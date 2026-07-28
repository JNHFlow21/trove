from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterator, TypeVar

from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import (
    MutationOutsideCoordinator,
    VaultOperationCoordinator,
    VaultWriteSession,
    require_vault_write_session,
)
from trove_core.vault.generation import coordinated_vault_generation_publish


@dataclass(frozen=True, slots=True)
class VaultMutationSpec:
    """One public mutation family and its lock/approval policy.

    Approval is intentionally descriptive here.  The writer coordinator never
    creates or consumes approvals: unattended maintenance and sync must remain
    possible, while sensitive commands keep enforcing approval at their
    application boundary.
    """

    operation: str
    owner: str
    approval_policy: str


_SPECS = (
    VaultMutationSpec('decrypt_snapshot', 'decrypt', 'separate-sensitive-boundary'),
    VaultMutationSpec('full_import', 'import', 'separate-destructive-boundary'),
    VaultMutationSpec('auxiliary_import', 'aux-import', 'separate-adapter-boundary'),
    VaultMutationSpec('reset_index_cache', 'reset-index', 'separate-destructive-boundary'),
    VaultMutationSpec('scope_rebuild', 'scope-rebuild', 'separate-destructive-boundary'),
    VaultMutationSpec('sync', 'sync', 'unattended'),
    VaultMutationSpec('maintain', 'maintain', 'unattended'),
    VaultMutationSpec('vector_index', 'vector-index', 'separate-if-purge'),
    VaultMutationSpec('vector_rebuild', 'vector-rebuild', 'separate-destructive-boundary'),
    VaultMutationSpec('content_kind_backfill', 'content-backfill', 'separate-destructive-boundary'),
    VaultMutationSpec('appmsg_backfill', 'appmsg-backfill', 'separate-destructive-boundary'),
    VaultMutationSpec('message_media_backfill', 'media-reference-backfill', 'separate-destructive-boundary'),
    VaultMutationSpec('rebuild_chunks', 'rebuild-chunks', 'separate-adapter-boundary'),
    VaultMutationSpec('rebuild_fts', 'rebuild-fts', 'separate-adapter-boundary'),
    VaultMutationSpec('initialize_index', 'initialize-index', 'separate-adapter-boundary'),
    VaultMutationSpec('files_archive', 'files-archive', 'separate-export-boundary'),
    VaultMutationSpec('media_fetch', 'media-fetch', 'separate-adapter-boundary'),
    VaultMutationSpec('media_annotate', 'media-annotate', 'separate-adapter-boundary'),
    VaultMutationSpec('media_invalidate', 'media-invalidate', 'separate-destructive-boundary'),
    VaultMutationSpec('media_transcribe', 'media-transcribe', 'separate-if-cloud'),
    VaultMutationSpec('media_observe', 'media-observe', 'separate-if-cloud'),
    VaultMutationSpec('observation_write', 'observation', 'separate-agent-boundary'),
    VaultMutationSpec('entity_reconcile', 'entity-reconcile', 'separate-destructive-boundary'),
    VaultMutationSpec('profile_enrichment', 'profile-enrichment', 'explicit-agent-boundary'),
    VaultMutationSpec('profile_automation', 'profile-automation', 'explicit-opt-in-then-unattended'),
    VaultMutationSpec('derived_data_purge', 'derived-data-purge', 'separate-destructive-boundary'),
    VaultMutationSpec('process_config_write', 'process-config', 'separate-adapter-boundary'),
    VaultMutationSpec('source_manifest_write', 'source-manifest', 'separate-adapter-boundary'),
    VaultMutationSpec('wiki_write', 'wiki-write', 'separate-adapter-boundary'),
    VaultMutationSpec('fixture_generation', 'fixture-index', 'fixture-only'),
)

VAULT_MUTATION_INVENTORY = MappingProxyType({spec.operation: spec for spec in _SPECS})


# These operations can remove, replace, or rebuild artifacts consumed by one
# logical read.  SQLite transactions protect an individual statement, but not
# a multi-statement search spanning a destructive reset/rebuild.  Holding the
# generation publication lease makes the whole mutation one old-or-new
# boundary for API, MCP, and runtime readers.  Vector publication has a more
# precise staged-swap boundary in ``runtime.py``; fixture publication already
# owns the same root-inode lock in ``fixture_vault_session``.
_GENERATION_EXCLUSIVE_OPERATIONS = frozenset(
    {
        'full_import',
        'reset_index_cache',
        'scope_rebuild',
        'content_kind_backfill',
        'rebuild_chunks',
        'rebuild_fts',
        'initialize_index',
        'derived_data_purge',
    }
)


@contextmanager
def _generation_boundary(
    vault: VaultConfig | str | Path,
    *,
    operation: str,
    write_session: VaultWriteSession,
) -> Iterator[VaultWriteSession]:
    if operation not in _GENERATION_EXCLUSIVE_OPERATIONS:
        yield write_session
        return
    with coordinated_vault_generation_publish(vault, operation=operation):
        yield write_session


@contextmanager
def coordinated_vault_mutation(
    vault: VaultConfig | str | Path,
    *,
    operation: str,
    write_session: VaultWriteSession | None = None,
) -> Iterator[VaultWriteSession]:
    """Acquire one Vault writer or validate an explicitly propagated session.

    Nested mutation orchestration reuses an explicitly supplied leaf session;
    it never tries to take a second process lock.  This keeps re-entrancy
    visible in function signatures and prevents the former ``use_lock=False``
    escape hatch from becoming an uncoordinated public write path.
    """

    attempts = _COORDINATION_ATTEMPTS.get()
    if attempts is not None:
        attempts.append(operation)
    spec = VAULT_MUTATION_INVENTORY.get(operation)
    if spec is None:
        raise ValueError(f'unknown Vault mutation operation: {operation}')
    if write_session is not None:
        active_session = require_vault_write_session(vault, write_session)
        with _generation_boundary(
            vault,
            operation=operation,
            write_session=active_session,
        ):
            yield active_session
        return
    coordinator = VaultOperationCoordinator(vault)
    with coordinator.write(owner=spec.owner) as acquired:
        with _generation_boundary(
            vault,
            operation=operation,
            write_session=acquired,
        ):
            yield acquired


def record_vault_mutation_noop(*, operation: str) -> None:
    """Record a proven no-write return from a registered mutation command."""

    if operation not in VAULT_MUTATION_INVENTORY:
        raise ValueError(f'unknown Vault mutation operation: {operation}')
    attempts = _COORDINATION_ATTEMPTS.get()
    if attempts is not None:
        attempts.append(f'noop:{operation}')


F = TypeVar('F', bound=Callable)
_COORDINATION_ATTEMPTS: ContextVar[list[str] | None] = ContextVar(
    'trove_vault_mutation_coordination_attempts',
    default=None,
)


def mutation_entrypoint(operation: str) -> Callable[[F], F]:
    """Label and runtime-audit one public mutation entrypoint.

    A successful top-level call must enter ``coordinated_vault_mutation``
    directly or through another registered mutation entrypoint.  Validation
    exceptions may happen before lock acquisition and are preserved.
    """

    if operation not in VAULT_MUTATION_INVENTORY:
        raise ValueError(f'unknown Vault mutation operation: {operation}')

    def decorate(function: F) -> F:
        @wraps(function)
        def guarded(*args, **kwargs):
            inherited = _COORDINATION_ATTEMPTS.get()
            root_call = inherited is None
            attempts: list[str] = [] if root_call else inherited
            token = _COORDINATION_ATTEMPTS.set(attempts) if root_call else None
            try:
                result = function(*args, **kwargs)
            except BaseException:
                raise
            else:
                covered = operation in attempts or f'noop:{operation}' in attempts
                if root_call and not covered:
                    raise MutationOutsideCoordinator(
                        f'Public Vault mutation {operation!r} bypassed coordination',
                        code='mutation_entrypoint_without_coordinator',
                    )
                return result
            finally:
                if token is not None:
                    _COORDINATION_ATTEMPTS.reset(token)

        setattr(guarded, '__trove_vault_mutation__', operation)
        return guarded  # type: ignore[return-value]

    return decorate


def mutation_operation(function: Callable) -> str | None:
    value = getattr(function, '__trove_vault_mutation__', None)
    return value if isinstance(value, str) else None
