"""Protocol-neutral TROVE mutation orchestration.

This layer deliberately delegates safety-critical execution to the existing
ApprovalGrant, writer coordinator and generation-aware leaves.  It centralizes
which leaf is selected without recreating any approval or lock semantics in a
wire adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from trove_core.approvals import ApprovalGrant
from trove_core.vault.config import VaultConfig


AuxiliaryImportKind = Literal['contacts', 'moments', 'favorites']
VectorAction = Literal['index', 'purge', 'rebuild']
DerivedDataScope = Literal['entity', 'source', 'run', 'task']


def _exact_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f'{field} must be an exact boolean')
    return value


def _optional_positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int | None:
    if value is None:
        return None
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        qualifier = 'non-negative' if allow_zero else 'positive'
        raise TypeError(f'{field} must be a {qualifier} exact integer or null')
    return value


@dataclass(frozen=True)
class FullImportCommand:
    sources: tuple[str | Path, ...]
    reset_index_cache: bool = False
    limit_per_sqlite: int | None = None
    process_config: object | None = None
    force_rescan: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'sources', tuple(self.sources))
        _exact_bool(self.reset_index_cache, field='reset_index_cache')
        _exact_bool(self.force_rescan, field='force_rescan')
        _optional_positive_int(self.limit_per_sqlite, field='limit_per_sqlite')


@dataclass(frozen=True)
class SyncCommand:
    account_ids: tuple[str, ...] = ()
    full: bool = False
    since: str | datetime | None = None
    snapshot_dir: str | Path | None = None
    snapshot_command: str | None = None
    limit_per_shard: int | None = None
    vector_backend: str = 'zvec'
    media_discovery_mode: str = 'message_delta'
    profile_refresh_budget: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, 'account_ids', tuple(self.account_ids))
        if len(self.account_ids) > 32 or any(
            type(value) is not str or not value.strip() for value in self.account_ids
        ):
            raise ValueError('account_ids must be bounded non-empty strings')
        if len(set(self.account_ids)) != len(self.account_ids):
            raise ValueError('account_ids must be unique')
        _exact_bool(self.full, field='full')
        _optional_positive_int(self.limit_per_shard, field='limit_per_shard')
        if self.vector_backend not in {'zvec', 'sqlite'}:
            raise ValueError('unsupported vector backend')
        if self.media_discovery_mode not in {'full', 'message_delta'}:
            raise ValueError('unsupported media discovery mode')
        if type(self.profile_refresh_budget) is not int or not 0 <= self.profile_refresh_budget <= 5:
            raise ValueError('profile_refresh_budget must be from 0 to 5')


@dataclass(frozen=True)
class MaintainCommand:
    auto_rebuild: bool = False
    backup_retention: int = 3
    log_retention: int = 5
    max_log_bytes: int = 5 * 1024 * 1024
    vector_backend: str = 'zvec'
    model_path: str | None = None
    vacuum: bool = False
    always_backup: bool = False
    media_voice_budget: int = 0
    media_image_budget: int = 0
    media_caption_budget: int = 0
    full_scan: bool = False

    def __post_init__(self) -> None:
        _exact_bool(self.auto_rebuild, field='auto_rebuild')
        _exact_bool(self.vacuum, field='vacuum')
        _exact_bool(self.full_scan, field='full_scan')
        _exact_bool(self.always_backup, field='always_backup')
        for field_name in ('backup_retention', 'log_retention', 'max_log_bytes'):
            _optional_positive_int(getattr(self, field_name), field=field_name)
        for field_name in ('media_voice_budget', 'media_image_budget', 'media_caption_budget'):
            _optional_positive_int(getattr(self, field_name), field=field_name, allow_zero=True)
        if self.vector_backend not in {'zvec', 'sqlite'}:
            raise ValueError('unsupported vector backend')


@dataclass(frozen=True)
class AuxiliaryImportCommand:
    kind: AuxiliaryImportKind
    source: str | Path
    account_id: str
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {'contacts', 'moments', 'favorites'}:
            raise ValueError('unsupported auxiliary import kind')
        if not str(self.account_id or '').strip():
            raise ValueError('account_id is required')
        _optional_positive_int(self.limit, field='limit')


@dataclass(frozen=True)
class VectorCommand:
    action: VectorAction = 'index'
    model_path: str | None = None
    backend: str = 'zvec'
    batch_size: int = 256
    max_messages: int | None = None
    cloud: bool = False

    def __post_init__(self) -> None:
        if self.action not in {'index', 'purge', 'rebuild'}:
            raise ValueError('unsupported vector action')
        if self.backend not in {'zvec', 'sqlite'}:
            raise ValueError('unsupported vector backend')
        _optional_positive_int(self.batch_size, field='batch_size')
        _optional_positive_int(self.max_messages, field='max_messages')
        _exact_bool(self.cloud, field='cloud')


@dataclass(frozen=True)
class DerivedDataPurgeCommand:
    scope_type: DerivedDataScope
    scope_id: str
    audit_retention_days: int = 365

    def __post_init__(self) -> None:
        if self.scope_type not in {'entity', 'source', 'run', 'task'}:
            raise ValueError('unsupported derived-data purge scope')
        if type(self.scope_id) is not str or not self.scope_id or len(self.scope_id) > 1000:
            raise ValueError('scope_id must be non-empty bounded text')
        if type(self.audit_retention_days) is not int or not 1 <= self.audit_retention_days <= 3650:
            raise ValueError('audit_retention_days must be from 1 to 3650')


@dataclass(frozen=True)
class PreparedVectorCommand:
    command: VectorCommand
    provider: object
    approval_action: str | None = None
    approval_danger_class: str | None = None
    approval_payload: dict[str, Any] | None = None

    @property
    def requires_approval(self) -> bool:
        return self.approval_action is not None


class TroveCommands:
    """One orchestration boundary shared by CLI, HTTP, MCP and agent tools."""

    def __init__(self, config: VaultConfig | str | Path | None) -> None:
        self.config = config if isinstance(config, VaultConfig) else VaultConfig.resolve(
            None if config is None else str(config)
        )

    def full_import_payload(self, command: FullImportCommand) -> dict[str, Any]:
        from .sensitive_commands import full_import_payload

        return full_import_payload(
            command.sources,
            reset_index_cache=command.reset_index_cache,
            limit_per_sqlite=command.limit_per_sqlite,
            process_config=command.process_config,
            force_rescan=command.force_rescan,
        )

    def full_import(self, command: FullImportCommand, *, approval_grant: ApprovalGrant) -> dict[str, Any]:
        from .sensitive_commands import execute_full_import

        return execute_full_import(
            self.config.root,
            command.sources,
            reset_index_cache=command.reset_index_cache,
            limit_per_sqlite=command.limit_per_sqlite,
            process_config=command.process_config,
            force_rescan=command.force_rescan,
            approval_grant=approval_grant,
        )

    @staticmethod
    def reset_index_cache_payload() -> dict[str, Any]:
        from .sensitive_commands import reset_index_cache_payload

        return reset_index_cache_payload()

    def reset_index_cache(self, *, approval_grant: ApprovalGrant) -> dict[str, Any]:
        from .sensitive_commands import execute_reset_index_cache

        return execute_reset_index_cache(self.config.root, approval_grant=approval_grant)

    @staticmethod
    def scope_rebuild_payload() -> dict[str, Any]:
        from .sensitive_commands import scope_rebuild_payload

        return scope_rebuild_payload()

    def scope_rebuild(self, *, approval_grant: ApprovalGrant) -> dict[str, Any]:
        from .sensitive_commands import execute_scope_rebuild

        return execute_scope_rebuild(self.config.root, approval_grant=approval_grant)

    @staticmethod
    def derived_data_purge_payload(command: DerivedDataPurgeCommand) -> dict[str, Any]:
        from .sensitive_commands import derived_data_purge_payload

        return derived_data_purge_payload(
            scope_type=command.scope_type,
            scope_id=command.scope_id,
            audit_retention_days=command.audit_retention_days,
        )

    def derived_data_purge(
        self,
        command: DerivedDataPurgeCommand,
        *,
        approval_grant: ApprovalGrant,
    ) -> dict[str, Any]:
        from .sensitive_commands import execute_derived_data_purge

        return execute_derived_data_purge(
            self.config.root,
            scope_type=command.scope_type,
            scope_id=command.scope_id,
            audit_retention_days=command.audit_retention_days,
            approval_grant=approval_grant,
        )

    def sync(self, command: SyncCommand) -> dict[str, Any]:
        from trove_core.sync import SyncOptions, parse_since, run_sync

        since = command.since
        if isinstance(since, str) or since is None:
            since = parse_since(since)
        return run_sync(
            self.config.root,
            options=SyncOptions(
                account_ids=command.account_ids,
                full=command.full,
                since=since,
                snapshot_dir=Path(command.snapshot_dir).expanduser() if command.snapshot_dir is not None else None,
                snapshot_command=command.snapshot_command,
                limit_per_shard=command.limit_per_shard,
                vector_backend=command.vector_backend,
                media_discovery_mode=command.media_discovery_mode,
                profile_refresh_budget=command.profile_refresh_budget,
            ),
        )

    def watch_sync(self, command: SyncCommand) -> None:
        from trove_core.sync import SyncOptions, parse_since, watch_sync

        since = command.since
        if isinstance(since, str) or since is None:
            since = parse_since(since)
        watch_sync(
            self.config.root,
            options=SyncOptions(
                full=command.full,
                since=since,
                snapshot_dir=Path(command.snapshot_dir).expanduser() if command.snapshot_dir is not None else None,
                snapshot_command=command.snapshot_command,
                limit_per_shard=command.limit_per_shard,
                vector_backend=command.vector_backend,
            ),
        )

    def maintain(self, command: MaintainCommand) -> dict[str, Any]:
        from trove_core.maintain import MaintainOptions, run_maintain

        return run_maintain(
            self.config.root,
            options=MaintainOptions(
                auto_rebuild=command.auto_rebuild,
                backup_retention=command.backup_retention,
                log_retention=command.log_retention,
                max_log_bytes=command.max_log_bytes,
                vector_backend=command.vector_backend,
                model_path=command.model_path,
                vacuum=command.vacuum,
                always_backup=command.always_backup,
                media_voice_budget=command.media_voice_budget,
                media_image_budget=command.media_image_budget,
                media_caption_budget=command.media_caption_budget,
                full_scan=command.full_scan,
            ),
        )

    def auxiliary_import(self, command: AuxiliaryImportCommand) -> dict[str, Any]:
        from trove_core.store import change_journal
        from trove_core.store.repositories import MultimodalRepository
        from trove_core.store.sqlite_store import SQLiteStore
        from trove_core.vault.mutations import coordinated_vault_mutation
        from trove_core.wechat.auxiliary_import import (
            commit_prepared_auxiliary_sources,
            prepare_auxiliary_sources,
        )

        source = Path(command.source).expanduser()
        family = {
            'contacts': 'contact',
            'moments': 'moment',
            'favorites': 'favorite',
        }[command.kind]

        # Admission/schema work is short.  Source SQLite parsing and hashing
        # happen after releasing the Vault writer.
        with coordinated_vault_mutation(self.config, operation='auxiliary_import'):
            admission_store = SQLiteStore(self.config.paths.sqlite_path)
            try:
                expected_generation = change_journal.recover_sync_commit_generation(admission_store)
            finally:
                admission_store.close_all()

        prepared = prepare_auxiliary_sources(
            source.parent,
            account_id=command.account_id,
            limit=command.limit,
            only={family},
            source_overrides={family: source},
        )

        with coordinated_vault_mutation(self.config, operation='auxiliary_import'):
            store = SQLiteStore(self.config.paths.sqlite_path)
            try:
                claimed_generation = change_journal.claim_sync_commit_generation(
                    store,
                    expected_generation,
                )
                if claimed_generation is None:
                    result = {
                        f'imported_{command.kind}': 0,
                        'status': 'retry_required',
                        'errors': ['sync_commit_generation_changed'],
                        'chunks': None,
                        'dirty_citations': 0,
                        'removed_count': 0,
                        'raw_content_included': False,
                    }
                    if command.kind == 'moments':
                        result['coverage'] = {
                            'scope_counts': {},
                            'excluded_counts': {},
                            'raw_content_included': False,
                        }
                    return result
                repository = MultimodalRepository(store)
                report = commit_prepared_auxiliary_sources(
                    prepared,
                    store=store,
                    repo=repository,
                )
                changed = set(report.changed_citations.get(family, ()))
                removed = set(report.removed_citations.get(family, ()))
                projected = sorted(changed | removed)
                chunks = (
                    store.rebuild_evidence_chunks_for_source_citations(family, projected)
                    if projected else None
                )
                if removed:
                    store.record_citation_tombstones([
                        {
                            'citation': citation,
                            'account_id': command.account_id,
                            'conversation_id': '',
                            'source_type': family,
                        }
                        for citation in sorted(removed)
                    ])
                if changed:
                    store.clear_citation_tombstones(sorted(changed))
                dirty_count = change_journal.record_dirty_citations(store, report.dirty_refs())
                result = {
                    f'imported_{command.kind}': getattr(report, f'{command.kind}_imported'),
                    'chunks': chunks,
                    'dirty_citations': dirty_count,
                    'removed_count': len(removed),
                    'raw_content_included': False,
                }
                if command.kind == 'moments':
                    result['coverage'] = {
                        'scope_counts': dict(sorted(report.scope_counts.items())),
                        'excluded_counts': dict(sorted(report.excluded_counts.items())),
                        'raw_content_included': False,
                    }
                change_journal.complete_sync_commit_generation(store, claimed_generation)
                return result
            finally:
                store.close_all()

    def prepare_vector(self, command: VectorCommand) -> PreparedVectorCommand:
        from trove_core.runtime import configured_embedding_provider, vector_cloud_approval_payload
        from .sensitive_commands import vector_mutation_payload

        provider = configured_embedding_provider(
            command.model_path,
            strict=True,
            vault_root=self.config.root,
            prefer_cloud=command.cloud,
        )
        if provider is None:
            raise RuntimeError(
                'vector indexing requires model_path/TROVE_EMBEDDING_MODEL_PATH '
                'or explicit cloud embedding opt-in with credentials'
            )
        if command.action in {'purge', 'rebuild'}:
            payload = vector_mutation_payload(
                provider,
                backend=command.backend,
                batch_size=command.batch_size,
                max_messages=command.max_messages,
                purge=True,
            )
            return PreparedVectorCommand(
                command,
                provider,
                approval_action='vector_rebuild' if command.action == 'rebuild' else 'vector_purge_rebuild',
                approval_danger_class='vector_purge_rebuild',
                approval_payload=payload,
            )
        if getattr(provider, 'egress_kind', None) is not None:
            payload = vector_cloud_approval_payload(
                self.config,
                provider,
                backend=command.backend,
                batch_size=command.batch_size,
                max_messages=command.max_messages,
                purge=False,
            )
            return PreparedVectorCommand(
                command,
                provider,
                approval_action='cloud_vector_index',
                approval_danger_class='cloud_embedding_upload',
                approval_payload=payload,
            )
        return PreparedVectorCommand(command, provider)

    def vector(
        self,
        prepared: PreparedVectorCommand,
        *,
        approval_grant: ApprovalGrant | None = None,
    ) -> dict[str, Any]:
        from trove_core.runtime import index_vectors
        from .sensitive_commands import execute_vector_mutation

        command = prepared.command
        if command.action in {'purge', 'rebuild'}:
            if approval_grant is None:
                raise TypeError('destructive vector command requires ApprovalGrant')
            return execute_vector_mutation(
                self.config.root,
                prepared.provider,
                action='vector_rebuild' if command.action == 'rebuild' else 'vector_purge_rebuild',
                backend=command.backend,
                batch_size=command.batch_size,
                max_messages=command.max_messages,
                purge=True,
                approval_grant=approval_grant,
            )
        return index_vectors(
            self.config,
            prepared.provider,
            backend=command.backend,
            batch_size=command.batch_size,
            max_messages=command.max_messages,
            purge=False,
            approval_grant=approval_grant,
            approval_payload=prepared.approval_payload,
        )


__all__ = [
    'AuxiliaryImportCommand',
    'FullImportCommand',
    'MaintainCommand',
    'PreparedVectorCommand',
    'SyncCommand',
    'TroveCommands',
    'VectorCommand',
]
