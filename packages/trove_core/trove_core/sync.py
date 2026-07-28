from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import time
from typing import Any, Mapping

from trove_protocol.provider import Provider

from trove_core.runtime import configured_embedding_provider, index_vectors, vector_status_payload
from trove_core.media_pipeline import enqueue_media_jobs
from trove_core.knowledge.profile_automation import ProfileAutomationService, process_profile_refresh_queue
from trove_core.store.repositories import MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.change_journal import (
    MAX_INLINE_DIRTY_CITATIONS,
    claim_sync_commit_generation,
    clear_all_dirty_citations,
    clear_dirty_citation_batch,
    clear_dirty_citations,
    complete_sync_commit_generation,
    dirty_citation_count,
    ensure_sync_state,
    read_aux_fingerprints,
    read_dirty_citation_batch,
    read_dirty_citations,
    read_sync_commit_generation,
    read_waterlines,
    record_dirty_citations,
    recover_sync_commit_generation,
    write_aux_fingerprints,
    write_waterlines,
)
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.locks import VaultOperationLocked, active_vector_progress
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.vault.tracing import TraceTimeline
from trove_core.wechat.source_discovery import is_wechat_decrypted_account_dir, iter_importable_files
from trove_core.wechat.decrypt.manifest import load_snapshot_guard
from trove_core.wechat.auxiliary_import import (
    PreparedAuxiliaryImport,
    auxiliary_source_fingerprints,
    commit_prepared_auxiliary_sources,
    family_for_auxiliary_source_key,
    prepare_auxiliary_sources,
)
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.message_refs import message_media_references_for_messages
from trove_core.wechat.media.resources import discover_media_assets_delta
from trove_core.wechat.media.source_registry import (
    SourceSnapshot,
    account_dir_hash,
    bind_account_assets,
    inspect_source_snapshot,
    persist_source_snapshot,
    rebind_account_assets,
)
from trove_core.wechat.media_mapping_assessment import d0_mapping_conclusion
from trove_core.wechat.process_config import read_latest_process_config
from trove_core.wechat.snapshot_media import DEFAULT_MAX_COPY_BYTES, DEFAULT_WECHAT_FILES_ROOT, refresh_snapshot_media_cache
from trove_core.watch import WatchBackend, create_watch_backend

SYNC_CONFIG_PATH = ('jobs', 'sync_config.redacted.json')
DEFAULT_SNAPSHOT_RELATIVE = ('sources', 'wechat-kos-decrypted', 'current')
SYNC_WATCH_MANIFEST = ('jobs', 'sync_watch_manifest.redacted.json')


@dataclass(frozen=True)
class SyncConfig:
    snapshot_dir: Path | None = None
    snapshot_command: str | None = None
    snapshot_media_enabled: bool = True
    snapshot_media_root: Path | None = None
    snapshot_media_max_bytes: int = DEFAULT_MAX_COPY_BYTES
    debounce_seconds: float = 3.0
    command_timeout_seconds: int = 300
    vector_inline_dirty_limit: int = MAX_INLINE_DIRTY_CITATIONS

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'snapshot_dir_configured': self.snapshot_dir is not None,
            'snapshot_command_configured': bool(self.snapshot_command),
            'snapshot_media_enabled': self.snapshot_media_enabled,
            'snapshot_media_root_configured': self.snapshot_media_root is not None,
            'snapshot_media_max_bytes': self.snapshot_media_max_bytes,
            'debounce_seconds': self.debounce_seconds,
            'command_timeout_seconds': self.command_timeout_seconds,
            'vector_inline_dirty_limit': self.vector_inline_dirty_limit,
        }


@dataclass(frozen=True)
class SyncOptions:
    account_ids: tuple[str, ...] = ()
    full: bool = False
    since: datetime | None = None
    snapshot_dir: Path | None = None
    snapshot_command: str | None = None
    limit_per_shard: int | None = None
    vector_backend: str = 'zvec'
    snapshot_media_enabled: bool | None = None
    media_discovery_mode: str = 'full'
    profile_refresh_budget: int = 5


@dataclass(frozen=True)
class SyncReport:
    ok: bool
    status: str
    snapshot: dict[str, Any]
    sources_seen: int
    sources_imported: int
    messages_imported: int
    conversations_changed: int
    dirty_count: int
    chunks: dict[str, Any] | None
    media: dict[str, Any]
    auxiliary: dict[str, Any]
    waterlines_updated: int
    vector: dict[str, Any]
    elapsed_ms: float
    trace_id: str
    errors: list[str]
    raw_content_included: bool = False
    profiles: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PreparedSyncSource:
    importer: WeChatDecryptedAccountImporter
    accounts: list[Any]
    conversations: list[Any]
    messages: list[Any]
    media_refs: tuple[Any, ...]
    media_source_states: tuple[dict[str, Any], ...]
    media_discovery_counters: dict[str, int]
    changed_aux_keys: dict[str, str]
    auxiliary: PreparedAuxiliaryImport | None
    account_hash: str


class SyncCommitGenerationChanged(RuntimeError):
    code = 'sync_commit_generation_changed'


def provider_change_batch(
    provider: Provider,
    *,
    account_id: str,
    cursor: str | None = None,
) -> Mapping[str, Any]:
    """Read one normalized, replayable source batch through the Provider contract."""
    payload: dict[str, Any] = {'operation': 'changes', 'account_id': account_id}
    if cursor is not None:
        payload['cursor'] = cursor
    result = provider.invoke('read', payload)
    required = {'account_id', 'records', 'watermark', 'change_cursor', 'replayed'}
    if not isinstance(result, Mapping) or not required <= set(result):
        raise RuntimeError('provider_change_batch_invalid')
    if result['account_id'] != account_id or not isinstance(result['records'], list):
        raise RuntimeError('provider_change_batch_invalid')
    return result


def parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)


def read_sync_config(cfg: VaultConfig) -> SyncConfig:
    path = cfg.root.joinpath(*SYNC_CONFIG_PATH)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            payload = {}
    nested = payload.get('sync') if isinstance(payload.get('sync'), dict) else payload
    snapshot_dir_raw = os.environ.get('TROVE_SYNC_SNAPSHOT_DIR') or nested.get('snapshot_dir')
    snapshot_command = os.environ.get('TROVE_SYNC_SNAPSHOT_COMMAND') or nested.get('snapshot_command')
    media_enabled_raw = os.environ.get('TROVE_SYNC_SNAPSHOT_MEDIA_ENABLED')
    if media_enabled_raw is None:
        media_enabled_raw = nested.get('snapshot_media_enabled', True)
    snapshot_media_enabled = str(media_enabled_raw).strip().lower() not in {'0', 'false', 'no', 'off'}
    snapshot_media_root_raw = os.environ.get('TROVE_SYNC_SNAPSHOT_MEDIA_ROOT') or nested.get('snapshot_media_root')
    media_max_raw = os.environ.get('TROVE_SYNC_SNAPSHOT_MEDIA_MAX_BYTES') or nested.get('snapshot_media_max_bytes') or DEFAULT_MAX_COPY_BYTES
    debounce = float(os.environ.get('TROVE_SYNC_DEBOUNCE_SECONDS') or nested.get('debounce_seconds') or 3.0)
    timeout = int(os.environ.get('TROVE_SYNC_COMMAND_TIMEOUT_SECONDS') or nested.get('command_timeout_seconds') or 300)
    inline_limit_raw = os.environ.get('TROVE_SYNC_VECTOR_INLINE_DIRTY_LIMIT')
    if inline_limit_raw is None:
        inline_limit_raw = nested.get('vector_inline_dirty_limit', MAX_INLINE_DIRTY_CITATIONS)
    if inline_limit_raw is None:
        inline_limit_raw = MAX_INLINE_DIRTY_CITATIONS
    try:
        vector_inline_dirty_limit = min(MAX_INLINE_DIRTY_CITATIONS, max(0, int(inline_limit_raw)))
    except (TypeError, ValueError):
        vector_inline_dirty_limit = MAX_INLINE_DIRTY_CITATIONS
    try:
        snapshot_media_max_bytes = max(0, int(media_max_raw))
    except (TypeError, ValueError):
        snapshot_media_max_bytes = DEFAULT_MAX_COPY_BYTES
    snapshot_dir = Path(snapshot_dir_raw).expanduser() if snapshot_dir_raw else None
    snapshot_media_root = Path(snapshot_media_root_raw).expanduser() if snapshot_media_root_raw else None
    return SyncConfig(
        snapshot_dir=snapshot_dir,
        snapshot_command=snapshot_command or None,
        snapshot_media_enabled=snapshot_media_enabled,
        snapshot_media_root=snapshot_media_root,
        snapshot_media_max_bytes=snapshot_media_max_bytes,
        debounce_seconds=debounce,
        command_timeout_seconds=timeout,
        vector_inline_dirty_limit=vector_inline_dirty_limit,
    )


def default_snapshot_dir(cfg: VaultConfig) -> Path:
    return cfg.root.joinpath(*DEFAULT_SNAPSHOT_RELATIVE)


def resolve_snapshot_dir(cfg: VaultConfig, config: SyncConfig, override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser()
    if config.snapshot_dir is not None:
        return config.snapshot_dir.expanduser()
    return default_snapshot_dir(cfg)


def run_snapshot_command(cfg: VaultConfig, config: SyncConfig, snapshot_dir: Path, override: str | None = None) -> dict[str, Any]:
    command = override if override is not None else config.snapshot_command
    if not command:
        return {'refreshed': False, 'command_configured': False, 'skipped_reason': 'snapshot_command_empty'}
    env = dict(os.environ)
    env['TROVE_VAULT_ROOT'] = str(cfg.root)
    env['TROVE_SYNC_SNAPSHOT_DIR'] = str(snapshot_dir)
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(Path.cwd()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=config.command_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            'refreshed': False,
            'command_configured': True,
            'returncode': None,
            'elapsed_ms': round((time.time() - started) * 1000, 3),
            'error_code': 'snapshot_command_timeout',
        }
    return {
        'refreshed': proc.returncode == 0,
        'command_configured': True,
        'returncode': proc.returncode,
        'elapsed_ms': round((time.time() - started) * 1000, 3),
        'stdout_bytes': len(proc.stdout.encode('utf-8')) if proc.stdout else 0,
        'stderr_bytes': len(proc.stderr.encode('utf-8')) if proc.stderr else 0,
        'error_code': None if proc.returncode == 0 else 'snapshot_command_failed',
    }


def refresh_snapshot_media_if_enabled(config: SyncConfig, snapshot_dir: Path) -> dict[str, Any]:
    if not config.snapshot_media_enabled:
        return {'enabled': False, 'status': 'skipped', 'reason': 'disabled', 'raw_paths_included': False}
    try:
        return refresh_snapshot_media_cache(
            snapshot_dir,
            wechat_root=config.snapshot_media_root or DEFAULT_WECHAT_FILES_ROOT,
            max_bytes=config.snapshot_media_max_bytes,
        )
    except PermissionError:
        reason = 'permission_denied'
    except InterruptedError:
        reason = 'interrupted'
    except OSError:
        reason = 'source_unavailable'
    return {
        'enabled': True,
        'status': 'skipped',
        'reason': reason,
        'raw_paths_included': False,
        'raw_content_included': False,
    }


@mutation_entrypoint('sync')
def run_sync(
    vault_root: str | Path | None = None,
    *,
    options: SyncOptions | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    started = time.time()
    try:
        return _run_sync_coordinated(
            vault_root,
            options=options,
            write_session=write_session,
        )
    except VaultOperationLocked as exc:
        return SyncReport(
            ok=False,
            status='locked',
            snapshot={'exists': False, 'status': 'skipped', 'reason': 'writer_lock'},
            sources_seen=0,
            sources_imported=0,
            messages_imported=0,
            conversations_changed=0,
            dirty_count=0,
            chunks=None,
            media={'status': 'skipped', 'reason': 'writer_lock'},
            auxiliary={'status': 'skipped', 'reason': 'writer_lock'},
            waterlines_updated=0,
            vector={'status': 'blocked', 'reason': 'writer_lock'},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id='',
            errors=[exc.__class__.__name__],
        ).to_dict()


def _run_sync_coordinated(
    vault_root: str | Path | None = None,
    *,
    options: SyncOptions | None = None,
    write_session: VaultWriteSession | None,
) -> dict[str, Any]:
    started = time.time()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    cfg.ensure()
    # Preserve the immediate writer-conflict contract and initialize a new
    # Vault in one short database window.  Existing Vault cursor reads below
    # are read-only; no writer spans source or provider work.
    with coordinated_vault_mutation(cfg, operation='sync', write_session=write_session):
        initial_store = SQLiteStore(cfg.paths.sqlite_path)
        initial_store.initialize()
        # An odd value can only survive a process failure after the prior sync
        # claimed its publication window.  Close it before this fresh scan so
        # no prepared work can inherit a partially committed state.
        recover_sync_commit_generation(initial_store)
    options = options or SyncOptions()
    sync_config = read_sync_config(cfg)
    if options.snapshot_media_enabled is not None:
        sync_config = replace(sync_config, snapshot_media_enabled=bool(options.snapshot_media_enabled))
    snapshot_dir = resolve_snapshot_dir(cfg, sync_config, options.snapshot_dir)
    trace = TraceTimeline(cfg.root)
    trace_id = trace.start('sync', {'full': options.full, 'snapshot': sync_config.to_redacted_dict()})
    errors: list[str] = []
    snapshot_report = run_snapshot_command(cfg, sync_config, snapshot_dir, override=options.snapshot_command)
    if snapshot_report.get('error_code'):
        errors.append(str(snapshot_report['error_code']))
    if not snapshot_dir.exists():
        report = SyncReport(
            ok=False,
            status='no_snapshot',
            snapshot={**snapshot_report, 'exists': False},
            sources_seen=0,
            sources_imported=0,
            messages_imported=0,
            conversations_changed=0,
            dirty_count=0,
            chunks=None,
            media={'status': 'skipped', 'reason': 'no_snapshot'},
            auxiliary={'status': 'skipped', 'reason': 'no_snapshot'},
            waterlines_updated=0,
            vector={'status': 'skipped', 'reason': 'no_new_messages'},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=errors,
        )
        trace.fail(trace_id, {'status': report.status, 'errors': errors})
        return report.to_dict()
    snapshot_media_report = refresh_snapshot_media_if_enabled(sync_config, snapshot_dir)
    snapshot_guard = load_snapshot_guard(snapshot_dir)

    vector_progress = active_vector_progress(cfg)
    if vector_progress:
        report = SyncReport(
            ok=False,
            status='blocked_by_vector_rebuild',
            snapshot={**snapshot_report, 'exists': True, 'media_cache': snapshot_media_report},
            sources_seen=0,
            sources_imported=0,
            messages_imported=0,
            conversations_changed=0,
            dirty_count=0,
            chunks=None,
            media={'status': 'skipped', 'reason': 'blocked_by_vector_rebuild'},
            auxiliary={'status': 'skipped', 'reason': 'blocked_by_vector_rebuild'},
            waterlines_updated=0,
            vector={'status': 'blocked', 'reason': 'active_vector_progress', **vector_progress},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=errors,
        )
        trace.fail(trace_id, {'status': report.status})
        return report.to_dict()

    try:
        state_store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
        ensure_sync_state(state_store)
        prepared_sync_generation = read_sync_commit_generation(state_store)
        waterlines = {} if options.full else read_waterlines(state_store)
        aux_fingerprints = {} if options.full else read_aux_fingerprints(state_store)

        prepared_sources: list[_PreparedSyncSource] = []
        waterline_updates: dict[tuple[str, str, str], dict[str, Any]] = {}
        aux_fingerprint_updates: dict[str, str] = {}
        changed_conversations: set[tuple[str, str]] = set()
        message_chunk_reports: list[dict[str, Any]] = []
        message_dirty_count = 0
        sources_seen = 0
        sources_imported = 0
        messages_imported = 0
        contacts_imported = 0
        moments_imported = 0
        favorites_imported = 0
        aux_sources_seen = 0
        aux_sources_imported = 0
        aux_dirty_refs: list[dict[str, str]] = []
        aux_changed_families: dict[str, int] = {}
        aux_removed_counts: dict[str, int] = {}
        aux_projection_citations: dict[str, set[str]] = {}
        aux_removed_refs: list[dict[str, str]] = []
        auxiliary_profile_citations: set[str] = set()
        auxiliary_profile_removal = False
        changed_profile_identity_values: set[str] = set()
        profile_refresh_all = False
        profile_refresh_queue: dict[str, Any] = {
            'status': 'skipped',
            'reason': 'no_relevant_changes',
            'queued': 0,
            'raw_content_included': False,
        }
        media_seen = media_upserted = media_links = media_accepted = media_excluded = 0
        media_changed_asset_ids: set[str] = set()
        media_discovery_counters: dict[str, int] = {}
        media_persist_metrics: dict[str, int] = {}
        media_excluded_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        excluded_counts: dict[str, int] = {}

        scan_store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
        for source in iter_importable_files(snapshot_dir):
            if not is_wechat_decrypted_account_dir(source):
                continue
            sources_seen += 1
            if not snapshot_guard.allows(source):
                errors.append(f'{source.name}: not_selected_account')
                excluded_counts['not_selected_account'] = excluded_counts.get('not_selected_account', 0) + 1
                continue
            imp = WeChatDecryptedAccountImporter(source)
            if options.account_ids and imp.account_id not in options.account_ids:
                excluded_counts['not_requested_account'] = excluded_counts.get('not_requested_account', 0) + 1
                continue
            try:
                accounts, conversations, messages = imp.load(
                    limit_per_shard=options.limit_per_shard,
                    waterlines=waterlines,
                    since=options.since,
                )
                if options.media_discovery_mode == 'message_delta':
                    media_refs = tuple(message_media_references_for_messages(messages))
                    media_source_states: tuple[dict[str, Any], ...] = ()
                    source_media_counters = {'message_delta_rows': len(media_refs)}
                else:
                    media_discovery = discover_media_assets_delta(
                        source,
                        store=scan_store,
                        account_id=imp.account_id,
                        limit_per_table=options.limit_per_shard,
                    )
                    media_refs = tuple(media_discovery.refs)
                    media_source_states = tuple(media_discovery.source_states)
                    source_media_counters = dict(media_discovery.counters)
                current_aux_fingerprints = auxiliary_source_fingerprints(
                    source,
                    account_id=imp.account_id,
                )
                changed_aux_keys = {
                    key: fingerprint
                    for key, fingerprint in current_aux_fingerprints.items()
                    if options.full or aux_fingerprints.get(key) != fingerprint
                }
                prepared_auxiliary = None
                if changed_aux_keys:
                    changed_families = None if options.full else {
                        family
                        for key in changed_aux_keys
                        for family in [family_for_auxiliary_source_key(key)]
                        if family
                    }
                    prepared_auxiliary = prepare_auxiliary_sources(
                        source,
                        account_id=imp.account_id,
                        only=changed_families,
                    )
            except (sqlite3.DatabaseError, OSError) as exc:
                errors.append(f'source_{sources_seen}: {exc.__class__.__name__}')
                continue

            aux_sources_seen += len(current_aux_fingerprints)
            for key, value in source_media_counters.items():
                media_discovery_counters[str(key)] = media_discovery_counters.get(str(key), 0) + int(value)
            for key, value in imp.last_scope_counts.items():
                scope_counts[key] = scope_counts.get(key, 0) + value
            for key, value in imp.last_excluded_counts.items():
                excluded_counts[key] = excluded_counts.get(key, 0) + value
            waterline_updates.update(imp.last_waterline_updates)
            prepared_sources.append(_PreparedSyncSource(
                importer=imp,
                accounts=list(accounts),
                conversations=list(conversations),
                messages=list(messages),
                media_refs=media_refs,
                media_source_states=media_source_states,
                media_discovery_counters=source_media_counters,
                changed_aux_keys=changed_aux_keys,
                auxiliary=prepared_auxiliary,
                account_hash=account_dir_hash(source),
            ))

        # Every immutable decrypt generation must take ownership of the media
        # bindings for every imported account before older generations can be
        # retained away.  Only changed Moments need a full coordinate rebuild;
        # ordinary message deltas can move their immutable coordinates with a
        # set-based rebind.
        accounts_to_refresh = {
            item.importer.account_id
            for item in prepared_sources
            if item.auxiliary is not None and item.auxiliary.moment_importer is not None
        }
        source_snapshot: SourceSnapshot | None = None
        if prepared_sources:
            source_snapshot = inspect_source_snapshot(cfg, snapshot_dir)

        # All remaining work is target-database mutation/publication.  The
        # writer no longer covers source scans, model/provider work, or file I/O.
        with coordinated_vault_mutation(
            cfg,
            operation='sync',
            write_session=write_session,
        ):
            store = SQLiteStore(cfg.paths.sqlite_path)
            claimed_sync_generation = claim_sync_commit_generation(
                store,
                prepared_sync_generation,
            )
            if claimed_sync_generation is None:
                # Another sync published after this scan started.  Drop all
                # prepared rows before any prune, fingerprint update, or source
                # binding can overwrite that newer result.
                raise SyncCommitGenerationChanged()
            repo = WeChatRepository(store)
            media_repo = MultimodalRepository(store)
            media_linker = MediaLinker(media_repo)
            if source_snapshot is not None:
                persist_source_snapshot(store, source_snapshot)

            for item in prepared_sources:
                media_report = media_linker.link_references(
                    item.media_refs,
                    source_states=item.media_source_states,
                ).to_dict()
                media_seen += int(media_report.get('assets_seen') or 0)
                media_upserted += int(media_report.get('assets_upserted') or 0)
                media_links += int(media_report.get('links_upserted') or 0)
                media_accepted += int(media_report.get('accepted_links') or 0)
                media_excluded += int(media_report.get('excluded_links') or 0)
                media_changed_asset_ids.update(str(value) for value in media_report.get('changed_asset_ids') or ())
                for key, value in (media_report.get('metrics') or {}).items():
                    media_persist_metrics[str(key)] = media_persist_metrics.get(str(key), 0) + int(value)
                for key, value in (media_report.get('excluded_counts') or {}).items():
                    media_excluded_counts[str(key)] = media_excluded_counts.get(str(key), 0) + int(value)

                if item.auxiliary is not None:
                    aux_report = commit_prepared_auxiliary_sources(
                        item.auxiliary,
                        store=store,
                        repo=media_repo,
                        bind_source=False,
                    )
                    contacts_imported += aux_report.contacts_imported
                    moments_imported += aux_report.moments_imported
                    favorites_imported += aux_report.favorites_imported
                    aux_sources_imported += len(item.changed_aux_keys)
                    aux_fingerprint_updates.update(item.changed_aux_keys)
                    aux_dirty_refs.extend(aux_report.dirty_refs())
                    for family, citations in aux_report.changed_citations.items():
                        aux_projection_citations.setdefault(family, set()).update(citations)
                        if family in {'contact', 'moment'}:
                            auxiliary_profile_citations.update(citations)
                    for family, citations in aux_report.removed_citations.items():
                        aux_projection_citations.setdefault(family, set()).update(citations)
                        if family in {'contact', 'moment'} and citations:
                            auxiliary_profile_removal = True
                        aux_removed_refs.extend({
                            'citation': citation,
                            'account_id': item.importer.account_id,
                            'conversation_id': '',
                            'source_type': family,
                        } for citation in citations)
                    for family, count in aux_report.changed_families().items():
                        aux_changed_families[family] = aux_changed_families.get(family, 0) + count
                    for family, count in aux_report.removed_counts.items():
                        aux_removed_counts[family] = aux_removed_counts.get(family, 0) + count
                    for key, value in aux_report.scope_counts.items():
                        scope_counts[key] = scope_counts.get(key, 0) + value
                    for key, value in aux_report.excluded_counts.items():
                        excluded_counts[key] = excluded_counts.get(key, 0) + value

                if item.accounts or item.conversations or item.messages or (
                    options.full and options.since is None and options.limit_per_shard is None
                ):
                    delta = repo.apply_delta(
                        item.accounts,
                        item.conversations,
                        item.messages,
                        source_key=f'wechat:{item.importer.account_id}',
                        source_snapshot_complete=bool(
                            options.full and options.since is None and options.limit_per_shard is None
                        ),
                    )
                    changed_refs = list(delta.get('changed_refs') or ())
                    message_dirty_count += int(delta.get('dirty_recorded') or 0)
                    if int(delta.get('citations_changed') or 0):
                        message_chunk_reports.append(dict(delta.get('chunks') or {}))
                    changed_now = int(delta.get('messages_changed') or 0)
                    messages_imported += changed_now
                    if changed_now:
                        sources_imported += 1
                    profile_identities = {
                        str(value) for value in (delta.get('profile_identity_values') or [])
                        if str(value or '').strip()
                    }
                    changed_profile_identity_values.update(profile_identities)
                    if delta.get('profile_scope_changed') and not profile_identities:
                        profile_refresh_all = True
                    for ref in changed_refs:
                        if ref.get('account_id') and ref.get('conversation_id'):
                            changed_conversations.add((str(ref['account_id']), str(ref['conversation_id'])))

            if source_snapshot is not None:
                for item in prepared_sources:
                    bind = (
                        bind_account_assets
                        if item.importer.account_id in accounts_to_refresh
                        else rebind_account_assets
                    )
                    bind(
                        store,
                        account_id=item.importer.account_id,
                        snapshot=source_snapshot,
                        account_hash=item.account_hash,
                    )

            chunks = None
            if message_chunk_reports:
                chunks = {
                    'parents': sum(int(item.get('parents') or 0) for item in message_chunk_reports),
                    'chunks': sum(int(item.get('chunks') or 0) for item in message_chunk_reports),
                    'citations': sum(int(item.get('citations') or 0) for item in message_chunk_reports),
                    'deleted_chunks': sum(int(item.get('deleted_chunks') or 0) for item in message_chunk_reports),
                    'deleted_vectors': sum(int(item.get('deleted_vectors') or 0) for item in message_chunk_reports),
                    'conversations': len(changed_conversations),
                }
            family_chunk_reports: list[dict[str, Any]] = []
            for family, citations in sorted(aux_projection_citations.items()):
                if citations:
                    family_chunk_reports.append(store.rebuild_evidence_chunks_for_source_citations(family, sorted(citations)))
            family_chunks = None
            if family_chunk_reports:
                family_chunks = {
                    'parents': sum(int(item.get('parents') or 0) for item in family_chunk_reports),
                    'chunks': sum(int(item.get('chunks') or 0) for item in family_chunk_reports),
                    'citations': sum(int(item.get('citations') or 0) for item in family_chunk_reports),
                    'source_types': sorted(aux_projection_citations),
                }
            if aux_removed_refs:
                store.record_citation_tombstones(aux_removed_refs)
            for family, citations in aux_projection_citations.items():
                live = set(citations) - {ref['citation'] for ref in aux_removed_refs if ref['source_type'] == family}
                if live:
                    store.clear_citation_tombstones(live)
            updated = write_waterlines(store, waterline_updates)
            aux_state_updated = write_aux_fingerprints(store, aux_fingerprint_updates)
            dirty_count = message_dirty_count + record_dirty_citations(store, aux_dirty_refs)
            media_queue = enqueue_media_jobs(store, asset_ids=sorted(media_changed_asset_ids))
            profile_service = ProfileAutomationService(store)
            subscriptions_enabled = profile_service.has_enabled_subscriptions()
            auxiliary_identity_values = (
                profile_service.identity_values_for_changes(
                    citations=auxiliary_profile_citations,
                    asset_ids=media_changed_asset_ids,
                )
                if subscriptions_enabled else set()
            )
            changed_profile_identity_values.update(auxiliary_identity_values)
            unresolved_auxiliary_delta = bool(
                subscriptions_enabled
                and (
                    auxiliary_profile_removal
                    or (
                        (auxiliary_profile_citations or media_changed_asset_ids)
                        and not auxiliary_identity_values
                    )
                )
            )
            if not subscriptions_enabled:
                profile_refresh_queue = {
                    'status': 'skipped',
                    'reason': 'no_enabled_subscriptions',
                    'queued': 0,
                    'raw_content_included': False,
                }
            elif unresolved_auxiliary_delta or profile_refresh_all:
                profile_refresh_queue = ProfileAutomationService(store).enqueue_all(
                    reason=(
                        'sync_auxiliary_scope_delta'
                        if unresolved_auxiliary_delta else 'sync_message_scope_delta'
                    ),
                ) | {'status': 'queued'}
            elif changed_profile_identity_values:
                profile_refresh_queue = profile_service.enqueue_impacted(
                    changed_profile_identity_values,
                    reason='sync_message_delta',
                ) | {'status': 'queued'}
            pending_dirty_count = dirty_citation_count(store)
            pending_dirty_batch = read_dirty_citation_batch(
                store,
                limit=min(pending_dirty_count, sync_config.vector_inline_dirty_limit),
            )
            pending_dirty_citations = [citation for citation, _updated_at in pending_dirty_batch]
            complete_sync_commit_generation(store, claimed_sync_generation)

        if write_session is not None:
            vector = {
                'status': 'deferred',
                'reason': 'active_parent_writer',
                'dirty_count': pending_dirty_count,
                'processed_dirty_count': 0,
                'remaining_dirty_count': pending_dirty_count,
                'deferred_dirty_count': pending_dirty_count,
                'deferred': pending_dirty_count > 0,
            }
        else:
            vector = maybe_index_vectors(
                cfg,
                changed=bool(pending_dirty_count),
                backend=options.vector_backend,
                citations=pending_dirty_citations,
                total_dirty_count=pending_dirty_count,
                write_session=None,
            )
        vector['pending_dirty_count'] = pending_dirty_count
        processed = min(
            len(pending_dirty_batch),
            max(0, int(vector.get('processed_dirty_count') or 0)),
        )
        remaining_dirty_count = pending_dirty_count
        vector['dirty_cleared'] = 0
        if vector.get('status') == 'indexed' and processed:
            try:
                with coordinated_vault_mutation(cfg, operation='sync'):
                    clear_store = SQLiteStore(cfg.paths.sqlite_path)
                    vector['dirty_cleared'] = clear_dirty_citation_batch(
                        clear_store,
                        pending_dirty_batch[:processed],
                    )
                    remaining_dirty_count = dirty_citation_count(clear_store)
            except VaultOperationLocked:
                vector['dirty_clear_status'] = 'deferred_writer_lock'
        vector['processed_dirty_count'] = processed
        vector['remaining_dirty_count'] = remaining_dirty_count
        vector['deferred_dirty_count'] = remaining_dirty_count
        vector['deferred'] = remaining_dirty_count > 0
        vector['dirty_clear_cas_misses'] = max(0, processed - int(vector['dirty_cleared']))
        if write_session is None and options.profile_refresh_budget > 0:
            try:
                profile_refresh_worker = process_profile_refresh_queue(
                    cfg,
                    limit=min(5, options.profile_refresh_budget),
                )
            except Exception as exc:
                profile_refresh_worker = {
                    'ok': False,
                    'status': 'failed',
                    'error_code': str(getattr(exc, 'code', exc.__class__.__name__))[:100],
                    'processed': 0,
                    'raw_content_included': False,
                    'raw_paths_included': False,
                }
        else:
            profile_refresh_worker = {
                'ok': True,
                'status': 'deferred',
                'reason': (
                    'active_parent_writer'
                    if write_session is not None else 'bounded_sync_budget'
                ),
                'processed': 0,
                'raw_content_included': False,
                'raw_paths_included': False,
            }
        profile_refresh_failed = not bool(profile_refresh_worker.get('ok'))
        if profile_refresh_failed:
            errors.append('profile_refresh_partial')
        status = (
            'partial'
            if profile_refresh_failed
            else ('completed' if not errors else ('partial' if messages_imported else 'failed'))
        )
        report = SyncReport(
            ok=status in {'completed', 'partial'},
            status=status,
            snapshot={**snapshot_report, 'exists': True, 'media_cache': snapshot_media_report},
            sources_seen=sources_seen,
            sources_imported=sources_imported,
            messages_imported=messages_imported,
            conversations_changed=len(changed_conversations),
            dirty_count=dirty_count,
            chunks=chunks,
            media={
                'status': 'queued',
                'discovery_mode': options.media_discovery_mode,
                'assets_seen': media_seen,
                'assets_upserted': media_upserted,
                'links_upserted': media_links,
                'accepted_links': media_accepted,
                'excluded_links': media_excluded,
                'excluded_counts': dict(sorted(media_excluded_counts.items())),
                'changed_assets': len(media_changed_asset_ids),
                'discovery': dict(sorted(media_discovery_counters.items())),
                'persistence': dict(sorted(media_persist_metrics.items())),
                'jobs': media_queue,
                'raw_content_included': False,
                'raw_paths_included': False,
            },
            auxiliary={
                'status': 'imported' if aux_sources_imported else 'skipped',
                'sources_seen': aux_sources_seen,
                'sources_imported': aux_sources_imported,
                'contacts_imported': contacts_imported,
                'moments_imported': moments_imported,
                'favorites_imported': favorites_imported,
                'dirty_citations': len({ref['citation'] for ref in aux_dirty_refs if ref.get('citation')}),
                'changed_families': dict(sorted(aux_changed_families.items())),
                'removed_counts': dict(sorted(aux_removed_counts.items())),
                'chunks': family_chunks,
                'state_updated': aux_state_updated,
                'raw_content_included': False,
                'raw_paths_included': False,
                'd0_mapping_conclusion': d0_mapping_conclusion(),
            },
            waterlines_updated=updated,
            vector=vector,
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=errors[:50],
            profiles={
                'refresh_queue': profile_refresh_queue,
                'worker': profile_refresh_worker,
                'raw_content_included': False,
                'raw_paths_included': False,
            },
        )
    except SyncCommitGenerationChanged as exc:
        report = SyncReport(
            ok=False,
            status='retry_required',
            snapshot={**snapshot_report, 'exists': True, 'media_cache': snapshot_media_report},
            sources_seen=sources_seen,
            sources_imported=0,
            messages_imported=0,
            conversations_changed=0,
            dirty_count=0,
            chunks=None,
            media={'status': 'skipped', 'reason': exc.code},
            auxiliary={'status': 'skipped', 'reason': exc.code},
            waterlines_updated=0,
            vector={'status': 'deferred', 'reason': exc.code, 'retryable': True},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=[exc.code],
        )
    except VaultOperationLocked as exc:
        report = SyncReport(
            ok=False,
            status='locked',
            snapshot={**snapshot_report, 'exists': True, 'media_cache': snapshot_media_report},
            sources_seen=0,
            sources_imported=0,
            messages_imported=0,
            conversations_changed=0,
            dirty_count=0,
            chunks=None,
            media={'status': 'skipped', 'reason': 'writer_lock'},
            auxiliary={'status': 'skipped', 'reason': 'writer_lock'},
            waterlines_updated=0,
            vector={'status': 'blocked', 'reason': 'writer_lock'},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=[exc.__class__.__name__],
        )
    if report.ok:
        trace.complete(trace_id, {'messages_imported': report.messages_imported, 'conversations_changed': report.conversations_changed, 'chunks': report.chunks, 'media': report.media})
    else:
        trace.fail(trace_id, {'status': report.status, 'errors': report.errors})
    return report.to_dict()


def maybe_index_vectors(
    cfg: VaultConfig,
    *,
    changed: bool,
    backend: str = 'zvec',
    citations=None,
    total_dirty_count: int | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    citation_filter = None if citations is None else list(dict.fromkeys(str(c) for c in citations if c))
    effective_dirty_count = int(total_dirty_count) if total_dirty_count is not None else (len(citation_filter) if citation_filter is not None else None)

    def dirty_progress(processed: int = 0) -> dict[str, Any]:
        total = max(0, int(effective_dirty_count or 0))
        done = min(total, max(0, int(processed)))
        remaining = max(0, total - done)
        return {
            'dirty_count': total,
            'processed_dirty_count': done,
            'remaining_dirty_count': remaining,
            'deferred_dirty_count': remaining,
            'deferred': remaining > 0,
        }

    def deferred(reason: str, *, inline_limit: int | None = None) -> dict[str, Any]:
        result = {'status': 'deferred', 'reason': reason, **dirty_progress()}
        if inline_limit is not None:
            result['inline_limit'] = inline_limit
        return result

    inline_limit = None
    if citation_filter is not None:
        inline_limit = read_sync_config(cfg).vector_inline_dirty_limit
        citation_filter = citation_filter[:inline_limit]
        if not effective_dirty_count:
            return {'status': 'skipped', 'reason': 'no_dirty_citations', **dirty_progress()}
    if not changed and citation_filter is None:
        return {'status': 'skipped', 'reason': 'no_new_messages'}
    pcfg = (read_latest_process_config(cfg.root).get('config') or {})
    if pcfg.get('vector_index', 'diagnose_only') != 'incremental':
        if citation_filter is not None:
            return deferred('process_config_vector_index_not_incremental', inline_limit=inline_limit)
        return {'status': 'skipped', 'reason': 'process_config_vector_index_not_incremental', 'dirty_count': effective_dirty_count}
    if citation_filter is not None and not citation_filter:
        return deferred('vector_inline_dirty_limit_zero', inline_limit=inline_limit)
    try:
        provider = configured_embedding_provider(strict=False, vault_root=cfg.root)
    except Exception:
        provider = None
    if provider is None:
        if citation_filter is not None:
            return deferred('embedding_provider_unavailable', inline_limit=inline_limit)
        return {'status': 'skipped', 'reason': 'embedding_provider_unavailable', 'dirty_count': effective_dirty_count}
    vector_status = vector_status_payload(cfg, backend=backend, provider=provider)
    backend_status = vector_status.get(backend) if isinstance(vector_status.get(backend), dict) else {}
    zvec_rebuild_reason = _zvec_rebuild_gate_reason(backend, backend_status, vector_status)
    if zvec_rebuild_reason is not None:
        return {
            'status': 'recommend_rebuild',
            'backend': backend,
            'reason_code': zvec_rebuild_reason,
            **(dirty_progress() if citation_filter is not None else {'dirty_count': effective_dirty_count}),
            'auto_rebuild': False,
        }
    if backend_status.get('rebuild_required'):
        return {
            'status': 'recommend_rebuild',
            'backend': backend,
            'reason_code': backend_status.get('reason_code') or vector_status.get('reason_code'),
            **(dirty_progress() if citation_filter is not None else {'dirty_count': effective_dirty_count}),
            'auto_rebuild': False,
        }
    try:
        data = index_vectors(
            cfg,
            provider,
            backend=backend,
            purge=False,
            citations=citation_filter,
            write_session=write_session,
        )
    except Exception as exc:
        return {
            'status': 'failed',
            'reason': exc.__class__.__name__,
            **(dirty_progress() if citation_filter is not None else {'dirty_count': effective_dirty_count}),
        }
    report = {
        'status': 'indexed',
        'backend': data.get('backend'),
        'indexed': data.get('indexed'),
        'dirty_count': effective_dirty_count if citation_filter is not None else data.get('dirty_count'),
    }
    if citation_filter is not None:
        report.update(dirty_progress(len(citation_filter)))
        report['inline_limit'] = inline_limit
    return report


def _zvec_rebuild_gate_reason(backend: str, backend_status: dict[str, Any], vector_status: dict[str, Any]) -> str | None:
    if backend != 'zvec':
        return None
    reason = backend_status.get('reason_code') or vector_status.get('reason_code') or 'zvec_rebuild_required'
    if backend_status.get('collection_exists') is False:
        return 'zvec_collection_missing'
    if 'metadata_complete' in backend_status and backend_status.get('metadata_complete') is not True:
        return reason
    if backend_status.get('provider_mismatch') or backend_status.get('stale'):
        return reason
    return None


def watch_sync(
    vault_root: str | Path | None = None,
    *,
    options: SyncOptions | None = None,
    once: bool = False,
    backend: WatchBackend | None = None,
) -> None:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    cfg.require_configured_for_write('watch sync')
    sync_config = read_sync_config(cfg)
    options = options or SyncOptions()
    snapshot_dir = resolve_snapshot_dir(cfg, sync_config, options.snapshot_dir)
    active_backend = backend or create_watch_backend(
        snapshot_dir,
        cfg.root.joinpath(*SYNC_WATCH_MANIFEST),
    )
    pending_change = False
    last_change_at = 0.0
    stable_manifest_digest: str | None = None
    stable_scan_at = 0.0
    next_profile_poll_at = 0.0
    try:
        while True:
            tick = active_backend.poll(timeout=0.0 if once else 1.0)
            now = time.monotonic()
            if tick.changed or tick.event_loss:
                pending_change = True
                last_change_at = now
                stable_manifest_digest = None
                stable_scan_at = 0.0
            if (
                pending_change
                and tick.scan_complete
                and not tick.scan_discarded
                and tick.error_code is None
                and tick.manifest_digest
                and tick.manifest_digest != stable_manifest_digest
            ):
                stable_manifest_digest = tick.manifest_digest
                stable_scan_at = now
            debounce = max(0.0, sync_config.debounce_seconds)
            stable_since = max(last_change_at, stable_scan_at)
            if (
                pending_change
                and stable_manifest_digest
                and not tick.scan_active
                and tick.error_code is None
                and now - stable_since >= debounce
            ):
                data = run_sync(cfg.root, options=options)
                print(json.dumps(data, ensure_ascii=False), flush=True)
                pending_change = False
                stable_manifest_digest = None
                next_profile_poll_at = 0.0
                active_backend.request_repair(reason='post_sync')
                if once:
                    return
                continue
            if (
                not pending_change
                and not tick.scan_active
                and now >= next_profile_poll_at
            ):
                try:
                    profile_refresh = process_profile_refresh_queue(cfg, limit=1)
                except Exception as exc:
                    profile_refresh = {
                        'ok': False,
                        'status': 'failed',
                        'error_code': str(getattr(exc, 'code', exc.__class__.__name__))[:100],
                        'processed': 0,
                        'raw_content_included': False,
                        'raw_paths_included': False,
                    }
                # Drain a known capped backlog immediately, but keep a quiet
                # watcher from opening SQLite every five seconds all day.
                next_profile_poll_at = now + (
                    0.0 if profile_refresh.get('drained') is False else 60.0
                )
                if profile_refresh.get('processed') or not profile_refresh.get('ok'):
                    print(json.dumps({'profile_automation': profile_refresh}, ensure_ascii=False), flush=True)
            if once:
                if pending_change and stable_manifest_digest:
                    time.sleep(max(0.0, debounce - (now - stable_since)))
                    continue
                if not pending_change and (tick.scan_complete or tick.error_code):
                    return
    finally:
        active_backend.close()
