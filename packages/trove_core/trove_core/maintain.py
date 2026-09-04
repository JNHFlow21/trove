from __future__ import annotations

from contextlib import closing

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import time
from typing import Any

from trove_core.runtime import configured_embedding_provider, index_vectors, vector_status_payload
from trove_core.media_pipeline import media_status_payload
from trove_core.knowledge.profile_automation import ProfileAutomationService, process_profile_refresh_queue
from trove_core.store.schema import SCHEMA_VERSION
from trove_core.store.sqlite_store import FTS_TOKENIZER_VERSION, SQLiteStore
from trove_core.store.change_journal import (
    MAX_INLINE_DIRTY_CITATIONS,
    clear_dirty_citation_batch,
    dirty_citation_count,
    read_dirty_citation_batch,
)
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.locks import VaultOperationLocked, active_vector_progress
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint, record_vault_mutation_noop
from trove_core.vault.tracing import TraceTimeline
from trove_core.wechat.process_config import read_latest_process_config


@dataclass(frozen=True)
class MaintainOptions:
    auto_rebuild: bool = False
    backup_retention: int = 3
    log_retention: int = 5
    max_log_bytes: int = 5 * 1024 * 1024
    vector_backend: str = 'zvec'
    model_path: str | None = None
    vacuum: bool = False
    always_backup: bool = False
    # Scheduled maintenance is database/index housekeeping. Model work belongs
    # to dedicated commands so a nightly maintain cannot spend minutes
    # downloading/loading ASR while monopolizing the Vault writer.
    media_voice_budget: int = 0
    media_image_budget: int = 0
    media_caption_budget: int = 0
    # Expensive corpus-wide integrity scans are explicit repair work. Ordinary
    # maintain consumes persisted dirty journals/queues only.
    full_scan: bool = False


@dataclass(frozen=True)
class MaintainReport:
    ok: bool
    status: str
    schema: dict[str, Any]
    chunks: dict[str, Any]
    fts: dict[str, Any]
    vector: dict[str, Any]
    media: dict[str, Any]
    storage: dict[str, Any]
    backups: dict[str, Any]
    logs: dict[str, Any]
    elapsed_ms: float
    trace_id: str
    errors: list[str]
    raw_content_included: bool = False
    profiles: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@mutation_entrypoint('maintain')
def run_maintain(
    vault_root: str | Path | None = None,
    *,
    options: MaintainOptions | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    started = time.time()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    try:
        return _run_maintain_coordinated(
            vault_root,
            options=options,
            write_session=write_session,
        )
    except VaultOperationLocked as exc:
        return MaintainReport(
            ok=False,
            status='locked',
            schema={},
            chunks={},
            fts={},
            vector={'status': 'blocked', 'reason': 'writer_lock'},
            media={'status': 'skipped', 'reason': 'writer_lock'},
            storage={},
            backups={},
            logs={},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id='',
            errors=[exc.__class__.__name__],
        ).to_dict()


def _run_maintain_coordinated(
    vault_root: str | Path | None = None,
    *,
    options: MaintainOptions | None = None,
    write_session: VaultWriteSession | None,
) -> dict[str, Any]:
    started = time.time()
    options = options or MaintainOptions()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
    cfg.ensure()
    trace = TraceTimeline(cfg.root)
    trace_id = trace.start('maintain', {'auto_rebuild': options.auto_rebuild, 'vector_backend': options.vector_backend})
    vector_progress = active_vector_progress(cfg)
    if vector_progress:
        # This is a proven no-write admission refusal; keep the public mutation
        # coverage audit explicit even though no writer should be acquired.
        record_vault_mutation_noop(operation='maintain')
        report = MaintainReport(
            ok=False,
            status='blocked_by_vector_rebuild',
            schema={},
            chunks={},
            fts={},
            vector={'status': 'blocked', 'reason': 'active_vector_progress', **vector_progress},
            media={'status': 'skipped', 'reason': 'blocked_by_vector_rebuild'},
            storage={},
            backups={},
            logs={},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=[],
        )
        trace.fail(trace_id, {'status': report.status})
        return report.to_dict()

    errors: list[str] = []
    initial_meta = inspect_schema_meta(cfg.paths.sqlite_path)
    try:
        # A pre-migration backup is a consistent SQLite snapshot and does not
        # require blocking unrelated Vault writers while bytes are copied.
        pre_init_backup = options.always_backup or options.vacuum or _initial_meta_needs_backup(initial_meta)
        backups = rotate_sqlite_backups(
            cfg.paths.sqlite_path,
            retention=options.backup_retention,
            create=pre_init_backup,
        )

        # Acquire the writer only for schema initialization/migration, then
        # release it before integrity planning and other read-only scans.
        with coordinated_vault_mutation(
            cfg,
            operation='maintain',
            write_session=write_session,
        ):
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()

        read_store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
        read_store.initialize()
        orphan_needed = orphan_cleanup_needed(read_store, full_scan=options.full_scan)
        fts_needed = fts_repair_needed(read_store, initial_meta, full_scan=options.full_scan)
        vacuum_needed = storage_vacuum_needed(read_store, vacuum=options.vacuum)
        destructive_reasons: list[str] = []
        if options.always_backup:
            destructive_reasons.append('always_backup')
        if orphan_needed:
            destructive_reasons.append('purge_orphans')
        if fts_needed:
            destructive_reasons.append('rebuild_fts')
        if vacuum_needed:
            destructive_reasons.append('vacuum')
        if not backups.get('created') and destructive_reasons:
            backups = rotate_sqlite_backups(
                cfg.paths.sqlite_path,
                retention=options.backup_retention,
                create=True,
            )
        backups['destructive_reasons'] = destructive_reasons

        # The writer now covers only the actual SQLite mutation/commit window.
        with coordinated_vault_mutation(
            cfg,
            operation='maintain',
            write_session=write_session,
        ) as active_session:
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()
            schema = ensure_schema_report(store, initial_meta)
            chunks = cleanup_orphan_chunks(store) if orphan_needed else {
                'removed_orphan_chunks': 0,
                'removed_orphan_vectors': 0,
                'scan_mode': 'dirty_only',
            }
            fts = repair_fts_if_needed(
                store,
                initial_meta,
                repair_needed=fts_needed,
                full_scan=options.full_scan,
            )
            storage = optimize_storage(
                store,
                vacuum=options.vacuum,
                optimize_fts=options.full_scan,
            )

        # Routine maintain never downloads/loads a model or calls a provider.
        # Dedicated vector/media commands own those explicit workloads; this
        # command reports their durable backlog without holding the DB writer.
        vector = maintain_vectors(cfg, options=options, execute=False)
        media_backlog = media_status_payload(cfg.root).get('queue', {})
        requested_media = (
            int(options.media_voice_budget)
            + int(options.media_image_budget)
            + int(options.media_caption_budget)
        )
        media = {
            'status': 'deferred' if requested_media else 'healthy',
            'reason': 'explicit_media_command_required' if requested_media else 'no_model_work_requested',
            'requested_voice_budget': int(options.media_voice_budget),
            'requested_image_budget': int(options.media_image_budget),
            'requested_caption_budget': int(options.media_caption_budget),
            'backlog': media_backlog,
            'raw_content_included': False,
            'raw_paths_included': False,
        }
        # Daily maintenance is the safety net for profile-affecting local
        # writes that did not originate from sync. Content hashes keep this
        # reconciliation idempotent and prevent duplicate snapshot versions.
        try:
            with coordinated_vault_mutation(
                cfg,
                operation='profile_automation',
                write_session=write_session,
            ):
                profile_store = SQLiteStore(cfg.paths.sqlite_path)
                try:
                    profile_reconcile = ProfileAutomationService(profile_store).enqueue_all_if_source_changed(
                        reason='maintenance_reconcile', debounce_override_seconds=0,
                    )
                finally:
                    profile_store.close()
            profiles = process_profile_refresh_queue(
                cfg, limit=20, write_session=write_session,
            ) | {
                'reconcile_queue': profile_reconcile,
            }
        except Exception as exc:
            profiles = {
                'ok': False,
                'status': 'failed',
                'error_code': str(getattr(exc, 'code', exc.__class__.__name__))[:100],
                'processed': 0,
                'raw_content_included': False,
                'raw_paths_included': False,
            }
        logs = rotate_logs(cfg.paths.logs_dir, retention=options.log_retention, max_bytes=options.max_log_bytes)
        if not profiles.get('ok'):
            errors.append('profile_refresh_partial')
        status = 'action_required' if (
            options.auto_rebuild or requested_media or not profiles.get('ok')
        ) else 'completed'
    except VaultOperationLocked as exc:
        report = MaintainReport(
            ok=False,
            status='locked',
            schema={},
            chunks={},
            fts={},
            vector={'status': 'blocked', 'reason': 'writer_lock'},
            media={'status': 'skipped', 'reason': 'writer_lock'},
            storage={},
            backups={},
            logs={},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=[exc.__class__.__name__],
        )
        trace.fail(trace_id, {'status': report.status})
        return report.to_dict()
    except Exception as exc:
        errors.append(exc.__class__.__name__)
        report = MaintainReport(
            ok=False,
            status='failed',
            schema={},
            chunks={},
            fts={},
            vector={},
            media={},
            storage={},
            backups={},
            logs={},
            elapsed_ms=round((time.time() - started) * 1000, 3),
            trace_id=trace_id,
            errors=errors,
        )
        trace.fail(trace_id, {'errors': errors})
        return report.to_dict()

    report = MaintainReport(
        ok=status == 'completed',
        status=status,
        schema=schema,
        chunks=chunks,
        fts=fts,
        vector=vector,
        media=media,
        storage=storage,
        backups=backups,
        logs=logs,
        elapsed_ms=round((time.time() - started) * 1000, 3),
        trace_id=trace_id,
        errors=errors,
        profiles=profiles,
    )
    trace.complete(trace_id, {
        'status': status,
        'commit_count': storage.get('commit_count', 0),
        'wal_bytes': storage.get('wal_bytes', 0),
        'resource_count': storage.get('resource_count', 0),
        'scan_count': int(chunks.get('removed_orphan_chunks', 0)) + int(chunks.get('removed_orphan_vectors', 0)),
    })
    return report.to_dict()


def inspect_schema_meta(sqlite_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {'exists': False}
    try:
        with closing(sqlite3.connect(sqlite_path)) as conn:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'").fetchone()
            user_version = int(conn.execute('PRAGMA user_version').fetchone()[0])
            user_tables = [
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table') AND name NOT LIKE 'sqlite_%'"
                )
            ]
            meta: dict[str, Any] = {'exists': True, 'user_version': user_version, 'table_count': len(user_tables)}
            if table:
                for key in ('schema_version', 'fts_tokenizer'):
                    row = conn.execute('SELECT value FROM schema_meta WHERE key=?', (key,)).fetchone()
                    if row:
                        meta[key] = row[0]
            return meta
    except sqlite3.DatabaseError as exc:
        return {'exists': True, 'error_code': exc.__class__.__name__}


def ensure_schema_report(store: SQLiteStore, initial_meta: dict[str, Any]) -> dict[str, Any]:
    version = store.schema_version()
    with store.connect() as conn:
        tokenizer = _schema_value(conn, 'fts_tokenizer')
    return {
        'schema_version': version,
        'expected_schema_version': SCHEMA_VERSION,
        'schema_repaired': int(initial_meta.get('schema_version') or initial_meta.get('user_version') or 0) != SCHEMA_VERSION,
        'fts_tokenizer': tokenizer,
        'expected_fts_tokenizer': FTS_TOKENIZER_VERSION,
        'fts_tokenizer_repaired': bool(initial_meta.get('fts_tokenizer') and initial_meta.get('fts_tokenizer') != FTS_TOKENIZER_VERSION and tokenizer == FTS_TOKENIZER_VERSION),
    }


def _initial_meta_needs_backup(initial_meta: dict[str, Any]) -> bool:
    if not initial_meta.get('exists'):
        return False
    if initial_meta.get('error_code'):
        return True
    if int(initial_meta.get('table_count') or 0) == 0:
        return False
    if initial_meta.get('schema_version') is None:
        return True
    return initial_meta.get('fts_tokenizer') != FTS_TOKENIZER_VERSION


def orphan_cleanup_needed(store: SQLiteStore, *, full_scan: bool = True) -> bool:
    if not full_scan:
        return False
    store.initialize()
    with store.connect() as conn:
        if _table_exists(conn, 'evidence_chunks'):
            row = conn.execute(
                """SELECT 1 FROM evidence_chunks e
                   WHERE e.source_type='message'
                     AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.citation=e.parent_citation)
                   LIMIT 1"""
            ).fetchone()
            if row is not None:
                return True
        if _table_exists(conn, 'vector_entries'):
            row = conn.execute(
                """SELECT 1 FROM vector_entries
                   WHERE citation NOT IN (SELECT citation FROM messages)
                     AND citation NOT IN (SELECT chunk_citation FROM evidence_chunks WHERE status='active')
                   LIMIT 1"""
            ).fetchone()
            if row is not None:
                return True
    return False


def fts_repair_needed(
    store: SQLiteStore,
    initial_meta: dict[str, Any],
    *,
    full_scan: bool = True,
) -> bool:
    store.initialize()
    with store.connect() as conn:
        tokenizer = _schema_value(conn, 'fts_tokenizer')
    drift = tokenizer != FTS_TOKENIZER_VERSION
    mismatch = False
    if full_scan:
        with store.connect() as conn:
            before = _fts_count_report(conn)
        mismatch = before.get('message_rows') != before.get('message_fts_rows') or before.get('chunk_rows') != before.get('chunk_fts_rows')
    return bool(drift or mismatch or initial_meta.get('fts_tokenizer') not in {None, FTS_TOKENIZER_VERSION})


def cleanup_orphan_chunks(store: SQLiteStore) -> dict[str, Any]:
    store.initialize()
    removed_chunks = 0
    removed_vectors = 0
    with store.connect() as conn:
        if not _table_exists(conn, 'evidence_chunks'):
            return {'removed_orphan_chunks': 0, 'removed_orphan_vectors': 0}
        orphan_rows = list(conn.execute(
                """SELECT chunk_citation,parent_citation,account_id,source_id,source_type FROM evidence_chunks e
                   WHERE e.source_type='message'
                     AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.citation=e.parent_citation)"""
            ))
        orphans = [row['chunk_citation'] for row in orphan_rows]
        dirty_recorded = store._record_dirty_refs_conn(conn, (
            {
                'citation': str(row['parent_citation'] or ''),
                'account_id': str(row['account_id'] or ''),
                'conversation_id': str(row['source_id'] or ''),
                'source_type': str(row['source_type'] or 'message'),
            }
            for row in orphan_rows
        ))
        if orphans:
            for start in range(0, len(orphans), 500):
                batch = orphans[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                conn.execute(f'DELETE FROM evidence_chunks WHERE chunk_citation IN ({placeholders})', batch)
                removed_chunks += len(batch)
                if _table_exists(conn, 'vector_entries'):
                    cursor = conn.execute(f'DELETE FROM vector_entries WHERE citation IN ({placeholders})', batch)
                    removed_vectors += max(0, cursor.rowcount)
        if _table_exists(conn, 'vector_entries'):
            cursor = conn.execute(
                """DELETE FROM vector_entries
                   WHERE citation NOT IN (SELECT citation FROM messages)
                     AND citation NOT IN (SELECT chunk_citation FROM evidence_chunks WHERE status='active')"""
            )
            removed_vectors += max(0, cursor.rowcount)
        conn.commit()
    return {'removed_orphan_chunks': removed_chunks, 'removed_orphan_vectors': removed_vectors, 'dirty_recorded': dirty_recorded}


def repair_fts_if_needed(
    store: SQLiteStore,
    initial_meta: dict[str, Any],
    *,
    repair_needed: bool | None = None,
    full_scan: bool = True,
) -> dict[str, Any]:
    store.initialize()
    with store.connect() as conn:
        tokenizer = _schema_value(conn, 'fts_tokenizer')
        before = _fts_count_report(conn) if full_scan else {'verification': 'structural'}
    if repair_needed is None:
        repair_needed = fts_repair_needed(store, initial_meta, full_scan=full_scan)
    repaired = False
    rebuild_report = None
    if repair_needed:
        rebuild_report = store.rebuild_fts()
        repaired = True
    with store.connect() as conn:
        after = _fts_count_report(conn) if full_scan else {
            'verification': 'structural',
            'fts_tokenizer': _schema_value(conn, 'fts_tokenizer'),
        }
    return {
        'repaired': repaired,
        'reason': 'tokenizer_or_count_drift' if repaired else 'healthy',
        'scan_mode': 'full' if full_scan else 'structural',
        'before': before,
        'after': after,
        'rebuild': rebuild_report,
    }


def optimize_storage(
    store: SQLiteStore,
    *,
    vacuum: bool = False,
    optimize_fts: bool = True,
) -> dict[str, Any]:
    store.initialize()
    with store.connect() as conn:
        fts_optimized: list[str] = []
        if optimize_fts:
            for table in ('message_fts', 'chunk_fts'):
                if _table_exists(conn, table):
                    try:
                        conn.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")
                        fts_optimized.append(table)
                    except sqlite3.DatabaseError:
                        pass
        pragma_optimized = False
        if optimize_fts:
            try:
                conn.execute('PRAGMA optimize')
                pragma_optimized = True
            except sqlite3.DatabaseError:
                pass
        page_count = int(conn.execute('PRAGMA page_count').fetchone()[0])
        freelist_count = int(conn.execute('PRAGMA freelist_count').fetchone()[0])
        conn.commit()
    vacuumed = False
    if vacuum or (page_count > 0 and freelist_count / max(1, page_count) > 0.5 and store.path.exists() and store.path.stat().st_size < 128 * 1024 * 1024):
        with closing(sqlite3.connect(store.path)) as conn:
            conn.execute('VACUUM')
            conn.commit()
        vacuumed = True
    wal_path = Path(str(store.path) + '-wal')
    try:
        wal_bytes = int(wal_path.stat().st_size)
    except OSError:
        wal_bytes = 0
    return {
        'fts_optimized': fts_optimized,
        'pragma_optimize': pragma_optimized,
        'page_count': page_count,
        'freelist_count': freelist_count,
        'vacuumed': vacuumed,
        'commit_count': 1 + int(vacuumed),
        'wal_bytes': wal_bytes,
        'resource_count': int(store.active_connection_count),
    }


def storage_vacuum_needed(store: SQLiteStore, *, vacuum: bool = False) -> bool:
    if vacuum:
        return True
    store.initialize()
    with store.connect() as conn:
        page_count = int(conn.execute('PRAGMA page_count').fetchone()[0])
        freelist_count = int(conn.execute('PRAGMA freelist_count').fetchone()[0])
    return bool(page_count > 0 and freelist_count / max(1, page_count) > 0.5 and store.path.exists() and store.path.stat().st_size < 128 * 1024 * 1024)


def maintain_vectors(
    cfg: VaultConfig,
    *,
    options: MaintainOptions,
    write_session: VaultWriteSession | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    if write_session is not None:
        write_session.validate_for(cfg)
        if execute:
            raise RuntimeError('vector preparation cannot run inside an outer maintain writer session')
    pcfg = (read_latest_process_config(cfg.root).get('config') or {})
    provider = configured_embedding_provider(options.model_path, strict=False, vault_root=cfg.root)
    status = vector_status_payload(cfg, backend=options.vector_backend, provider=provider)
    backend_status = status.get(options.vector_backend) if isinstance(status.get(options.vector_backend), dict) else {}
    # Status-only maintenance must remain a true read path after the writer
    # window closes.  It must not run schema initialization or WAL setup on a
    # writable handle by accident.
    store = SQLiteStore(cfg.paths.sqlite_path, readonly=not execute)
    dirty_count = dirty_citation_count(store)
    dirty_batch = (
        read_dirty_citation_batch(
            store,
            limit=dirty_count if options.auto_rebuild else min(dirty_count, MAX_INLINE_DIRTY_CITATIONS),
        )
        if execute and dirty_count
        else []
    )
    dirty_citations = [citation for citation, _updated_at in dirty_batch]
    if not execute and options.auto_rebuild:
        return {
            'status': 'deferred',
            'backend': options.vector_backend,
            'reason_code': 'explicit_vector_rebuild_required',
            'dirty_count': dirty_count,
        }
    if provider is None:
        return {'status': 'skipped', 'reason': 'embedding_provider_unavailable', 'backend': options.vector_backend, 'state': status.get('state'), 'dirty_count': dirty_count}
    stale = bool(backend_status.get('stale') or backend_status.get('provider_mismatch'))
    incomplete = bool(backend_status.get('incomplete'))
    catchup_pending = bool(backend_status.get('catchup_pending'))
    incremental_configured = pcfg.get('vector_index') == 'incremental'
    incremental_repair_needed = incremental_configured and (incomplete or catchup_pending or dirty_count > 0) and not stale
    zvec_rebuild_reason = _zvec_rebuild_gate_reason(options.vector_backend, backend_status, status)
    if not execute:
        if zvec_rebuild_reason is not None or backend_status.get('rebuild_required'):
            return {
                'status': 'recommend_rebuild',
                'backend': options.vector_backend,
                'reason_code': zvec_rebuild_reason or backend_status.get('reason_code') or status.get('reason_code'),
                'dirty_count': dirty_count,
            }
        if dirty_count > 0:
            return {
                'status': 'deferred',
                'backend': options.vector_backend,
                'reason_code': 'explicit_vector_index_required',
                'dirty_count': dirty_count,
                'dirty_batch_count': min(dirty_count, MAX_INLINE_DIRTY_CITATIONS),
            }
        if incremental_repair_needed:
            return {
                'status': 'full_scan_required',
                'backend': options.vector_backend,
                'reason_code': 'dirty_journal_gap',
                'dirty_count': 0,
            }
        return {
            'status': 'healthy',
            'backend': options.vector_backend,
            'state': backend_status.get('state') or status.get('state'),
            'dirty_count': dirty_count,
        }
    if dirty_count > 0 and incremental_repair_needed and not options.auto_rebuild:
        if zvec_rebuild_reason is not None:
            return {
                'status': 'recommend_rebuild',
                'backend': options.vector_backend,
                'reason_code': zvec_rebuild_reason,
                'auto_rebuild': False,
                'dirty_count': dirty_count,
            }
        try:
            data = index_vectors(
                cfg,
                provider,
                backend=options.vector_backend,
                purge=False,
                citations=dirty_citations,
                write_session=None,
            )
        except Exception as exc:
            return {'status': 'failed', 'reason': exc.__class__.__name__, 'backend': options.vector_backend, 'dirty_count': dirty_count}
        with coordinated_vault_mutation(cfg, operation='maintain'):
            cleared = clear_dirty_citation_batch(store, dirty_batch)
        return {
            'status': 'indexed',
            'backend': data.get('backend'),
            'indexed': data.get('indexed'),
            'auto_rebuild': False,
            'dirty_count': dirty_count,
            'dirty_batch_count': len(dirty_citations),
            'dirty_cleared': cleared,
            'dirty_remaining': max(0, dirty_count - cleared),
        }
    if incremental_repair_needed and dirty_count == 0 and not options.auto_rebuild:
        return {
            'status': 'full_scan_required',
            'backend': options.vector_backend,
            'reason_code': 'dirty_journal_gap',
            'auto_rebuild': False,
            'dirty_count': 0,
        }
    if options.auto_rebuild:
        try:
            data = index_vectors(
                cfg,
                provider,
                backend=options.vector_backend,
                purge=False,
                write_session=None,
            )
        except Exception as exc:
            return {'status': 'failed', 'reason': exc.__class__.__name__, 'backend': options.vector_backend, 'dirty_count': dirty_count}
        if dirty_batch:
            with coordinated_vault_mutation(cfg, operation='maintain'):
                cleared = clear_dirty_citation_batch(store, dirty_batch)
        else:
            cleared = 0
        return {'status': 'indexed', 'backend': data.get('backend'), 'indexed': data.get('indexed'), 'auto_rebuild': options.auto_rebuild, 'dirty_count': dirty_count, 'dirty_cleared': cleared}
    if zvec_rebuild_reason is not None:
        return {'status': 'recommend_rebuild', 'backend': options.vector_backend, 'reason_code': zvec_rebuild_reason, 'auto_rebuild': False, 'dirty_count': dirty_count}
    if backend_status.get('rebuild_required'):
        return {'status': 'recommend_rebuild', 'backend': options.vector_backend, 'reason_code': backend_status.get('reason_code') or status.get('reason_code'), 'auto_rebuild': False, 'dirty_count': dirty_count}
    return {'status': 'healthy', 'backend': options.vector_backend, 'state': backend_status.get('state') or status.get('state'), 'dirty_count': dirty_count}


def _zvec_rebuild_gate_reason(vector_backend: str, backend_status: dict[str, Any], status: dict[str, Any]) -> str | None:
    if vector_backend != 'zvec':
        return None
    reason = backend_status.get('reason_code') or status.get('reason_code') or 'zvec_rebuild_required'
    if backend_status.get('collection_exists') is False:
        return 'zvec_collection_missing'
    if 'metadata_complete' in backend_status and backend_status.get('metadata_complete') is not True:
        return reason
    if backend_status.get('provider_mismatch') or backend_status.get('stale'):
        return reason
    return None


def rotate_sqlite_backups(sqlite_path: Path, *, retention: int, create: bool = True) -> dict[str, Any]:
    retention = max(0, int(retention))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    created = None
    if create and sqlite_path.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        dest = sqlite_path.with_name(f'{sqlite_path.name}.bak-{stamp}')
        with closing(sqlite3.connect(sqlite_path)) as source, closing(sqlite3.connect(dest)) as target:
            source.backup(target)
            target.commit()
        dest.chmod(0o600)
        created = dest.name
    backups = sorted(sqlite_path.parent.glob(f'{sqlite_path.name}.bak-*'), key=lambda p: p.stat().st_mtime, reverse=True)
    removed: list[str] = []
    for old in backups[retention:]:
        try:
            old.unlink()
            removed.append(old.name)
        except OSError:
            pass
    retained = [p.name for p in sorted(sqlite_path.parent.glob(f'{sqlite_path.name}.bak-*'), key=lambda p: p.stat().st_mtime, reverse=True)]
    return {
        'created': created,
        'skipped_reason': None if create else 'no_destructive_repair',
        'retention': retention,
        'retained': retained,
        'removed': removed,
        'retained_count': len(retained),
        'removed_count': len(removed),
    }


def rotate_logs(logs_dir: Path, *, retention: int, max_bytes: int) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    rotated: list[str] = []
    removed: list[str] = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for log in sorted(logs_dir.glob('*.log')):
        try:
            if log.stat().st_size <= max_bytes:
                continue
            dest = log.with_name(f'{log.name}.bak-{stamp}')
            log.replace(dest)
            log.write_text('', encoding='utf-8')
            rotated.append(dest.name)
        except OSError:
            continue
    for family in set(p.name.split('.bak-')[0] for p in logs_dir.glob('*.log.bak-*')):
        entries = sorted(logs_dir.glob(f'{family}.bak-*'), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in entries[max(0, retention):]:
            try:
                old.unlink()
                removed.append(old.name)
            except OSError:
                pass
    return {'rotated': rotated, 'removed': removed, 'retention': retention, 'max_bytes': max_bytes}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _schema_value(conn: sqlite3.Connection, key: str) -> str | None:
    if not _table_exists(conn, 'schema_meta'):
        return None
    row = conn.execute('SELECT value FROM schema_meta WHERE key=?', (key,)).fetchone()
    return row[0] if row else None


def _safe_count(conn: sqlite3.Connection, table: str, where: str = '1=1') -> int:
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {where}').fetchone()[0])


def _fts_count_report(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        'message_rows': _safe_count(conn, 'messages'),
        'message_fts_rows': _safe_count(conn, 'message_fts'),
        'chunk_rows': _safe_count(conn, 'evidence_chunks', "status='active'"),
        'chunk_fts_rows': _safe_count(conn, 'chunk_fts'),
    }
