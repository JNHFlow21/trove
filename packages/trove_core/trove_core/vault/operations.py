from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os
import shutil
from typing import Any

from .config import VaultConfig, path_is_under
from .coordinator import VaultWriteSession
from .mutations import coordinated_vault_mutation, mutation_entrypoint


DERIVED_DATA_LIFECYCLE_VERSION = 'derived-data/v1'
DERIVED_DATA_AUDIT_RETENTION_DAYS = 365
DERIVED_DATA_LIFECYCLE = (
    {'order': 10, 'artifact': 'task_capabilities_and_leases', 'retention': 'until completion, expiry, cancellation, or purge', 'purge': 'database overwrite/delete'},
    {'order': 20, 'artifact': 'temporary_materialization_files', 'retention': 'request lifetime only', 'purge': 'always unlink on success, failure, interruption, expiry, cancellation, or purge'},
    {'order': 30, 'artifact': 'previews_and_keyframes', 'retention': 'bounded local LRU', 'purge': 'clear preview cache for every derived-data purge'},
    {'order': 40, 'artifact': 'provider_jobs_and_payloads', 'retention': 'status, usage, cost, and request hash only; request/response bodies never logged', 'purge': 'delete jobs linked to selected assets/citations'},
    {'order': 50, 'artifact': 'transcripts_image_observations_understanding', 'retention': 'until source/person/task purge or explicit invalidation', 'purge': 'delete projections and content-hash cache'},
    {'order': 60, 'artifact': 'evidence_chunks_vectors_payloads', 'retention': 'while parent evidence remains queryable', 'purge': 'delete citation closure and search derivatives'},
    {'order': 70, 'artifact': 'profile_tasks_and_snapshots', 'retention': 'until owner revoke, source/person/task purge, or replacement', 'purge': 'delete tasks/snapshots and cancel surviving partial runs'},
    {'order': 80, 'artifact': 'materialized_media_and_decode_derivatives', 'retention': 'local cache until purge', 'purge': 'unlink only Vault-owned cache paths'},
    {'order': 90, 'artifact': 'related_approval_records', 'retention': 'until linked task/source/person purge', 'purge': 'delete exact redacted record and consumption claim; keep independent purge audit'},
    {'order': 100, 'artifact': 'sqlite_backups', 'retention': 'operator configured normally', 'purge': 'remove all pre-purge backups and create one post-purge sanitized backup'},
    {'order': 110, 'artifact': 'redacted_purge_audit', 'retention': f'{DERIVED_DATA_AUDIT_RETENTION_DAYS} days', 'purge': 'expiry rotation only; contains hashes and counts, never target identifiers'},
)


def derived_data_lifecycle() -> dict[str, Any]:
    return {
        'ok': True,
        'version': DERIVED_DATA_LIFECYCLE_VERSION,
        'artifacts': [dict(item) for item in DERIVED_DATA_LIFECYCLE],
        'provider_request_response_logging': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def ensure_safe_child(root: Path, target: Path) -> None:
    if not path_is_under(target, root):
        raise ValueError(f'Refusing to operate outside target Vault: {target}')


def _vault_relative_target(cfg: VaultConfig, relative: Path) -> Path:
    """Build a lexical Vault child without following pre-existing links."""

    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Vault artifact reference must be a safe relative path')
    root = cfg.root.resolve()
    target = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        try:
            if cursor.is_symlink():
                raise ValueError('Vault artifact parent must not be a symlink')
        except OSError as exc:
            raise ValueError('Vault artifact parent cannot be inspected safely') from exc
    if target == root or root not in target.parents:
        raise ValueError('Vault artifact reference escapes the target Vault')
    return target


def _remove_cache_ref(cfg: VaultConfig, ref: str) -> bool:
    candidate = Path(ref)
    if candidate.is_absolute() or '..' in candidate.parts:
        return False
    parts = candidate.parts
    if len(parts) < 2 or parts[0] != 'media' or parts[1] not in {'materialized', 'decoded', 'previews', 'tmp'}:
        return False
    target = _vault_relative_target(cfg, candidate)
    try:
        os.lstat(target)
    except FileNotFoundError:
        return False
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)
    return True


def _clear_runtime_cache_dir(cfg: VaultConfig, relative: Path) -> int:
    target = _vault_relative_target(cfg, relative)
    if target.is_symlink():
        target.unlink()
        return 1
    if not target.exists() or not target.is_dir():
        return 0
    removed = sum(1 for item in target.rglob('*') if item.is_file() or item.is_symlink())
    shutil.rmtree(target)
    return removed


def _remove_source_root(cfg: VaultConfig, root_ref: str) -> bool:
    candidate = Path(root_ref)
    if (
        candidate.is_absolute()
        or '..' in candidate.parts
        or len(candidate.parts) < 2
        or candidate.parts[0] not in {'sources', 'decrypted'}
    ):
        return False
    target = _vault_relative_target(cfg, candidate)
    if target.is_symlink():
        target.unlink()
        return True
    if not target.exists():
        return False
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)
    return True


@mutation_entrypoint('derived_data_purge')
def purge_derived_data(
    vault_root: Path,
    *,
    scope_type: str,
    scope_id: str,
    audit_retention_days: int = DERIVED_DATA_AUDIT_RETENTION_DAYS,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.approvals import ApprovalManager
    from trove_core.maintain import rotate_sqlite_backups
    from trove_core.store.sqlite_store import SQLiteStore

    cfg = VaultConfig.resolve(str(vault_root), env={})
    cfg.validate_runtime_path()
    scope_hash = hashlib.sha256(f'{scope_type}:{scope_id}'.encode('utf-8')).hexdigest()
    purge_id = 'purge-' + hashlib.sha256(
        f'{scope_hash}:{datetime.now(timezone.utc).isoformat()}'.encode('utf-8')
    ).hexdigest()[:20]
    with coordinated_vault_mutation(cfg, operation='derived_data_purge', write_session=write_session):
        store = SQLiteStore(cfg.paths.sqlite_path)
        try:
            internal = store.purge_derived_data(
                scope_type=scope_type,
                scope_id=scope_id,
                purge_id=purge_id,
                scope_hash=scope_hash,
                lifecycle_version=DERIVED_DATA_LIFECYCLE_VERSION,
                audit_retention_days=audit_retention_days,
            )
        finally:
            store.close()
        try:
            removed_cache_files = sum(_remove_cache_ref(cfg, ref) for ref in internal.pop('_file_refs', []))
            removed_preview_files = _clear_runtime_cache_dir(cfg, Path('media/previews'))
            removed_temp_files = _clear_runtime_cache_dir(cfg, Path('media/tmp'))
            removed_job_temp_files = _clear_runtime_cache_dir(cfg, Path('jobs/tmp'))
            removed_source_roots = 0
            if scope_type == 'source':
                removed_source_roots = sum(_remove_source_root(cfg, ref) for ref in internal.pop('_source_root_refs', []))
            else:
                internal.pop('_source_root_refs', None)
            approval_cleanup = ApprovalManager(cfg.root).purge_records(internal.pop('_approval_ids', []))
            prior_backups = list(cfg.paths.index_dir.glob(f'{cfg.paths.sqlite_path.name}.bak-*'))
            removed_backups = 0
            for backup in prior_backups:
                try:
                    backup.unlink()
                    removed_backups += 1
                except FileNotFoundError:
                    continue
            backup = rotate_sqlite_backups(cfg.paths.sqlite_path, retention=1, create=True)
        except BaseException:
            # The database cascade is durable before filesystem cleanup. Mark
            # a partial cleanup honestly so acceptance and later maintenance
            # cannot treat it as complete.
            failed_store = SQLiteStore(cfg.paths.sqlite_path)
            try:
                with failed_store.connect() as conn:
                    conn.execute(
                        "UPDATE derived_data_purge_audit SET status='failed' WHERE purge_id=?",
                        (purge_id,),
                    )
                    conn.commit()
            except Exception:
                pass
            finally:
                failed_store.close()
            raise
    return {
        **internal,
        'scope_hash': scope_hash[:20],
        'lifecycle_version': DERIVED_DATA_LIFECYCLE_VERSION,
        'filesystem': {
            'cache_files_removed': removed_cache_files,
            'preview_files_removed': removed_preview_files,
            'temporary_files_removed': removed_temp_files + removed_job_temp_files,
            'source_roots_removed': removed_source_roots,
        },
        'approvals': approval_cleanup,
        'backups': {
            'pre_purge_removed': removed_backups,
            'post_purge_retained': int(backup.get('retained_count') or 0),
            'policy': 'replace_all_pre_purge_backups_with_one_post_purge_backup',
        },
        'provider_request_response_logging': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


@mutation_entrypoint('reset_index_cache')
def reset_index_cache(
    vault_root: Path,
    *,
    write_session: VaultWriteSession | None = None,
) -> dict:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(
        cfg,
        operation='reset_index_cache',
        write_session=write_session,
    ):
        cfg.validate_runtime_path()
        root = cfg.root.resolve()
        removed: list[str] = []
        for target in [cfg.paths.sqlite_path, cfg.paths.index_dir / 'trove.sqlite-shm', cfg.paths.index_dir / 'trove.sqlite-wal']:
            target = target.resolve()
            ensure_safe_child(root, target)
            if target.exists() and target.is_file():
                target.unlink()
                removed.append(target.name)
        for directory in [cfg.paths.vector_dir, root / 'jobs' / 'tmp']:
            directory = directory.resolve()
            ensure_safe_child(root, directory)
            if directory.exists() and directory.is_dir():
                shutil.rmtree(directory)
                removed.append(directory.name + '/')
        cfg.ensure()
    return {'vault': str(cfg.root), 'removed': removed}


@mutation_entrypoint('scope_rebuild')
def rebuild_scope(
    vault_root: Path,
    *,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.store.sqlite_store import SQLiteStore

    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='scope_rebuild', write_session=write_session):
        return SQLiteStore(cfg.paths.sqlite_path).purge_excluded_scope()


@mutation_entrypoint('content_kind_backfill')
def backfill_content_kind(
    vault_root: Path,
    *,
    limit: int,
    backup_retention: int = 3,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.maintain import rotate_sqlite_backups
    from trove_core.store.sqlite_store import SQLiteStore

    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='content_kind_backfill', write_session=write_session):
        backup = rotate_sqlite_backups(cfg.paths.sqlite_path, retention=backup_retention, create=True)
        report = SQLiteStore(cfg.paths.sqlite_path).backfill_message_content_kind(limit=limit)
    return {'backup': backup, 'backfill': report}


@mutation_entrypoint('rebuild_chunks')
def rebuild_chunks(
    vault_root: Path,
    *,
    max_chars: int,
    overlap_chars: int,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.store.sqlite_store import SQLiteStore

    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='rebuild_chunks', write_session=write_session):
        return SQLiteStore(cfg.paths.sqlite_path).rebuild_evidence_chunks(
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )


@mutation_entrypoint('rebuild_fts')
def rebuild_fts(
    vault_root: Path,
    *,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.store.sqlite_store import SQLiteStore

    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='rebuild_fts', write_session=write_session):
        return SQLiteStore(cfg.paths.sqlite_path).rebuild_fts()


@mutation_entrypoint('initialize_index')
def initialize_index(
    vault_root: Path,
    *,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.store.sqlite_store import SQLiteStore

    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='initialize_index', write_session=write_session):
        cfg.ensure()
        SQLiteStore(cfg.paths.sqlite_path).initialize()
    return {'ok': True, 'indexed': 0, 'note': 'empty index initialized; fixture and real indexers are separate commands'}


def _coverage_payload(cfg: VaultConfig, job: dict[str, Any] | None = None) -> dict[str, Any]:
    from trove_core.store.sqlite_store import open_store

    if not cfg.paths.sqlite_path.is_file():
        indexed = {'contacts': 0, 'moments': 0, 'interactions': 0, 'favorites': 0}
    else:
        store = open_store(cfg.paths.sqlite_path, readonly=True)
        try:
            with store.connect() as conn:
                def count(table: str) -> int:
                    if not store._table_exists(conn, table):
                        return 0
                    return int(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
                indexed = {
                    'contacts': count('entities'),
                    'moments': count('moment_items'),
                    'interactions': count('moment_interactions'),
                    'favorites': count('favorites'),
                }
        finally:
            store.close()
    job = job or {}
    return {
        'contacts': {'source_rows': int(job.get('contacts_imported') or 0), 'indexed_rows': indexed['contacts']},
        'moments': {'source_rows': int(job.get('moments_imported') or 0), 'indexed_rows': indexed['moments']},
        'interactions': {'source_rows': int(((job.get('scope_counts') or {}).get('moment_interactions') or 0)), 'indexed_rows': indexed['interactions']},
        'favorites': {'source_rows': int(job.get('favorites_imported') or 0), 'indexed_rows': indexed['favorites']},
        'raw_content_included': False,
    }


def read_last_import_status(vault_root: Path) -> dict:
    from trove_core.vault.generation import vault_generation_read

    cfg = VaultConfig.resolve(str(vault_root), env={})
    cfg.validate_runtime_path()
    if not cfg.root.exists():
        return {'status': 'missing', 'job': None, 'coverage': _coverage_payload(cfg)}
    with vault_generation_read(cfg):
        path = cfg.paths.jobs_dir / 'last_import.json'
        if not path.exists():
            return {'status': 'missing', 'job': None, 'coverage': _coverage_payload(cfg)}
        import json
        job = json.loads(path.read_text(encoding='utf-8'))
        stale = bool(job.get('vault') and Path(str(job.get('vault'))).expanduser().resolve() != cfg.root.expanduser().resolve())
        return {'status': 'ok', 'job': job, 'coverage': _coverage_payload(cfg, job), 'stale_from_other_vault': stale}
