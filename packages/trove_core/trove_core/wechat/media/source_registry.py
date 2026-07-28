from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.wechat.decrypt.manifest import load_snapshot_guard


_COORDINATE_KEYS = {
    'source_type', 'db', 'table', 'rowid', 'message_citation', 'conversation_id',
    'conversation_type', 'message_shard_id', 'message_local_id', 'content_kind',
    'moment_citation', 'media_idx', 'cache_key', 'cache_mapping_source',
    'url_hash', 'url_md5', 'thumb_hash', 'thumb_md5',
}
_SOURCE_FILE_SUFFIXES = {'.db', '.sqlite', '.sqlite3', '.jsonl', '.dat', '.amr', '.silk', '.wav', '.m4a', '.mp3', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.mp4', '.mov'}
_MAX_FINGERPRINT_FILES = 50_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def opaque_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode('utf-8', errors='replace')).hexdigest()


def account_dir_hash(path_or_name: str | Path) -> str:
    return opaque_hash(Path(path_or_name).name)


def _manifest_hash(root: Path) -> str:
    digest = hashlib.sha256()
    seen = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not (Path(directory) / name).is_symlink())
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_symlink() or (path.suffix.lower() not in _SOURCE_FILE_SUFFIXES and not name.startswith('.trove_')):
                continue
            try:
                stat = path.stat(follow_symlinks=False)
                relative_hash = opaque_hash(str(path.relative_to(root)))
            except (OSError, ValueError):
                continue
            digest.update(f'{relative_hash}:{stat.st_size}:{stat.st_mtime_ns}'.encode('ascii'))
            seen += 1
            if seen >= _MAX_FINGERPRINT_FILES:
                break
        if seen >= _MAX_FINGERPRINT_FILES:
            break
    digest.update(f'files:{seen}'.encode('ascii'))
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceSnapshot:
    snapshot_revision: str
    root_ref: str | None
    manifest_hash: str
    guard_run_id_hash: str | None
    state: str

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'snapshot_revision': self.snapshot_revision,
            'state': self.state,
            'root_bound': bool(self.root_ref),
            'raw_paths_included': False,
        }


def inspect_source_snapshot(cfg: VaultConfig, snapshot_root: str | Path) -> SourceSnapshot:
    """Fingerprint a source snapshot without touching the Vault database."""

    resolved = Path(snapshot_root).expanduser().resolve()
    manifest_hash = _manifest_hash(resolved) if resolved.is_dir() else opaque_hash('unavailable')
    guard = load_snapshot_guard(resolved) if resolved.is_dir() else None
    guard_hash = opaque_hash(guard.run_id) if guard is not None and guard.run_id else None
    if resolved.is_dir() and path_is_under(resolved, cfg.root):
        root_ref = str(resolved.relative_to(cfg.root.resolve()))
        state = 'available'
        root_identity = root_ref
    else:
        root_ref = None
        state = 'external_unbound' if resolved.is_dir() else 'unavailable'
        root_identity = 'unbound'
    revision = opaque_hash(f'{root_identity}:{manifest_hash}:{guard_hash or ""}')
    return SourceSnapshot(revision, root_ref, manifest_hash, guard_hash, state)


def persist_source_snapshot(store: SQLiteStore, snapshot: SourceSnapshot) -> SourceSnapshot:
    """Persist a pre-inspected source snapshot in one short database commit."""

    store.initialize()
    now = _now()
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO source_snapshots(
                   snapshot_revision,root_ref,manifest_hash,guard_run_id_hash,state,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(snapshot_revision) DO UPDATE SET
                   root_ref=excluded.root_ref,manifest_hash=excluded.manifest_hash,
                   guard_run_id_hash=excluded.guard_run_id_hash,state=excluded.state,
                   updated_at=excluded.updated_at""",
            (
                snapshot.snapshot_revision,
                snapshot.root_ref,
                snapshot.manifest_hash,
                snapshot.guard_run_id_hash,
                snapshot.state,
                now,
                now,
            ),
        )
        conn.commit()
    return snapshot


def register_source_snapshot(cfg: VaultConfig, store: SQLiteStore, snapshot_root: str | Path) -> SourceSnapshot:
    snapshot = inspect_source_snapshot(cfg, snapshot_root)
    return persist_source_snapshot(store, snapshot)


def source_coordinates(metadata: dict[str, Any] | None, *, source_type: str) -> dict[str, Any]:
    metadata = metadata or {}
    coordinates: dict[str, Any] = {'source_type': source_type}
    for key in _COORDINATE_KEYS - {'source_type'}:
        value = metadata.get(key)
        if value is None or value == '':
            continue
        if isinstance(value, bool):
            coordinates[key] = value
        elif isinstance(value, int):
            coordinates[key] = value
        else:
            text = str(value)
            if len(text) <= 512 and not Path(text).is_absolute() and '://' not in text:
                coordinates[key] = text
    return coordinates


def bind_account_assets(
    store: SQLiteStore,
    *,
    account_id: str,
    snapshot: SourceSnapshot,
    account_hash: str,
) -> dict[str, int]:
    if snapshot.state not in {'available', 'external_unbound'}:
        return {'seen': 0, 'bound': 0}
    store.initialize()
    now = _now()
    seen = 0
    bound = 0
    with store.connect() as conn:
        # Joining with ``ma.citation LIKE mi.citation || '#%'`` forces an
        # account-media x Moments scan on real Vaults. Load each indexed family
        # once and resolve the citation base in Python instead.
        moment_metadata_by_citation = {
            str(row['citation']): str(row['metadata_json'] or '{}')
            for row in conn.execute(
                'SELECT citation,metadata_json FROM moment_items WHERE account_id=?',
                (account_id,),
            )
        }
        rows = list(conn.execute(
            """SELECT asset_id,source_type,citation,metadata_json
                 FROM media_assets WHERE account_id=?""",
            (account_id,),
        ))
        for row in rows:
            seen += 1
            try:
                metadata = json.loads(row['metadata_json'] or '{}')
            except json.JSONDecodeError:
                metadata = {}
            moment_metadata_raw = None
            if str(row['source_type']) == 'moment':
                moment_metadata_raw = moment_metadata_by_citation.get(
                    str(row['citation']).split('#', 1)[0]
                )
            if moment_metadata_raw:
                try:
                    moment_metadata = json.loads(moment_metadata_raw)
                except json.JSONDecodeError:
                    moment_metadata = {}
                for key in ('table', 'rowid'):
                    if key in moment_metadata and key not in metadata:
                        metadata[key] = moment_metadata[key]
            coordinates = source_coordinates(metadata, source_type=str(row['source_type']))
            payload = json.dumps(coordinates, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            cursor = conn.execute(
                """INSERT INTO media_source_bindings(
                       asset_id,snapshot_revision,account_dir_hash,source_coordinates_json,
                       locator_state,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(asset_id) DO UPDATE SET
                       snapshot_revision=excluded.snapshot_revision,
                       account_dir_hash=excluded.account_dir_hash,
                       source_coordinates_json=excluded.source_coordinates_json,
                       locator_state=CASE
                           WHEN media_source_bindings.locator_state='materialized'
                           THEN 'materialized'
                           ELSE 'bound'
                       END,
                       updated_at=excluded.updated_at
                   WHERE media_source_bindings.snapshot_revision IS NOT excluded.snapshot_revision
                      OR media_source_bindings.account_dir_hash IS NOT excluded.account_dir_hash
                      OR media_source_bindings.source_coordinates_json IS NOT excluded.source_coordinates_json
                      OR media_source_bindings.locator_state NOT IN ('bound','materialized')""",
                (row['asset_id'], snapshot.snapshot_revision, account_hash, payload, 'bound', now, now),
            )
            bound += max(cursor.rowcount, 0)
        if bound:
            conn.commit()
        else:
            conn.rollback()
    return {'seen': seen, 'bound': bound}


def rebind_account_assets(
    store: SQLiteStore,
    *,
    account_id: str,
    snapshot: SourceSnapshot,
    account_hash: str,
) -> dict[str, int]:
    """Move an account's immutable media coordinates to a newer snapshot.

    Incremental message syncs normally add only a few media assets, but every
    decrypted generation is immutable and older generations are retained only
    briefly.  Recomputing coordinates for the whole account on every message
    delta is unnecessarily expensive; leaving existing bindings untouched is
    incorrect because retention will eventually remove their source snapshot.

    Existing coordinates are therefore moved in one set-based update.  Only
    assets that have never been bound require metadata parsing.
    """

    if snapshot.state not in {'available', 'external_unbound'}:
        return {'seen': 0, 'bound': 0}
    store.initialize()
    now = _now()
    seen = 0
    bound = 0
    with store.connect() as conn:
        seen = int(conn.execute(
            'SELECT COUNT(*) FROM media_assets WHERE account_id=?',
            (account_id,),
        ).fetchone()[0])
        cursor = conn.execute(
            """UPDATE media_source_bindings
                  SET snapshot_revision=?,account_dir_hash=?,
                      locator_state=CASE
                          WHEN locator_state='materialized' THEN 'materialized'
                          ELSE 'bound'
                      END,
                      updated_at=?
                WHERE asset_id IN (
                          SELECT asset_id FROM media_assets WHERE account_id=?
                      )
                  AND (
                          snapshot_revision IS NOT ?
                       OR account_dir_hash IS NOT ?
                       OR locator_state NOT IN ('bound','materialized')
                  )""",
            (
                snapshot.snapshot_revision,
                account_hash,
                now,
                account_id,
                snapshot.snapshot_revision,
                account_hash,
            ),
        )
        bound += max(cursor.rowcount, 0)

        missing = list(conn.execute(
            """SELECT ma.asset_id,ma.source_type,ma.citation,ma.metadata_json
                 FROM media_assets ma
                 LEFT JOIN media_source_bindings binding
                   ON binding.asset_id=ma.asset_id
                WHERE ma.account_id=? AND binding.asset_id IS NULL""",
            (account_id,),
        ))
        moment_metadata_by_citation: dict[str, str] = {}
        if any(str(row['source_type']) == 'moment' for row in missing):
            moment_metadata_by_citation = {
                str(row['citation']): str(row['metadata_json'] or '{}')
                for row in conn.execute(
                    'SELECT citation,metadata_json FROM moment_items WHERE account_id=?',
                    (account_id,),
                )
            }
        for row in missing:
            try:
                metadata = json.loads(row['metadata_json'] or '{}')
            except json.JSONDecodeError:
                metadata = {}
            moment_metadata_raw = None
            if str(row['source_type']) == 'moment':
                moment_metadata_raw = moment_metadata_by_citation.get(
                    str(row['citation']).split('#', 1)[0]
                )
            if moment_metadata_raw:
                try:
                    moment_metadata = json.loads(moment_metadata_raw)
                except json.JSONDecodeError:
                    moment_metadata = {}
                for key in ('table', 'rowid'):
                    if key in moment_metadata and key not in metadata:
                        metadata[key] = moment_metadata[key]
            coordinates = source_coordinates(
                metadata,
                source_type=str(row['source_type']),
            )
            payload = json.dumps(
                coordinates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            )
            cursor = conn.execute(
                """INSERT INTO media_source_bindings(
                       asset_id,snapshot_revision,account_dir_hash,
                       source_coordinates_json,locator_state,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    row['asset_id'],
                    snapshot.snapshot_revision,
                    account_hash,
                    payload,
                    'bound',
                    now,
                    now,
                ),
            )
            bound += max(cursor.rowcount, 0)
        if bound:
            conn.commit()
        else:
            conn.rollback()
    return {'seen': seen, 'bound': bound}


def resolve_snapshot_root(cfg: VaultConfig, store: SQLiteStore, snapshot_revision: str) -> tuple[Path | None, str | None]:
    with store.connect() as conn:
        row = conn.execute(
            'SELECT root_ref,manifest_hash,state FROM source_snapshots WHERE snapshot_revision=?',
            (snapshot_revision,),
        ).fetchone()
    if row is None:
        return None, 'source_snapshot_unavailable'
    if str(row['state']) != 'available' or not row['root_ref']:
        return None, 'source_snapshot_unavailable'
    root = (cfg.root / str(row['root_ref'])).resolve()
    if not path_is_under(root, cfg.root) or not root.is_dir():
        return None, 'source_snapshot_unavailable'
    if _manifest_hash(root) != str(row['manifest_hash']):
        return None, 'source_snapshot_changed'
    return root, None


def resolve_account_dir(snapshot_root: Path, expected_hash: str) -> Path | None:
    candidates = [snapshot_root]
    try:
        candidates.extend(path for path in snapshot_root.iterdir() if path.is_dir() and not path.is_symlink())
    except OSError:
        return None
    for candidate in candidates:
        if account_dir_hash(candidate) == expected_hash:
            return candidate
    return None
