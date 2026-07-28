from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable

from trove_core.wechat.parsers.packed_info import parse_packed_info_blob
from trove_core.wechat.scope import classify_wechat_identity

MEDIA_EXT_RE = re.compile(r'(?i)\.(jpg|jpeg|png|gif|webp|heic|dat|mp3|m4a|wav|amr|silk|mp4|mov)(\?|$)')
VOICE_TYPES = {'34', 'voice', 'audio', 'amr', 'silk', 'm4a', 'mp3', 'wav'}
IMAGE_TYPES = {'3', '47', 'image', 'img', 'pic', 'picture', 'jpeg', 'jpg', 'png', 'gif', 'webp', 'heic', 'dat'}
VIDEO_TYPES = {'43', '62', 'video', 'mp4', 'mov'}
PATH_HINT_WORDS = ('path', 'file', 'thumb', 'media', 'url', 'cdn', 'md5', 'sha', 'hash', 'aes')
ID_WORDS = ('local_id', 'message_id', 'msg_id', 'msgid', 'server_id', 'svrid', 'id')


@dataclass(frozen=True)
class MediaReference:
    asset_id: str
    account_id: str
    source_type: str
    source_id: str
    modality: str
    media_type: str
    citation: str
    local_type: str | None = None
    path_hint: str | None = None
    path_ref: str | None = None
    content_hash: str | None = None
    cache_state: str = 'metadata_only'
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaDiscoveryResult:
    references: tuple[MediaReference, ...]
    source_states: tuple[dict[str, Any], ...]
    deleted_asset_ids: tuple[str, ...]
    counters: dict[str, int]

    @property
    def refs(self) -> tuple[MediaReference, ...]:
        return self.references

    def to_dict(self) -> dict[str, Any]:
        return {
            'references': len(self.references),
            'source_states': len(self.source_states),
            'deleted_asset_ids': len(self.deleted_asset_ids),
            'counters': dict(self.counters),
            'raw_paths_included': False,
            'raw_content_included': False,
        }


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}'


def _stable12(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def message_media_asset_id(citation: str, modality: str, media_type: str) -> str:
    """Stable placeholder identity shared by message import and resource upgrades."""
    if modality == 'voice' and media_type == 'voice':
        payload = json.dumps(['voice-asset', citation], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return f'voice-asset-{hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]}'
    payload = json.dumps(
        ['message-media', citation, modality, media_type],
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return f'message-asset-{hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]}'


def _safe_text(value: Any, limit: int = 512) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        text = value[:limit].decode('utf-8', errors='ignore')
    else:
        text = str(value)
    return text.replace('\x00', '').strip()[:limit]


def _media_type_from(local_type: str | None, path_hint: str | None) -> tuple[str, str]:
    value = (local_type or '').lower()
    ext = ''
    if path_hint:
        match = MEDIA_EXT_RE.search(path_hint)
        if match:
            ext = match.group(1).lower()
    marker = value or ext
    if marker in VOICE_TYPES or ext in VOICE_TYPES:
        return 'voice', 'voice'
    if marker in VIDEO_TYPES or ext in VIDEO_TYPES:
        return 'video', 'video'
    if marker in IMAGE_TYPES or ext in IMAGE_TYPES:
        return 'image', 'image'
    if ext:
        return ('image', 'image') if ext in IMAGE_TYPES else ('voice', 'voice') if ext in VOICE_TYPES else ('video', 'video')
    return 'attachment', 'unknown'


def _resolve_path_hint(path_hint: str | None, roots: list[Path]) -> Path | None:
    if not path_hint:
        return None
    candidate = Path(path_hint).expanduser()
    if candidate.is_absolute() and candidate.exists():
        resolved = candidate.resolve()
        if any(_path_under(resolved, root) for root in roots):
            return resolved
        return None
    for root in roots:
        for maybe in (root / path_hint, root / Path(path_hint).name):
            if maybe.exists():
                return maybe.resolve()
    return None


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _path_exists(path_hint: str | None, roots: list[Path]) -> bool:
    return _resolve_path_hint(path_hint, roots) is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
    except sqlite3.DatabaseError:
        return []


def _iter_rows(conn: sqlite3.Connection, table: str, columns: list[str], limit: int | None = None):
    selected = ', '.join(f'"{c}"' for c in columns)
    query = f'SELECT rowid AS __rowid__, {selected} FROM "{table}"'
    if limit:
        query += f' LIMIT {int(limit)}'
    try:
        yield from conn.execute(query)
    except sqlite3.DatabaseError:
        return


def _iter_candidate_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    *,
    after_rowid: int | None = None,
    limit: int | None = None,
):
    selected = ', '.join(f'"{c}"' for c in columns)
    query = f'SELECT rowid AS __rowid__, {selected} FROM "{table}"'
    params: list[Any] = []
    if after_rowid is not None:
        query += ' WHERE rowid>?'
        params.append(int(after_rowid))
    query += ' ORDER BY rowid'
    if limit:
        query += ' LIMIT ?'
        params.append(int(limit))
    try:
        yield from conn.execute(query, params)
    except sqlite3.DatabaseError:
        return


def _candidate_columns(table: str, columns: list[str]) -> list[str]:
    """Return only columns that can affect media identity/classification."""
    selected: list[str] = []
    for column in columns:
        lower = column.lower()
        if (
            column in ID_WORDS
            or lower in ID_WORDS
            or lower in {'local_type', 'type', 'media_type', 'msg_type', 'message_type', 'sub_type'}
            or lower in {'packed_info', 'packed_info_data', 'pack_info_buf'}
            or any(word in lower for word in PATH_HINT_WORDS)
        ):
            selected.append(column)
    # Message citations require local_id even when no path-like column exists.
    if table.startswith('Msg_') and 'local_id' in columns and 'local_id' not in selected:
        selected.append('local_id')
    return list(dict.fromkeys(selected))


def _row_fingerprint(row: sqlite3.Row, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode('utf-8'))
        digest.update(b'\0')
        value = row[column]
        if isinstance(value, bytes):
            digest.update(hashlib.sha256(value).digest())
        else:
            digest.update(str(value if value is not None else '').encode('utf-8', errors='replace'))
        digest.update(b'\0')
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode('utf-8')).hexdigest()


def _table_fingerprint(table: str, columns: list[str]) -> str:
    return hashlib.sha256(json.dumps([table, columns], separators=(',', ':')).encode('utf-8')).hexdigest()


def _row_path_hint(row: sqlite3.Row, columns: list[str]) -> str | None:
    hints: list[str] = []
    for col in columns:
        if col.lower() in {'packed_info', 'packed_info_data', 'pack_info_buf'}:
            parsed = parse_packed_info_blob(row[col])
            hints.extend(parsed.path_hints)
        if any(word in col.lower() for word in PATH_HINT_WORDS):
            text = _safe_text(row[col])
            if text:
                hints.append(text)
    for col in columns:
        text = _safe_text(row[col])
        if MEDIA_EXT_RE.search(text):
            hints.append(text)
    return hints[0] if hints else None


def _row_local_type(row: sqlite3.Row, columns: list[str]) -> str | None:
    for name in ('local_type', 'type', 'media_type', 'msg_type', 'message_type', 'sub_type'):
        if name in columns:
            value = _safe_text(row[name], limit=64)
            if value:
                return value
    return None


def _row_source_id(db_path: Path, table: str, row: sqlite3.Row, columns: list[str]) -> str:
    for name in ID_WORDS:
        if name in columns:
            value = _safe_text(row[name], limit=128)
            if value:
                return f'{db_path.stem}:{table}:{value}'
    return f'{db_path.stem}:{table}:row-{row["__rowid__"]}'


def _source_type_from(db_path: Path, table: str) -> str:
    value = f'{db_path.stem}:{table}'.lower()
    if 'sns' in value or 'moment' in value:
        return 'moment'
    if 'fav' in value or 'favorite' in value:
        return 'favorite'
    if 'contact' in value:
        return 'contact'
    if 'message' in value or 'msg_' in value or 'resource' in value or 'voice' in value or 'img' in value:
        return 'message'
    return 'wechat_metadata'


def _load_name2id(conn: sqlite3.Connection) -> dict[int, str]:
    try:
        return {int(row['rowid']): str(row['user_name']) for row in conn.execute('SELECT rowid, user_name FROM Name2Id') if row['user_name']}
    except sqlite3.DatabaseError:
        return {}


def _message_citation_metadata(
    account_dir: Path,
    account_id: str,
    db_path: Path,
    table: str,
    row: sqlite3.Row,
    columns: list[str],
    table_by_username: dict[str, str],
) -> tuple[str | None, str | None, dict[str, Any]]:
    if not table.startswith('Msg_') or 'local_id' not in columns:
        return None, None, {}
    username = table_by_username.get(table)
    if not username:
        return None, None, {}
    decision = classify_wechat_identity(username, has_chat_history=True)
    if not decision.allowed or decision.scope_type not in {'private_chat', 'group_chat'}:
        return None, decision.scope_type, {'scope_type': decision.scope_type, 'scope_allowed': False}
    conv_id = _stable12('conv', f'{account_dir.name}:{username}')
    local_id = int(row['local_id'])
    citation = f'trove://wechat/{account_id}/{conv_id}/{db_path.stem}/{local_id}'
    conversation_type = 'group' if decision.scope_type == 'group_chat' else 'private'
    return citation, decision.scope_type, {
        'message_citation': citation,
        'conversation_id': conv_id,
        'conversation_type': conversation_type,
        'scope_type': decision.scope_type,
        'message_shard_id': db_path.stem,
        'message_local_id': local_id,
    }


def _reference_from_row(
    account_dir: Path,
    account_id: str,
    db_path: Path,
    table: str,
    row: sqlite3.Row,
    columns: list[str],
    table_by_username: dict[str, str],
) -> MediaReference | None:
    maybe_media_table = any(token in table.lower() for token in ('resource', 'media', 'img', 'voice', 'sns', 'favorite', 'contact'))
    path_hint = _row_path_hint(row, columns)
    local_type = _row_local_type(row, columns)
    modality, media_type = _media_type_from(local_type, path_hint)
    if media_type == 'unknown' and not maybe_media_table:
        return None
    if media_type == 'unknown' and not path_hint:
        return None
    source_id = _row_source_id(db_path, table, row, columns)
    message_citation, message_scope_type, message_metadata = _message_citation_metadata(
        account_dir, account_id, db_path, table, row, columns, table_by_username,
    )
    source_type = message_scope_type or _source_type_from(db_path, table)
    citation = message_citation or f'trove://wechat/{account_id}/media/{source_id}'
    dedupe_key = path_hint or local_type or source_id
    basis = f'{account_id}:{modality}:{media_type}:{dedupe_key}'
    asset_id = message_media_asset_id(message_citation, modality, media_type) if message_citation else _stable('asset', basis)
    resolved_path = _resolve_path_hint(path_hint, [account_dir])
    # Source paths stay in the immutable snapshot and are recovered lazily via
    # the source binding.  Never persist a raw decrypted-source path as a Vault
    # cache reference.
    cache_state = 'source_available' if resolved_path is not None else ('missing_local_cache' if path_hint else 'metadata_only')
    metadata = {
        'db': db_path.name,
        'table': table,
        'rowid': int(row['__rowid__']),
        'has_path_hint': bool(path_hint),
    } | message_metadata
    return MediaReference(
        asset_id=asset_id,
        account_id=account_id,
        source_type=source_type,
        source_id=message_citation or source_id,
        modality=modality,
        media_type=media_type,
        local_type=local_type,
        citation=citation,
        path_hint=path_hint,
        path_ref=None,
        content_hash=None,
        cache_state=cache_state,
        metadata=metadata,
    )


def discover_media_assets(account_dir: Path, *, account_id: str | None = None, limit_per_table: int | None = None) -> list[MediaReference]:
    """Discover media-like references from local WeChat SQLite metadata without reading media bytes."""
    account_dir = Path(account_dir)
    account_id = account_id or _stable('acct', account_dir.name)
    roots = [account_dir]
    refs: dict[str, MediaReference] = {}
    for db_path in sorted(account_dir.glob('*.db')):
        if not any(token in db_path.name.lower() for token in ('message', 'resource', 'contact', 'sns', 'favorite', 'fav', 'media', 'hardlink', 'head_image')):
            continue
        try:
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                name_by_id = _load_name2id(conn) if db_path.name.startswith('message_') else {}
                table_by_username = {'Msg_' + hashlib.md5(username.encode('utf-8')).hexdigest(): username for username in name_by_id.values()}
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                for table in tables:
                    columns = _table_columns(conn, table)
                    if not columns:
                        continue
                    for row in _iter_rows(conn, table, columns, limit=limit_per_table):
                        ref = _reference_from_row(account_dir, account_id, db_path, table, row, columns, table_by_username)
                        if ref is not None:
                            refs[ref.asset_id] = ref
        except sqlite3.DatabaseError:
            continue
    return sorted(refs.values(), key=lambda r: (r.account_id, r.source_id, r.asset_id))


def discover_media_assets_delta(
    account_dir: Path,
    *,
    store: Any,
    account_id: str | None = None,
    limit_per_table: int | None = None,
) -> MediaDiscoveryResult:
    """Discover only changed media rows using persisted file/table watermarks.

    Unchanged files are rejected before SQLite is opened.  Pure append growth
    selects ``rowid > watermark`` and only candidate columns.  Opaque edits or
    deletions fall back to a narrow reconciliation scan because an external
    SQLite snapshot has no change journal; even then only rows whose persisted
    fingerprint differs are emitted to the projection writer.
    """
    account_dir = Path(account_dir)
    account_id = account_id or _stable('acct', account_dir.name)
    # A sync scan receives an already-published read-only Vault.  Opening that
    # reader validates the schema without running writable initialization.
    if not getattr(store, 'readonly', False):
        store.initialize()
    counters = {
        'files_seen': 0,
        'files_skipped': 0,
        'tables_seen': 0,
        'tables_skipped': 0,
        'appended_tables': 0,
        'reconciled_tables': 0,
        'source_rows_scanned': 0,
        'candidate_rows': 0,
        'candidate_columns': 0,
    }
    refs: dict[str, MediaReference] = {}
    source_states: list[dict[str, Any]] = []
    deleted_asset_ids: set[str] = set()

    for db_path in sorted(account_dir.glob('*.db')):
        if not any(token in db_path.name.lower() for token in ('message', 'resource', 'contact', 'sns', 'favorite', 'fav', 'media', 'hardlink', 'head_image')):
            continue
        counters['files_seen'] += 1
        try:
            file_fingerprint = _file_fingerprint(db_path)
        except OSError:
            continue
        prefix = f'{account_id}:{db_path.name}:'
        with store.connect() as state_conn:
            previous_states = {
                str(row['source_key']): dict(row)
                for row in state_conn.execute(
                    'SELECT * FROM media_source_state WHERE source_key LIKE ?',
                    (prefix + '%',),
                )
            }
        if previous_states and all(str(row['file_fingerprint']) == file_fingerprint for row in previous_states.values()):
            counters['files_skipped'] += 1
            continue

        try:
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                name_by_id = _load_name2id(conn) if db_path.name.startswith('message_') else {}
                table_by_username = {'Msg_' + hashlib.md5(username.encode('utf-8')).hexdigest(): username for username in name_by_id.values()}
                tables = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                for table in tables:
                    all_columns = _table_columns(conn, table)
                    columns = _candidate_columns(table, all_columns)
                    if not columns:
                        counters['tables_skipped'] += 1
                        continue
                    counters['tables_seen'] += 1
                    counters['candidate_columns'] += len(columns)
                    source_key = prefix + table
                    table_fingerprint = _table_fingerprint(table, columns)
                    try:
                        row_count, max_rowid = conn.execute(
                            f'SELECT COUNT(*),COALESCE(MAX(rowid),0) FROM "{table}"'
                        ).fetchone()
                    except sqlite3.DatabaseError:
                        continue
                    row_count = int(row_count or 0)
                    max_rowid = int(max_rowid or 0)
                    previous = previous_states.get(source_key)
                    previous_count = int(previous['row_count']) if previous is not None else 0
                    previous_watermark = int(previous['row_watermark']) if previous is not None else 0
                    pure_append = bool(
                        previous is not None
                        and str(previous['table_fingerprint']) == table_fingerprint
                        and row_count >= previous_count
                        and max_rowid >= previous_watermark
                        and (row_count - previous_count) == (max_rowid - previous_watermark)
                        and max_rowid > previous_watermark
                    )
                    scan_after = previous_watermark if pure_append else None
                    if pure_append:
                        counters['appended_tables'] += 1
                        previous_rows: dict[int, dict[str, Any]] = {}
                    else:
                        counters['reconciled_tables'] += 1
                        with store.connect() as state_conn:
                            previous_rows = {
                                int(row['row_id']): dict(row)
                                for row in state_conn.execute(
                                    'SELECT row_id,row_fingerprint,asset_id,citation FROM media_source_rows WHERE source_key=?',
                                    (source_key,),
                                )
                            }

                    current_row_ids: set[int] = set()
                    row_updates: list[dict[str, Any]] = []
                    stale_asset_ids: set[str] = set()
                    scanned_max = previous_watermark if pure_append else 0
                    for row in _iter_candidate_rows(
                        conn,
                        table,
                        columns,
                        after_rowid=scan_after,
                        limit=limit_per_table,
                    ):
                        row_id = int(row['__rowid__'])
                        current_row_ids.add(row_id)
                        scanned_max = max(scanned_max, row_id)
                        counters['source_rows_scanned'] += 1
                        row_fingerprint = _row_fingerprint(row, columns)
                        old = previous_rows.get(row_id)
                        if old is not None and str(old['row_fingerprint']) == row_fingerprint:
                            continue
                        ref = _reference_from_row(
                            account_dir,
                            account_id,
                            db_path,
                            table,
                            row,
                            columns,
                            table_by_username,
                        )
                        asset_id = ref.asset_id if ref is not None else ''
                        citation = ref.citation if ref is not None else ''
                        if old is not None and old.get('asset_id') and str(old['asset_id']) != asset_id:
                            stale_asset_ids.add(str(old['asset_id']))
                        row_updates.append({
                            'row_id': row_id,
                            'row_fingerprint': row_fingerprint,
                            'asset_id': asset_id,
                            'citation': citation,
                        })
                        if ref is not None:
                            refs[ref.asset_id] = ref
                            counters['candidate_rows'] += 1

                    deleted_row_ids: list[int] = []
                    if not pure_append:
                        deleted_row_ids = sorted(set(previous_rows) - current_row_ids)
                        for row_id in deleted_row_ids:
                            old_asset_id = str(previous_rows[row_id].get('asset_id') or '')
                            if old_asset_id:
                                stale_asset_ids.add(old_asset_id)
                    deleted_asset_ids.update(stale_asset_ids)
                    source_states.append({
                        'source_key': source_key,
                        'file_fingerprint': file_fingerprint,
                        'table_fingerprint': table_fingerprint,
                        'row_watermark': scanned_max if limit_per_table else max_rowid,
                        'row_count': row_count,
                        'row_updates': row_updates,
                        'deleted_row_ids': deleted_row_ids,
                        'stale_asset_ids': sorted(stale_asset_ids),
                    })
        except sqlite3.DatabaseError:
            continue

    return MediaDiscoveryResult(
        references=tuple(sorted(refs.values(), key=lambda item: (item.account_id, item.source_id, item.asset_id))),
        source_states=tuple(source_states),
        deleted_asset_ids=tuple(sorted(deleted_asset_ids)),
        counters=counters,
    )


def summarize_media_references(refs: Iterable[MediaReference]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in refs:
        for key in (ref.modality, ref.cache_state):
            counts[key] = counts.get(key, 0) + 1
        if ref.media_type != ref.modality:
            counts[ref.media_type] = counts.get(ref.media_type, 0) + 1
    return counts
