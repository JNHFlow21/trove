from __future__ import annotations

from contextlib import closing

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.wechat.decrypt.manifest import load_account_identity
from trove_core.wechat.importers.wechat_decrypted import decode_content, msg_table_for, stable_id
from trove_core.wechat.media.source_registry import resolve_account_dir, resolve_snapshot_root
from trove_core.wechat.parsers.packed_info import parse_packed_info_blob


_PATH_WORDS = ('path', 'file', 'thumb', 'media', 'cache')
_MEDIA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.dat', '.amr', '.silk', '.m4a', '.mp3', '.wav', '.mp4', '.mov', '.pdf'}
_MAX_TABLES = 300
_MAX_CACHE_FILES = 50_000


@dataclass(frozen=True)
class MediaLocatorResult:
    status: str
    route: str | None = None
    path: Path | None = None
    source_root: Path | None = None
    embedded_bytes: bytes | None = None
    remote_url: str | None = None
    locator_hash: str | None = None
    snapshot_revision: str | None = None
    reason: str | None = None

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'route': self.route,
            'locator_hash': self.locator_hash,
            'snapshot_revision': self.snapshot_revision,
            'reason': self.reason,
            'raw_paths_included': False,
            'remote_url_included': False,
        }


def _json_obj(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or '{}')
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resolve_path_hint(account_dir: Path, hint: str) -> Path | None:
    hint = str(hint or '').strip()
    if not hint or '://' in hint:
        return None
    candidate = Path(hint).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [account_dir / candidate, account_dir / candidate.name]
    for path in candidates:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if path_is_under(resolved, account_dir) and resolved.is_file():
            return resolved
    return None


def _row_path_hints(row: sqlite3.Row, columns: list[str]) -> list[str]:
    hints: list[str] = []
    for column in columns:
        lower = column.lower()
        value = row[column]
        if lower in {'packed_info', 'packed_info_data', 'pack_info_buf'}:
            hints.extend(parse_packed_info_blob(value).path_hints)
        if any(word in lower for word in _PATH_WORDS):
            if isinstance(value, bytes):
                text = value[:4096].decode('utf-8', errors='ignore').strip('\x00')
            else:
                text = str(value or '').strip('\x00')
            if text and len(text) <= 4096:
                hints.append(text)
    return list(dict.fromkeys(hints))[:20]


def _account_usernames(account_dir: Path, preferred_db: Path) -> list[str]:
    paths = [preferred_db, *(path for path in sorted(account_dir.glob('message_*.db')) if path != preferred_db)]
    names: list[str] = []
    for path in paths:
        try:
            with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                names.extend(str(row['user_name']) for row in conn.execute('SELECT user_name FROM Name2Id') if row['user_name'])
        except sqlite3.DatabaseError:
            continue
        if names:
            break
    return list(dict.fromkeys(names))


def _path_from_exact_row(account_dir: Path, coordinates: dict[str, Any]) -> Path | None:
    db_name = str(coordinates.get('db') or '')
    table = str(coordinates.get('table') or '')
    rowid = coordinates.get('rowid')
    if not db_name or not table or type(rowid) is not int:
        return None
    db_path = account_dir / Path(db_name).name
    if not db_path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
            if not columns:
                return None
            row = conn.execute(f'SELECT * FROM "{table}" WHERE rowid=?', (rowid,)).fetchone()
            if row is None:
                return None
            for hint in _row_path_hints(row, columns):
                found = _resolve_path_hint(account_dir, hint)
                if found is not None:
                    return found
    except sqlite3.DatabaseError:
        return None
    return None


def _path_from_message_row(account_dir: Path, coordinates: dict[str, Any]) -> Path | None:
    shard = str(coordinates.get('message_shard_id') or '')
    conversation_id = str(coordinates.get('conversation_id') or '')
    local_id = coordinates.get('message_local_id')
    if not shard or not conversation_id or type(local_id) is not int:
        return None
    db_path = account_dir / f'{Path(shard).name}.db'
    if not db_path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            names = _account_usernames(account_dir, db_path)
            username = next(
                (name for name in names if stable_id('conv', f'{account_dir.name}:{name}') == conversation_id),
                None,
            )
            if username is None:
                return None
            table = msg_table_for(username)
            columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
            row = conn.execute(f'SELECT * FROM "{table}" WHERE local_id=? LIMIT 1', (local_id,)).fetchone()
            if row is not None:
                for hint in _row_path_hints(row, columns):
                    found = _resolve_path_hint(account_dir, hint)
                    if found is not None:
                        return found
    except sqlite3.DatabaseError:
        return None
    return None


def _message_row_identity(
    account_dir: Path,
    coordinates: dict[str, Any],
) -> tuple[str, int | None, int | None, int | None] | None:
    shard = str(coordinates.get('message_shard_id') or '')
    conversation_id = str(coordinates.get('conversation_id') or '')
    local_id = coordinates.get('message_local_id')
    if not shard or not conversation_id or type(local_id) is not int:
        return None
    message_db = account_dir / f'{Path(shard).name}.db'
    if not message_db.is_file():
        return None
    try:
        with closing(sqlite3.connect(f'file:{message_db}?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            names = _account_usernames(account_dir, message_db)
            username = next(
                (name for name in names if stable_id('conv', f'{account_dir.name}:{name}') == conversation_id),
                None,
            )
            if username is None:
                return None
            table = msg_table_for(username)
            columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
            server_column = next(
                (name for name in ('server_id', 'svr_id', 'message_svr_id', 'msg_svr_id') if name in columns),
                None,
            )
            create_time_column = next(
                (name for name in ('create_time', 'timestamp', 'time') if name in columns),
                None,
            )
            selected = ['local_id']
            if server_column is not None:
                selected.append(f'"{server_column}" AS server_id')
            if create_time_column is not None:
                selected.append(f'"{create_time_column}" AS create_time')
            row = conn.execute(
                f'SELECT {",".join(selected)} FROM "{table}" WHERE local_id=? LIMIT 1',
                (local_id,),
            ).fetchone()
            if row is None:
                return None
            chat_rows = list(conn.execute(
                'SELECT rowid FROM Name2Id WHERE user_name=? LIMIT 2',
                (username,),
            ))
            chat_name_id = int(chat_rows[0][0]) if len(chat_rows) == 1 else None
            server_id = (
                int(row['server_id'])
                if server_column is not None and row['server_id'] is not None
                else None
            )
            create_time = (
                int(row['create_time'])
                if create_time_column is not None and row['create_time'] is not None
                else None
            )
            return username, chat_name_id, server_id, create_time
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


def _message_server_id(account_dir: Path, coordinates: dict[str, Any]) -> int | None:
    identity = _message_row_identity(account_dir, coordinates)
    return identity[2] if identity is not None else None


def _voice_blob_from_media_db(account_dir: Path, coordinates: dict[str, Any]) -> bytes | None:
    """Resolve the exact WeChat VoiceInfo row paired with a message citation."""
    shard = str(coordinates.get('message_shard_id') or '')
    local_id = coordinates.get('message_local_id')
    if not shard or type(local_id) is not int:
        return None
    message_identity = _message_row_identity(account_dir, coordinates)
    if message_identity is None:
        return None
    username, message_chat_name_id, server_id, create_time = message_identity

    shard_suffix = Path(shard).name.removeprefix('message_')
    media_db = account_dir / f'media_{shard_suffix}.db'
    if not media_db.is_file():
        return None
    try:
        with closing(sqlite3.connect(f'file:{media_db}?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            columns = [str(row[1]) for row in conn.execute('PRAGMA table_info("VoiceInfo")')]
            required = {'local_id', 'voice_data'}
            if not required.issubset(columns):
                return None
            if server_id is not None and 'svr_id' in columns:
                rows = list(conn.execute(
                    'SELECT voice_data FROM VoiceInfo WHERE local_id=? AND svr_id=? LIMIT 2',
                    (local_id, server_id),
                ))
                if len(rows) == 1:
                    value = rows[0]['voice_data']
                    return (
                        bytes(value)
                        if isinstance(value, (bytes, bytearray, memoryview)) and value
                        else None
                    )
                if len(rows) > 1:
                    return None

            # Some WeChat generations update the message row's server id
            # without rewriting VoiceInfo.  The media database's conversation
            # id plus local id and create time remain the exact immutable
            # identity for the voice payload.
            media_chat_rows: list[sqlite3.Row] = []
            try:
                media_chat_rows = list(conn.execute(
                    'SELECT rowid FROM Name2Id WHERE user_name=? LIMIT 2',
                    (username,),
                ))
            except sqlite3.DatabaseError:
                pass
            chat_name_id = (
                int(media_chat_rows[0][0])
                if len(media_chat_rows) == 1
                else message_chat_name_id
            )
            if chat_name_id is None or 'chat_name_id' not in columns:
                return None
            predicate = 'chat_name_id=? AND local_id=?'
            params: list[Any] = [chat_name_id, local_id]
            if create_time is not None and 'create_time' in columns:
                predicate += ' AND create_time=?'
                params.append(create_time)
            rows = list(conn.execute(
                f'SELECT voice_data FROM VoiceInfo WHERE {predicate} LIMIT 2',
                params,
            ))
            if len(rows) != 1:
                return None
            value = rows[0]['voice_data']
            return bytes(value) if isinstance(value, (bytes, bytearray, memoryview)) and value else None
    except (sqlite3.DatabaseError, TypeError, ValueError):
        return None


def _wechat_documents_root() -> Path:
    return Path.home() / 'Library' / 'Containers' / 'com.tencent.xinWeChat' / 'Data' / 'Documents'


def _safe_component(value: Any) -> str | None:
    text = str(value or '')
    if not text or '\x00' in text or text in {'.', '..'} or Path(text).name != text or '/' in text or '\\' in text:
        return None
    return text


def _live_account_name(account_dir: Path) -> str | None:
    """Resolve the live ``xwechat_files`` account without exposing it in SQLite.

    Integrated decrypted snapshots intentionally use opaque directory names.
    Their private, permission-checked identity sidecar is therefore the
    authority for the live client account; legacy snapshots may still encode
    the wxid in the directory basename.
    """

    identity_name = _safe_component(load_account_identity(account_dir).get('own_wxid'))
    if identity_name and identity_name.startswith('wxid_'):
        return identity_name
    legacy_name = _safe_component(account_dir.name.rsplit('__', 1)[-1])
    if legacy_name and legacy_name.startswith('wxid_'):
        return legacy_name
    return None


def _message_resource_keys(account_dir: Path, coordinates: dict[str, Any]) -> list[str]:
    server_id = _message_server_id(account_dir, coordinates)
    resource_db = account_dir / 'message_resource.db'
    if server_id is None or not resource_db.is_file():
        return []
    try:
        with closing(sqlite3.connect(f'file:{resource_db}?mode=ro', uri=True)) as conn:
            rows = list(conn.execute(
                'SELECT packed_info FROM MessageResourceInfo WHERE message_svr_id=? LIMIT 3',
                (server_id,),
            ))
    except sqlite3.DatabaseError:
        return []
    keys: list[str] = []
    for row in rows:
        packed = row[0]
        raw = bytes(packed) if isinstance(packed, (bytes, bytearray, memoryview)) else str(packed or '').encode()
        keys.extend(match.decode().lower() for match in re.findall(rb'(?i)[0-9a-f]{32}', raw))
    return list(dict.fromkeys(keys))


def _path_from_live_hardlink_cache(
    account_dir: Path,
    coordinates: dict[str, Any],
    modality: str,
) -> tuple[Path, Path] | None:
    """Resolve one exact visual cache file using immutable message and hardlink metadata."""
    if modality not in {'image', 'video'}:
        return None
    hardlink_db = account_dir / 'hardlink.db'
    resource_keys = _message_resource_keys(account_dir, coordinates)
    if not resource_keys or not hardlink_db.is_file():
        return None
    try:
        with closing(sqlite3.connect(f'file:{hardlink_db}?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            for key in resource_keys:
                if modality == 'image':
                    matches = list(conn.execute(
                        """SELECT file_name,file_size,dir1,dir2
                             FROM image_hardlink_info_v4
                            WHERE file_name=? OR file_name=?
                            LIMIT 3""",
                        (f'{key}.dat', key),
                    ))
                else:
                    matches = [row for row in conn.execute(
                        """SELECT file_name,file_size,dir1,dir2
                             FROM video_hardlink_info_v4
                            WHERE file_name LIKE ?
                            LIMIT 6""",
                        (f'{key}.%',),
                    ) if Path(str(row['file_name'] or '')).suffix.lower() in {'.mp4', '.mov', '.m4v'}]
                if len(matches) != 1:
                    continue
                row = matches[0]
                dir1_row = conn.execute('SELECT username FROM dir2id WHERE rowid=?', (row['dir1'],)).fetchone()
                dir2_row = conn.execute('SELECT username FROM dir2id WHERE rowid=?', (row['dir2'],)).fetchone() if modality == 'image' else None
                if dir1_row is None or (modality == 'image' and dir2_row is None):
                    continue
                dir1 = _safe_component(dir1_row['username'])
                dir2 = _safe_component(dir2_row['username']) if dir2_row is not None else None
                filename = _safe_component(row['file_name'])
                if not dir1 or (modality == 'image' and not dir2) or not filename:
                    continue
                account_name = _live_account_name(account_dir)
                if not account_name:
                    continue
                live_root = _wechat_documents_root() / 'xwechat_files' / account_name
                candidate = (
                    live_root / 'msg' / 'attach' / dir1 / str(dir2) / 'Img' / filename
                    if modality == 'image'
                    else live_root / 'msg' / 'video' / dir1 / filename
                )
                try:
                    resolved_root = live_root.resolve(strict=True)
                    resolved = candidate.resolve(strict=True)
                except OSError:
                    continue
                if not path_is_under(resolved, resolved_root) or not resolved.is_file():
                    continue
                expected_size = row['file_size']
                if type(expected_size) is int and expected_size > 0:
                    actual_size = resolved.stat().st_size
                    exact_or_v2_wrapper = actual_size == expected_size or (
                        filename.lower().endswith('.dat') and expected_size < actual_size <= expected_size + 31
                    )
                    if not exact_or_v2_wrapper:
                        continue
                return resolved, resolved_root
    except sqlite3.DatabaseError:
        return None
    return None


def _path_from_live_file_cache(
    account_dir: Path,
    metadata: dict[str, Any],
) -> tuple[tuple[Path, Path] | None, bool]:
    """Resolve one exact document from WeChat's live ``msg/file`` cache.

    AppMsg file rows carry the original name and byte count but deliberately
    do not persist a raw local path.  The hardlink index binds those immutable
    message facts to the month directory used by the live desktop client.  A
    missing file therefore remains a typed lazy-materialization gap instead of
    being confused with an unparsed message.
    """

    file_name = _safe_component(metadata.get('file_name'))
    if not file_name:
        return None, False
    try:
        expected_size = int(metadata.get('file_size') or metadata.get('size_bytes') or 0)
    except (TypeError, ValueError):
        expected_size = 0
    account_name = _live_account_name(account_dir)
    if not account_name:
        return None, False
    # A safe original filename plus a bound WeChat account is sufficient to
    # identify this as a client-cache-backed attachment.  The hardlink table
    # narrows the month when present, but an older snapshot may legitimately
    # predate the client's eventual download.
    cache_identity_known = True
    live_root = _wechat_documents_root() / 'xwechat_files' / account_name
    hardlink_db = account_dir / 'hardlink.db'
    month_names: list[str] = []
    if hardlink_db.is_file():
        try:
            with closing(sqlite3.connect(f'file:{hardlink_db}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_hardlink_info_v4'",
                ).fetchone()
                if table is not None:
                    params: list[Any] = [file_name]
                    size_clause = ''
                    if expected_size > 0:
                        size_clause = ' AND file_size=?'
                        params.append(expected_size)
                    rows = list(conn.execute(
                        f"""SELECT file_name,file_size,dir1
                               FROM file_hardlink_info_v4
                              WHERE file_name=?{size_clause}
                              LIMIT 4""",
                        params,
                    ))
                    for row in rows:
                        mapped = conn.execute(
                            'SELECT username FROM dir2id WHERE rowid=?', (row['dir1'],),
                        ).fetchone()
                        month = _safe_component(mapped['username']) if mapped is not None else None
                        if month and month not in month_names:
                            month_names.append(month)
        except sqlite3.DatabaseError:
            pass

    candidates: list[Path] = []
    for month in month_names:
        candidates.append(live_root / 'msg' / 'file' / month / file_name)
    # A live file may arrive before the next decrypted hardlink snapshot.  The
    # fallback remains bounded to direct month children and still requires an
    # exact safe filename and byte count.
    file_root = live_root / 'msg' / 'file'
    try:
        for month_dir in list(file_root.iterdir())[:120] if file_root.is_dir() else []:
            month = _safe_component(month_dir.name)
            if month:
                candidate = month_dir / file_name
                if candidate not in candidates:
                    candidates.append(candidate)
    except OSError:
        pass

    resolved_matches: list[Path] = []
    try:
        resolved_root = live_root.resolve(strict=True)
    except OSError:
        return None, cache_identity_known
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            stat_size = resolved.stat().st_size
        except OSError:
            continue
        if not path_is_under(resolved, resolved_root) or not resolved.is_file():
            continue
        if expected_size > 0 and stat_size != expected_size:
            continue
        if resolved not in resolved_matches:
            resolved_matches.append(resolved)
    if len(resolved_matches) != 1:
        return None, cache_identity_known
    return (resolved_matches[0], resolved_root), cache_identity_known


def _path_from_resource_tables(account_dir: Path, local_id: int | None) -> Path | None:
    if type(local_id) is not int:
        return None
    tables_seen = 0
    for db_path in sorted(account_dir.glob('*.db')):
        if not any(token in db_path.name.lower() for token in ('resource', 'media', 'message', 'hardlink', 'voice', 'image', 'img')):
            continue
        try:
            with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                for table_row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                    table = str(table_row['name'])
                    tables_seen += 1
                    if tables_seen > _MAX_TABLES:
                        return None
                    columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
                    id_column = next((name for name in ('local_id', 'message_id', 'msg_id', 'msgid') if name in columns), None)
                    if id_column is None or not any(any(word in col.lower() for word in _PATH_WORDS) for col in columns):
                        continue
                    rows = conn.execute(f'SELECT * FROM "{table}" WHERE "{id_column}"=? LIMIT 5', (local_id,))
                    for row in rows:
                        for hint in _row_path_hints(row, columns):
                            found = _resolve_path_hint(account_dir, hint)
                            if found is not None:
                                return found
        except sqlite3.DatabaseError:
            continue
    return None


def _path_from_cache_key(account_dir: Path, cache_key: str | None) -> Path | None:
    key = str(cache_key or '').lower().strip()
    if not key or len(key) < 8:
        return None
    seen = 0
    for path in account_dir.rglob('*'):
        seen += 1
        if seen > _MAX_CACHE_FILES:
            break
        if path.is_symlink() or not path.is_file():
            continue
        compact = ''.join(path.parts[-3:]).lower()
        exact_key_match = key in path.name.lower() or key in compact
        if not exact_key_match:
            continue
        if path.suffix and path.suffix.lower() not in _MEDIA_SUFFIXES:
            continue
        if exact_key_match:
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                continue
            if path_is_under(resolved, account_dir):
                return resolved
    return None


def _local_name(tag: str) -> str:
    return str(tag).rsplit('}', 1)[-1]


def _remote_moment_url(account_dir: Path, coordinates: dict[str, Any]) -> str | None:
    table = str(coordinates.get('table') or 'SnsTimeLine')
    rowid = coordinates.get('rowid')
    media_idx = coordinates.get('media_idx')
    if type(rowid) is not int or type(media_idx) is not int:
        return None
    for db_name in ('sns.db', 'moment.db', 'moments.db'):
        db_path = account_dir / db_name
        if not db_path.is_file():
            continue
        try:
            with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
                content_column = next((name for name in ('content', 'text', 'xml') if name in columns), None)
                if content_column is None:
                    continue
                row = conn.execute(f'SELECT "{content_column}" AS raw FROM "{table}" WHERE rowid=?', (rowid,)).fetchone()
                if row is None:
                    continue
                raw = row['raw']
                text = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else str(raw or '')
                start = text.find('<')
                if start < 0:
                    continue
                root = ET.fromstring(text[start:])
                media_nodes = [node for node in root.iter() if _local_name(node.tag) == 'media']
                if not 0 <= media_idx < len(media_nodes):
                    continue
                node = media_nodes[media_idx]
                values: dict[str, str] = {}
                for child in node.iter():
                    name = _local_name(child.tag)
                    if name in {'url', 'thumb'} and child.text:
                        values[name] = child.text.strip()
                for key in ('url', 'thumb'):
                    value = values.get(key)
                    if not value:
                        continue
                    parsed = urlsplit(value)
                    expected_short = str(coordinates.get(f'{key}_hash') or '')
                    expected_md5 = str(coordinates.get(f'{key}_md5') or '')
                    short_ok = not expected_short or hashlib.sha256(value.encode()).hexdigest().startswith(expected_short)
                    md5_ok = not expected_md5 or hashlib.md5(value.encode()).hexdigest() == expected_md5
                    if parsed.scheme == 'https' and parsed.hostname and short_ok and md5_ok:
                        return value
        except (sqlite3.DatabaseError, ET.ParseError, ValueError):
            continue
    return None


_MESSAGE_STICKER_CDN_HOSTS = {'vweixinf.tc.qq.com', 'wxapp.tc.qq.com'}


def _remote_message_sticker_url(account_dir: Path, coordinates: dict[str, Any]) -> str | None:
    """Recover a direct sticker CDN URL from one exact bound message row.

    The URL is never persisted.  It is reconstructed from the immutable source
    snapshot only long enough to build the existing hashed approval payload.
    Encrypted-only URLs are deliberately ignored because the materializer does
    not yet own the corresponding AES transform.
    """

    shard = str(coordinates.get('message_shard_id') or '')
    conversation_id = str(coordinates.get('conversation_id') or '')
    local_id = coordinates.get('message_local_id')
    if not shard or not conversation_id or type(local_id) is not int:
        return None
    db_path = account_dir / f'{Path(shard).name}.db'
    if not db_path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            username = next(
                (
                    name for name in _account_usernames(account_dir, db_path)
                    if stable_id('conv', f'{account_dir.name}:{name}') == conversation_id
                ),
                None,
            )
            if username is None:
                return None
            table = msg_table_for(username)
            columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
            content_columns = [
                name for name in ('message_content', 'compress_content', 'WCDB_CT_message_content')
                if name in columns
            ]
            if not content_columns:
                return None
            selected = ','.join(f'"{name}"' for name in content_columns)
            row = conn.execute(
                f'SELECT {selected} FROM "{table}" WHERE local_id=? LIMIT 1',
                (local_id,),
            ).fetchone()
            if row is None:
                return None
            text = ''
            for name in content_columns:
                text = text or decode_content(row[name])
            start = text.find('<')
            if start < 0:
                return None
            root = ET.fromstring(text[start:])
            values: dict[str, str] = {}
            for node in root.iter():
                for key in ('cdnurl', 'externurl', 'tpurl'):
                    if node.attrib.get(key):
                        values.setdefault(key, str(node.attrib[key]).strip())
            for key in ('externurl', 'cdnurl', 'tpurl'):
                value = values.get(key)
                if not value:
                    continue
                parsed = urlsplit(value)
                host = (parsed.hostname or '').lower()
                if parsed.scheme not in {'http', 'https'} or host not in _MESSAGE_STICKER_CDN_HOSTS:
                    continue
                return parsed._replace(scheme='https').geturl()
    except (sqlite3.DatabaseError, ET.ParseError, ValueError):
        return None
    return None


def locate_media_asset(cfg: VaultConfig, store: SQLiteStore, asset: sqlite3.Row | dict[str, Any]) -> MediaLocatorResult:
    asset_id = str(asset['asset_id'])
    path_ref = str(asset['path_ref'] or '')
    if path_ref:
        path = Path(path_ref).expanduser()
        candidate = path if path.is_absolute() else cfg.root / path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            resolved = None
        if resolved is not None and path_is_under(resolved, cfg.root) and resolved.is_file():
            return MediaLocatorResult('found', route='vault_cache', path=resolved, locator_hash=hashlib.sha256(path_ref.encode()).hexdigest())

    with store.connect() as conn:
        binding = conn.execute(
            'SELECT * FROM media_source_bindings WHERE asset_id=?', (asset_id,),
        ).fetchone()
    if binding is None:
        return MediaLocatorResult('unavailable', reason='source_binding_missing')
    revision = str(binding['snapshot_revision'])
    snapshot_root, error = resolve_snapshot_root(cfg, store, revision)
    if snapshot_root is None:
        return MediaLocatorResult('unavailable', snapshot_revision=revision, reason=error or 'source_snapshot_unavailable')
    account_dir = resolve_account_dir(snapshot_root, str(binding['account_dir_hash']))
    if account_dir is None:
        return MediaLocatorResult('unavailable', snapshot_revision=revision, reason='source_account_unavailable')
    coordinates = _json_obj(str(binding['source_coordinates_json'] or '{}'))
    metadata = _json_obj(str(asset['metadata_json'] or '{}'))
    file_cache_identified = False

    if str(asset['modality'] or '') == 'voice':
        voice_blob = _voice_blob_from_media_db(account_dir, coordinates)
        if voice_blob is not None:
            return MediaLocatorResult(
                'found', route='source_voice_info_blob', embedded_bytes=voice_blob,
                locator_hash=hashlib.sha256(f'{revision}:source_voice_info_blob:{asset_id}'.encode()).hexdigest(),
                snapshot_revision=revision,
            )

    if str(asset['modality'] or '') in {'image', 'video'}:
        live_cache = _path_from_live_hardlink_cache(account_dir, coordinates, str(asset['modality']))
        if live_cache is not None:
            path, source_root = live_cache
            return MediaLocatorResult(
                'found', route='live_wechat_hardlink_cache', path=path, source_root=source_root,
                locator_hash=hashlib.sha256(f'{revision}:live_wechat_hardlink_cache:{asset_id}'.encode()).hexdigest(),
                snapshot_revision=revision,
            )

    if str(asset['modality'] or '') in {'file', 'attachment', 'document'}:
        live_file_cache, file_cache_identified = _path_from_live_file_cache(account_dir, metadata)
        if live_file_cache is not None:
            path, source_root = live_file_cache
            return MediaLocatorResult(
                'found', route='live_wechat_file_cache', path=path, source_root=source_root,
                locator_hash=hashlib.sha256(f'{revision}:live_wechat_file_cache:{asset_id}'.encode()).hexdigest(),
                snapshot_revision=revision,
            )

    routes = (
        ('source_exact_row', _path_from_exact_row(account_dir, coordinates)),
        ('source_message_row', _path_from_message_row(account_dir, coordinates)),
        ('source_resource_table', _path_from_resource_tables(account_dir, coordinates.get('message_local_id'))),
        ('source_account_cache', _path_from_cache_key(account_dir, coordinates.get('cache_key') or metadata.get('cache_key'))),
    )
    for route, path in routes:
        if path is not None:
            return MediaLocatorResult(
                'found', route=route, path=path,
                locator_hash=hashlib.sha256(f'{revision}:{route}:{asset_id}'.encode()).hexdigest(),
                snapshot_revision=revision,
            )

    if str(asset['source_type']) == 'moment':
        remote = _remote_moment_url(account_dir, coordinates)
        if remote:
            return MediaLocatorResult(
                'remote', route='wechat_cdn', remote_url=remote,
                locator_hash=hashlib.sha256(remote.encode()).hexdigest(),
                snapshot_revision=revision,
            )
    if str(asset['modality'] or '') == 'image' and str(asset['local_type'] or '').lower() in {'47', 'sticker', 'emoji'}:
        remote = _remote_message_sticker_url(account_dir, coordinates)
        if remote:
            return MediaLocatorResult(
                'remote', route='wechat_cdn', remote_url=remote,
                locator_hash=hashlib.sha256(remote.encode()).hexdigest(),
                snapshot_revision=revision,
            )
    if str(asset['modality'] or '') == 'video' and _message_resource_keys(account_dir, coordinates):
        return MediaLocatorResult(
            'unavailable', route='live_wechat_hardlink_cache', snapshot_revision=revision,
            reason='local_video_cache_missing',
        )
    if str(asset['modality'] or '') in {'file', 'attachment', 'document'} and file_cache_identified:
        return MediaLocatorResult(
            'unavailable', route='live_wechat_file_cache', snapshot_revision=revision,
            reason='local_file_cache_missing',
        )
    return MediaLocatorResult('unavailable', snapshot_revision=revision, reason='locator_routes_exhausted')
