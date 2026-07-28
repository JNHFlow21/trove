from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping

from .models import (
    ContactIdentity,
    LiveMessage,
    WeChatLiveConfig,
    stable_ref,
)


SQLCIPHER_CANDIDATES = (
    Path('/opt/homebrew/bin/sqlcipher'),
    Path('/usr/local/bin/sqlcipher'),
    Path('/opt/homebrew/opt/sqlcipher/bin/sqlcipher'),
)
SPECIAL_TARGETS = frozenset({
    'filehelper', 'fmessage', 'medianote', 'newsapp', 'notifymessage',
    'notification_messages', 'weixin', 'weixinreminder',
})
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'
MAX_DECOMPRESSED_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class WorkAccount:
    account_id: str
    root: Path
    contact_db: Path
    session_db: Path
    message_dbs: tuple[Path, ...]


def _account_hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in (path, Path(f'{path}-wal'), Path(f'{path}-shm')):
        if not candidate.exists():
            continue
        metadata = candidate.stat()
        digest.update(candidate.name.encode('utf-8'))
        digest.update(str(metadata.st_size).encode('ascii'))
        digest.update(str(metadata.st_mtime_ns).encode('ascii'))
    return digest.hexdigest()


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    if re.fullmatch(r'[A-Za-z0-9_]+', value) is None:
        raise ValueError('unsafe_sql_identifier')
    return f'[{value}]'


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _resolve_sqlcipher() -> Path:
    for candidate in SQLCIPHER_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError('sqlcipher_binary_missing')


def _derived_key(path: Path, key_store: Mapping[str, Mapping[str, Any]]) -> str:
    with path.open('rb') as stream:
        salt = stream.read(16).hex().lower()
    record = key_store.get(salt) or {}
    key = str(record.get('dk') or '')
    if (
        len(key) not in {32, 64, 96, 128}
        or any(char not in '0123456789abcdefABCDEF' for char in key)
    ):
        raise RuntimeError('missing_or_invalid_key_for_source')
    return key


def _snapshot_family(source: Path, root: Path, *, retries: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    target_root = root / _source_fingerprint(source)[:16]
    target_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target_root, 0o700)
    target = target_root / source.name
    for _attempt in range(retries):
        before = _source_fingerprint(source)
        for suffix in ('', '-wal', '-shm'):
            original = Path(f'{source}{suffix}')
            copied = Path(f'{target}{suffix}')
            copied.unlink(missing_ok=True)
            if original.exists():
                shutil.copy2(original, copied)
                os.chmod(copied, 0o600)
        if before == _source_fingerprint(source):
            return target
        time.sleep(0.1)
    raise RuntimeError('source_changed_during_snapshot')


def _decode_content(raw: bytes) -> bytes:
    if not raw:
        return b''
    if raw.startswith(ZSTD_MAGIC):
        try:
            import zstandard
            return zstandard.ZstdDecompressor().decompress(
                raw, max_output_size=MAX_DECOMPRESSED_BYTES,
            )
        except Exception:
            return b''
    return raw


def _xml_field(text: str, tag: str) -> str:
    match = re.search(
        rf'<{tag}[^>]*>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE,
    )
    return html.unescape(match.group(1).strip()) if match else ''


def _decode_message_text(local_type: int, raw: bytes) -> str:
    decoded = _decode_content(raw)
    text = decoded.decode('utf-8', errors='replace') if decoded else ''
    if local_type == 1:
        return text.replace('\x00', '').strip()[:8_000]
    fixed = {
        3: '[图片]',
        34: '[语音]',
        43: '[视频]',
        47: '[表情]',
        48: '[位置]',
    }
    if local_type in fixed:
        return fixed[local_type]
    if local_type == 49 or local_type > 100:
        title = _xml_field(text, 'title')
        description = _xml_field(text, 'des') or _xml_field(text, 'desc')
        if title and description:
            return f'[分享] {title}: {description}'[:1_000]
        if title:
            return f'[分享] {title}'[:1_000]
        return '[分享]' if local_type == 49 else f'[消息类型 {local_type}]'
    return f'[消息类型 {local_type}]'


def _kind(local_type: int) -> str:
    return {
        1: 'text',
        3: 'image',
        34: 'voice',
        43: 'video',
        47: 'sticker',
        49: 'link',
    }.get(local_type, 'other')


class SQLCipher:
    def __init__(self, binary: str | Path | None = None) -> None:
        self.binary = Path(binary) if binary is not None else _resolve_sqlcipher()

    def query(
        self,
        database: Path,
        key_hex: str,
        sql: str,
        *,
        timeout: int = 20,
    ) -> list[dict[str, Any]]:
        if (
            len(key_hex) not in {32, 64, 96, 128}
            or any(char not in '0123456789abcdefABCDEF' for char in key_hex)
        ):
            raise ValueError('invalid_key_hex')
        marker = '__TROVE_WECHAT_LIVE_RESULT__'
        script = '\n'.join((
            f'PRAGMA key = "x\'{key_hex}\'";',
            'PRAGMA query_only = ON;',
            '.mode json',
            f'.print {marker}BEGIN',
            str(sql).rstrip().rstrip(';') + ';',
            f'.print {marker}END',
        )) + '\n'
        result = subprocess.run(
            [str(self.binary), str(database), '-batch'],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError('sqlcipher_query_failed')
        lines = result.stdout.splitlines()
        try:
            begin = lines.index(f'{marker}BEGIN') + 1
            end = lines.index(f'{marker}END', begin)
        except ValueError as exc:
            raise RuntimeError('sqlcipher_result_marker_missing') from exc
        raw = '\n'.join(
            line for line in lines[begin:end]
            if line.strip() and line.strip().lower() != 'ok'
        )
        if not raw:
            return []
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise RuntimeError('sqlcipher_json_not_list')
        return [item for item in payload if isinstance(item, dict)]


class WeChatLiveSource:
    """Read exact live deltas from the configured work account."""

    def __init__(
        self,
        config: WeChatLiveConfig,
        key_store: Mapping[str, Mapping[str, Any]],
        *,
        runtime_root: str | Path,
        sqlcipher: SQLCipher | None = None,
        account: WorkAccount | None = None,
        context_limit: int = 50,
    ) -> None:
        self.config = config
        self.key_store = key_store
        self.runtime_root = Path(runtime_root)
        self.snapshot_root = self.runtime_root / 'snapshots'
        self.sqlcipher = sqlcipher or SQLCipher()
        self.account = account or self._discover_account()
        self.context_limit = max(1, min(int(context_limit), 200))

    def _discover_account(self) -> WorkAccount:
        documents = (
            Path.home()
            / 'Library/Containers'
            / self.config.container_name
            / 'Data/Documents/xwechat_files'
        )
        if not documents.is_dir():
            raise FileNotFoundError('work_container_missing')
        matches: list[WorkAccount] = []
        for root in documents.iterdir():
            if (
                not root.is_dir()
                or root.is_symlink()
                or root.name != self.config.source_account
                or _account_hash(root.name) != self.config.account_id_sha256
            ):
                continue
            db_root = root / 'db_storage'
            contact = db_root / 'contact/contact.db'
            session = db_root / 'session/session.db'
            messages = tuple(
                path
                for path in sorted((db_root / 'message').glob('message_*.db'))
                if path.stem.removeprefix('message_').isdigit()
            )
            if contact.is_file() and session.is_file() and messages:
                matches.append(
                    WorkAccount(root.name, root, contact, session, messages)
                )
        if len(matches) != 1:
            raise RuntimeError(f'work_account_match_count:{len(matches)}')
        return matches[0]

    def _query(self, source: Path, sql: str) -> list[dict[str, Any]]:
        self.snapshot_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.snapshot_root, 0o700)
        temporary = Path(tempfile.mkdtemp(
            prefix='query-', dir=self.snapshot_root,
        ))
        try:
            snapshot = _snapshot_family(source, temporary)
            return self.sqlcipher.query(
                snapshot, _derived_key(source, self.key_store), sql,
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _eligible(self, target_id: str) -> bool:
        if not target_id or target_id in SPECIAL_TARGETS:
            return False
        if target_id.endswith('@chatroom'):
            return self.config.groups_enabled
        return self.config.private_chats_enabled and not target_id.startswith('gh_')

    def sessions(self) -> tuple[tuple[str, str, int], ...]:
        rows = self._query(
            self.account.session_db,
            """
            SELECT username,
                   COALESCE(last_msg_locald_id, 0) AS last_local_id,
                   COALESCE(sort_timestamp, 0) AS sort_timestamp
              FROM SessionTable
             WHERE COALESCE(last_msg_locald_id, 0) > 0
             ORDER BY sort_timestamp, username
            """,
        )
        return tuple(
            (target_id, stable_ref(target_id), _as_int(row.get('last_local_id')))
            for row in rows
            if (target_id := str(row.get('username') or '').strip())
            and self._eligible(target_id)
        )

    def resolve_identity(self, target_ref: str) -> ContactIdentity:
        matches = [
            target_id
            for target_id, current_ref, _position in self.sessions()
            if current_ref == target_ref
        ]
        if len(matches) != 1:
            raise RuntimeError(f'target_ref_match_count:{len(matches)}')
        target_id = matches[0]
        rows = self._query(
            self.account.contact_db,
            f"""
            SELECT COALESCE(alias, '') AS alias,
                   COALESCE(remark, '') AS remark,
                   COALESCE(nick_name, '') AS nick_name
              FROM contact
             WHERE username = {_sql_literal(target_id)}
             LIMIT 2
            """,
        )
        if len(rows) != 1:
            raise RuntimeError(f'contact_match_count:{len(rows)}')
        alias = str(rows[0].get('alias') or '').strip()
        remark = str(rows[0].get('remark') or '').strip()
        nickname = str(rows[0].get('nick_name') or '').strip()
        display = remark or nickname
        if not display:
            raise RuntimeError('contact_display_name_missing')
        search_query = alias or display
        if alias:
            count = self._query(
                self.account.contact_db,
                f'SELECT COUNT(*) AS n FROM contact '
                f'WHERE alias={_sql_literal(alias)}',
            )
        else:
            count = self._query(
                self.account.contact_db,
                """
                SELECT COUNT(*) AS n FROM contact
                 WHERE COALESCE(NULLIF(remark, ''), NULLIF(nick_name, ''))
                """
                + f'={_sql_literal(display)}',
            )
        unique = bool(count and _as_int(count[0].get('n')) == 1)
        if not unique:
            raise RuntimeError('contact_search_not_unique')
        return ContactIdentity(
            target_id,
            target_ref,
            search_query,
            tuple(dict.fromkeys(
                item for item in (remark, nickname, display) if item
            )),
            True,
        )

    def _account_rowid(self, source: Path) -> int:
        values = {self.account.account_id}
        match = re.fullmatch(r'(.+)_([0-9a-fA-F]{4})', self.account.account_id)
        if match:
            values.add(match.group(1))
        rows = self._query(
            source,
            'SELECT rowid FROM Name2Id WHERE user_name IN ('
            + ','.join(_sql_literal(value) for value in sorted(values))
            + ') ORDER BY rowid',
        )
        if len(rows) != 1:
            raise RuntimeError(f'account_rowid_match_count:{len(rows)}')
        return _as_int(rows[0].get('rowid'))

    def _message_rows(
        self,
        source: Path,
        target_id: str,
        *,
        after: int | None,
        through: int | None,
        limit: int | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        table = 'Msg_' + hashlib.md5(target_id.encode('utf-8')).hexdigest()
        table_identifier = _sql_identifier(table)
        exists = self._query(
            source,
            'SELECT COUNT(*) AS n FROM sqlite_master '
            f"WHERE type='table' AND name={_sql_literal(table)}",
        )
        if not exists or _as_int(exists[0].get('n')) != 1:
            return [], False
        columns = {
            str(row.get('name') or '')
            for row in self._query(
                source, f'PRAGMA table_info({_sql_literal(table)})',
            )
        }
        required = {
            'local_id', 'local_type', 'real_sender_id',
            'create_time', 'message_content',
        }
        if not required <= columns:
            raise RuntimeError('message_schema_mismatch')
        server = 'server_id' if 'server_id' in columns else "''"
        where = []
        if after is not None:
            where.append(f'local_id>{int(after)}')
        if through is not None:
            where.append(f'local_id<={int(through)}')
        where_sql = ' WHERE ' + ' AND '.join(where) if where else ''
        if limit is None:
            order = ' ORDER BY local_id ASC,rowid ASC'
            limit_sql = ''
        else:
            order = ' ORDER BY local_id DESC,rowid DESC'
            limit_sql = f' LIMIT {max(1, min(int(limit), 200))}'
        rows = self._query(
            source,
            f"""
            SELECT local_id,{server} AS server_id,local_type,real_sender_id,
                   create_time,hex(message_content) AS content_hex
              FROM {table_identifier}{where_sql}{order}{limit_sql}
            """,
        )
        if limit is not None:
            rows.reverse()
        return rows, True

    def messages(
        self,
        target_id: str,
        *,
        after_source_position: int | None = None,
        through_source_position: int | None = None,
        limit: int | None = None,
    ) -> tuple[LiveMessage, ...]:
        result: list[LiveMessage] = []
        found = False
        for source in self.account.message_dbs:
            rows, table_found = self._message_rows(
                source,
                target_id,
                after=after_source_position,
                through=through_source_position,
                limit=limit,
            )
            found = found or table_found
            if not table_found:
                continue
            account_rowid = self._account_rowid(source)
            for row in rows:
                content_hex = str(row.get('content_hex') or '')
                try:
                    raw = bytes.fromhex(content_hex) if content_hex else b''
                except ValueError:
                    raw = b''
                local_type = _as_int(row.get('local_type')) or 1
                result.append(LiveMessage(
                    target_id,
                    stable_ref(target_id),
                    source.name,
                    _as_int(row.get('local_id')),
                    str(row.get('server_id') or ''),
                    local_type,
                    _as_int(row.get('create_time')),
                    _as_int(row.get('real_sender_id')) == account_rowid,
                    _decode_message_text(local_type, raw),
                ))
        if not found:
            raise RuntimeError('conversation_table_missing')
        result.sort(
            key=lambda item: (
                item.source_position, item.create_time, item.source_name,
            )
        )
        return tuple(result[-limit:] if limit is not None else result)

    def current_position(self, target_id: str) -> int:
        rows = self.messages(target_id, limit=1)
        return rows[-1].source_position if rows else 0

    def wait_for_outgoing_echo(
        self,
        target_id: str,
        *,
        after_source_position: int,
        expected_text: str,
        timeout_seconds: float,
        poll_seconds: float = 0.5,
    ) -> LiveMessage | None:
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        expected = expected_text.replace('\r\n', '\n').replace('\r', '\n').strip()
        while time.monotonic() < deadline:
            rows = self.messages(
                target_id, after_source_position=after_source_position,
            )
            for row in rows:
                actual = row.text.replace('\r\n', '\n').replace('\r', '\n').strip()
                if (
                    row.is_outgoing
                    and row.server_acknowledged
                    and actual == expected
                ):
                    return row
            time.sleep(max(0.1, poll_seconds))
        return None

    def _display_name(self, target_id: str) -> str:
        rows = self._query(
            self.account.contact_db,
            """
            SELECT COALESCE(
                NULLIF(remark, ''),NULLIF(nick_name, ''),NULLIF(alias, ''),username
            ) AS display_name
              FROM contact
            """
            + f' WHERE username={_sql_literal(target_id)} LIMIT 1',
        )
        return str(rows[0].get('display_name') or '').strip() if rows else ''

    def events(
        self,
        cursors: Mapping[str, int],
        *,
        observed_at: float,
    ) -> Mapping[str, Any]:
        events: list[dict[str, Any]] = []
        acknowledgements: list[dict[str, Any]] = []
        for target_id, target_ref, position in self.sessions():
            cursor = cursors.get(target_ref)
            if cursor is None:
                try:
                    latest = self.messages(
                        target_id,
                        through_source_position=position,
                        limit=1,
                    )
                except RuntimeError as exc:
                    if str(exc) != 'conversation_table_missing':
                        raise
                    acknowledgements.append({
                        'target_ref': target_ref,
                        'source_position': position,
                        'reason': 'conversation_unavailable',
                    })
                    continue
                if not latest or latest[-1].create_time < int(observed_at) - 5:
                    acknowledgements.append({
                        'target_ref': target_ref,
                        'source_position': position,
                        'reason': 'bootstrap_seed',
                    })
                    continue
                cursor = max(0, latest[-1].source_position - 1)
            if position <= cursor:
                continue
            try:
                delta = self.messages(
                    target_id,
                    after_source_position=cursor,
                    through_source_position=position,
                )
            except RuntimeError as exc:
                if str(exc) != 'conversation_table_missing':
                    raise
                acknowledgements.append({
                    'target_ref': target_ref,
                    'source_position': position,
                    'reason': 'conversation_unavailable',
                })
                continue
            if not delta:
                continue
            latest = delta[-1]
            if latest.is_outgoing or not self._display_name(target_id):
                acknowledgements.append({
                    'target_ref': target_ref,
                    'source_position': position,
                    'reason': 'outgoing_or_unresolved',
                    'fingerprint': latest.fingerprint,
                })
                continue
            messages = [{
                'citation': (
                    f'provider://wechat/live/{target_ref}/{item.source_position}'
                ),
                'source_position': item.source_position,
                'observed_at': float(observed_at),
                'kind': _kind(item.local_type),
                **({'text': item.text} if item.text else {}),
                'trust': 'untrusted_evidence',
            } for item in delta if not item.is_outgoing]
            if not messages:
                acknowledgements.append({
                    'target_ref': target_ref,
                    'source_position': position,
                    'reason': 'no_inbound_delta',
                })
                continue
            events.append({
                'event_id': (
                    'wechat-live-'
                    + hashlib.sha256(
                        f'{self.config.account_id}\0{target_ref}\0{position}'
                        .encode('utf-8')
                    ).hexdigest()
                ),
                'account_id': self.config.account_id,
                'conversation_id': _stable_id(
                    'conv', f'{self.config.conversation_scope}:{target_id}',
                ),
                'target_ref': target_ref,
                'source_position': position,
                'latest_fingerprint': latest.fingerprint,
                'messages': messages,
                'observed_at': float(observed_at),
            })
        return {
            'events': events,
            'acknowledgements': acknowledgements,
        }


__all__ = ['SQLCipher', 'WeChatLiveSource', 'WorkAccount']
