from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
import sqlite3
from typing import Any

try:
    import zstandard
except ImportError:  # pragma: no cover - runtime doctor covers the required dependency.
    zstandard = None

from trove_core.domain.messages import Account, Conversation, Message
from trove_core.domain.content import classify_content_kind, display_content_for_kind
from trove_core.wechat.parsers.appmsg import parse_appmsg
from trove_core.wechat.scope import ScopeDecision, classify_wechat_identity
from trove_core.wechat.decrypt.manifest import load_account_identity


ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'
MAX_DECOMPRESSED_MESSAGE_BYTES = 16 * 1024 * 1024


def stable_id(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def msg_table_for(username: str) -> str:
    return 'Msg_' + hashlib.md5(username.encode('utf-8')).hexdigest()


def parse_wechat_ts(value: Any) -> datetime:
    try:
        num = int(value)
    except Exception:
        num = 0
    if num > 10_000_000_000:
        num = num // 1000
    if num <= 0:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromtimestamp(num, tz=timezone.utc)


def decode_content(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        if value.startswith(ZSTD_MAGIC) and zstandard is not None:
            try:
                value = zstandard.ZstdDecompressor().decompress(
                    value, max_output_size=MAX_DECOMPRESSED_MESSAGE_BYTES,
                )
            except zstandard.ZstdError:
                pass
        for enc in ('utf-8', 'utf-16le', 'gb18030'):
            try:
                return value.decode(enc).strip('\x00')
            except UnicodeDecodeError:
                continue
        return value.decode('utf-8', errors='ignore').strip('\x00')
    return str(value).strip('\x00')


def _extract_wxid(value: str) -> str:
    match = re.search(r'wxid_[A-Za-z0-9]+', str(value or ''))
    return match.group(0) if match else ''


def _looks_like_wxid(value: str) -> bool:
    return bool(re.fullmatch(r'wxid_[A-Za-z0-9]+', str(value or '')))


def _record_waterline(
    updates: dict[tuple[str, str, str], dict[str, Any]],
    key: tuple[str, str, str],
    *,
    local_id: int,
    raw_create_time: int,
    timestamp: datetime,
) -> None:
    prev = updates.get(key, {'max_local_id': -1, 'max_create_time': -1, 'max_timestamp': ''})
    if local_id > int(prev.get('max_local_id') or -1):
        prev['max_local_id'] = local_id
    if raw_create_time > int(prev.get('max_create_time') or -1):
        prev['max_create_time'] = raw_create_time
    iso = timestamp.isoformat().replace('+00:00', 'Z')
    if iso > str(prev.get('max_timestamp') or ''):
        prev['max_timestamp'] = iso
    updates[key] = prev


class WeChatDecryptedAccountImporter:
    """Importer for decrypted WeChat KOS account directories.

    Expected account dir shape:
      account_dir/contact.db
      account_dir/session.db
      account_dir/message_*.db with Name2Id and Msg_<md5> tables
    """
    def __init__(self, account_dir: Path):
        self.account_dir = Path(account_dir)
        self.account_id = stable_id('acct', self.account_dir.name)
        self.account_label = self.account_dir.name.split('__')[0] + '-' + self.account_id.split('-')[1][:6]
        self.last_scope_counts: dict[str, int] = {}
        self.last_excluded_counts: dict[str, int] = {}
        self.last_scope_decisions: dict[str, ScopeDecision] = {}
        self.last_waterline_updates: dict[tuple[str, str, str], dict[str, Any]] = {}

    def load(
        self,
        limit_per_shard: int | None = None,
        *,
        waterlines: dict[tuple[str, str, str], dict[str, Any]] | None = None,
        since: datetime | None = None,
        content_kinds: set[str] | None = None,
    ) -> tuple[list[Account], list[Conversation], list[Message]]:
        requested_kinds = set(content_kinds or ())
        titles = self._load_contact_titles()
        member_counts = self._load_member_counts()
        own_wxid = self._load_own_wxid()
        conversations: dict[str, Conversation] = {}
        messages: list[Message] = []
        scope_counts: dict[str, int] = {}
        excluded_counts: dict[str, int] = {}
        scope_decisions: dict[str, ScopeDecision] = {}
        waterline_updates: dict[tuple[str, str, str], dict[str, Any]] = {}
        waterlines = waterlines or {}
        for db_path in sorted(self.account_dir.glob('message_*.db')):
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                name_by_id = self._load_name2id(conn)
                table_by_username = {msg_table_for(username): username for username in name_by_id.values()}
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Msg_%'")]
                for table in tables:
                    username = table_by_username.get(table)
                    if not username:
                        continue
                    decision = classify_wechat_identity(username, has_chat_history=True)
                    scope_decisions[username] = decision
                    scope_counts[decision.scope_type] = scope_counts.get(decision.scope_type, 0) + 1
                    if not decision.allowed or decision.scope_type not in {'private_chat', 'group_chat'}:
                        excluded_counts[decision.scope_type] = excluded_counts.get(decision.scope_type, 0) + 1
                        continue
                    conv_id = stable_id('conv', f'{self.account_dir.name}:{username}')
                    title = titles.get(username) or username
                    ctype = 'group' if decision.scope_type == 'group_chat' else 'private'
                    conversations.setdefault(conv_id, Conversation(conv_id, self.account_id, title, ctype, member_counts.get(username, 1)))
                    key = (self.account_id, conv_id, db_path.stem)
                    cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
                    content_cols = [
                        c for c in ('message_content', 'compress_content', 'WCDB_CT_message_content')
                        if c in cols
                    ]
                    if not {'local_id', 'real_sender_id', 'create_time'}.issubset(cols) or not content_cols:
                        continue
                    select_cols = ['local_id', 'real_sender_id', 'create_time', *content_cols]
                    if 'local_type' in cols:
                        select_cols.append('local_type')
                    query = 'SELECT ' + ', '.join(f'"{c}"' for c in select_cols) + f' FROM "{table}"'
                    params: list[Any] = []
                    conditions: list[str] = []
                    waterline = waterlines.get(key) or {}
                    if waterline:
                        conditions.append('(local_id > ? OR create_time > ?)')
                        params.extend([int(waterline.get('max_local_id') or -1), int(waterline.get('max_create_time') or -1)])
                    if requested_kinds == {'appmsg'}:
                        appmsg_conditions: list[str] = []
                        if 'local_type' in cols:
                            appmsg_conditions.append(
                                'CAST("local_type" AS INTEGER) IN (42,48,49,50,66,67)'
                            )
                        for content_col in content_cols:
                            quoted = f'CAST("{content_col}" AS TEXT)'
                            appmsg_conditions.extend((
                                f'instr(lower({quoted}),\'<appmsg\')>0',
                                f'ltrim(lower({quoted})) LIKE \'<msg>%\'',
                                f'ltrim(lower({quoted})) LIKE \'<?xml%\'',
                                f'substr(CAST("{content_col}" AS BLOB),1,4)=X\'28B52FFD\'',
                            ))
                        conditions.append('(' + ' OR '.join(appmsg_conditions) + ')')
                    if conditions:
                        query += ' WHERE ' + ' AND '.join(conditions)
                    query += ' ORDER BY create_time, local_id'
                    if limit_per_shard:
                        query += f' LIMIT {int(limit_per_shard)}'
                    for row in conn.execute(query, params):
                        timestamp = parse_wechat_ts(row['create_time'])
                        local_id = int(row['local_id'])
                        raw_create_time = int(row['create_time'] or 0)
                        _record_waterline(waterline_updates, key, local_id=local_id, raw_create_time=raw_create_time, timestamp=timestamp)
                        raw_content = ''
                        for content_col in content_cols:
                            raw_content = raw_content or decode_content(row[content_col])
                        local_type = row['local_type'] if 'local_type' in row.keys() else None
                        content_kind = classify_content_kind(raw_content, local_type=local_type)
                        if requested_kinds and content_kind not in requested_kinds:
                            continue
                        if not raw_content and content_kind == 'text':
                            continue
                        normalized_payload = None
                        if content_kind == 'appmsg':
                            parsed_payload = parse_appmsg(raw_content)
                            normalized_payload = parsed_payload.to_dict()
                            content = parsed_payload.display_text
                        else:
                            content = display_content_for_kind(raw_content, content_kind)
                        if since is not None and timestamp < since:
                            continue
                        real_sender_raw = name_by_id.get(row['real_sender_id'], '')
                        sender_raw = real_sender_raw or username
                        sender_is_known = bool(real_sender_raw)
                        sent_by_me = bool(own_wxid and sender_is_known and real_sender_raw == own_wxid)
                        direction_hint = None if own_wxid and sender_is_known else 'unknown'
                        sender_id = stable_id('sender', f'{self.account_dir.name}:{sender_raw}')
                        sender_name = titles.get(sender_raw) or sender_raw
                        messages.append(Message(
                            account_id=self.account_id,
                            account_label=self.account_label,
                            conversation_id=conv_id,
                            conversation_title=title,
                            conversation_type=ctype,
                            sender_id=sender_id,
                            sender_name=sender_name,
                            timestamp=timestamp,
                            content=content,
                            shard_id=db_path.stem,
                            local_id=local_id,
                            sent_by_me=sent_by_me,
                            source_type='message',
                            content_kind=content_kind,
                            direction_hint=direction_hint,
                            normalized_payload=normalized_payload,
                        ))
        self.last_scope_counts = scope_counts
        self.last_excluded_counts = excluded_counts
        self.last_scope_decisions = scope_decisions
        self.last_waterline_updates = waterline_updates
        return [Account(self.account_id, self.account_label, self.account_label)], list(conversations.values()), messages

    def waterline_snapshot(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        """Read exact shard maxima without decoding historical message bodies.

        This seeds incremental sync only after a separate full import has
        completed. It does not claim that source content itself was imported.
        """

        updates: dict[tuple[str, str, str], dict[str, Any]] = {}
        for db_path in sorted(self.account_dir.glob('message_*.db')):
            try:
                with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
                    conn.row_factory = sqlite3.Row
                    name_by_id = self._load_name2id(conn)
                    table_by_username = {msg_table_for(username): username for username in name_by_id.values()}
                    tables = [
                        str(row[0])
                        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Msg_%'")
                    ]
                    for table in tables:
                        username = table_by_username.get(table)
                        if not username:
                            continue
                        decision = classify_wechat_identity(username, has_chat_history=True)
                        if not decision.allowed or decision.scope_type not in {'private_chat', 'group_chat'}:
                            continue
                        columns = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
                        if not {'local_id', 'create_time'}.issubset(columns):
                            continue
                        maximum = conn.execute(
                            f'SELECT COALESCE(MAX(CAST(local_id AS INTEGER)),-1) AS max_local_id,'
                            f' COALESCE(MAX(CAST(create_time AS INTEGER)),-1) AS max_create_time FROM "{table}"'
                        ).fetchone()
                        max_local_id = int(maximum['max_local_id'] or -1)
                        max_create_time = int(maximum['max_create_time'] or -1)
                        if max_local_id < 0 and max_create_time < 0:
                            continue
                        conversation_id = stable_id('conv', f'{self.account_dir.name}:{username}')
                        _record_waterline(
                            updates,
                            (self.account_id, conversation_id, db_path.stem),
                            local_id=max_local_id,
                            raw_create_time=max_create_time,
                            timestamp=parse_wechat_ts(max_create_time),
                        )
            except (sqlite3.DatabaseError, OSError):
                continue
        return updates

    def _load_name2id(self, conn: sqlite3.Connection) -> dict[int, str]:
        try:
            return {int(row['rowid']): row['user_name'] for row in conn.execute('SELECT rowid, user_name FROM Name2Id')}
        except sqlite3.DatabaseError:
            return {}

    def _load_own_wxid(self) -> str:
        private_identity = load_account_identity(self.account_dir)
        own = str(private_identity.get('own_wxid') or '')
        if own:
            return own
        own = _extract_wxid(self.account_dir.name)
        if own:
            return own
        for db_name in ('account.db', 'userinfo.db', 'user_info.db', 'contact.db'):
            path = self.account_dir / db_name
            if not path.exists():
                continue
            try:
                with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
                    conn.row_factory = sqlite3.Row
                    for table_row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                        table = str(table_row['name'] or '')
                        if not any(token in table.lower() for token in ('self', 'user', 'account', 'profile')):
                            continue
                        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                        wanted = [c for c in cols if c.lower() in {'wxid', 'user_name', 'username', 'account', 'account_name', 'self_wxid'}]
                        for col in wanted:
                            try:
                                rows = conn.execute(f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT 20')
                            except sqlite3.DatabaseError:
                                continue
                            for row in rows:
                                candidate = str(row[col] or '')
                                if _looks_like_wxid(candidate):
                                    return candidate
            except sqlite3.DatabaseError:
                continue
        return ''

    def _load_contact_titles(self) -> dict[str, str]:
        path = self.account_dir / 'contact.db'
        if not path.exists():
            return {}
        titles: dict[str, str] = {}
        try:
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                for row in conn.execute('SELECT username, remark, nick_name, alias FROM contact'):
                    username = row['username']
                    title = row['remark'] or row['nick_name'] or row['alias'] or username
                    if username:
                        titles[username] = title
        except sqlite3.DatabaseError:
            pass
        return titles

    def _load_member_counts(self) -> dict[str, int]:
        path = self.account_dir / 'contact.db'
        if not path.exists():
            return {}
        counts: dict[str, int] = {}
        try:
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                cols = [r[1] for r in conn.execute('PRAGMA table_info(chatroom_member)')]
                if 'chatroom' in cols:
                    for row in conn.execute('SELECT chatroom, COUNT(*) AS n FROM chatroom_member GROUP BY chatroom'):
                        counts[row['chatroom']] = int(row['n'])
                elif 'username' in cols:
                    for row in conn.execute('SELECT username, COUNT(*) AS n FROM chatroom_member GROUP BY username'):
                        counts[row['username']] = int(row['n'])
        except sqlite3.DatabaseError:
            pass
        return counts
