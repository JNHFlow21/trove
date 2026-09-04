from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import sqlite3

from trove_core.domain.messages import Account, Conversation, Message
from .jsonl_export import parse_ts

CONTENT_COLUMNS = ('content', 'msg', 'message', 'text', 'StrContent')
TIME_COLUMNS = ('timestamp', 'create_time', 'CreateTime', 'time', 'msg_time')
SENDER_COLUMNS = ('sender_name', 'sender', 'from_user', 'talker', 'Sender', 'FromUserName')
CONV_COLUMNS = ('conversation_id', 'room_id', 'talker', 'Talker', 'strTalker')
ID_COLUMNS = ('local_id', 'msg_id', 'id', 'MsgSvrID', 'CreateTime')


def choose(columns: list[str], names: tuple[str, ...]) -> str | None:
    by_lower = {c.lower(): c for c in columns}
    for name in names:
        if name in columns:
            return name
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SQLiteArchiveImporter:
    def __init__(self, path: Path, *, account_id: str | None = None, account_label: str | None = None):
        self.path = Path(path)
        digest = hashlib.sha256(str(self.path.resolve()).encode('utf-8')).hexdigest()[:8]
        self.account_id = account_id or f'acct-{digest}'
        self.account_label = account_label or self.account_id

    def candidate_tables(self, conn: sqlite3.Connection) -> list[str]:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        out = []
        for table in tables:
            lower = table.lower()
            if 'message' in lower or lower in {'msg', 'messages'}:
                out.append(table)
        return out

    def load(self, limit: int | None = None) -> tuple[list[Account], list[Conversation], list[Message]]:
        uri = f'file:{self.path}?mode=ro'
        messages: list[Message] = []
        conversations: dict[str, Conversation] = {}
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            for table in self.candidate_tables(conn):
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info({quote(table)})')]
                content_col = choose(cols, CONTENT_COLUMNS)
                if not content_col:
                    continue
                time_col = choose(cols, TIME_COLUMNS)
                sender_col = choose(cols, SENDER_COLUMNS)
                conv_col = choose(cols, CONV_COLUMNS)
                id_col = choose(cols, ID_COLUMNS)
                query = f'SELECT * FROM {quote(table)}'
                if limit:
                    query += f' LIMIT {int(limit)}'
                for idx, row in enumerate(conn.execute(query), start=1):
                    content = row[content_col]
                    if content is None or not str(content).strip():
                        continue
                    conv_id = str(row[conv_col]) if conv_col and row[conv_col] is not None else table
                    conv_title = conv_id
                    ctype = 'group' if '@chatroom' in conv_id else 'private'
                    sender_id = str(row[sender_col]) if sender_col and row[sender_col] is not None else 'unknown'
                    sender_name = sender_id
                    ts_value = row[time_col] if time_col else None
                    local_id_raw = row[id_col] if id_col and row[id_col] is not None else idx
                    try:
                        local_id = int(local_id_raw)
                    except Exception:
                        local_id = idx
                    msg = Message(
                        account_id=self.account_id,
                        account_label=self.account_label,
                        conversation_id=conv_id,
                        conversation_title=conv_title,
                        conversation_type=ctype,
                        sender_id=sender_id,
                        sender_name=sender_name,
                        timestamp=parse_ts(ts_value),
                        content=str(content),
                        shard_id=table,
                        local_id=local_id,
                        sent_by_me=False,
                        source_type='message',
                    )
                    conversations.setdefault(conv_id, Conversation(conv_id, self.account_id, conv_title, ctype, 1))
                    messages.append(msg)
        return [Account(self.account_id, self.account_label, self.account_label)], list(conversations.values()), messages
