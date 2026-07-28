from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

from trove_core.domain.messages import Account, Conversation, Message


def parse_ts(value) -> datetime:
    if isinstance(value, (int, float)):
        # WeChat-ish timestamps may be seconds or milliseconds.
        if value > 10_000_000_000:
            value = value / 1000
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value or '')
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class JsonlExportImporter:
    def __init__(self, path: Path, *, account_id: str | None = None, account_label: str | None = None):
        self.path = Path(path)
        self.account_id = account_id
        self.account_label = account_label

    def load(self) -> tuple[list[Account], list[Conversation], list[Message]]:
        accounts: dict[str, Account] = {}
        conversations: dict[tuple[str, str], Conversation] = {}
        messages: list[Message] = []
        with self.path.open('r', encoding='utf-8') as fh:
            for idx, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                account_id = str(obj.get('account_id') or self.account_id or 'acct-import')
                label = str(obj.get('account_label') or self.account_label or account_id)
                conv_id = str(obj.get('conversation_id') or obj.get('room_id') or obj.get('talker') or 'conv-import')
                conv_title = str(obj.get('conversation_title') or obj.get('title') or conv_id)
                conv_type = obj.get('conversation_type') or ('group' if '@chatroom' in conv_id or obj.get('is_group') else 'private')
                sender_id = str(obj.get('sender_id') or obj.get('sender') or obj.get('from_user') or 'unknown')
                sender_name = str(obj.get('sender_name') or obj.get('sender') or sender_id)
                content = str(obj.get('content') or obj.get('text') or obj.get('message') or '')
                if not content:
                    continue
                shard = str(obj.get('shard_id') or self.path.stem)
                local_id = int(obj.get('local_id') or obj.get('msg_id') or obj.get('id') or idx)
                msg = Message(
                    account_id=account_id,
                    account_label=label,
                    conversation_id=conv_id,
                    conversation_title=conv_title,
                    conversation_type='group' if conv_type == 'group' else 'private',
                    sender_id=sender_id,
                    sender_name=sender_name,
                    timestamp=parse_ts(obj.get('timestamp') or obj.get('create_time') or obj.get('time')),
                    content=content,
                    shard_id=shard,
                    local_id=local_id,
                    sent_by_me=bool(obj.get('sent_by_me') or obj.get('is_self') or obj.get('is_sender')),
                    source_type=obj.get('source_type') or 'message',
                )
                accounts.setdefault(account_id, Account(account_id, label, label))
                conversations.setdefault((account_id, conv_id), Conversation(conv_id, account_id, conv_title, msg.conversation_type, int(obj.get('member_count') or 1)))
                messages.append(msg)
        return list(accounts.values()), list(conversations.values()), messages
