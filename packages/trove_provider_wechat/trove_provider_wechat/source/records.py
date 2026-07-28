from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class SourceRecordError(ValueError):
    code = 'source_record_invalid'


def _bounded_text(value: object, field: str, *, limit: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise SourceRecordError(f'{field} is invalid')
    return value


def source_accounts(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    accounts = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()):
        metadata_path = directory / 'account.json'
        records_path = directory / 'messages.jsonl'
        if not metadata_path.is_file() or not records_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        account_id = _bounded_text(metadata.get('account_id'), 'account_id', limit=256)
        label = _bounded_text(metadata.get('label'), 'label', limit=256)
        rows, watermark = load_account_records(directory, account_id=account_id)
        accounts.append({
            'account_id': account_id, 'label': label,
            'message_count': len(rows), 'watermark': watermark,
        })
    return accounts


def load_account_records(directory: Path, *, account_id: str) -> tuple[list[dict[str, Any]], str]:
    path = directory / 'messages.jsonl'
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise SourceRecordError('account record file is unavailable') from exc
    for index, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
        try:
            row = json.loads(raw.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SourceRecordError('account record is not UTF-8 JSON') from exc
        required = {
            'conversation_id', 'conversation_title', 'conversation_type',
            'sender_id', 'sender_name', 'timestamp', 'content', 'shard_id', 'local_id',
        }
        if not isinstance(row, dict) or not required <= set(row):
            raise SourceRecordError('account record fields are incomplete')
        conversation_id = _bounded_text(row['conversation_id'], 'conversation_id', limit=256)
        shard_id = _bounded_text(row['shard_id'], 'shard_id', limit=256)
        local_id = row['local_id']
        if type(local_id) is not int or local_id < 0:
            raise SourceRecordError('local_id is invalid')
        record = {
            'account_id': account_id,
            'conversation_id': conversation_id,
            'conversation_title': _bounded_text(row['conversation_title'], 'conversation_title'),
            'conversation_type': row['conversation_type'],
            'sender_id': _bounded_text(row['sender_id'], 'sender_id', limit=256),
            'sender_name': _bounded_text(row['sender_name'], 'sender_name', limit=256),
            'timestamp': _bounded_text(row['timestamp'], 'timestamp', limit=64),
            'content': str(row['content'])[:256 * 1024],
            'shard_id': shard_id,
            'local_id': local_id,
            'direction': row.get('direction', 'unknown'),
            'content_kind': row.get('content_kind', 'text'),
            'citation': f'trove://wechat/{account_id}/{conversation_id}/{shard_id}/{local_id}',
            'trust': 'untrusted_evidence',
        }
        if record['conversation_type'] not in {'private', 'group'}:
            raise SourceRecordError('conversation_type is invalid')
        if record['direction'] not in {'incoming', 'outgoing', 'unknown'}:
            raise SourceRecordError('direction is invalid')
        records.append(record)
    records.sort(key=lambda item: (item['timestamp'], item['shard_id'], item['local_id']))
    return records, digest.hexdigest()
