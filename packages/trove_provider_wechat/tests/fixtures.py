from __future__ import annotations

import json
from pathlib import Path


def write_account(root: Path, account_id: str, *, label: str, token: str, rows: int = 2) -> Path:
    account = root / account_id
    account.mkdir(parents=True)
    (account / 'account.json').write_text(json.dumps({
        'account_id': account_id, 'label': label,
    }) + '\n', encoding='utf-8')
    records = []
    for index in range(rows):
        records.append({
            'conversation_id': f'conversation-{account_id}',
            'conversation_title': f'Conversation {label}',
            'conversation_type': 'private',
            'sender_id': f'peer-{account_id}', 'sender_name': f'Peer {label}',
            'timestamp': f'2026-01-01T00:00:0{index}Z',
            'content': f'{token}-{index}', 'shard_id': 'fixture',
            'local_id': index + 1, 'direction': 'incoming',
        })
    (account / 'messages.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records),
        encoding='utf-8',
    )
    return account
