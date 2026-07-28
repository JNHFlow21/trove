from __future__ import annotations
from trove_core.store.sqlite_store import SQLiteStore


def build_conversation_card(store: SQLiteStore, account_id: str = 'acct-work', conversation_id: str = 'conv-trove-team') -> dict:
    rows = store.messages_for_conversation(account_id, conversation_id)
    decisions = []
    owners = []
    open_questions = []
    evidence = []
    for row in rows:
        item = {'citation': row['citation'], 'speaker': row['sender_name'], 'timestamp': row['timestamp'], 'text': row['content']}
        evidence.append(item)
        if '决定' in row['content'] or '必须' in row['content']:
            decisions.append({'value': row['content'], 'citation': row['citation']})
        if '负责' in row['content']:
            owners.append({'value': row['content'], 'citation': row['citation']})
        if '？' in row['content'] or '?' in row['content']:
            open_questions.append({'value': row['content'], 'citation': row['citation']})
    return {
        'type': 'conversation_card',
        'conversation_id': conversation_id,
        'conversation_title': rows[0]['conversation_title'] if rows else conversation_id,
        'decisions': decisions,
        'owners': owners,
        'open_questions': open_questions,
        'evidence': evidence,
        'citation_policy': 'Only cited messages are summarized; empty sections mean no cited evidence found.',
    }
