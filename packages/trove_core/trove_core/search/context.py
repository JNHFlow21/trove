from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from trove_core.bounds import BoundSpec, BoundedLimit, CONTEXT_WINDOW, PRIVATE_LIST
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.domain.content import display_content_for_kind
from trove_core.domain.messages import ContextMessage

@dataclass
class ContextService:
    store: SQLiteStore
    max_window: int = 200

    def fetch(self, citation: str, before: int = 5, after: int = 5) -> dict[str, Any]:
        maximum = min(BoundedLimit(self.max_window, field='max_window', spec=CONTEXT_WINDOW), CONTEXT_WINDOW.maximum)
        window = BoundSpec(0, maximum, min(CONTEXT_WINDOW.default, maximum))
        before = BoundedLimit(before, field='before', spec=window)
        after = BoundedLimit(after, field='after', spec=window)
        rows = self.store.context_window(citation, before=before, after=after)
        if rows:
            hints = self.store.media_hints_for_citations([r['citation'] for r in rows])
            messages = []
            for r in rows:
                content = display_content_for_kind(r['content'], r['content_kind'] if 'content_kind' in r.keys() else 'text')
                item = ContextMessage(citation=r['citation'], sender_name=r['sender_name'], timestamp=r['timestamp'], content=content, direction=r['direction']).to_dict()
                if r['citation'] in hints:
                    item['media_hint'] = hints[r['citation']]
                messages.append(item)
            return {'citation': citation, 'before': before, 'after': after, 'messages': messages, 'evidence': None}
        evidence = self.store.evidence_by_citation(citation)
        if evidence is None:
            return {'citation': citation, 'before': before, 'after': after, 'messages': [], 'evidence': None}
        hint = self.store.media_hints_for_citations([citation, evidence['citation']]).get(citation) or self.store.media_hints_for_citations([evidence['citation']]).get(evidence['citation'])
        content = evidence['content']
        if hasattr(evidence, 'keys') and 'content_kind' in evidence.keys():
            content = display_content_for_kind(content, evidence['content_kind'])
        return {
            'citation': citation,
            'before': 0,
            'after': 0,
            'messages': [],
            'evidence': {
                'citation': evidence['citation'],
                'source_type': evidence['source_type'],
                'title': evidence['conversation_title'],
                'actor': evidence['sender_name'],
                'timestamp': evidence['timestamp'],
                'content': content[:1200],
                'media_hint': hint,
            },
        }

    def fetch_conversation(self, conversation_id: str, *, limit: int = 20) -> dict[str, Any]:
        limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
        if not self.store.path.exists():
            return {'conversation_id': conversation_id, 'limit': limit, 'messages': [], 'raw_content_included': False}
        with self.store.connect() as conn:
            rows = list(conn.execute(
                """SELECT * FROM (
                       SELECT * FROM messages
                       WHERE conversation_id=?
                       ORDER BY timestamp DESC, shard_id DESC, local_id DESC
                       LIMIT ?
                   ) ORDER BY timestamp ASC, shard_id ASC, local_id ASC""",
                (conversation_id, limit),
            ))
        hints = self.store.media_hints_for_citations([r['citation'] for r in rows]) if rows else {}
        messages = []
        for r in rows:
            content = display_content_for_kind(r['content'], r['content_kind'] if 'content_kind' in r.keys() else 'text')
            item = ContextMessage(citation=r['citation'], sender_name=r['sender_name'], timestamp=r['timestamp'], content=content, direction=r['direction']).to_dict()
            if r['citation'] in hints:
                item['media_hint'] = hints[r['citation']]
            messages.append(item)
        return {'conversation_id': conversation_id, 'limit': limit, 'messages': messages, 'evidence': None, 'raw_content_included': True}
