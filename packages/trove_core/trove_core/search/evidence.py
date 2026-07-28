from __future__ import annotations
from sqlite3 import Row

from trove_core.domain.messages import Evidence
from trove_core.wechat.normalize import normalize_text
from trove_core.domain.content import display_content_for_kind


def snippet(text: str, query: str, radius: int = 60) -> str:
    text = normalize_text(text)
    if not query:
        return text[: radius * 2]
    idx = text.lower().find(query.lower())
    if idx < 0:
        return text[: radius * 2]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(query) + radius)
    prefix = '…' if start else ''
    suffix = '…' if end < len(text) else ''
    return prefix + text[start:end] + suffix


def row_to_evidence(row: Row, query: str, retrieval_paths: list[str], score: float, media_hint: dict | None = None) -> Evidence:
    content = row['content']
    if hasattr(row, 'keys') and 'content_kind' in row.keys():
        content = display_content_for_kind(content, row['content_kind'])
    return Evidence(
        citation=row['citation'],
        account_id=row['account_id'],
        account_label=row['account_label'],
        conversation_id=row['conversation_id'],
        conversation_title=row['conversation_title'],
        conversation_type=row['conversation_type'],
        sender_name=row['sender_name'],
        timestamp=row['timestamp'],
        snippet=snippet(content, query),
        source_type=row['source_type'],
        direction=row['direction'],
        score=score,
        retrieval_paths=retrieval_paths,
        context_anchor=row['parent_citation'] if hasattr(row, 'keys') and 'parent_citation' in row.keys() else row['citation'],
        media_hint=media_hint,
    )
