from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_citation: str
    parent_citation: str
    account_id: str
    account_label: str
    source_type: str
    source_id: str
    title: str
    actor: str
    timestamp: str
    content: str
    chunk_index: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_text(text: str, *, max_chars: int = 900, overlap_chars: int = 120) -> list[str]:
    text = re.sub(r'\s+', ' ', (text or '')).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        window = text[start:end]
        if end < len(text):
            # Prefer a natural CJK/Latin sentence boundary near the end.
            boundary = max(window.rfind('。'), window.rfind('！'), window.rfind('？'), window.rfind('. '), window.rfind('; '))
            if boundary > max_chars * 0.55:
                end = start + boundary + 1
                window = text[start:end]
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks
