from __future__ import annotations
import re

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\u3000', ' ')).strip()

def safe_label(value: str) -> str:
    cleaned = normalize_text(value)
    cleaned = re.sub(r'[/\\\x00-\x1f]+', '-', cleaned)
    return cleaned[:80] or 'untitled'
