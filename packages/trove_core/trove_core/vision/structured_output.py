from __future__ import annotations

import json
from typing import Any


def parse_structured_vision_text(text: str) -> dict[str, Any]:
    text = (text or '').strip()
    if not text:
        raise ValueError('empty vision output')
    if text.startswith('```'):
        text = text.strip('`')
        if text.lower().startswith('json'):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError('vision output is not valid JSON') from exc
    if not isinstance(data, dict):
        raise ValueError('vision output must be a JSON object')
    caption = str(data.get('caption') or '').strip()
    confidence = float(data.get('confidence', 0) or 0)
    if not caption:
        raise ValueError('vision output missing caption')
    if confidence < 0 or confidence > 1:
        raise ValueError('vision confidence must be between 0 and 1')
    return {
        'caption': caption,
        'visible_text': str(data.get('visible_text') or ''),
        'objects': data.get('objects') if isinstance(data.get('objects'), list) else [],
        'business_signals': data.get('business_signals') if isinstance(data.get('business_signals'), list) else [],
        'entity_mentions': data.get('entity_mentions') if isinstance(data.get('entity_mentions'), list) else [],
        'confidence': confidence,
    }
