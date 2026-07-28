from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

KNOWN_FIELDS = {
    'signature': 'signature',
    'region': 'region',
    'country': 'country',
    'province': 'province',
    'city': 'city',
    'gender': 'gender',
    'avatar_ref': 'avatar_ref',
    'avatar': 'avatar_ref',
    'head_img_url': 'avatar_ref',
}


@dataclass(frozen=True)
class ParsedContactExtra:
    fields: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip('\x00\ufeff')
    return str(value).strip('\x00\ufeff')


def _clean_text(value: Any) -> str:
    text = _decode(value)
    # Defend against arbitrary binary/string leakage into profile facts.
    if not text or len(text) > 240 or any(ord(ch) < 9 for ch in text):
        return ''
    return text.strip()


def parse_contact_extra_buffer(value: bytes | str | None) -> ParsedContactExtra:
    """Parse only explicit, known contact-extra schemas.

    Unknown buffers return diagnostics only. We intentionally do not scrape
    arbitrary readable substrings because that would turn noise into fake facts.
    """
    text = _decode(value)
    if not text:
        return ParsedContactExtra({}, {'status': 'empty'})
    payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = None
    if payload is None and text.startswith('trove_contact_extra_v1;'):
        payload = {}
        for part in text.split(';')[1:]:
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            payload[k.strip()] = v.strip()
    if payload is None:
        return ParsedContactExtra({}, {'status': 'unknown_schema', 'bytes': len(value or b'') if isinstance(value, bytes) else len(text)})
    fields: dict[str, str] = {}
    for key, target in KNOWN_FIELDS.items():
        if key not in payload:
            continue
        cleaned = _clean_text(payload.get(key))
        if cleaned:
            fields[target] = cleaned
    return ParsedContactExtra(fields, {'status': 'parsed' if fields else 'known_schema_no_fields', 'schema': 'json_or_trove_contact_extra_v1'})
