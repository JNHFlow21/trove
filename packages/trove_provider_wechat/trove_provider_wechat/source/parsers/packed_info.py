from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MEDIA_PATH_RE = re.compile(r'(?i)([^\s\x00"\']+\.(?:jpg|jpeg|png|gif|webp|heic|dat|mp3|m4a|wav|amr|silk|mp4|mov))')
KNOWN_KEYS = ('path', 'file_path', 'local_path', 'relative_path', 'file_name', 'filename', 'cache_key', 'md5', 'sha256', 'media_type', 'type')


@dataclass(frozen=True)
class ParsedPackedInfo:
    fields: dict[str, Any] = field(default_factory=dict)
    path_hints: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip('\x00\ufeff')
    return str(value).strip('\x00\ufeff')


def _clean(value: Any) -> str:
    text = _decode(value).strip()
    if not text or len(text) > 1024:
        return ''
    return text


def parse_packed_info_blob(value: bytes | str | None) -> ParsedPackedInfo:
    """Schema-driven packed_info parser with a conservative fallback.

    Facts are extracted only from JSON / explicit trove prefix schemas. The
    fallback returns path hints for media linking only; it does not promote
    unknown fields into customer facts.
    """
    text = _decode(value)
    if not text:
        return ParsedPackedInfo({}, [], {'status': 'empty'})
    payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = None
    if payload is None and text.startswith('trove_packed_info_v1;'):
        payload = {}
        for part in text.split(';')[1:]:
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            payload[k.strip()] = v.strip()
    fields: dict[str, Any] = {}
    path_hints: list[str] = []
    if payload is not None:
        for key in KNOWN_KEYS:
            if key in payload:
                cleaned = _clean(payload.get(key))
                if cleaned:
                    fields[key] = cleaned
                    if key in {'path', 'file_path', 'local_path', 'relative_path', 'file_name', 'filename'}:
                        path_hints.append(cleaned)
        return ParsedPackedInfo(fields, list(dict.fromkeys(path_hints)), {'status': 'parsed' if fields else 'known_schema_no_fields'})
    # Conservative media-link fallback only: a concrete media-looking path can
    # help cache mapping, but no semantic facts are inferred from it.
    hints = [m.group(1) for m in MEDIA_PATH_RE.finditer(text[:4096])]
    return ParsedPackedInfo({}, list(dict.fromkeys(hints))[:5], {'status': 'path_hint_only' if hints else 'unknown_schema'})
