from __future__ import annotations

import re
from typing import Any


PLACEHOLDERS = {
    'voice': '[voice]',
    'image': '[image]',
    'video': '[video]',
    'sticker': '[sticker]',
    'quote': '[引用消息]',
    'appmsg': '[appmsg]',
    'unknown_binary': '[unknown_binary]',
}

LOCAL_TYPE_KIND = {
    1: 'text',
    3: 'image',
    34: 'voice',
    42: 'appmsg',
    43: 'video',
    47: 'sticker',
    48: 'appmsg',
    49: 'appmsg',
    50: 'appmsg',
    62: 'video',
    66: 'appmsg',
    67: 'appmsg',
    10000: 'text',
}
LOCAL_TYPE_NAME_KIND = {
    'text': 'text',
    'image': 'image',
    'img': 'image',
    'pic': 'image',
    'voice': 'voice',
    'audio': 'voice',
    'video': 'video',
    'sticker': 'sticker',
    'emoji': 'sticker',
    'appmsg': 'appmsg',
    'app': 'appmsg',
}


def decode_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        for enc in ('utf-8', 'utf-16le', 'gb18030'):
            try:
                return value.decode(enc).strip('\x00')
            except UnicodeDecodeError:
                continue
        return value.decode('utf-8', errors='ignore').strip('\x00')
    return str(value).strip('\x00')


def classify_content_kind(value: Any, local_type: Any = None) -> str:
    local_kind = _kind_from_local_type(local_type)
    if local_kind:
        return local_kind
    text = decode_text(value).strip()
    if not text:
        return 'unknown_binary' if isinstance(value, (bytes, bytearray)) else 'text'
    lowered = text.lower()
    if '\ufffd' in text or _control_ratio(text) > 0.08:
        return 'unknown_binary'
    if '<voicemsg' in lowered or '<voice' in lowered:
        return 'voice'
    if '<img' in lowered or '<image' in lowered or '<mediaobject' in lowered and 'image' in lowered:
        return 'image'
    if '<emoji' in lowered or '<sticker' in lowered:
        return 'sticker'
    if '<videomsg' in lowered or '<video' in lowered:
        return 'video'
    if '<quotemsg' in lowered or '<refermsg' in lowered or '<quote' in lowered:
        return 'quote'
    if '<appmsg' in lowered or lowered.startswith('<msg>') or lowered.startswith('<?xml'):
        return 'appmsg'
    if isinstance(value, (bytes, bytearray)) and not _looks_textual(text):
        return 'unknown_binary'
    return 'text'


def display_content_for_kind(content: str, content_kind: str) -> str:
    kind = content_kind or 'text'
    if kind == 'text':
        return str(content or '')
    if kind == 'appmsg' and str(content or '').startswith('[appmsg/') and len(str(content or '')) <= 1400:
        # Only the schema-driven AppMsg parser emits this marker.  It contains
        # bounded allowlisted fields, never the source XML or a fetchable URL.
        return str(content)
    return PLACEHOLDERS.get(kind, '[unknown_binary]')


def _control_ratio(text: str) -> float:
    if not text:
        return 0.0
    controls = sum(1 for ch in text if ord(ch) < 32 and ch not in '\r\n\t')
    return controls / max(len(text), 1)


def _looks_textual(text: str) -> bool:
    sample = text[:200]
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', sample))


def _kind_from_local_type(local_type: Any) -> str:
    if local_type is None or local_type == '':
        return ''
    try:
        type_value = int(local_type)
    except (TypeError, ValueError):
        return LOCAL_TYPE_NAME_KIND.get(str(local_type).strip().lower(), '') if isinstance(local_type, str) else ''
    base_type = type_value & 0xFFFFFFFF if type_value > 0xFFFFFFFF else type_value
    return LOCAL_TYPE_KIND.get(base_type, '')
