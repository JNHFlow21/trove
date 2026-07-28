from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET


PARSER_VERSION = 'appmsg-v2'
MAX_SOURCE_BYTES = 256 * 1024
MAX_TEXT = 500
MAX_DISPLAY = 1400

_WECHAT_GROUP_SENDER_PREFIX = re.compile(
    r'\A[A-Za-z0-9_@.-]{1,128}:\r?\n(?=<)'
)

_TYPE_NAMES = {
    1: 'text_card',
    8: 'image_card',
    3: 'music',
    4: 'video_card',
    5: 'link',
    6: 'file',
    17: 'location',
    19: 'merged_chat',
    24: 'note',
    33: 'mini_program',
    36: 'mini_program',
    48: 'location',
    57: 'quote',
    62: 'generic_card',
    2000: 'transfer_notice',
}


@dataclass(frozen=True)
class ParsedAppMessage:
    source_hash: str
    appmsg_type: int | None
    normalized_type: str
    parse_status: str
    fields: dict[str, Any] = field(default_factory=dict)
    display_text: str = '[appmsg/unsupported]'
    unsupported_reason: str | None = None
    parser_version: str = PARSER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b''
    if isinstance(value, bytes):
        return value
    return str(value).encode('utf-8', errors='replace')


def _source_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode(value: bytes) -> str:
    for encoding in ('utf-8', 'utf-16le', 'gb18030'):
        try:
            return value.decode(encoding).strip('\x00\ufeff')
        except UnicodeDecodeError:
            continue
    return value.decode('utf-8', errors='replace').strip('\x00\ufeff')


def _tag_name(tag: str) -> str:
    return str(tag).split('}', 1)[-1].lower()


def _clean_text(value: Any, *, limit: int = MAX_TEXT, redact_links: bool = True) -> str:
    text = str(value or '').replace('\x00', '').strip()
    text = ''.join(ch if ch in '\r\n\t' or ord(ch) >= 32 else ' ' for ch in text)
    text = re.sub(r'\s+', ' ', text).strip()
    if redact_links:
        text = re.sub(r'(?i)https?://\S+', '[link]', text)
    return text[:limit]


def _find(element: ET.Element, *names: str) -> ET.Element | None:
    wanted = {name.lower() for name in names}
    for child in element.iter():
        if _tag_name(child.tag) in wanted:
            return child
    return None


def _find_text(element: ET.Element, *names: str, limit: int = MAX_TEXT, redact_links: bool = True) -> str:
    found = _find(element, *names)
    return _clean_text(found.text if found is not None else '', limit=limit, redact_links=redact_links)


def _bounded_int(value: str, *, minimum: int = 0, maximum: int = 10**15) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if minimum <= parsed <= maximum else None


def _bounded_float(value: str, *, minimum: float, maximum: float) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return round(parsed, 6) if minimum <= parsed <= maximum else None


def _safe_link_identity(value: str) -> dict[str, Any]:
    raw = _clean_text(value, limit=4096, redact_links=False)
    if not raw:
        return {}
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or '').rstrip('.').lower().encode('idna').decode('ascii')
        if scheme not in {'http', 'https'} or not host or len(host) > 253:
            return {}
        port = parsed.port
    except (UnicodeError, ValueError):
        return {}
    identity: dict[str, Any] = {
        'scheme': scheme,
        'host': host,
        'path_hash': hashlib.sha256((parsed.path or '/').encode('utf-8', errors='replace')).hexdigest(),
    }
    if port is not None and port not in {80, 443}:
        identity['port'] = port
    return identity


def _display(normalized_type: str, fields: dict[str, Any]) -> str:
    parts = [f'[appmsg/{normalized_type}]']
    for key in ('title', 'description', 'file_name', 'location_label', 'quote_sender', 'quote_title'):
        value = _clean_text(fields.get(key), limit=MAX_TEXT)
        if value and value not in parts:
            parts.append(value)
    if normalized_type == 'call' and fields.get('duration_seconds') is not None:
        parts.append(f"时长{int(fields['duration_seconds'])}秒")
    if normalized_type == 'image_card':
        if fields.get('file_extension'):
            parts.append(str(fields['file_extension']))
        if fields.get('file_size') is not None:
            parts.append(f"{int(fields['file_size'])}B")
    return ' '.join(parts)[:MAX_DISPLAY]


def parse_appmsg(value: bytes | str | None) -> ParsedAppMessage:
    source = _source_bytes(value)
    digest = _source_hash(source)
    if not source:
        return ParsedAppMessage(digest, None, 'unsupported', 'unsupported', {}, '[appmsg/unsupported]', 'empty_payload')
    if len(source) > MAX_SOURCE_BYTES:
        return ParsedAppMessage(digest, None, 'unsupported', 'rejected', {}, '[appmsg/unsupported]', 'payload_too_large')
    text = _decode(source)
    lowered = text.lower()
    if '<!doctype' in lowered or '<!entity' in lowered:
        return ParsedAppMessage(digest, None, 'unsupported', 'rejected', {}, '[appmsg/unsupported]', 'unsafe_xml_construct')
    # WeChat group rows prepend the real sender identifier before the XML
    # payload (for example, ``wxid_sender:\n<msg>...``).  Preserve the hash of
    # the original bytes, but parse only the bounded XML envelope.
    text = _WECHAT_GROUP_SENDER_PREFIX.sub('', text, count=1)
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, ValueError):
        return ParsedAppMessage(digest, None, 'unsupported', 'malformed', {}, '[appmsg/unsupported]', 'malformed_xml')
    appmsg = root if _tag_name(root.tag) == 'appmsg' else _find(root, 'appmsg')
    forced_type: str | None = None
    forced_appmsg_type: int | None = None
    if appmsg is None:
        if _find(root, 'location') is not None:
            appmsg = root
            forced_type = 'location'
            forced_appmsg_type = 48
        elif _tag_name(root.tag) == 'voipmsg' or _find(root, 'voipbubblemsg') is not None:
            appmsg = root
            forced_type = 'call'
            forced_appmsg_type = 50
        else:
            return ParsedAppMessage(digest, None, 'unsupported', 'unsupported', {}, '[appmsg/unsupported]', 'missing_appmsg')

    raw_type = _find_text(appmsg, 'type', limit=20) if forced_type is None else ''
    appmsg_type = forced_appmsg_type if forced_type is not None else _bounded_int(raw_type, maximum=10**9)
    normalized_type = forced_type or _TYPE_NAMES.get(appmsg_type or -1, 'unsupported')
    if normalized_type == 'unsupported':
        if _find(appmsg, 'weappinfo') is not None:
            normalized_type = 'mini_program'
        elif _find(appmsg, 'refermsg') is not None:
            normalized_type = 'quote'
        elif _find(appmsg, 'location') is not None:
            normalized_type = 'location'

    fields: dict[str, Any] = {}
    title = _find_text(appmsg, 'title')
    description = _find_text(appmsg, 'des', 'description')
    if title:
        fields['title'] = title
    if description:
        fields['description'] = description

    link = _safe_link_identity(_find_text(appmsg, 'url', 'lowurl', limit=4096, redact_links=False))
    if link:
        fields['link_identity'] = link

    if normalized_type in {'file', 'image_card'}:
        appattach = _find(appmsg, 'appattach')
        if appattach is None:
            appattach = appmsg
        file_name = _find_text(appmsg, 'filename', 'file_name') or title
        extension = _find_text(appattach, 'fileext', limit=24).lower().lstrip('.')
        file_size = _bounded_int(_find_text(appattach, 'totallen', 'filesize', limit=32))
        if file_name:
            fields['file_name'] = file_name
        if extension and re.fullmatch(r'[a-z0-9]{1,16}', extension):
            fields['file_extension'] = extension
        if file_size is not None:
            fields['file_size'] = file_size
    elif normalized_type == 'mini_program':
        weapp = _find(appmsg, 'weappinfo')
        if weapp is None:
            weapp = appmsg
        app_id = _find_text(weapp, 'appid', limit=128)
        username = _find_text(weapp, 'username', limit=128)
        page_path = _find_text(weapp, 'pagepath', limit=2048)
        if app_id:
            fields['mini_program_app_id'] = app_id
        if username:
            fields['mini_program_username'] = username
        if page_path:
            fields['mini_program_page_hash'] = hashlib.sha256(page_path.split('?', 1)[0].encode('utf-8')).hexdigest()
    elif normalized_type == 'quote':
        refer = _find(appmsg, 'refermsg')
        if refer is None:
            refer = appmsg
        quote_type = _bounded_int(_find_text(refer, 'type', limit=20), maximum=10**9)
        quote_sender = _find_text(refer, 'displayname', 'sender', limit=200)
        quote_title = _find_text(refer, 'title', limit=MAX_TEXT)
        if quote_type is not None:
            fields['quote_type'] = quote_type
        if quote_sender:
            fields['quote_sender'] = quote_sender
        if quote_title:
            fields['quote_title'] = quote_title
    elif normalized_type == 'location':
        location = _find(appmsg, 'location')
        if location is None:
            location = appmsg
        label = _clean_text(location.attrib.get('label') or location.attrib.get('poiname') or location.text)
        latitude = _bounded_float(location.attrib.get('x', ''), minimum=-90.0, maximum=90.0)
        longitude = _bounded_float(location.attrib.get('y', ''), minimum=-180.0, maximum=180.0)
        if label:
            fields['location_label'] = label
        if latitude is not None:
            fields['latitude'] = latitude
        if longitude is not None:
            fields['longitude'] = longitude
    elif normalized_type == 'call':
        duration = _bounded_int(_find_text(appmsg, 'duration', limit=32), maximum=7 * 24 * 60 * 60)
        room_type = _bounded_int(_find_text(appmsg, 'room_type', limit=20), maximum=1000)
        message_type = _bounded_int(_find_text(appmsg, 'msg_type', limit=20), maximum=1000)
        if duration is not None:
            fields['duration_seconds'] = duration
        if room_type is not None:
            fields['room_type'] = room_type
        if message_type is not None:
            fields['message_type'] = message_type

    if normalized_type == 'unsupported':
        return ParsedAppMessage(
            digest, appmsg_type, normalized_type, 'unsupported', fields,
            _display(normalized_type, fields), 'unsupported_appmsg_type',
        )
    return ParsedAppMessage(
        digest, appmsg_type, normalized_type, 'parsed', fields,
        _display(normalized_type, fields), None,
    )
