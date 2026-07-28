from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CATALOG_BY_ID, PROTOCOL_VERSION, validate_input
from .errors import ProtocolError


MAX_FRAME_BYTES = 256 * 1024
SOFT_RESPONSE_BYTES = 64 * 1024
_REQUEST_FIELDS = frozenset({
    'protocol', 'request_id', 'capability', 'input', 'deadline_ms',
    'response_budget', 'vault_identity', 'build_hash', 'catalog_hash',
})


@dataclass(frozen=True)
class RequestFrame:
    request_id: str
    capability: str
    input: Mapping[str, Any]
    deadline_ms: int
    response_budget: int
    vault_identity: str | None = None
    build_hash: str | None = None
    catalog_hash: str | None = None
    protocol: str = PROTOCOL_VERSION


def encode_frame(payload: Mapping[str, Any], *, max_frame_bytes: int = MAX_FRAME_BYTES) -> bytes:
    try:
        body = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ProtocolError('invalid_frame', 'frame is not JSON serializable') from exc
    if len(body) > max_frame_bytes:
        raise ProtocolError(
            'frame_too_large', 'frame exceeds the protocol hard cap',
            details={'max_bytes': max_frame_bytes},
        )
    return struct.pack('>I', len(body)) + body


class FrameDecoder:
    def __init__(self, *, max_frame_bytes: int = MAX_FRAME_BYTES):
        if max_frame_bytes < 1:
            raise ValueError('max_frame_bytes must be positive')
        self.max_frame_bytes = max_frame_bytes
        self._buffer = bytearray()
        self._expected: int | None = None

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        if not isinstance(chunk, bytes):
            raise ProtocolError('invalid_frame', 'frame chunk must be bytes')
        self._buffer.extend(chunk)
        frames: list[dict[str, Any]] = []
        while True:
            if self._expected is None:
                if len(self._buffer) < 4:
                    break
                self._expected = struct.unpack('>I', self._buffer[:4])[0]
                del self._buffer[:4]
                if self._expected > self.max_frame_bytes:
                    self._buffer.clear()
                    expected = self._expected
                    self._expected = None
                    raise ProtocolError(
                        'frame_too_large', 'declared frame exceeds the protocol hard cap',
                        details={'declared_bytes': expected, 'max_bytes': self.max_frame_bytes},
                    )
                if self._expected == 0:
                    self._expected = None
                    raise ProtocolError('invalid_frame', 'empty frames are not allowed')
            if len(self._buffer) < self._expected:
                break
            body = bytes(self._buffer[:self._expected])
            del self._buffer[:self._expected]
            self._expected = None
            try:
                payload = json.loads(body.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProtocolError('invalid_frame', 'frame must contain one UTF-8 JSON object') from exc
            if not isinstance(payload, dict):
                raise ProtocolError('invalid_frame', 'frame JSON must be an object')
            frames.append(payload)
        return frames

    def finish(self) -> None:
        if self._buffer or self._expected is not None:
            self._buffer.clear()
            self._expected = None
            raise ProtocolError('partial_frame', 'stream ended before a complete frame arrived')


def decode_request(payload: Mapping[str, Any], *, now_ms: int | None = None) -> RequestFrame:
    if not isinstance(payload, Mapping):
        raise ProtocolError('invalid_request', 'request must be a JSON object')
    unknown = set(payload) - _REQUEST_FIELDS
    if unknown:
        raise ProtocolError('invalid_request', 'request contains unknown fields', details={'fields': sorted(unknown)})
    required = {'protocol', 'request_id', 'capability', 'input', 'deadline_ms', 'response_budget'}
    missing = required - set(payload)
    if missing:
        raise ProtocolError('invalid_request', 'request is missing required fields', details={'fields': sorted(missing)})
    if payload.get('protocol') != PROTOCOL_VERSION:
        raise ProtocolError(
            'protocol_mismatch', 'request protocol is incompatible',
            details={'supported': [PROTOCOL_VERSION]},
        )
    request_id = payload.get('request_id')
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ProtocolError('invalid_request', 'request_id is invalid')
    capability = payload.get('capability')
    if capability not in CATALOG_BY_ID:
        raise ProtocolError('unknown_capability', 'capability is not in the reviewed catalog')
    deadline_ms = payload.get('deadline_ms')
    if type(deadline_ms) is not int:
        raise ProtocolError('invalid_request', 'deadline_ms must be an integer')
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if deadline_ms <= current_ms:
        raise ProtocolError('deadline_expired', 'request deadline has expired', retryable=True)
    response_budget = payload.get('response_budget')
    if type(response_budget) is not int or not 1 <= response_budget <= MAX_FRAME_BYTES:
        raise ProtocolError('invalid_request', 'response_budget is outside protocol bounds')
    try:
        normalized_input = validate_input(CATALOG_BY_ID[str(capability)], payload.get('input'))
    except ValueError as exc:
        raise ProtocolError('invalid_request', str(exc)) from exc
    optional_strings: dict[str, str | None] = {}
    for field in ('vault_identity', 'build_hash', 'catalog_hash'):
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ProtocolError('invalid_request', f'{field} must be a non-empty string')
        optional_strings[field] = value
    return RequestFrame(
        request_id=request_id,
        capability=str(capability),
        input=normalized_input,
        deadline_ms=deadline_ms,
        response_budget=response_budget,
        **optional_strings,
    )


__all__ = [
    'FrameDecoder', 'MAX_FRAME_BYTES', 'RequestFrame', 'SOFT_RESPONSE_BYTES',
    'decode_request', 'encode_frame',
]
