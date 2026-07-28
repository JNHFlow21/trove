from __future__ import annotations

import json
import math
import socket
from typing import Any

from .daemon_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PROTOCOL_VERSION,
    DaemonIdentity,
    DaemonProtocolError,
    safe_error_code,
    validate_texts,
)


class EmbeddingDaemonClient:
    def __init__(
        self,
        socket_path: str,
        *,
        identity: DaemonIdentity | None = None,
        timeout_ms: int = 500,
    ) -> None:
        self.socket_path = socket_path
        self.identity = identity
        self.timeout_ms = max(1, min(int(timeout_ms), 120_000))

    def _request(self, payload: dict[str, Any], *, timeout_ms: int | None = None) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'
        if len(encoded) > MAX_REQUEST_BYTES:
            raise DaemonProtocolError('daemon_request_too_large')
        timeout = max(1, min(int(timeout_ms or self.timeout_ms), 120_000)) / 1000.0
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(self.socket_path)
                sock.sendall(encoded)
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise DaemonProtocolError('daemon_response_too_large')
                    if b'\n' in chunk:
                        break
        except socket.timeout:
            raise DaemonProtocolError('daemon_request_timeout') from None
        except FileNotFoundError:
            raise DaemonProtocolError('daemon_unavailable') from None
        except ConnectionRefusedError:
            raise DaemonProtocolError('daemon_unavailable') from None
        except OSError:
            raise DaemonProtocolError('daemon_transport_error') from None
        line = b''.join(chunks).split(b'\n', 1)[0]
        try:
            data = json.loads(line.decode('utf-8'))
        except Exception:
            raise DaemonProtocolError('daemon_protocol_error') from None
        if type(data) is not dict:
            raise DaemonProtocolError('daemon_protocol_error')
        error = data.get('error')
        if type(error) is dict and type(error.get('code')) is str:
            code = safe_error_code(error['code'])
            raise DaemonProtocolError(code if code != 'daemon_internal_error' else 'daemon_protocol_error')
        return data

    def handshake(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'op': 'handshake', 'protocol_version': PROTOCOL_VERSION}
        if self.identity is not None:
            payload['identity'] = self.identity.to_dict()
        data = self._request(payload)
        actual = DaemonIdentity.from_dict(data.get('identity'))
        if self.identity is not None:
            self.identity.require_match(actual)
        return data

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        texts = validate_texts(texts)
        if self.identity is None:
            raise DaemonProtocolError('daemon_identity_missing')
        data = self._request({
            'op': 'embed',
            'protocol_version': PROTOCOL_VERSION,
            'identity': self.identity.to_dict(),
            'timeout_ms': self.timeout_ms,
            'texts': texts,
        })
        actual = DaemonIdentity.from_dict(data.get('identity'))
        self.identity.require_match(actual)
        vectors = data.get('vectors')
        if type(vectors) is not list or len(vectors) != len(texts):
            raise DaemonProtocolError('daemon_vector_count_mismatch')
        parsed: list[list[float]] = []
        for vector in vectors:
            if type(vector) is not list:
                raise DaemonProtocolError('daemon_vector_invalid')
            try:
                row = [float(value) for value in vector]
            except (TypeError, ValueError):
                raise DaemonProtocolError('daemon_vector_invalid') from None
            if actual.dimensions and len(row) != actual.dimensions:
                raise DaemonProtocolError('daemon_dimensions_mismatch')
            if any(not math.isfinite(value) for value in row):
                raise DaemonProtocolError('daemon_vector_invalid')
            parsed.append(row)
        telemetry = data.get('telemetry') if type(data.get('telemetry')) is dict else {}
        return parsed, telemetry
