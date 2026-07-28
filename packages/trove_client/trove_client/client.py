from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from typing import Any, Mapping

from trove_daemon.lifecycle import RuntimeIdentity
from trove_daemon.session import SessionContract
from trove_protocol.capabilities import CATALOG_BY_ID
from trove_protocol.codec import MAX_FRAME_BYTES, encode_frame

from .autostart import AutostartCoordinator, AutostartError


_DEFAULT_AUTOSTART = object()


class TroveClientError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        response: Mapping[str, Any] | None = None,
    ):
        self.code = code
        self.retryable = retryable
        self.response = dict(response or {})
        super().__init__(message)


def _receive(connection: socket.socket) -> dict[str, Any]:
    header = bytearray()
    while len(header) < 4:
        chunk = connection.recv(4 - len(header))
        if not chunk:
            raise OSError('daemon disconnected before response header')
        header.extend(chunk)
    length = struct.unpack('>I', header)[0]
    if not 1 <= length <= MAX_FRAME_BYTES:
        raise OSError('daemon response frame exceeds bounds')
    body = bytearray()
    while len(body) < length:
        chunk = connection.recv(length - len(body))
        if not chunk:
            raise OSError('daemon disconnected during response')
        body.extend(chunk)
    payload = json.loads(body.decode('utf-8'))
    if not isinstance(payload, dict):
        raise OSError('daemon response is not an object')
    return payload


class TroveClient:
    def __init__(
        self,
        identity: RuntimeIdentity,
        *,
        pool_size: int = 4,
        autostart: AutostartCoordinator | None | object = _DEFAULT_AUTOSTART,
        role: str = 'sdk',
        connect_timeout: float = 2.0,
    ):
        if pool_size < 1:
            raise ValueError('pool_size must be positive')
        self.identity = identity
        self.pool_size = pool_size
        self.role = role
        self.connect_timeout = float(connect_timeout)
        self.autostart = (
            AutostartCoordinator.system(identity)
            if autostart is _DEFAULT_AUTOSTART else autostart
        )
        self._pool: list[socket.socket] = []
        self._pool_lock = threading.Lock()
        self._leases = threading.BoundedSemaphore(pool_size)
        self._closed = False

    def __enter__(self) -> 'TroveClient':
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _connect(self) -> socket.socket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.connect_timeout)
        try:
            connection.connect(str(self.identity.socket_path))
            hello = SessionContract(self.identity).client_hello(
                client_id=f'{self.role}-{os.getpid()}-{threading.get_ident()}',
                role=self.role,
            )
            connection.sendall(encode_frame(hello))
            response = _receive(connection)
            if response.get('ok') is not True:
                self._raise_response(response)
            return connection
        except BaseException:
            connection.close()
            raise

    def _checkout(self) -> socket.socket:
        with self._pool_lock:
            if self._closed:
                raise TroveClientError('client_closed', 'TROVE client is closed.')
            if self._pool:
                return self._pool.pop()
        return self._connect()

    def _checkin(self, connection: socket.socket) -> None:
        with self._pool_lock:
            if not self._closed and len(self._pool) < self.pool_size:
                self._pool.append(connection)
                return
        connection.close()

    def call(
        self,
        capability: str,
        payload: Mapping[str, Any],
        *,
        request_id: str,
        timeout: float = 30.0,
        response_budget: int | None = None,
    ) -> dict[str, Any]:
        spec = CATALOG_BY_ID.get(capability)
        if spec is None:
            raise TroveClientError('unknown_capability', 'Capability is not in the reviewed catalog.')
        deadline = time.time() + max(0.001, float(timeout))
        if not self._leases.acquire(timeout=max(0.001, timeout)):
            raise TroveClientError('busy', 'Client connection pool is saturated.', retryable=True)
        try:
            restarted = False
            while True:
                connection = None
                try:
                    connection = self._checkout()
                    connection.settimeout(max(0.001, deadline - time.time()))
                    connection.sendall(encode_frame({
                        'protocol': 'trove/1', 'request_id': request_id,
                        'capability': capability, 'input': dict(payload),
                        'deadline_ms': int(deadline * 1000),
                        'response_budget': min(
                            spec.response_budget,
                            spec.response_budget if response_budget is None else response_budget,
                        ),
                        'vault_identity': self.identity.vault_identity,
                        'build_hash': self.identity.build_hash,
                        'catalog_hash': self.identity.catalog_hash,
                    }))
                    response = _receive(connection)
                    self._checkin(connection)
                    connection = None
                    if response.get('ok') is not True:
                        self._raise_response(response)
                    return response
                except TroveClientError:
                    if connection is not None:
                        connection.close()
                    raise
                except socket.timeout as exc:
                    if connection is not None:
                        connection.close()
                    raise TroveClientError(
                        'timeout', 'Daemon request exceeded its deadline.', retryable=True,
                    ) from exc
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    if connection is not None:
                        connection.close()
                    if restarted or self.autostart is None:
                        raise TroveClientError(
                            'daemon_unavailable', 'Daemon transport is unavailable.', retryable=True,
                        ) from exc
                    restarted = True
                    try:
                        self.autostart.ensure_running()
                    except AutostartError as start_error:
                        raise TroveClientError(
                            'daemon_unavailable', 'Daemon restart failed.', retryable=True,
                        ) from start_error
        finally:
            self._leases.release()

    @staticmethod
    def _raise_response(response: Mapping[str, Any]) -> None:
        error = response.get('error') if isinstance(response.get('error'), Mapping) else {}
        raise TroveClientError(
            str(error.get('code') or 'daemon_error'),
            str(error.get('message') or 'Daemon rejected the request.'),
            retryable=bool(error.get('retryable')),
            response=response,
        )

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            connections, self._pool = self._pool, []
        for connection in connections:
            connection.close()


__all__ = ['TroveClient', 'TroveClientError']
