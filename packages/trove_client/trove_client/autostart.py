from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
from typing import Callable, Literal

from trove_core.managed_process import ManagedProcessError, ManagedProcessManager
from trove_protocol.codec import MAX_FRAME_BYTES, encode_frame
from trove_daemon.lifecycle import RuntimeIdentity
from trove_daemon.session import SessionContract


ProbeState = Literal['compatible', 'incompatible', 'unavailable']
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class AutostartError(RuntimeError):
    code = 'daemon_autostart_failed'


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _read_frame(connection: socket.socket) -> dict:
    header = bytearray()
    while len(header) < 4:
        chunk = connection.recv(4 - len(header))
        if not chunk:
            raise OSError('daemon closed before frame header')
        header.extend(chunk)
    size = struct.unpack('>I', header)[0]
    if size < 1 or size > MAX_FRAME_BYTES:
        raise OSError('daemon frame size is invalid')
    body = bytearray()
    while len(body) < size:
        chunk = connection.recv(size - len(body))
        if not chunk:
            raise OSError('daemon closed during frame')
        body.extend(chunk)
    value = json.loads(body.decode('utf-8'))
    if not isinstance(value, dict):
        raise OSError('daemon frame is not an object')
    return value


def probe_daemon(identity: RuntimeIdentity, *, timeout: float = 0.3) -> ProbeState:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect(str(identity.socket_path))
        hello = SessionContract(identity).client_hello(
            client_id=f'autostart-{os.getpid()}', role='sdk', persistent=False,
        )
        connection.sendall(encode_frame(hello))
        response = _read_frame(connection)
    except (OSError, ValueError, json.JSONDecodeError):
        return 'unavailable'
    finally:
        connection.close()
    if response.get('ok') is True:
        return 'compatible'
    code = (response.get('error') or {}).get('code')
    if code in {
        'version_incompatible', 'catalog_mismatch', 'vault_identity_mismatch',
        'protocol_mismatch',
    }:
        return 'incompatible'
    return 'unavailable'


class AutostartCoordinator:
    """Cross-thread/process single-attempt daemon start and build replacement."""

    def __init__(
        self,
        identity: RuntimeIdentity,
        *,
        probe: Callable[[], ProbeState],
        start: Callable[[], object],
        stop: Callable[[float], object] | None = None,
        replace_timeout: float = 5.0,
    ):
        self.identity = identity
        self.probe = probe
        self.start_callback = start
        self.stop_callback = stop
        self.replace_timeout = float(replace_timeout)

    @classmethod
    def system(cls, identity: RuntimeIdentity, *, replace_timeout: float = 5.0) -> 'AutostartCoordinator':
        manager = ManagedProcessManager(identity.runtime_dir)

        def start():
            command = [
                sys.executable, '-m', 'trove_daemon.main',
                '--vault', str(identity.vault_root),
                '--build-hash', identity.build_hash,
                '--catalog-hash', identity.catalog_hash,
            ]
            return manager.start(
                'daemon', command,
                health_endpoint=f'unix://{identity.socket_path}',
                cwd=Path.cwd(), readiness_timeout=8.0,
            )

        def stop(timeout: float):
            return manager.stop('daemon', timeout=timeout)

        return cls(
            identity, probe=lambda: probe_daemon(identity), start=start, stop=stop,
            replace_timeout=replace_timeout,
        )

    def ensure_running(self) -> None:
        self.identity.prepare()
        lock_path = self.identity.lock_path
        with _thread_lock(lock_path):
            fd = os.open(lock_path, os.O_RDWR | int(getattr(os, 'O_NOFOLLOW', 0)))
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                state = self.probe()
                if state == 'compatible':
                    return
                if state == 'incompatible':
                    if self.stop_callback is None:
                        raise AutostartError('An incompatible daemon is running and cannot be replaced.')
                    stopped = self.stop_callback(self.replace_timeout)
                    if isinstance(stopped, dict) and stopped.get('ok') is not True:
                        raise AutostartError('Incompatible daemon did not drain within the replacement bound.')
                try:
                    self.start_callback()
                except (OSError, ManagedProcessError) as exc:
                    raise AutostartError('Daemon failed to start.') from exc
                if self.probe() != 'compatible':
                    raise AutostartError('Daemon did not become compatible after one start attempt.')
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


__all__ = [
    'AutostartCoordinator', 'AutostartError', 'ProbeState', 'probe_daemon',
]
