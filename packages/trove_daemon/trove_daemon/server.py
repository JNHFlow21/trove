from __future__ import annotations

import concurrent.futures
from contextlib import nullcontext
import json
import os
import queue
import socket
import struct
import threading
import time
from typing import Any, Callable, Mapping

from trove_core.vault.generation import vault_generation_read
from trove_protocol.capabilities import CATALOG_BY_ID
from trove_protocol.codec import MAX_FRAME_BYTES, decode_request, encode_frame
from trove_protocol.errors import ProtocolError

from .cursors import DaemonCursorStore
from .lifecycle import RuntimeIdentity, require_macos
from .session import SessionContract, SessionError


DEFAULT_IDLE_TIMEOUT_SECONDS = 60.0


def _typed_error(
    code: str,
    message: str,
    *,
    request_id: str = 'unknown',
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {'code': code, 'retryable': retryable, 'message': message}
    if details:
        error['details'] = dict(details)
    return {'ok': False, 'request_id': request_id, 'error': error}


def _current_peer_uid(connection: socket.socket) -> int:
    require_macos()
    try:
        raw = connection.getsockopt(0, socket.LOCAL_PEERCRED, 12)
        _version, uid, _group_count = struct.unpack('=IIh', raw[:10])
    except (AttributeError, OSError, struct.error) as exc:
        raise SessionError('peer_credential_unavailable', 'macOS peer credentials are unavailable.') from exc
    return int(uid)


def _current_peer_pid(connection: socket.socket) -> int:
    require_macos()
    try:
        raw = connection.getsockopt(0, 2, 4)
        pid = struct.unpack('=I', raw)[0]
    except (AttributeError, OSError, struct.error) as exc:
        raise SessionError(
            'peer_credential_unavailable',
            'macOS peer process identity is unavailable.',
        ) from exc
    if pid <= 1:
        raise SessionError(
            'peer_credential_unavailable',
            'macOS peer process identity is invalid.',
        )
    return int(pid)


def _recv_frame(connection: socket.socket) -> dict[str, Any] | None:
    header = bytearray()
    while len(header) < 4:
        chunk = connection.recv(4 - len(header))
        if not chunk:
            if not header:
                return None
            raise ProtocolError('partial_frame', 'stream ended during the frame header')
        header.extend(chunk)
    expected = struct.unpack('>I', header)[0]
    if expected == 0:
        raise ProtocolError('invalid_frame', 'empty frames are not allowed')
    if expected > MAX_FRAME_BYTES:
        raise ProtocolError(
            'frame_too_large', 'declared frame exceeds the protocol hard cap',
            details={'declared_bytes': expected, 'max_bytes': MAX_FRAME_BYTES},
        )
    body = bytearray()
    while len(body) < expected:
        chunk = connection.recv(expected - len(body))
        if not chunk:
            raise ProtocolError('partial_frame', 'stream ended during the frame body')
        body.extend(chunk)
    try:
        payload = json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError('invalid_frame', 'frame must contain one UTF-8 JSON object') from exc
    if not isinstance(payload, dict):
        raise ProtocolError('invalid_frame', 'frame JSON must be an object')
    return payload


class _BoundedExecutor:
    def __init__(self, *, max_workers: int, capacity: int):
        if max_workers < 1 or capacity < 1:
            raise ValueError('daemon worker bounds must be positive')
        self._capacity = threading.BoundedSemaphore(capacity)
        self._queue: queue.Queue[tuple[concurrent.futures.Future, Callable[[], Mapping[str, Any]]] | None] = queue.Queue()
        self._active = 0
        self._condition = threading.Condition()
        self._shutdown = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f'troved-request-{index}',
                daemon=True,
            )
            for index in range(max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, function: Callable[[], Mapping[str, Any]]):
        if not self._capacity.acquire(blocking=False):
            return None
        with self._condition:
            if self._shutdown:
                self._capacity.release()
                return None
            self._active += 1
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((future, function))
        return future

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, function = item
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(function())
                    except BaseException as exc:
                        future.set_exception(exc)
            finally:
                self._release()

    def _release(self) -> None:
        self._capacity.release()
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def drain(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._active and time.monotonic() < deadline:
                self._condition.wait(max(0.0, deadline - time.monotonic()))
            return self._active == 0

    def shutdown(self) -> None:
        with self._condition:
            if self._shutdown:
                return
            self._shutdown = True
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                future, _function = item
                future.cancel()
                self._release()
        for _thread in self._threads:
            self._queue.put(None)


class DaemonServer:
    """Bounded macOS Unix-domain server for the application dispatcher."""

    def __init__(
        self,
        identity: RuntimeIdentity,
        dispatcher: object,
        *,
        max_workers: int = 8,
        max_pending: int = 32,
        max_connections: int = 64,
        idle_timeout: float | None = DEFAULT_IDLE_TIMEOUT_SECONDS,
        managed_nonce: str | None = None,
        peer_uid: Callable[[socket.socket], int] = _current_peer_uid,
        keepalive: Callable[[], bool] | None = None,
        peer_pid: Callable[[socket.socket], int] = _current_peer_pid,
        operator_authorizer: Callable[[int], bool] | None = None,
        operator_control: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ):
        require_macos()
        if max_connections < 1:
            raise ValueError('max_connections must be positive')
        self.identity = identity
        self.dispatcher = dispatcher
        self.idle_timeout = idle_timeout
        self.managed_nonce = managed_nonce
        self.keepalive = keepalive
        self.session_contract = SessionContract(identity)
        self.restart_id = os.urandom(16).hex()
        self.cursors = DaemonCursorStore(
            vault_identity=identity.vault_identity, restart_id=self.restart_id,
        )
        self._peer_uid = peer_uid
        self._peer_pid = peer_pid
        self.operator_authorizer = operator_authorizer
        self.operator_control = operator_control
        self._requests = _BoundedExecutor(max_workers=max_workers, capacity=max_pending)
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._state_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._session_threads: set[threading.Thread] = set()
        self._persistent_mcp = 0
        self._last_activity = time.monotonic()

    @property
    def address_family(self) -> int:
        return socket.AF_UNIX

    def start(self) -> None:
        with self._state_lock:
            if self._started.is_set():
                return
            self.identity.prepare()
            self.identity.remove_stale_socket()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(self.identity.socket_path))
                os.chmod(self.identity.socket_path, 0o600)
                listener.listen(64)
                listener.settimeout(0.2)
            except BaseException:
                listener.close()
                self.identity.socket_path.unlink(missing_ok=True)
                raise
            self._listener = listener
            self.identity.write_metadata(pid=os.getpid(), restart_id=self.restart_id)
            self._started.set()
            self._accept_thread = threading.Thread(
                target=self._accept_loop, name='troved-accept', daemon=True,
            )
            self._accept_thread.start()

    def serve_forever(self) -> None:
        self.start()
        self._stop.wait()
        self.stop()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                break
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                if self._should_idle_exit():
                    self._stop.set()
                continue
            except OSError:
                break
            connection.settimeout(30.0)
            if not self._connection_slots.acquire(blocking=False):
                try:
                    connection.sendall(encode_frame(_typed_error(
                        'busy', 'Daemon connection queue is saturated.', retryable=True,
                    )))
                except OSError:
                    pass
                connection.close()
                continue
            with self._state_lock:
                self._connections.add(connection)
                self._last_activity = time.monotonic()
            thread = threading.Thread(
                target=self._serve_connection,
                args=(connection,),
                name='troved-session',
                daemon=True,
            )
            with self._state_lock:
                self._session_threads.add(thread)
            thread.start()

    def _should_idle_exit(self) -> bool:
        if self.idle_timeout is None:
            return False
        if self.keepalive is not None:
            try:
                if self.keepalive():
                    return False
            except Exception:
                pass
        with self._state_lock:
            return (
                self._persistent_mcp == 0
                and not self._connections
                and time.monotonic() - self._last_activity >= self.idle_timeout
            )

    def _serve_connection(self, connection: socket.socket) -> None:
        role = None
        try:
            try:
                peer_uid = self._peer_uid(connection)
            except SessionError as exc:
                connection.sendall(encode_frame(exc.to_response()))
                return
            try:
                first = _recv_frame(connection)
            except ProtocolError as exc:
                connection.sendall(encode_frame(_typed_error(
                    exc.code, str(exc), retryable=exc.retryable, details=exc.details,
                )))
                return
            if first is None:
                return
            if first.get('type') == 'health':
                self._serve_health(connection, first, peer_uid=peer_uid)
                return
            try:
                hello = self.session_contract.accept_hello(first, peer_uid=peer_uid)
            except SessionError as exc:
                connection.sendall(encode_frame(exc.to_response()))
                return
            role = str(first['role'])
            if role == 'operator':
                try:
                    peer_pid = self._peer_pid(connection)
                except SessionError as exc:
                    connection.sendall(encode_frame(exc.to_response()))
                    return
                authorized = False
                if self.operator_authorizer is not None:
                    try:
                        authorized = self.operator_authorizer(peer_pid)
                    except Exception:
                        authorized = False
                if not authorized or self.operator_control is None:
                    connection.sendall(encode_frame(_typed_error(
                        'operator_unauthorized',
                        'Signed operator application is not trusted.',
                    )))
                    return
            if role == 'mcp' and first['persistent']:
                with self._state_lock:
                    self._persistent_mcp += 1
            connection.sendall(encode_frame(hello))
            while not self._stop.is_set():
                try:
                    payload = _recv_frame(connection)
                except socket.timeout:
                    continue
                except ProtocolError as exc:
                    connection.sendall(encode_frame(_typed_error(
                        exc.code, str(exc), retryable=exc.retryable, details=exc.details,
                    )))
                    return
                if payload is None:
                    break
                with self._state_lock:
                    self._last_activity = time.monotonic()
                if role == 'operator':
                    self._serve_operator_request(connection, payload)
                else:
                    self._serve_request(connection, payload)
        except (ConnectionError, OSError, ProtocolError, SessionError):
            return
        finally:
            if role == 'mcp':
                with self._state_lock:
                    self._persistent_mcp = max(0, self._persistent_mcp - 1)
            with self._state_lock:
                self._connections.discard(connection)
                self._session_threads.discard(threading.current_thread())
                self._last_activity = time.monotonic()
            connection.close()
            self._connection_slots.release()

    def _serve_health(self, connection: socket.socket, payload: Mapping[str, Any], *, peer_uid: int) -> None:
        expected = {'type', 'managed_nonce'}
        ok = (
            peer_uid == os.getuid()
            and set(payload) == expected
            and isinstance(self.managed_nonce, str)
            and payload.get('managed_nonce') == self.managed_nonce
        )
        connection.sendall(encode_frame({
            'ok': ok,
            'managed_nonce': self.managed_nonce if ok else None,
            'transport': 'unix',
        }))

    def _serve_request(self, connection: socket.socket, payload: Mapping[str, Any]) -> None:
        request_id = payload.get('request_id') if isinstance(payload.get('request_id'), str) else 'unknown'
        try:
            frame = decode_request(payload)
            if frame.vault_identity != self.identity.vault_identity:
                raise ProtocolError('vault_identity_mismatch', 'Request Vault identity does not match the session.')
            if frame.build_hash != self.identity.build_hash:
                raise ProtocolError('version_incompatible', 'Request build hash does not match the daemon.')
            if frame.catalog_hash != self.identity.catalog_hash:
                raise ProtocolError('catalog_mismatch', 'Request catalog hash does not match the daemon.')
        except ProtocolError as exc:
            connection.sendall(encode_frame(_typed_error(
                exc.code, str(exc), request_id=request_id,
                retryable=exc.retryable, details=exc.details,
            )))
            return
        future = self._requests.submit(lambda: self._dispatch(frame))
        if future is None:
            connection.sendall(encode_frame(_typed_error(
                'busy', 'Daemon request queue is saturated.',
                request_id=frame.request_id, retryable=True,
            )))
            return
        remaining = max(0.001, (frame.deadline_ms - int(time.time() * 1000)) / 1000)
        try:
            response = future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            response = _typed_error(
                'timeout', 'Request exceeded its deadline.',
                request_id=frame.request_id, retryable=True,
            )
        except Exception:
            response = _typed_error(
                'capability_unavailable', 'Request failed inside the daemon.',
                request_id=frame.request_id,
            )
        connection.sendall(encode_frame(response))

    def _serve_operator_request(
        self,
        connection: socket.socket,
        payload: Mapping[str, Any],
    ) -> None:
        request_id = (
            payload.get('request_id')
            if isinstance(payload.get('request_id'), str)
            else 'unknown'
        )
        common = {
            'type', 'request_id', 'action', 'vault_identity',
            'build_hash', 'catalog_hash',
        }
        action = payload.get('action')
        expected = (
            common | {'review_id'}
            if action in {'reply.approve', 'reply.reject', 'reply.retry'}
            else (
                common | {'mode'}
                if action == 'reply.set_mode'
                else common
            )
        )
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected
            or payload.get('type') != 'operator_request'
            or not isinstance(payload.get('request_id'), str)
            or not request_id
            or action not in {
                'reply.arm', 'reply.disarm',
                'reply.set_mode', 'reply.approve', 'reply.reject',
                'reply.retry',
            }
            or payload.get('vault_identity') != self.identity.vault_identity
            or payload.get('build_hash') != self.identity.build_hash
            or payload.get('catalog_hash') != self.identity.catalog_hash
            or (
                'review_id' in expected
                and (
                    not isinstance(payload.get('review_id'), str)
                    or not payload.get('review_id')
                )
            )
            or (
                'mode' in expected
                and payload.get('mode') not in {
                    'shadow', 'review_queue', 'live',
                }
            )
        ):
            connection.sendall(encode_frame(_typed_error(
                'invalid_request',
                'Operator request does not match the exact contract.',
                request_id=request_id,
            )))
            return
        try:
            assert self.operator_control is not None
            result = self.operator_control(
                str(action),
                {
                    key: value
                    for key, value in payload.items()
                    if key not in common
                },
            )
            response = {
                'protocol': 'trove/1',
                'request_id': request_id,
                'ok': True,
                'data': {'reply_control': dict(result)},
            }
        except Exception as exc:
            response = _typed_error(
                str(getattr(exc, 'code', 'operator_control_failed')),
                'Operator action could not be completed.',
                request_id=request_id,
            )
        connection.sendall(encode_frame(response))

    def _dispatch(self, frame) -> Mapping[str, Any]:
        spec = CATALOG_BY_ID[frame.capability]
        lease = vault_generation_read(self.identity.vault_root) if spec.replay_policy == 'read' else nullcontext()
        with lease:
            result = self.dispatcher.dispatch(
                frame.capability,
                frame.input,
                request_id=frame.request_id,
                response_budget=frame.response_budget,
            )
        if not isinstance(result, Mapping):
            raise TypeError('dispatcher result must be an object')
        return dict(result)

    def stop(self, timeout: float = 5.0) -> bool:
        with self._state_lock:
            if not self._started.is_set():
                return True
            self._stop.set()
            listener = self._listener
            self._listener = None
        if listener is not None:
            listener.close()
        thread = self._accept_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(1.0, max(0.0, timeout)))
        drained = self._requests.drain(timeout=max(0.0, timeout - 1.0))
        with self._state_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self._requests.shutdown()
        self.identity.socket_path.unlink(missing_ok=True)
        self.identity.remove_metadata()
        self._started.clear()
        return drained


__all__ = ['DEFAULT_IDLE_TIMEOUT_SECONDS', 'DaemonServer']
