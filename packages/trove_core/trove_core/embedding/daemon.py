from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from trove_core.embedding.daemon_client import EmbeddingDaemonClient
from trove_core.embedding.daemon_protocol import (
    MAX_DIMENSIONS,
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    DaemonIdentity,
    DaemonIdentityMismatch,
    DaemonProtocolError,
    DaemonQueueSaturated,
    DaemonRequestTimeout,
    error_payload,
    identity_for_model,
    validate_texts,
)
from trove_core.embedding.local_provider import DEFAULT_EMBED_SOCKET, LocalEmbeddingProvider
from trove_core.embedding.model_registry import default_local_model_path

PID_PATH = Path('/tmp/trove-embed.sock.pid')


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _pid_path(socket_path: str) -> Path:
    return PID_PATH if socket_path == DEFAULT_EMBED_SOCKET else Path(f'{socket_path}.pid')


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _socket_status(socket_path: str) -> dict[str, Any]:
    return {
        'socket': Path(DEFAULT_EMBED_SOCKET).name if socket_path == DEFAULT_EMBED_SOCKET else 'custom-socket',
        'socket_id': __import__('hashlib').sha256(socket_path.encode('utf-8')).hexdigest()[:16],
        'socket_exists': Path(socket_path).exists(),
        'raw_paths_included': False,
        'secret_values_included': False,
    }


def _read_pid(socket_path: str) -> int | None:
    try:
        return int(_pid_path(socket_path).read_text(encoding='utf-8').strip())
    except Exception:
        return None


def status_payload(
    socket_path: str = DEFAULT_EMBED_SOCKET,
    *,
    probe: bool = True,
    timeout: float = 0.2,
    expected_identity: DaemonIdentity | None = None,
) -> dict[str, Any]:
    pid = _read_pid(socket_path)
    running = bool(pid and _pid_running(pid))
    responsive = False
    identity: dict[str, Any] | None = None
    telemetry: dict[str, Any] = {}
    reason_code: str | None = None
    if Path(socket_path).exists() and probe:
        try:
            client = EmbeddingDaemonClient(
                socket_path,
                identity=expected_identity,
                timeout_ms=max(1, int(timeout * 1000)),
            )
            response = client.handshake()
            identity = DaemonIdentity.from_dict(response.get('identity')).to_dict()
            telemetry = response.get('telemetry') if type(response.get('telemetry')) is dict else {}
            responsive = True
        except DaemonProtocolError as exc:
            reason_code = exc.code
    return {
        **_socket_status(socket_path),
        'pid': pid,
        'running': running,
        'responsive': responsive,
        'identity': identity,
        'telemetry': telemetry,
        'reason_code': reason_code,
        'protocol_version': PROTOCOL_VERSION,
        'raw_content_included': False,
        'secret_values_included': False,
    }


def warmup_daemon(
    socket_path: str = DEFAULT_EMBED_SOCKET,
    *,
    timeout: float = 120.0,
    expected_identity: DaemonIdentity | None = None,
) -> dict[str, Any]:
    """Load the daemon with static non-private text and verify its identity."""

    start = time.perf_counter()
    try:
        if expected_identity is None:
            handshake = EmbeddingDaemonClient(
                socket_path,
                timeout_ms=max(1, int(timeout * 1000)),
            ).handshake()
            expected_identity = DaemonIdentity.from_dict(handshake.get('identity'))
        vectors, telemetry = EmbeddingDaemonClient(
            socket_path,
            identity=expected_identity,
            timeout_ms=max(1, int(timeout * 1000)),
        ).embed(['trove daemon warmup'])
        dimensions = len(vectors[0]) if vectors else 0
        return {
            'ok': bool(vectors and dimensions),
            'identity': replace(expected_identity, dimensions=dimensions or expected_identity.dimensions).to_dict(),
            'dimensions': dimensions,
            'telemetry': telemetry,
            'elapsed_ms': round((time.perf_counter() - start) * 1000, 3),
            'private_text_used': False,
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    except DaemonProtocolError as exc:
        return {
            'ok': False,
            'reason_code': exc.code,
            'elapsed_ms': round((time.perf_counter() - start) * 1000, 3),
            'private_text_used': False,
            'raw_content_included': False,
            'raw_paths_included': False,
        }


_SECRET_ENV_RE = re.compile(r'(?i)(secret|token|api[_-]?key|access[_-]?key|password|credential|authorization)')


def _daemon_environment() -> dict[str, str]:
    """Copy runtime settings while dropping credentials the local model cannot use."""

    from trove_core.providers.config import ProviderConfig

    source = os.environ
    config = ProviderConfig.resolve(dict(source), check_agent_switch=False)
    configured_secret_names = {
        config.asr_secret_name,
        config.vision_secret_name,
        config.cloud_embedding_secret_name,
        config.cloud_rerank_secret_name,
    }
    return {
        key: value
        for key, value in source.items()
        if key not in configured_secret_names and not _SECRET_ENV_RE.search(key)
    }


def start_daemon(model_path: str | None = None, socket_path: str = DEFAULT_EMBED_SOCKET) -> dict[str, Any]:
    existing = status_payload(socket_path)
    if existing.get('responsive') and model_path is None:
        warmup = warmup_daemon(socket_path)
        return {'ok': bool(warmup.get('ok')), 'already_running': True, 'warmup': warmup, **existing}

    path = Path(model_path).expanduser() if model_path else default_local_model_path()
    if path is None or not path.exists():
        raise RuntimeError('No local embedding model is available for the daemon.')
    expected_identity = identity_for_model(path)
    if existing.get('responsive'):
        matched = status_payload(socket_path, expected_identity=expected_identity)
        if not matched.get('responsive'):
            return {
                **matched,
                'ok': False,
                'already_running': True,
                'reason_code': matched.get('reason_code') or 'daemon_identity_mismatch',
            }
        warmup = warmup_daemon(socket_path, expected_identity=expected_identity)
        return {'ok': bool(warmup.get('ok')), 'already_running': True, 'warmup': warmup, **matched}
    if existing.get('running'):
        return {'ok': False, 'started': False, 'reason_code': 'daemon_unresponsive_process_running', **existing}

    cmd = [sys.executable, '-m', 'trove_core.embedding.daemon', 'serve', '--model-path', str(path), '--socket', socket_path]
    env = _daemon_environment()
    repo = Path(__file__).resolve().parents[4]
    env['PYTHONPATH'] = os.pathsep.join([
        str(repo / 'packages' / 'trove_core'),
        str(repo / 'packages' / 'trove_cli'),
        env.get('PYTHONPATH', ''),
    ]).strip(os.pathsep)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    pid_path = _pid_path(socket_path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(proc.pid), encoding='utf-8')
    status: dict[str, Any] = {}
    for _ in range(900):
        time.sleep(0.1)
        status = status_payload(socket_path, expected_identity=expected_identity, timeout=0.5)
        if status.get('responsive') and status.get('running'):
            warmup = warmup_daemon(socket_path, expected_identity=expected_identity)
            return {'ok': bool(warmup.get('ok')), 'started': True, 'warmup': warmup, **status}
        if not _pid_running(proc.pid):
            break
    return {**status, 'ok': False, 'started': False, 'reason_code': status.get('reason_code') or 'daemon_start_failed'}


def stop_daemon(socket_path: str = DEFAULT_EMBED_SOCKET) -> dict[str, Any]:
    status = status_payload(socket_path)
    pid = status.get('pid')
    if pid and status.get('running'):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
    for _ in range(30):
        time.sleep(0.1)
        if not (pid and _pid_running(int(pid))):
            break
    _pid_path(socket_path).unlink(missing_ok=True)
    Path(socket_path).unlink(missing_ok=True)
    return {'ok': True, **status_payload(socket_path)}


@dataclass
class _EmbeddingJob:
    texts: list[str]
    expected_identity: DaemonIdentity
    event: threading.Event = field(default_factory=threading.Event)
    vectors: list[list[float]] | None = None
    error: DaemonProtocolError | None = None
    cancelled: bool = False


class EmbeddingDaemonRuntime:
    """Bounded batching runtime with singleflight model construction."""

    def __init__(
        self,
        provider_factory: Callable[[], Any],
        identity: DaemonIdentity,
        *,
        queue_size: int = 32,
        max_batch_requests: int = 8,
        max_batch_texts: int = 64,
        batch_wait_ms: int = 5,
    ) -> None:
        self._provider_factory = provider_factory
        self._identity = identity
        self._queue: queue.Queue[_EmbeddingJob | None] = queue.Queue(maxsize=max(1, queue_size))
        self._max_batch_requests = max(1, min(max_batch_requests, 64))
        self._max_batch_texts = max(1, min(max_batch_texts, 256))
        self._batch_wait_ms = max(0, min(batch_wait_ms, 100))
        self._provider: Any | None = None
        self._load_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._telemetry = {
            'load_count': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'timed_out_requests': 0,
            'saturated_requests': 0,
            'batched_requests': 0,
            'batches': 0,
        }
        self._closed = False
        self._state_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._deferred_job: _EmbeddingJob | None = None
        self._worker = threading.Thread(target=self._run, name='trove-embed-batcher', daemon=True)
        self._worker.start()

    @property
    def identity(self) -> DaemonIdentity:
        return self._identity

    def _record(self, key: str, amount: int = 1) -> None:
        with self._telemetry_lock:
            self._telemetry[key] += amount

    def telemetry(self) -> dict[str, Any]:
        with self._telemetry_lock:
            data = dict(self._telemetry)
        data['queue_depth'] = self._queue.qsize()
        data['queue_capacity'] = self._queue.maxsize
        data['raw_content_included'] = False
        data['raw_paths_included'] = False
        data['secret_values_included'] = False
        return data

    def _ensure_provider(self) -> Any:
        if self._provider is not None:
            return self._provider
        with self._load_lock:
            if self._provider is None:
                provider = self._provider_factory()
                dimensions = int(getattr(provider, 'dimensions', 0) or 0)
                if dimensions < 0 or dimensions > MAX_DIMENSIONS:
                    raise DaemonProtocolError('daemon_dimensions_invalid')
                if self._identity.dimensions and dimensions and dimensions != self._identity.dimensions:
                    raise DaemonIdentityMismatch()
                if dimensions:
                    self._identity = replace(self._identity, dimensions=dimensions)
                self._provider = provider
                self._record('load_count')
        return self._provider

    def submit(
        self,
        texts: list[str],
        *,
        expected_identity: DaemonIdentity,
        timeout_ms: int,
    ) -> list[list[float]]:
        texts = validate_texts(texts)
        if len(texts) > self._max_batch_texts:
            raise DaemonProtocolError('daemon_text_batch_too_large')
        expected_identity.require_match(self._identity)
        job = _EmbeddingJob(texts=texts, expected_identity=expected_identity)
        with self._state_lock:
            if self._closed:
                raise DaemonProtocolError('daemon_stopped')
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:
                self._record('saturated_requests')
                raise DaemonQueueSaturated() from exc
        if not job.event.wait(max(1, min(timeout_ms, 120_000)) / 1000.0):
            job.cancelled = True
            self._record('timed_out_requests')
            raise DaemonRequestTimeout()
        if job.error is not None:
            raise job.error
        if job.vectors is None:
            raise DaemonProtocolError('daemon_vector_missing')
        return job.vectors

    def _collect_batch(self, first: _EmbeddingJob) -> list[_EmbeddingJob]:
        batch = [first]
        text_count = len(first.texts)
        deadline = time.monotonic() + self._batch_wait_ms / 1000.0
        while len(batch) < self._max_batch_requests and text_count < self._max_batch_texts:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                candidate = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if candidate is None:
                self._queue.task_done()
                self._stop_requested.set()
                break
            if text_count + len(candidate.texts) > self._max_batch_texts:
                # Keep the hard batch bound. The job has already been removed
                # from the bounded queue, so retain exactly one deferred slot
                # for the next worker iteration.
                self._deferred_job = candidate
                break
            batch.append(candidate)
            text_count += len(candidate.texts)
        return batch

    def _complete_error(self, jobs: list[_EmbeddingJob], error: DaemonProtocolError) -> None:
        for job in jobs:
            job.error = error
            job.event.set()
            self._queue.task_done()
        self._record('failed_requests', len(jobs))

    def _run(self) -> None:
        while True:
            if self._stop_requested.is_set() and self._deferred_job is None and self._queue.empty():
                break
            if self._deferred_job is not None:
                first = self._deferred_job
                self._deferred_job = None
            else:
                try:
                    first = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            if first is None:
                self._queue.task_done()
                self._stop_requested.set()
                continue
            batch = self._collect_batch(first)
            active = [job for job in batch if not job.cancelled]
            if not active:
                for job in batch:
                    job.event.set()
                    self._queue.task_done()
                continue
            try:
                provider = self._ensure_provider()
                for job in active:
                    job.expected_identity.require_match(self._identity)
                flattened = [text for job in active for text in job.texts]
                vectors = provider.embed_many(flattened)
                if type(vectors) is not list or len(vectors) != len(flattened):
                    raise DaemonProtocolError('daemon_vector_count_mismatch')
                dimensions = int(getattr(provider, 'dimensions', 0) or (len(vectors[0]) if vectors else 0))
                if dimensions <= 0 or dimensions > MAX_DIMENSIONS:
                    raise DaemonProtocolError('daemon_dimensions_invalid')
                resolved_identity = replace(self._identity, dimensions=dimensions)
                self._identity.require_match(resolved_identity)
                for job in active:
                    job.expected_identity.require_match(resolved_identity)
                self._identity = resolved_identity
                offset = 0
                for job in active:
                    rows = vectors[offset:offset + len(job.texts)]
                    try:
                        parsed = [[float(value) for value in row] for row in rows]
                    except (TypeError, ValueError):
                        raise DaemonProtocolError('daemon_vector_invalid') from None
                    if any(len(row) != dimensions for row in parsed):
                        raise DaemonProtocolError('daemon_dimensions_mismatch')
                    if any(not math.isfinite(value) for row in parsed for value in row):
                        raise DaemonProtocolError('daemon_vector_invalid')
                    job.vectors = parsed
                    offset += len(job.texts)
            except DaemonProtocolError as exc:
                self._complete_error(batch, exc)
                continue
            except Exception:
                self._complete_error(batch, DaemonProtocolError('daemon_provider_failed'))
                continue
            for job in batch:
                job.event.set()
                self._queue.task_done()
            self._record('completed_requests', len(active))
            self._record('batches')
            if len(active) > 1:
                self._record('batched_requests', len(active))


    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_requested.set()
        self._worker.join(timeout=2)


def _read_request(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    while True:
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            raise DaemonRequestTimeout() from None
        except OSError:
            raise DaemonProtocolError('daemon_transport_error') from None
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise DaemonProtocolError('daemon_request_too_large')
        if b'\n' in chunk:
            break
    line = b''.join(chunks).split(b'\n', 1)[0]
    try:
        payload = json.loads(line.decode('utf-8'))
    except Exception as exc:
        raise DaemonProtocolError('daemon_protocol_error') from exc
    if type(payload) is not dict:
        raise DaemonProtocolError('daemon_protocol_error')
    return payload


def _response_for_request(runtime: EmbeddingDaemonRuntime, payload: dict[str, Any]) -> dict[str, Any]:
    protocol = payload.get('protocol_version')
    if type(protocol) is not int or protocol != PROTOCOL_VERSION:
        raise DaemonProtocolError('daemon_protocol_mismatch')
    operation = payload.get('op')
    expected: DaemonIdentity | None = None
    if payload.get('identity') is not None:
        expected = DaemonIdentity.from_dict(payload['identity'])
        expected.require_match(runtime.identity)
    if operation == 'handshake':
        return {
            'ok': True,
            'identity': runtime.identity.to_dict(),
            'telemetry': runtime.telemetry(),
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    if operation != 'embed' or expected is None:
        raise DaemonProtocolError('daemon_operation_invalid')
    texts = validate_texts(payload.get('texts'))
    timeout_ms = _bounded_int(payload.get('timeout_ms'), default=500, minimum=1, maximum=120_000)
    vectors = runtime.submit(texts, expected_identity=expected, timeout_ms=timeout_ms)
    return {
        'ok': True,
        'vectors': vectors,
        'identity': runtime.identity.to_dict(),
        'telemetry': runtime.telemetry(),
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def _handle_connection(conn: socket.socket, runtime: EmbeddingDaemonRuntime, semaphore: threading.BoundedSemaphore) -> None:
    try:
        with conn:
            try:
                payload = _read_request(conn)
                response = _response_for_request(runtime, payload)
            except DaemonProtocolError as exc:
                response = error_payload(exc.code)
            except Exception:
                response = error_payload('daemon_internal_error')
            try:
                conn.sendall(json.dumps(response, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
    finally:
        semaphore.release()


def serve(
    model_path: str,
    socket_path: str = DEFAULT_EMBED_SOCKET,
    *,
    stop_event: threading.Event | None = None,
    provider_factory: Callable[[], Any] | None = None,
    queue_size: int | None = None,
    max_batch_requests: int | None = None,
    max_batch_texts: int | None = None,
    batch_wait_ms: int | None = None,
) -> int:
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError('Local embedding model path does not exist.')
    identity = identity_for_model(path)
    provider_factory = provider_factory or (lambda: LocalEmbeddingProvider(
        path,
        dimensions=identity.dimensions,
        use_daemon=False,
    ))
    queue_size = _bounded_int(
        queue_size if queue_size is not None else os.environ.get('TROVE_EMBEDDING_DAEMON_QUEUE_SIZE'),
        default=32,
        minimum=1,
        maximum=1024,
    )
    runtime = EmbeddingDaemonRuntime(
        provider_factory,
        identity,
        queue_size=queue_size,
        max_batch_requests=_bounded_int(
            max_batch_requests if max_batch_requests is not None else os.environ.get('TROVE_EMBEDDING_DAEMON_BATCH_REQUESTS'),
            default=8,
            minimum=1,
            maximum=64,
        ),
        max_batch_texts=_bounded_int(
            max_batch_texts if max_batch_texts is not None else os.environ.get('TROVE_EMBEDDING_DAEMON_BATCH_TEXTS'),
            default=64,
            minimum=1,
            maximum=256,
        ),
        batch_wait_ms=_bounded_int(
            batch_wait_ms if batch_wait_ms is not None else os.environ.get('TROVE_EMBEDDING_DAEMON_BATCH_WAIT_MS'),
            default=5,
            minimum=0,
            maximum=100,
        ),
    )
    socket_path_obj = Path(socket_path)
    socket_path_obj.parent.mkdir(parents=True, exist_ok=True)
    socket_path_obj.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        old_umask = os.umask(0o177)
        try:
            server.bind(socket_path)
        finally:
            os.umask(old_umask)
        os.chmod(socket_path, 0o600)
        server.listen(min(queue_size + 4, 1024))
        server.settimeout(0.2)
        connection_limit = min(queue_size + 4, 1024)
        semaphore = threading.BoundedSemaphore(connection_limit)
        while stop_event is None or not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            if not semaphore.acquire(blocking=False):
                try:
                    conn.sendall(json.dumps(error_payload('daemon_queue_saturated')).encode('utf-8') + b'\n')
                except OSError:
                    pass
                conn.close()
                continue
            conn.settimeout(5.0)
            threading.Thread(
                target=_handle_connection,
                args=(conn, runtime, semaphore),
                name='trove-embed-client',
                daemon=True,
            ).start()
    finally:
        runtime.close()
        server.close()
        socket_path_obj.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['start', 'stop', 'status', 'serve']:
        command = sub.add_parser(name)
        command.add_argument('--socket', default=DEFAULT_EMBED_SOCKET)
        if name in {'start', 'serve'}:
            command.add_argument('--model-path')
        command.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'serve':
        model_path = args.model_path or str(default_local_model_path() or '')
        if not model_path:
            raise SystemExit('missing --model-path and no default local model')
        return serve(model_path, socket_path=args.socket)
    if args.command == 'start':
        data = start_daemon(args.model_path, socket_path=args.socket)
    elif args.command == 'stop':
        data = stop_daemon(socket_path=args.socket)
    else:
        data = status_payload(socket_path=args.socket)
        data['ok'] = True
    print(json.dumps(data, ensure_ascii=False, indent=2 if args.json else None, sort_keys=True))
    return 0 if data.get('ok', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
