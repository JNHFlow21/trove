from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trove_core.bounds import BoundedLimit, TRACE_EVENTS_APPROVALS
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import uuid


_TRACE_FILE = 'trace-timeline.redacted.jsonl'
_TRACE_LOCK_FILE = '.trace-timeline.lock'
_MAX_TRACE_BYTES = 5 * 1024 * 1024
_TRACE_FILE_COUNT = 5
_MAX_TRACE_EVENT_BYTES = 64 * 1024
_MAX_TRACE_DEPTH = 16
_MAX_TRACE_NODES = 2000
_SAFE_TRACE_FIELDS = {
    'action', 'approval_id', 'backend', 'batch_size', 'candidate_count',
    'changed', 'commit_count', 'consumption_id', 'count', 'danger_class',
    'dirty_count', 'duration_ms', 'elapsed_ms', 'error_code', 'generation',
    'indexed', 'lock_conflicts', 'max_messages', 'mode', 'ok', 'provider',
    'purge', 'queries', 'reason_code', 'request_hash', 'resource_count',
    'scan_count', 'scope', 'sql_count', 'status', 'wal_bytes',
    'active_workers', 'cache_hit', 'cancel_requested', 'job_id', 'max_queue',
    'max_workers', 'phase_timings', 'queued_workers', 'request_id', 'route',
    'resource_counts',
}
_SAFE_TRACE_NUMERIC_FIELDS = {
    'batch_size', 'candidate_count', 'commit_count', 'count', 'dirty_count',
    'duration_ms', 'elapsed_ms', 'generation', 'indexed', 'lock_conflicts',
    'max_messages', 'queries', 'resource_count', 'scan_count', 'sql_count',
    'wal_bytes',
    'active_workers', 'max_queue', 'max_workers', 'queued_workers',
}
_SAFE_TRACE_BOOL_FIELDS = {'cache_hit', 'cancel_requested', 'changed', 'ok', 'purge'}
_SAFE_TRACE_STAGES = {
    'approval', 'chunking', 'complete', 'dirty_citations', 'evaluation', 'fail',
    'import', 'import_source', 'maintain', 'sync', 'vector_index',
}
_SAFE_TRACE_STATUSES = {
    'approved', 'cancelled', 'complete', 'completed', 'consumed', 'expired',
    'fail', 'failed', 'locked', 'ok', 'pending', 'progress', 'rejected',
    'running', 'start', 'succeeded',
}


class TraceStorageError(RuntimeError):
    def __init__(self, message: str = 'trace storage is unavailable', *, code: str = 'trace_storage_unavailable'):
        super().__init__(message)
        self.code = code


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _redacted_text(value: str) -> str:
    if '/Users/' in value or '/Volumes/' in value or '\\Users\\' in value:
        prefix = 'redacted-path'
    else:
        prefix = 'redacted-text'
    return f'{prefix}:{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _safe_trace_text(key: str, value: str) -> bool:
    if key == 'request_hash':
        return bool(re.fullmatch(r'[0-9a-f]{64}', value))
    if key == 'approval_id':
        return bool(re.fullmatch(r'appr-[0-9a-f]{16}', value))
    if key == 'consumption_id':
        return bool(re.fullmatch(r'consume-[0-9a-f]{16}', value))
    if key == 'request_id':
        return bool(re.fullmatch(r'req-[0-9a-f]{16}', value))
    if key == 'job_id':
        return bool(re.fullmatch(r'job-[0-9a-f]{16}', value))
    if key == 'status':
        return value in _SAFE_TRACE_STATUSES
    if key in {'action', 'danger_class'}:
        return bool(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', value))
    if key == 'backend':
        return value in {'sqlite', 'zvec', 'none'}
    return False


def redact_value(
    value: Any,
    *,
    _key: str = '',
    _depth: int = 0,
    _budget: list[int] | None = None,
) -> Any:
    """Retain counts/booleans while reducing every free-form string to a hash."""

    budget = _budget if _budget is not None else [_MAX_TRACE_NODES]
    budget[0] -= 1
    if budget[0] < 0 or _depth > _MAX_TRACE_DEPTH:
        return 'redacted-structure'
    if isinstance(value, Path):
        return _redacted_text(str(value))
    if type(value) is str:
        return value if _safe_trace_text(_key, value) else _redacted_text(value)
    if type(value) is dict:
        redacted: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:200]:
            key = str(raw_key)
            persisted_key = key if key in _SAFE_TRACE_FIELDS else f'field_{hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]}'
            redacted[persisted_key] = redact_value(
                item,
                _key=key,
                _depth=_depth + 1,
                _budget=budget,
            )
        return redacted
    if type(value) in {list, tuple}:
        return [
            redact_value(item, _key=_key, _depth=_depth + 1, _budget=budget)
            for item in value[:50]
        ]
    if value is None:
        return None
    if type(value) is bool and _key in _SAFE_TRACE_BOOL_FIELDS:
        return value
    if type(value) is int and _key in _SAFE_TRACE_NUMERIC_FIELDS:
        if 0 <= value <= 1_000_000_000_000_000:
            return value
        raise ValueError('trace numeric field is out of range')
    if type(value) is float and _key in _SAFE_TRACE_NUMERIC_FIELDS:
        if math.isfinite(value) and 0 <= value <= 1_000_000_000_000_000:
            return value
        raise ValueError('trace numeric field is out of range')
    if type(value) in {bool, int, float}:
        return 'redacted-number'
    return _redacted_text(type(value).__name__)


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    stage: str
    status: str
    created_at: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceTimeline:
    def __init__(self, vault_root: str | Path):
        self.vault_root = Path(vault_root).expanduser().resolve()
        self.path = self.vault_root / 'logs' / _TRACE_FILE

    @staticmethod
    def _identity(result: os.stat_result) -> tuple[int, int]:
        return int(result.st_dev), int(result.st_ino)

    def _open_logs_dir(self, *, create: bool) -> tuple[int, tuple[int, int]] | None:
        logs_dir = self.path.parent
        if create:
            try:
                logs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError as exc:
                raise TraceStorageError() from exc
        try:
            listed = os.lstat(logs_dir)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TraceStorageError() from exc
        if stat.S_ISLNK(listed.st_mode) or not stat.S_ISDIR(listed.st_mode):
            raise TraceStorageError('trace storage is not a private directory')
        flags = os.O_RDONLY
        if hasattr(os, 'O_DIRECTORY'):
            flags |= os.O_DIRECTORY
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        dir_fd: int | None = None
        try:
            dir_fd = os.open(logs_dir, flags)
            bound = os.fstat(dir_fd)
            current = os.lstat(logs_dir)
            identity = self._identity(bound)
            if (
                not stat.S_ISDIR(bound.st_mode)
                or identity != self._identity(listed)
                or identity != self._identity(current)
            ):
                raise TraceStorageError('trace directory identity changed')
            os.fchmod(dir_fd, 0o700)
            return dir_fd, identity
        except OSError as exc:
            if dir_fd is not None:
                os.close(dir_fd)
            raise TraceStorageError() from exc
        except Exception:
            if dir_fd is not None:
                os.close(dir_fd)
            raise

    def _validate_dir(self, dir_fd: int, identity: tuple[int, int]) -> None:
        try:
            bound = os.fstat(dir_fd)
            current = os.lstat(self.path.parent)
        except OSError as exc:
            raise TraceStorageError('trace directory identity changed') from exc
        if self._identity(bound) != identity or self._identity(current) != identity:
            raise TraceStorageError('trace directory identity changed')

    @staticmethod
    def _trace_name(index: int) -> str:
        return _TRACE_FILE if index == 0 else f'{_TRACE_FILE}.{index}'

    def _open_lock(self, dir_fd: int) -> int:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        fd = os.open(_TRACE_LOCK_FILE, flags, 0o600, dir_fd=dir_fd)
        opened = os.fstat(fd)
        listed = os.stat(_TRACE_LOCK_FILE, dir_fd=dir_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_mode & 0o077
            or self._identity(opened) != self._identity(listed)
        ):
            os.close(fd)
            raise TraceStorageError('trace lock is not a private regular file')
        return fd

    def _open_trace(self, dir_fd: int, name: str, *, create: bool, write: bool) -> int:
        flags = (os.O_WRONLY | os.O_APPEND) if write else os.O_RDONLY
        if create:
            flags |= os.O_CREAT
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
        opened = os.fstat(fd)
        listed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_mode & 0o077
            or self._identity(opened) != self._identity(listed)
        ):
            os.close(fd)
            raise TraceStorageError('trace storage is not a bounded private file')
        return fd

    def _validate_rotated_name(self, dir_fd: int, name: str) -> bool:
        try:
            listed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(listed.st_mode) or listed.st_nlink != 1 or listed.st_mode & 0o077:
            raise TraceStorageError('rotated trace storage is not a private regular file')
        return True

    def _rotate_locked(self, dir_fd: int) -> None:
        oldest = self._trace_name(_TRACE_FILE_COUNT - 1)
        if self._validate_rotated_name(dir_fd, oldest):
            os.unlink(oldest, dir_fd=dir_fd)
        for index in range(_TRACE_FILE_COUNT - 2, -1, -1):
            source = self._trace_name(index)
            if not self._validate_rotated_name(dir_fd, source):
                continue
            destination = self._trace_name(index + 1)
            if self._validate_rotated_name(dir_fd, destination):
                os.unlink(destination, dir_fd=dir_fd)
            os.replace(source, destination, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)

    @staticmethod
    def _safe_marker(value: str, *, kind: str, allowed: set[str]) -> str:
        if value in allowed:
            return value
        return f'redacted-{kind}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'

    def append(self, stage: str, status: str, payload: dict[str, Any] | None = None, *, trace_id: str | None = None) -> str:
        if (
            type(stage) is not str
            or not re.fullmatch(r'[a-z0-9_.:-]{1,64}', stage)
            or type(status) is not str
            or not re.fullmatch(r'[a-z0-9_.:-]{1,64}', status)
            or (payload is not None and type(payload) is not dict)
            or (trace_id is not None and (type(trace_id) is not str or not re.fullmatch(r'trace-[0-9a-f]{16}', trace_id)))
        ):
            raise TraceStorageError('trace event is malformed', code='invalid_trace_event')
        opened_dir = self._open_logs_dir(create=True)
        if opened_dir is None:  # pragma: no cover - create=True makes this impossible
            raise TraceStorageError()
        dir_fd, directory_identity = opened_dir
        tid = trace_id or ('trace-' + uuid.uuid4().hex[:16])
        try:
            event = TraceEvent(
                tid,
                self._safe_marker(stage, kind='stage', allowed=_SAFE_TRACE_STAGES),
                self._safe_marker(status, kind='status', allowed=_SAFE_TRACE_STATUSES),
                now_iso(),
                redact_value(payload or {}),
            )
            encoded = (
                json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False) + '\n'
            ).encode('utf-8')
        except (TypeError, ValueError, RecursionError) as exc:
            raise TraceStorageError('trace event is malformed', code='invalid_trace_event') from exc
        if len(encoded) > _MAX_TRACE_EVENT_BYTES:
            raise TraceStorageError('trace event exceeds size bound', code='invalid_trace_event')
        fd: int | None = None
        lock_fd: int | None = None
        try:
            self._validate_dir(dir_fd, directory_identity)
            lock_fd = self._open_lock(dir_fd)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            fd = self._open_trace(dir_fd, _TRACE_FILE, create=True, write=True)
            opened = os.fstat(fd)
            if opened.st_size > 0 and opened.st_size + len(encoded) > _MAX_TRACE_BYTES:
                os.close(fd)
                fd = None
                self._rotate_locked(dir_fd)
                fd = self._open_trace(dir_fd, _TRACE_FILE, create=True, write=True)
                opened = os.fstat(fd)
            file_identity = self._identity(opened)
            self._validate_dir(dir_fd, directory_identity)
            current = os.stat(_TRACE_FILE, dir_fd=dir_fd, follow_symlinks=False)
            if self._identity(current) != file_identity:
                raise TraceStorageError('trace file identity changed')
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise TraceStorageError('trace append failed')
                view = view[written:]
            os.fsync(fd)
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            self._validate_dir(dir_fd, directory_identity)
            current = os.stat(_TRACE_FILE, dir_fd=dir_fd, follow_symlinks=False)
            if self._identity(current) != file_identity:
                raise TraceStorageError('trace file identity changed')
        except OSError as exc:
            raise TraceStorageError() from exc
        finally:
            if fd is not None:
                os.close(fd)
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(dir_fd)
        return tid

    def start(self, stage: str, payload: dict[str, Any] | None = None) -> str:
        return self.append(stage, 'start', payload)

    def progress(self, trace_id: str, stage: str, payload: dict[str, Any] | None = None) -> None:
        self.append(stage, 'progress', payload, trace_id=trace_id)

    def complete(self, trace_id: str, payload: dict[str, Any] | None = None) -> None:
        self.append('complete', 'complete', payload, trace_id=trace_id)

    def fail(self, trace_id: str, payload: dict[str, Any] | None = None) -> None:
        self.append('fail', 'fail', payload, trace_id=trace_id)

    @staticmethod
    def _tail_lines(fd: int, limit: int) -> list[bytes]:
        size = int(os.fstat(fd).st_size)
        if size <= 0 or limit <= 0:
            return []
        cursor = size
        chunks: list[bytes] = []
        newline_count = 0
        # A valid event is at most 64 KiB.  For ordinary short JSONL rows this
        # reads only a few 8 KiB blocks, proportional to the requested limit.
        byte_budget = min(size, max(8192, limit * _MAX_TRACE_EVENT_BYTES))
        read_bytes = 0
        while cursor > 0 and read_bytes < byte_budget and newline_count <= limit:
            amount = min(8192, cursor, byte_budget - read_bytes)
            cursor -= amount
            chunk = os.pread(fd, amount, cursor)
            if not chunk:
                break
            chunks.append(chunk)
            newline_count += chunk.count(b'\n')
            read_bytes += len(chunk)
        raw = b''.join(reversed(chunks))
        if cursor > 0:
            _discarded, separator, remainder = raw.partition(b'\n')
            raw = remainder if separator else b''
        return raw.splitlines()[-limit:]

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = BoundedLimit(limit, field='limit', spec=TRACE_EVENTS_APPROVALS)
        opened_dir = self._open_logs_dir(create=False)
        if opened_dir is None:
            return []
        dir_fd, directory_identity = opened_dir
        lock_fd: int | None = None
        lines: list[bytes] = []
        try:
            if not any(
                self._validate_rotated_name(dir_fd, self._trace_name(index))
                for index in range(_TRACE_FILE_COUNT)
            ):
                return []
            lock_fd = self._open_lock(dir_fd)
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            self._validate_dir(dir_fd, directory_identity)
            # Newest file first; prepend older rows so output remains
            # chronological while spanning rotation boundaries.
            for index in range(_TRACE_FILE_COUNT):
                remaining = int(limit) - len(lines)
                if remaining <= 0:
                    break
                name = self._trace_name(index)
                try:
                    fd = self._open_trace(dir_fd, name, create=False, write=False)
                except FileNotFoundError:
                    continue
                try:
                    rows = self._tail_lines(fd, remaining)
                finally:
                    os.close(fd)
                lines = rows + lines
        except (OSError, UnicodeError) as exc:
            raise TraceStorageError() from exc
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(dir_fd)
        out: list[dict[str, Any]] = []
        for line in lines:
            try:
                item = json.loads(line.decode('utf-8'))
            except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
                continue
            if type(item) is dict:
                out.append(item)
        return out
