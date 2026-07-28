from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import sys
import time
from typing import Callable, Iterator, Protocol
import uuid

MANIFEST_VERSION = 1
MAX_SCAN_ENTRIES_PER_TICK = 4096
MAX_DIRECTORY_WATCHES = 1024
MAX_EVENTS_PER_TICK = 256
MIN_IDLE_BACKOFF_SECONDS = 1.0
MAX_IDLE_BACKOFF_SECONDS = 300.0
SCAN_YIELD_SECONDS = 0.01
_DIGEST_MASK = (1 << 128) - 1


@dataclass(frozen=True)
class TreeSnapshot:
    root_present: bool
    digest: str
    entry_count: int
    max_mtime_ns: int

    @property
    def signature(self) -> tuple[bool, str, int, int]:
        return (self.root_present, self.digest, self.entry_count, self.max_mtime_ns)


@dataclass(frozen=True)
class WatchManifest:
    root_present: bool
    digest: str
    entry_count: int
    max_mtime_ns: int
    scan_generation: int
    completed_at: float

    @property
    def signature(self) -> tuple[bool, str, int, int]:
        return (self.root_present, self.digest, self.entry_count, self.max_mtime_ns)

    def to_dict(self) -> dict[str, object]:
        return {
            'version': MANIFEST_VERSION,
            'root_present': self.root_present,
            'digest': self.digest,
            'entry_count': self.entry_count,
            'max_mtime_ns': self.max_mtime_ns,
            'scan_generation': self.scan_generation,
            'completed_at': self.completed_at,
            'raw_paths_included': False,
            'raw_content_included': False,
        }

    @classmethod
    def from_dict(cls, payload: object) -> WatchManifest | None:
        if not isinstance(payload, dict) or payload.get('version') != MANIFEST_VERSION:
            return None
        try:
            digest = str(payload['digest'])
            if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
                return None
            return cls(
                root_present=bool(payload['root_present']),
                digest=digest,
                entry_count=max(0, int(payload['entry_count'])),
                max_mtime_ns=max(0, int(payload['max_mtime_ns'])),
                scan_generation=max(0, int(payload['scan_generation'])),
                completed_at=float(payload['completed_at']),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class WatchTick:
    backend: str
    changed: bool = False
    change_source: str | None = None
    scan_complete: bool = False
    scan_discarded: bool = False
    scan_active: bool = False
    entries_processed: int = 0
    repair_pending: bool = False
    event_loss: bool = False
    descriptor_overflow: bool = False
    error_code: str | None = None
    manifest_digest: str | None = None


class WatchBackend(Protocol):
    name: str

    def poll(self, timeout: float = 1.0) -> WatchTick: ...

    def request_repair(self, *, reason: str = 'manual') -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _ScanStep:
    entries_processed: int
    snapshot: TreeSnapshot | None = None
    error_code: str | None = None


class BoundedManifestScanner:
    """Streaming, order-independent tree scanner with bounded per-tick work."""

    def __init__(self, root: Path, *, on_directory: Callable[[Path], None] | None = None):
        self.root = Path(root)
        self.on_directory = on_directory
        self._stack: list[tuple[Iterator[os.DirEntry[str]], str]] = []
        self._root_pending = False
        self._root_follow_pending = False
        self._active = False
        self._xor_digest = 0
        self._sum_digest = 0
        self._entry_count = 0
        self._max_mtime_ns = 0
        self._root_present = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self.reset()
        self._root_pending = True
        self._active = True

    def reset(self) -> None:
        while self._stack:
            iterator, _ = self._stack.pop()
            try:
                iterator.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._root_pending = False
        self._root_follow_pending = False
        self._active = False
        self._xor_digest = 0
        self._sum_digest = 0
        self._entry_count = 0
        self._max_mtime_ns = 0
        self._root_present = False

    def _add(self, relative_path: str, row: os.stat_result) -> None:
        payload = b'\0'.join((
            os.fsencode(relative_path),
            str(int(row.st_mode)).encode('ascii'),
            str(int(row.st_size)).encode('ascii'),
            str(int(row.st_mtime_ns)).encode('ascii'),
        ))
        digest = hashlib.blake2b(payload, digest_size=32).digest()
        self._xor_digest ^= int.from_bytes(digest[:16], 'big')
        self._sum_digest = (self._sum_digest + int.from_bytes(digest[16:], 'big')) & _DIGEST_MASK
        self._entry_count += 1
        self._max_mtime_ns = max(self._max_mtime_ns, int(row.st_mtime_ns))

    def _snapshot(self) -> TreeSnapshot:
        digest = f'{self._xor_digest & _DIGEST_MASK:032x}{self._sum_digest:032x}'
        return TreeSnapshot(
            root_present=self._root_present,
            digest=digest,
            entry_count=self._entry_count,
            max_mtime_ns=self._max_mtime_ns,
        )

    def _fail(self, exc: OSError, processed: int) -> _ScanStep:
        error_code = f'scan_{exc.__class__.__name__.lower()}'
        self.reset()
        return _ScanStep(entries_processed=processed, error_code=error_code)

    def step(self, *, limit: int = MAX_SCAN_ENTRIES_PER_TICK) -> _ScanStep:
        limit = max(1, min(MAX_SCAN_ENTRIES_PER_TICK, int(limit)))
        if not self._active:
            self.start()
        processed = 0
        while processed < limit:
            if self._root_pending:
                self._root_pending = False
                try:
                    root_stat = self.root.stat(follow_symlinks=False)
                except FileNotFoundError:
                    snapshot = self._snapshot()
                    self._active = False
                    return _ScanStep(entries_processed=processed, snapshot=snapshot)
                except OSError as exc:
                    return self._fail(exc, processed)
                self._root_present = True
                self._add('.', root_stat)
                processed += 1
                if stat.S_ISLNK(root_stat.st_mode):
                    self._root_follow_pending = True
                elif stat.S_ISDIR(root_stat.st_mode):
                    try:
                        self._stack.append((os.scandir(self.root), ''))
                    except OSError as exc:
                        return self._fail(exc, processed)
                    if self.on_directory is not None:
                        self.on_directory(self.root)
                else:
                    snapshot = self._snapshot()
                    self._active = False
                    return _ScanStep(entries_processed=processed, snapshot=snapshot)
                continue

            if self._root_follow_pending:
                self._root_follow_pending = False
                try:
                    target_stat = self.root.stat()
                except FileNotFoundError:
                    snapshot = self._snapshot()
                    self._active = False
                    return _ScanStep(entries_processed=processed, snapshot=snapshot)
                except OSError as exc:
                    return self._fail(exc, processed)
                self._add('.@target', target_stat)
                processed += 1
                if stat.S_ISDIR(target_stat.st_mode):
                    try:
                        self._stack.append((os.scandir(self.root), ''))
                    except OSError as exc:
                        return self._fail(exc, processed)
                    if self.on_directory is not None:
                        self.on_directory(self.root)
                else:
                    snapshot = self._snapshot()
                    self._active = False
                    return _ScanStep(entries_processed=processed, snapshot=snapshot)
                continue

            if not self._stack:
                snapshot = self._snapshot()
                self._active = False
                return _ScanStep(entries_processed=processed, snapshot=snapshot)

            iterator, prefix = self._stack[-1]
            try:
                entry = next(iterator)
            except StopIteration:
                try:
                    iterator.close()  # type: ignore[attr-defined]
                finally:
                    self._stack.pop()
                continue
            except OSError as exc:
                return self._fail(exc, processed)

            relative = f'{prefix}/{entry.name}' if prefix else entry.name
            try:
                row = entry.stat(follow_symlinks=False)
            except OSError as exc:
                return self._fail(exc, processed)
            self._add(relative, row)
            processed += 1
            if stat.S_ISDIR(row.st_mode) and not stat.S_ISLNK(row.st_mode):
                try:
                    child_iterator = os.scandir(entry.path)
                except OSError as exc:
                    return self._fail(exc, processed)
                self._stack.append((child_iterator, relative))
                if self.on_directory is not None:
                    self.on_directory(Path(entry.path))

        return _ScanStep(entries_processed=processed)


def load_watch_manifest(path: Path) -> WatchManifest | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return WatchManifest.from_dict(payload)


def write_watch_manifest(path: Path, manifest: WatchManifest) -> None:
    """Atomically publish one complete aggregate manifest; never raw paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    data = (json.dumps(manifest.to_dict(), ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class ManifestPollingBackend:
    name = 'persisted-manifest'

    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        *,
        max_entries_per_tick: int = MAX_SCAN_ENTRIES_PER_TICK,
        min_backoff_seconds: float = MIN_IDLE_BACKOFF_SECONDS,
        max_backoff_seconds: float = MAX_IDLE_BACKOFF_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_directory: Callable[[Path], None] | None = None,
        completion_guard: Callable[[], bool] | None = None,
        scan_yield_seconds: float = SCAN_YIELD_SECONDS,
    ):
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.max_entries_per_tick = max(1, min(MAX_SCAN_ENTRIES_PER_TICK, int(max_entries_per_tick)))
        self.min_backoff_seconds = max(0.0, float(min_backoff_seconds))
        self.max_backoff_seconds = max(self.min_backoff_seconds, float(max_backoff_seconds))
        self._clock = clock
        self._sleep = sleep
        self._completion_guard = completion_guard
        self._scan_yield_seconds = max(0.0, float(scan_yield_seconds))
        self._scanner = BoundedManifestScanner(self.root, on_directory=on_directory)
        self._manifest = load_watch_manifest(self.manifest_path)
        self._backoff = self.min_backoff_seconds
        self._next_scan_at = 0.0
        self._scan_invalidated = False
        self._closed = False

    @property
    def scan_active(self) -> bool:
        return self._scanner.active

    def scan_due(self) -> bool:
        return self._scanner.active or self._clock() >= self._next_scan_at

    def request_repair(self, *, reason: str = 'manual') -> None:
        del reason
        if self._scanner.active:
            self._scan_invalidated = True
        else:
            self._next_scan_at = 0.0

    def _idle_tick(self) -> WatchTick:
        return WatchTick(backend=self.name, repair_pending=False)

    def poll(self, timeout: float = 1.0) -> WatchTick:
        if self._closed:
            raise RuntimeError('watch backend is closed')
        timeout = max(0.0, float(timeout))
        now = self._clock()
        if not self._scanner.active and now < self._next_scan_at:
            self._sleep(min(timeout, max(0.0, self._next_scan_at - now)))
            now = self._clock()
            if now < self._next_scan_at:
                return self._idle_tick()
        if not self._scanner.active:
            self._scanner.start()
            self._scan_invalidated = False
        elif timeout > 0 and self._scan_yield_seconds > 0:
            self._sleep(min(timeout, self._scan_yield_seconds))

        step = self._scanner.step(limit=self.max_entries_per_tick)
        if step.error_code:
            self._backoff = min(
                self.max_backoff_seconds,
                max(self.min_backoff_seconds, self._backoff * 2 or self.min_backoff_seconds),
            )
            self._next_scan_at = self._clock() + self._backoff
            return WatchTick(
                backend=self.name,
                entries_processed=step.entries_processed,
                repair_pending=True,
                event_loss=True,
                error_code=step.error_code,
            )
        if step.snapshot is None:
            return WatchTick(
                backend=self.name,
                entries_processed=step.entries_processed,
                repair_pending=True,
                scan_active=True,
            )
        if self._scan_invalidated:
            self._scanner.start()
            self._scan_invalidated = False
            self._next_scan_at = 0.0
            return WatchTick(
                backend=self.name,
                scan_discarded=True,
                scan_active=True,
                entries_processed=step.entries_processed,
                repair_pending=True,
                event_loss=True,
            )
        if self._completion_guard is not None and not self._completion_guard():
            self._scanner.start()
            self._next_scan_at = 0.0
            return WatchTick(
                backend=self.name,
                scan_discarded=True,
                scan_active=True,
                entries_processed=step.entries_processed,
                repair_pending=True,
                event_loss=True,
            )

        prior = self._manifest
        changed = bool(
            (prior is None and step.snapshot.root_present)
            or (prior is not None and prior.signature != step.snapshot.signature)
        )
        manifest = WatchManifest(
            root_present=step.snapshot.root_present,
            digest=step.snapshot.digest,
            entry_count=step.snapshot.entry_count,
            max_mtime_ns=step.snapshot.max_mtime_ns,
            scan_generation=(prior.scan_generation if prior is not None else 0) + 1,
            completed_at=time.time(),
        )
        try:
            write_watch_manifest(self.manifest_path, manifest)
        except OSError as exc:
            self._next_scan_at = self._clock() + self.min_backoff_seconds
            return WatchTick(
                backend=self.name,
                scan_complete=True,
                entries_processed=step.entries_processed,
                repair_pending=True,
                event_loss=True,
                error_code=f'manifest_{exc.__class__.__name__.lower()}',
            )
        self._manifest = manifest
        if changed:
            self._backoff = self.min_backoff_seconds
        else:
            self._backoff = min(
                self.max_backoff_seconds,
                max(self.min_backoff_seconds, self._backoff * 2 or self.min_backoff_seconds),
            )
        self._next_scan_at = self._clock() + self._backoff
        return WatchTick(
            backend=self.name,
            changed=changed,
            change_source='manifest' if changed else None,
            scan_complete=True,
            entries_processed=step.entries_processed,
            repair_pending=False,
            manifest_digest=manifest.digest,
        )

    def close(self) -> None:
        self._closed = True
        self._scanner.reset()


class KqueueWatchBackend:
    """Native macOS vnode notifications plus bounded manifest repair."""

    name = 'macos-kqueue+persisted-manifest'

    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        *,
        max_entries_per_tick: int = MAX_SCAN_ENTRIES_PER_TICK,
        max_directory_watches: int = MAX_DIRECTORY_WATCHES,
        max_events_per_tick: int = MAX_EVENTS_PER_TICK,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if sys.platform != 'darwin' or not hasattr(select, 'kqueue'):
            raise RuntimeError('kqueue backend is unavailable')
        self.max_directory_watches = max(1, int(max_directory_watches))
        self.max_events_per_tick = max(1, int(max_events_per_tick))
        self._kqueue = select.kqueue()
        self._fd_to_path: dict[int, Path] = {}
        self._watched_paths: set[Path] = set()
        self._descriptor_overflow = False
        self._closed = False
        self._late_changed = False
        self._late_event_loss = False
        self._polling = ManifestPollingBackend(
            root,
            manifest_path,
            max_entries_per_tick=max_entries_per_tick,
            clock=clock,
            sleep=sleep,
            on_directory=self._register_directory,
            completion_guard=self._completion_guard,
        )
        self._register_directory(Path(root).parent)
        self._register_directory(Path(root))

    def _register_directory(self, path: Path) -> None:
        normalized = Path(path)
        if normalized in self._watched_paths or self._closed:
            return
        if len(self._fd_to_path) >= self.max_directory_watches:
            self._descriptor_overflow = True
            return
        flags = getattr(os, 'O_EVTONLY', os.O_RDONLY) | getattr(os, 'O_CLOEXEC', 0)
        fd: int | None = None
        try:
            fd = os.open(normalized, flags)
            notes = 0
            for name in ('KQ_NOTE_WRITE', 'KQ_NOTE_DELETE', 'KQ_NOTE_RENAME', 'KQ_NOTE_ATTRIB', 'KQ_NOTE_EXTEND', 'KQ_NOTE_REVOKE'):
                notes |= int(getattr(select, name, 0))
            event = select.kevent(
                fd,
                filter=select.KQ_FILTER_VNODE,
                flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                fflags=notes,
            )
            self._kqueue.control([event], 0, 0)
        except FileNotFoundError:
            if fd is not None:
                os.close(fd)
            return
        except OSError:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._descriptor_overflow = True
            return
        if fd is None:
            return
        self._fd_to_path[fd] = normalized
        self._watched_paths.add(normalized)

    def _drop_fd(self, fd: int) -> None:
        path = self._fd_to_path.pop(fd, None)
        if path is not None:
            self._watched_paths.discard(path)
        try:
            os.close(fd)
        except OSError:
            pass

    def request_repair(self, *, reason: str = 'manual') -> None:
        self._polling.request_repair(reason=reason)

    def _consume_events(self, events: list[object]) -> tuple[bool, bool]:
        event_loss = len(events) >= self.max_events_per_tick
        terminal_notes = int(getattr(select, 'KQ_NOTE_DELETE', 0)) | int(getattr(select, 'KQ_NOTE_RENAME', 0)) | int(getattr(select, 'KQ_NOTE_REVOKE', 0))
        for event in events:
            if int(getattr(event, 'flags', 0)) & int(getattr(select, 'KQ_EV_ERROR', 0)):
                event_loss = True
            if int(getattr(event, 'fflags', 0)) & terminal_notes:
                self._drop_fd(int(getattr(event, 'ident')))
                event_loss = True
        return bool(events), event_loss

    def _completion_guard(self) -> bool:
        try:
            events = self._kqueue.control(None, self.max_events_per_tick, 0.0)
        except OSError:
            self._late_event_loss = True
            return False
        self._late_changed, self._late_event_loss = self._consume_events(events)
        return not self._late_changed and not self._late_event_loss

    def poll(self, timeout: float = 1.0) -> WatchTick:
        if self._closed:
            raise RuntimeError('watch backend is closed')
        self._late_changed = False
        self._late_event_loss = False
        wait = 0.0 if self._polling.scan_due() else max(0.0, float(timeout))
        try:
            events = self._kqueue.control(None, self.max_events_per_tick, wait)
        except OSError as exc:
            events = []
            event_loss = True
            error_code = f'kqueue_{exc.__class__.__name__.lower()}'
        else:
            _, event_loss = self._consume_events(events)
            error_code = None
        changed = bool(events)
        if changed or event_loss:
            self._polling.request_repair(reason='event_loss' if event_loss else 'event')
        repair = self._polling.poll(timeout=min(max(0.0, float(timeout)), SCAN_YIELD_SECONDS))
        changed = changed or self._late_changed
        event_loss = event_loss or self._late_event_loss
        return WatchTick(
            backend=self.name,
            changed=changed or repair.changed,
            change_source='event' if changed else repair.change_source,
            scan_complete=repair.scan_complete,
            scan_discarded=repair.scan_discarded,
            scan_active=repair.scan_active,
            entries_processed=repair.entries_processed,
            repair_pending=repair.repair_pending or event_loss or self._descriptor_overflow,
            event_loss=event_loss or repair.event_loss,
            descriptor_overflow=self._descriptor_overflow,
            error_code=error_code or repair.error_code,
            manifest_digest=repair.manifest_digest,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._polling.close()
        for fd in list(self._fd_to_path):
            self._drop_fd(fd)
        self._kqueue.close()


def create_watch_backend(root: Path, manifest_path: Path) -> WatchBackend:
    if sys.platform == 'darwin' and hasattr(select, 'kqueue'):
        try:
            return KqueueWatchBackend(root, manifest_path)
        except (OSError, RuntimeError):
            pass
    return ManifestPollingBackend(root, manifest_path)
