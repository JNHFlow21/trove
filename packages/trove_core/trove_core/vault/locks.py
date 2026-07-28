from __future__ import annotations

from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, TYPE_CHECKING
from uuid import uuid4
import weakref

from trove_core.vault.config import VaultConfig

if TYPE_CHECKING:
    from trove_core.vault.coordinator import VaultOperationCoordinator, VaultWriteSession


LOCK_STALE_SECONDS = 24 * 60 * 60
RUNNING_VECTOR_STATES = {'opening', 'running', 'resetting'}
_STABLE_LOCK_MARKER = b'TROVE_VAULT_WRITER_LOCK_V1\n'
_SAFE_OWNER_LABELS = {'import', 'maintain', 'sync', 'vector-index', 'vector-rebuild'}
_LOCK_FILE_NAME = 'trove-index-writer.flock'
_PID_FILE_NAME = 'trove-index-writer.pid'
_INFO_FILE_NAME = 'trove-index-writer.lock.json'
_MAX_PID_BYTES = 64
_MAX_INFO_BYTES = 16 * 1024
_NONCE_RE = re.compile(r'^[0-9a-f]{32}$')
_LOCK_OBJECTS: weakref.WeakSet['_StableVaultWriterLock'] = weakref.WeakSet()
_FORK_LOCK_OBJECTS: tuple['_StableVaultWriterLock', ...] = ()


class VaultOperationLocked(RuntimeError):
    """Raised when a local Vault writer cannot be acquired safely."""

    def __init__(self, message: str = 'Vault writer is already active', *, code: str = 'vault_writer_locked'):
        super().__init__(message)
        self.code = code


def _identity(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_dev), int(stat_result.st_ino)


def _safe_int(value: Any, *, minimum: int | None = None) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value and len(value) <= 20 and value.isascii():
        try:
            result = int(value, 10)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None
    if minimum is not None and result < minimum:
        return None
    return result


def _safe_float(value: Any, *, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str) and (not value or len(value) > 64):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _is_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except (TypeError, ValueError):
        return False
    return value == value.lower()


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError('short write')
        view = view[written:]


def _fsync_directory(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, getattr(errno, 'ENOTSUP', errno.EINVAL)}:
            raise


def _open_flags(*, directory: bool = False, writable: bool = False) -> int:
    flags = os.O_RDWR if writable else os.O_RDONLY
    if directory and hasattr(os, 'O_DIRECTORY'):
        flags |= os.O_DIRECTORY
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    return flags


def _read_fd_limited(fd: int, maximum: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 4096))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b''.join(chunks)
    return payload if len(payload) <= maximum else None


def _read_leaf_at(dir_fd: int, name: str, *, maximum: int) -> dict[str, Any]:
    """Read one diagnostic leaf without following links or trusting its content."""

    result: dict[str, Any] = {'present': False, 'identity': None, 'mtime': None, 'regular': False, 'payload': None}
    try:
        listed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return result
    except OSError:
        result['present'] = True
        return result
    result.update(
        present=True,
        identity=_identity(listed),
        mtime=float(listed.st_mtime),
        regular=stat.S_ISREG(listed.st_mode) and int(listed.st_nlink) == 1,
    )
    if not result['regular']:
        return result
    fd: int | None = None
    try:
        fd = os.open(name, _open_flags(), dir_fd=dir_fd)
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or _identity(opened) != result['identity']
        ):
            return result
        result['payload'] = _read_fd_limited(fd, maximum)
    except OSError:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    return result


def _read_lock_info_at(dir_fd: int) -> dict[str, Any]:
    pid_leaf = _read_leaf_at(dir_fd, _PID_FILE_NAME, maximum=_MAX_PID_BYTES)
    info_leaf = _read_leaf_at(dir_fd, _INFO_FILE_NAME, maximum=_MAX_INFO_BYTES)
    data: dict[str, Any] = {
        '_pid_present': pid_leaf['present'],
        '_pid_identity': pid_leaf['identity'],
        '_pid_mtime': pid_leaf['mtime'],
        '_pid_regular': pid_leaf['regular'],
        '_info_present': info_leaf['present'],
        '_info_identity': info_leaf['identity'],
        '_info_regular': info_leaf['regular'],
    }
    info_payload = info_leaf['payload']
    if isinstance(info_payload, bytes):
        try:
            loaded = json.loads(info_payload.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            # Copy only protocol fields; arbitrary keys never become control
            # metadata and are never reflected in an exception.
            for key in (
                'schema',
                'pid',
                'owner',
                'created_at',
                'process_start_time',
                'process_birth_hash',
                'owner_nonce',
                'vault_hash',
            ):
                if key in loaded:
                    data[key] = loaded[key]
            data['_info_pid'] = _safe_int(loaded.get('pid'), minimum=1)
    pid_payload = pid_leaf['payload']
    if isinstance(pid_payload, bytes):
        try:
            decoded_pid = pid_payload.decode('ascii')
        except UnicodeError:
            decoded_pid = ''
        # A newline is the publication-complete delimiter used by both the
        # legacy writer and this implementation.  Digits without it may be a
        # paused partial write and therefore remain permanently fail-closed.
        pid_text = decoded_pid[:-1] if re.fullmatch(r'[1-9][0-9]{0,19}\n', decoded_pid) else ''
        pid = _safe_int(pid_text, minimum=1)
        if pid is not None:
            data['pid'] = pid
            data['_pid_valid'] = True
        else:
            data['_pid_valid'] = False
    else:
        data['_pid_valid'] = False
    return data


def _atomic_write_json_at(dir_fd: int, name: str, payload: dict[str, Any]) -> tuple[int, int]:
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    tmp_name = f'.{name}.{os.getpid()}.{uuid4().hex}.tmp'
    fd: int | None = None
    tmp_identity: tuple[int, int] | None = None
    linked = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp_name, flags, 0o600, dir_fd=dir_fd)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            raise OSError('diagnostic temporary file is unsafe')
        tmp_identity = _identity(opened)
        _write_all(fd, encoded)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.link(
            tmp_name,
            name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(tmp_name, dir_fd=dir_fd)
        _fsync_directory(dir_fd)
        listed = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if (
            tmp_identity is None
            or _identity(listed) != tmp_identity
            or not stat.S_ISREG(listed.st_mode)
            or int(listed.st_nlink) != 1
        ):
            raise OSError('diagnostic identity changed')
        return tmp_identity
    except BaseException:
        if linked and tmp_identity is not None:
            _unlink_identity_at(dir_fd, name, tmp_identity, require_regular=False)
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name, dir_fd=dir_fd)
        except OSError:
            pass


def _publish_pid_marker_at(dir_fd: int, *, pid: int) -> tuple[int, int]:
    """Publish a fully written legacy marker with an atomic no-replace link."""

    payload = f'{pid}\n'.encode('ascii')
    tmp_name = f'.{_PID_FILE_NAME}.{pid}.{uuid4().hex}.tmp'
    fd: int | None = None
    tmp_identity: tuple[int, int] | None = None
    linked = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp_name, flags, 0o600, dir_fd=dir_fd)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            raise OSError('PID temporary file is unsafe')
        tmp_identity = _identity(opened)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        # link(2), unlike rename, is an atomic no-replace publication primitive.
        os.link(
            tmp_name,
            _PID_FILE_NAME,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(tmp_name, dir_fd=dir_fd)
        _fsync_directory(dir_fd)
        listed = os.stat(_PID_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
        if (
            tmp_identity is None
            or _identity(listed) != tmp_identity
            or not stat.S_ISREG(listed.st_mode)
            or int(listed.st_nlink) != 1
        ):
            raise OSError('PID marker identity changed')
        return tmp_identity
    except BaseException:
        # If publication succeeded and a later durability/verification step
        # failed, remove only the inode created by this attempt.
        if linked and tmp_identity is not None:
            # The temporary hardlink may still exist if its unlink was the
            # failing step, so permit link-count two for this known inode.
            _unlink_identity_at(dir_fd, _PID_FILE_NAME, tmp_identity, require_regular=False)
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name, dir_fd=dir_fd)
        except OSError:
            pass


def _unlink_identity_at(
    dir_fd: int,
    name: str,
    expected: tuple[int, int],
    *,
    require_regular: bool,
) -> bool:
    try:
        current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if _identity(current) != expected:
        return False
    if require_regular and (not stat.S_ISREG(current.st_mode) or int(current.st_nlink) != 1):
        return False
    if stat.S_ISDIR(current.st_mode):
        return False
    try:
        os.unlink(name, dir_fd=dir_fd)
        return True
    except OSError:
        return False


class _StableVaultWriterLock:
    """Kernel lock plus an inode-bound rolling-upgrade marker protocol."""

    def __init__(
        self,
        cfg: VaultConfig,
        *,
        owner: str,
        owner_nonce: str,
        process_birth: str,
        vault_hash: str,
        stale_seconds: int = LOCK_STALE_SECONDS,
    ):
        self.cfg = cfg
        self.owner = owner
        self.owner_nonce = owner_nonce
        self.process_birth = process_birth
        self.vault_hash = vault_hash
        self.stale_seconds = stale_seconds
        self._fd: int | None = None
        self._root_fd: int | None = None
        self._dir_fd: int | None = None
        self._root_path = Path(os.path.abspath(os.path.expanduser(str(cfg.root))))
        self._root_identity: tuple[int, int] | None = None
        self._dir_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._pid_identity: tuple[int, int] | None = None
        self._pid_fd: int | None = None
        self._info_identity: tuple[int, int] | None = None
        self._owner_pid: int | None = None
        _LOCK_OBJECTS.add(self)

    @property
    def lock_path(self) -> Path:
        return self._root_path / 'logs' / _LOCK_FILE_NAME

    @property
    def pid_path(self) -> Path:
        return self._root_path / 'logs' / _PID_FILE_NAME

    @property
    def info_path(self) -> Path:
        return self._root_path / 'logs' / _INFO_FILE_NAME

    def acquire(self) -> None:
        if self._fd is not None:
            raise VaultOperationLocked('Vault writer lock object is already active')
        self.cfg.validate_runtime_path()
        fd: int | None = None
        try:
            self._prepare_bound_directories()
            dir_fd = self._require_dir_fd()
            flags = os.O_CREAT | _open_flags(writable=True)
            fd = os.open(_LOCK_FILE_NAME, flags, 0o600, dir_fd=dir_fd)
            self._fd = fd
            opened = os.fstat(fd)
            listed = os.stat(_LOCK_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != 1
                or _identity(opened) != _identity(listed)
            ):
                raise VaultOperationLocked(
                    'Vault writer lock file is unsafe',
                    code='vault_writer_lock_unavailable',
                )
            # Validate type, link count, and pathname binding before mutating
            # permissions or bytes; a hardlink must have zero side effects.
            self._lock_identity = _identity(opened)
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise VaultOperationLocked() from exc

            initialized = self._is_initialized(fd)
            self._claim_legacy_marker()
            if not initialized:
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                _write_all(fd, _STABLE_LOCK_MARKER)
                os.fsync(fd)
            self._write_info_diagnostic()
            fd = None
            self.validate_bound_identity()
        except BaseException as exc:
            self._clear_owned_diagnostics()
            owned_fd = self._fd
            self._fd = None
            if owned_fd is not None:
                try:
                    fcntl.flock(owned_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(owned_fd)
                except OSError:
                    pass
            self._close_bound_directories()
            if isinstance(exc, VaultOperationLocked):
                raise
            raise VaultOperationLocked(
                'Vault writer lock could not be acquired safely',
                code='vault_writer_lock_unavailable',
            ) from exc

    def _prepare_bound_directories(self) -> None:
        root = self._root_path
        try:
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            listed_root = os.lstat(root)
            if stat.S_ISLNK(listed_root.st_mode) or not stat.S_ISDIR(listed_root.st_mode):
                raise OSError('Vault root is not a real directory')
            root_fd = os.open(root, _open_flags(directory=True))
            opened_root = os.fstat(root_fd)
            if not stat.S_ISDIR(opened_root.st_mode) or _identity(opened_root) != _identity(listed_root):
                raise OSError('Vault root identity changed')
            self._root_fd = root_fd
            self._root_identity = _identity(opened_root)

            try:
                os.mkdir('logs', 0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            listed_logs = os.stat('logs', dir_fd=root_fd, follow_symlinks=False)
            if stat.S_ISLNK(listed_logs.st_mode) or not stat.S_ISDIR(listed_logs.st_mode):
                raise OSError('Vault writer directory is not a real directory')
            dir_fd = os.open('logs', _open_flags(directory=True), dir_fd=root_fd)
            self._dir_fd = dir_fd
            opened_logs = os.fstat(dir_fd)
            if not stat.S_ISDIR(opened_logs.st_mode) or _identity(opened_logs) != _identity(listed_logs):
                raise OSError('Vault writer directory identity changed')
            os.fchmod(dir_fd, 0o700)
            self._dir_identity = _identity(opened_logs)
        except OSError as exc:
            self._close_bound_directories()
            raise VaultOperationLocked(
                'Vault writer directory could not be bound safely',
                code='vault_writer_lock_unavailable',
            ) from exc

    def _require_dir_fd(self) -> int:
        if self._dir_fd is None:
            raise OSError('writer directory is not bound')
        return self._dir_fd

    @staticmethod
    def _is_initialized(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        return os.read(fd, len(_STABLE_LOCK_MARKER)) == _STABLE_LOCK_MARKER

    def _claim_legacy_marker(self) -> None:
        dir_fd = self._require_dir_fd()
        pid_fd: int | None = None
        identity: tuple[int, int] | None = None
        try:
            identity = _publish_pid_marker_at(dir_fd, pid=os.getpid())
            pid_fd = os.open(_PID_FILE_NAME, _open_flags(), dir_fd=dir_fd)
            opened = os.fstat(pid_fd)
            listed = os.stat(_PID_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != 1
                or _identity(opened) != identity
                or _identity(listed) != identity
            ):
                raise OSError('PID marker identity changed')
        except FileExistsError as exc:
            # Never auto-delete a pre-existing rolling-upgrade marker.  An old
            # binary may already have classified that inode as stale and be
            # paused immediately before its unconditional unlink; replacing it
            # here would let that old writer erase our marker and run beside us.
            # Recovery is deliberately an explicit offline operator action.
            raise VaultOperationLocked(
                'Vault writer marker requires explicit offline recovery',
                code='vault_writer_marker_recovery_required',
            ) from exc
        except OSError as exc:
            if pid_fd is not None:
                os.close(pid_fd)
            if identity is not None:
                _unlink_identity_at(dir_fd, _PID_FILE_NAME, identity, require_regular=True)
            raise VaultOperationLocked(
                'Vault writer compatibility marker could not be published',
                code='vault_writer_diagnostics_unavailable',
            ) from exc
        self._pid_identity = identity
        self._pid_fd = pid_fd
        self._owner_pid = os.getpid()

    def _write_info_diagnostic(self) -> None:
        payload = {
            'schema': 1,
            'pid': os.getpid(),
            'owner': self.owner if self.owner in _SAFE_OWNER_LABELS else 'other',
            'created_at': time.time(),
            'process_birth_hash': hashlib.sha256(self.process_birth.encode('utf-8')).hexdigest(),
            'owner_nonce': self.owner_nonce,
            'vault_hash': self.vault_hash,
        }
        try:
            dir_fd = self._require_dir_fd()
            existing = _read_leaf_at(dir_fd, _INFO_FILE_NAME, maximum=_MAX_INFO_BYTES)
            if existing.get('present'):
                identity = existing.get('identity')
                if (
                    not existing.get('regular')
                    or not isinstance(identity, tuple)
                    or not _unlink_identity_at(dir_fd, _INFO_FILE_NAME, identity, require_regular=True)
                ):
                    raise OSError('diagnostic slot is unsafe')
                _fsync_directory(dir_fd)
            self._info_identity = _atomic_write_json_at(dir_fd, _INFO_FILE_NAME, payload)
        except Exception as exc:
            raise VaultOperationLocked(
                'Vault writer diagnostics could not be written',
                code='vault_writer_diagnostics_unavailable',
            ) from exc

    def validate_bound_identity(self) -> None:
        """Fail closed if any pathname no longer names the acquired objects."""

        if self._fd is None or self._root_fd is None or self._dir_fd is None:
            raise VaultOperationLocked('Vault writer lock is inactive', code='vault_writer_lock_unavailable')
        try:
            opened_root = os.fstat(self._root_fd)
            listed_root = os.lstat(self._root_path)
            opened_logs = os.fstat(self._dir_fd)
            listed_logs = os.stat('logs', dir_fd=self._root_fd, follow_symlinks=False)
            opened_lock = os.fstat(self._fd)
            listed_lock = os.stat(_LOCK_FILE_NAME, dir_fd=self._dir_fd, follow_symlinks=False)
        except OSError as exc:
            raise VaultOperationLocked(
                'Vault writer path identity changed',
                code='vault_writer_path_changed',
            ) from exc
        if (
            self._root_identity is None
            or self._dir_identity is None
            or self._lock_identity is None
            or stat.S_ISLNK(listed_root.st_mode)
            or not stat.S_ISDIR(opened_root.st_mode)
            or _identity(opened_root) != self._root_identity
            or _identity(listed_root) != self._root_identity
            or not stat.S_ISDIR(opened_logs.st_mode)
            or _identity(opened_logs) != self._dir_identity
            or _identity(listed_logs) != self._dir_identity
            or not stat.S_ISREG(opened_lock.st_mode)
            or int(opened_lock.st_nlink) != 1
            or _identity(opened_lock) != self._lock_identity
            or _identity(listed_lock) != self._lock_identity
        ):
            raise VaultOperationLocked(
                'Vault writer path identity changed',
                code='vault_writer_path_changed',
            )

    def _owned_pid_content_matches(self, dir_fd: int) -> bool:
        if self._pid_identity is None or self._pid_fd is None or self._owner_pid is None:
            return False
        try:
            opened = os.fstat(self._pid_fd)
        except OSError:
            return False
        if _identity(opened) != self._pid_identity or not stat.S_ISREG(opened.st_mode):
            return False
        leaf = _read_leaf_at(dir_fd, _PID_FILE_NAME, maximum=_MAX_PID_BYTES)
        if leaf.get('identity') != self._pid_identity or not leaf.get('regular'):
            return False
        payload = leaf.get('payload')
        if not isinstance(payload, bytes):
            return False
        try:
            value = _safe_int(payload.decode('ascii').strip(), minimum=1)
        except UnicodeError:
            return False
        return value == self._owner_pid

    def _owned_info_content_matches(self, dir_fd: int) -> bool:
        if self._info_identity is None:
            return False
        leaf = _read_leaf_at(dir_fd, _INFO_FILE_NAME, maximum=_MAX_INFO_BYTES)
        if leaf.get('identity') != self._info_identity or not leaf.get('regular'):
            return False
        payload = leaf.get('payload')
        if not isinstance(payload, bytes):
            return False
        try:
            loaded = json.loads(payload.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
        return (
            isinstance(loaded, dict)
            and loaded.get('owner_nonce') == self.owner_nonce
            and _safe_int(loaded.get('pid'), minimum=1) == self._owner_pid
            and loaded.get('vault_hash') == self.vault_hash
        )

    def _clear_owned_diagnostics(self) -> None:
        dir_fd = self._dir_fd
        if dir_fd is None:
            return
        # Non-authoritative observations must never make release fail.  Each
        # unlink requires both the inode captured at publication and the owner
        # identity stored inside that exact inode.
        if self._info_identity is not None and self._owned_info_content_matches(dir_fd):
            _unlink_identity_at(dir_fd, _INFO_FILE_NAME, self._info_identity, require_regular=True)
        if self._pid_identity is not None and self._owned_pid_content_matches(dir_fd):
            _unlink_identity_at(dir_fd, _PID_FILE_NAME, self._pid_identity, require_regular=True)
        try:
            _fsync_directory(dir_fd)
        except OSError:
            pass

    def _close_bound_directories(self) -> None:
        for attr in ('_pid_fd', '_dir_fd', '_root_fd'):
            fd = getattr(self, attr)
            setattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _after_fork_in_child(self) -> None:
        """Drop inherited open-file descriptions without unlocking the parent."""

        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        self._close_bound_directories()

    def _before_fork(self) -> None:
        """Close the flock description before fork so a child cannot inherit it.

        An ``after_in_child`` close alone has a scheduler race: the parent can
        crash immediately after ``fork(2)`` while the child has not yet run its
        Python callback, leaving the child as the final reference to the shared
        flock description.  Closing here removes ownership before the kernel
        clones descriptors.  Durable PID/info markers remain published, so a
        normal contender still fails closed during the tiny parent-reopen
        window.
        """

        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _after_fork_in_parent(self) -> None:
        """Reopen the bound stable inode on a parent-only file description."""

        if self._lock_identity is None or self._dir_fd is None:
            return
        dir_fd = self._require_dir_fd()
        flags = _open_flags(writable=True)
        fd = os.open(_LOCK_FILE_NAME, flags, dir_fd=dir_fd)
        try:
            opened = os.fstat(fd)
            listed = os.stat(_LOCK_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != 1
                or _identity(opened) != self._lock_identity
                or _identity(listed) != self._lock_identity
            ):
                raise VaultOperationLocked(
                    'Vault writer lock identity changed across fork',
                    code='vault_writer_path_changed',
                )
            # Product contenders observe the still-published diagnostic marker
            # and immediately fail closed, so this normally returns at once.
            # Blocking is intentional: continuing a parent mutation without
            # serialization would be worse than waiting for a raw-flock race.
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._fd = fd
            fd = -1
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            self._close_bound_directories()
            return
        self._fd = None
        try:
            self._clear_owned_diagnostics()
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            self._close_bound_directories()


def _prepare_lock_objects_for_fork() -> None:
    """Remove every active flock description before descriptor cloning."""

    global _FORK_LOCK_OBJECTS
    inherited = tuple(lock for lock in _LOCK_OBJECTS if lock._fd is not None)
    _FORK_LOCK_OBJECTS = inherited
    for lock in inherited:
        lock._before_fork()


def _restore_parent_lock_objects() -> None:
    """Give the parent fresh, non-shared flock descriptions after fork."""

    global _FORK_LOCK_OBJECTS
    inherited = _FORK_LOCK_OBJECTS
    _FORK_LOCK_OBJECTS = ()
    for lock in inherited:
        lock._after_fork_in_parent()


def _close_inherited_lock_objects() -> None:
    """Cover a fork that lands while a coordinator is still acquiring."""

    global _LOCK_OBJECTS, _FORK_LOCK_OBJECTS
    inherited = tuple(dict.fromkeys((*_FORK_LOCK_OBJECTS, *_LOCK_OBJECTS)))
    _FORK_LOCK_OBJECTS = ()
    _LOCK_OBJECTS = weakref.WeakSet()
    for lock in inherited:
        lock._after_fork_in_child()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(
        before=_prepare_lock_objects_for_fork,
        after_in_parent=_restore_parent_lock_objects,
        after_in_child=_close_inherited_lock_objects,
    )


@dataclass(frozen=True)
class VaultOperationLock:
    """Compatibility facade backed by :class:`VaultOperationCoordinator`."""

    cfg: VaultConfig
    owner: str
    stale_seconds: int = LOCK_STALE_SECONDS
    owner_nonce: str = ''
    _coordinator: 'VaultOperationCoordinator | None' = field(init=False, default=None, repr=False, compare=False)
    _write_session: 'VaultWriteSession | None' = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        nonce = self.owner_nonce or uuid4().hex
        if not _NONCE_RE.fullmatch(nonce):
            raise ValueError('owner_nonce must be 32 lowercase hexadecimal characters')
        object.__setattr__(self, 'owner_nonce', nonce)

    @property
    def lock_path(self) -> Path:
        return self.cfg.paths.logs_dir / _LOCK_FILE_NAME

    @property
    def pid_path(self) -> Path:
        return self.cfg.paths.logs_dir / _PID_FILE_NAME

    @property
    def info_path(self) -> Path:
        return self.cfg.paths.logs_dir / _INFO_FILE_NAME

    @property
    def write_session(self) -> 'VaultWriteSession | None':
        return self._write_session

    def __enter__(self) -> 'VaultOperationLock':
        from trove_core.vault.coordinator import VaultOperationCoordinator

        if self._write_session is not None:
            raise VaultOperationLocked('Vault writer lock object is already active')
        coordinator = VaultOperationCoordinator(self.cfg, stale_seconds=self.stale_seconds)
        session = coordinator.acquire(owner=self.owner, owner_nonce=self.owner_nonce)
        object.__setattr__(self, '_coordinator', coordinator)
        object.__setattr__(self, '_write_session', session)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def release(self) -> None:
        coordinator = self._coordinator
        session = self._write_session
        if coordinator is None or session is None:
            return
        coordinator.release(session)
        object.__setattr__(self, '_write_session', None)
        object.__setattr__(self, '_coordinator', None)


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


def _process_start_time(pid: int) -> str | None:
    if pid <= 0:
        return None
    try:
        proc = subprocess.run(
            ['ps', '-o', 'lstart=', '-p', str(pid)],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value or None


def _lock_owner_running(lock: dict[str, Any]) -> bool:
    pid = _safe_int(lock.get('pid'), minimum=1)
    if pid is None or not _pid_running(pid):
        return False
    locked_start = lock.get('process_start_time')
    if isinstance(locked_start, str) and 0 < len(locked_start) <= 256:
        current_start = _process_start_time(pid)
        if current_start and current_start != locked_start:
            return False
    return True


def _legacy_marker_blocks(lock: dict[str, Any], *, stale_seconds: int) -> bool:
    if bool(lock.get('_pid_present')) and not bool(lock.get('_pid_valid')):
        # A legacy binary publishes with O_EXCL before writing its PID.  There
        # is no bounded time after which an empty/corrupt inode proves that
        # writer is dead, so automatic recovery would violate exclusivity.
        # Leave it for an explicit offline repair procedure instead.
        return True
    pid = _safe_int(lock.get('pid'), minimum=1)
    if pid is not None:
        return _lock_owner_running(lock)
    created_at = _safe_float(lock.get('created_at'), minimum=0)
    return bool(created_at is not None and time.time() - created_at < stale_seconds)


def _is_matching_new_diagnostic(lock: dict[str, Any], *, expected_vault_hash: str) -> bool:
    return (
        lock.get('_pid_regular') is True
        and lock.get('_info_regular') is True
        and _safe_int(lock.get('schema'), minimum=1) == 1
        and _safe_int(lock.get('_info_pid'), minimum=1) == _safe_int(lock.get('pid'), minimum=1)
        and lock.get('vault_hash') == expected_vault_hash
        and _is_hex(lock.get('owner_nonce'), 32)
        and _is_hex(lock.get('process_birth_hash'), 64)
    )


def _read_lock_info(pid_path: Path, info_path: Path) -> dict[str, Any]:
    """Compatibility diagnostic reader; malformed content is always inert."""

    try:
        if pid_path.parent != info_path.parent:
            return {}
        dir_fd = os.open(pid_path.parent, _open_flags(directory=True))
    except OSError:
        return {}
    try:
        return _read_lock_info_at(dir_fd)
    finally:
        os.close(dir_fd)


def _writer_lock_held(cfg: VaultConfig) -> bool:
    logs_dir = cfg.paths.logs_dir
    vault_hash = hashlib.sha256(str(cfg.root.expanduser().resolve()).encode('utf-8')).hexdigest()
    try:
        listed_logs = os.lstat(logs_dir)
        if stat.S_ISLNK(listed_logs.st_mode) or not stat.S_ISDIR(listed_logs.st_mode):
            return True
        dir_fd = os.open(logs_dir, _open_flags(directory=True))
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        try:
            listed = os.stat(_LOCK_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            legacy = _read_lock_info_at(dir_fd)
            return bool(legacy and _legacy_marker_blocks(legacy, stale_seconds=LOCK_STALE_SECONDS))
        except OSError:
            return True
        if not stat.S_ISREG(listed.st_mode) or int(listed.st_nlink) != 1:
            return True
        try:
            fd = os.open(_LOCK_FILE_NAME, _open_flags(writable=True), dir_fd=dir_fd)
        except OSError:
            return True
        try:
            opened = os.fstat(fd)
            if _identity(opened) != _identity(listed) or int(opened.st_nlink) != 1:
                return True
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                return True
            try:
                legacy = _read_lock_info_at(dir_fd)
                return bool(
                    legacy
                    and not _is_matching_new_diagnostic(legacy, expected_vault_hash=vault_hash)
                    and _legacy_marker_blocks(legacy, stale_seconds=LOCK_STALE_SECONDS)
                )
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)


def active_vector_progress(cfg: VaultConfig, *, max_age_seconds: int = 12 * 60 * 60) -> dict[str, Any] | None:
    """Return recent ZVEC progress only while an authoritative writer is active."""

    progress_path = cfg.paths.vector_dir / 'zvec' / 'messages.trove-progress.json'
    try:
        data = json.loads(progress_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    state_value = data.get('state')
    state = state_value if isinstance(state_value, str) else ''
    updated_at = _safe_float(data.get('updated_at'), minimum=0)
    if (
        state in RUNNING_VECTOR_STATES
        and updated_at is not None
        and time.time() - updated_at <= max_age_seconds
        and _writer_lock_held(cfg)
    ):
        return {'state': state, 'updated_at': updated_at}
    return None
