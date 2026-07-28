"""Explicit offline recovery for the writer marker protocol in ``locks``."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable
import weakref

from trove_core.vault.config import VaultConfig
from trove_core.vault.locks import (
    _INFO_FILE_NAME,
    _LOCK_FILE_NAME,
    _MAX_INFO_BYTES,
    _MAX_PID_BYTES,
    _PID_FILE_NAME,
    _fsync_directory,
    _identity,
    _is_hex,
    _open_flags,
    _pid_running,
    _process_start_time,
    _read_fd_limited,
    _safe_int,
    _unlink_identity_at,
)


_PID_TEMP_RE = re.compile(
    r'^\.trove-index-writer\.pid\.([1-9][0-9]{0,19})\.[0-9a-f]{32}\.tmp$'
)
_INFO_TEMP_RE = re.compile(
    r'^\.trove-index-writer\.lock\.json\.([1-9][0-9]{0,19})\.[0-9a-f]{32}\.tmp$'
)
_RECOVERY_OBJECTS: weakref.WeakSet['_OfflineWriterMarkerRecovery'] = weakref.WeakSet()


class WriterMarkerRecoveryError(RuntimeError):
    """Typed, path-free failure from the explicit offline recovery protocol."""

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WriterMarkerRecoveryResult:
    """A redacted recovery report safe for CLI and application adapters."""

    code: str
    recovered: bool
    pid_marker_removed: bool
    info_marker_removed: bool
    temporary_markers_removed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': True,
            'code': self.code,
            'recovered': self.recovered,
            'pid_marker_removed': self.pid_marker_removed,
            'info_marker_removed': self.info_marker_removed,
            'temporary_markers_removed': self.temporary_markers_removed,
            'paths_included': False,
        }


class _OfflineWriterMarkerRecovery:
    """One-shot, inode-bound recovery while the stable authority is held.

    Legacy binaries do not participate in the stable flock protocol.  The
    application boundary therefore requires an explicit operator assertion
    that every legacy writer has been stopped before this helper can be used.
    This class still proves that the PID recorded by the bound marker is dead,
    or that the live PID has a different process birth, before cleanup.
    """

    def __init__(self, cfg: VaultConfig, *, legacy_writers_stopped: bool):
        if type(legacy_writers_stopped) is not bool or legacy_writers_stopped is not True:
            raise WriterMarkerRecoveryError(
                'Legacy writer shutdown must be explicitly confirmed',
                code='writer_marker_recovery_confirmation_required',
            )
        self.cfg = cfg
        self._root_path = Path(os.path.abspath(os.path.expanduser(str(cfg.root))))
        self._vault_hash: str | None = None
        self._root_fd: int | None = None
        self._dir_fd: int | None = None
        self._lock_fd: int | None = None
        self._marker_fd: int | None = None
        self._root_identity: tuple[int, int] | None = None
        self._dir_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._marker_identity: tuple[int, int] | None = None
        self._marker_payload: bytes | None = None
        self._marker_pid: int | None = None
        self._marker_links = 0
        self._info_identity: tuple[int, int] | None = None
        self._info_payload: bytes | None = None
        self._info_links = 0
        self._info_temp_name: str | None = None
        self._info_temp_identity: tuple[int, int] | None = None
        self._temp_name: str | None = None
        self._temp_identity: tuple[int, int] | None = None
        self._prepared = False
        _RECOVERY_OBJECTS.add(self)

    def prepare(self) -> None:
        """Bind authority and prove the observed marker cannot own a writer."""

        try:
            self.cfg.require_configured_for_write(action='Writer marker recovery')
            self._bind_authority()
            self._snapshot_marker()
            self._snapshot_info()
            if self._marker_identity is not None:
                self._prove_marker_owner_inactive()
            self._validate_all_snapshots()
            self._prepared = True
        except WriterMarkerRecoveryError:
            self.close()
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            self.close()
            raise WriterMarkerRecoveryError(
                'Writer marker recovery could not bind the Vault safely',
                code='writer_marker_recovery_unavailable',
            ) from exc

    def _bind_authority(self) -> None:
        try:
            listed_root = os.lstat(self._root_path)
            if stat.S_ISLNK(listed_root.st_mode) or not stat.S_ISDIR(listed_root.st_mode):
                raise OSError('unsafe root')
            root_fd = os.open(self._root_path, _open_flags(directory=True))
            self._root_fd = root_fd
            opened_root = os.fstat(root_fd)
            if not stat.S_ISDIR(opened_root.st_mode) or _identity(opened_root) != _identity(listed_root):
                raise OSError('root identity changed')
            self._root_identity = _identity(opened_root)
            canonical_root = str(self._root_path.resolve())
            self._vault_hash = hashlib.sha256(canonical_root.encode('utf-8')).hexdigest()

            listed_logs = os.stat('logs', dir_fd=root_fd, follow_symlinks=False)
            if stat.S_ISLNK(listed_logs.st_mode) or not stat.S_ISDIR(listed_logs.st_mode):
                raise OSError('unsafe writer directory')
            dir_fd = os.open('logs', _open_flags(directory=True), dir_fd=root_fd)
            self._dir_fd = dir_fd
            opened_logs = os.fstat(dir_fd)
            if not stat.S_ISDIR(opened_logs.st_mode) or _identity(opened_logs) != _identity(listed_logs):
                raise OSError('writer directory identity changed')
            self._dir_identity = _identity(opened_logs)

            flags = os.O_CREAT | _open_flags(writable=True)
            lock_fd = os.open(_LOCK_FILE_NAME, flags, 0o600, dir_fd=dir_fd)
            self._lock_fd = lock_fd
            opened_lock = os.fstat(lock_fd)
            listed_lock = os.stat(_LOCK_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened_lock.st_mode)
                or int(opened_lock.st_nlink) != 1
                or _identity(opened_lock) != _identity(listed_lock)
            ):
                raise OSError('unsafe authority lock')
            self._lock_identity = _identity(opened_lock)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError) as exc:
                raise WriterMarkerRecoveryError(
                    'A Vault writer may still be active',
                    code='writer_marker_recovery_writer_active',
                ) from exc
            self._validate_authority()
        except WriterMarkerRecoveryError:
            raise
        except (OSError, RuntimeError) as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker recovery authority is unavailable',
                code='writer_marker_recovery_unavailable',
            ) from exc

    def _snapshot_marker(self) -> None:
        dir_fd = self._require_dir_fd()
        try:
            listed = os.stat(_PID_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker identity could not be verified',
                code='writer_marker_recovery_path_changed',
            ) from exc
        links = int(listed.st_nlink)
        if not stat.S_ISREG(listed.st_mode) or links not in {1, 2}:
            raise WriterMarkerRecoveryError(
                'Writer marker is not a recoverable regular file',
                code='writer_marker_recovery_marker_unsafe',
            )
        marker_fd: int | None = None
        try:
            marker_fd = os.open(_PID_FILE_NAME, _open_flags(), dir_fd=dir_fd)
            opened = os.fstat(marker_fd)
        except OSError as exc:
            if marker_fd is not None:
                try:
                    os.close(marker_fd)
                except OSError:
                    pass
            raise WriterMarkerRecoveryError(
                'Writer marker identity could not be verified',
                code='writer_marker_recovery_path_changed',
            ) from exc
        self._marker_fd = marker_fd
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != links
            or _identity(opened) != _identity(listed)
        ):
            raise WriterMarkerRecoveryError(
                'Writer marker identity changed',
                code='writer_marker_recovery_path_changed',
            )
        payload = _read_fd_limited(marker_fd, _MAX_PID_BYTES)
        if payload is None:
            raise WriterMarkerRecoveryError(
                'Writer marker content cannot prove an inactive owner',
                code='writer_marker_recovery_owner_unverifiable',
            )
        try:
            decoded = payload.decode('ascii')
        except UnicodeError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker content cannot prove an inactive owner',
                code='writer_marker_recovery_owner_unverifiable',
            ) from exc
        pid_text = decoded[:-1] if re.fullmatch(r'[1-9][0-9]{0,19}\n', decoded) else ''
        pid = _safe_int(pid_text, minimum=1)
        if pid is None or pid > 2_147_483_647:
            raise WriterMarkerRecoveryError(
                'Writer marker content cannot prove an inactive owner',
                code='writer_marker_recovery_owner_unverifiable',
            )
        self._marker_identity = _identity(opened)
        self._marker_payload = payload
        self._marker_pid = pid
        self._marker_links = links

        if links == 2:
            matches: list[str] = []
            try:
                names = os.listdir(dir_fd)
            except OSError as exc:
                raise WriterMarkerRecoveryError(
                    'Interrupted marker publication could not be verified',
                    code='writer_marker_recovery_marker_unsafe',
                ) from exc
            for name in names:
                match = _PID_TEMP_RE.fullmatch(name)
                if match is None or _safe_int(match.group(1), minimum=1) != pid:
                    continue
                try:
                    candidate = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError:
                    continue
                if (
                    stat.S_ISREG(candidate.st_mode)
                    and int(candidate.st_nlink) == 2
                    and _identity(candidate) == self._marker_identity
                ):
                    matches.append(name)
            if len(matches) != 1:
                raise WriterMarkerRecoveryError(
                    'Writer marker has an unrecognized hardlink shape',
                    code='writer_marker_recovery_marker_unsafe',
                )
            self._temp_name = matches[0]
            self._temp_identity = self._marker_identity

    def _snapshot_info(self) -> None:
        dir_fd = self._require_dir_fd()
        try:
            listed = os.stat(_INFO_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer diagnostic identity could not be verified',
                code='writer_marker_recovery_path_changed',
            ) from exc
        links = int(listed.st_nlink)
        if not stat.S_ISREG(listed.st_mode) or links not in {1, 2}:
            raise WriterMarkerRecoveryError(
                'Writer diagnostic is not a recoverable regular file',
                code='writer_marker_recovery_marker_unsafe',
            )
        info_fd: int | None = None
        try:
            info_fd = os.open(_INFO_FILE_NAME, _open_flags(), dir_fd=dir_fd)
            opened = os.fstat(info_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_nlink) != links
                or _identity(opened) != _identity(listed)
            ):
                raise OSError('diagnostic identity changed')
            payload = _read_fd_limited(info_fd, _MAX_INFO_BYTES)
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer diagnostic identity could not be verified',
                code='writer_marker_recovery_path_changed',
            ) from exc
        finally:
            if info_fd is not None:
                try:
                    os.close(info_fd)
                except OSError:
                    pass
        if payload is None:
            raise WriterMarkerRecoveryError(
                'Writer diagnostic cannot be read safely',
                code='writer_marker_recovery_marker_unsafe',
            )
        self._info_identity = _identity(listed)
        self._info_payload = payload
        self._info_links = links

        if links == 2:
            pid = self._marker_pid
            if pid is None:
                raise WriterMarkerRecoveryError(
                    'Writer diagnostic has an unrecognized hardlink shape',
                    code='writer_marker_recovery_marker_unsafe',
                )
            matches: list[str] = []
            try:
                names = os.listdir(dir_fd)
            except OSError as exc:
                raise WriterMarkerRecoveryError(
                    'Interrupted diagnostic publication could not be verified',
                    code='writer_marker_recovery_marker_unsafe',
                ) from exc
            for name in names:
                match = _INFO_TEMP_RE.fullmatch(name)
                if match is None or _safe_int(match.group(1), minimum=1) != pid:
                    continue
                try:
                    candidate = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError:
                    continue
                if (
                    stat.S_ISREG(candidate.st_mode)
                    and int(candidate.st_nlink) == 2
                    and _identity(candidate) == self._info_identity
                ):
                    matches.append(name)
            if len(matches) != 1:
                raise WriterMarkerRecoveryError(
                    'Writer diagnostic has an unrecognized hardlink shape',
                    code='writer_marker_recovery_marker_unsafe',
                )
            self._info_temp_name = matches[0]
            self._info_temp_identity = self._info_identity

    def _decoded_info(self) -> dict[str, Any]:
        payload = self._info_payload
        if payload is None:
            return {}
        try:
            loaded = json.loads(payload.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _prove_marker_owner_inactive(self) -> None:
        pid = self._marker_pid
        if pid is None:
            raise WriterMarkerRecoveryError(
                'Writer marker owner cannot be proven inactive',
                code='writer_marker_recovery_owner_unverifiable',
            )
        if not _pid_running(pid):
            return

        current_birth = _process_start_time(pid)
        if not current_birth:
            raise WriterMarkerRecoveryError(
                'Writer marker owner may still be active',
                code='writer_marker_recovery_writer_active',
            )
        info = self._decoded_info()
        info_pid = _safe_int(info.get('pid'), minimum=1)
        if info_pid == pid:
            birth_hash = info.get('process_birth_hash')
            if (
                _safe_int(info.get('schema'), minimum=1) == 1
                and _is_hex(birth_hash, 64)
                and _is_hex(info.get('owner_nonce'), 32)
                and self._vault_hash is not None
                and info.get('vault_hash') == self._vault_hash
                and hashlib.sha256(current_birth.encode('utf-8')).hexdigest() != birth_hash
            ):
                return
            legacy_birth = info.get('process_start_time')
            if (
                isinstance(legacy_birth, str)
                and 0 < len(legacy_birth) <= 256
                and legacy_birth != current_birth
            ):
                return
        raise WriterMarkerRecoveryError(
            'Writer marker owner may still be active',
            code='writer_marker_recovery_writer_active',
        )

    def _require_dir_fd(self) -> int:
        if self._dir_fd is None:
            raise WriterMarkerRecoveryError(
                'Writer marker recovery is inactive',
                code='writer_marker_recovery_unavailable',
            )
        return self._dir_fd

    def _validate_authority(self) -> None:
        if (
            self._root_fd is None
            or self._dir_fd is None
            or self._lock_fd is None
            or self._root_identity is None
            or self._dir_identity is None
            or self._lock_identity is None
        ):
            raise WriterMarkerRecoveryError(
                'Writer marker recovery authority is inactive',
                code='writer_marker_recovery_unavailable',
            )
        try:
            opened_root = os.fstat(self._root_fd)
            listed_root = os.lstat(self._root_path)
            opened_logs = os.fstat(self._dir_fd)
            listed_logs = os.stat('logs', dir_fd=self._root_fd, follow_symlinks=False)
            opened_lock = os.fstat(self._lock_fd)
            listed_lock = os.stat(_LOCK_FILE_NAME, dir_fd=self._dir_fd, follow_symlinks=False)
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker recovery path identity changed',
                code='writer_marker_recovery_path_changed',
            ) from exc
        if (
            stat.S_ISLNK(listed_root.st_mode)
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
            raise WriterMarkerRecoveryError(
                'Writer marker recovery path identity changed',
                code='writer_marker_recovery_path_changed',
            )

    def _validate_marker(self, *, links: int) -> None:
        if self._marker_fd is None or self._marker_identity is None or self._marker_payload is None:
            raise WriterMarkerRecoveryError(
                'Writer marker identity changed',
                code='writer_marker_recovery_path_changed',
            )
        dir_fd = self._require_dir_fd()
        try:
            opened = os.fstat(self._marker_fd)
            listed = os.stat(_PID_FILE_NAME, dir_fd=dir_fd, follow_symlinks=False)
            os.lseek(self._marker_fd, 0, os.SEEK_SET)
            payload = _read_fd_limited(self._marker_fd, _MAX_PID_BYTES)
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker identity changed',
                code='writer_marker_recovery_path_changed',
            ) from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(listed.st_mode)
            or int(opened.st_nlink) != links
            or int(listed.st_nlink) != links
            or _identity(opened) != self._marker_identity
            or _identity(listed) != self._marker_identity
            or payload != self._marker_payload
        ):
            raise WriterMarkerRecoveryError(
                'Writer marker identity changed',
                code='writer_marker_recovery_path_changed',
            )

    def _validate_marker_absent(self) -> None:
        try:
            os.stat(_PID_FILE_NAME, dir_fd=self._require_dir_fd(), follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker identity could not be verified',
                code='writer_marker_recovery_path_changed',
            ) from exc
        raise WriterMarkerRecoveryError(
            'Writer marker identity changed',
            code='writer_marker_recovery_path_changed',
        )

    def _validate_auxiliary(self, name: str, identity: tuple[int, int] | None, *, links: int = 1) -> None:
        try:
            current = os.stat(name, dir_fd=self._require_dir_fd(), follow_symlinks=False)
        except FileNotFoundError:
            if identity is None:
                return
            raise WriterMarkerRecoveryError(
                'Writer marker recovery snapshot changed',
                code='writer_marker_recovery_path_changed',
            )
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker recovery snapshot changed',
                code='writer_marker_recovery_path_changed',
            ) from exc
        if (
            identity is None
            or not stat.S_ISREG(current.st_mode)
            or int(current.st_nlink) != links
            or _identity(current) != identity
        ):
            raise WriterMarkerRecoveryError(
                'Writer marker recovery snapshot changed',
                code='writer_marker_recovery_path_changed',
            )

    def _validate_all_snapshots(self) -> None:
        self._validate_authority()
        if self._marker_identity is None:
            self._validate_marker_absent()
            return
        self._validate_marker(links=self._marker_links)
        self._validate_auxiliary(
            _INFO_FILE_NAME,
            self._info_identity,
            links=self._info_links or 1,
        )
        if self._info_temp_name is not None:
            self._validate_auxiliary(
                self._info_temp_name,
                self._info_temp_identity,
                links=2,
            )
        if self._temp_name is not None:
            self._validate_auxiliary(self._temp_name, self._temp_identity, links=2)

    def cleanup(self) -> WriterMarkerRecoveryResult:
        """Conditionally remove the exact snapshot, leaving the PID until last."""

        if not self._prepared:
            raise WriterMarkerRecoveryError(
                'Writer marker recovery was not prepared',
                code='writer_marker_recovery_unavailable',
            )
        self._validate_all_snapshots()
        if self._marker_identity is None:
            return WriterMarkerRecoveryResult(
                code='writer_marker_absent',
                recovered=False,
                pid_marker_removed=False,
                info_marker_removed=False,
                temporary_markers_removed=0,
            )

        dir_fd = self._require_dir_fd()
        info_removed = False
        temp_removed = 0
        removed_temp_names: list[str] = []

        if self._info_identity is not None:
            self._validate_authority()
            self._validate_marker(links=self._marker_links)
            self._validate_auxiliary(
                _INFO_FILE_NAME,
                self._info_identity,
                links=self._info_links,
            )
            if self._info_temp_name is not None and self._info_temp_identity is not None:
                self._validate_auxiliary(
                    self._info_temp_name,
                    self._info_temp_identity,
                    links=2,
                )
                if not _unlink_identity_at(
                    dir_fd,
                    self._info_temp_name,
                    self._info_temp_identity,
                    require_regular=False,
                ):
                    raise WriterMarkerRecoveryError(
                        'Interrupted diagnostic publication cleanup was incomplete',
                        code='writer_marker_recovery_cleanup_incomplete',
                    )
                removed_info_temp = self._info_temp_name
                self._info_temp_name = None
                self._info_temp_identity = None
                self._info_links = 1
                self._validate_auxiliary(removed_info_temp, None)
                self._validate_auxiliary(_INFO_FILE_NAME, self._info_identity, links=1)
                temp_removed += 1
                removed_temp_names.append(removed_info_temp)
            if not _unlink_identity_at(
                dir_fd,
                _INFO_FILE_NAME,
                self._info_identity,
                require_regular=True,
            ):
                raise WriterMarkerRecoveryError(
                    'Writer marker cleanup was interrupted safely',
                    code='writer_marker_recovery_cleanup_incomplete',
                )
            self._info_identity = None
            self._info_payload = None
            self._info_links = 0
            self._validate_auxiliary(_INFO_FILE_NAME, None)
            info_removed = True

        if self._temp_name is not None and self._temp_identity is not None:
            self._validate_authority()
            self._validate_marker(links=2)
            self._validate_auxiliary(self._temp_name, self._temp_identity, links=2)
            if not _unlink_identity_at(
                dir_fd,
                self._temp_name,
                self._temp_identity,
                require_regular=False,
            ):
                raise WriterMarkerRecoveryError(
                    'Interrupted marker publication cleanup was incomplete',
                    code='writer_marker_recovery_cleanup_incomplete',
                )
            removed_name = self._temp_name
            self._temp_name = None
            self._temp_identity = None
            self._marker_links = 1
            self._validate_auxiliary(removed_name, None)
            self._validate_marker(links=1)
            temp_removed += 1
            removed_temp_names.append(removed_name)

        self._validate_authority()
        self._validate_auxiliary(_INFO_FILE_NAME, None)
        for removed_name in removed_temp_names:
            self._validate_auxiliary(removed_name, None)
        self._validate_marker(links=1)
        if not _unlink_identity_at(
            dir_fd,
            _PID_FILE_NAME,
            self._marker_identity,
            require_regular=True,
        ):
            raise WriterMarkerRecoveryError(
                'Writer marker cleanup was interrupted safely',
                code='writer_marker_recovery_cleanup_incomplete',
            )
        self._validate_marker_absent()
        self._validate_authority()
        self._validate_auxiliary(_INFO_FILE_NAME, None)
        for removed_name in removed_temp_names:
            self._validate_auxiliary(removed_name, None)
        try:
            _fsync_directory(dir_fd)
        except OSError as exc:
            raise WriterMarkerRecoveryError(
                'Writer marker cleanup durability is uncertain',
                code='writer_marker_recovery_durability_uncertain',
            ) from exc
        self._validate_authority()
        self._validate_marker_absent()
        self._validate_auxiliary(_INFO_FILE_NAME, None)
        for removed_name in removed_temp_names:
            self._validate_auxiliary(removed_name, None)
        return WriterMarkerRecoveryResult(
            code='writer_marker_recovered',
            recovered=True,
            pid_marker_removed=True,
            info_marker_removed=info_removed,
            temporary_markers_removed=temp_removed,
        )

    def close(self) -> None:
        marker_fd = self._marker_fd
        self._marker_fd = None
        if marker_fd is not None:
            try:
                os.close(marker_fd)
            except OSError:
                pass
        lock_fd = self._lock_fd
        self._lock_fd = None
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass
        for attr in ('_dir_fd', '_root_fd'):
            fd = getattr(self, attr)
            setattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _after_fork_in_child(self) -> None:
        """Close inherited descriptions without unlocking the parent flock."""

        for attr in ('_marker_fd', '_lock_fd', '_dir_fd', '_root_fd'):
            fd = getattr(self, attr)
            setattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def _recover_writer_marker_offline(
    cfg: VaultConfig,
    *,
    legacy_writers_stopped: bool,
    claim: Callable[[], None],
) -> WriterMarkerRecoveryResult:
    """Internal sequencing seam: prove, claim approval, revalidate, cleanup."""

    if not callable(claim):
        raise WriterMarkerRecoveryError(
            'Writer marker recovery approval claim is required',
            code='writer_marker_recovery_approval_required',
        )
    recovery = _OfflineWriterMarkerRecovery(
        cfg,
        legacy_writers_stopped=legacy_writers_stopped,
    )
    recovery.prepare()
    try:
        # The application-owned approval is intentionally claimed only after
        # the dead-owner proof, but before the first cleanup mutation.
        claim()
        return recovery.cleanup()
    finally:
        recovery.close()


def _close_inherited_recovery_objects() -> None:
    global _RECOVERY_OBJECTS
    inherited = tuple(_RECOVERY_OBJECTS)
    _RECOVERY_OBJECTS = weakref.WeakSet()
    for recovery in inherited:
        recovery._after_fork_in_child()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_close_inherited_recovery_objects)
