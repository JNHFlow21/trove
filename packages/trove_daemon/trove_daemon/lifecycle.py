from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
from typing import Any

from trove_protocol.capabilities import catalog_snapshot


_HASH_RE = re.compile(r'^[0-9a-f]{64}$')
_UNIX_PATH_LIMIT = 103


class LifecycleError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def require_macos() -> None:
    if sys.platform != 'darwin':
        raise LifecycleError('platform_unsupported', 'The trove/1 local daemon supports macOS only.')


def catalog_identity() -> str:
    payload = json.dumps(
        catalog_snapshot(), ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def build_identity() -> str:
    """Hash the installed runtime code used by both daemon and client."""
    configured = os.environ.get('TROVE_BUILD_HASH')
    if configured is not None:
        if not _HASH_RE.fullmatch(configured):
            raise LifecycleError('build_identity_invalid', 'TROVE_BUILD_HASH must be one lowercase sha256.')
        return configured
    digest = hashlib.sha256(b'trove-build-v1\0')
    package_names = (
        'trove_protocol', 'trove_core', 'trove_client', 'trove_daemon',
        'trove_cli', 'trove_mcp',
    )
    files: dict[str, Path] = {}
    anchor = Path(__file__).resolve()
    installed_root = anchor.parent.parent
    installed = all((installed_root / name).is_dir() for name in package_names)
    repo = anchor.parents[3] if not installed else None
    for package_name in package_names:
        root = (
            installed_root / package_name
            if installed
            else repo / 'packages' / package_name / package_name
        )
        if not root.is_dir():
            continue
        for path in root.rglob('*'):
            if (
                path.is_file() and path.suffix in {'.py', '.json'}
                and '__pycache__' not in path.parts
            ):
                files[path.relative_to(root.parent).as_posix()] = path
    if not files:
        files['trove_daemon/lifecycle.py'] = Path(__file__)
    for relative_text, path in sorted(files.items()):
        relative = relative_text.encode('utf-8')
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, 'big'))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()


def _canonical_vault(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve(strict=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
        raise LifecycleError('vault_identity_unsafe', 'Vault root must be a current-user directory.')
    return path


def _ensure_owner_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise LifecycleError('runtime_metadata_unsafe', 'Daemon runtime directory is unsafe.')
    os.chmod(path, 0o700)


@dataclass(frozen=True)
class RuntimeIdentity:
    vault_root: Path
    vault_identity: str
    build_hash: str
    catalog_hash: str
    runtime_dir: Path
    socket_path: Path
    pid_path: Path
    lock_path: Path
    cursor_dir: Path
    cache_dir: Path

    @classmethod
    def for_vault(
        cls,
        root: str | Path,
        *,
        build_hash: str | None = None,
        catalog_hash: str | None = None,
    ) -> 'RuntimeIdentity':
        canonical = _canonical_vault(root)
        vault_hash = hashlib.sha256(
            b'trove-vault-identity-v1\0' + os.fsencode(canonical),
        ).hexdigest()
        exact_build = build_hash or build_identity()
        exact_catalog = catalog_hash or catalog_identity()
        if not _HASH_RE.fullmatch(exact_build) or not _HASH_RE.fullmatch(exact_catalog):
            raise LifecycleError('runtime_identity_invalid', 'Build and catalog identities must be lowercase sha256 values.')
        candidate = canonical / '.trove-runtime' / vault_hash[:20]
        if len(os.fsencode(candidate / 'daemon.sock')) > _UNIX_PATH_LIMIT:
            # The transport path must not depend on ambient TMPDIR. Agent
            # clients sanitize environments differently; varying this root
            # would create two daemons for one canonical Vault.
            candidate = Path('/tmp') / f'trove-{os.getuid()}' / vault_hash[:20]
        return cls(
            canonical, vault_hash, exact_build, exact_catalog, candidate,
            candidate / 'daemon.sock', candidate / 'daemon.json', candidate / 'autostart.lock',
            candidate / 'cursors', candidate / 'cache',
        )

    def prepare(self) -> None:
        require_macos()
        _ensure_owner_directory(self.runtime_dir.parent)
        _ensure_owner_directory(self.runtime_dir)
        _ensure_owner_directory(self.cursor_dir)
        _ensure_owner_directory(self.cache_dir)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | int(getattr(os, 'O_NOFOLLOW', 0)), 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise LifecycleError('runtime_metadata_unsafe', 'Daemon lock file is unsafe.')
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)

    def write_metadata(self, *, pid: int, restart_id: str) -> None:
        if type(pid) is not int or pid <= 1 or not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', restart_id):
            raise LifecycleError('runtime_metadata_invalid', 'Daemon runtime metadata values are invalid.')
        payload = {
            'format': 'troved-runtime', 'version': 1, 'pid': pid,
            'vault_identity': self.vault_identity, 'build_hash': self.build_hash,
            'catalog_hash': self.catalog_hash, 'restart_id': restart_id,
            'transport': 'unix',
        }
        temporary = self.pid_path.with_suffix(f'.tmp-{os.getpid()}')
        fd: int | None = None
        try:
            fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, 'O_NOFOLLOW', 0)), 0o600,
            )
            encoded = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError('short daemon metadata write')
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(temporary, self.pid_path)
            os.chmod(self.pid_path, 0o600)
        finally:
            if fd is not None:
                os.close(fd)
            temporary.unlink(missing_ok=True)

    def remove_metadata(self) -> None:
        self.pid_path.unlink(missing_ok=True)

    def remove_stale_socket(self) -> None:
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode):
            raise LifecycleError('runtime_socket_unsafe', 'Existing daemon socket path is unsafe.')
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            probe.connect(str(self.socket_path))
        except OSError:
            self.socket_path.unlink(missing_ok=True)
        else:
            raise LifecycleError('daemon_already_running', 'A daemon is already listening for this Vault.')
        finally:
            probe.close()


__all__ = [
    'LifecycleError', 'RuntimeIdentity', 'build_identity', 'catalog_identity',
    'require_macos',
]
