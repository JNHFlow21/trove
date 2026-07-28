from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Iterator, Literal
import weakref

from trove_core.vault.config import VaultConfig


GenerationFileIdentity = tuple[str, int, int, int, int]
_PUBLISH_MARKER = ".trove-generation-publish.json"
_FIXTURE_PENDING = ".trove-fixture-generation-state.json"
_VECTOR_PENDING = "vectors/zvec/messages.trove-swap.json"
_OPERATION_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MAX_MARKER_BYTES = 1024


class VaultGenerationUnavailable(RuntimeError):
    """Raised when a Vault root cannot be bound for a generation lease."""

    def __init__(self, message: str, *, code: str = "vault_generation_unavailable"):
        super().__init__(message)
        self.code = code


def generation_unavailable_payload(error: VaultGenerationUnavailable) -> dict[str, object]:
    """Return a redacted, actionable adapter contract for blocked reads."""

    return {
        "error": {
            "code": error.code,
            "message": "Vault generation is unavailable until the interrupted publication is recovered.",
            "action": "retry_the_same_mutation_then_retry_the_read",
        },
        "raw_content_included": False,
    }


@dataclass(frozen=True, slots=True)
class VaultGenerationToken:
    """A redacted identity for every artifact used by the search runtime.

    The token intentionally contains only inode/mtime/size metadata.  It never
    reads Vault content.  It is captured while the root generation lease is
    held, so a destructive publisher cannot produce a torn token.
    """

    root_identity: tuple[int, int]
    sqlite: GenerationFileIdentity
    sqlite_wal: GenerationFileIdentity
    sqlite_shm: GenerationFileIdentity
    fixture_ready: GenerationFileIdentity
    vector_collection: GenerationFileIdentity
    vector_metadata: GenerationFileIdentity
    vector_progress: GenerationFileIdentity
    publish_pending: GenerationFileIdentity
    fixture_pending: GenerationFileIdentity
    vector_pending: GenerationFileIdentity

    def cache_key(self) -> tuple[object, ...]:
        return (
            "vault-generation-v1",
            self.root_identity,
            self.sqlite,
            self.sqlite_wal,
            # SHM contains SQLite reader coordination, not durable data.  A
            # reader may update it, so it must not self-invalidate the cache.
            self.fixture_ready,
            self.vector_collection,
            self.vector_metadata,
            self.vector_progress,
            self.publish_pending,
            self.fixture_pending,
            self.vector_pending,
        )

    @property
    def recovery_required(self) -> bool:
        return any(
            item[0] != "missing"
            for item in (self.publish_pending, self.fixture_pending, self.vector_pending)
        )


def _identity(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _file_identity(root_fd: int, relative: str) -> GenerationFileIdentity:
    try:
        info = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return ("missing", 0, 0, 0, 0)
    except OSError as exc:
        return (f"error:{int(exc.errno or 0)}", 0, 0, 0, 0)
    if stat.S_ISREG(info.st_mode):
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    elif stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    else:
        kind = "other"
    return (
        kind,
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mtime_ns),
        int(info.st_size),
    )


def _token(root_fd: int) -> VaultGenerationToken:
    root = os.fstat(root_fd)
    return VaultGenerationToken(
        root_identity=_identity(root),
        sqlite=_file_identity(root_fd, "index/trove.sqlite"),
        sqlite_wal=_file_identity(root_fd, "index/trove.sqlite-wal"),
        sqlite_shm=_file_identity(root_fd, "index/trove.sqlite-shm"),
        fixture_ready=_file_identity(root_fd, ".trove-fixture-ready.json"),
        vector_collection=_file_identity(root_fd, "vectors/zvec/messages"),
        vector_metadata=_file_identity(root_fd, "vectors/zvec/messages.trove-meta.json"),
        vector_progress=_file_identity(root_fd, "vectors/zvec/messages.trove-progress.json"),
        publish_pending=_file_identity(root_fd, _PUBLISH_MARKER),
        fixture_pending=_file_identity(root_fd, _FIXTURE_PENDING),
        vector_pending=_file_identity(root_fd, _VECTOR_PENDING),
    )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short generation marker write")
        view = view[written:]


def _read_publish_marker(root_fd: int) -> tuple[str, tuple[int, int]] | None:
    try:
        listed = os.stat(_PUBLISH_MARKER, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise VaultGenerationUnavailable(
            "Generation recovery marker is unavailable",
            code="vault_generation_marker_unsafe",
        ) from exc
    if not stat.S_ISREG(listed.st_mode) or int(listed.st_nlink) != 1 or listed.st_mode & 0o077:
        raise VaultGenerationUnavailable(
            "Generation recovery marker is unsafe",
            code="vault_generation_marker_unsafe",
        )
    flags = os.O_RDONLY | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(_PUBLISH_MARKER, flags, dir_fd=root_fd)
    except OSError as exc:
        raise VaultGenerationUnavailable(
            "Generation recovery marker is unavailable",
            code="vault_generation_marker_unsafe",
        ) from exc
    try:
        opened = os.fstat(fd)
        if _identity(opened) != _identity(listed):
            raise VaultGenerationUnavailable(
                "Generation recovery marker identity changed",
                code="vault_generation_marker_unsafe",
            )
        raw = os.read(fd, _MAX_MARKER_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _MAX_MARKER_BYTES:
        raise VaultGenerationUnavailable(
            "Generation recovery marker is invalid",
            code="vault_generation_marker_unsafe",
        )
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VaultGenerationUnavailable(
            "Generation recovery marker is invalid",
            code="vault_generation_marker_unsafe",
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format", "version", "operation", "nonce"}
        or payload.get("format") != "trove-generation-publish"
        or payload.get("version") != 1
        or not isinstance(payload.get("operation"), str)
        or not _OPERATION_RE.fullmatch(payload["operation"])
        or not isinstance(payload.get("nonce"), str)
        or len(payload["nonce"]) != 32
    ):
        raise VaultGenerationUnavailable(
            "Generation recovery marker is invalid",
            code="vault_generation_marker_unsafe",
        )
    try:
        int(payload["nonce"], 16)
    except ValueError as exc:
        raise VaultGenerationUnavailable(
            "Generation recovery marker is invalid",
            code="vault_generation_marker_unsafe",
        ) from exc
    return payload["operation"], _identity(opened)


def _prepare_publish_marker(root_fd: int, operation: str) -> tuple[int, int]:
    if not isinstance(operation, str) or not _OPERATION_RE.fullmatch(operation):
        raise ValueError("generation publication operation must be a short lowercase identifier")
    existing = _read_publish_marker(root_fd)
    if existing is not None:
        prior_operation, identity = existing
        if prior_operation != operation:
            raise VaultGenerationUnavailable(
                f"Generation recovery requires retrying {prior_operation}",
                code="vault_generation_recovery_required",
            )
        return identity

    nonce = secrets.token_hex(16)
    payload = (
        json.dumps(
            {
                "format": "trove-generation-publish",
                "nonce": nonce,
                "operation": operation,
                "version": 1,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    temporary = f".{_PUBLISH_MARKER}.tmp-{os.getpid()}-{nonce}"
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
            dir_fd=root_fd,
        )
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, _PUBLISH_MARKER, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
        created = _read_publish_marker(root_fd)
        if created is None or created[0] != operation:
            raise VaultGenerationUnavailable(
                "Generation recovery marker publication failed",
                code="vault_generation_marker_unsafe",
            )
        return created[1]
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except OSError:
            pass


def _clear_publish_marker(root_fd: int, expected: tuple[int, int]) -> None:
    marker = _read_publish_marker(root_fd)
    if marker is None:
        raise VaultGenerationUnavailable(
            "Generation recovery marker disappeared",
            code="vault_generation_marker_unsafe",
        )
    _, current = marker
    if current != expected:
        raise VaultGenerationUnavailable(
            "Generation recovery marker identity changed",
            code="vault_generation_marker_unsafe",
        )
    os.unlink(_PUBLISH_MARKER, dir_fd=root_fd)
    os.fsync(root_fd)


def _open_bound_root(cfg: VaultConfig) -> int:
    cfg.validate_runtime_path()
    root = Path(os.path.abspath(os.path.expanduser(str(cfg.root))))
    try:
        listed = os.lstat(root)
    except FileNotFoundError as exc:
        raise VaultGenerationUnavailable(
            "Vault root does not exist",
            code="vault_generation_root_missing",
        ) from exc
    except OSError as exc:
        raise VaultGenerationUnavailable("Vault root is unavailable") from exc
    if stat.S_ISLNK(listed.st_mode) or not stat.S_ISDIR(listed.st_mode):
        raise VaultGenerationUnavailable(
            "Vault root is not a real directory",
            code="vault_generation_root_unsafe",
        )
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(root, flags)
    except OSError as exc:
        code = "vault_generation_root_unsafe" if exc.errno in {errno.ELOOP, errno.EMLINK} else "vault_generation_unavailable"
        raise VaultGenerationUnavailable("Vault root could not be opened", code=code) from exc
    opened = os.fstat(fd)
    if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(listed):
        os.close(fd)
        raise VaultGenerationUnavailable(
            "Vault root identity changed",
            code="vault_generation_root_changed",
        )
    return fd


def _validate_bound_root(cfg: VaultConfig, fd: int) -> None:
    root = Path(os.path.abspath(os.path.expanduser(str(cfg.root))))
    try:
        opened = os.fstat(fd)
        listed = os.lstat(root)
    except OSError as exc:
        raise VaultGenerationUnavailable(
            "Vault root changed while waiting for a generation lease",
            code="vault_generation_root_changed",
        ) from exc
    if (
        stat.S_ISLNK(listed.st_mode)
        or not stat.S_ISDIR(listed.st_mode)
        or _identity(opened) != _identity(listed)
    ):
        raise VaultGenerationUnavailable(
            "Vault root changed while waiting for a generation lease",
            code="vault_generation_root_changed",
        )


_LEASES: weakref.WeakSet["VaultGenerationLease"] = weakref.WeakSet()
_LEASES_LOCK = threading.Lock()
_ACTIVE_PUBLICATIONS: ContextVar[tuple[tuple[str, "VaultGenerationLease"], ...]] = ContextVar(
    "trove_active_generation_publications",
    default=(),
)
_ACTIVE_READS: ContextVar[tuple[tuple[str, "VaultGenerationLease"], ...]] = ContextVar(
    "trove_active_generation_reads",
    default=(),
)


def _vault_key(cfg: VaultConfig) -> str:
    return str(Path(os.path.abspath(os.path.expanduser(str(cfg.root)))))


class VaultGenerationLease:
    """Cross-process lease on the immutable Vault root inode.

    Readers take a shared lock for the complete logical read.  A publisher
    takes an exclusive lock only for the destructive/publication window.  The
    kernel releases the lease after crashes, so retry never depends on a stale
    PID marker.
    """

    def __init__(self, vault: VaultConfig | str | Path, *, mode: Literal["read", "publish"] = "read"):
        self.cfg = vault if isinstance(vault, VaultConfig) else VaultConfig.resolve(str(vault), env={})
        if mode not in {"read", "publish"}:
            raise ValueError("generation lease mode must be read or publish")
        self.mode = mode
        self._fd: int | None = None
        self._owner_pid: int | None = None
        self._token: VaultGenerationToken | None = None
        self._publish_marker_identity: tuple[int, int] | None = None
        self._consistent_on_error = False
        with _LEASES_LOCK:
            _LEASES.add(self)

    @property
    def active(self) -> bool:
        return self._fd is not None and self._owner_pid == os.getpid()

    @property
    def token(self) -> VaultGenerationToken:
        if not self.active or self._token is None:
            raise VaultGenerationUnavailable(
                "Vault generation lease is inactive",
                code="vault_generation_lease_inactive",
            )
        return self._token

    def acquire(self) -> "VaultGenerationLease":
        if self._fd is not None:
            raise VaultGenerationUnavailable(
                "Vault generation lease is already active",
                code="vault_generation_lease_active",
            )
        fd = _open_bound_root(self.cfg)
        try:
            operation = fcntl.LOCK_SH if self.mode == "read" else fcntl.LOCK_EX
            # Publication waits for every old-generation reader.  New readers
            # then wait for publication and can only capture the complete new
            # token.  The stable writer lock still serializes publishers.
            fcntl.flock(fd, operation)
            _validate_bound_root(self.cfg, fd)
            token = _token(fd)
            if self.mode == "read" and token.recovery_required:
                raise VaultGenerationUnavailable(
                    "Vault generation requires writer recovery before reads resume",
                    code="vault_generation_recovery_required",
                )
            self._fd = fd
            self._owner_pid = os.getpid()
            self._token = token
        except BaseException:
            os.close(fd)
            raise
        return self

    def refresh_token(self) -> VaultGenerationToken:
        if not self.active or self._fd is None:
            return self.token
        self._token = _token(self._fd)
        return self._token

    def begin_publication(self, operation: str) -> None:
        if self.mode != "publish" or not self.active or self._fd is None:
            raise VaultGenerationUnavailable(
                "An exclusive generation lease is required",
                code="vault_generation_lease_inactive",
            )
        if self._publish_marker_identity is not None:
            raise VaultGenerationUnavailable(
                "Generation publication is already active",
                code="vault_generation_publication_active",
            )
        self._publish_marker_identity = _prepare_publish_marker(self._fd, operation)
        self._consistent_on_error = False
        self.refresh_token()

    def mark_consistent(self) -> None:
        """Declare that caller recovery restored a complete old/new state."""

        if self._publish_marker_identity is None:
            raise VaultGenerationUnavailable(
                "Generation publication is not active",
                code="vault_generation_publication_inactive",
            )
        self._consistent_on_error = True

    def finish_publication(self) -> None:
        if self._publish_marker_identity is None or self._fd is None:
            return
        identity = self._publish_marker_identity
        _clear_publish_marker(self._fd, identity)
        self._publish_marker_identity = None
        self._consistent_on_error = False
        self.refresh_token()

    def release(self) -> None:
        fd = self._fd
        owner_pid = self._owner_pid
        self._fd = None
        self._owner_pid = None
        self._token = None
        self._publish_marker_identity = None
        self._consistent_on_error = False
        if fd is None:
            return
        # A child must only close an inherited descriptor.  LOCK_UN on the
        # duplicated open-file description could release the parent's lease.
        if owner_pid == os.getpid():
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass

    def _after_fork_in_child(self) -> None:
        self.release()

    def __enter__(self) -> "VaultGenerationLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False


def _after_fork_in_child() -> None:
    global _LEASES, _LEASES_LOCK
    inherited = tuple(_LEASES)
    _LEASES = weakref.WeakSet()
    _LEASES_LOCK = threading.Lock()
    _ACTIVE_PUBLICATIONS.set(())
    _ACTIVE_READS.set(())
    for lease in inherited:
        lease._after_fork_in_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


@contextmanager
def vault_generation_read(vault: VaultConfig | str | Path) -> Iterator[VaultGenerationToken]:
    """Hold one shared generation lease for a complete, safely nested read.

    Query adapters and the search runtime deliberately overlap this boundary.
    Reusing the active lease avoids opening a second ``flock`` descriptor and,
    more importantly, guarantees that an inner helper cannot release the
    outer logical read's publication barrier.
    """

    cfg = vault if isinstance(vault, VaultConfig) else VaultConfig.resolve(str(vault), env={})
    key = _vault_key(cfg)
    active = _ACTIVE_READS.get()
    for active_key, active_lease in reversed(active):
        if active_key == key and active_lease.active:
            yield active_lease.token
            return
    with VaultGenerationLease(cfg, mode="read") as lease:
        reset = _ACTIVE_READS.set((*active, (key, lease)))
        try:
            yield lease.token
        finally:
            _ACTIVE_READS.reset(reset)


@contextmanager
def vault_generation_publish(
    vault: VaultConfig | str | Path,
    *,
    operation: str = "generation-publish",
) -> Iterator[VaultGenerationLease]:
    with VaultGenerationLease(vault, mode="publish") as lease:
        lease.begin_publication(operation)
        try:
            yield lease
        except BaseException:
            if lease._consistent_on_error:
                lease.finish_publication()
            raise
        else:
            lease.finish_publication()


@contextmanager
def coordinated_vault_generation_publish(
    vault: VaultConfig | str | Path,
    *,
    operation: str,
) -> Iterator[VaultGenerationLease]:
    """Reuse an explicit outer publication lease for nested mutation slices."""

    cfg = vault if isinstance(vault, VaultConfig) else VaultConfig.resolve(str(vault), env={})
    key = _vault_key(cfg)
    active = _ACTIVE_PUBLICATIONS.get()
    for active_key, active_lease in reversed(active):
        if active_key == key:
            yield active_lease
            return
    with vault_generation_publish(cfg, operation=operation) as lease:
        reset = _ACTIVE_PUBLICATIONS.set((*active, (key, lease)))
        try:
            yield lease
        finally:
            _ACTIVE_PUBLICATIONS.reset(reset)
