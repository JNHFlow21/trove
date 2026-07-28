from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import tempfile
from typing import Iterator


FIXTURE_MARKER_NAME = ".trove-fixture-vault.json"
_FIXTURE_MARKER = {
    "format": "trove-fixture-vault",
    "generator": "trove",
    "version": 1,
}
FIXTURE_MARKER_BYTES = (
    json.dumps(_FIXTURE_MARKER, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
).encode("ascii")
_LEGACY_GENERATOR = "trove" + "-wechat"
_LEGACY_FIXTURE_MARKER_BYTES = (
    json.dumps(
        {**_FIXTURE_MARKER, "generator": _LEGACY_GENERATOR},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode("ascii")
FIXTURE_READY_NAME = ".trove-fixture-ready.json"
FIXTURE_GENERATION_STATE_NAME = ".trove-fixture-generation-state.json"
_PRODUCT_DIRECTORIES = ("index", "api", "logs", "manifests", "jobs", "sources", "proof")


class FixtureVaultGuardError(RuntimeError):
    """A fixture mutation was rejected before product data was touched."""

    code = "fixture_vault_guard_rejected"

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__("Fixture Vault mutation refused by the safety marker guard.")


def fixture_vault_guard_error_payload(exc: FixtureVaultGuardError) -> dict:
    return {
        "error": {
            "code": exc.code,
            "reason_code": exc.reason_code,
            "message": "Fixture Vault mutation refused by the safety marker guard.",
            "action": "use_a_dedicated_empty_fixture_vault",
        }
    }


def normalize_fixture_root(path: Path) -> Path:
    """Normalize a fixture root lexically without following symlinks."""

    expanded = Path(path).expanduser()
    if ".." in expanded.parts:
        raise FixtureVaultGuardError("fixture_path_contains_parent_traversal")
    return Path(os.path.abspath(os.fspath(expanded)))


def _no_follow_flag() -> int:
    return int(getattr(os, "O_NOFOLLOW", 0))


def _directory_flag() -> int:
    return int(getattr(os, "O_DIRECTORY", 0))


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


@dataclass(frozen=True, slots=True)
class FixtureReadyGeneration:
    identity: tuple[int, int]
    nonce: str
    sqlite_sha256: str
    sqlite_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FixtureVaultIdentity:
    root: tuple[int, int]
    marker: tuple[int, int]
    ready: FixtureReadyGeneration | None = None

    @property
    def provisional(self) -> bool:
        return self.ready is None


def _reject_untrusted_path_symlinks(root: Path) -> None:
    """Reject caller-controlled symlink components without breaking macOS /var."""

    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_path_unavailable") from exc
        if not stat.S_ISLNK(info.st_mode):
            continue
        if current == root:
            raise FixtureVaultGuardError("fixture_root_is_symlink")
        try:
            parent_info = os.stat(current.parent, follow_symlinks=False)
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_path_contains_symlink") from exc
        trusted_platform_alias = (
            hasattr(info, "st_uid")
            and info.st_uid == 0
            and parent_info.st_uid == 0
            and not (parent_info.st_mode & 0o022)
        )
        if not trusted_platform_alias:
            raise FixtureVaultGuardError("fixture_path_contains_symlink")


def _open_root(root: Path, *, allow_create: bool) -> int:
    _reject_untrusted_path_symlinks(root)
    try:
        before = os.lstat(root)
    except FileNotFoundError:
        if not allow_create:
            raise FixtureVaultGuardError("fixture_root_unavailable")
        try:
            root.mkdir(parents=True, exist_ok=False, mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_root_unavailable") from exc
        _reject_untrusted_path_symlinks(root)
        try:
            before = os.lstat(root)
        except FileNotFoundError as exc:
            raise FixtureVaultGuardError("fixture_root_unavailable") from exc
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_root_unavailable") from exc
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_root_unavailable") from exc

    if stat.S_ISLNK(before.st_mode):
        raise FixtureVaultGuardError("fixture_root_is_symlink")
    if not stat.S_ISDIR(before.st_mode):
        raise FixtureVaultGuardError("fixture_root_is_not_directory")

    try:
        fd = os.open(root, os.O_RDONLY | _directory_flag() | _no_follow_flag())
    except OSError as exc:
        reason = "fixture_root_is_symlink" if exc.errno in {errno.ELOOP, errno.EMLINK} else "fixture_root_unavailable"
        raise FixtureVaultGuardError(reason) from exc
    after = os.fstat(fd)
    if _identity(before) != _identity(after):
        os.close(fd)
        raise FixtureVaultGuardError("fixture_root_changed_during_validation")
    return fd


def _marker_stat(root_fd: int) -> os.stat_result | None:
    try:
        return os.stat(FIXTURE_MARKER_NAME, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_marker_unreadable") from exc


def _list_names(root_fd: int) -> list[str]:
    try:
        return os.listdir(root_fd)
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_tree_unreadable") from exc


def _read_and_validate_marker(root_fd: int) -> os.stat_result:
    marker_stat = _marker_stat(root_fd)
    if marker_stat is None:
        raise FixtureVaultGuardError("fixture_marker_missing")
    if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_nlink != 1:
        raise FixtureVaultGuardError("fixture_marker_invalid_type")
    if marker_stat.st_mode & 0o077:
        raise FixtureVaultGuardError("fixture_marker_permissions_unsafe")
    try:
        marker_fd = os.open(FIXTURE_MARKER_NAME, os.O_RDONLY | _no_follow_flag(), dir_fd=root_fd)
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_marker_unreadable") from exc
    try:
        opened = os.fstat(marker_fd)
        if _identity(marker_stat) != _identity(opened):
            raise FixtureVaultGuardError("fixture_marker_changed_during_validation")
        data = os.read(
            marker_fd,
            max(len(FIXTURE_MARKER_BYTES), len(_LEGACY_FIXTURE_MARKER_BYTES)) + 1,
        )
    finally:
        os.close(marker_fd)
    if data not in {FIXTURE_MARKER_BYTES, _LEGACY_FIXTURE_MARKER_BYTES}:
        raise FixtureVaultGuardError("fixture_marker_invalid_content")
    return marker_stat


def _ready_stat(root_fd: int) -> os.stat_result | None:
    try:
        return os.stat(FIXTURE_READY_NAME, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_ready_marker_unreadable") from exc


def _read_and_validate_ready(root_fd: int) -> FixtureReadyGeneration | None:
    ready_stat = _ready_stat(root_fd)
    if ready_stat is None:
        return None
    if not stat.S_ISREG(ready_stat.st_mode) or ready_stat.st_nlink != 1:
        raise FixtureVaultGuardError("fixture_ready_marker_invalid_type")
    if ready_stat.st_mode & 0o077:
        raise FixtureVaultGuardError("fixture_ready_marker_permissions_unsafe")
    try:
        ready_fd = os.open(FIXTURE_READY_NAME, os.O_RDONLY | _no_follow_flag(), dir_fd=root_fd)
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_ready_marker_unreadable") from exc
    try:
        opened = os.fstat(ready_fd)
        if _identity(ready_stat) != _identity(opened):
            raise FixtureVaultGuardError("fixture_ready_marker_changed_during_validation")
        raw = bytearray()
        while True:
            chunk = os.read(ready_fd, 4096)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > 4096:
                raise FixtureVaultGuardError("fixture_ready_marker_invalid_content")
    finally:
        os.close(ready_fd)
    try:
        data = json.loads(bytes(raw).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureVaultGuardError("fixture_ready_marker_invalid_content") from exc
    expected_keys = {
        "format",
        "generator",
        "version",
        "generation_nonce",
        "sqlite_sha256",
        "sqlite_device",
        "sqlite_inode",
    }
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise FixtureVaultGuardError("fixture_ready_marker_invalid_content")
    nonce = data.get("generation_nonce")
    sqlite_sha256 = data.get("sqlite_sha256")
    sqlite_device = data.get("sqlite_device")
    sqlite_inode = data.get("sqlite_inode")
    if (
        data.get("format") != "trove-fixture-ready"
        or data.get("generator") not in {"trove", _LEGACY_GENERATOR}
        or data.get("version") != 2
        or not isinstance(nonce, str)
        or len(nonce) != 32
        or not isinstance(sqlite_sha256, str)
        or len(sqlite_sha256) != 64
        or not isinstance(sqlite_device, int)
        or isinstance(sqlite_device, bool)
        or sqlite_device < 0
        or not isinstance(sqlite_inode, int)
        or isinstance(sqlite_inode, bool)
        or sqlite_inode <= 0
    ):
        raise FixtureVaultGuardError("fixture_ready_marker_invalid_content")
    try:
        int(nonce, 16)
        int(sqlite_sha256, 16)
    except ValueError as exc:
        raise FixtureVaultGuardError("fixture_ready_marker_invalid_content") from exc
    return FixtureReadyGeneration(
        _identity(ready_stat),
        nonce,
        sqlite_sha256,
        (sqlite_device, sqlite_inode),
    )


def _ready_bytes(
    *,
    nonce: str,
    sqlite_sha256: str,
    sqlite_identity: tuple[int, int],
) -> bytes:
    return (
        json.dumps(
            {
                "format": "trove-fixture-ready",
                "generator": "trove",
                "version": 2,
                "generation_nonce": nonce,
                "sqlite_sha256": sqlite_sha256,
                "sqlite_device": sqlite_identity[0],
                "sqlite_inode": sqlite_identity[1],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


@dataclass(slots=True)
class _FixedPublication:
    fd: int
    temp_name: str
    identity: tuple[int, int]
    _active: bool = True

    def finalize(self, root_fd: int) -> None:
        if not self._active:
            return
        self._active = False
        os.close(self.fd)
        os.unlink(self.temp_name, dir_fd=root_fd)
        os.fsync(root_fd)

    def revoke(self, root_fd: int) -> None:
        """Invalidate only the inode created by this publication.

        The retained file descriptor is the authority. We intentionally never
        stat-then-unlink the public marker name, so a replacement marker cannot
        be deleted by this rollback path.
        """

        if not self._active:
            return
        self._active = False
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            os.ftruncate(self.fd, 0)
            os.write(self.fd, b"TROVE_FIXTURE_CLAIM_REVOKED\n")
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            try:
                os.unlink(self.temp_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
            os.fsync(root_fd)


def _write_fixed_file_once(
    root_fd: int,
    *,
    name: str,
    data: bytes,
    validate_existing,
) -> _FixedPublication | None:
    temp_name = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    temp_fd: int | None = None
    try:
        temp_fd = os.open(
            temp_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
            0o600,
            dir_fd=root_fd,
        )
        os.fchmod(temp_fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError("short write while publishing fixture marker")
            view = view[written:]
        os.fsync(temp_fd)
        publication = _FixedPublication(temp_fd, temp_name, _identity(os.fstat(temp_fd)))
        try:
            os.link(
                temp_name,
                name,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            os.close(temp_fd)
            temp_fd = None
            os.unlink(temp_name, dir_fd=root_fd)
            os.fsync(root_fd)
            validate_existing(root_fd)
            return None
        os.fsync(root_fd)
        temp_fd = None  # ownership moved to publication
        return publication
    except FixtureVaultGuardError:
        raise
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_marker_create_failed") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
            try:
                os.unlink(temp_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass


def _write_marker_once(root_fd: int) -> _FixedPublication | None:
    return _write_fixed_file_once(
        root_fd,
        name=FIXTURE_MARKER_NAME,
        data=FIXTURE_MARKER_BYTES,
        validate_existing=_read_and_validate_marker,
    )


def _hash_fd(fd: int) -> str:
    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.lseek(fd, position, os.SEEK_SET)


def _verify_sqlite_fd(fd: int, expected_sha256: str) -> None:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
        raise FixtureVaultGuardError("fixture_generation_sqlite_invalid")
    if _hash_fd(fd) != expected_sha256:
        raise FixtureVaultGuardError("fixture_generation_hash_mismatch")
    with tempfile.TemporaryDirectory(prefix="trove-fixture-verify-") as directory:
        copy_path = Path(directory) / "generation.sqlite"
        copied_digest = hashlib.sha256()
        with copy_path.open("wb") as handle:
            position = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    copied_digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                os.lseek(fd, position, os.SEEK_SET)
        after = os.fstat(fd)
        if _identity(before) != _identity(after) or before.st_size != after.st_size:
            raise FixtureVaultGuardError("fixture_generation_changed_during_verification")
        if copied_digest.hexdigest() != expected_sha256 or _hash_fd(fd) != expected_sha256:
            raise FixtureVaultGuardError("fixture_generation_changed_during_verification")
        try:
            conn = sqlite3.connect(f"file:{copy_path}?mode=ro", uri=True)
            try:
                rows = conn.execute("PRAGMA integrity_check").fetchall()
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise FixtureVaultGuardError("fixture_generation_integrity_failed") from exc
        if rows != [("ok",)]:
            raise FixtureVaultGuardError("fixture_generation_integrity_failed")


def _open_index_dir(root_fd: int) -> int:
    try:
        listed = os.stat("index", dir_fd=root_fd, follow_symlinks=False)
        index_fd = os.open(
            "index",
            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
            dir_fd=root_fd,
        )
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_index_unavailable") from exc
    if not stat.S_ISDIR(listed.st_mode) or _identity(listed) != _identity(os.fstat(index_fd)):
        os.close(index_fd)
        raise FixtureVaultGuardError("fixture_index_identity_changed")
    return index_fd


def _open_regular_at(directory_fd: int, name: str) -> int | None:
    try:
        listed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(listed.st_mode) or listed.st_nlink < 1:
        raise FixtureVaultGuardError("fixture_generation_file_invalid")
    try:
        fd = os.open(name, os.O_RDONLY | _no_follow_flag(), dir_fd=directory_fd)
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_file_unavailable") from exc
    opened = os.fstat(fd)
    if _identity(opened) != _identity(listed):
        os.close(fd)
        raise FixtureVaultGuardError("fixture_generation_file_changed")
    return fd


def _write_json_at(root_fd: int, name: str, payload: dict) -> tuple[int, int]:
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    temp_name = f".{name}.{secrets.token_hex(12)}.tmp"
    temp_fd: int | None = None
    try:
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
            0o600,
            dir_fd=root_fd,
        )
        os.fchmod(temp_fd, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError("short generation metadata write")
            view = view[written:]
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = None
        os.replace(temp_name, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise FixtureVaultGuardError("fixture_generation_metadata_invalid")
        return _identity(current)
    except FixtureVaultGuardError:
        raise
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_metadata_write_failed") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class FixtureGenerationState:
    nonce: str
    phase: str
    new_sha256: str
    old_sha256: str | None
    old_ready_nonce: str | None

    @property
    def candidate_name(self) -> str:
        return f".trove.sqlite.fixture-{self.nonce}.candidate"

    @property
    def anchor_name(self) -> str:
        return f".trove.sqlite.fixture-{self.nonce}.anchor"

    @property
    def previous_name(self) -> str:
        return f".trove.sqlite.fixture-{self.nonce}.previous"

    def to_payload(self) -> dict:
        return {
            "format": "trove-fixture-generation-state",
            "version": 1,
            "nonce": self.nonce,
            "phase": self.phase,
            "new_sha256": self.new_sha256,
            "old_sha256": self.old_sha256,
            "old_ready_nonce": self.old_ready_nonce,
        }


def _read_generation_state(root_fd: int) -> FixtureGenerationState | None:
    try:
        listed = os.stat(FIXTURE_GENERATION_STATE_NAME, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(listed.st_mode) or listed.st_nlink != 1 or listed.st_mode & 0o077:
        raise FixtureVaultGuardError("fixture_generation_state_invalid")
    fd = os.open(FIXTURE_GENERATION_STATE_NAME, os.O_RDONLY | _no_follow_flag(), dir_fd=root_fd)
    try:
        if _identity(os.fstat(fd)) != _identity(listed):
            raise FixtureVaultGuardError("fixture_generation_state_changed")
        raw = bytearray()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > 8192:
                raise FixtureVaultGuardError("fixture_generation_state_invalid")
    finally:
        os.close(fd)
    try:
        data = json.loads(bytes(raw).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureVaultGuardError("fixture_generation_state_invalid") from exc
    keys = {"format", "version", "nonce", "phase", "new_sha256", "old_sha256", "old_ready_nonce"}
    if not isinstance(data, dict) or set(data) != keys:
        raise FixtureVaultGuardError("fixture_generation_state_invalid")
    nonce = data.get("nonce")
    phase = data.get("phase")
    new_sha256 = data.get("new_sha256")
    old_sha256 = data.get("old_sha256")
    old_ready_nonce = data.get("old_ready_nonce")
    if (
        data.get("format") != "trove-fixture-generation-state"
        or data.get("version") != 1
        or not isinstance(nonce, str)
        or len(nonce) != 32
        or phase not in {"preparing", "prepared", "swapped", "ready"}
        or not isinstance(new_sha256, str)
        or len(new_sha256) != 64
        or (old_sha256 is not None and (not isinstance(old_sha256, str) or len(old_sha256) != 64))
        or (old_ready_nonce is not None and (not isinstance(old_ready_nonce, str) or len(old_ready_nonce) != 32))
    ):
        raise FixtureVaultGuardError("fixture_generation_state_invalid")
    try:
        int(nonce, 16)
        int(new_sha256, 16)
        if old_sha256 is not None:
            int(old_sha256, 16)
        if old_ready_nonce is not None:
            int(old_ready_nonce, 16)
    except ValueError as exc:
        raise FixtureVaultGuardError("fixture_generation_state_invalid") from exc
    return FixtureGenerationState(nonce, phase, new_sha256, old_sha256, old_ready_nonce)


def _write_generation_state(root_fd: int, state: FixtureGenerationState) -> None:
    _write_json_at(root_fd, FIXTURE_GENERATION_STATE_NAME, state.to_payload())


@dataclass(slots=True)
class FixturePublishedArtifact:
    fd: int
    identity: tuple[int, int]
    sha256: str
    nonce: str
    reused: bool = False
    _closed: bool = False

    def verify(self, root_fd: int, *, integrity: bool = False) -> None:
        if self._closed:
            raise FixtureVaultGuardError("fixture_published_artifact_closed")
        held = os.fstat(self.fd)
        if _identity(held) != self.identity or _hash_fd(self.fd) != self.sha256:
            raise FixtureVaultGuardError("fixture_published_artifact_changed")
        index_fd = _open_index_dir(root_fd)
        try:
            current_fd = _open_regular_at(index_fd, "trove.sqlite")
            if current_fd is None:
                raise FixtureVaultGuardError("fixture_published_artifact_missing")
            try:
                if _identity(os.fstat(current_fd)) != self.identity or _hash_fd(current_fd) != self.sha256:
                    raise FixtureVaultGuardError("fixture_published_artifact_replaced")
                if integrity:
                    _verify_sqlite_fd(current_fd, self.sha256)
            finally:
                os.close(current_fd)
        finally:
            os.close(index_fd)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.fd)


@dataclass(slots=True)
class _ReadyPublication:
    fd: int
    identity: tuple[int, int]
    nonce: str
    sqlite_sha256: str
    sqlite_identity: tuple[int, int]
    _closed: bool = False

    def verify(self, root_fd: int) -> FixtureReadyGeneration:
        if self._closed:
            raise FixtureVaultGuardError("fixture_ready_publication_closed")
        held = os.fstat(self.fd)
        ready = _read_and_validate_ready(root_fd)
        if ready is None or _identity(held) != self.identity or ready.identity != self.identity:
            raise FixtureVaultGuardError("fixture_ready_marker_changed_during_publication")
        if (
            ready.nonce != self.nonce
            or ready.sqlite_sha256 != self.sqlite_sha256
            or ready.sqlite_identity != self.sqlite_identity
        ):
            raise FixtureVaultGuardError("fixture_ready_marker_changed_during_publication")
        return ready

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self.fd)


def _replace_ready_generation(
    root_fd: int,
    *,
    nonce: str,
    sqlite_sha256: str,
    sqlite_identity: tuple[int, int],
) -> _ReadyPublication:
    data = _ready_bytes(
        nonce=nonce,
        sqlite_sha256=sqlite_sha256,
        sqlite_identity=sqlite_identity,
    )
    temp_name = f".{FIXTURE_READY_NAME}.{nonce}.tmp"
    temp_fd: int | None = None
    publication: _ReadyPublication | None = None
    try:
        temp_fd = os.open(
            temp_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
            0o600,
            dir_fd=root_fd,
        )
        os.fchmod(temp_fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError("short fixture ready write")
            view = view[written:]
        os.fsync(temp_fd)
        identity = _identity(os.fstat(temp_fd))
        os.replace(temp_name, FIXTURE_READY_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        os.fsync(root_fd)
        publication = _ReadyPublication(
            temp_fd,
            identity,
            nonce,
            sqlite_sha256,
            sqlite_identity,
        )
        temp_fd = None
        try:
            publication.verify(root_fd)
        except BaseException:
            publication.close()
            raise
        return publication
    except FixtureVaultGuardError:
        raise
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_ready_marker_write_failed") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass


@dataclass(slots=True)
class _GenerationFile:
    fd: int
    identity: tuple[int, int]
    sha256: str

    def close(self) -> None:
        os.close(self.fd)


def _generation_file(index_fd: int, name: str) -> _GenerationFile | None:
    fd = _open_regular_at(index_fd, name)
    if fd is None:
        return None
    try:
        return _GenerationFile(fd, _identity(os.fstat(fd)), _hash_fd(fd))
    except BaseException:
        os.close(fd)
        raise


def _assert_no_sqlite_sidecars(index_fd: int, *, allow_empty_wal: bool = False) -> None:
    """Never discard a possibly-live SQLite generation's WAL state.

    A read-only SQLite client can leave a zero-byte WAL and an SHM file behind.
    Those files contain no durable delta and may remain untouched when the
    certified main database is reused. Any transaction that would switch the
    main inode still requires both public sidecars to be absent.
    """

    sidecars: dict[str, os.stat_result] = {}
    for name in ("trove.sqlite-wal", "trove.sqlite-shm"):
        try:
            info = os.stat(name, dir_fd=index_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_sqlite_sidecars_unreadable") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FixtureVaultGuardError("fixture_sqlite_sidecars_active")
        sidecars[name] = info
    if not sidecars:
        return
    wal = sidecars.get("trove.sqlite-wal")
    if allow_empty_wal and (wal is None or wal.st_size == 0):
        return
    raise FixtureVaultGuardError("fixture_sqlite_sidecars_active")


def _copy_generation_candidate(
    source_fd: int,
    index_fd: int,
    state: FixtureGenerationState,
) -> None:
    """Durably copy the staged database under the transaction nonce."""

    candidate_fd: int | None = None
    try:
        candidate_fd = os.open(
            state.candidate_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
            0o600,
            dir_fd=index_fd,
        )
        os.fchmod(candidate_fd, 0o600)
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(candidate_fd, view)
                if written <= 0:
                    raise OSError("short fixture generation write")
                view = view[written:]
        os.fsync(candidate_fd)
        if _hash_fd(candidate_fd) != state.new_sha256:
            raise FixtureVaultGuardError("fixture_generation_copy_mismatch")
        _verify_sqlite_fd(candidate_fd, state.new_sha256)
    except FixtureVaultGuardError:
        raise
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_publish_failed") from exc
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)


def _sync_generation_directory(index_fd: int) -> None:
    try:
        os.fsync(index_fd)
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_fsync_failed") from exc


def _switch_generation(index_fd: int, state: FixtureGenerationState) -> None:
    """Perform only the single-file switch; old sidecars are never removed."""

    _assert_no_sqlite_sidecars(index_fd)
    try:
        os.replace(
            state.candidate_name,
            "trove.sqlite",
            src_dir_fd=index_fd,
            dst_dir_fd=index_fd,
        )
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_switch_failed") from exc
    _sync_generation_directory(index_fd)


def _state_in_phase(state: FixtureGenerationState, phase: str) -> FixtureGenerationState:
    return FixtureGenerationState(
        nonce=state.nonce,
        phase=phase,
        new_sha256=state.new_sha256,
        old_sha256=state.old_sha256,
        old_ready_nonce=state.old_ready_nonce,
    )


def _unlink_generation_name(index_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=index_fd)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_cleanup_failed") from exc


def _remove_generation_state(root_fd: int) -> None:
    try:
        os.unlink(FIXTURE_GENERATION_STATE_NAME, dir_fd=root_fd)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_generation_cleanup_failed") from exc
    try:
        os.fsync(root_fd)
    except OSError as exc:
        # The ready marker already commits the generation. A retry can verify it
        # whether this unlink was persisted or not.
        raise FixtureVaultGuardError("fixture_generation_fsync_failed") from exc


def _scan_tree(root_fd: int) -> None:
    pending = [os.dup(root_fd)]
    try:
        while pending:
            directory_fd = pending.pop()
            try:
                for name in os.listdir(directory_fd):
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        reason = "fixture_marker_invalid_type" if name in {FIXTURE_MARKER_NAME, FIXTURE_READY_NAME} else "fixture_tree_contains_symlink"
                        raise FixtureVaultGuardError(reason)
                    if stat.S_ISDIR(info.st_mode):
                        child_fd = os.open(
                            name,
                            os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                            dir_fd=directory_fd,
                        )
                        if _identity(info) != _identity(os.fstat(child_fd)):
                            os.close(child_fd)
                            raise FixtureVaultGuardError("fixture_tree_changed_during_validation")
                        pending.append(child_fd)
                    elif not stat.S_ISREG(info.st_mode):
                        raise FixtureVaultGuardError("fixture_tree_contains_special_file")
                    elif info.st_nlink != 1:
                        reason = "fixture_marker_invalid_type" if name in {FIXTURE_MARKER_NAME, FIXTURE_READY_NAME} else "fixture_tree_contains_hardlink"
                        raise FixtureVaultGuardError(reason)
            finally:
                os.close(directory_fd)
    except FixtureVaultGuardError:
        for fd in pending:
            os.close(fd)
        raise
    except OSError as exc:
        for fd in pending:
            os.close(fd)
        raise FixtureVaultGuardError("fixture_tree_unreadable") from exc


def _ensure_directory(root_fd: int, name: str) -> None:
    try:
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(name, mode=0o700, dir_fd=root_fd)
            os.fsync(root_fd)
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise FixtureVaultGuardError("fixture_product_directory_invalid")
        child_fd = os.open(name, os.O_RDONLY | _directory_flag() | _no_follow_flag(), dir_fd=root_fd)
        try:
            if _identity(info) != _identity(os.fstat(child_fd)):
                raise FixtureVaultGuardError("fixture_tree_changed_during_validation")
        finally:
            os.close(child_fd)
    except FixtureVaultGuardError:
        raise
    except OSError as exc:
        raise FixtureVaultGuardError("fixture_product_directory_unavailable") from exc


@dataclass(slots=True)
class FixtureVaultGuardSession:
    root: Path
    root_fd: int
    identity: FixtureVaultIdentity
    _closed: bool = False

    def validate_current(self, *, scan_tree: bool = True) -> None:
        if self._closed:
            raise FixtureVaultGuardError("fixture_guard_session_closed")
        held_root = os.fstat(self.root_fd)
        held_marker = _read_and_validate_marker(self.root_fd)
        held_ready = _read_and_validate_ready(self.root_fd)
        if (
            _identity(held_root) != self.identity.root
            or _identity(held_marker) != self.identity.marker
            or held_ready != self.identity.ready
        ):
            raise FixtureVaultGuardError("fixture_identity_changed")

        current_fd = _open_root(self.root, allow_create=False)
        try:
            current_root = os.fstat(current_fd)
            current_marker = _read_and_validate_marker(current_fd)
            current_ready = _read_and_validate_ready(current_fd)
            if (
                _identity(current_root) != self.identity.root
                or _identity(current_marker) != self.identity.marker
                or current_ready != self.identity.ready
            ):
                raise FixtureVaultGuardError("fixture_root_changed_during_operation")
        finally:
            os.close(current_fd)
        if scan_tree:
            _scan_tree(self.root_fd)

    def ensure_product_directories(self) -> None:
        self.validate_current()
        for name in _PRODUCT_DIRECTORIES:
            _ensure_directory(self.root_fd, name)
        self.validate_current()

    def validate_provisional_layout(self, *, published: bool = False, jsonl: bool = False) -> None:
        """Prove that an unfinished claim still contains only our scaffolding."""

        self.validate_current()
        if not self.identity.provisional:
            return
        allowed_root = {FIXTURE_MARKER_NAME, *_PRODUCT_DIRECTORIES}
        if published and jsonl:
            allowed_root.add("fixtures")
        if set(_list_names(self.root_fd)) != allowed_root:
            raise FixtureVaultGuardError("fixture_provisional_layout_changed")
        allowed_logs = {
            "trove-index-writer.flock",
            "trove-index-writer.pid",
            "trove-index-writer.lock.json",
        }
        for name in _PRODUCT_DIRECTORIES:
            directory_fd = os.open(
                name,
                os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                dir_fd=self.root_fd,
            )
            try:
                entries = set(_list_names(directory_fd))
            finally:
                os.close(directory_fd)
            if name == "logs":
                if not entries.issubset(allowed_logs):
                    raise FixtureVaultGuardError("fixture_provisional_layout_changed")
            elif name == "index" and published:
                if entries != {"trove.sqlite"}:
                    raise FixtureVaultGuardError("fixture_provisional_layout_changed")
            elif entries:
                raise FixtureVaultGuardError("fixture_provisional_layout_changed")
        if published and jsonl:
            fixtures_fd = os.open(
                "fixtures",
                os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                dir_fd=self.root_fd,
            )
            try:
                if set(_list_names(fixtures_fd)) != {"synthetic"}:
                    raise FixtureVaultGuardError("fixture_provisional_layout_changed")
                synthetic_fd = os.open(
                    "synthetic",
                    os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                    dir_fd=fixtures_fd,
                )
                try:
                    if set(_list_names(synthetic_fd)) != {"messages.jsonl"}:
                        raise FixtureVaultGuardError("fixture_provisional_layout_changed")
                finally:
                    os.close(synthetic_fd)
            finally:
                os.close(fixtures_fd)

    def publish_file(
        self,
        source: Path,
        relative_parts: tuple[str, ...],
        *,
        require_absent: bool,
    ) -> None:
        """Copy a staged artifact and atomically publish it through held dirfds."""

        self.validate_current()
        if len(relative_parts) < 1 or any(not part or part in {".", ".."} or "/" in part for part in relative_parts):
            raise FixtureVaultGuardError("fixture_publish_location_invalid")
        source = Path(source)
        source_info = source.stat()
        if not stat.S_ISREG(source_info.st_mode):
            raise FixtureVaultGuardError("fixture_stage_artifact_invalid")

        parent_fd = os.dup(self.root_fd)
        try:
            for part in relative_parts[:-1]:
                try:
                    info = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    info = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode):
                    raise FixtureVaultGuardError("fixture_publish_location_invalid")
                child_fd = os.open(
                    part,
                    os.O_RDONLY | _directory_flag() | _no_follow_flag(),
                    dir_fd=parent_fd,
                )
                if _identity(info) != _identity(os.fstat(child_fd)):
                    os.close(child_fd)
                    raise FixtureVaultGuardError("fixture_tree_changed_during_validation")
                os.close(parent_fd)
                parent_fd = child_fd

            target_name = relative_parts[-1]
            temp_name = f".{target_name}.fixture-{secrets.token_hex(12)}.tmp"
            temp_fd: int | None = None
            try:
                temp_fd = os.open(
                    temp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
                    0o600,
                    dir_fd=parent_fd,
                )
                os.fchmod(temp_fd, 0o600)
                with source.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        view = memoryview(chunk)
                        while view:
                            written = os.write(temp_fd, view)
                            if written <= 0:
                                raise OSError("short write while publishing staged fixture")
                            view = view[written:]
                os.fsync(temp_fd)
                os.close(temp_fd)
                temp_fd = None

                if require_absent:
                    try:
                        os.link(
                            temp_name,
                            target_name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise FixtureVaultGuardError("fixture_claim_target_appeared") from exc
                    os.unlink(temp_name, dir_fd=parent_fd)
                else:
                    try:
                        target = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        target = None
                    if target is not None and (not stat.S_ISREG(target.st_mode) or target.st_nlink != 1):
                        raise FixtureVaultGuardError("fixture_publish_target_invalid")
                    os.replace(temp_name, target_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FixtureVaultGuardError:
                raise
            except OSError as exc:
                raise FixtureVaultGuardError("fixture_publish_failed") from exc
            finally:
                if temp_fd is not None:
                    os.close(temp_fd)
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(parent_fd)
        self.validate_current()

    def verify_ready_generation(self, *, integrity: bool = True) -> FixtureReadyGeneration | None:
        """Authenticate the public database against the ready certificate."""

        state = _read_generation_state(self.root_fd)
        self.validate_current(scan_tree=state is None)
        if state is not None:
            raise FixtureVaultGuardError("fixture_generation_recovery_required")

        try:
            index_fd = _open_index_dir(self.root_fd)
        except FixtureVaultGuardError as exc:
            if self.identity.ready is None and exc.reason_code == "fixture_index_unavailable":
                return None
            raise
        try:
            _assert_no_sqlite_sidecars(index_fd, allow_empty_wal=True)
            target = _generation_file(index_fd, "trove.sqlite")
            try:
                ready = self.identity.ready
                if ready is None:
                    if target is not None:
                        raise FixtureVaultGuardError("fixture_uncertified_sqlite_present")
                    return None
                if target is None:
                    raise FixtureVaultGuardError("fixture_ready_generation_missing")
                if target.identity != ready.sqlite_identity or target.sha256 != ready.sqlite_sha256:
                    raise FixtureVaultGuardError("fixture_ready_generation_mismatch")
                if integrity:
                    _verify_sqlite_fd(target.fd, ready.sqlite_sha256)
                return ready
            finally:
                if target is not None:
                    target.close()
        finally:
            os.close(index_fd)

    @staticmethod
    def _validate_old_target(
        state: FixtureGenerationState,
        target: _GenerationFile | None,
    ) -> None:
        if state.old_sha256 is None:
            if target is not None:
                raise FixtureVaultGuardError("fixture_generation_target_changed")
            return
        if target is None or target.sha256 != state.old_sha256:
            raise FixtureVaultGuardError("fixture_generation_target_changed")

    def _cleanup_ready_state(
        self,
        state: FixtureGenerationState,
        artifact: FixturePublishedArtifact,
        ready_publication: _ReadyPublication,
    ) -> None:
        artifact.verify(self.root_fd, integrity=True)
        ready_publication.verify(self.root_fd)
        index_fd = _open_index_dir(self.root_fd)
        try:
            _assert_no_sqlite_sidecars(index_fd)
            candidate = _generation_file(index_fd, state.candidate_name)
            anchor = _generation_file(index_fd, state.anchor_name)
            previous = _generation_file(index_fd, state.previous_name)
            try:
                if candidate is not None and candidate.sha256 != state.new_sha256:
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if anchor is not None and (
                    anchor.sha256 != state.new_sha256
                    or anchor.identity != artifact.identity
                ):
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if previous is not None and (
                    state.old_sha256 is None or previous.sha256 != state.old_sha256
                ):
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
            finally:
                for entry in (candidate, anchor, previous):
                    if entry is not None:
                        entry.close()

            # The previous complete database remains linked until both the
            # target and ready certificate have passed their final checks.
            _unlink_generation_name(index_fd, state.candidate_name)
            _unlink_generation_name(index_fd, state.previous_name)
            _unlink_generation_name(index_fd, state.anchor_name)
            _sync_generation_directory(index_fd)
        finally:
            os.close(index_fd)
        _remove_generation_state(self.root_fd)
        artifact.verify(self.root_fd, integrity=True)
        ready_publication.verify(self.root_fd)
        self.validate_current()

    def recover_pending_generation(self) -> None:
        """Resume or roll back only the nonce-owned interrupted transaction."""

        state = _read_generation_state(self.root_fd)
        self.validate_current(scan_tree=state is None)
        if state is None:
            self.verify_ready_generation(integrity=True)
            return

        index_fd = _open_index_dir(self.root_fd)
        entries: list[_GenerationFile] = []
        try:
            _assert_no_sqlite_sidecars(index_fd)
            target = _generation_file(index_fd, "trove.sqlite")
            candidate = _generation_file(index_fd, state.candidate_name)
            anchor = _generation_file(index_fd, state.anchor_name)
            previous = _generation_file(index_fd, state.previous_name)
            entries = [entry for entry in (target, candidate, anchor, previous) if entry is not None]

            if state.phase == "preparing":
                self._validate_old_target(state, target)
                if anchor is not None and anchor.sha256 != state.new_sha256:
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if candidate is not None and anchor is not None and candidate.identity != anchor.identity:
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if previous is not None and (
                    state.old_sha256 is None
                    or previous.sha256 != state.old_sha256
                    or target is None
                    or previous.identity != target.identity
                ):
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                for entry in entries:
                    entry.close()
                entries = []
                _unlink_generation_name(index_fd, state.candidate_name)
                _unlink_generation_name(index_fd, state.anchor_name)
                _unlink_generation_name(index_fd, state.previous_name)
                _sync_generation_directory(index_fd)
                os.close(index_fd)
                index_fd = -1
                _remove_generation_state(self.root_fd)
                self.validate_current()
                return

            target_is_new = target is not None and target.sha256 == state.new_sha256
            target_is_old = (
                (state.old_sha256 is None and target is None)
                or (target is not None and target.sha256 == state.old_sha256)
            )

            if state.phase == "ready":
                if not target_is_new or target is None:
                    raise FixtureVaultGuardError("fixture_generation_target_changed")
                ready = self.identity.ready
                if ready is None or (
                    ready.nonce != state.nonce
                    or ready.sqlite_sha256 != state.new_sha256
                    or ready.sqlite_identity != target.identity
                ):
                    raise FixtureVaultGuardError("fixture_generation_ready_mismatch")
                _verify_sqlite_fd(target.fd, state.new_sha256)
                ready_fd = os.open(FIXTURE_READY_NAME, os.O_RDONLY | _no_follow_flag(), dir_fd=self.root_fd)
                ready_publication = _ReadyPublication(
                    ready_fd,
                    ready.identity,
                    ready.nonce,
                    ready.sqlite_sha256,
                    ready.sqlite_identity,
                )
                artifact_fd = os.dup(target.fd)
                artifact = FixturePublishedArtifact(
                    artifact_fd,
                    target.identity,
                    target.sha256,
                    state.nonce,
                )
                for entry in entries:
                    entry.close()
                entries = []
                os.close(index_fd)
                index_fd = -1
                try:
                    self._cleanup_ready_state(state, artifact, ready_publication)
                finally:
                    artifact.close()
                    ready_publication.close()
                return

            if state.phase == "swapped" and not target_is_new:
                raise FixtureVaultGuardError("fixture_generation_target_changed")
            if not target_is_new and not target_is_old:
                raise FixtureVaultGuardError("fixture_generation_target_changed")

            if target_is_old:
                if state.phase != "prepared" or candidate is None or anchor is None:
                    raise FixtureVaultGuardError("fixture_generation_artifact_missing")
                if (
                    candidate.sha256 != state.new_sha256
                    or anchor.sha256 != state.new_sha256
                    or candidate.identity != anchor.identity
                ):
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if state.old_sha256 is None:
                    if previous is not None:
                        raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                elif (
                    previous is None
                    or target is None
                    or previous.sha256 != state.old_sha256
                    or previous.identity != target.identity
                ):
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                for entry in entries:
                    entry.close()
                entries = []
                _switch_generation(index_fd, state)
                state = _state_in_phase(state, "swapped")
                _write_generation_state(self.root_fd, state)
            else:
                if target is None or anchor is None:
                    raise FixtureVaultGuardError("fixture_generation_artifact_missing")
                if anchor.sha256 != state.new_sha256 or anchor.identity != target.identity:
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if candidate is not None:
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                if previous is not None and (
                    state.old_sha256 is None or previous.sha256 != state.old_sha256
                ):
                    raise FixtureVaultGuardError("fixture_generation_artifact_changed")
                for entry in entries:
                    entry.close()
                entries = []
                if state.phase != "swapped":
                    state = _state_in_phase(state, "swapped")
                    _write_generation_state(self.root_fd, state)

            _assert_no_sqlite_sidecars(index_fd)
            target_fd = _open_regular_at(index_fd, "trove.sqlite")
            if target_fd is None:
                raise FixtureVaultGuardError("fixture_generation_target_changed")
            artifact = FixturePublishedArtifact(
                target_fd,
                _identity(os.fstat(target_fd)),
                state.new_sha256,
                state.nonce,
            )
        finally:
            for entry in entries:
                entry.close()
            if index_fd >= 0:
                os.close(index_fd)

        ready_publication: _ReadyPublication | None = None
        try:
            artifact.verify(self.root_fd, integrity=True)
            ready_publication = self.mark_generation_ready(artifact)
            self._cleanup_ready_state(state, artifact, ready_publication)
        finally:
            artifact.close()
            if ready_publication is not None:
                ready_publication.close()

    def publish_sqlite_generation(self, source: Path) -> FixturePublishedArtifact:
        """Publish a durable SQLite generation and retain its exact inode."""

        self.recover_pending_generation()
        self.validate_current()
        source = Path(source)
        try:
            listed = os.stat(source, follow_symlinks=False)
            source_fd = os.open(source, os.O_RDONLY | _no_follow_flag())
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_stage_artifact_invalid") from exc
        try:
            opened = os.fstat(source_fd)
            if not stat.S_ISREG(listed.st_mode) or _identity(listed) != _identity(opened):
                raise FixtureVaultGuardError("fixture_stage_artifact_invalid")
            new_sha256 = _hash_fd(source_fd)
            _verify_sqlite_fd(source_fd, new_sha256)

            self.verify_ready_generation(integrity=True)
            index_fd = _open_index_dir(self.root_fd)
            try:
                _assert_no_sqlite_sidecars(index_fd, allow_empty_wal=True)
                target = _generation_file(index_fd, "trove.sqlite")
                try:
                    ready = self.identity.ready
                    if ready is not None and ready.sqlite_sha256 == new_sha256:
                        if target is None or target.identity != ready.sqlite_identity:
                            raise FixtureVaultGuardError("fixture_ready_generation_mismatch")
                        retained_fd = os.dup(target.fd)
                        artifact = FixturePublishedArtifact(
                            retained_fd,
                            target.identity,
                            new_sha256,
                            ready.nonce,
                            reused=True,
                        )
                        artifact.verify(self.root_fd, integrity=True)
                        return artifact

                    _assert_no_sqlite_sidecars(index_fd)
                    old_sha256 = ready.sqlite_sha256 if ready is not None else None
                    old_ready_nonce = ready.nonce if ready is not None else None
                    self._validate_old_target(
                        FixtureGenerationState("0" * 32, "preparing", new_sha256, old_sha256, old_ready_nonce),
                        target,
                    )
                    nonce = secrets.token_hex(16)
                    state = FixtureGenerationState(
                        nonce,
                        "preparing",
                        new_sha256,
                        old_sha256,
                        old_ready_nonce,
                    )
                    _write_generation_state(self.root_fd, state)
                    _copy_generation_candidate(source_fd, index_fd, state)

                    try:
                        os.link(
                            state.candidate_name,
                            state.anchor_name,
                            src_dir_fd=index_fd,
                            dst_dir_fd=index_fd,
                            follow_symlinks=False,
                        )
                        if target is not None:
                            os.link(
                                "trove.sqlite",
                                state.previous_name,
                                src_dir_fd=index_fd,
                                dst_dir_fd=index_fd,
                                follow_symlinks=False,
                            )
                    except OSError as exc:
                        raise FixtureVaultGuardError("fixture_generation_anchor_failed") from exc
                    _sync_generation_directory(index_fd)
                    state = _state_in_phase(state, "prepared")
                    _write_generation_state(self.root_fd, state)
                finally:
                    if target is not None:
                        target.close()

                _switch_generation(index_fd, state)
                state = _state_in_phase(state, "swapped")
                _write_generation_state(self.root_fd, state)
                _assert_no_sqlite_sidecars(index_fd)
                target_fd = _open_regular_at(index_fd, "trove.sqlite")
                if target_fd is None:
                    raise FixtureVaultGuardError("fixture_generation_target_changed")
                artifact = FixturePublishedArtifact(
                    target_fd,
                    _identity(os.fstat(target_fd)),
                    new_sha256,
                    nonce,
                )
            finally:
                os.close(index_fd)
            artifact.verify(self.root_fd, integrity=True)
            return artifact
        finally:
            os.close(source_fd)

    def mark_generation_ready(self, artifact: FixturePublishedArtifact) -> _ReadyPublication:
        if artifact.reused:
            raise FixtureVaultGuardError("fixture_generation_already_ready")
        state = _read_generation_state(self.root_fd)
        if state is None or (
            state.nonce != artifact.nonce
            or state.phase != "swapped"
            or state.new_sha256 != artifact.sha256
        ):
            raise FixtureVaultGuardError("fixture_generation_state_mismatch")
        artifact.verify(self.root_fd, integrity=True)
        index_fd = _open_index_dir(self.root_fd)
        try:
            _assert_no_sqlite_sidecars(index_fd)
        finally:
            os.close(index_fd)

        publication = _replace_ready_generation(
            self.root_fd,
            nonce=artifact.nonce,
            sqlite_sha256=artifact.sha256,
            sqlite_identity=artifact.identity,
        )
        try:
            # Both checks occur after the public ready marker exists. A target
            # replace, including same-byte/inode swaps, leaves a certificate
            # whose recorded identity cannot authenticate the replacement.
            artifact.verify(self.root_fd, integrity=True)
            ready = publication.verify(self.root_fd)
            self.identity = FixtureVaultIdentity(
                root=self.identity.root,
                marker=self.identity.marker,
                ready=ready,
            )
            _write_generation_state(self.root_fd, _state_in_phase(state, "ready"))
            artifact.verify(self.root_fd, integrity=True)
            publication.verify(self.root_fd)
            return publication
        except BaseException:
            publication.close()
            raise

    def finalize_generation(
        self,
        artifact: FixturePublishedArtifact,
        ready_publication: _ReadyPublication,
    ) -> None:
        if artifact.reused:
            artifact.verify(self.root_fd, integrity=True)
            return
        state = _read_generation_state(self.root_fd)
        if state is None or state.phase != "ready" or state.nonce != artifact.nonce:
            raise FixtureVaultGuardError("fixture_generation_state_mismatch")
        self._cleanup_ready_state(state, artifact, ready_publication)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.flock(self.root_fd, fcntl.LOCK_UN)
        finally:
            os.close(self.root_fd)


@contextmanager
def fixture_vault_session(
    root: Path,
    *,
    expected_identity: FixtureVaultIdentity | None = None,
    allow_create: bool = True,
) -> Iterator[FixtureVaultGuardSession]:
    normalized = normalize_fixture_root(root)
    root_fd = _open_root(normalized, allow_create=allow_create)
    session: FixtureVaultGuardSession | None = None
    try:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FixtureVaultGuardError("fixture_claim_locked") from exc
        except OSError as exc:
            raise FixtureVaultGuardError("fixture_claim_lock_unavailable") from exc

        # The path may have been replaced while this caller waited for flock.
        current_fd = _open_root(normalized, allow_create=False)
        try:
            if _identity(os.fstat(current_fd)) != _identity(os.fstat(root_fd)):
                raise FixtureVaultGuardError("fixture_root_changed_during_validation")
        finally:
            os.close(current_fd)

        marker = _marker_stat(root_fd)
        if marker is None:
            if expected_identity is not None:
                raise FixtureVaultGuardError("fixture_root_changed_during_operation")
            if not allow_create:
                raise FixtureVaultGuardError("fixture_marker_missing")
            if _list_names(root_fd):
                raise FixtureVaultGuardError("fixture_marker_missing_nonempty_root")
            publication = _write_marker_once(root_fd)
            try:
                if publication is not None:
                    current = _marker_stat(root_fd)
                    if current is None or _identity(current) != publication.identity:
                        raise FixtureVaultGuardError("fixture_marker_changed_during_publication")
                    expected_names = {FIXTURE_MARKER_NAME, publication.temp_name}
                else:
                    expected_names = {FIXTURE_MARKER_NAME}
                if set(_list_names(root_fd)) != expected_names:
                    raise FixtureVaultGuardError("fixture_claim_raced_with_nonempty_root")
                if publication is not None:
                    publication.finalize(root_fd)
            except BaseException:
                if publication is not None:
                    publication.revoke(root_fd)
                raise
        marker = _read_and_validate_marker(root_fd)
        ready = _read_and_validate_ready(root_fd)
        identity = FixtureVaultIdentity(
            root=_identity(os.fstat(root_fd)),
            marker=_identity(marker),
            ready=ready,
        )
        if expected_identity is not None and identity != expected_identity:
            raise FixtureVaultGuardError("fixture_root_changed_during_operation")

        session = FixtureVaultGuardSession(normalized, root_fd, identity)
        # Interrupted fixture transactions intentionally contain nonce-owned
        # hard-link anchors. Recovery validates those exact artifacts before
        # any cleanup; the normal whole-tree hard-link ban resumes afterward.
        session.validate_current(scan_tree=_read_generation_state(root_fd) is None)
        yield session
    finally:
        if session is not None:
            session.close()
        else:
            try:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            finally:
                os.close(root_fd)


def prepare_fixture_vault(root: Path) -> FixtureVaultIdentity:
    """Claim an empty fixture root or validate its strict existing marker."""

    with fixture_vault_session(root) as session:
        if _read_generation_state(session.root_fd) is None:
            session.verify_ready_generation(integrity=True)
        return session.identity
