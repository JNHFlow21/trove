from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.wechat.media.source_registry import account_dir_hash
from trove_core.wechat.source_discovery import is_wechat_decrypted_account_dir


RECEIPT_FORMAT = 'trove-import-source-receipts'
RECEIPT_VERSION = 1
# Bump whenever a complete source must be reinterpreted even when its bytes and
# process configuration are unchanged.
IMPORTER_CONTRACT_VERSION = 'wechat-full-import/v2'
RECEIPT_FILE_NAME = 'import_source_receipts.redacted.json'
_READ_CHUNK_BYTES = 4 * 1024 * 1024


class SourceFingerprintUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    manifest_sha256: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def stable_import_source_key(path: Path) -> str:
    path = Path(path).expanduser()
    if is_wechat_decrypted_account_dir(path):
        return f'wechat-account:{account_dir_hash(path)}'
    identity = hashlib.sha256(str(path.absolute()).encode('utf-8')).hexdigest()
    return f'file:{identity}'


def _entry_snapshot(root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    try:
        if root.is_symlink():
            raise SourceFingerprintUnavailable('source_symlink')
        if root.is_file():
            candidates = [root]
            base = root.parent
        elif root.is_dir():
            base = root
            candidates = []
            for directory, dirnames, filenames in os.walk(root, followlinks=False):
                directory_path = Path(directory)
                if any((directory_path / name).is_symlink() for name in dirnames):
                    raise SourceFingerprintUnavailable('source_contains_symlink')
                dirnames[:] = sorted(dirnames)
                for name in sorted(filenames):
                    candidates.append(directory_path / name)
        else:
            raise SourceFingerprintUnavailable('source_missing')
        entries: list[tuple[str, int, int, int, int]] = []
        for candidate in candidates:
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise SourceFingerprintUnavailable('source_contains_symlink')
            if not stat.S_ISREG(info.st_mode):
                raise SourceFingerprintUnavailable('source_contains_non_regular_file')
            relative = candidate.relative_to(base).as_posix()
            entries.append((relative, int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns)))
        return tuple(sorted(entries))
    except (OSError, ValueError) as exc:
        raise SourceFingerprintUnavailable(exc.__class__.__name__) from exc


def strong_source_fingerprint(path: Path) -> SourceFingerprint:
    """Hash every byte and prove the source did not change during hashing.

    A receipt is intentionally stronger than the stat fingerprints used by
    ordinary media delta discovery.  It is allowed to skip semantic parsing
    only after a full content digest, an exact importer-contract match, and a
    completed prior run all agree.
    """

    root = Path(path).expanduser()
    before = _entry_snapshot(root)
    base = root.parent if root.is_file() else root
    digest = hashlib.sha256()
    total_bytes = 0
    for relative, expected_dev, expected_ino, expected_size, expected_mtime_ns in before:
        candidate = base / relative
        try:
            info_before = candidate.lstat()
            if (
                int(info_before.st_dev), int(info_before.st_ino), int(info_before.st_size), int(info_before.st_mtime_ns)
            ) != (expected_dev, expected_ino, expected_size, expected_mtime_ns):
                raise SourceFingerprintUnavailable('source_changed_during_hash')
            file_digest = hashlib.sha256()
            with candidate.open('rb') as handle:
                while True:
                    chunk = handle.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    file_digest.update(chunk)
            info_after = candidate.lstat()
        except OSError as exc:
            raise SourceFingerprintUnavailable(exc.__class__.__name__) from exc
        if (
            int(info_after.st_dev), int(info_after.st_ino), int(info_after.st_size), int(info_after.st_mtime_ns)
        ) != (expected_dev, expected_ino, expected_size, expected_mtime_ns):
            raise SourceFingerprintUnavailable('source_changed_during_hash')
        relative_bytes = relative.encode('utf-8')
        digest.update(len(relative_bytes).to_bytes(4, 'big'))
        digest.update(relative_bytes)
        digest.update(expected_size.to_bytes(8, 'big', signed=False))
        digest.update(file_digest.digest())
        total_bytes += expected_size
    if _entry_snapshot(root) != before:
        raise SourceFingerprintUnavailable('source_changed_during_hash')
    digest.update(len(before).to_bytes(8, 'big', signed=False))
    digest.update(total_bytes.to_bytes(16, 'big', signed=False))
    return SourceFingerprint(digest.hexdigest(), len(before), total_bytes)


def source_stat_token(path: Path) -> str:
    """Return a metadata-only token for a cheap post-parse stability check."""

    digest = hashlib.sha256()
    entries = _entry_snapshot(Path(path).expanduser())
    for relative, device, inode, size, mtime_ns in entries:
        relative_bytes = relative.encode('utf-8')
        digest.update(len(relative_bytes).to_bytes(4, 'big'))
        digest.update(relative_bytes)
        digest.update(f'{device}:{inode}:{size}:{mtime_ns}'.encode('ascii'))
    digest.update(len(entries).to_bytes(8, 'big', signed=False))
    return digest.hexdigest()


def _receipt_path(vault_root: Path) -> Path:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    return cfg.paths.jobs_dir / RECEIPT_FILE_NAME


def load_import_receipts(vault_root: Path) -> dict[str, dict[str, Any]]:
    path = _receipt_path(vault_root)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get('format') != RECEIPT_FORMAT or payload.get('version') != RECEIPT_VERSION:
        return {}
    sources = payload.get('sources')
    if not isinstance(sources, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in sources.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def receipt_matches(
    receipt: dict[str, Any] | None,
    fingerprint: SourceFingerprint,
    *,
    process_config_hash: str,
) -> bool:
    return bool(
        isinstance(receipt, dict)
        and receipt.get('status') == 'completed'
        and receipt.get('importer_contract_version') == IMPORTER_CONTRACT_VERSION
        and receipt.get('manifest_sha256') == fingerprint.manifest_sha256
        and int(receipt.get('file_count') or -1) == fingerprint.file_count
        and int(receipt.get('total_bytes') or -1) == fingerprint.total_bytes
        and receipt.get('process_config_hash') == process_config_hash
    )


def completed_receipt(
    fingerprint: SourceFingerprint,
    *,
    process_config_hash: str,
    snapshot_revision: str | None,
) -> dict[str, Any]:
    return {
        'status': 'completed',
        'importer_contract_version': IMPORTER_CONTRACT_VERSION,
        'manifest_sha256': fingerprint.manifest_sha256,
        'file_count': fingerprint.file_count,
        'total_bytes': fingerprint.total_bytes,
        'process_config_hash': process_config_hash,
        'snapshot_revision': snapshot_revision,
        'completed_at': _now(),
        'raw_paths_included': False,
    }


def write_import_receipts(
    vault_root: Path,
    receipts: dict[str, dict[str, Any]],
    *,
    write_session: VaultWriteSession,
) -> None:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    write_session.validate_for(cfg)
    cfg.paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    path = _receipt_path(cfg.root)
    payload = {
        'format': RECEIPT_FORMAT,
        'version': RECEIPT_VERSION,
        'sources': dict(sorted(receipts.items())),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, 'O_NOFOLLOW', 0)), 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('short import receipt write')
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    'IMPORTER_CONTRACT_VERSION',
    'SourceFingerprint',
    'SourceFingerprintUnavailable',
    'completed_receipt',
    'load_import_receipts',
    'receipt_matches',
    'stable_import_source_key',
    'strong_source_fingerprint',
    'source_stat_token',
    'write_import_receipts',
]
