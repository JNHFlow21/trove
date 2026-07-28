from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping


MAX_BLOB_BYTES = 256 * 1024 * 1024


class BlobContractError(ValueError):
    code = 'blob_contract_invalid'


@dataclass(frozen=True)
class BlobTransfer:
    path: str
    size: int
    sha256: str
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise BlobContractError('blob path is required')
        if type(self.size) is not int or not 0 <= self.size <= MAX_BLOB_BYTES:
            raise BlobContractError('blob size is outside bounds')
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(c not in '0123456789abcdef' for c in self.sha256):
            raise BlobContractError('blob sha256 is invalid')
        if not isinstance(self.expires_at, (int, float)):
            raise BlobContractError('blob expires_at is invalid')

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'BlobTransfer':
        if not isinstance(payload, Mapping):
            raise BlobContractError('blob metadata must be an object')
        unknown = set(payload) - {'path', 'size', 'sha256', 'expires_at'}
        if unknown:
            raise BlobContractError(f'unknown blob metadata fields:{sorted(unknown)}')
        missing = {'path', 'size', 'sha256', 'expires_at'} - set(payload)
        if missing:
            raise BlobContractError(f'missing blob metadata fields:{sorted(missing)}')
        return cls(
            path=payload['path'],
            size=payload['size'],
            sha256=payload['sha256'],
            expires_at=payload['expires_at'],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'path': self.path,
            'size': self.size,
            'sha256': self.sha256,
            'expires_at': self.expires_at,
        }


def validate_staged_blob(
    transfer: BlobTransfer,
    *,
    staging_root: Path,
    now: float | None = None,
    max_bytes: int = MAX_BLOB_BYTES,
) -> Path:
    current = time.time() if now is None else now
    if transfer.expires_at <= current:
        raise BlobContractError('staged blob metadata has expired')
    if transfer.size > max_bytes:
        raise BlobContractError('staged blob exceeds the configured size cap')
    root = staging_root.expanduser().resolve(strict=True)
    raw = Path(transfer.path).expanduser()
    try:
        raw_stat = raw.lstat()
    except FileNotFoundError as exc:
        raise BlobContractError('staged blob does not exist') from exc
    if stat.S_ISLNK(raw_stat.st_mode):
        raise BlobContractError('staged blob cannot be a symlink')
    candidate = raw.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BlobContractError('staged blob escapes the assigned staging root') from exc
    file_stat = candidate.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise BlobContractError('staged blob must be a regular file')
    if file_stat.st_uid != os.getuid():
        raise BlobContractError('staged blob must be owned by the current user')
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise BlobContractError('staged blob permissions must be owner-only')
    if file_stat.st_size != transfer.size:
        raise BlobContractError('staged blob size does not match metadata')
    digest = hashlib.sha256()
    observed = 0
    with candidate.open('rb') as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise BlobContractError('staged blob exceeds the configured size cap')
            digest.update(chunk)
    if observed != transfer.size or digest.hexdigest() != transfer.sha256:
        raise BlobContractError('staged blob hash does not match metadata')
    return candidate


__all__ = ['BlobContractError', 'BlobTransfer', 'MAX_BLOB_BYTES', 'validate_staged_blob']
