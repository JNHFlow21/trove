from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import stat
from typing import Any

from .redaction import redact_obj, stable_hash

MANIFEST_NAME = 'decrypt_manifest.redacted.json'
INTERNAL_GUARD_NAME = '.trove_decrypt_guard.json'
INTERNAL_ACCOUNT_IDENTITY_NAME = '.trove_account_identity.json'
_ACCOUNT_IDENTITY_FORMAT = 'trove-account-identity'
# Accepted read-only so existing private Vault snapshots continue to work.
_LEGACY_ACCOUNT_IDENTITY_FORMAT = 'trove' + '-wechat-account-identity'
_WXID_RE = re.compile(r'^wxid_[A-Za-z0-9]+$')


@dataclass(frozen=True)
class SnapshotGuard:
    strict: bool
    account_dir_name_hashes: frozenset[str]
    run_id: str | None = None

    def allows(self, account_dir: Path) -> bool:
        if not self.strict:
            return True
        return stable_hash(account_dir.name) in self.account_dir_name_hashes

    def to_dict(self) -> dict[str, Any]:
        return {
            'strict': self.strict,
            'run_id': self.run_id,
            'account_dir_name_hashes': sorted(self.account_dir_name_hashes),
            'raw_paths_included': False,
        }


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    payload = redact_obj(payload)
    payload['raw_paths_included'] = False
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return path


def write_guard(run_dir: Path, *, account_dir_names: list[str], run_id: str) -> Path:
    payload = {
        'strict': True,
        'run_id': run_id,
        'account_dir_name_hashes': sorted({stable_hash(name) for name in account_dir_names}),
        'raw_paths_included': False,
    }
    path = run_dir / INTERNAL_GUARD_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return path


def load_snapshot_guard(snapshot_dir: Path) -> SnapshotGuard:
    guard_path = snapshot_dir / INTERNAL_GUARD_NAME
    if not guard_path.exists():
        return SnapshotGuard(strict=False, account_dir_name_hashes=frozenset())
    try:
        payload = json.loads(guard_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return SnapshotGuard(strict=True, account_dir_name_hashes=frozenset(), run_id=None)
    hashes = payload.get('account_dir_name_hashes') if isinstance(payload, dict) else []
    run_id = str(payload.get('run_id') or '') if isinstance(payload, dict) else ''
    return SnapshotGuard(
        strict=bool(payload.get('strict', True)) if isinstance(payload, dict) else True,
        account_dir_name_hashes=frozenset(str(v) for v in hashes if str(v or '').strip()),
        run_id=run_id or None,
    )


def write_account_identity(account_dir: Path, *, account_ref_hash: str, own_wxid: str) -> Path | None:
    """Persist private import identity inside the Vault, never in redacted output.

    Anonymous output directory names must not erase the local account identity
    needed to distinguish incoming from outgoing messages.  This file is an
    internal source input: callers must never include its value in reports or
    logs.
    """

    own_wxid = str(own_wxid or '').strip()
    account_ref_hash = str(account_ref_hash or '').strip()
    if not _WXID_RE.fullmatch(own_wxid) or not re.fullmatch(r'[0-9a-f]{16}', account_ref_hash):
        return None
    account_dir.mkdir(parents=True, exist_ok=True)
    path = account_dir / INTERNAL_ACCOUNT_IDENTITY_NAME
    payload = {
        'format': _ACCOUNT_IDENTITY_FORMAT,
        'version': 1,
        'account_ref_hash': account_ref_hash,
        'own_wxid': own_wxid,
    }
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    fd: int | None = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, 'O_NOFOLLOW', 0)), 0o600)
        encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('short account identity write')
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def load_account_identity(account_dir: Path) -> dict[str, str]:
    path = Path(account_dir) / INTERNAL_ACCOUNT_IDENTITY_NAME
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096 or info.st_mode & 0o077:
            return {}
        payload = json.loads(path.read_text(encoding='ascii'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get('format') not in {_ACCOUNT_IDENTITY_FORMAT, _LEGACY_ACCOUNT_IDENTITY_FORMAT}
        or payload.get('version') != 1
    ):
        return {}
    own_wxid = str(payload.get('own_wxid') or '')
    account_ref_hash = str(payload.get('account_ref_hash') or '')
    if not _WXID_RE.fullmatch(own_wxid) or not re.fullmatch(r'[0-9a-f]{16}', account_ref_hash):
        return {}
    return {'own_wxid': own_wxid, 'account_ref_hash': account_ref_hash}
