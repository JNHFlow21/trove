from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import os
from pathlib import Path
import plistlib
import re
import stat
import subprocess
from typing import Any, Callable

from trove_core.reply.service import _private_write_json


_BUNDLE_ID = re.compile(r'^[A-Za-z0-9.-]{3,255}$')
_CDHASH = re.compile(r'^[0-9a-f]{40,64}$')


class OperatorTrustError(RuntimeError):
    code = 'operator_trust_invalid'


@dataclass(frozen=True)
class OperatorAppIdentity:
    bundle_identifier: str
    app_path: str
    executable_path: str
    cdhash: str
    team_identifier: str | None = None

    def __post_init__(self) -> None:
        if _BUNDLE_ID.fullmatch(self.bundle_identifier) is None:
            raise OperatorTrustError('operator bundle identifier is invalid')
        if not Path(self.app_path).is_absolute() or not Path(
            self.executable_path
        ).is_absolute():
            raise OperatorTrustError('operator paths must be absolute')
        if _CDHASH.fullmatch(self.cdhash) is None:
            raise OperatorTrustError('operator code hash is invalid')
        if self.team_identifier is not None and not self.team_identifier:
            raise OperatorTrustError('operator team identifier is invalid')

    def redacted(self) -> dict[str, Any]:
        return {
            'bundle_identifier': self.bundle_identifier,
            'app_name': Path(self.app_path).name,
            'executable_name': Path(self.executable_path).name,
            'cdhash': self.cdhash,
            'team_identifier': self.team_identifier,
        }


def trust_path(vault_root: str | Path) -> Path:
    return Path(vault_root) / 'jobs' / 'reply' / 'operator.json'


def save_operator_trust(
    vault_root: str | Path,
    identity: OperatorAppIdentity,
) -> None:
    _private_write_json(
        trust_path(vault_root),
        {'schema_version': 1, **asdict(identity)},
    )


def load_operator_trust(
    vault_root: str | Path,
) -> OperatorAppIdentity | None:
    path = trust_path(vault_root)
    if not path.is_file():
        return None
    import json

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('schema_version') != 1:
            raise ValueError
        return OperatorAppIdentity(**{
            name: payload.get(name)
            for name in OperatorAppIdentity.__dataclass_fields__
        })
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise OperatorTrustError('operator trust file is invalid') from exc


def _owned_regular(path: Path) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise OperatorTrustError('operator executable is unavailable') from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        raise OperatorTrustError('operator executable ownership is invalid')
    return path.resolve(strict=True)


def inspect_operator_app(
    app_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> OperatorAppIdentity:
    raw = Path(app_path).expanduser()
    try:
        app_info = raw.lstat()
        app = raw.resolve(strict=True)
    except OSError as exc:
        raise OperatorTrustError('operator app is unavailable') from exc
    if (
        stat.S_ISLNK(app_info.st_mode)
        or not stat.S_ISDIR(app_info.st_mode)
        or app_info.st_uid != os.getuid()
        or app.suffix != '.app'
    ):
        raise OperatorTrustError('operator app path is invalid')
    plist_path = app / 'Contents' / 'Info.plist'
    try:
        with plist_path.open('rb') as handle:
            plist = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise OperatorTrustError('operator app metadata is invalid') from exc
    bundle_id = str(plist.get('CFBundleIdentifier') or '')
    executable_name = str(plist.get('CFBundleExecutable') or '')
    if (
        _BUNDLE_ID.fullmatch(bundle_id) is None
        or not executable_name
        or '/' in executable_name
    ):
        raise OperatorTrustError('operator app identity is invalid')
    executable = _owned_regular(
        app / 'Contents' / 'MacOS' / executable_name,
    )
    verified = runner(
        [
            '/usr/bin/codesign', '--verify', '--deep', '--strict',
            '--verbose=2', str(app),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if verified.returncode != 0:
        raise OperatorTrustError('operator app code signature is invalid')
    described = runner(
        ['/usr/bin/codesign', '-dv', '--verbose=4', str(executable)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if described.returncode != 0:
        raise OperatorTrustError('operator signature metadata is unavailable')
    metadata: dict[str, str] = {}
    for line in (described.stderr + '\n' + described.stdout).splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        metadata[key.strip()] = value.strip()
    identifier = metadata.get('Identifier', '')
    cdhash = metadata.get('CDHash', '').lower()
    team = metadata.get('TeamIdentifier')
    if identifier != bundle_id or _CDHASH.fullmatch(cdhash) is None:
        raise OperatorTrustError('operator signature identity does not match')
    return OperatorAppIdentity(
        bundle_identifier=bundle_id,
        app_path=str(app),
        executable_path=str(executable),
        cdhash=cdhash,
        team_identifier=team if team and team != 'not set' else None,
    )


def process_executable_path(pid: int) -> Path:
    if type(pid) is not int or pid <= 1:
        raise OperatorTrustError('operator process id is invalid')
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    function = library.proc_pidpath
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    length = function(pid, buffer, len(buffer))
    if length <= 0:
        raise OperatorTrustError('operator process path is unavailable')
    return Path(os.fsdecode(buffer.value)).resolve(strict=True)


class SignedOperatorAuthorizer:
    def __init__(
        self,
        vault_root: str | Path,
        *,
        process_path: Callable[[int], Path] = process_executable_path,
        inspect_app: Callable[[str | Path], OperatorAppIdentity] = inspect_operator_app,
    ) -> None:
        self.vault_root = Path(vault_root)
        self.process_path = process_path
        self.inspect_app = inspect_app

    def authorize(self, pid: int) -> bool:
        try:
            trusted = load_operator_trust(self.vault_root)
            if trusted is None:
                return False
            actual_path = self.process_path(pid)
            if actual_path != Path(trusted.executable_path).resolve(strict=True):
                return False
            actual = self.inspect_app(trusted.app_path)
            return actual == trusted
        except Exception:
            return False


__all__ = [
    'OperatorAppIdentity', 'OperatorTrustError', 'SignedOperatorAuthorizer',
    'inspect_operator_app', 'load_operator_trust', 'process_executable_path',
    'save_operator_trust', 'trust_path',
]
