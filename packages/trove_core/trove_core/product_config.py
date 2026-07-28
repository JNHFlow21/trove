from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping


CONFIG_SCHEMA_VERSION = 1
PACKS = frozenset({'standard', 'operations', 'admin'})
_PROVIDER_ID = re.compile(r'^[a-z0-9][a-z0-9-]{1,63}$')
_SECRET_NAME = re.compile(r'^[A-Z][A-Z0-9_]{1,127}$')
_KEYS = frozenset({
    'schema_version', 'vault_root', 'runtime_root', 'provider_id',
    'mcp_pack', 'secret_names',
})


class ProductConfigError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _optional_absolute_path(value: Any, field: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ProductConfigError('config_invalid', f'{field} must be an absolute path or null')
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ProductConfigError('config_invalid', f'{field} must be absolute')
    return path


@dataclass(frozen=True)
class ProductConfig:
    vault_root: Path | None = None
    runtime_root: Path | None = None
    provider_id: str = 'wechat-source'
    mcp_pack: str = 'standard'
    secret_names: tuple[str, ...] = ()
    explicit: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, explicit: bool = True) -> 'ProductConfig':
        if set(payload) != _KEYS or payload.get('schema_version') != CONFIG_SCHEMA_VERSION:
            raise ProductConfigError('config_schema_invalid', 'configuration keys or version are invalid')
        provider_id = payload.get('provider_id')
        pack = payload.get('mcp_pack')
        secret_names = payload.get('secret_names')
        if not isinstance(provider_id, str) or not _PROVIDER_ID.fullmatch(provider_id):
            raise ProductConfigError('config_invalid', 'provider_id is invalid')
        if pack not in PACKS:
            raise ProductConfigError('config_invalid', 'mcp_pack is invalid')
        if (
            not isinstance(secret_names, list)
            or len(secret_names) != len(set(secret_names))
            or any(not isinstance(name, str) or not _SECRET_NAME.fullmatch(name) for name in secret_names)
        ):
            raise ProductConfigError('secret_value_forbidden', 'only unique secret names are allowed')
        return cls(
            vault_root=_optional_absolute_path(payload.get('vault_root'), 'vault_root'),
            runtime_root=_optional_absolute_path(payload.get('runtime_root'), 'runtime_root'),
            provider_id=provider_id,
            mcp_pack=str(pack),
            secret_names=tuple(secret_names),
            explicit=explicit,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': CONFIG_SCHEMA_VERSION,
            'vault_root': str(self.vault_root) if self.vault_root else None,
            'runtime_root': str(self.runtime_root) if self.runtime_root else None,
            'provider_id': self.provider_id,
            'mcp_pack': self.mcp_pack,
            'secret_names': list(self.secret_names),
        }


def default_config_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / '.config/trove/config.json'


def _secure_read(path: Path) -> dict[str, Any]:
    listed = path.lstat()
    if (
        stat.S_ISLNK(listed.st_mode) or not stat.S_ISREG(listed.st_mode)
        or listed.st_uid != os.getuid() or listed.st_mode & 0o077
    ):
        raise ProductConfigError('config_permissions_invalid', 'configuration must be owner-only')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductConfigError('config_invalid', 'configuration is unreadable') from exc
    if not isinstance(value, dict):
        raise ProductConfigError('config_invalid', 'configuration must be an object')
    return value


def load_product_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    for_write: bool = False,
) -> ProductConfig:
    environment = os.environ if env is None else env
    selected = Path(path).expanduser() if path is not None else default_config_path(home)
    if selected.is_file() or selected.is_symlink():
        return ProductConfig.from_dict(_secure_read(selected), explicit=True)
    if for_write:
        raise ProductConfigError(
            'explicit_config_required',
            'write operations require an owner-created product configuration',
        )
    vault_text = environment.get('TROVE_VAULT_ROOT')
    default_vault = (home or Path.home()) / 'Trove/trove-vault'
    vault = Path(vault_text).expanduser() if vault_text else (default_vault if default_vault.is_dir() else None)
    if vault is not None and not vault.is_absolute():
        vault = None
    return ProductConfig(vault_root=vault, explicit=False)


def write_product_config(path: str | Path, config: ProductConfig) -> None:
    destination = Path(path).expanduser()
    payload = ProductConfig.from_dict(config.to_dict(), explicit=True).to_dict()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = destination.parent.lstat()
    if stat.S_ISLNK(parent.st_mode) or parent.st_uid != os.getuid():
        raise ProductConfigError('config_permissions_invalid', 'configuration directory is unsafe')
    os.chmod(destination.parent, 0o700)
    fd, temporary_name = tempfile.mkstemp(prefix='.trove-config-', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    'CONFIG_SCHEMA_VERSION', 'PACKS', 'ProductConfig', 'ProductConfigError',
    'default_config_path', 'load_product_config', 'write_product_config',
]
