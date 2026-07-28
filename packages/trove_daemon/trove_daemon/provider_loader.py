from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from trove_core.providers.registry import (
    ProviderAllowlistEntry, ProviderRegistry, ProviderRegistryError,
)
from trove_protocol.provider import ProviderManifest


MAX_STAGING_BYTES = 64 * 1024 * 1024
OFFICIAL_WECHAT_PACKAGE_SHA256 = (
    'f73aa19d6539606ad24d94f2bb7cea5c'
    '3bdfde7c6b2d68ecd9a734744a8e2e88'
)


def official_provider_registry() -> ProviderRegistry:
    return ProviderRegistry({
        'wechat-source': ProviderAllowlistEntry(
            provider_id='wechat-source',
            package_sha256=OFFICIAL_WECHAT_PACKAGE_SHA256,
            owner_uid=os.getuid(),
            capabilities=frozenset({'read', 'media', 'action'}),
            source_types=frozenset({'wechat'}),
            secret_names=frozenset({'TROVE_WECHAT_KEY_STORE'}),
        ),
    })


class ProviderLoadError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ProviderDistribution:
    provider_id: str
    distribution_name: str
    version: str
    module_name: str
    factory_name: str
    package_root: Path


def discover_provider_distributions(
    distributions: Any | None = None,
) -> tuple[ProviderDistribution, ...]:
    """Read distribution entry-point metadata without importing Provider code."""
    installed = importlib.metadata.distributions() if distributions is None else distributions
    result: list[ProviderDistribution] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for distribution in installed:
        for entry_point in distribution.entry_points:
            if entry_point.group != 'trove.providers':
                continue
            module_name, separator, factory_name = entry_point.value.partition(':')
            if (
                not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', module_name)
                or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', factory_name)
                or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,63}', entry_point.name)
            ):
                continue
            relative_module = module_name.replace('.', '/')
            root = Path(distribution.locate_file(relative_module))
            if not root.is_dir():
                try:
                    direct_url_text = distribution.read_text('direct_url.json')
                    direct_url = json.loads(direct_url_text or '{}')
                    parsed = urlparse(str(direct_url.get('url') or ''))
                    editable = (
                        isinstance(direct_url.get('dir_info'), dict)
                        and direct_url['dir_info'].get('editable') is True
                    )
                    project_root = Path(unquote(parsed.path))
                    candidate = project_root / relative_module
                    if (
                        parsed.scheme == 'file'
                        and editable
                        and candidate.is_dir()
                    ):
                        root = candidate
                except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
                    pass
            try:
                canonical_root = str(root.resolve(strict=True))
            except (FileNotFoundError, NotADirectoryError, OSError):
                canonical_root = str(root.absolute())
            key = (
                entry_point.name,
                str(distribution.metadata['Name']).lower(),
                str(distribution.version),
                module_name,
                factory_name,
                canonical_root,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(ProviderDistribution(
                provider_id=entry_point.name,
                distribution_name=str(distribution.metadata['Name']),
                version=str(distribution.version),
                module_name=module_name,
                factory_name=factory_name,
                package_root=Path(canonical_root),
            ))
    return tuple(sorted(result, key=lambda item: (item.provider_id, item.distribution_name)))


def canonical_package_blob(root: Path) -> bytes:
    chunks = [b'trove-provider-package-v1\0']
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name in {'manifest.json', 'package.seal'} or '__pycache__' in path.parts:
            continue
        listed = path.lstat()
        if stat.S_ISLNK(listed.st_mode) or listed.st_uid != os.getuid() or listed.st_mode & 0o022:
            raise ProviderLoadError('provider_package_unsafe', 'Provider distribution file is unsafe.')
        relative = path.relative_to(root).as_posix().encode('utf-8')
        content = path.read_bytes()
        chunks.extend((len(relative).to_bytes(4, 'big'), relative, len(content).to_bytes(8, 'big'), content))
    return b''.join(chunks)


@dataclass(frozen=True)
class ProviderLoadResult:
    ok: bool
    provider_id: str
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok, 'provider_id': self.provider_id,
            **({'error': {'code': self.error_code, 'retryable': False}} if self.error_code else {}),
            **({
                'next_action': {
                    'capability': 'trove.provider_status',
                    'action': 'repair_or_reinstall_provider',
                },
            } if self.error_code else {}),
            'pure_vault_read_available': True,
            'private_paths_included': False,
            'secret_values_included': False,
        }


class ProviderLoader:
    """Verify distribution metadata and core allowlist before in-process import."""

    def __init__(self, registry: ProviderRegistry, *, runtime_dir: str | Path):
        self.registry = registry
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.runtime_dir, 0o700)

    def load(
        self,
        package_root: str | Path,
        *,
        module_name: str,
        factory_name: str = 'create_provider',
        provider_kwargs: Mapping[str, Any] | None = None,
    ) -> ProviderLoadResult:
        provider_id = 'unknown-provider'
        seal_copy: Path | None = None
        try:
            root = Path(package_root).resolve(strict=True)
            manifest_payload = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
            manifest = ProviderManifest.from_dict(manifest_payload)
            provider_id = manifest.provider_id
            blob = canonical_package_blob(root)
            seal = (root / 'package.seal').read_bytes()
            if seal != blob or hashlib.sha256(seal).hexdigest() != manifest.package_sha256:
                raise ProviderLoadError('provider_package_hash_mismatch', 'Provider distribution hash does not match its manifest.')
            seal_copy = self.runtime_dir / f'{provider_id}-{secrets.token_hex(8)}.pkg'
            fd = os.open(seal_copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, 'O_NOFOLLOW', 0)), 0o600)
            try:
                view = memoryview(seal)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError('short provider seal write')
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            self.registry.preflight(manifest, package_path=seal_copy)
            module = importlib.import_module(module_name)
            factory = getattr(module, factory_name)
            provider = factory(manifest, **dict(provider_kwargs or {}))
            self.registry.register(manifest, provider, package_path=seal_copy)
            return ProviderLoadResult(True, provider_id)
        except Exception as exc:
            code = getattr(exc, 'code', 'provider_load_failed')
            return ProviderLoadResult(False, provider_id, str(code))
        finally:
            if seal_copy is not None:
                seal_copy.unlink(missing_ok=True)

    def load_distribution(
        self,
        distribution: ProviderDistribution,
        *,
        provider_kwargs: Mapping[str, Any] | None = None,
    ) -> ProviderLoadResult:
        return self.load(
            distribution.package_root,
            module_name=distribution.module_name,
            factory_name=distribution.factory_name,
            provider_kwargs=provider_kwargs,
        )


@dataclass(frozen=True)
class StagingGrant:
    handle: str
    path: Path
    max_bytes: int


class StagingTransfer:
    def __init__(self, root: str | Path, *, max_bytes: int = MAX_STAGING_BYTES):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._grants: dict[str, Path] = {}

    def allocate(self) -> StagingGrant:
        handle = secrets.token_urlsafe(32)
        path = self.root / f'{handle}.staging'
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, 'O_NOFOLLOW', 0)), 0o600)
        os.close(fd)
        self._grants[handle] = path
        return StagingGrant(handle, path, self.max_bytes)

    def accept(
        self,
        handle: str,
        *,
        size: int,
        sha256: str,
        cas_dir: str | Path,
    ) -> Path:
        path = self._grants.pop(handle, None)
        if path is None:
            raise ProviderLoadError('staging_invalid', 'Staging grant is unknown.')
        try:
            listed = path.lstat()
            if (
                stat.S_ISLNK(listed.st_mode) or not stat.S_ISREG(listed.st_mode)
                or listed.st_uid != os.getuid() or listed.st_mode & 0o077
                or path.parent.resolve(strict=True) != self.root.resolve(strict=True)
            ):
                raise ProviderLoadError('staging_unsafe', 'Staging object is unsafe.')
            if type(size) is not int or not 0 <= size <= self.max_bytes or listed.st_size != size:
                raise ProviderLoadError('staging_size_mismatch', 'Staging size is invalid.')
            if not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256):
                raise ProviderLoadError('staging_hash_mismatch', 'Staging hash is invalid.')
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != sha256:
                raise ProviderLoadError('staging_hash_mismatch', 'Staging hash does not match.')
            destination_root = Path(cas_dir)
            destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(destination_root, 0o700)
            destination = destination_root / sha256
            if destination.exists():
                path.unlink()
            else:
                os.replace(path, destination)
                os.chmod(destination, 0o600)
            return destination
        except BaseException:
            self._discard(path)
            raise

    def _discard(self, path: Path) -> None:
        try:
            if path.parent.resolve(strict=True) != self.root.resolve(strict=True):
                return
            listed = path.lstat()
            if stat.S_ISDIR(listed.st_mode) and not stat.S_ISLNK(listed.st_mode):
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
        except OSError:
            return


__all__ = [
    'MAX_STAGING_BYTES', 'ProviderLoadError', 'ProviderLoadResult',
    'ProviderDistribution', 'ProviderLoader', 'StagingGrant', 'StagingTransfer',
    'canonical_package_blob', 'discover_provider_distributions',
    'official_provider_registry',
]
