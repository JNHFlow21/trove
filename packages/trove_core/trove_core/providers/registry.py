from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
from contextlib import nullcontext
from typing import Any, Mapping, TextIO

from trove_protocol.provider import (
    PROVIDER_METHODS,
    Provider,
    ProviderAccount,
    ProviderContractError,
    ProviderManifest,
    ProtocolRange,
    canonical_provider_schema_hash,
    validate_provider_hello,
)


class ProviderRegistryError(RuntimeError):
    code = 'provider_registry_rejected'


@dataclass(frozen=True)
class ProviderAllowlistEntry:
    provider_id: str
    package_sha256: str
    owner_uid: int
    capabilities: frozenset[str]
    source_types: frozenset[str]
    secret_names: frozenset[str]
    protocol_range: ProtocolRange = field(default_factory=lambda: ProtocolRange('trove/1', 'trove/1'))

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ProviderRegistryError('allowlist provider_id is invalid')
        if not isinstance(self.package_sha256, str) or len(self.package_sha256) != 64:
            raise ProviderRegistryError('allowlist package hash is invalid')
        if type(self.owner_uid) is not int or self.owner_uid < 0:
            raise ProviderRegistryError('allowlist owner uid is invalid')
        for value, name in (
            (self.capabilities, 'capabilities'),
            (self.source_types, 'source_types'),
            (self.secret_names, 'secret_names'),
        ):
            if not isinstance(value, frozenset):
                raise ProviderRegistryError(f'allowlist {name} must be a frozenset')

    def to_dict(self) -> dict[str, Any]:
        return {
            'provider_id': self.provider_id,
            'package_sha256': self.package_sha256,
            'owner_uid': self.owner_uid,
            'capabilities': self.capabilities,
            'source_types': self.source_types,
            'secret_names': self.secret_names,
            'protocol_range': self.protocol_range,
        }


@dataclass(frozen=True)
class RegisteredProvider:
    manifest: ProviderManifest
    provider: Provider
    package_path: Path


class ProviderRegistry:
    """Core-owned verification boundary for v1 in-process providers."""

    def __init__(self, allowlist: Mapping[str, ProviderAllowlistEntry]):
        self._allowlist = dict(allowlist)
        self._providers: dict[str, RegisteredProvider] = {}
        self._core_range = ProtocolRange('trove/1', 'trove/1')

    def register(
        self,
        manifest: ProviderManifest,
        provider: Provider,
        *,
        package_path: str | Path,
    ) -> None:
        path = self.preflight(manifest, package_path=package_path)
        if manifest.provider_id in self._providers:
            raise ProviderRegistryError('provider id is already registered')
        for method in PROVIDER_METHODS:
            if not callable(getattr(provider, method, None)):
                raise ProviderRegistryError(f'provider contract method is missing:{method}')
        try:
            validate_provider_hello(manifest, provider.hello())
            declared = provider.capabilities()
            if not isinstance(declared, Mapping) or set(declared) != {'capabilities', 'source_types'}:
                raise ProviderContractError('provider capabilities response is invalid')
            if set(declared['capabilities']) != set(manifest.capabilities):
                raise ProviderContractError('provider capabilities do not match manifest')
            if set(declared['source_types']) != set(manifest.source_types):
                raise ProviderContractError('provider source_types do not match manifest')
            health = provider.health()
            if not isinstance(health, Mapping) or type(health.get('ok')) is not bool or not health.get('ok'):
                raise ProviderContractError('provider health is not ready')
            self._validate_accounts(provider.accounts())
        except (ProviderContractError, TypeError, ValueError) as exc:
            raise ProviderRegistryError(str(exc)) from exc
        self._providers[manifest.provider_id] = RegisteredProvider(manifest, provider, path)

    def preflight(
        self,
        manifest: ProviderManifest,
        *,
        package_path: str | Path,
    ) -> Path:
        """Validate core-owned pins before any Provider implementation import."""
        if manifest.provider_id in self._providers:
            raise ProviderRegistryError('provider id is already registered')
        allow = self._allowlist.get(manifest.provider_id)
        if allow is None:
            raise ProviderRegistryError('provider is not in the core allowlist')
        path = self._verify_package(Path(package_path), allow)
        if manifest.package_sha256 != allow.package_sha256:
            raise ProviderRegistryError('provider manifest package hash does not match allowlist')
        if not manifest.protocol_range.intersects(self._core_range) or not manifest.protocol_range.intersects(allow.protocol_range):
            raise ProviderRegistryError('provider protocol range is incompatible')
        if set(manifest.capabilities) - allow.capabilities:
            raise ProviderRegistryError('provider capability privilege expansion was not approved')
        if set(manifest.source_types) - allow.source_types:
            raise ProviderRegistryError('provider source type privilege expansion was not approved')
        if set(manifest.secret_names) - allow.secret_names:
            raise ProviderRegistryError('provider secret privilege expansion was not approved')
        if manifest.schema_sha256 != canonical_provider_schema_hash():
            raise ProviderRegistryError('provider schema hash is incompatible')
        return path

    def accounts(self, provider_id: str) -> list[dict[str, Any]]:
        registered = self._get(provider_id)
        return [item.to_dict() for item in self._validate_accounts(registered.provider.accounts())]

    def invoke(self, provider_id: str, method: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        registered = self._get(provider_id)
        if method not in registered.manifest.capabilities:
            raise ProviderRegistryError('provider method was not declared')
        if not isinstance(payload, Mapping):
            raise ProviderRegistryError('provider payload must be an object')
        result = registered.provider.invoke(method, dict(payload))
        if not isinstance(result, Mapping):
            raise ProviderRegistryError('provider result must be an object')
        return dict(result)

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                'provider_id': item.manifest.provider_id,
                'version': item.manifest.version,
                'capabilities': list(item.manifest.capabilities),
                'source_types': list(item.manifest.source_types),
                'secret_names': list(item.manifest.secret_names),
                'package_sha256': item.manifest.package_sha256,
                'schema_sha256': item.manifest.schema_sha256,
                'secret_values_included': False,
                'private_paths_included': False,
            }
            for item in sorted(self._providers.values(), key=lambda entry: entry.manifest.provider_id)
        ]

    def replace_allowlist_entry(
        self,
        entry: ProviderAllowlistEntry,
        *,
        terminal: TextIO | None = None,
    ) -> None:
        current = self._allowlist.get(entry.provider_id)
        expands = current is None or (
            not entry.capabilities <= current.capabilities
            or not entry.source_types <= current.source_types
            or not entry.secret_names <= current.secret_names
        )
        if expands:
            self._confirm_privilege_expansion(entry, terminal=terminal)
        self._allowlist[entry.provider_id] = entry

    @staticmethod
    def _confirm_privilege_expansion(
        entry: ProviderAllowlistEntry,
        *,
        terminal: TextIO | None,
    ) -> None:
        expected = f'APPROVE PROVIDER-PIN {entry.provider_id} {entry.package_sha256}'
        try:
            context = nullcontext(terminal) if terminal is not None else open(
                '/dev/tty', 'r+', encoding='utf-8', buffering=1,
            )
        except OSError as exc:
            raise ProviderRegistryError('provider privilege expansion requires a controlling terminal') from exc
        with context as tty:
            if tty is None or not tty.isatty():
                raise ProviderRegistryError('provider privilege expansion requires a live controlling terminal')
            tty.write('TROVE provider privilege expansion (exact pin):\n')
            tty.write(json.dumps({
                'provider_id': entry.provider_id,
                'package_sha256': entry.package_sha256,
                'capabilities': sorted(entry.capabilities),
                'source_types': sorted(entry.source_types),
                'secret_names': sorted(entry.secret_names),
            }, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
            tty.write(f'Type exactly: {expected}\n> ')
            tty.flush()
            if tty.readline().rstrip('\r\n') != expected:
                raise ProviderRegistryError('provider privilege expansion was not confirmed')

    def _get(self, provider_id: str) -> RegisteredProvider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ProviderRegistryError('provider is not registered') from exc

    @staticmethod
    def _verify_package(path: Path, allow: ProviderAllowlistEntry) -> Path:
        try:
            raw = path.expanduser()
            info = raw.lstat()
        except OSError as exc:
            raise ProviderRegistryError('provider package is unavailable') from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ProviderRegistryError('provider package must be a regular file')
        if info.st_uid != allow.owner_uid or info.st_uid != os.getuid():
            raise ProviderRegistryError('provider package owner is invalid')
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ProviderRegistryError('provider package permissions must be owner-only')
        digest = hashlib.sha256()
        with raw.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != allow.package_sha256:
            raise ProviderRegistryError('provider package hash does not match allowlist')
        return raw.resolve(strict=True)

    @staticmethod
    def _validate_accounts(rows: object) -> list[ProviderAccount]:
        if not isinstance(rows, list) or len(rows) > 64:
            raise ProviderContractError('provider accounts response is invalid')
        accounts = [ProviderAccount.from_dict(row) for row in rows]
        ids = [item.account_id for item in accounts]
        if len(ids) != len(set(ids)):
            raise ProviderContractError('provider account ids are duplicated')
        return accounts


__all__ = [
    'ProviderAllowlistEntry', 'ProviderRegistry', 'ProviderRegistryError',
    'RegisteredProvider',
]
