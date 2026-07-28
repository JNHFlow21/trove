from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Protocol, runtime_checkable


PROVIDER_METHODS = frozenset({'hello', 'capabilities', 'health', 'accounts', 'invoke'})
PROVIDER_CAPABILITIES = frozenset({'read', 'media', 'action'})
PROVIDER_SCHEMA = {
    'version': 1,
    'methods': {
        'hello': {'input': {}, 'output': ['provider_id', 'version', 'protocol', 'schema_sha256']},
        'capabilities': {'input': {}, 'output': ['capabilities', 'source_types']},
        'health': {'input': {}, 'output': ['ok', 'state']},
        'accounts': {'input': {}, 'output': ['accounts']},
        'invoke': {'input': ['method', 'payload'], 'output': ['result']},
    },
    'bulk_transport': 'staging_path_size_sha256_ttl',
}


class ProviderContractError(ValueError):
    code = 'provider_contract_invalid'


def canonical_provider_schema_hash() -> str:
    return hashlib.sha256(json.dumps(
        PROVIDER_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()


def _protocol_number(value: str) -> int:
    match = re.fullmatch(r'trove/([1-9][0-9]*)', str(value or ''))
    if not match:
        raise ProviderContractError('invalid provider protocol version')
    return int(match.group(1))


@dataclass(frozen=True)
class ProtocolRange:
    minimum: str
    maximum: str

    def __post_init__(self) -> None:
        if _protocol_number(self.minimum) > _protocol_number(self.maximum):
            raise ProviderContractError('provider protocol range is inverted')

    def intersects(self, other: 'ProtocolRange') -> bool:
        return max(_protocol_number(self.minimum), _protocol_number(other.minimum)) <= min(
            _protocol_number(self.maximum), _protocol_number(other.maximum),
        )

    def contains(self, version: str) -> bool:
        value = _protocol_number(version)
        return _protocol_number(self.minimum) <= value <= _protocol_number(self.maximum)


@dataclass(frozen=True)
class ProviderManifest:
    provider_id: str
    version: str
    protocol_range: ProtocolRange
    capabilities: tuple[str, ...]
    source_types: tuple[str, ...]
    secret_names: tuple[str, ...]
    package_sha256: str
    schema_sha256: str
    resource_class: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r'[a-z][a-z0-9.-]{0,63}', self.provider_id):
            raise ProviderContractError('provider_id is invalid')
        if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?', self.version):
            raise ProviderContractError('provider version is invalid')
        if not self.capabilities or len(self.capabilities) != len(set(self.capabilities)):
            raise ProviderContractError('provider capabilities must be unique and non-empty')
        if set(self.capabilities) - PROVIDER_CAPABILITIES:
            raise ProviderContractError('provider requested an unknown capability')
        if not self.source_types or len(self.source_types) != len(set(self.source_types)):
            raise ProviderContractError('provider source_types must be unique and non-empty')
        if any(not re.fullmatch(r'[a-z][a-z0-9._-]{0,63}', item) for item in self.source_types):
            raise ProviderContractError('provider source_type is invalid')
        if len(self.secret_names) != len(set(self.secret_names)) or any(
            not re.fullmatch(r'[A-Z][A-Z0-9_]{0,127}', item) for item in self.secret_names
        ):
            raise ProviderContractError('provider secret_names are invalid')
        for value, field in (
            (self.package_sha256, 'package_sha256'),
            (self.schema_sha256, 'schema_sha256'),
        ):
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ProviderContractError(f'provider {field} is invalid')
        if not re.fullmatch(r'[a-z][a-z0-9._-]{0,63}', self.resource_class):
            raise ProviderContractError('provider resource_class is invalid')

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'ProviderManifest':
        if not isinstance(payload, Mapping):
            raise ProviderContractError('provider manifest must be an object')
        fields = {
            'provider_id', 'version', 'protocol_min', 'protocol_max',
            'capabilities', 'source_types', 'secret_names', 'package_sha256',
            'schema_sha256', 'resource_class',
        }
        unknown = set(payload) - fields
        missing = fields - set(payload)
        if unknown or missing:
            raise ProviderContractError('provider manifest fields do not match the contract')
        for field in ('capabilities', 'source_types', 'secret_names'):
            if not isinstance(payload[field], list) or any(not isinstance(item, str) for item in payload[field]):
                raise ProviderContractError(f'provider {field} must be a string list')
        return cls(
            provider_id=payload['provider_id'],
            version=payload['version'],
            protocol_range=ProtocolRange(payload['protocol_min'], payload['protocol_max']),
            capabilities=tuple(payload['capabilities']),
            source_types=tuple(payload['source_types']),
            secret_names=tuple(payload['secret_names']),
            package_sha256=payload['package_sha256'],
            schema_sha256=payload['schema_sha256'],
            resource_class=payload['resource_class'],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'provider_id': self.provider_id,
            'version': self.version,
            'protocol_min': self.protocol_range.minimum,
            'protocol_max': self.protocol_range.maximum,
            'capabilities': list(self.capabilities),
            'source_types': list(self.source_types),
            'secret_names': list(self.secret_names),
            'package_sha256': self.package_sha256,
            'schema_sha256': self.schema_sha256,
            'resource_class': self.resource_class,
        }


@dataclass(frozen=True)
class ProviderAccount:
    account_id: str
    label: str
    message_count: int
    watermark: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'ProviderAccount':
        fields = {'account_id', 'label', 'message_count', 'watermark'}
        if not isinstance(payload, Mapping) or set(payload) - fields or not {'account_id', 'label', 'message_count'} <= set(payload):
            raise ProviderContractError('provider account metadata is invalid')
        if not isinstance(payload['account_id'], str) or not payload['account_id']:
            raise ProviderContractError('provider account_id is required')
        if not isinstance(payload['label'], str) or not payload['label']:
            raise ProviderContractError('provider account label is required')
        if type(payload['message_count']) is not int or payload['message_count'] < 0:
            raise ProviderContractError('provider account message_count is invalid')
        watermark = payload.get('watermark')
        if watermark is not None and not isinstance(watermark, str):
            raise ProviderContractError('provider account watermark is invalid')
        return cls(payload['account_id'], payload['label'], payload['message_count'], watermark)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            'account_id': self.account_id,
            'label': self.label,
            'message_count': self.message_count,
        }
        if self.watermark is not None:
            payload['watermark'] = self.watermark
        return payload


@runtime_checkable
class Provider(Protocol):
    def hello(self) -> Mapping[str, Any]: ...
    def capabilities(self) -> Mapping[str, Any]: ...
    def health(self) -> Mapping[str, Any]: ...
    def accounts(self) -> list[Mapping[str, Any]]: ...
    def invoke(self, method: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


def validate_provider_hello(manifest: ProviderManifest, payload: Mapping[str, Any]) -> None:
    expected = {'provider_id', 'version', 'protocol', 'schema_sha256'}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ProviderContractError('provider hello fields do not match the contract')
    if payload['provider_id'] != manifest.provider_id or payload['version'] != manifest.version:
        raise ProviderContractError('provider hello identity does not match manifest')
    if not manifest.protocol_range.contains(str(payload['protocol'])):
        raise ProviderContractError('provider hello protocol is outside manifest range')
    if payload['schema_sha256'] != manifest.schema_sha256 or payload['schema_sha256'] != canonical_provider_schema_hash():
        raise ProviderContractError('provider schema hash is incompatible')


__all__ = [
    'PROVIDER_CAPABILITIES', 'PROVIDER_METHODS', 'PROVIDER_SCHEMA', 'Provider',
    'ProviderAccount', 'ProviderContractError', 'ProviderManifest', 'ProtocolRange',
    'canonical_provider_schema_hash', 'validate_provider_hello',
]
