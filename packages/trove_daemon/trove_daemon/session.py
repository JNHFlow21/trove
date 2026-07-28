from __future__ import annotations

import os
import re
from typing import Any, Mapping

from trove_protocol.provider import ProtocolRange, ProviderContractError

from .lifecycle import RuntimeIdentity


_HELLO_FIELDS = {
    'type', 'client_id', 'role', 'protocol_min', 'protocol_max',
    'vault_identity', 'build_hash', 'catalog_hash', 'persistent',
}


class SessionError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(message)

    def to_response(self, request_id: str = 'hello') -> dict[str, Any]:
        return {
            'ok': False, 'request_id': request_id,
            'error': {'code': self.code, 'retryable': self.retryable, 'message': str(self)},
        }


class SessionContract:
    def __init__(self, identity: RuntimeIdentity):
        self.identity = identity
        self.protocol_range = ProtocolRange('trove/1', 'trove/1')

    def client_hello(
        self,
        *,
        client_id: str,
        role: str = 'sdk',
        persistent: bool = True,
    ) -> dict[str, Any]:
        return {
            'type': 'hello', 'client_id': client_id, 'role': role,
            'protocol_min': self.protocol_range.minimum,
            'protocol_max': self.protocol_range.maximum,
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
            'persistent': persistent,
        }

    def accept_hello(self, payload: Mapping[str, Any], *, peer_uid: int) -> dict[str, Any]:
        if type(peer_uid) is not int or peer_uid != os.getuid():
            raise SessionError('peer_unauthorized', 'Daemon peer is not the current local user.')
        if not isinstance(payload, Mapping) or set(payload) != _HELLO_FIELDS or payload.get('type') != 'hello':
            raise SessionError('invalid_request', 'Session hello fields do not match the contract.')
        if not isinstance(payload.get('client_id'), str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,128}', payload['client_id']):
            raise SessionError('invalid_request', 'Session client_id is invalid.')
        if payload.get('role') not in {'cli', 'mcp', 'sdk', 'operator'} or type(payload.get('persistent')) is not bool:
            raise SessionError('invalid_request', 'Session role or persistence flag is invalid.')
        try:
            offered = ProtocolRange(str(payload['protocol_min']), str(payload['protocol_max']))
        except (ProviderContractError, KeyError) as exc:
            raise SessionError('protocol_mismatch', 'Session protocol range is invalid.') from exc
        if not offered.intersects(self.protocol_range):
            raise SessionError('protocol_mismatch', 'Session protocol ranges do not intersect.')
        if payload.get('vault_identity') != self.identity.vault_identity:
            raise SessionError('vault_identity_mismatch', 'Client is bound to a different Vault.')
        if payload.get('build_hash') != self.identity.build_hash:
            raise SessionError('version_incompatible', 'Client and daemon build hashes differ.')
        if payload.get('catalog_hash') != self.identity.catalog_hash:
            raise SessionError('catalog_mismatch', 'Client and daemon catalog hashes differ.')
        return {
            'ok': True, 'type': 'hello_ack', 'protocol': 'trove/1',
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
            'role': payload['role'], 'persistent': payload['persistent'],
        }


__all__ = ['SessionContract', 'SessionError']
