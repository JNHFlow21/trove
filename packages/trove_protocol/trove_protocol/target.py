from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping


class TargetRefError(ValueError):
    code = 'target_ref_invalid'


class AmbiguousTargetError(TargetRefError):
    code = 'target_ref_ambiguous'


@dataclass(frozen=True)
class TargetRef:
    provider_id: str
    account_id: str
    kind: str
    stable_id: str
    conversation_id: str | None = None
    peer_id: str | None = None
    display_hints: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ('provider_id', 'account_id', 'kind', 'stable_id'):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise TargetRefError(f'{field_name} is required and bounded')
        if not re.fullmatch(r'[a-z][a-z0-9._-]{0,63}', self.kind):
            raise TargetRefError('target kind is invalid')
        for field_name in ('conversation_id', 'peer_id'):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value or len(value) > 512):
                raise TargetRefError(f'{field_name} is invalid')
        if not isinstance(self.display_hints, Mapping) or len(self.display_hints) > 16:
            raise TargetRefError('display_hints must be a bounded object')
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            or len(key) > 64 or len(value) > 512
            for key, value in self.display_hints.items()
        ):
            raise TargetRefError('display_hints contain invalid text')

    @property
    def identity_key(self) -> str:
        return '\x1f'.join((
            self.provider_id,
            self.account_id,
            self.kind,
            self.stable_id,
            self.conversation_id or '',
            self.peer_id or '',
        ))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'TargetRef':
        fields = {
            'provider_id', 'account_id', 'kind', 'stable_id',
            'conversation_id', 'peer_id', 'display_hints',
        }
        required = {'provider_id', 'account_id', 'kind', 'stable_id'}
        if not isinstance(payload, Mapping) or set(payload) - fields or not required <= set(payload):
            raise TargetRefError('TargetRef fields do not match the contract')
        return cls(
            provider_id=payload['provider_id'], account_id=payload['account_id'],
            kind=payload['kind'], stable_id=payload['stable_id'],
            conversation_id=payload.get('conversation_id'), peer_id=payload.get('peer_id'),
            display_hints=payload.get('display_hints') or {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'provider_id': self.provider_id,
            'account_id': self.account_id,
            'kind': self.kind,
            'stable_id': self.stable_id,
        }
        if self.conversation_id:
            payload['conversation_id'] = self.conversation_id
        if self.peer_id:
            payload['peer_id'] = self.peer_id
        if self.display_hints:
            payload['display_hints'] = dict(self.display_hints)
        # Assert JSON compatibility at the boundary.
        json.dumps(payload, ensure_ascii=False)
        return payload


def resolve_unique_display_target(candidates: list[TargetRef], display_name: str) -> TargetRef:
    """Resolve a display hint only when it identifies exactly one stable target."""
    if not isinstance(candidates, list) or any(not isinstance(item, TargetRef) for item in candidates):
        raise TargetRefError('target candidates must be a TargetRef list')
    if not isinstance(display_name, str) or not display_name:
        raise TargetRefError('display name is required')
    matches = [item for item in candidates if item.display_hints.get('name') == display_name]
    if len(matches) != 1:
        raise AmbiguousTargetError('display name does not identify one stable target')
    return matches[0]


__all__ = [
    'AmbiguousTargetError', 'TargetRef', 'TargetRefError',
    'resolve_unique_display_target',
]
