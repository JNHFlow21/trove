from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Protocol

from trove_protocol.target import TargetRef


EXPERIMENTAL = True


class ActionContractError(ValueError):
    code = 'experimental_action_contract_invalid'


def _validate_action(action: str, arguments: Mapping[str, Any]) -> None:
    if not isinstance(action, str) or not action or len(action) > 128:
        raise ActionContractError('action is required and bounded')
    if not isinstance(arguments, Mapping):
        raise ActionContractError('action arguments must be an object')
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ActionContractError('action arguments must be canonical JSON') from exc
    if len(encoded) > 64 * 1024:
        raise ActionContractError('action arguments exceed the hard cap')


@dataclass(frozen=True)
class ActionPreflightRequest:
    action: str
    target: TargetRef
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_action(self.action, self.arguments)
        if not isinstance(self.target, TargetRef):
            raise ActionContractError('action target must be a TargetRef')


@dataclass(frozen=True)
class ActionPreflightResult:
    preflight_token: str
    exact_payload: Mapping[str, Any]
    approval_required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.preflight_token, str) or len(self.preflight_token) < 16:
            raise ActionContractError('preflight token is invalid')
        if not isinstance(self.exact_payload, Mapping):
            raise ActionContractError('preflight exact payload is invalid')
        if type(self.approval_required) is not bool:
            raise ActionContractError('approval_required must be boolean')


@dataclass(frozen=True)
class ActionExecuteRequest:
    action: str
    target: TargetRef
    arguments: Mapping[str, Any]
    preflight_token: str
    idempotency_key: str

    def __post_init__(self) -> None:
        _validate_action(self.action, self.arguments)
        if not isinstance(self.target, TargetRef):
            raise ActionContractError('action target must be a TargetRef')
        if not isinstance(self.preflight_token, str) or len(self.preflight_token) < 16:
            raise ActionContractError('preflight token is invalid')
        if not isinstance(self.idempotency_key, str) or len(self.idempotency_key) < 16:
            raise ActionContractError('idempotency key is required')

    def digest(self) -> str:
        payload = {
            'action': self.action,
            'target': self.target.to_dict(),
            'arguments': self.arguments,
            'preflight_token': self.preflight_token,
        }
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ActionStatusRequest:
    operation_id: str


@dataclass(frozen=True)
class ActionCancelRequest:
    operation_id: str
    idempotency_key: str


class ActionProvider(Protocol):
    def preflight(self, request: ActionPreflightRequest) -> ActionPreflightResult: ...
    def execute(self, request: ActionExecuteRequest) -> str: ...
    def status(self, request: ActionStatusRequest) -> Mapping[str, Any]: ...
    def cancel(self, request: ActionCancelRequest) -> Mapping[str, Any]: ...


class IdempotentActionExecutor:
    """Small contract harness used until a real ActionProvider exists."""

    def __init__(self, execute: Callable[[ActionExecuteRequest], str]):
        self._execute = execute
        self._results: dict[str, tuple[str, str]] = {}

    def execute(self, request: ActionExecuteRequest) -> str:
        existing = self._results.get(request.idempotency_key)
        digest = request.digest()
        if existing is not None:
            existing_digest, operation_id = existing
            if existing_digest != digest:
                raise ActionContractError('idempotency key is bound to a different action')
            return operation_id
        operation_id = self._execute(request)
        if not isinstance(operation_id, str) or not operation_id:
            raise ActionContractError('action execute must return an operation id')
        self._results[request.idempotency_key] = (digest, operation_id)
        return operation_id


__all__ = [
    'EXPERIMENTAL', 'ActionCancelRequest', 'ActionContractError',
    'ActionExecuteRequest', 'ActionPreflightRequest', 'ActionPreflightResult',
    'ActionProvider', 'ActionStatusRequest', 'IdempotentActionExecutor',
]
