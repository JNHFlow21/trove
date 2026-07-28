from __future__ import annotations

import hashlib
from typing import Any, Mapping

from trove_core.application.operation_journal import OperationConflict, OperationNotFound

from .base import HandlerOutcome


def _operation_failure(exc: BaseException) -> HandlerOutcome:
    code = getattr(exc, 'code', 'operation_failed')
    return HandlerOutcome.failure(str(code), str(exc), retryable=isinstance(exc, OperationConflict))


def status(context: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    try:
        record = context.operations.status(str(payload['operation_id']))
    except (OperationConflict, OperationNotFound) as exc:
        return _operation_failure(exc)
    return HandlerOutcome.success({'operation': record.to_dict()})


def continue_operation(context: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    operation_id = str(payload['operation_id'])
    continuation = context.continuations.get(operation_id)
    if continuation is None:
        return HandlerOutcome.failure(
            'capability_unavailable',
            'No caller continuation is registered for this operation.',
            details={'operation_id': operation_id},
        )
    try:
        record = context.operations.continue_operation(
            operation_id,
            token=str(payload['token']),
            payload=payload['payload'],
            continuation=continuation,
        )
    except (OperationConflict, OperationNotFound) as exc:
        return _operation_failure(exc)
    return HandlerOutcome.success({'operation': record.to_dict()})


def cancel(context: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    try:
        record = context.operations.cancel(str(payload['operation_id']))
    except (OperationConflict, OperationNotFound) as exc:
        return _operation_failure(exc)
    return HandlerOutcome.success({'operation': record.to_dict()})


def start_operation(
    context: Any,
    capability_id: str,
    payload: Mapping[str, Any],
    *,
    request_id: str,
    replay_policy: str = 'journaled',
) -> HandlerOutcome:
    provided = payload.get('idempotency_key')
    key = str(provided) if provided else hashlib.sha256(
        f'{capability_id}\x00{request_id}'.encode('utf-8')
    ).hexdigest()
    try:
        record, replayed = context.operations.start(
            capability_id,
            payload,
            idempotency_key=key,
            replay_policy=replay_policy,
        )
    except OperationConflict as exc:
        return _operation_failure(exc)
    return HandlerOutcome.success({'operation': record.to_dict(), 'replayed': replayed})


__all__ = ['cancel', 'continue_operation', 'start_operation', 'status']
