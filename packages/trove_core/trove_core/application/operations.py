from __future__ import annotations

from collections.abc import Callable, Mapping
import secrets
from typing import Any

from .operation_journal import (
    OperationConflict,
    OperationJournal,
    OperationRecord,
)


Continuation = Callable[[OperationRecord, Mapping[str, Any]], Mapping[str, Any]]


class OperationService:
    """Semantic operation lifecycle; internal task bookkeeping stays private."""

    def __init__(self, journal: OperationJournal):
        self.journal = journal

    def start(
        self,
        capability_id: str,
        request: Mapping[str, Any],
        *,
        idempotency_key: str,
        replay_policy: str,
        owner: str = 'daemon',
    ) -> tuple[OperationRecord, bool]:
        return self.journal.start(
            capability_id,
            request,
            idempotency_key=idempotency_key,
            replay_policy=replay_policy,
            owner=owner,
        )

    def status(self, operation_id: str) -> OperationRecord:
        return self.journal.get(operation_id)

    def await_agent(self, operation_id: str, *, stage: str) -> tuple[OperationRecord, str]:
        current = self.journal.get(operation_id)
        token = secrets.token_urlsafe(32)
        token_hash = self.journal.continuation_hash(current, token)
        waiting = self.journal.transition(
            operation_id,
            expected_states={'pending', 'running'},
            state='awaiting_agent',
            stage=stage,
            owner='agent',
            continuation_token_hash=token_hash,
        )
        return waiting, token

    def continue_operation(
        self,
        operation_id: str,
        *,
        token: str,
        payload: Mapping[str, Any],
        continuation: Continuation,
    ) -> OperationRecord:
        if not isinstance(payload, Mapping):
            raise OperationConflict('continuation payload must be an object')
        running = self.journal.consume_continuation(operation_id, token=token)
        try:
            result = continuation(running, dict(payload))
            if not isinstance(result, Mapping):
                raise TypeError('continuation result must be an object')
        except BaseException as exc:
            self.journal.transition(
                operation_id,
                expected_states={'running'},
                state='failed',
                stage='terminal',
                owner='none',
                error={'code': 'continuation_failed', 'retryable': False, 'type': exc.__class__.__name__},
            )
            raise
        return self.journal.transition(
            operation_id,
            expected_states={'running'},
            state='completed',
            stage='terminal',
            owner='none',
            result=dict(result),
        )

    def cancel(self, operation_id: str) -> OperationRecord:
        current = self.journal.get(operation_id)
        if current.state not in {'pending', 'awaiting_agent'}:
            raise OperationConflict('operation is not in a cancellable stage')
        return self.journal.transition(
            operation_id,
            expected_states={current.state},
            state='cancelled',
            stage='terminal',
            owner='none',
        )

    def mark_external_dispatched(self, operation_id: str, *, external_ref: str) -> OperationRecord:
        if not isinstance(external_ref, str) or not external_ref:
            raise OperationConflict('external_ref is required for reconciliation')
        current = self.journal.get(operation_id)
        return self.journal.transition(
            operation_id,
            expected_states={current.state} if current.state in {'pending', 'running'} else set(),
            state='reconciling',
            stage='external_dispatched',
            owner='provider',
            external_ref=external_ref,
        )

    def reconcile_external(
        self,
        operation_id: str,
        *,
        terminal: bool,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> OperationRecord:
        if not terminal:
            return self.journal.transition(
                operation_id,
                expected_states={'reconciling'},
                state='reconciling',
                stage='external_pending',
                owner='provider',
                external_ref=self.journal.get(operation_id).external_ref,
            )
        if error is not None:
            return self.journal.transition(
                operation_id,
                expected_states={'reconciling'},
                state='failed',
                stage='terminal',
                owner='none',
                error=dict(error),
            )
        return self.journal.transition(
            operation_id,
            expected_states={'reconciling'},
            state='completed',
            stage='terminal',
            owner='none',
            result=dict(result or {}),
        )


__all__ = ['Continuation', 'OperationService']
