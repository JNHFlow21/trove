from __future__ import annotations

from typing import Any, Mapping

from trove_core.application.operation_journal import OperationConflict

from .base import HandlerOutcome


def add(context: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    """Persist one idempotent observation behind the durable operation journal."""

    key = str(payload['idempotency_key'])
    try:
        record, replayed = context.operations.start(
            'trove.observe_add', payload,
            idempotency_key=key, replay_policy='journaled',
        )
        if record.state == 'completed':
            return HandlerOutcome.success({
                'operation': record.to_dict(), 'replayed': True,
                'observation': dict(record.result or {}),
            })

        from trove_core.agent_tools.tools import observe_add

        result = observe_add(
            context.config.root,
            entity=str(payload['target']),
            text=str(payload['text']),
        )
        observation = result.get('observation') if isinstance(result, Mapping) else None
        if not isinstance(observation, Mapping):
            raise OperationConflict('observation write did not return a record')
        completed = context.operations.journal.transition(
            record.operation_id,
            expected_states={'pending'},
            state='completed', stage='terminal', owner='none',
            result=dict(observation),
        )
        return HandlerOutcome.success({
            'operation': completed.to_dict(), 'replayed': replayed,
            'observation': dict(observation),
        })
    except OperationConflict as exc:
        return HandlerOutcome.failure(
            exc.code, str(exc), retryable=True,
        )


def list_observations(context: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    from trove_core.agent_tools.tools import observe_list

    target = str(payload.get('target') or '').strip()
    limit = int(payload.get('limit') or 100)
    if target:
        result = observe_list(context.config.root, entity=target, limit=limit)
        return HandlerOutcome.success({
            'observations': list(result.get('observations') or []),
            'count': len(result.get('observations') or []),
        })

    store = context.operations.journal.store
    with store.connect() as connection:
        rows = [dict(row) for row in connection.execute(
            """SELECT observation_id,entity_id,observation_type,value_json,status,
                      confidence,citation,source_type,updated_at
                 FROM observations
                WHERE status IN ('active','needs_review','merge_candidate')
                ORDER BY updated_at DESC,observation_id
                LIMIT ?""",
            (limit,),
        )]
    return HandlerOutcome.success({'observations': rows, 'count': len(rows)})


__all__ = ['add', 'list_observations']
