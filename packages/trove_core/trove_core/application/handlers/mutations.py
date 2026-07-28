from __future__ import annotations

from typing import Any, Mapping

from trove_core.application.commands import SyncCommand, TroveCommands
from trove_core.application.operation_journal import OperationConflict
from trove_core.approvals import ApprovalManager, ApprovalRequired, ApprovalValidationError
from trove_core.wechat.files import archive_approval_payload

from .base import HandlerOutcome


def _operation_failure(exc: BaseException) -> HandlerOutcome:
    return HandlerOutcome.failure(
        str(getattr(exc, 'code', 'operation_failed')),
        'The journaled operation could not be completed.',
        retryable=isinstance(exc, OperationConflict),
    )


def _provider_accounts(context: Any) -> set[str]:
    owner = context.runtime_owner
    registry = getattr(owner, 'provider_registry', None) if owner is not None else None
    if registry is None:
        return set()
    rows = registry.accounts('wechat-source')
    return {
        str(item.get('account_id') or '')
        for item in rows
        if isinstance(item, Mapping) and str(item.get('account_id') or '')
    }


def sync(context: Any, payload: Mapping[str, Any], *, request_id: str) -> HandlerOutcome:
    """Run one bounded synchronous sync under the durable operation journal."""

    requested = tuple(str(value) for value in (payload.get('account_ids') or ()))
    available = _provider_accounts(context)
    selected = requested or tuple(sorted(available))
    if not available or not selected or set(selected) - available:
        return HandlerOutcome.failure(
            'account_scope_unavailable',
            'One or more selected source accounts are unavailable.',
            details={
                'requested_count': len(selected),
                'available_count': len(available),
            },
            next={'capability': 'trove.provider_status', 'action': 'inspect_accounts'},
        )
    try:
        record, replayed = context.operations.start(
            'trove.sync',
            payload,
            idempotency_key=str(payload['idempotency_key']),
            replay_policy='journaled',
        )
    except OperationConflict as exc:
        return _operation_failure(exc)
    if replayed:
        return HandlerOutcome.success({'operation': record.to_dict(), 'replayed': True})
    try:
        running = context.operations.journal.transition(
            record.operation_id,
            expected_states={'pending'},
            state='running',
            stage='syncing',
            owner='daemon',
        )
        report = TroveCommands(context.config).sync(SyncCommand(
            account_ids=selected,
            full=bool(payload.get('full', False)),
            media_discovery_mode='message_delta',
            profile_refresh_budget=0,
        ))
        if not bool(report.get('ok')):
            failed = context.operations.journal.transition(
                running.operation_id,
                expected_states={'running'},
                state='failed',
                stage='terminal',
                owner='none',
                error={
                    'code': 'sync_failed',
                    'retryable': report.get('status') in {'locked', 'retry_required'},
                    'status': str(report.get('status') or 'failed'),
                },
            )
            return HandlerOutcome.failure(
                'sync_failed',
                'The selected-account sync did not complete.',
                retryable=bool(failed.error and failed.error.get('retryable')),
                details={'operation': failed.to_dict()},
            )
        completed = context.operations.journal.transition(
            running.operation_id,
            expected_states={'running'},
            state='completed',
            stage='terminal',
            owner='none',
            result=dict(report),
        )
    except BaseException as exc:
        try:
            context.operations.journal.transition(
                record.operation_id,
                expected_states={'pending', 'running'},
                state='failed',
                stage='terminal',
                owner='none',
                error={
                    'code': 'sync_failed',
                    'retryable': False,
                    'failure_type': exc.__class__.__name__,
                },
            )
        except Exception:
            pass
        return _operation_failure(exc)
    return HandlerOutcome.success({'operation': completed.to_dict(), 'replayed': False})


def files_export(context: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    """Request or consume an exact human-approved local file export."""

    from trove_core.application.sensitive_commands import execute_files_archive

    selection = {
        'asset_ids': list(payload['selection']),
        'account_id': payload.get('account_id'),
    }
    try:
        approval_payload = archive_approval_payload(
            context.config,
            selection=selection,
            dest_dir=str(payload['destination']),
            mode='copy',
        )
        approval_id = payload.get('approval_id')
        if not approval_id:
            requested = context.approvals.request(
                'files_archive',
                'local-file-export',
                approval_payload,
            )
            approval = requested['approval']
            return HandlerOutcome.failure(
                'approval_required',
                'Exact human approval is required before exporting local files.',
                details={'approval': approval},
                next={
                    'capability': 'trove.files_export',
                    'action': 'retry_with_approval_id',
                    'approval_id': approval['approval_id'],
                },
            )
        grant = ApprovalManager(context.config.root).consume(
            'files_archive',
            'local-file-export',
            approval_payload,
            approval_id=str(approval_id),
        )
        report = execute_files_archive(
            context.config.root,
            selection=selection,
            dest_dir=str(payload['destination']),
            mode='copy',
            approval_grant=grant,
        )
    except (ApprovalRequired, ApprovalValidationError) as exc:
        return HandlerOutcome.failure(
            str(getattr(exc, 'code', 'approval_required')),
            'The export approval is missing, expired, mismatched, or already consumed.',
        )
    except (OSError, TypeError, ValueError) as exc:
        return HandlerOutcome.failure(
            'invalid_export_request',
            'The exact file export request is invalid.',
            details={'failure_type': exc.__class__.__name__},
        )
    return HandlerOutcome.success(dict(report))


__all__ = ['files_export', 'sync']
