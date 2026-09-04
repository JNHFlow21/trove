from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from trove_core.vault.config import VaultConfig
from trove_protocol.capabilities import CATALOG, CATALOG_BY_ID, CapabilitySpec, validate_input
from trove_protocol.envelope import Envelope, sanitize_untrusted_evidence
from trove_protocol.errors import ErrorDetail

from .approval_control import ApprovalControl
from .handlers.base import HandlerOutcome
from .operation_journal import OperationJournal
from .operations import OperationService


CapabilityHandler = Callable[['DispatchContext', Mapping[str, Any], str], HandlerOutcome]
EntitlementCheck = Callable[[CapabilitySpec], bool]


@dataclass
class DispatchContext:
    config: VaultConfig
    operations: OperationService
    approvals: ApprovalControl
    continuations: dict[str, Callable[..., Mapping[str, Any]]] = field(default_factory=dict)
    entitlement_check: EntitlementCheck = lambda _spec: True
    runtime_owner: Any | None = None
    installed_capabilities: frozenset[str] = frozenset()


class CapabilityDispatcher:
    def __init__(self, context: DispatchContext, handlers: Mapping[str, CapabilityHandler]):
        self.context = context
        self.handlers = dict(handlers)

    def dispatch(
        self,
        capability_id: str,
        payload: Mapping[str, Any],
        *,
        request_id: str,
        response_budget: int | None = None,
    ) -> dict[str, Any]:
        spec = CATALOG_BY_ID.get(capability_id)
        if spec is None:
            return self._failure(
                request_id,
                'unknown_capability',
                'Capability is not in the reviewed protocol catalog.',
            )
        try:
            normalized = validate_input(spec, payload)
        except ValueError as exc:
            return self._failure(request_id, 'invalid_request', str(exc))
        try:
            allowed = self.context.entitlement_check(spec)
        except Exception:
            allowed = False
        if not allowed:
            return self._failure(
                request_id,
                'capability_unavailable',
                'Capability is not entitled for this runtime.',
                details={'capability': capability_id},
                next={
                    'capability': 'trove.capabilities',
                    'action': 'inspect_entitlement',
                },
            )
        handler = self.handlers.get(capability_id)
        if handler is None:
            return self._failure(
                request_id,
                'capability_unavailable',
                'Capability has no installed handler.',
                details={'capability': capability_id},
            )
        try:
            outcome = handler(self.context, normalized, request_id)
        except Exception as exc:
            # The daemon logs the concrete exception locally.  The Agent-facing
            # contract never returns a traceback, path, or exception message.
            return self._failure(
                request_id,
                'capability_unavailable',
                'Capability failed before producing a result.',
                retryable=False,
                details={'capability': capability_id, 'failure_type': exc.__class__.__name__},
            )
        if not outcome.ok:
            return self._failure(
                request_id,
                str(outcome.error_code),
                str(outcome.error_message or 'Capability failed.'),
                retryable=outcome.retryable,
                details=outcome.error_details,
                next=outcome.next,
            )
        if (
            spec.replay_policy != 'read'
            and spec.risk != 'approval_request'
            and self.context.runtime_owner is not None
            and callable(getattr(self.context.runtime_owner, 'mark_generation_dirty', None))
        ):
            self.context.runtime_owner.mark_generation_dirty()
        data = dict(outcome.data or {})
        page = outcome.page
        coverage = outcome.coverage
        if spec.paginated and page is None:
            page = {'has_more': False}
            coverage = {'state': 'complete'}
        if spec.trust_class == 'untrusted_evidence':
            data = sanitize_untrusted_evidence(data)
            provenance = {
                'trust': 'untrusted_evidence',
                'source_type': 'vault',
            }
            account_id = normalized.get('account_id')
            if account_id:
                provenance['account_id'] = account_id
        else:
            provenance = None
        envelope = Envelope.success(
            data,
            request_id=request_id,
            page=page,
            coverage=coverage,
            provenance=provenance,
        ).to_dict()
        budget = spec.response_budget if response_budget is None else min(response_budget, spec.response_budget)
        encoded = json.dumps(envelope, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        if len(encoded) > budget:
            return self._failure(
                request_id,
                'response_too_large',
                'Result exceeded the reviewed response budget.',
                details={'budget_bytes': budget},
                next={'capability': capability_id, 'action': 'retry_with_narrower_scope'},
            )
        return envelope

    @staticmethod
    def _failure(
        request_id: str,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        next: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_request_id = request_id if isinstance(request_id, str) and request_id else 'invalid-request'
        return Envelope.failure(
            ErrorDetail(code, retryable=retryable, details=dict(details or {}), message=message),
            request_id=safe_request_id,
            next=next,
        ).to_dict()


def _unavailable(spec: CapabilitySpec) -> CapabilityHandler:
    def handler(_context: DispatchContext, _payload: Mapping[str, Any], _request_id: str) -> HandlerOutcome:
        details: dict[str, Any] = {'capability': spec.capability_id}
        next_action = None
        if spec.provider_requirement:
            details['provider_requirement'] = spec.provider_requirement
            next_action = {
                'capability': 'trove.provider_status',
                'action': 'inspect_provider',
            }
        return HandlerOutcome.failure(
            'capability_unavailable',
            'Capability is not available in the current runtime.',
            details=details,
            next=next_action,
        )

    return handler


def build_default_dispatcher(
    config: VaultConfig | str | Path,
    *,
    entitlement_check: EntitlementCheck | None = None,
    runtime_owner: Any | None = None,
) -> CapabilityDispatcher:
    from .handlers import mutations, operations as operation_handlers
    from .handlers import favorites, media_plan, message_kinds, moments, observations, pending, queries, reply, stats, system

    cfg = config if isinstance(config, VaultConfig) else VaultConfig.resolve(str(config), env={})
    journal = OperationJournal(cfg.paths.sqlite_path)
    context = DispatchContext(
        config=cfg,
        operations=OperationService(journal),
        approvals=ApprovalControl(cfg.root),
        entitlement_check=entitlement_check or (lambda _spec: True),
        runtime_owner=runtime_owner,
    )
    handlers: dict[str, CapabilityHandler] = {
        spec.capability_id: _unavailable(spec) for spec in CATALOG
    }

    def query(function: Callable[[Any, Mapping[str, Any]], HandlerOutcome]) -> CapabilityHandler:
        return lambda ctx, payload, _request_id: function(ctx.runtime_owner or ctx.config, payload)

    handlers.update({
        'trove.capabilities': lambda ctx, payload, _request_id: system.capabilities(
            ctx.runtime_owner or ctx.config,
            payload,
            installed_capabilities=ctx.installed_capabilities,
        ),
        'trove.resolve': query(system.resolve),
        'trove.diagnostics': query(system.diagnostics),
        'trove.provider_status': query(system.provider_status),
        'trove.recall': query(queries.recall),
        'trove.group_summary': query(queries.group_summary),
        'trove.search': query(queries.search),
        'trove.context': query(queries.context),
        'trove.profile': query(queries.profile),
        'trove.files_list': query(queries.files_list),
        'trove.favorites_list': query(favorites.favorites_list),
        'trove.moment_timeline': query(moments.moment_timeline),
        'trove.moment_interactions': query(moments.moment_interactions),
        'trove.message_stats': query(stats.message_stats),
        'trove.pending_replies': query(pending.pending_replies),
        'trove.messages_by_kind': query(message_kinds.messages_by_kind),
        'trove.reply_status': query(reply.status),
        'trove.reply_reviews': query(reply.reviews),
        'trove.reply_activity': query(reply.activity),
        'trove.media_fetch': query(queries.media_fetch),
        'trove.media_enrich_plan': query(media_plan.media_enrich_plan),
        'trove.observe_add': lambda ctx, payload, _request_id: observations.add(ctx, payload),
        'trove.observe_list': lambda ctx, payload, _request_id: observations.list_observations(ctx, payload),
        'trove.operation_status': lambda ctx, payload, _request_id: operation_handlers.status(ctx, payload),
        'trove.operation_continue': lambda ctx, payload, _request_id: operation_handlers.continue_operation(ctx, payload),
        'trove.operation_cancel': lambda ctx, payload, _request_id: operation_handlers.cancel(ctx, payload),
        'trove.sync': lambda ctx, payload, request_id: mutations.sync(
            ctx, payload, request_id=request_id,
        ),
        'trove.files_export': lambda ctx, payload, _request_id: mutations.files_export(ctx, payload),
        'trove.media_enrich': lambda ctx, payload, request_id: operation_handlers.start_operation(
            ctx, 'trove.media_enrich', payload, request_id=request_id,
        ),
        'trove.profile_build': lambda ctx, payload, request_id: operation_handlers.start_operation(
            ctx, 'trove.profile_build', payload, request_id=request_id,
        ),
        'trove.approval_request': lambda ctx, payload, _request_id: HandlerOutcome.success(
            ctx.approvals.request(
                str(payload['action']),
                str(payload['danger_class']),
                payload['payload'],
            )
        ),
        'trove.approval_status': lambda ctx, payload, _request_id: HandlerOutcome.success(
            ctx.approvals.status(str(payload['approval_id']))
        ),
    })
    context.installed_capabilities = frozenset({
        'trove.capabilities', 'trove.resolve', 'trove.diagnostics',
        'trove.provider_status', 'trove.recall', 'trove.group_summary',
        'trove.search', 'trove.context', 'trove.profile', 'trove.files_list',
        'trove.favorites_list',
        'trove.moment_timeline', 'trove.moment_interactions',
        'trove.message_stats',
        'trove.pending_replies', 'trove.messages_by_kind',
        'trove.reply_status', 'trove.reply_reviews', 'trove.reply_activity',
        'trove.media_fetch', 'trove.media_enrich_plan', 'trove.observe_add', 'trove.observe_list',
        'trove.operation_status', 'trove.operation_continue',
        'trove.operation_cancel', 'trove.media_enrich', 'trove.profile_build',
        'trove.approval_request', 'trove.approval_status', 'trove.sync',
        'trove.files_export',
    })
    return CapabilityDispatcher(context, handlers)


__all__ = [
    'CapabilityDispatcher', 'CapabilityHandler', 'DispatchContext',
    'EntitlementCheck', 'build_default_dispatcher',
]
