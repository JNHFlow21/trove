from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class HandlerOutcome:
    data: Mapping[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: Mapping[str, Any] = field(default_factory=dict)
    retryable: bool = False
    page: Mapping[str, Any] | None = None
    coverage: Mapping[str, Any] | None = None
    next: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error_code is None

    @classmethod
    def success(
        cls,
        data: Mapping[str, Any],
        *,
        page: Mapping[str, Any] | None = None,
        coverage: Mapping[str, Any] | None = None,
    ) -> 'HandlerOutcome':
        return cls(data=dict(data), page=page, coverage=coverage)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
        next: Mapping[str, Any] | None = None,
    ) -> 'HandlerOutcome':
        return cls(
            error_code=code,
            error_message=message,
            retryable=retryable,
            error_details=dict(details or {}),
            next=next,
        )


def from_query_result(result: Any, *, paginated: bool = False) -> HandlerOutcome:
    if not bool(getattr(result, 'ok', False)):
        error = getattr(result, 'error', None)
        code = str(getattr(error, 'code', None) or getattr(result, 'code', None) or 'query_failed')
        if code in {'ambiguous_contact', 'ambiguous_conversation'}:
            code = 'ambiguous_target'
        details = dict(getattr(error, 'details', {}) or {})
        return HandlerOutcome.failure(
            code,
            str(getattr(error, 'message', None) or 'Capability query failed.'),
            details=details,
        )
    data = dict(getattr(result, 'data', {}) or {})
    legacy_coverage = data.pop('coverage', None)
    if paginated:
        has_more = bool((legacy_coverage or {}).get('has_more', False))
        page = {'has_more': has_more}
        if has_more:
            page['legacy_offset'] = (legacy_coverage or {}).get('next_offset')
        coverage = {
            'state': 'partial' if has_more else 'complete',
            'returned': (legacy_coverage or {}).get('returned'),
            'remaining': (legacy_coverage or {}).get('remaining'),
        }
        coverage = {key: value for key, value in coverage.items() if value is not None}
        return HandlerOutcome.success(data, page=page, coverage=coverage)
    return HandlerOutcome.success(data)


__all__ = ['HandlerOutcome', 'from_query_result']
