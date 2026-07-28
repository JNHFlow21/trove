from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import ErrorDetail


PROTOCOL_VERSION = 'trove/1'
_ALLOWED_TOP_LEVEL = frozenset({
    'protocol', 'request_id', 'ok', 'data', 'error', 'page', 'coverage',
    'next', 'warnings', 'provenance',
})
_RESERVED_EVIDENCE_KEYS = frozenset({
    'next', 'approval', 'approval_status', 'capability', 'capability_id',
    'action', 'action_arguments', 'control', 'ok', 'error', 'protocol',
})


class EnvelopeValidationError(ValueError):
    pass


def sanitize_untrusted_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            (f'evidence_{key}' if str(key).lower() in _RESERVED_EVIDENCE_KEYS else str(key)): sanitize_untrusted_evidence(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_untrusted_evidence(item) for item in value]
    return value


@dataclass(frozen=True)
class Envelope:
    request_id: str
    ok: bool
    data: Mapping[str, Any] | None = None
    error: ErrorDetail | None = None
    page: Mapping[str, Any] | None = None
    coverage: Mapping[str, Any] | None = None
    next: Mapping[str, Any] | None = None
    warnings: Sequence[Mapping[str, Any]] = ()
    provenance: Mapping[str, Any] | None = None
    protocol: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL_VERSION:
            raise EnvelopeValidationError('protocol mismatch')
        if not isinstance(self.request_id, str) or not self.request_id:
            raise EnvelopeValidationError('request_id is required')
        if type(self.ok) is not bool:
            raise EnvelopeValidationError('ok must be boolean')
        if self.ok != (self.data is not None):
            raise EnvelopeValidationError('success requires data and failure forbids it')
        if self.ok == (self.error is not None):
            raise EnvelopeValidationError('envelope requires exactly one of data/error')
        if (self.page is None) != (self.coverage is None):
            raise EnvelopeValidationError('page and coverage must appear together')
        if self.page is not None:
            if type(self.page.get('has_more')) is not bool:
                raise EnvelopeValidationError('page.has_more must be boolean')
            if self.coverage.get('state') not in {'complete', 'partial'}:
                raise EnvelopeValidationError('coverage.state must be complete or partial')
        if self.next is not None and not isinstance(self.next, Mapping):
            raise EnvelopeValidationError('next must be an object')

    @classmethod
    def success(
        cls,
        data: Mapping[str, Any],
        *,
        request_id: str,
        page: Mapping[str, Any] | None = None,
        coverage: Mapping[str, Any] | None = None,
        warnings: Sequence[Mapping[str, Any]] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> 'Envelope':
        return cls(
            request_id=request_id,
            ok=True,
            data=dict(data),
            page=page,
            coverage=coverage,
            warnings=warnings,
            provenance=provenance,
        )

    @classmethod
    def success_evidence(
        cls,
        data: Mapping[str, Any],
        *,
        request_id: str,
        source_type: str,
        account_id: str,
        page: Mapping[str, Any] | None = None,
        coverage: Mapping[str, Any] | None = None,
    ) -> 'Envelope':
        return cls.success(
            sanitize_untrusted_evidence(data),
            request_id=request_id,
            page=page,
            coverage=coverage,
            provenance={
                'trust': 'untrusted_evidence',
                'source_type': source_type,
                'account_id': account_id,
            },
        )

    @classmethod
    def failure(
        cls,
        error: ErrorDetail,
        *,
        request_id: str,
        next: Mapping[str, Any] | None = None,
    ) -> 'Envelope':
        return cls(request_id=request_id, ok=False, error=error, next=next)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'protocol': self.protocol,
            'request_id': self.request_id,
            'ok': self.ok,
        }
        if self.data is not None:
            payload['data'] = dict(self.data)
        if self.error is not None:
            payload['error'] = self.error.to_dict()
        if self.page is not None:
            payload['page'] = dict(self.page)
            payload['coverage'] = dict(self.coverage or {})
        if self.next is not None:
            payload['next'] = dict(self.next)
        if self.warnings:
            payload['warnings'] = [dict(item) for item in self.warnings]
        if self.provenance is not None:
            payload['provenance'] = dict(self.provenance)
        return payload


def parse_envelope(payload: Mapping[str, Any]) -> Envelope:
    if not isinstance(payload, Mapping):
        raise EnvelopeValidationError('envelope must be an object')
    unknown = set(payload) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise EnvelopeValidationError(f'unknown envelope fields: {sorted(unknown)}')
    error_payload = payload.get('error')
    error = None
    if error_payload is not None:
        if not isinstance(error_payload, Mapping):
            raise EnvelopeValidationError('error must be an object')
        if 'code' not in error_payload or 'retryable' not in error_payload:
            raise EnvelopeValidationError('error requires code and retryable')
        if set(error_payload) - {'code', 'retryable', 'message', 'details'}:
            raise EnvelopeValidationError('unknown error fields')
        try:
            error = ErrorDetail(
                str(error_payload['code']),
                retryable=error_payload['retryable'],
                message=error_payload.get('message'),
                details=error_payload.get('details') or {},
            )
        except ValueError as exc:
            raise EnvelopeValidationError(str(exc)) from exc
    return Envelope(
        protocol=str(payload.get('protocol') or ''),
        request_id=str(payload.get('request_id') or ''),
        ok=payload.get('ok'),
        data=payload.get('data'),
        error=error,
        page=payload.get('page'),
        coverage=payload.get('coverage'),
        next=payload.get('next'),
        warnings=payload.get('warnings') or (),
        provenance=payload.get('provenance'),
    )


__all__ = [
    'Envelope', 'EnvelopeValidationError', 'PROTOCOL_VERSION', 'parse_envelope',
    'sanitize_untrusted_evidence',
]
