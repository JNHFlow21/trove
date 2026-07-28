from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ErrorDetail:
    code: str
    retryable: bool
    details: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError('error code must be a non-empty string')
        if type(self.retryable) is not bool:
            raise ValueError('error retryable must be boolean')
        if not isinstance(self.details, Mapping):
            raise ValueError('error details must be an object')

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'code': self.code,
            'retryable': self.retryable,
        }
        if self.message:
            payload['message'] = self.message
        if self.details:
            payload['details'] = dict(self.details)
        return payload


class ProtocolError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ):
        self.code = code
        self.retryable = retryable
        self.details = dict(details or {})
        super().__init__(message)

    def to_detail(self) -> ErrorDetail:
        return ErrorDetail(
            self.code,
            retryable=self.retryable,
            details=self.details,
            message=str(self),
        )


ERROR_CODES = frozenset({
    'invalid_request',
    'protocol_mismatch',
    'unknown_capability',
    'frame_too_large',
    'partial_frame',
    'invalid_frame',
    'deadline_expired',
    'response_too_large',
    'no_results',
    'ambiguous_target',
    'approval_required',
    'busy',
    'timeout',
    'capability_unavailable',
    'provider_unavailable',
    'version_incompatible',
    'vault_identity_mismatch',
    'catalog_mismatch',
    'peer_unauthorized',
    'peer_credential_unavailable',
    'platform_unsupported',
    'daemon_unavailable',
    'cursor_invalid',
    'cursor_expired',
    'cursor_stale',
    'cursor_mismatch',
})


__all__ = ['ERROR_CODES', 'ErrorDetail', 'ProtocolError']
