from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


EXPERIMENTAL = True


@dataclass(frozen=True)
class EntitlementDecision:
    allowed: bool
    reason_code: str | None = None
    approval_still_required: bool = True

    @classmethod
    def allow(cls) -> 'EntitlementDecision':
        return cls(True, approval_still_required=True)

    @classmethod
    def deny(cls, reason_code: str) -> 'EntitlementDecision':
        return cls(False, reason_code=reason_code, approval_still_required=True)


class EntitlementProvider(Protocol):
    def check(self, capability_id: str) -> EntitlementDecision: ...


__all__ = ['EXPERIMENTAL', 'EntitlementDecision', 'EntitlementProvider']
