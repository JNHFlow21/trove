from __future__ import annotations

from typing import Mapping

from trove_protocol.capabilities import CATALOG_BY_ID
from trove_protocol.experimental.entitlements import EntitlementDecision


class AllowAllEntitlementProvider:
    """Internal v1 implementation; it never weakens approval requirements."""

    def check(self, capability_id: str) -> EntitlementDecision:
        if not isinstance(capability_id, str) or capability_id not in CATALOG_BY_ID:
            return EntitlementDecision.deny('invalid_capability')
        return EntitlementDecision.allow()


class StaticEntitlementProvider:
    """Test-only deny seam kept outside the public protocol catalog."""

    def __init__(self, decisions: Mapping[str, bool]):
        self._decisions = {str(key): bool(value) for key, value in decisions.items()}

    def check(self, capability_id: str) -> EntitlementDecision:
        if capability_id not in CATALOG_BY_ID:
            return EntitlementDecision.deny('invalid_capability')
        if self._decisions.get(capability_id, True):
            return EntitlementDecision.allow()
        return EntitlementDecision.deny('entitlement_denied')


def dispatcher_entitlement_check(provider: object):
    def check(spec) -> bool:
        decision = provider.check(spec.capability_id)
        return bool(decision.allowed and decision.approval_still_required is True)

    return check


__all__ = [
    'AllowAllEntitlementProvider', 'StaticEntitlementProvider',
    'dispatcher_entitlement_check',
]
