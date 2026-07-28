from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from trove_core.approvals import ApprovalManager


class ApprovalControl:
    """Agent-facing request/status boundary with no decision authority."""

    def __init__(self, vault_root: str | Path):
        self._manager = ApprovalManager(vault_root)

    def request(
        self,
        action: str,
        danger_class: str,
        payload: Mapping[str, Any],
        *,
        ttl_minutes: int = 60,
    ) -> dict[str, Any]:
        record = self._manager.request(
            action,
            danger_class,
            dict(payload),
            ttl_minutes=ttl_minutes,
        )
        return {'approval': record.to_dict()}

    def status(self, approval_id: str) -> dict[str, Any]:
        return {'approval': self._manager.load(approval_id).to_dict()}

    def list(self, *, limit: int = 50) -> dict[str, Any]:
        return {'approvals': self._manager.list(limit=limit)}


__all__ = ['ApprovalControl']
