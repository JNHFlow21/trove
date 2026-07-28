from __future__ import annotations

from pathlib import Path
from typing import Any

from trove_client.control import decide_approval


def decide(vault_root: str | Path, approval_id: str, decision: str, *, note: str | None = None) -> dict[str, Any]:
    """Human-only control path. There is deliberately no stdin/--yes/env seam."""
    return decide_approval(vault_root, approval_id, decision, note=note)


__all__ = ['decide']
