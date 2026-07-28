from __future__ import annotations

from typing import Any, Mapping

from trove_client import TroveClient
from trove_protocol.capabilities import CapabilitySpec, validate_input


def execute_query(
    client: TroveClient,
    spec: CapabilitySpec,
    payload: Mapping[str, Any],
    *,
    request_id: str,
    timeout: float,
) -> dict[str, Any]:
    normalized = validate_input(spec, payload)
    return client.call(
        spec.capability_id, normalized,
        request_id=request_id, timeout=timeout,
        response_budget=spec.response_budget,
    )


__all__ = ['execute_query']
