from __future__ import annotations

from typing import Mapping

from trove_protocol.capabilities import PACK_ORDER, capabilities_for_pack


class MCPPackError(ValueError):
    code = 'mcp_pack_invalid'


def resolve_pack(value: str | None, env: Mapping[str, str] | None = None) -> str:
    selected = value
    if selected is None and env is not None:
        selected = env.get('TROVE_MCP_PACK')
    pack = str(selected or 'standard').strip().lower()
    if pack not in PACK_ORDER:
        raise MCPPackError('MCP pack must be standard, operations, or admin')
    return pack


def tool_names(pack: str) -> frozenset[str]:
    return frozenset(spec.mcp_name for spec in capabilities_for_pack(resolve_pack(pack)))


__all__ = ['MCPPackError', 'resolve_pack', 'tool_names']
