"""Shared client for CLI and MCP adapters."""

from .client import TroveClient, TroveClientError
from .product_config import resolve_vault_root

__all__ = ['TroveClient', 'TroveClientError', 'resolve_vault_root']
