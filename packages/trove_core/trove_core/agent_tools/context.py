from __future__ import annotations
from pathlib import Path

from trove_core.agent_tools.tools import import_status, model_status, provider_status, vault_status, vector_status


def build_agent_context(vault_root: str | Path) -> dict:
    """Return safe state for agent prompt injection.

    This intentionally excludes message bodies, snippets, full private paths, and credentials.
    """
    return {
        'vault': vault_status(vault_root),
        'import': import_status(vault_root),
        'model': model_status(),
        'vector': vector_status(vault_root),
        'providers': provider_status(),
        'raw_content_included': False,
    }
