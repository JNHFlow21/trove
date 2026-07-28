from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for rel in (
    'packages/trove_protocol', 'packages/trove_core', 'packages/trove_client',
    'packages/trove_daemon', 'packages/trove_provider_wechat',
    'packages/trove_cli', 'packages/trove_mcp',
):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)
