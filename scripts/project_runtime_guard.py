#!/usr/bin/env python3
"""Small trampoline that keeps direct script execution inside TROVE .venv."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_runtime(script_file: str) -> None:
    root = Path(script_file).resolve().parents[1]
    venv = (root / '.venv').resolve()
    if Path(sys.prefix).resolve() == venv:
        return
    if os.environ.get('TROVE_PROJECT_RUNTIME_GUARD') == '1':
        raise SystemExit('TROVE project runtime guard could not enter .venv; run bash scripts/bootstrap_runtime.sh')
    wrapper = root / 'scripts' / 'trove-python'
    if not wrapper.exists():
        raise SystemExit('missing TROVE runtime wrapper: scripts/trove-python')
    env = os.environ
    env['TROVE_PROJECT_RUNTIME_GUARD'] = '1'
    os.execv(str(wrapper), [str(wrapper), str(Path(script_file).resolve()), *sys.argv[1:]])
