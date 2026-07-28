#!/usr/bin/env python3
"""Report the active TROVE project Python runtime without touching Vault data."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATHS = [
    ROOT / "packages" / "trove_protocol",
    ROOT / "packages" / "trove_core",
    ROOT / "packages" / "trove_client",
    ROOT / "packages" / "trove_daemon",
    ROOT / "packages" / "trove_provider_wechat",
    ROOT / "packages" / "trove_cli",
    ROOT / "packages" / "trove_mcp",
]


def import_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False, "origin": None}
    origin = spec.origin or "built-in"
    return {"available": True, "origin": origin if str(ROOT) in origin else "venv-or-system-site"}


def payload() -> dict[str, Any]:
    expected = ROOT / ".venv" / "bin" / "python"
    expected_venv = ROOT / ".venv"
    executable = Path(sys.executable)
    project_path_text = [str(p) for p in PROJECT_PATHS]
    sys_path_text = [str(Path(p).resolve()) for p in sys.path if p]
    project_paths_active = all(str(p.resolve()) in sys_path_text for p in PROJECT_PATHS)
    uses_project_venv = Path(sys.prefix).resolve() == expected_venv.resolve()
    return {
        "ok": uses_project_venv and project_paths_active,
        "repo": str(ROOT),
        "python_executable": str(executable),
        "python_executable_resolved": str(executable.resolve()),
        "expected_project_python": str(expected),
        "venv_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "uses_project_python": uses_project_venv,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        "project_pythonpath": project_path_text,
        "project_paths_active": project_paths_active,
        "imports": {
            "trove_protocol": import_status("trove_protocol"),
            "trove_core": import_status("trove_core"),
            "trove_client": import_status("trove_client"),
            "trove_daemon": import_status("trove_daemon"),
            "trove_provider_wechat": import_status("trove_provider_wechat"),
            "trove_cli": import_status("trove_cli"),
            "trove_mcp": import_status("trove_mcp"),
            "zvec": import_status("zvec"),
            "mcp": import_status("mcp"),
            "httpx": import_status("httpx"),
            "sentence_transformers": import_status("sentence_transformers"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = payload()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"TROVE project runtime ok: {str(data['ok']).lower()}")
        print(f"python: {data['python_executable']}")
        print(f"zvec: {str(data['imports']['zvec']['available']).lower()}")
    return 0 if data["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
