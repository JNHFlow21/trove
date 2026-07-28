#!/usr/bin/env python3
"""Safety preflight for creating the TROVE product workspace."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

APPROVED_TARGET = Path.home() / "Trove" / "trove"


def nearest_git_root(path: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def is_icloud_or_knowledge_os(path: Path) -> bool:
    text = str(path)
    return "/Library/Mobile Documents/" in text or "/Knowledge_OS/" in text or text.endswith("/Knowledge_OS")


def inspect_target(target: Path, owner: str | None = None, cwd: Path | None = None) -> dict:
    target = target.expanduser().resolve()
    cwd = (cwd or Path.cwd()).resolve()
    parent = target if target.exists() and target.is_dir() else target.parent
    parent.mkdir(parents=True, exist_ok=True) if target == APPROVED_TARGET and not parent.exists() else None
    git_root = nearest_git_root(parent)
    report = {
        "target_path": str(target),
        "current_cwd": str(cwd),
        "nearest_parent_git_root": git_root,
        "is_in_icloud_or_knowledge_os": is_icloud_or_knowledge_os(target),
        "planned_github_owner": owner or os.environ.get("TROVE_GITHUB_OWNER", "<unknown>"),
        "approved_target": str(APPROVED_TARGET),
    }
    errors: list[str] = []
    if target != APPROVED_TARGET:
        errors.append(f"target must resolve to {APPROVED_TARGET}")
    if report["is_in_icloud_or_knowledge_os"]:
        errors.append("target is under iCloud or Knowledge_OS")
    if git_root and Path(git_root).resolve() != target:
        errors.append(f"target is inside existing git root {git_root}")
    report["ok"] = not errors
    report["errors"] = errors
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(APPROVED_TARGET))
    parser.add_argument("--github-owner", default=os.environ.get("TROVE_GITHUB_OWNER"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = inspect_target(Path(args.target), owner=args.github_owner)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("TROVE migration safety preflight")
        print(f"target path: {report['target_path']}")
        print(f"current cwd: {report['current_cwd']}")
        print(f"nearest parent git root: {report['nearest_parent_git_root'] or '<none>'}")
        print(f"is in iCloud / Knowledge_OS: {str(report['is_in_icloud_or_knowledge_os']).lower()}")
        print(f"planned GitHub owner: {report['planned_github_owner']}")
        print(f"approved target: {report['approved_target']}")
        print(f"ok: {str(report['ok']).lower()}")
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
