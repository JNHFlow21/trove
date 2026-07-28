#!/usr/bin/env python3
"""Allowlist-only migration helper from the exploration project.

This helper copies safe source assets only. It never moves or deletes old runtime data.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)
import argparse
import fnmatch
import json
import shutil
from pathlib import Path

DEFAULT_ALLOWLIST = [
    'PROJECT.md', 'AGENTS.md', 'README.md', 'index.md', 'log.md',
    'docs/**/*.md', 'wiki/**/*.md', 'raw/**/*.md',
    'wechat-chat-analysis/requirements.txt',
    'wechat-chat-analysis/scripts/vault_config.py',
    'wechat-chat-analysis/scripts/vault_cli.py',
    'wechat-chat-analysis/scripts/vault_index.py',
    'wechat-chat-analysis/tests/test_vault_index.py',
]
DENYLIST = [
    '*/decrypted/*', '*/logs/*', '*/output/*', '*/.venv/*', '*/node_modules/*', '*/desktop/dist/*',
    '*key_store.json', '*.db', '*.sqlite', '*.sqlite3', '*-wal', '*-shm', '*messages.jsonl', '*.zip', '*.tar', '*.gz',
    '*wechat_auto_reply.py', '*auto_reply_*', '*wechat_sender.swift', '*wechat_auto_reply_helper.swift', '*send_wechat_message.applescript',
    '*LaunchAgent*', '*.plist',
]


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def eligible_files(source: Path, allowlist: list[str] = DEFAULT_ALLOWLIST) -> list[Path]:
    files: list[Path] = []
    for pattern in allowlist:
        files.extend(p for p in source.glob(pattern) if p.is_file())
    unique = []
    seen = set()
    for path in files:
        rel = path.relative_to(source).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        if matches_any(rel, DENYLIST):
            continue
        unique.append(path)
    return sorted(unique)


def migrate(source: Path, target: Path, dry_run: bool = False) -> dict:
    source = source.resolve()
    target = target.resolve()
    copied = []
    skipped = []
    for path in eligible_files(source):
        rel = path.relative_to(source)
        if matches_any(rel.as_posix(), DENYLIST):
            skipped.append(rel.as_posix())
            continue
        copied.append(rel.as_posix())
        if not dry_run:
            dest = target / 'migration-reference' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    return {"source": str(source), "target": str(target), "copied": copied, "skipped": skipped, "deleted_from_source": []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Old exploration project root')
    parser.add_argument('--target', default='.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    report = migrate(Path(args.source), Path(args.target), args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
