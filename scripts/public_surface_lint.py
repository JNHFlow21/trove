#!/usr/bin/env python3
"""Reject product surfaces that are outside the reviewed TROVE v1 boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PREFIXES = (
    'apps/web_console/',
    'packages/trove_api/',
)
FORBIDDEN_PATHS = frozenset({
    'package.json',
    'package-lock.json',
    'docs/adr/0001-web-console-source-of-truth.md',
    'docs/natural-language-guide.md',
    'tests/e2e/test_api_fixture_flow.py',
    'tests/e2e/test_web_console_fixture_flow.py',
    'tests/test_web_console_acceptance_contract.py',
})
PUBLIC_PACKAGE_PREFIXES = ('packages/trove_cli/trove_cli/', 'packages/trove_mcp/trove_mcp/')
LEGACY_PUBLIC_TOKENS = frozenset({
    'trove-api',
    'chat-recall',
    'customer-profile',
    'person-profile',
    'files-list',
    'media-fetch',
    'trove_chat_recall',
    'trove_customer_profile',
    'trove_person_profile',
    'trove-chat-recall',
})
EXPECTED_SCRIPTS = frozenset({'trove', 'trove-mcp', 'troved'})


def _git_snapshot(root: Path) -> dict[str, str] | None:
    listed = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '-z'],
        check=False,
        capture_output=True,
    )
    if listed.returncode:
        return None
    result: dict[str, str] = {}
    for raw_path in listed.stdout.split(b'\0'):
        if not raw_path:
            continue
        path = raw_path.decode('utf-8')
        result[path] = ''
        needs_content = (
            path == 'pyproject.toml'
            or path == 'packages/trove_daemon/trove_daemon/server.py'
            or path.startswith(PUBLIC_PACKAGE_PREFIXES)
            or path.startswith('skills/')
        )
        if not needs_content:
            continue
        shown = subprocess.run(
            ['git', '-C', str(root), 'show', f':{path}'],
            check=False,
            capture_output=True,
        )
        if shown.returncode == 0:
            result[path] = shown.stdout.decode('utf-8', errors='replace')
    return result


def _filesystem_snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in root.rglob('*'):
        if not path.is_file() or any(part in {'.git', '.venv', '__pycache__'} for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            result[relative] = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
    return result


def repository_snapshot(root: Path = ROOT) -> dict[str, str]:
    return _git_snapshot(root) or _filesystem_snapshot(root)


def _import_findings(path: str, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return [f'{path}: invalid Python syntax']
    findings: list[str] = []
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        for name in names:
            if name == 'trove_core' or name.startswith('trove_core.'):
                findings.append(f'{path}:{node.lineno}: public adapter imports trove_core')
            if name == 'trove_provider_wechat' or name.startswith('trove_provider_wechat.'):
                findings.append(f'{path}:{node.lineno}: public adapter imports a source provider')
            if name == 'trove_api' or name.startswith('trove_api.'):
                findings.append(f'{path}:{node.lineno}: public adapter imports removed API')
    return findings


def scan_snapshot(files: Mapping[str, str]) -> list[str]:
    findings: list[str] = []
    paths = set(files)
    for path in sorted(paths):
        if path in FORBIDDEN_PATHS or path.startswith(FORBIDDEN_PREFIXES):
            findings.append(f'{path}: forbidden legacy product surface')
        if path.endswith(('.js', '.mjs', '.cjs', '.ts', '.tsx')):
            findings.append(f'{path}: Node/JavaScript product dependency is forbidden')

    try:
        project = tomllib.loads(files['pyproject.toml'])
        scripts = frozenset((project.get('project') or {}).get('scripts') or {})
        if scripts != EXPECTED_SCRIPTS:
            findings.append(
                'pyproject.toml: public executables must be exactly '
                + ', '.join(sorted(EXPECTED_SCRIPTS))
            )
    except (KeyError, tomllib.TOMLDecodeError):
        findings.append('pyproject.toml: missing or invalid project metadata')

    for path, source in sorted(files.items()):
        if path.startswith(PUBLIC_PACKAGE_PREFIXES) and path.endswith('.py'):
            findings.extend(_import_findings(path, source))
        if (
            path.startswith(PUBLIC_PACKAGE_PREFIXES)
            or path.startswith('skills/')
        ) and not path.endswith('.pyc'):
            for token in sorted(LEGACY_PUBLIC_TOKENS):
                if token in source:
                    findings.append(f'{path}: contains legacy public token {json.dumps(token)}')
    daemon_source = files.get('packages/trove_daemon/trove_daemon/server.py', '')
    if 'AF_INET' in daemon_source or 'AF_INET6' in daemon_source:
        findings.append('packages/trove_daemon/trove_daemon/server.py: TCP listener is forbidden')
    return sorted(set(findings))


def scan_public_surface(root: Path = ROOT) -> list[str]:
    return scan_snapshot(repository_snapshot(root))


def main() -> int:
    findings = scan_public_surface()
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print('public surface lint passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
