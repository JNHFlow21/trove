#!/usr/bin/env python3
"""Content and filename privacy scanner for TROVE."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)
import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import tokenize
from io import StringIO
from pathlib import Path

NAME_DENY_RE = re.compile(r'(^|/)(decrypted|logs|output)(/|$)|key_store\.json$|wechat_auto_reply\.py$|auto_reply_[^/]*|wechat_sender\.swift$|wechat_auto_reply_helper\.swift$|send_wechat_message\.applescript$|(^|/)LaunchAgents(/|$)|\.(db|sqlite|sqlite3)$|(-wal|-shm|\.db-wal|\.db-shm)$', re.I)
RUNTIME_CLOUD_DENY_RE = re.compile(r'(^|/)(transcripts?|provider[_-]?payloads?|raw[_-]?provider|proof/raw|jobs/raw|media[_-]?understanding)(/|$)|(^|/)(raw[_-]?transcript|provider[_-]?payload|image[_-]?observations?|voice[_-]?transcripts?|media[_-]?understanding)\.(json|jsonl|txt|ndjson)$', re.I)
RUNTIME_DERIVED_DENY_RE = re.compile(
    r'(^|/)(media/(previews|materialized|decoded|tmp)|approvals|proof/lazy-profile-enrichment|profile[_-]?snapshots?|profile[_-]?enrichment)(/|$)'
    r'|(^|/)(profile[_-]?snapshot|profile[_-]?enrichment[_-]manifest)\.(json|jsonl|txt|ndjson)$',
    re.I,
)
MEDIA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.mp3', '.m4a', '.wav', '.amr', '.silk', '.mp4', '.mov', '.dat'}
LONG_HEX_RE = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{48,}(?![0-9a-fA-F])')
PRIVATE_PATH_RE = re.compile(r'/Users/[A-Za-z0-9_.-]+/[^\s\"\'<>)]*')
TOKEN_RE = re.compile(r'(?i)(api[_-]?key|secret|token)\s*[:=]\s*[\"\']?[A-Za-z0-9_\-]{24,}')
EMAIL_RE = re.compile(r'(?<![\w.+-])[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})(?![\w.-])', re.I)
CN_MOBILE_RE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
WECHAT_ID_RE = re.compile(r'\bwxid_[A-Za-z0-9]{12,}\b')
SYNTHETIC_ID_MARKERS = ('fixture', 'sample', 'test', 'unknown', 'owner', 'example')
SAFE_EMAIL_DOMAINS = {'example.com', 'example.org', 'example.net', 'users.noreply.github.com'}
DOCS_REAL_WECHAT_CITATION_RE = re.compile(r'trove://wechat/(acct-[A-Za-z0-9_.-]+)/')
SQLITE_HEADER = b'SQLite format 3\x00'
SQLCIPHER_HINTS = (b'cipher_' + b'page_size', b'PRAGMA ' + b'key')
SKIP_DIRS = {'.git', '.venv', 'node_modules', '__pycache__', '.mypy_cache', '.pytest_cache', 'target', 'dist'}
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.toml', '.yml', '.yaml', '.js', '.ts', '.tsx', '.css', '.html', '.rs', '.sh', '.mjs'}
SOURCE_SAFE_EVIDENCE_PREFIXES = {
    'docs/perf/agent-runtime-budgets.json',
    'packages/trove_provider_wechat/trove_provider_wechat/manifest.json',
    'skills/manifest.json',
    'tests/fixtures/synthetic/evidence/',
    'tests/golden/trove_protocol_v1.json',
}
EVIDENCE_HASH_KEY_PARTS = {'hash', 'sha', 'sha256', 'commit', 'digest', 'generation', 'provenance'}
FORMAL_DOCS = frozenset({
    'README.md',
    'docs/architecture.md',
    'docs/architecture/application-boundary.md',
    'docs/mcp.md',
    'docs/capability-map.md',
    'docs/protocol.md',
    'docs/provider-sdk.md',
    'docs/operations.md',
    'docs/providers/wechat.md',
    'docs/testing.md',
    'docs/release.md',
})
FORMAL_DOC_LEGACY_TOKENS = (
    'trove-api', 'trove_api', 'trove-wechat', 'trove_cli.main',
    'chat-recall', 'customer-profile', 'person-profile', 'files-list',
    'web_console', 'web console', 'npm ', 'localhost:', '127.0.0.1:',
    'trove up', 'trove down', 'trove ps',
)


def _evidence_hash_context_is_safe(rel: str, path: Path, text: str) -> bool:
    """Allow full digests only in the reviewed source-safe evidence bundle.

    The broad long-hex detector remains active everywhere else. Inside the
    frozen bundle, JSON values are allowed only below explicitly hash-shaped
    keys; a long hex string under query/content/token remains a finding.
    """

    if not any(rel.startswith(prefix) for prefix in SOURCE_SAFE_EVIDENCE_PREFIXES):
        return False
    if path.suffix.lower() == '.md':
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if not LONG_HEX_RE.search(line):
                continue
            context = ' '.join(lines[max(0, index - 1):index + 1]).lower()
            if not any(part in context for part in EVIDENCE_HASH_KEY_PARTS):
                return False
        return True
    if path.suffix.lower() not in {'.json', '.jsonl'}:
        return False

    try:
        values = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix.lower() == '.jsonl' else [json.loads(text)]
    except (ValueError, TypeError):
        return False

    def safe(value, key_path: tuple[str, ...]) -> bool:
        if isinstance(value, dict):
            return all(safe(item, (*key_path, str(key))) for key, item in value.items())
        if isinstance(value, list):
            return all(safe(item, key_path) for item in value)
        if not isinstance(value, str) or not LONG_HEX_RE.search(value):
            return True
        normalized = '_'.join(key_path).lower().replace('-', '_')
        return any(part in normalized for part in EVIDENCE_HASH_KEY_PARTS)

    return all(safe(value, ()) for value in values)


def is_fixture_messages(path: Path) -> bool:
    parts = path.as_posix().split('/')
    return path.name == 'messages.jsonl' and 'fixtures' in parts and 'synthetic' in parts


def _string_values(path: Path, text: str):
    """Yield data-bearing strings without treating source numeric literals as phones."""

    suffix = path.suffix.lower()
    if suffix in {'.json', '.jsonl'}:
        try:
            values = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if suffix == '.jsonl'
                else [json.loads(text)]
            )
        except (ValueError, TypeError):
            yield text
            return

        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, str):
                yield value

        for value in values:
            yield from walk(value)
        return
    if suffix == '.py':
        try:
            for token in tokenize.generate_tokens(StringIO(text).readline):
                if token.type != tokenize.STRING:
                    continue
                try:
                    value = ast.literal_eval(token.string)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(value, str):
                    yield value
        except (tokenize.TokenError, IndentationError):
            yield text
        return
    yield text


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def staged_files(root: Path) -> list[Path]:
    try:
        out = subprocess.check_output(['git', '-C', str(root), 'diff', '--cached', '--name-only', '--diff-filter=ACMR'], text=True)
    except Exception:
        return list(iter_files(root))
    return [root / line.strip() for line in out.splitlines() if line.strip()]


def scan_file(root: Path, path: Path) -> list[str]:
    findings: list[str] = []
    rel = path.relative_to(root).as_posix()
    if NAME_DENY_RE.search(rel) and not is_fixture_messages(Path(rel)):
        findings.append(f'{rel}: denied sensitive filename/path')
    if RUNTIME_CLOUD_DENY_RE.search(rel):
        findings.append(f'{rel}: denied runtime cloud/transcript/provider payload path')
    if RUNTIME_DERIVED_DENY_RE.search(rel):
        findings.append(f'{rel}: denied runtime derived-data/cache/proof path')
    if path.suffix.lower() in MEDIA_SUFFIXES:
        findings.append(f'{rel}: raw media file is forbidden in source Git')
    if path.name == 'messages.jsonl' and not is_fixture_messages(Path(rel)):
        findings.append(f'{rel}: non-fixture messages.jsonl is forbidden')
    try:
        head = path.read_bytes()[:4096]
    except Exception as exc:
        findings.append(f'{rel}: cannot read file: {exc}')
        return findings
    if head.startswith(SQLITE_HEADER):
        findings.append(f'{rel}: SQLite header detected')
    if any(hint.lower() in head.lower() for hint in SQLCIPHER_HINTS):
        findings.append(f'{rel}: SQLCipher/key material hint detected')
    if path.suffix.lower() in TEXT_SUFFIXES or not b'\x00' in head:
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = head.decode('utf-8', errors='ignore')
        if PRIVATE_PATH_RE.search(text):
            findings.append(f'{rel}: private absolute path detected')
        if LONG_HEX_RE.search(text) and not _evidence_hash_context_is_safe(rel, path, text):
            findings.append(f'{rel}: long hex key-like value detected')
        if TOKEN_RE.search(text):
            findings.append(f'{rel}: token/key-like assignment detected')
        if any(
            match.group(1).lower() not in SAFE_EMAIL_DOMAINS
            for match in EMAIL_RE.finditer(text)
        ):
            findings.append(f'{rel}: non-example email address detected')
        strings = tuple(_string_values(path, text))
        if any(CN_MOBILE_RE.search(value) for value in strings):
            findings.append(f'{rel}: phone-like value detected')
        if any(
            any(
                not any(marker in value.lower() for marker in SYNTHETIC_ID_MARKERS)
                for value in WECHAT_ID_RE.findall(item)
            )
            for item in strings
        ):
            findings.append(f'{rel}: non-synthetic WeChat identifier detected')
        if rel == 'docs' or rel.startswith('docs/'):
            for match in DOCS_REAL_WECHAT_CITATION_RE.finditer(text):
                account_id = match.group(1)
                if account_id not in {'acct-work'}:
                    findings.append(f'{rel}: raw real WeChat citation detected in docs')
                    break
        if rel in FORMAL_DOCS:
            lowered = text.lower()
            for token in FORMAL_DOC_LEGACY_TOKENS:
                if token in lowered:
                    findings.append(f'{rel}: legacy public surface in formal documentation')
                    break
    return findings


def scan(root: Path, staged: bool = False) -> list[str]:
    root = root.resolve()
    files = staged_files(root) if staged else list(iter_files(root))
    findings: list[str] = []
    for path in files:
        if path.exists() and path.is_file():
            findings.extend(scan_file(root, path.resolve()))
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='.')
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args(argv)
    findings = scan(Path(args.root), staged=args.staged)
    if findings:
        print('TROVE privacy scan FAILED', file=sys.stderr)
        for finding in findings:
            print(f'- {finding}', file=sys.stderr)
        return 1
    print('TROVE privacy scan passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
