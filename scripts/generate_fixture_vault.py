#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages' / 'trove_core'))

from trove_core.wechat.fixture_guard import FixtureVaultGuardError, fixture_vault_guard_error_payload
from trove_core.wechat.indexer import index_fixture_vault


def redacted_fixture_metadata(report: dict, *, seed: int) -> dict:
    stable = {
        'seed': seed,
        'changed': int(report.get('changed') or 0),
        'chunks': report.get('chunks') or {},
        'counts': report.get('counts') or {},
    }
    digest = hashlib.sha256(json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    return {
        'schema_version': 1,
        'artifact_type': 'fixture_vault_metadata_redacted',
        **stable,
        'fixture_sha256': digest,
        'privacy': {
            'content_included': False,
            'absolute_paths_included': False,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Generate a synthetic WeChat-like TROVE fixture Vault.')
    parser.add_argument('--vault', required=True)
    parser.add_argument('--seed', type=int, default=20260621)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--jsonl', action='store_true', help='Also write synthetic/messages.jsonl inside the runtime Vault for inspection.')
    parser.add_argument('--redacted-metadata', action='store_true', help='Print deterministic counts/hash metadata without local paths.')
    args = parser.parse_args(argv)
    vault = Path(args.vault).expanduser().absolute()
    try:
        report = index_fixture_vault(vault, seed=args.seed, reset=args.reset, write_jsonl=args.jsonl)
    except FixtureVaultGuardError as exc:
        print(json.dumps(fixture_vault_guard_error_payload(exc), ensure_ascii=False), file=sys.stderr)
        return 2
    output = redacted_fixture_metadata(report, seed=args.seed) if args.redacted_metadata else report
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
