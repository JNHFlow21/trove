#!/usr/bin/env python3
"""Redacted hard-stop gate for real cloud ASR/Vision processing."""
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'packages' / 'trove_core') not in sys.path:
    sys.path.insert(0, str(ROOT / 'packages' / 'trove_core'))

from trove_core.providers.readiness import CloudReadinessInput, check_cloud_processing_readiness
from trove_core.providers.volcengine_docs import verify_volcengine_official_docs


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_out(path: Path, vault: Path | None) -> Path:
    resolved = path.expanduser().resolve()
    if is_relative_to(resolved, ROOT.resolve()):
        raise SystemExit('cloud readiness report must not be written inside source repo')
    if vault is not None:
        proof_root = (vault / 'proof').resolve()
        if not is_relative_to(resolved, proof_root):
            raise SystemExit('cloud readiness report must stay under runtime Vault proof directory')
    return resolved


def load_doc_verification(path: str | None, *, verify_live: bool) -> dict:
    if verify_live:
        return verify_volcengine_official_docs().to_dict()
    if not path:
        return {}
    return json.loads(Path(path).expanduser().read_text(encoding='utf-8'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Check cloud processing readiness without uploading data.')
    parser.add_argument('--repo', default=str(ROOT))
    parser.add_argument('--vault')
    parser.add_argument('--cost-cap-rmb', type=float)
    parser.add_argument('--estimated-cost-rmb', type=float)
    parser.add_argument('--doc-verification-json')
    parser.add_argument('--verify-docs-live', action='store_true')
    parser.add_argument('--selected-account-id', action='append', default=[])
    parser.add_argument('--discovered-account-id', action='append', default=[])
    parser.add_argument('--undecryptable-account-id', action='append', default=[])
    parser.add_argument('--coverage-gap-account-id', action='append', default=[])
    parser.add_argument('--redaction-probe', default='')
    parser.add_argument('--allow-dirty-git-for-test', action='store_true')
    parser.add_argument('--skip-usage-store-for-test', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args(argv)

    docs = load_doc_verification(args.doc_verification_json, verify_live=args.verify_docs_live)
    params = CloudReadinessInput(
        repo_root=Path(args.repo).expanduser().resolve(),
        vault_root=Path(args.vault).expanduser().resolve() if args.vault else None,
        cost_cap_rmb=args.cost_cap_rmb,
        estimated_cost_rmb=args.estimated_cost_rmb,
        doc_verification_date=docs.get('verification_date'),
        provider_docs_ok=bool(docs.get('ok')),
        selected_account_ids=args.selected_account_id,
        discovered_account_ids=args.discovered_account_id,
        undecryptable_account_ids=args.undecryptable_account_id,
        coverage_gap_account_ids=args.coverage_gap_account_id,
        redaction_probe=args.redaction_probe,
        require_clean_git=not args.allow_dirty_git_for_test,
        require_usage_store=not args.skip_usage_store_for_test,
    )
    report = check_cloud_processing_readiness(params)
    payload = report.to_dict()
    if docs:
        payload['doc_verification'] = {
            'ok': bool(docs.get('ok')),
            'verification_date': docs.get('verification_date'),
            'verified_at': docs.get('verified_at'),
            'official_sources': docs.get('official_sources', []),
            'doc_updates': docs.get('doc_updates', {}),
            'ambiguities': docs.get('ambiguities', []),
        }
    if args.out:
        out = validate_out(Path(args.out), params.vault_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
