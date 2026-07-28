#!/usr/bin/env python3
"""Redacted multimodal real-Vault acceptance gate.

This script never uploads data by itself. It first writes a redacted proof report
under the runtime Vault proof directory. If cloud readiness fails, the report is
recoverable and no provider job is started.
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
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'packages' / 'trove_core') not in sys.path:
    sys.path.insert(0, str(ROOT / 'packages' / 'trove_core'))

from trove_core.providers.readiness import CloudReadinessInput, check_cloud_processing_readiness
from trove_core.providers.volcengine_docs import verify_volcengine_official_docs


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_out(path: Path, vault: Path) -> Path:
    resolved = path.expanduser().resolve()
    proof_root = (vault / 'proof').resolve()
    if resolved == ROOT.resolve() or is_relative_to(resolved, ROOT.resolve()):
        raise SystemExit('acceptance proof must not be written inside source repo')
    if not is_relative_to(resolved, proof_root):
        raise SystemExit('acceptance proof must stay under runtime Vault proof directory')
    return resolved


def table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
    except sqlite3.DatabaseError:
        return 0


def vault_counts(vault: Path) -> dict[str, Any]:
    db = vault / 'index' / 'trove.sqlite'
    if not db.exists():
        return {'available': False, 'counts': {}}
    conn = sqlite3.connect(db)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {name: table_count(conn, name) for name in ['accounts', 'conversations', 'messages', 'media_assets', 'provider_jobs', 'transcripts', 'image_observations', 'moment_items', 'favorites', 'entities', 'observations', 'relationships'] if name in tables}
        return {'available': True, 'counts': counts}
    finally:
        conn.close()


def redact_guard(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    forbidden = ['Bearer ', 'Authorization:', '/Users/', '/Volumes/', ]
    leaked = [item for item in forbidden if item in text]
    if leaked:
        raise SystemExit(f'redaction guard failed: {leaked[0]}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', required=True)
    parser.add_argument('--out')
    parser.add_argument('--cost-cap-rmb', type=float)
    parser.add_argument('--estimated-cost-rmb', type=float)
    parser.add_argument('--selected-account-id', action='append', default=[])
    parser.add_argument('--discovered-account-id', action='append', default=[])
    parser.add_argument('--undecryptable-account-id', action='append', default=[])
    parser.add_argument('--coverage-gap-account-id', action='append', default=[])
    parser.add_argument('--verify-docs-live', action='store_true')
    parser.add_argument('--small-batch', action='store_true')
    parser.add_argument('--full-import', action='store_true')
    args = parser.parse_args(argv)

    vault = Path(args.vault).expanduser().resolve()
    out = validate_out(Path(args.out).expanduser() if args.out else vault / 'proof' / 'multimodal' / 'acceptance.redacted.json', vault)
    docs = verify_volcengine_official_docs().to_dict() if args.verify_docs_live else {}
    readiness = check_cloud_processing_readiness(CloudReadinessInput(
        repo_root=ROOT,
        vault_root=vault,
        cost_cap_rmb=args.cost_cap_rmb,
        estimated_cost_rmb=args.estimated_cost_rmb,
        doc_verification_date=docs.get('verification_date'),
        provider_docs_ok=bool(docs.get('ok')),
        selected_account_ids=args.selected_account_id,
        discovered_account_ids=args.discovered_account_id,
        undecryptable_account_ids=args.undecryptable_account_id,
        coverage_gap_account_ids=args.coverage_gap_account_id,
    ))
    report = {
        'schema_version': 1,
        'created_at': now_iso(),
        'mode': 'full_import' if args.full_import else 'small_batch' if args.small_batch else 'preflight',
        'overall_ok': False,
        'status': 'blocked' if not readiness.ok else 'ready_for_manual_cloud_run',
        'cloud_upload_started': False,
        'privacy': {
            'raw_message_bodies_included': False,
            'raw_transcripts_included': False,
            'raw_image_observations_included': False,
            'provider_payloads_included': False,
            'private_paths_included': False,
            'media_files_included': False,
        },
        'vault': vault_counts(vault),
        'scope': readiness.scope,
        'cost': readiness.cost,
        'hard_stops': [issue.to_dict() for issue in readiness.hard_stops],
        'warnings': [issue.to_dict() for issue in readiness.warnings],
        'provider_docs': {
            'verified_today': bool(docs.get('ok')),
            'verification_date': docs.get('verification_date'),
            'official_sources_count': len(docs.get('official_sources', [])) if docs else 0,
            'ambiguities': docs.get('ambiguities', []),
        },
        'next_resume_action': 'Fix hard_stops, rerun small batch, then rerun with --full-import after small batch passes.' if not readiness.ok else 'Run bounded ASR/Vision jobs from the runtime Vault; persist job usage before full import.',
    }
    report['overall_ok'] = readiness.ok and args.small_batch
    redact_guard(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'written': out.name, 'status': report['status'], 'overall_ok': report['overall_ok'], 'hard_stop_count': len(report['hard_stops'])}, ensure_ascii=False))
    return 0 if report['overall_ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
