#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from trove_core.approvals import ApprovalManager, ApprovalRequired, approval_required_payload
from trove_core.maintain import rotate_sqlite_backups
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.repositories import MultimodalRepository
from trove_core.sync import record_dirty_citations
from trove_core.vault.config import VaultConfig
from trove_core.wechat.auxiliary_import import import_auxiliary_sources


def _counts(store: SQLiteStore) -> dict[str, int]:
    store.initialize()
    with store.connect() as conn:
        def count(table: str, where: str = '1=1') -> int:
            if not store._table_exists(conn, table):
                return 0
            return int(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {where}').fetchone()[0])
        return {
            'moment_items': count('moment_items'),
            'moment_interactions': count('moment_interactions'),
            'moment_chunks': count('evidence_chunks', "source_type='moment'"),
            'sqlite_vectors': count('vector_entries'),
        }


def _redacted_source(source: Path) -> dict[str, Any]:
    sns = source / 'sns.db'
    return {'source_name_hash': __import__('hashlib').sha256(source.name.encode()).hexdigest()[:12], 'sns_db_present': sns.exists()}


def migrate(vault: str | None, account_dir: Path, *, approval_id: str | None = None, yes: bool = False, backup_retention: int = 3, target: Path | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    preflight = subprocess.run(
        [str(repo_root / 'scripts' / 'trove-python'), str(repo_root / 'scripts' / 'preflight_workspace.py'), '--target', str(target or repo_root)],
        cwd=str(repo_root), capture_output=True, text=True, check=False,
    )
    if preflight.returncode != 0:
        return {'ok': False, 'status': 'preflight_failed', 'preflight_returncode': preflight.returncode, 'raw_content_included': False, 'raw_paths_included': False}
    cfg = VaultConfig.resolve(vault, env={} if vault is not None else None)
    cfg.ensure()
    payload = {'account_sources': 1, 'family': 'moment', 'zvec_full_rebuild': False}
    try:
        approval = ApprovalManager(cfg.root).require('moment_semantics_migration', 'full_import', payload, approval_id=approval_id, one_step_approval=yes)
    except ApprovalRequired as exc:
        data = approval_required_payload(exc.record)
        data.update({'ok': False, 'status': 'approval_required', 'raw_content_included': False, 'raw_paths_included': False})
        return data
    store = SQLiteStore(cfg.paths.sqlite_path)
    before = _counts(store)
    backups = rotate_sqlite_backups(cfg.paths.sqlite_path, retention=backup_retention, create=True)
    report = import_auxiliary_sources(account_dir, account_id='acct-' + __import__('hashlib').sha256(account_dir.name.encode()).hexdigest()[:12], store=store, repo=MultimodalRepository(store), only={'moment'})
    chunks = store.rebuild_evidence_chunks_for_source_types(['moment'])
    dirty = record_dirty_citations(store, report.dirty_refs()) if report.dirty_refs() else 0
    after = _counts(store)
    return {
        'ok': True,
        'status': 'completed',
        'approval': approval,
        'source': _redacted_source(account_dir),
        'before_counts': before,
        'after_counts': after,
        'import_report': report.to_dict(),
        'chunks': chunks,
        'dirty_citations_recorded': dirty,
        'backup': {'created': bool(backups.get('created')), 'retention': backup_retention},
        'zvec_full_rebuild': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Approve-backed moment semantics migration; no full zvec rebuild.')
    parser.add_argument('--vault')
    parser.add_argument('--account-dir', required=True)
    parser.add_argument('--approval-id')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--backup-retention', type=int, default=3)
    parser.add_argument('--target', default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    data = migrate(args.vault, Path(args.account_dir).expanduser(), approval_id=args.approval_id, yes=args.yes, backup_retention=args.backup_retention, target=Path(args.target).expanduser())
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if data.get('ok') else 3 if data.get('status') == 'approval_required' else 2


if __name__ == '__main__':
    raise SystemExit(main())
