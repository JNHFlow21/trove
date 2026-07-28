#!/usr/bin/env python3
"""Repair and verify one complete-profile loop using redacted Vault-only proof.

The private customer selector is accepted only through an inherited descriptor.
No message, media, transcript, annotation, citation, source path, or profile
content is written to the proof artifact or stdout.
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
from collections import Counter
import hashlib
import json
import os
import tempfile
from typing import Any

from trove_core.agent_tools import tools as agent_tools
from trove_core.approvals import ApprovalManager, ApprovalRequired
from trove_core.knowledge.profile_enrichment import ACTIVE_TASK_STATES
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig, path_is_under


REPORT_RELATIVE_PATH = Path('proof/lazy-profile-enrichment/acceptance-report.redacted.json')
ALLOWED_PHASES = {'dry-run', 'repair', 'verify', 'all'}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]


def _read_private_input(fd: int) -> dict[str, Any]:
    if type(fd) is not int or fd < 3:
        raise ValueError('input descriptor must be inherited and non-stdio')
    try:
        chunks: list[bytes] = []
        remaining = 64 * 1024 + 1
        while remaining > 0:
            chunk = os.read(fd, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    raw = b''.join(chunks)
    if not raw or len(raw) > 64 * 1024 or b'\x00' in raw:
        raise ValueError('private input is empty or out of bounds')
    value = json.loads(raw.decode('utf-8'))
    if type(value) is not dict:
        raise ValueError('private input must be an object')
    allowed = {'customer', 'actor', 'session', 'allow_cloud_asr'}
    if set(value) - allowed:
        raise ValueError('private input contains unsupported fields')
    customer = value.get('customer')
    actor = value.get('actor', 'operator')
    session = value.get('session', 'lazy-profile-acceptance')
    allow_cloud = value.get('allow_cloud_asr', False)
    if type(customer) is not str or not customer.strip() or len(customer) > 500:
        raise ValueError('customer selector is missing or out of bounds')
    if type(actor) is not str or not actor.strip() or len(actor) > 200:
        raise ValueError('actor is missing or out of bounds')
    if type(session) is not str or not session.strip() or len(session) > 160:
        raise ValueError('session is missing or out of bounds')
    if type(allow_cloud) is not bool:
        raise ValueError('allow_cloud_asr must be an exact boolean')
    return {
        'customer': customer.strip(),
        'actor': actor.strip(),
        'session': session.strip(),
        'allow_cloud_asr': allow_cloud,
    }


def _latest_bound_source(cfg: VaultConfig) -> Path | None:
    store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    try:
        with store.connect() as conn:
            row = conn.execute(
                """SELECT root_ref FROM source_snapshots
                    WHERE state='available' AND root_ref IS NOT NULL
                    ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()
    finally:
        store.close()
    if row is None:
        return None
    source = (cfg.root / str(row['root_ref'])).resolve()
    if not path_is_under(source, cfg.root) or not source.is_dir():
        return None
    return source


def _task_counts(manifest: dict[str, Any]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in (manifest.get('counts') or {}).items()))


def _actionable(manifest: dict[str, Any]) -> int:
    return sum(int((manifest.get('counts') or {}).get(state) or 0) for state in ACTIVE_TASK_STATES)


def _lifecycle_status(cfg: VaultConfig) -> dict[str, Any]:
    """Return only redacted lifecycle/audit facts used by release acceptance."""

    store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    try:
        with store.connect() as conn:
            row = conn.execute(
                """SELECT lifecycle_version,status,backup_policy,audit_retention_until
                     FROM derived_data_purge_audit ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
    finally:
        store.close()
    verified = bool(
        row is not None
        and row['status'] == 'completed'
        and row['lifecycle_version'] == 'derived-data/v1'
        and row['backup_policy'] == 'replace_all_pre_purge_backups_with_one_post_purge_backup'
        and row['audit_retention_until']
    )
    return {
        'matrix_version': 'derived-data/v1',
        'audit_present': row is not None,
        'purge_verified': verified,
        'latest_status': str(row['status']) if row is not None else None,
        'lifecycle_version': str(row['lifecycle_version']) if row is not None else None,
        'backup_policy': str(row['backup_policy']) if row is not None else None,
        'audit_retention_recorded': bool(row is not None and row['audit_retention_until']),
    }


def _run_artifact_counts(cfg: VaultConfig, run_id: str) -> dict[str, Any]:
    """Count authorized cache/evidence writes without exposing their contents."""

    store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    try:
        with store.connect() as conn:
            rows = conn.execute(
                """SELECT asset_id,modality,state FROM profile_enrichment_tasks
                     WHERE run_id=?""",
                (run_id,),
            ).fetchall()
            asset_ids = sorted({str(row['asset_id']) for row in rows if row['asset_id']})
            if asset_ids:
                marks = ','.join('?' for _ in asset_ids)
                transcript_count = int(conn.execute(
                    f"SELECT COUNT(*) FROM transcripts WHERE status='active' AND asset_id IN ({marks})",
                    asset_ids,
                ).fetchone()[0])
                image_count = int(conn.execute(
                    f"SELECT COUNT(*) FROM image_observations WHERE status IN ('active','proposed') AND asset_id IN ({marks})",
                    asset_ids,
                ).fetchone()[0])
                materialized = int(conn.execute(
                    f"SELECT COUNT(*) FROM media_assets WHERE cache_state='cached' AND asset_id IN ({marks})",
                    asset_ids,
                ).fetchone()[0])
                locator_rows = conn.execute(
                    f"""SELECT locator_state,COUNT(*) AS count FROM media_source_bindings
                          WHERE asset_id IN ({marks}) GROUP BY locator_state""",
                    asset_ids,
                ).fetchall()
            else:
                transcript_count = image_count = materialized = 0
                locator_rows = []
    finally:
        store.close()
    modality_counts = Counter(str(row['modality']) for row in rows)
    completed_counts = Counter(str(row['modality']) for row in rows if row['state'] == 'completed')
    return {
        'task_modalities': dict(sorted(modality_counts.items())),
        'completed_modalities': dict(sorted(completed_counts.items())),
        'active_transcript_writes': transcript_count,
        'active_image_observation_writes': image_count,
        'materialized_cache_assets': materialized,
        'locator_states': {
            str(row['locator_state']): int(row['count']) for row in locator_rows
        },
    }


def _repair_summary(identity: dict, appmsg: dict | None, media: dict) -> dict[str, Any]:
    appmsg_backfill = (appmsg or {}).get('backfill') or {}
    link = media.get('link_result') or {}
    return {
        'identity': {
            'status': identity.get('status'),
            'duplicate_count': len(identity.get('duplicate_entity_ids') or []),
            'applied': bool(identity.get('applied')),
        },
        'appmsg': {
            'source_available': appmsg is not None,
            'source_payloads': int((appmsg or {}).get('source_payloads') or appmsg_backfill.get('source_payloads') or 0),
            'changed': int(appmsg_backfill.get('changed') or (appmsg or {}).get('would_change') or 0),
            'unsupported': int(appmsg_backfill.get('unsupported') or 0),
        },
        'message_media': {
            'eligible': int(media.get('eligible_messages') or 0),
            'missing_assets': int(media.get('missing_assets') or 0),
            'missing_links': int(media.get('missing_links') or 0),
            'assets_upserted': int(link.get('assets_upserted') or 0),
            'links_upserted': int(link.get('links_upserted') or 0),
        },
    }


def _run_repairs(cfg: VaultConfig, private: dict[str, Any], *, apply: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    customer = private['customer']
    identity_plan = agent_tools.identity_reconcile_plan(cfg.root, customer)
    source = _latest_bound_source(cfg)
    appmsg_plan = agent_tools.appmsg_repair_plan(cfg.root, source=source) if source is not None else None
    media_plan = agent_tools.message_media_repair_plan(cfg.root)
    if not apply:
        return _repair_summary(identity_plan, appmsg_plan, media_plan), identity_plan
    identity = agent_tools.identity_reconcile(cfg.root, customer=customer, yes=True)
    appmsg = agent_tools.appmsg_repair(cfg.root, source=source, yes=True) if source is not None else None
    media = agent_tools.message_media_repair(cfg.root, yes=True)
    return _repair_summary(identity, appmsg, media), identity


def _plan(cfg: VaultConfig, private: dict[str, Any], *, session: str) -> dict[str, Any]:
    return agent_tools.profile_enrichment_plan(
        cfg.root,
        private['customer'],
        actor=private['actor'],
        session=session,
        mode='complete',
        item_budget=500,
        cost_budget_rmb=10.0 if private['allow_cloud_asr'] else 0.0,
        execution_location='local',
        processor_identity='codex-local-agent/v1',
        prompt_version='profile-image/v1',
    )


def _process_voice_tasks(
    cfg: VaultConfig,
    private: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    paths: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    current = manifest
    for item in list(current.get('items') or []):
        if item.get('modality') != 'voice' or item.get('state') not in {'pending', 'retryable_failure'}:
            continue
        worker = 'lazy-profile-acceptance'
        claim = agent_tools.profile_enrichment_claim(
            cfg.root,
            current['run_id'],
            actor=private['actor'],
            session=private['session'],
            worker=worker,
            task_id=item['task_id'],
            execution_location='local',
            lease_seconds=300,
        )
        action = claim.get('agent_action') or {}
        try:
            result = agent_tools.profile_enrichment_voice_execute(
                cfg.root,
                item['task_id'],
                actor=private['actor'],
                session=private['session'],
                worker=worker,
                claim_token=action['claim_token'],
            )
        except ApprovalRequired as exc:
            paths['awaiting_approval'] += 1
            if not private['allow_cloud_asr']:
                continue
            ApprovalManager(cfg.root).decide(exc.record.approval_id, 'approved')
            result = agent_tools.profile_enrichment_voice_execute(
                cfg.root,
                item['task_id'],
                actor=private['actor'],
                session=private['session'],
                worker=worker,
                claim_token=action['claim_token'],
                approval_id=exc.record.approval_id,
            )
        paths[str(result.get('execution_path') or result.get('status') or 'unknown')] += 1
        route = result.get('route')
        if isinstance(route, str) and route:
            routes[route] += 1
    return dict(sorted(paths.items())), dict(sorted(routes.items()))


def _verify(cfg: VaultConfig, private: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    status = agent_tools.profile_enrichment_status(
        cfg.root, manifest['run_id'], actor=private['actor'], session=private['session'],
    )
    actionable = _actionable(status)
    final_state = str(status.get('state') or '')
    if actionable or final_state not in {'complete', 'complete_with_terminal_gaps'}:
        return {
            'state': final_state,
            'task_counts': _task_counts(status),
            'actionable': actionable,
            'snapshot': {'finalized': False, 'cache_hit': False},
            'rerun': {'performed': False, 'actionable': None, 'cache_hits': 0, 'snapshot_cache_hit': False},
            'authorized_cache_writes': _run_artifact_counts(cfg, status['run_id']),
        }, False
    snapshot = agent_tools.profile_enrichment_finalize(
        cfg.root, status['run_id'], actor=private['actor'], session=private['session'],
    )
    cache_session = private['session'] + '-cache'
    rerun = _plan(cfg, private, session=cache_session)
    rerun_actionable = _actionable(rerun)
    rerun_snapshot = None
    if rerun_actionable == 0 and rerun.get('state') in {'complete', 'complete_with_terminal_gaps'}:
        rerun_snapshot = agent_tools.profile_enrichment_finalize(
            cfg.root, rerun['run_id'], actor=private['actor'], session=cache_session,
        )
    rerun_cache_hits = int(((rerun.get('execution_summary') or {}).get('cache_hit')) or 0)
    ok = bool(
        snapshot.get('finalized')
        and rerun_actionable == 0
        and rerun_cache_hits == int((rerun.get('page') or {}).get('total') or 0)
        and rerun_snapshot
        and rerun_snapshot.get('cache_hit')
    )
    return {
        'state': final_state,
        'task_counts': _task_counts(status),
        'actionable': actionable,
        'terminal_gap_count': len(snapshot.get('unresolved_gaps') or []),
        'snapshot': {
            'finalized': bool(snapshot.get('finalized')),
            'created': bool(snapshot.get('created')),
            'cache_hit': bool(snapshot.get('cache_hit')),
            'version': int(snapshot.get('version') or 0),
            'evidence_citation_count': int(snapshot.get('evidence_citations_count') or 0),
            'stale': bool(snapshot.get('stale')),
        },
        'rerun': {
            'performed': True,
            'state': rerun.get('state'),
            'actionable': rerun_actionable,
            'cache_hits': rerun_cache_hits,
            'task_total': int((rerun.get('page') or {}).get('total') or 0),
            'snapshot_cache_hit': bool((rerun_snapshot or {}).get('cache_hit')),
        },
        'authorized_cache_writes': _run_artifact_counts(cfg, status['run_id']),
    }, ok


def _safe_write(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + '\n').encode('ascii')
    lowered = encoded.lower()
    for marker in (b'trove://', b'http://', b'https://', b'/users/', b'/volumes/', b'raw_text'):
        if marker in lowered:
            raise ValueError('redacted acceptance report contains a forbidden marker')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temp_name = tempfile.mkstemp(prefix='.lazy-profile-', suffix='.tmp', dir=path.parent)
    os.fchmod(fd, 0o600)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        path.chmod(0o600)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault')
    parser.add_argument('--input-fd', type=int, required=True)
    parser.add_argument('--phase', choices=sorted(ALLOWED_PHASES), default='all')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--require-purge-audit', action='store_true')
    args = parser.parse_args(argv)
    cfg = VaultConfig.resolve(args.vault)
    cfg.validate_runtime_path()
    private = _read_private_input(args.input_fd)
    if args.phase in {'repair', 'all'} and not args.yes:
        raise SystemExit('repair/all phase requires --yes')

    report: dict[str, Any] = {
        'schema': 'lazy-profile-enrichment-acceptance/v1',
        'phase': args.phase,
        'target_hash': _hash(private['customer']),
        'repair': None,
        'voice_execution_paths': {},
        'media_locator_routes': {},
        'enrichment': None,
        'lifecycle': {
            'matrix_version': 'derived-data/v1',
            'audit_present': False,
            'purge_verified': False,
            'latest_status': None,
            'lifecycle_version': None,
            'backup_policy': None,
            'audit_retention_recorded': False,
        },
        'privacy': {
            'names_included': False,
            'message_bodies_included': False,
            'paths_included': False,
            'urls_included': False,
            'keys_included': False,
            'media_included': False,
            'transcripts_included': False,
            'profile_content_included': False,
            'provider_payloads_included': False,
        },
        'ok': False,
    }
    try:
        apply_repairs = args.phase in {'repair', 'all'}
        repair, _identity = _run_repairs(cfg, private, apply=apply_repairs)
        report['repair'] = repair
        report['lifecycle'] = _lifecycle_status(cfg)
        manifest = _plan(cfg, private, session=private['session'])
        if apply_repairs:
            report['voice_execution_paths'], report['media_locator_routes'] = _process_voice_tasks(
                cfg, private, manifest,
            )
        if args.phase == 'dry-run':
            report['enrichment'] = {
                'state': manifest.get('state'),
                'task_counts': _task_counts(manifest),
                'actionable': _actionable(manifest),
                'snapshot': {'finalized': False, 'cache_hit': False},
                'rerun': {'performed': False},
                'authorized_cache_writes': _run_artifact_counts(cfg, manifest['run_id']),
            }
            report['ok'] = True
        elif args.phase == 'repair':
            status = agent_tools.profile_enrichment_status(
                cfg.root, manifest['run_id'], actor=private['actor'], session=private['session'],
            )
            report['enrichment'] = {
                'state': status.get('state'),
                'task_counts': _task_counts(status),
                'actionable': _actionable(status),
                'awaiting_local_agent': int((status.get('counts') or {}).get('pending') or 0)
                    + int((status.get('counts') or {}).get('awaiting_agent') or 0),
                'snapshot': {'finalized': False, 'cache_hit': False},
                'rerun': {'performed': False},
                'authorized_cache_writes': _run_artifact_counts(cfg, status['run_id']),
            }
            report['ok'] = True
        else:
            report['enrichment'], report['ok'] = _verify(cfg, private, manifest)
        if args.require_purge_audit and not report['lifecycle']['purge_verified']:
            report['ok'] = False
    except Exception as exc:
        report['failure'] = {
            'code': str(getattr(exc, 'code', exc.__class__.__name__)),
            'recoverable': True,
        }
        report['ok'] = False

    output = cfg.root / REPORT_RELATIVE_PATH
    _safe_write(output, report)
    print(json.dumps({
        'ok': report['ok'],
        'phase': args.phase,
        'written': str(REPORT_RELATIVE_PATH),
        'raw_content_included': False,
        'raw_paths_included': False,
    }, ensure_ascii=True, sort_keys=True))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
