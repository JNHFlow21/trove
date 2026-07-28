#!/usr/bin/env python3
"""Redacted real acceptance for integrated decrypt + lazy media.

Writes only counts/status into the runtime Vault proof directory. It never prints
or persists key values, message text, media bytes, raw DBs, or private paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

from trove_core.media_pipeline import media_status_payload
from trove_core.sync import SyncOptions, run_sync
from trove_core.vault.config import VaultConfig
from trove_core.wechat.decrypt import DecryptConfig, build_decrypt_plan, run_decrypt_plan
from trove_core.wechat.decrypt.config import selected_accounts_from_strings
from trove_core.wechat.decrypt.redaction import redact_obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', required=True)
    parser.add_argument('--live-root', required=True)
    parser.add_argument('--selected-account', action='append', default=[], required=True)
    parser.add_argument('--secret-name')
    parser.add_argument('--key-store')
    parser.add_argument('--output-source-name', default='wechat-integrated-decrypted')
    parser.add_argument('--run-decrypt', action='store_true')
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)

    if (args.run_decrypt or args.sync) and not args.yes:
        parser.error('--run-decrypt/--sync requires --yes')

    cfg = VaultConfig.resolve(args.vault, env={})
    cfg.ensure()
    dcfg = DecryptConfig(
        live_root=Path(args.live_root).expanduser(),
        vault_root=cfg.root,
        selected_accounts=selected_accounts_from_strings(args.selected_account, secret_name=args.secret_name),
        secret_name=args.secret_name,
        key_store_path=Path(args.key_store).expanduser() if args.key_store else None,
        output_source_name=args.output_source_name,
    )
    plan = build_decrypt_plan(dcfg)
    report: dict = {
        'ok': plan.ok,
        'preflight': plan.to_redacted_dict(),
        'decrypt': {'status': 'skipped', 'reason': 'run_decrypt_not_requested'},
        'sync': {'status': 'skipped', 'reason': 'sync_not_requested'},
        'media_status': media_status_payload(cfg.root),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
    if args.run_decrypt:
        decrypt_report = run_decrypt_plan(plan)
        report['decrypt'] = decrypt_report
        report['ok'] = bool(report['ok'] and decrypt_report.get('ok'))
    if args.sync and report.get('ok'):
        snapshot_dir = cfg.root / 'sources' / args.output_source_name / 'current'
        sync_report = run_sync(cfg.root, options=SyncOptions(snapshot_dir=snapshot_dir))
        report['sync'] = sync_report
        report['media_status'] = media_status_payload(cfg.root)
        report['ok'] = bool(sync_report.get('ok'))
    out_dir = cfg.root / 'proof' / 'decrypt-media-real'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'acceptance-report.redacted.json'
    out_path.write_text(json.dumps(redact_obj(report), ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'ok': report.get('ok'), 'written': 'proof/decrypt-media-real/acceptance-report.redacted.json', 'raw_paths_included': False, 'raw_content_included': False}, ensure_ascii=False, indent=2))
    return 0 if report.get('ok') else 2


if __name__ == '__main__':
    raise SystemExit(main())
