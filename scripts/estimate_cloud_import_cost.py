#!/usr/bin/env python3
"""Estimate accepted-source Volcengine ASR/Vision cost before any upload.

Reads only runtime Vault metadata. It does not inspect raw media bytes and never
prints private paths, message bodies, transcripts, image observations, provider
payloads, or secrets.
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

from trove_core.providers.pricing import ArkVisionLitePricing, estimate_asr_flash_rmb, pricing_payload


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def accepted_media(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    if not table_exists(conn, 'media_assets'):
        return []
    if table_exists(conn, 'media_asset_links'):
        return list(conn.execute(
            """
            SELECT DISTINCT a.*
            FROM media_assets a
            JOIN media_asset_links l ON l.asset_id=a.asset_id
            WHERE l.accepted=1 AND a.modality IN ('voice','audio','image')
            """
        ))
    return list(conn.execute("SELECT * FROM media_assets WHERE modality IN ('voice','audio','image')"))


def metadata_duration(row: sqlite3.Row) -> float | None:
    try:
        meta = json.loads(row['metadata_json'] or '{}')
    except Exception:
        meta = {}
    for key in ('duration_seconds', 'audio_duration_seconds', 'duration'):
        value = meta.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    return None


def build_estimate(vault: Path, *, default_audio_seconds: float = 0.0, ark_input_tokens_per_image: int = 1200, ark_output_tokens_per_image: int = 300) -> dict[str, Any]:
    db = vault / 'index' / 'trove.sqlite'
    if not db.exists():
        return {
            'schema_version': 1,
            'created_at': now_iso(),
            'status': 'blocked',
            'reason': 'vault_index_missing',
            'vault': 'configured-vault',
            'accepted_audio_duration_seconds': 0,
            'accepted_image_count': 0,
            'estimated_cost_rmb': None,
            'raw_paths_included': False,
        }
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = accepted_media(conn)
        audio_duration = 0.0
        audio_count = 0
        missing_duration_count = 0
        image_count = 0
        for row in rows:
            modality = (row['modality'] or '').lower()
            if modality in {'voice', 'audio'}:
                audio_count += 1
                duration = metadata_duration(row)
                if duration is None:
                    missing_duration_count += 1
                    duration = default_audio_seconds
                audio_duration += duration
            elif modality == 'image':
                image_count += 1
        ark = ArkVisionLitePricing()
        asr_cost = estimate_asr_flash_rmb(audio_duration)
        vision_cost = ark.estimate_rmb(input_tokens=image_count * ark_input_tokens_per_image, output_tokens=image_count * ark_output_tokens_per_image)
        total = round(asr_cost + vision_cost, 6)
        return {
            'schema_version': 1,
            'created_at': now_iso(),
            'status': 'estimated',
            'vault': 'configured-vault',
            'accepted_audio_count': audio_count,
            'accepted_audio_duration_seconds': round(audio_duration, 3),
            'audio_duration_source': 'metadata_or_default',
            'audio_missing_duration_count': missing_duration_count,
            'accepted_image_count': image_count,
            'provider_model_assumptions': pricing_payload(),
            'vision_token_assumptions': {
                'input_tokens_per_image': ark_input_tokens_per_image,
                'output_tokens_per_image': ark_output_tokens_per_image,
            },
            'cost_breakdown_rmb': {
                'asr_flash': asr_cost,
                'ark_vision_lite': vision_cost,
            },
            'estimated_cost_rmb': total,
            'raw_paths_included': False,
            'provider_payloads_included': False,
        }
    finally:
        conn.close()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_out(path: Path, vault: Path) -> Path:
    resolved = path.expanduser().resolve()
    proof_root = (vault / 'proof').resolve()
    if is_relative_to(resolved, ROOT.resolve()):
        raise SystemExit('cost estimate must not be written inside source repo')
    if not is_relative_to(resolved, proof_root):
        raise SystemExit('cost estimate must stay under runtime Vault proof directory')
    return resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', required=True)
    parser.add_argument('--out')
    parser.add_argument('--default-audio-seconds', type=float, default=0.0)
    parser.add_argument('--ark-input-tokens-per-image', type=int, default=1200)
    parser.add_argument('--ark-output-tokens-per-image', type=int, default=300)
    args = parser.parse_args(argv)
    vault = Path(args.vault).expanduser().resolve()
    payload = build_estimate(
        vault,
        default_audio_seconds=args.default_audio_seconds,
        ark_input_tokens_per_image=args.ark_input_tokens_per_image,
        ark_output_tokens_per_image=args.ark_output_tokens_per_image,
    )
    if args.out:
        out = validate_out(Path(args.out), vault)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get('estimated_cost_rmb') is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
