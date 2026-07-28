#!/usr/bin/env python3
"""Create a redacted fixed dev/holdout split for private retrieval eval packs."""
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
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from trove_core.search.eval_schema import expected_citations, load_case_pack, stable_hash, validate_redacted_artifact


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def source_family(case: dict[str, Any]) -> str:
    return str(case.get('source_family') or (case.get('filters') or {}).get('source_type') or 'message')


def is_negative_only(case: dict[str, Any]) -> bool:
    oracle = case.get('oracle') or {}
    return bool(oracle.get('negative_no_results') or oracle.get('negative_excluded_citations')) and not expected_citations(case)


def keyed_order(case: dict[str, Any], seed: str) -> str:
    return hashlib.sha256(f"{seed}:{stable_hash(case.get('case_id'))}".encode('utf-8')).hexdigest()


def allocate_holdout_counts(groups: dict[tuple[str, str], list[dict[str, Any]]], target: int) -> dict[tuple[str, str], int]:
    total = sum(len(rows) for rows in groups.values())
    if total == 0 or target <= 0:
        return {key: 0 for key in groups}
    target = min(target, total)
    allocations: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    for key, rows in groups.items():
        quota = len(rows) * target / total
        base = int(math.floor(quota))
        if len(rows) >= 2 and base == 0:
            base = 1
        base = min(base, len(rows))
        allocations[key] = base
        fractions.append((quota - math.floor(quota), key))
    current = sum(allocations.values())
    if current > target:
        for _frac, key in sorted(fractions, key=lambda item: (item[0], item[1][0], item[1][1])):
            if current <= target:
                break
            if allocations[key] > 0:
                allocations[key] -= 1
                current -= 1
    elif current < target:
        for _frac, key in sorted(fractions, key=lambda item: (-item[0], item[1][0], item[1][1])):
            if current >= target:
                break
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                current += 1
    return allocations


def build_split_manifest(cases_path: Path, *, holdout_positive_target: int = 32, seed: str = 'retrieval-quality-v1') -> dict[str, Any]:
    cases = load_case_pack(cases_path)
    positives = [case for case in cases if expected_citations(case)]
    negatives = [case for case in cases if is_negative_only(case)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in positives:
        grouped[(str(case.get('category') or 'unknown'), source_family(case))].append(case)
    allocations = allocate_holdout_counts(grouped, holdout_positive_target)
    dev_positive: list[dict[str, Any]] = []
    holdout_positive: list[dict[str, Any]] = []
    strata = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda case: keyed_order(case, seed))
        holdout_n = allocations[key]
        holdout_rows = ordered[:holdout_n]
        dev_rows = ordered[holdout_n:]
        holdout_positive.extend(holdout_rows)
        dev_positive.extend(dev_rows)
        strata.append({
            'category': key[0],
            'source_family': key[1],
            'positive_cases': len(rows),
            'dev_positive_cases': len(dev_rows),
            'holdout_positive_cases': len(holdout_rows),
        })

    def hashes(rows: list[dict[str, Any]]) -> list[str]:
        return sorted(str(stable_hash(case.get('case_id'))) for case in rows)

    negative_hashes = hashes(negatives)
    positive_by_category = Counter(str(case.get('category') or 'unknown') for case in positives)
    positive_by_source = Counter(source_family(case) for case in positives)
    payload = {
        'schema_version': 1,
        'artifact_type': 'retrieval_quality_split_redacted',
        'created_at': now_iso(),
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
            'token_values_included': False,
        },
        'case_pack_anchor': {
            'sha256_prefix': hashlib.sha256(cases_path.expanduser().read_bytes()).hexdigest()[:32],
            'loaded_cases': len(cases),
            'path_included': False,
        },
        'split_policy': {
            'name': 'category_source_family_stratified_positive_split_v1',
            'seed_hash': stable_hash(seed),
            'holdout_positive_target': holdout_positive_target,
            'negative_cases_reused_for_gate': False,
            'negative_assignment': 'dev_only',
        },
        'counts': {
            'cases': len(cases),
            'positive_cases': len(positives),
            'negative_only_cases': len(negatives),
            'dev_positive_cases': len(dev_positive),
            'holdout_positive_cases': len(holdout_positive),
        },
        'positive_distribution': {
            'by_category': dict(sorted(positive_by_category.items())),
            'by_source_family': dict(sorted(positive_by_source.items())),
        },
        'negative_case_hashes': negative_hashes,
        'splits': {
            'dev': {
                'positive_case_hashes': hashes(dev_positive),
                'negative_case_hashes': negative_hashes,
                'case_hashes': sorted([*hashes(dev_positive), *negative_hashes]),
            },
            'holdout': {
                'positive_case_hashes': hashes(holdout_positive),
                'negative_case_hashes': [],
                'case_hashes': hashes(holdout_positive),
            },
        },
        'strata': strata,
    }
    validate_redacted_artifact(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build a redacted fixed dev/holdout split manifest for retrieval eval.')
    parser.add_argument('--cases', required=True, help='Private or fixture case pack.')
    parser.add_argument('--out', required=True, help='Redacted split manifest output path.')
    parser.add_argument('--holdout-positive-target', type=int, default=32)
    parser.add_argument('--seed', default='retrieval-quality-v1')
    args = parser.parse_args(argv)
    manifest = build_split_manifest(Path(args.cases), holdout_positive_target=args.holdout_positive_target, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'redacted_file': out.name,
        'positive_cases': manifest['counts']['positive_cases'],
        'dev_positive_cases': manifest['counts']['dev_positive_cases'],
        'holdout_positive_cases': manifest['counts']['holdout_positive_cases'],
        'negative_only_cases': manifest['counts']['negative_only_cases'],
        'raw_queries_printed': False,
        'private_paths_printed': False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
