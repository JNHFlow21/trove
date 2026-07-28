#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

ROOT = Path(__file__).resolve().parents[1]
for rel in (
    'packages/trove_protocol', 'packages/trove_core', 'packages/trove_client',
    'packages/trove_daemon', 'packages/trove_provider_wechat',
    'packages/trove_cli', 'packages/trove_mcp',
):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from trove_core.runtime import build_search_engine  # noqa: E402
from trove_core.search.evidence_provenance import (  # noqa: E402
    build_artifact_provenance,
    evidence_manifest_path,
    write_evidence_artifact,
)
from trove_core.search.eval_schema import validate_redacted_artifact  # noqa: E402
from trove_core.search.query import SearchRequest  # noqa: E402
from trove_core.vault.config import VaultConfig  # noqa: E402


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _load_queries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            query = str(payload.get('query') or '')
            if not query:
                raise ValueError(f'{path}:{line_no}: query is required')
            filters = payload.get('filters') or {}
            if not isinstance(filters, dict):
                raise ValueError(f'{path}:{line_no}: filters must be an object')
            rows.append({'query': query, 'filters': {str(k): str(v) for k, v in filters.items() if v is not None}})
    if not rows:
        raise ValueError(f'{path}: no queries found')
    return rows


def _request_from_case(case: dict[str, Any], *, limit: int, semantic: str = 'auto', include_media_hints: bool = False) -> SearchRequest:
    filters = case.get('filters') or {}
    return SearchRequest(
        case['query'],
        limit=limit,
        account_id=filters.get('account_id'),
        conversation_id=filters.get('conversation_id'),
        conversation_type=filters.get('conversation_type'),
        sender=filters.get('sender'),
        source_type=filters.get('source_type'),
        source_family=filters.get('source_family'),
        scope_type=filters.get('scope_type'),
        semantic=semantic,
        include_media_hints=include_media_hints,
    )


def run_benchmark(
    vault: str,
    queries_path: Path,
    *,
    rounds: int,
    limit: int,
    semantic: str = 'auto',
    include_media_hints: bool = False,
    temperature: str = 'cold',
    warmups: int = 0,
    seed: int = 0,
    fixture_id: str = 'synthetic_or_redacted',
) -> dict[str, Any]:
    if temperature not in {'cold', 'warm'}:
        raise ValueError('temperature must be cold or warm')
    if warmups < 0:
        raise ValueError('warmups must be >= 0')
    if temperature == 'cold' and warmups:
        raise ValueError('cold benchmarks cannot include warmups')
    if temperature == 'warm' and warmups < 1:
        raise ValueError('warm benchmarks require at least one warmup')
    cfg = VaultConfig.resolve(vault)
    cases = _load_queries(queries_path)
    engine = None
    try:
        engine = build_search_engine(cfg) if temperature == 'warm' else None
        provider_for_provenance = engine.embedding_provider if engine is not None else None
        latencies: list[float] = []
        per_query: list[dict[str, Any]] = [
            {
                'query_index': idx,
                'runs': 0,
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'result_count_last': 0,
                'candidate_routes_last': {},
                'candidate_count_last': 0,
            }
            for idx in range(len(cases))
        ]
        per_query_latencies: list[list[float]] = [[] for _ in cases]
        route_totals: dict[str, int] = {}
        vector_states: dict[str, int] = {}

        for _warmup in range(warmups):
            for case in cases:
                assert engine is not None
                engine.search(_request_from_case(
                    case,
                    limit=limit,
                    semantic=semantic,
                    include_media_hints=include_media_hints,
                ))

        for _round in range(rounds):
            for idx, case in enumerate(cases):
                request = _request_from_case(case, limit=limit, semantic=semantic, include_media_hints=include_media_hints)
                active_engine = None
                try:
                    start = time.perf_counter()
                    active_engine = build_search_engine(cfg) if temperature == 'cold' else engine
                    assert active_engine is not None
                    response = active_engine.search(request)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    provider_for_provenance = active_engine.embedding_provider
                    latencies.append(elapsed_ms)
                    per_query_latencies[idx].append(elapsed_ms)
                    ranking = response.retrieval_status.get('ranking') or {}
                    candidate_routes = dict(ranking.get('candidate_routes') or {})
                    for route, count in candidate_routes.items():
                        route_totals[str(route)] = route_totals.get(str(route), 0) + int(count)
                    vector_state = ((response.retrieval_status.get('vector') or {}).get('state') or 'unknown')
                    vector_states[str(vector_state)] = vector_states.get(str(vector_state), 0) + 1
                    per_query[idx].update({
                        'runs': len(per_query_latencies[idx]),
                        'result_count_last': response.total,
                        'candidate_routes_last': candidate_routes,
                        'candidate_count_last': int(ranking.get('candidate_count') or 0),
                    })
                finally:
                    if temperature == 'cold' and active_engine is not None:
                        active_engine.close()

        for idx, values in enumerate(per_query_latencies):
            per_query[idx].update({
                'p50_ms': round(_percentile(values, 0.50), 3),
                'p95_ms': round(_percentile(values, 0.95), 3),
                'min_ms': round(min(values), 3) if values else 0.0,
                'max_ms': round(max(values), 3) if values else 0.0,
            })

        report = {
            'schema_version': 2,
            'artifact_type': 'search_benchmark_redacted',
            'benchmark': 'search',
            'vault_kind': 'fixture_or_redacted',
            'queries': len(cases),
            'rounds': rounds,
            'semantic_mode': semantic,
            'include_media_hints': include_media_hints,
            'total_searches': len(latencies),
            'measurement_contract': {
                'cold_and_warm_separate': True,
                'warmups_excluded_from_percentiles': True,
                'temperature': temperature,
                'warmups': warmups,
                'same_host_relative_gate_required': True,
                'warm_lexical_p95_budget_ms': 250,
            },
            'latency_ms': {
                'p50': round(_percentile(latencies, 0.50), 3),
                'p95': round(_percentile(latencies, 0.95), 3),
                'mean': round(statistics.fmean(latencies), 3) if latencies else 0.0,
                'min': round(min(latencies), 3) if latencies else 0.0,
                'max': round(max(latencies), 3) if latencies else 0.0,
            },
            'candidate_route_totals': dict(sorted(route_totals.items())),
            'vector_states': dict(sorted(vector_states.items())),
            'per_query': per_query,
            'provenance': build_artifact_provenance(
                repo_root=ROOT,
                sqlite_path=cfg.paths.sqlite_path,
                case_pack_path=queries_path,
                seed=seed,
                fixture_id=fixture_id,
                provider=provider_for_provenance,
                temperature=temperature,
                warmups=warmups,
                rounds=rounds,
            ),
            'privacy': {
                'raw_queries_included': False,
                'raw_snippets_included': False,
                'raw_citations_included': False,
                'private_paths_included': False,
                'token_values_included': False,
            },
        }
        validate_redacted_artifact(report)
        return report
    finally:
        if engine is not None:
            engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run source-safe TROVE search latency benchmark.')
    parser.add_argument('--vault', required=True)
    parser.add_argument('--queries', default='tests/golden/search_queries.jsonl')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--semantic', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--include-media-hints', action='store_true')
    parser.add_argument('--temperature', choices=['cold', 'warm'], default='cold')
    parser.add_argument('--warmups', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixture-id', default='synthetic_or_redacted', help='Synthetic/redacted fixture label; only its hash is recorded.')
    parser.add_argument('--out', help='Optional redacted artifact path; also writes an independent .manifest.json sidecar.')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error('--rounds must be >= 1')
    if args.limit < 1:
        parser.error('--limit must be >= 1')
    if args.warmups < 0:
        parser.error('--warmups must be >= 0')
    if args.temperature == 'cold' and args.warmups:
        parser.error('--temperature cold requires --warmups 0')
    if args.temperature == 'warm' and args.warmups < 1:
        parser.error('--temperature warm requires --warmups >= 1')
    report = run_benchmark(
        args.vault,
        Path(args.queries),
        rounds=args.rounds,
        limit=args.limit,
        semantic=args.semantic,
        include_media_hints=args.include_media_hints,
        temperature=args.temperature,
        warmups=args.warmups,
        seed=args.seed,
        fixture_id=args.fixture_id,
    )
    if args.out:
        write_evidence_artifact(report, Path(args.out))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        latency = report['latency_ms']
        print(f"queries={report['queries']} rounds={report['rounds']} p50_ms={latency['p50']} p95_ms={latency['p95']}")
        print('candidate_route_totals=' + json.dumps(report['candidate_route_totals'], ensure_ascii=False, sort_keys=True))
        if args.out:
            print('redacted_file=' + Path(args.out).name)
            print('manifest_file=' + evidence_manifest_path(args.out).name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
