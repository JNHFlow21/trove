#!/usr/bin/env python3
"""Redacted layer diagnosis for TROVE retrieval eval positives.

The script may read private local eval cases and the selected Vault index, but it
only writes aggregate counts, hashes, ranks, and route states.
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
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from trove_core.runtime import configured_embedding_provider, vector_registry
from trove_core.search.eval_schema import expected_citations, load_case_pack, stable_hash, validate_redacted_artifact
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.search.query_understanding import analyze_query
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig

LAYER_LABELS = {
    'hit_top3': 'already_recovered_in_top3',
    'L1_route': 'semantic_route_not_attempted_or_auto_skipped',
    'L2_recall': 'not_found_in_vector_top50_or_lexical_pool',
    'L3_fusion': 'candidate_available_but_not_ranked_top3',
    'L4_data': 'expected_evidence_text_missing_or_too_short',
}
VECTOR_TOP50_MISSING_REASONS = {
    'semantic_not_attempted': 'search did not run semantic retrieval for this case',
    'lexical_already_hit': 'expected evidence was absent from vector top50 but present in the lexical top50 pool',
    'topk_truncation': 'expected evidence was absent from vector top50 but present in the deeper vector window',
    'true_l2_recall': 'expected evidence was absent from lexical top50 and the deeper vector window',
}
PLACEHOLDER_RE = re.compile(r'^(\s|\[?(图片|语音|视频|文件|表情|位置|链接)\]?|<[^>]+>)+$', re.I)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def source_family(case: dict[str, Any]) -> str:
    return str(case.get('source_family') or (case.get('filters') or {}).get('source_type') or 'message')


def load_manifest_hashes(path: Path | None, group: str | None) -> set[str] | None:
    if path is None or not group:
        return None
    data = json.loads(path.expanduser().read_text(encoding='utf-8'))
    split = ((data.get('splits') or {}).get(group) or {})
    return {str(v) for v in (split.get('positive_case_hashes') or split.get('case_hashes') or []) if v}


def row_value(row: Any, key: str) -> str:
    try:
        if hasattr(row, 'keys') and key not in row.keys():
            return ''
        return str(row[key] or '')
    except Exception:
        return ''


def row_ids(row: Any) -> set[str]:
    ids = set()
    for key in ('citation', 'parent_citation', 'context_anchor'):
        value = row_value(row, key)
        if value:
            ids.add(value)
    return ids


def result_ids(result: dict[str, Any]) -> set[str]:
    return {str(result.get(key)) for key in ('citation', 'parent_citation', 'context_anchor') if result.get(key)}


def matches_expected_row(row: Any, expected: set[str]) -> bool:
    return bool(row_ids(row) & expected)


def matches_expected_result(result: dict[str, Any], expected: set[str]) -> bool:
    return bool(result_ids(result) & expected)


def first_rank_in_results(results: list[dict[str, Any]], expected: set[str], *, limit: int) -> int | None:
    for idx, result in enumerate(results[:limit], start=1):
        if matches_expected_result(result, expected):
            return idx
    return None


def first_rank_in_rows(rows: list[Any], expected: set[str]) -> int | None:
    for idx, row in enumerate(rows, start=1):
        if matches_expected_row(row, expected):
            return idx
    return None


def compact_len(value: str) -> int:
    return len(re.sub(r'\s+', '', str(value or '')))


def evidence_text_health(store: SQLiteStore, expected: set[str]) -> dict[str, Any]:
    rows: list[Any] = []
    rows.extend(store.evidence_by_citations(sorted(expected)).values())
    # If the oracle points at a parent message, inspect its active chunks too.
    if store.path.exists() and expected:
        with store.connect() as conn:
            if store._table_exists(conn, 'evidence_chunks'):
                placeholders = ','.join('?' for _ in expected)
                for chunk in conn.execute(
                    f"SELECT * FROM evidence_chunks WHERE status='active' AND parent_citation IN ({placeholders}) LIMIT 5",
                    tuple(sorted(expected)),
                ):
                    rows.append(chunk)
    max_len = 0
    placeholder_only = True if rows else False
    for row in rows:
        text = row_value(row, 'content') or row_value(row, 'text') or row_value(row, 'caption') or row_value(row, 'visible_text')
        max_len = max(max_len, compact_len(text))
        if text and not PLACEHOLDER_RE.match(text.strip()):
            placeholder_only = False
    return {
        'evidence_rows_found': len(rows),
        'max_compact_text_chars': max_len,
        'placeholder_only': placeholder_only,
        'data_issue': (not rows) or max_len < 4 or placeholder_only,
    }


def collect_lexical_pool(store: SQLiteStore, query: str, filters: dict[str, str], *, limit: int) -> dict[str, list[Any]]:
    understanding = analyze_query(query)
    fts_queries = understanding.expanded_queries[:3]
    pools: dict[str, list[Any]] = {
        'exact': store.exact_search(query, filters=filters, limit=limit),
        'evidence': store.chunk_search(query, filters=filters, limit=limit),
        'fts': [],
    }
    seen: set[str] = set()
    fts_rows: list[Any] = []
    for route_query in fts_queries:
        for row in store.fts_search_filtered(route_query, filters=filters, limit=limit, allow_like_fallback=False):
            citation = row_value(row, 'citation')
            if citation and citation not in seen:
                seen.add(citation)
                fts_rows.append(row)
            if len(fts_rows) >= limit:
                break
        if len(fts_rows) >= limit:
            break
    pools['fts'] = fts_rows
    return pools


def vector_top_rows(vector: Any, provider: Any, query: str, filters: dict[str, str], *, limit: int) -> tuple[list[Any], dict[str, Any]]:
    if vector is None or provider is None:
        return [], {'state': 'unavailable_fallback'}
    try:
        try:
            rows = vector.search(query, filters=filters, limit=limit, provider=provider)
        except TypeError:
            rows = vector.search(query, filters=filters, limit=limit)
        return rows, {'state': 'available'}
    except Exception as exc:
        return [], {'state': getattr(exc, 'vector_state', 'degraded'), 'reason_code': getattr(exc, 'reason_code', exc.__class__.__name__)}


def vector_top50_missing_reason(
    *,
    semantic_attempted: bool,
    lexical_hit: bool,
    vector_top50_rank: int | None,
    vector_deep_rank: int | None,
) -> str | None:
    if vector_top50_rank is not None:
        return None
    if not semantic_attempted:
        return 'semantic_not_attempted'
    if lexical_hit:
        return 'lexical_already_hit'
    if vector_deep_rank is not None:
        return 'topk_truncation'
    return 'true_l2_recall'


def diagnose(
    cases_path: Path,
    vault_root: Path,
    *,
    split_manifest: Path | None = None,
    split_group: str | None = None,
    k: int = 3,
    model_path: str | None = None,
    vector_top_n: int = 50,
    vector_deep_n: int = 200,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root))
    store = SQLiteStore(cfg.paths.sqlite_path)
    # Diagnostics are read-only: never initialize/migrate the selected Vault.
    provider = configured_embedding_provider(model_path)
    vector = None
    vector_status: dict[str, Any] = {'state': 'unavailable_fallback'}
    if provider is not None:
        selected, status = vector_registry(cfg, provider=provider).select('zvec')
        vector = selected
        vector_status = status.to_dict()
    search = HyperSearch(store, vector_store=vector, embedding_provider=provider, vector_status=vector_status)
    case_hash_filter = load_manifest_hashes(split_manifest, split_group)
    all_cases = load_case_pack(cases_path)
    cases = [case for case in all_cases if expected_citations(case)]
    if case_hash_filter is not None:
        cases = [case for case in cases if str(stable_hash(case.get('case_id'))) in case_hash_filter]

    layer_counts: Counter[str] = Counter()
    category_layer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_layer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    stratum_layer_counts: dict[str, Counter[str]] = defaultdict(Counter)
    route_counters: Counter[str] = Counter()
    vector_rank_buckets: Counter[str] = Counter()
    vector_missing_reason_counts: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []
    vector_top_n = max(1, int(vector_top_n))
    vector_deep_n = max(vector_top_n, int(vector_deep_n))

    for case in cases:
        filters = dict(case.get('filters') or {})
        expected = set(expected_citations(case))
        limit = max(int(case.get('limit') or 10), k, 10)
        req = SearchRequest(
            str(case.get('query') or ''),
            limit=limit,
            include_vector=True,
            semantic='auto',
            ranking_mode='weighted',
            **filters,
        )
        response = search.search(req)
        response_dict = response.to_dict()
        results = response_dict['results']
        top3_rank = first_rank_in_results(results, expected, limit=k)
        top_limit_rank = first_rank_in_results(results, expected, limit=limit)
        status = response.retrieval_status or {}
        route_plan = status.get('retrieval_plan') or {}
        vector_route_status = status.get('vector') or {}
        semantic_attempted = bool(vector_route_status.get('attempted'))
        if semantic_attempted:
            route_counters['semantic_attempted'] += 1
        else:
            route_counters['semantic_not_attempted'] += 1
        if vector_route_status.get('reason_code') == 'semantic_auto_satisfied':
            route_counters['semantic_auto_satisfied'] += 1
        text_health = evidence_text_health(store, expected)
        vec_rows_deep, vec_diag = vector_top_rows(vector, provider, str(case.get('query') or ''), filters, limit=vector_deep_n)
        vector_rank = first_rank_in_rows(vec_rows_deep[:vector_top_n], expected)
        vector_deep_rank = first_rank_in_rows(vec_rows_deep, expected)
        lexical_pool = collect_lexical_pool(store, str(case.get('query') or ''), filters, limit=50)
        lexical_ranks = {name: first_rank_in_rows(rows, expected) for name, rows in lexical_pool.items()}
        lexical_hit = any(rank is not None for rank in lexical_ranks.values())
        vector_hit = vector_rank is not None
        missing_reason = vector_top50_missing_reason(
            semantic_attempted=semantic_attempted,
            lexical_hit=lexical_hit,
            vector_top50_rank=vector_rank,
            vector_deep_rank=vector_deep_rank,
        )
        if missing_reason:
            vector_missing_reason_counts[missing_reason] += 1

        if top3_rank is not None:
            layer = 'hit_top3'
        elif text_health['data_issue']:
            layer = 'L4_data'
        elif not semantic_attempted:
            layer = 'L1_route'
        elif not vector_hit and not lexical_hit:
            layer = 'L2_recall'
        else:
            layer = 'L3_fusion'

        layer_counts[layer] += 1
        category = str(case.get('category') or 'unknown')
        family = source_family(case)
        category_layer_counts[category][layer] += 1
        source_layer_counts[family][layer] += 1
        stratum_layer_counts[f'{category}|{family}'][layer] += 1
        if vector_rank is None:
            vector_rank_buckets[f'missing_top{vector_top_n}'] += 1
        elif vector_rank <= 3:
            vector_rank_buckets['1_3'] += 1
        elif vector_rank <= 10:
            vector_rank_buckets['4_10'] += 1
        elif vector_rank <= 20:
            vector_rank_buckets['11_20'] += 1
        else:
            vector_rank_buckets['21_50'] += 1

        diagnostics.append({
            'case_hash': stable_hash(case.get('case_id')),
            'query_hash': stable_hash(case.get('query')),
            'category': category,
            'source_family': family,
            'query_type': str(case.get('query_type') or category),
            'layer': layer,
            'hit_top3': top3_rank is not None,
            'rank_top3': top3_rank,
            'rank_top_limit': top_limit_rank,
            'semantic_attempted': semantic_attempted,
            'semantic_reason_code': vector_route_status.get('reason_code'),
            'vector_state': vector_route_status.get('state') or vec_diag.get('state'),
            'vector_top50_rank': vector_rank,
            'vector_deep_rank': vector_deep_rank,
            'vector_top50_missing_reason': missing_reason,
            'lexical_top50_hit': lexical_hit,
            'lexical_top50_routes': sorted([name for name, rank in lexical_ranks.items() if rank is not None]),
            'first_stage_candidates': route_plan.get('first_stage_candidates'),
            'lexical_first_stage_candidates': route_plan.get('lexical_first_stage_candidates'),
            'exact_chunk_candidates': route_plan.get('exact_chunk_candidates'),
            'data_issue': bool(text_health['data_issue']),
            'evidence_rows_found': text_health['evidence_rows_found'],
            'max_compact_text_chars_bucket': '0' if text_health['max_compact_text_chars'] == 0 else ('1_3' if text_health['max_compact_text_chars'] < 4 else ('4_20' if text_health['max_compact_text_chars'] <= 20 else '21_plus')),
        })

    leak_counts = {layer: layer_counts.get(layer, 0) for layer in ('L1_route', 'L2_recall', 'L3_fusion', 'L4_data')}
    report = {
        'schema_version': 1,
        'artifact_type': 'retrieval_failure_diagnosis_redacted',
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
            'loaded_cases': len(all_cases),
            'selected_positive_cases': len(cases),
            'path_included': False,
        },
        'controls': {
            'k': k,
            'split_group': split_group,
            'split_filter_enabled': case_hash_filter is not None,
            'split_filter_count': len(case_hash_filter or set()),
            'vector_top_n': vector_top_n,
            'vector_deep_n': vector_deep_n,
            'model_configured': provider is not None,
        },
        'layer_definitions': LAYER_LABELS,
        'vector_top50_missing_reason_definitions': VECTOR_TOP50_MISSING_REASONS,
        'summary': {
            'positive_cases': len(cases),
            'hit_top3': layer_counts.get('hit_top3', 0),
            'miss_top3': len(cases) - layer_counts.get('hit_top3', 0),
            'leak_counts': leak_counts,
            'semantic_attempted_cases': route_counters.get('semantic_attempted', 0),
            'semantic_not_attempted_cases': route_counters.get('semantic_not_attempted', 0),
            'semantic_auto_satisfied_cases': route_counters.get('semantic_auto_satisfied', 0),
            'vector_top50_rank_buckets': dict(sorted(vector_rank_buckets.items())),
            'vector_top50_missing_reasons': dict(sorted(vector_missing_reason_counts.items())),
        },
        'by_category': {cat: dict(counter) for cat, counter in sorted(category_layer_counts.items())},
        'by_source_family': {src: dict(counter) for src, counter in sorted(source_layer_counts.items())},
        'by_category_source_family': {key: dict(counter) for key, counter in sorted(stratum_layer_counts.items())},
        'cases': diagnostics,
    }
    validate_redacted_artifact(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Diagnose redacted retrieval failure layers for private or fixture eval positives.')
    parser.add_argument('--vault', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--split-manifest')
    parser.add_argument('--split-group')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--model-path')
    parser.add_argument('--vector-top-n', type=int, default=50)
    parser.add_argument('--vector-deep-n', type=int, default=200)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = diagnose(
        Path(args.cases),
        Path(args.vault),
        split_manifest=Path(args.split_manifest) if args.split_manifest else None,
        split_group=args.split_group,
        k=args.k,
        model_path=args.model_path,
        vector_top_n=args.vector_top_n,
        vector_deep_n=args.vector_deep_n,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'redacted_file': out.name,
        'positive_cases': report['summary']['positive_cases'],
        'hit_top3': report['summary']['hit_top3'],
        'leak_counts': report['summary']['leak_counts'],
        'vector_top50_missing_reasons': report['summary']['vector_top50_missing_reasons'],
        'raw_queries_printed': False,
        'private_paths_printed': False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
