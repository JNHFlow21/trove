from __future__ import annotations
import json
import math
import time
from pathlib import Path
from typing import Any

from trove_core.search.eval_schema import stable_hash

from .hyper_search import HyperSearch
from .query import SearchRequest


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _expected_set(case: dict[str, Any]) -> set[str]:
    vals = case.get('expected_citations') or case.get('expected_any_citation') or []
    if isinstance(vals, str):
        vals = [vals]
    expected = set(vals)
    if case.get('expected_citation'):
        expected.add(case['expected_citation'])
    oracle = case.get('oracle') or {}
    for val in oracle.get('expected_any_citation') or []:
        expected.add(val)
    return expected


def _dcg(binary_relevance: list[int], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(binary_relevance[:k]))


def evaluate_golden(search: HyperSearch, path: Path, k: int = 3, *, semantic: str = 'auto') -> dict:
    cases = load_golden(path)
    hits = 0
    reciprocal = 0.0
    latencies = []
    completeness = 0
    precision_sum = 0.0
    ap_sum = 0.0
    ndcg3_sum = 0.0
    ndcg10_sum = 0.0
    path_counts: dict[str, int] = {}
    per_source: dict[str, dict[str, float]] = {}
    case_results = []
    for case in cases:
        filters = case.get('filters', {})
        start = time.perf_counter()
        resp = search.search(SearchRequest(case['query'], limit=max(k, 10), semantic=semantic, **filters))
        latencies.append((time.perf_counter() - start) * 1000)
        citations = [r.citation for r in resp.results]
        expected = _expected_set(case)
        if not expected:
            expected = {case.get('expected_citation', '')} - {''}
        relevant = [1 if c in expected else 0 for c in citations]
        if any(relevant[:k]):
            hits += 1
        if any(relevant):
            reciprocal += 1.0 / (relevant.index(1) + 1)
        precision_sum += sum(relevant[:k]) / max(k, 1)
        seen_relevant = 0
        ap = 0.0
        for idx, rel in enumerate(relevant, start=1):
            if rel:
                seen_relevant += 1
                ap += seen_relevant / idx
        ap_sum += ap / max(len(expected), 1)
        ideal = [1] * min(len(expected), len(relevant))
        ndcg3_sum += (_dcg(relevant, 3) / (_dcg(ideal, 3) or 1.0))
        ndcg10_sum += (_dcg(relevant, 10) / (_dcg(ideal, 10) or 1.0))
        if all(r.citation and r.account_label and r.conversation_title and r.sender_name and r.timestamp for r in resp.results[:k]):
            completeness += 1
        for r in resp.results:
            for path_name in r.retrieval_paths:
                path_counts[path_name] = path_counts.get(path_name, 0) + 1
            bucket = per_source.setdefault(r.source_type, {'results': 0, 'hits': 0})
            bucket['results'] += 1
            if r.citation in expected:
                bucket['hits'] += 1
        case_results.append({
            'case_id': case.get('case_id') or stable_hash(case.get('query')),
            'hit': any(relevant[:k]),
            'top_citation_hashes': [stable_hash(c) for c in citations[:k]],
            'retrieval_status': resp.retrieval_status,
        })
    total = len(cases) or 1
    return {
        'queries': len(cases),
        f'recall_at_{k}': hits / total,
        f'precision_at_{k}': precision_sum / total,
        'mrr': reciprocal / total,
        'map': ap_sum / total,
        'ndcg_at_3': ndcg3_sum / total,
        'ndcg_at_10': ndcg10_sum / total,
        'avg_latency_ms': round(sum(latencies) / total, 3),
        'p95_latency_ms': round(sorted(latencies)[int(max(0, min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)))] if latencies else 0, 3),
        'evidence_completeness': completeness / total,
        'retrieval_path_participation': path_counts,
        'per_source': per_source,
        'surface_parity': {'cli': 'covered-by-e2e', 'mcp': 'covered-by-e2e', 'daemon': 'covered-by-contract', 'agent': 'covered-by-capability-map'},
        'semantic_mode': semantic,
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
        },
        'cases': case_results,
    }
