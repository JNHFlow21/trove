from __future__ import annotations

import math
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trove_core.bounds import BoundedLimit, FUSION_CANDIDATES, RERANK_CANDIDATES, RETRIEVAL_CANDIDATES
from trove_core.runtime import build_search_engine, configured_embedding_provider, vector_registry, warm_search_engine
from trove_core.search.evidence import row_to_evidence
from trove_core.search.eval_schema import (
    EVAL_MODES,
    case_pack_quality_stats,
    expected_citations,
    expected_conversations,
    load_case_pack,
    stable_hash,
    validate_redacted_artifact,
)
from trove_core.search.evidence_provenance import build_artifact_provenance, write_evidence_artifact
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.local_reranker import warm_local_reranker
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore, open_store
from trove_core.vault.config import VaultConfig


class EvalCasePackCompatibilityError(RuntimeError):
    """Raised when positive case-pack oracles do not exist in the selected index."""

    code = 'case_pack_incompatible_with_index'

    def __init__(self, compatibility: dict[str, Any]):
        self.compatibility = compatibility
        super().__init__(self.code)

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'artifact_type': 'retrieval_eval_case_pack_compatibility_error',
            'ok': False,
            'error_code': self.code,
            'recommended_action': 'regenerate_or_rebind_case_pack_for_current_index',
            'compatibility': self.compatibility,
            'privacy': {
                'raw_queries_included': False,
                'raw_snippets_included': False,
                'raw_citations_included': False,
                'private_paths_included': False,
            },
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _dcg(binary_relevance: list[int], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(binary_relevance[:k]))


def _result_dicts_from_rows(rows: list[Any], query: str, path_name: str, base_score: float = 1.0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out.append(row_to_evidence(row, query, [path_name], base_score - (idx * 0.01)).to_dict())
    return out


class _DegradedVector:
    def search(self, *_args, **_kwargs):
        raise RuntimeError('simulated_vector_degraded')


class _DummyProvider:
    dimensions = 1

    def embed(self, _text: str) -> list[float]:
        return [0.0]


def _vector_search(cfg: VaultConfig, store: SQLiteStore, query: str, filters: dict[str, str], limit: int, model_path: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        provider = configured_embedding_provider(model_path)
    except Exception as exc:
        return [], {'state': 'unavailable_fallback', 'available': False, 'reason_code': exc.__class__.__name__}
    if provider is None:
        return [], {'state': 'unavailable_fallback', 'available': False, 'reason_code': 'embedding_provider_missing'}
    registry = vector_registry(cfg, provider=provider)
    vector, status = registry.select('zvec')
    status_dict = status.to_dict()
    if status_dict.get('state') != 'available' or vector is None:
        return [], status_dict
    try:
        rows = vector.search(query, filters=filters, limit=limit, provider=provider)
    except TypeError:
        rows = vector.search(query, filters=filters, limit=limit)
    except Exception as exc:
        return [], {'state': 'degraded', 'available': False, 'reason_code': exc.__class__.__name__}
    return _result_dicts_from_rows(rows, query, 'vector'), {'state': 'available', 'available': True, 'selected_backend': 'zvec'}


def _build_vector_context(
    cfg: VaultConfig,
    model_path: str | None,
    *,
    cloud: bool = False,
) -> tuple[object | None, object | None, dict[str, Any]]:
    try:
        provider = configured_embedding_provider(
            model_path,
            strict=cloud,
            vault_root=cfg.root if cloud else None,
            prefer_cloud=cloud,
        )
    except Exception as exc:
        return None, None, {'state': 'unavailable_fallback', 'available': False, 'reason_code': exc.__class__.__name__}
    if provider is None:
        return None, None, {'state': 'unavailable_fallback', 'available': False, 'reason_code': 'embedding_provider_missing'}
    registry = vector_registry(cfg, provider=provider)
    vector, status = registry.select('zvec')
    return provider, vector, status.to_dict()


def _episode_status_from_retrieval_status(status: dict[str, Any]) -> dict[str, Any]:
    direct = status.get('multi_hop_episode')
    if isinstance(direct, dict):
        return direct
    plan = status.get('retrieval_plan')
    if isinstance(plan, dict) and isinstance(plan.get('multi_hop_episode'), dict):
        return dict(plan['multi_hop_episode'])
    return {}


def run_mode(
    cfg: VaultConfig,
    store: SQLiteStore,
    case: dict[str, Any],
    mode: str,
    *,
    k: int,
    model_path: str | None = None,
    reranker_model_path: str | None = None,
    vector_context: tuple[object | None, object | None, dict[str, Any]] | None = None,
    hybrid_search: HyperSearch | None = None,
    cloud_queries: Any | None = None,
    retrieval_candidate_limit: int = 200,
    fusion_candidate_limit: int = 200,
    reranker_candidate_limit: int = 50,
) -> dict[str, Any]:
    retrieval_candidate_limit = int(BoundedLimit(
        retrieval_candidate_limit,
        field='retrieval_candidate_limit',
        spec=RETRIEVAL_CANDIDATES,
    ))
    fusion_candidate_limit = int(BoundedLimit(
        fusion_candidate_limit,
        field='fusion_candidate_limit',
        spec=FUSION_CANDIDATES,
    ))
    reranker_candidate_limit = int(BoundedLimit(
        reranker_candidate_limit,
        field='reranker_candidate_limit',
        spec=RERANK_CANDIDATES,
    ))
    if mode not in EVAL_MODES:
        raise ValueError(f'unsupported eval mode: {mode}')
    query = case['query']
    limit = max(int(case.get('limit') or k), k, 10)
    filters = dict(case.get('filters') or {})
    start = time.perf_counter()
    mode_note = None
    vector_status: dict[str, Any] = {'state': 'not_requested', 'available': False}
    reranker_status: dict[str, Any] = {'state': 'not_requested'}
    episode_status: dict[str, Any] = {'state': 'not_requested'}
    phase_latency_ms: dict[str, float] = {}
    candidate_citation_hashes: list[str] = []
    if mode == 'exact':
        results = _result_dicts_from_rows(store.exact_search(query, filters=filters, limit=limit), query, 'exact', 10.0)
    elif mode == 'fts':
        results = _result_dicts_from_rows(store.fts_search_filtered(query, filters=filters, limit=limit), query, 'fts', 7.0)
    elif mode == 'metadata':
        results = _result_dicts_from_rows(store.metadata_search(query, filters=filters, limit=limit) if filters else [], query, 'metadata', 3.0)
    elif mode == 'parent_child':
        results = _result_dicts_from_rows(store.chunk_search(query, filters=filters, limit=limit), query, 'parent_child', 9.0)
    elif mode == 'vector':
        if vector_context is None:
            results, vector_status = _vector_search(cfg, store, query, filters, limit, model_path)
        else:
            provider, vector, status = vector_context
            vector_status = dict(status)
            if status.get('state') != 'available' or vector is None or provider is None:
                results = []
            else:
                try:
                    try:
                        rows = vector.search(query, filters=filters, limit=limit, provider=provider)  # type: ignore[attr-defined]
                    except TypeError:
                        rows = vector.search(query, filters=filters, limit=limit)  # type: ignore[attr-defined]
                    results = _result_dicts_from_rows(rows, query, 'vector')
                    vector_status = {'state': 'available', 'available': True, 'selected_backend': status.get('selected_backend', 'zvec')}
                except Exception as exc:
                    results = []
                    vector_status = {'state': 'degraded', 'available': False, 'reason_code': exc.__class__.__name__}
    elif mode == 'vector-unavailable':
        resp = HyperSearch(store).search(SearchRequest(
            query,
            limit=limit,
            include_vector=True,
            retrieval_candidate_limit=retrieval_candidate_limit,
            fusion_candidate_limit=fusion_candidate_limit,
            reranker_candidate_limit=reranker_candidate_limit,
            **filters,
        ))
        results = resp.to_dict()['results']
        candidate_citation_hashes = [stable_hash(value) for value in resp.candidate_citations]
        vector_status = (resp.retrieval_status or {}).get('vector') or {}
    elif mode == 'vector-degraded':
        resp = HyperSearch(store, vector_store=_DegradedVector(), embedding_provider=_DummyProvider(), vector_status={'state': 'available', 'selected_backend': 'zvec'}).search(SearchRequest(
            query,
            limit=limit,
            include_vector=True,
            retrieval_candidate_limit=retrieval_candidate_limit,
            fusion_candidate_limit=fusion_candidate_limit,
            reranker_candidate_limit=reranker_candidate_limit,
            **filters,
        ))
        results = resp.to_dict()['results']
        candidate_citation_hashes = [stable_hash(value) for value in resp.candidate_citations]
        vector_status = (resp.retrieval_status or {}).get('vector') or {}
    else:
        search = hybrid_search or build_search_engine(cfg)
        if mode == 'hybrid-weighted':
            req = SearchRequest(query, limit=limit, include_vector=True, ranking_mode='weighted', retrieval_candidate_limit=retrieval_candidate_limit, fusion_candidate_limit=fusion_candidate_limit, reranker_candidate_limit=reranker_candidate_limit, **filters)
        elif mode == 'hybrid-rrf':
            req = SearchRequest(query, limit=limit, include_vector=True, ranking_mode='rrf', retrieval_candidate_limit=retrieval_candidate_limit, fusion_candidate_limit=fusion_candidate_limit, reranker_candidate_limit=reranker_candidate_limit, **filters)
        elif mode == 'feature-rerank':
            req = SearchRequest(query, limit=limit, include_vector=True, ranking_mode='feature', reranker_mode='features', retrieval_candidate_limit=retrieval_candidate_limit, fusion_candidate_limit=fusion_candidate_limit, reranker_candidate_limit=reranker_candidate_limit, **filters)
        elif mode == 'local-reranker':
            req = SearchRequest(query, limit=limit, include_vector=True, ranking_mode='feature', reranker_mode='local-bge', reranker_model_path=reranker_model_path, retrieval_candidate_limit=retrieval_candidate_limit, fusion_candidate_limit=fusion_candidate_limit, reranker_candidate_limit=reranker_candidate_limit, **filters)
        elif mode == 'cloud-reranker' and cloud_queries is not None:
            # Evaluate the actual selected-Vault product path: cloud hybrid
            # retrieval/episodes first, then the same bounded application
            # rerank method used by CLI/API/MCP.  HyperSearch's legacy direct
            # cloud mode intentionally lacks continuous-policy authority.
            from trove_core.application.queries import SearchQuery
            from trove_core.providers.config import DEFAULT_CLOUD_RERANK_TOP_K

            cloud_window = min(reranker_candidate_limit, DEFAULT_CLOUD_RERANK_TOP_K)
            local_limit = max(limit, cloud_window)
            req = SearchRequest(
                query,
                limit=local_limit,
                include_vector=True,
                ranking_mode='feature',
                reranker_mode='features',
                retrieval_candidate_limit=retrieval_candidate_limit,
                fusion_candidate_limit=fusion_candidate_limit,
                reranker_candidate_limit=reranker_candidate_limit,
                **filters,
            )
            resp = search.search(req)
            cloud_query = SearchQuery(
                query,
                limit=limit,
                ranking_mode='feature',
                reranker_mode='cloud-qwen3',
                allow_cloud_rerank=True,
                reranker_timeout_ms=5000,
                retrieval_candidate_limit=retrieval_candidate_limit,
                fusion_candidate_limit=fusion_candidate_limit,
                reranker_candidate_limit=reranker_candidate_limit,
                **filters,
            )
            resp = cloud_queries._cloud_rerank_response(
                cloud_query,
                resp,
                result_limit=limit,
                candidate_limit=cloud_window,
            )
            results = resp.to_dict()['results']
            candidate_citation_hashes = [stable_hash(value) for value in resp.candidate_citations]
            vector_status = (resp.retrieval_status or {}).get('vector') or {}
            reranker_status = (resp.retrieval_status or {}).get('reranker') or {}
            episode_status = _episode_status_from_retrieval_status(resp.retrieval_status or {})
            raw_phase_latency = (resp.retrieval_status or {}).get('phase_latency_ms') or {}
            phase_latency_ms = {
                key: round(float(raw_phase_latency.get(key) or 0.0), 3)
                for key in ('retrieval', 'fusion', 'rerank')
            }
            req = None
        elif mode == 'cloud-reranker':
            req = SearchRequest(query, limit=limit, include_vector=True, ranking_mode='feature', reranker_mode='cloud-qwen3', retrieval_candidate_limit=retrieval_candidate_limit, fusion_candidate_limit=fusion_candidate_limit, reranker_candidate_limit=reranker_candidate_limit, **filters)
        else:
            raise ValueError(f'unsupported eval mode: {mode}')
        if req is not None:
            resp = search.search(req)
            results = resp.to_dict()['results']
            candidate_citation_hashes = [stable_hash(value) for value in resp.candidate_citations]
            vector_status = (resp.retrieval_status or {}).get('vector') or {}
            reranker_status = (resp.retrieval_status or {}).get('reranker') or {}
            episode_status = _episode_status_from_retrieval_status(resp.retrieval_status or {})
            raw_phase_latency = (resp.retrieval_status or {}).get('phase_latency_ms') or {}
            phase_latency_ms = {
                key: round(float(raw_phase_latency.get(key) or 0.0), 3)
                for key in ('retrieval', 'fusion', 'rerank')
            }
        if mode == 'local-reranker' and reranker_status.get('state') != 'available':
            mode_note = reranker_status.get('reason_code') or 'local_reranker_unavailable_feature_fallback'
        if mode == 'cloud-reranker' and reranker_status.get('state') != 'available':
            mode_note = reranker_status.get('reason_code') or 'cloud_reranker_unavailable_feature_fallback'
    elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
    if not candidate_citation_hashes:
        candidate_citation_hashes = [
            stable_hash(value)
            for result in results
            for value in (result.get('citation'), result.get('context_anchor'), result.get('parent_citation'))
            if value
        ]
        candidate_citation_hashes = list(dict.fromkeys(candidate_citation_hashes))
    return {
        'mode': mode,
        'mode_note': mode_note,
        'elapsed_ms': elapsed_ms,
        'vector_status': vector_status,
        'reranker_status': reranker_status,
        'episode_status': episode_status,
        'phase_latency_ms': phase_latency_ms,
        'candidate_citation_hashes': candidate_citation_hashes,
        'results': results[:limit],
    }


def _case_relevance(case: dict[str, Any], results: list[dict[str, Any]]) -> tuple[list[int], set[str]]:
    expected = set(expected_citations(case))
    oracle = case.get('oracle') or {}
    if not expected and oracle.get('semantic_min_results'):
        min_results = int(oracle.get('min_results') or 1)
        relevant = [1 if idx < min_results else 0 for idx, _ in enumerate(results)]
        return relevant, expected
    relevant = []
    for r in results:
        relevant.append(1 if expected.intersection(_result_citations(r)) else 0)
    return relevant, expected


def _result_citations(result: dict[str, Any]) -> set[str]:
    values = {
        str(result.get(key))
        for key in ('citation', 'context_anchor', 'parent_citation')
        if result.get(key)
    }
    supporting = result.get('supporting_citations')
    if isinstance(supporting, (list, tuple)):
        values.update(str(value) for value in supporting if value)
    return values


def _matched_expected_citations(case: dict[str, Any], results: list[dict[str, Any]]) -> set[str]:
    expected = set(expected_citations(case))
    matched: set[str] = set()
    for r in results:
        matched.update(expected.intersection(_result_citations(r)))
    return matched


def _negative_excluded_hit(case: dict[str, Any], results: list[dict[str, Any]], *, k: int) -> bool:
    oracle = case.get('oracle') or {}
    excluded = {str(v) for v in (oracle.get('negative_excluded_citations') or []) if v}
    if not excluded:
        return False
    for r in results[:k]:
        if excluded.intersection(_result_citations(r)):
            return True
    return False


def _case_is_negative_only(case: dict[str, Any]) -> bool:
    oracle = case.get('oracle') or {}
    if oracle.get('negative_no_results'):
        return True
    return bool(oracle.get('negative_excluded_citations')) and not expected_citations(case)


def _case_has_positive_expectation(case: dict[str, Any]) -> bool:
    if _case_is_negative_only(case):
        return False
    oracle = case.get('oracle') or {}
    return bool(expected_citations(case) or oracle.get('semantic_min_results'))


def _conversation_context_success(case: dict[str, Any], results: list[dict[str, Any]], *, k: int) -> bool | None:
    expected_convs = set(expected_conversations(case))
    if not expected_convs:
        return None
    return any(str(r.get('conversation_id') or '') in expected_convs for r in results[:k])


def _failure_class(case: dict[str, Any], results: list[dict[str, Any]], relevant: list[int], *, k: int, context_ok: bool | None) -> str | None:
    oracle = case.get('oracle') or {}
    if oracle.get('negative_no_results') and results:
        return 'negative_false_positive'
    if oracle.get('negative_no_results') and not results:
        return None
    if _negative_excluded_hit(case, results, k=k):
        return 'negative_excluded_citation_in_topk'
    if oracle.get('negative_excluded_citations') and not expected_citations(case):
        return None
    if not results:
        return 'no_results'
    expected = set(expected_citations(case))
    expected_all = {str(v) for v in (oracle.get('expected_all_citations') or []) if v}
    if expected_all and not expected_all.issubset(_matched_expected_citations(case, results[:k])):
        return 'expected_all_missing'
    if expected and not any(relevant[:k]):
        return 'expected_missing'
    expected_paths = set(oracle.get('expected_retrieval_paths_any') or [])
    if expected_paths:
        seen_paths = {p for r in results[:k] for p in (r.get('retrieval_paths') or [])}
        if not expected_paths.intersection(seen_paths):
            return 'expected_retrieval_path_missing'
    expected_family = oracle.get('expected_source_family')
    if expected_family and not any(r.get('source_type') == expected_family for r in results[:k]):
        return 'source_family_missing'
    if context_ok is False:
        return 'context_failed'
    return None


def _context_success(store: SQLiteStore, case: dict[str, Any], results: list[dict[str, Any]], relevant: list[int]) -> bool | None:
    oracle = case.get('oracle') or {}
    context_oracle = case.get('context_oracle') or {}
    anchor = context_oracle.get('anchor_citation') or oracle.get('context_anchor')
    if not anchor:
        # If the case explicitly asks for context, use the first expected or first relevant hit.
        if not case.get('context'):
            return None
        citations = expected_citations(case)
        anchor = citations[0] if citations else None
    if not anchor:
        return None
    before = int(context_oracle.get('before') or 5)
    after = int(context_oracle.get('after') or 5)
    rows = store.context_window(anchor, before=before, after=after)
    return bool(rows and any(r['citation'] == anchor for r in rows) and len(rows) <= before + after + 1)


def _evaluate_case(store: SQLiteStore, case: dict[str, Any], mode_result: dict[str, Any], *, k: int) -> dict[str, Any]:
    results = mode_result['results']
    relevant, expected = _case_relevance(case, results)
    oracle = case.get('oracle') or {}
    negative = bool(oracle.get('negative_no_results'))
    negative_only = _case_is_negative_only(case)
    positive_expected = _case_has_positive_expectation(case)
    expected_all = {str(v) for v in (oracle.get('expected_all_citations') or []) if v}
    if negative:
        hit = not results
    elif negative_only:
        hit = not _negative_excluded_hit(case, results, k=k)
    elif expected_all:
        hit = expected_all.issubset(_matched_expected_citations(case, results[:k]))
    else:
        hit = any(relevant[:k])
    precision = (sum(relevant[:k]) / max(k, 1)) if positive_expected else 0.0
    reciprocal = 0.0
    if any(relevant):
        reciprocal = 1.0 / (relevant.index(1) + 1)
    seen_relevant = 0
    ap = 0.0
    for idx, rel in enumerate(relevant, start=1):
        if rel:
            seen_relevant += 1
            ap += seen_relevant / idx
    # Conversation/parent relaxation can make more retrieved rows relevant than
    # there are explicit expected citations.  AP/NDCG must normalize by the
    # same reachable relevant population used by the ideal ranking, otherwise
    # a one-citation oracle with ten same-conversation hits can score > 1.
    relevant_population = max(len(expected), sum(relevant), 1)
    average_precision = ap / relevant_population if positive_expected else 0.0
    ideal = [1] * relevant_population
    ndcg3 = _dcg(relevant, 3) / (_dcg(ideal, 3) or 1.0)
    ndcg10 = _dcg(relevant, 10) / (_dcg(ideal, 10) or 1.0)
    context_ok = _context_success(store, case, results, relevant)
    conversation_context_ok = _conversation_context_success(case, results, k=k)
    failure = None if hit else _failure_class(case, results, relevant, k=k, context_ok=context_ok)
    if hit:
        # Path/source/context oracles can still fail even when the citation matched.
        failure = _failure_class(case, results, relevant, k=k, context_ok=context_ok)
        hit = failure is None
    top = results[:k]
    expected_hashes = {stable_hash(c) for c in expected_citations(case)}
    candidate_hashes = {str(value) for value in (mode_result.get('candidate_citation_hashes') or []) if value}
    candidate_matches = expected_hashes.intersection(candidate_hashes)
    candidate_recall = (len(candidate_matches) / len(expected_hashes)) if expected_hashes else None
    reranker_status = mode_result.get('reranker_status') or {}
    episode_status = mode_result.get('episode_status') or {}
    selector_status = episode_status.get('selector') if isinstance(episode_status.get('selector'), dict) else {}
    reranker_identity = reranker_status.get('identity') if isinstance(reranker_status.get('identity'), dict) else None
    phase_latency = mode_result.get('phase_latency_ms') or {}
    return {
        'case_ref': stable_hash(case.get('case_id')),
        'case_hash': stable_hash(case.get('case_id')),
        'category': case.get('category'),
        'query_hash': stable_hash(case.get('query')),
        'query_length': len(case.get('query') or ''),
        'expected_citation_hashes': [stable_hash(c) for c in expected_citations(case)],
        'top_citation_hashes': [stable_hash(r.get('citation')) for r in top],
        'candidate_citation_hashes': sorted(candidate_hashes),
        'candidate_count': len(candidate_hashes),
        'candidate_hit': bool(candidate_matches) if expected_hashes else None,
        'candidate_recall': candidate_recall,
        'retrieval_paths': sorted({p for r in top for p in (r.get('retrieval_paths') or [])}),
        'source_families': sorted({str(r.get('source_type') or '') for r in top if r.get('source_type')}),
        'positive_expected': positive_expected,
        'negative_only': negative_only,
        'hit': bool(hit),
        'precision': precision,
        'reciprocal_rank': reciprocal,
        'average_precision': average_precision,
        'ndcg_at_3': ndcg3,
        'ndcg_at_10': ndcg10,
        'context_success': context_ok,
        'conversation_context_success': conversation_context_ok,
        'failure_class': failure,
        'result_count': len(results),
        'latency_ms': mode_result['elapsed_ms'],
        'phase_latency_ms': {
            key: round(float(phase_latency.get(key) or 0.0), 3)
            for key in ('retrieval', 'fusion', 'rerank')
        } if phase_latency else {},
        'vector_state': (mode_result.get('vector_status') or {}).get('state'),
        'reranker_state': reranker_status.get('state'),
        'reranker_invoked': bool(reranker_status.get('invoked')),
        'reranker_identity': reranker_identity,
        'reranker_elapsed_ms': (
            round(float(reranker_status.get('elapsed_ms')), 3)
            if isinstance(reranker_status.get('elapsed_ms'), (int, float))
            else None
        ),
        'reranker_input_tokens': (
            int(reranker_status.get('input_tokens'))
            if isinstance(reranker_status.get('input_tokens'), int)
            else 0
        ),
        'reranker_estimated_cost_usd': (
            float(reranker_status.get('estimated_cost_usd'))
            if isinstance(reranker_status.get('estimated_cost_usd'), (int, float))
            else 0.0
        ),
        'episode_state': episode_status.get('state'),
        'selector_state': selector_status.get('state'),
        'selector_input_tokens': (
            int(selector_status.get('input_tokens'))
            if isinstance(selector_status.get('input_tokens'), int)
            else 0
        ),
        'selector_output_tokens': (
            int(selector_status.get('output_tokens'))
            if isinstance(selector_status.get('output_tokens'), int)
            else 0
        ),
        'selector_estimated_cost_usd': (
            float(selector_status.get('estimated_cost_usd'))
            if isinstance(selector_status.get('estimated_cost_usd'), (int, float))
            else 0.0
        ),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct) - 1)))
    return round(ordered[idx], 3)


def _summarize_mode(case_results: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    def result_is_positive(c: dict[str, Any]) -> bool:
        return bool(c.get('positive_expected')) or (c.get('positive_expected') is None and bool(c.get('expected_citation_hashes')))

    total_cases = len(case_results)
    total = total_cases or 1
    positive_cases = [c for c in case_results if result_is_positive(c)]
    negative_cases = [c for c in case_results if bool(c.get('negative_only'))]
    positive_total = len(positive_cases)
    positive_denominator = positive_total or 1
    positive_hits = sum(1 for c in positive_cases if c['hit'])
    all_hits = sum(1 for c in case_results if c['hit'])
    negative_passes = sum(1 for c in negative_cases if c['hit'])
    latencies = [float(c.get('latency_ms') or 0) for c in case_results]
    phase_latencies = {
        phase: [
            float((c.get('phase_latency_ms') or {}).get(phase) or 0.0)
            for c in case_results
            if phase in (c.get('phase_latency_ms') or {})
        ]
        for phase in ('retrieval', 'fusion', 'rerank')
    }
    reranker_latencies = [
        float(c['reranker_elapsed_ms'])
        for c in case_results
        if isinstance(c.get('reranker_elapsed_ms'), (int, float))
    ]
    candidate_cases = [c for c in positive_cases if c.get('candidate_recall') is not None]
    reranker_states = Counter(str(c.get('reranker_state') or 'unknown') for c in case_results)
    reranker_identities = {
        json.dumps(c['reranker_identity'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        for c in case_results
        if isinstance(c.get('reranker_identity'), dict)
    }
    route_counts: Counter[str] = Counter()
    category_totals: dict[str, Counter[str]] = defaultdict(Counter)
    context_total = 0
    context_ok = 0
    conversation_context_total = 0
    conversation_context_ok = 0
    for c in case_results:
        for p in c.get('retrieval_paths') or []:
            route_counts[p] += 1
        category_totals[c['category']]['total'] += 1
        if result_is_positive(c):
            category_totals[c['category']]['positive_total'] += 1
            if c['hit']:
                category_totals[c['category']]['positive_hits'] += 1
        if c.get('negative_only'):
            category_totals[c['category']]['negative_total'] += 1
            if c['hit']:
                category_totals[c['category']]['negative_hits'] += 1
        if c['hit']:
            category_totals[c['category']]['hits'] += 1
        if c.get('context_success') is not None:
            context_total += 1
            if c.get('context_success'):
                context_ok += 1
        if c.get('conversation_context_success') is not None:
            conversation_context_total += 1
            if c.get('conversation_context_success'):
                conversation_context_ok += 1
    return {
        'queries': total_cases,
        'positive_queries': positive_total,
        'negative_only_queries': len(negative_cases),
        f'recall_at_{k}': positive_hits / positive_denominator,
        f'precision_at_{k}': sum(float(c['precision']) for c in positive_cases) / positive_denominator,
        'mrr': sum(float(c['reciprocal_rank']) for c in positive_cases) / positive_denominator,
        'map': sum(float(c['average_precision']) for c in positive_cases) / positive_denominator,
        'negative_pass_rate': (negative_passes / len(negative_cases)) if negative_cases else None,
        'candidate_evaluable_queries': len(candidate_cases),
        'candidate_hit_rate': (
            sum(1 for c in candidate_cases if c.get('candidate_hit')) / len(candidate_cases)
            if candidate_cases else None
        ),
        'candidate_recall': (
            sum(float(c.get('candidate_recall') or 0.0) for c in candidate_cases) / len(candidate_cases)
            if candidate_cases else None
        ),
        'reranker_invoked_queries': sum(1 for c in case_results if c.get('reranker_invoked')),
        'reranker_states': dict(sorted(reranker_states.items())),
        'reranker_identities': [json.loads(value) for value in sorted(reranker_identities)],
        'case_success_rate': all_hits / total,
        'ndcg_at_3': sum(float(c['ndcg_at_3']) for c in case_results) / total,
        'ndcg_at_10': sum(float(c['ndcg_at_10']) for c in case_results) / total,
        'avg_latency_ms': round(sum(latencies) / total, 3),
        'p50_latency_ms': _percentile(latencies, 0.50),
        'p95_latency_ms': _percentile(latencies, 0.95),
        'p99_latency_ms': _percentile(latencies, 0.99),
        'phase_latency_ms': {
            phase: {
                'samples': len(values),
                'p50': _percentile(values, 0.50),
                'p95': _percentile(values, 0.95),
                'p99': _percentile(values, 0.99),
            }
            for phase, values in phase_latencies.items()
        },
        'reranker_latency_ms': {
            'samples': len(reranker_latencies),
            'p50': _percentile(reranker_latencies, 0.50),
            'p95': _percentile(reranker_latencies, 0.95),
            'p99': _percentile(reranker_latencies, 0.99),
        },
        'cloud_usage': {
            'reranker_input_tokens': sum(int(c.get('reranker_input_tokens') or 0) for c in case_results),
            'reranker_estimated_cost_usd': round(sum(float(c.get('reranker_estimated_cost_usd') or 0.0) for c in case_results), 9),
            'selector_input_tokens': sum(int(c.get('selector_input_tokens') or 0) for c in case_results),
            'selector_output_tokens': sum(int(c.get('selector_output_tokens') or 0) for c in case_results),
            'selector_estimated_cost_usd': round(sum(float(c.get('selector_estimated_cost_usd') or 0.0) for c in case_results), 9),
        },
        'context_success_rate': (context_ok / context_total) if context_total else None,
        'conversation_context_success_rate': (conversation_context_ok / conversation_context_total) if conversation_context_total else None,
        'retrieval_path_participation': dict(sorted(route_counts.items())),
        'per_category': {
            cat: {
                'queries': cnt['total'],
                'positive_queries': cnt['positive_total'],
                'negative_only_queries': cnt['negative_total'],
                f'recall_at_{k}': cnt['positive_hits'] / max(cnt['positive_total'], 1),
                'negative_pass_rate': (cnt['negative_hits'] / cnt['negative_total']) if cnt['negative_total'] else None,
                'case_success_rate': cnt['hits'] / max(cnt['total'], 1),
            }
            for cat, cnt in sorted(category_totals.items())
        },
        'failure_classes': dict(sorted(Counter(c.get('failure_class') or 'none' for c in case_results).items())),
    }


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    category_filters: list[str] | None = None,
    case_hash_filters: set[str] | None = None,
    max_cases: int | None = None,
    sample_seed: int = 0,
) -> list[dict[str, Any]]:
    selected = list(cases)
    wanted = {c for c in (category_filters or []) if c}
    if wanted:
        selected = [case for case in selected if str(case.get('category')) in wanted]
    wanted_hashes = {str(c) for c in (case_hash_filters or set()) if c}
    if wanted_hashes:
        selected = [case for case in selected if str(stable_hash(case.get('case_id'))) in wanted_hashes]
    if max_cases is not None and max_cases >= 0 and len(selected) > max_cases:
        rng = random.Random(sample_seed)
        selected = sorted(selected, key=lambda c: stable_hash(c.get('case_id')) or '')
        rng.shuffle(selected)
        selected = sorted(selected[:max_cases], key=lambda c: stable_hash(c.get('case_id')) or '')
    return selected


_SEARCHABLE_CITATION_COLUMNS = (
    ('messages', 'citation'),
    ('evidence_chunks', 'chunk_citation'),
    ('evidence_chunks', 'parent_citation'),
    ('moment_items', 'citation'),
    ('moment_interactions', 'citation'),
    ('favorites', 'citation'),
    ('transcripts', 'citation'),
    ('image_observations', 'citation'),
    ('observations', 'citation'),
)


def _existing_citation_refs(store: SQLiteStore, citations: set[str]) -> set[str]:
    """Resolve citation presence using identifiers only; never load evidence content."""

    if not citations:
        return set()
    found: set[str] = set()
    with store.connect() as conn:
        for table, column in _SEARCHABLE_CITATION_COLUMNS:
            if not store._table_exists(conn, table):
                continue
            remaining = sorted(citations - found)
            for start in range(0, len(remaining), 400):
                batch = remaining[start:start + 400]
                placeholders = ','.join('?' for _ in batch)
                rows = conn.execute(
                    f'SELECT DISTINCT "{column}" AS ref FROM "{table}" '
                    f'WHERE "{column}" IN ({placeholders})',
                    batch,
                )
                found.update(str(row['ref']) for row in rows if row['ref'])
    return found


def _existing_conversation_refs(store: SQLiteStore, conversation_ids: set[str]) -> set[str]:
    """Resolve conversation presence using identifiers only."""

    if not conversation_ids:
        return set()
    found: set[str] = set()
    with store.connect() as conn:
        for table in ('conversations', 'messages'):
            if not store._table_exists(conn, table):
                continue
            remaining = sorted(conversation_ids - found)
            for start in range(0, len(remaining), 400):
                batch = remaining[start:start + 400]
                placeholders = ','.join('?' for _ in batch)
                rows = conn.execute(
                    f'SELECT DISTINCT conversation_id AS ref FROM "{table}" '
                    f'WHERE conversation_id IN ({placeholders})',
                    batch,
                )
                found.update(str(row['ref']) for row in rows if row['ref'])
    return found


def check_case_pack_compatibility(store: SQLiteStore, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Stop evaluation only when a case has no usable oracle anchor in the index."""

    citation_requirements = [expected_citations(case) for case in cases]
    conversation_requirements = [expected_conversations(case) for case in cases]
    citation_refs = {ref for refs in citation_requirements for ref in refs}
    conversation_refs = {
        ref
        for citation_refs_for_case, refs in zip(citation_requirements, conversation_requirements)
        if not citation_refs_for_case
        for ref in refs
    }
    existing_citations = _existing_citation_refs(store, citation_refs)
    existing_conversations = _existing_conversation_refs(store, conversation_refs)

    citation_oracle_cases = 0
    missing_citation_cases = 0
    conversation_oracle_cases = 0
    conversation_only_oracle_cases = 0
    missing_conversation_only_cases = 0
    incompatible_cases: set[int] = set()
    for idx, (expected_citations_for_case, expected_convs) in enumerate(zip(citation_requirements, conversation_requirements)):
        if expected_convs:
            conversation_oracle_cases += 1
        if expected_citations_for_case:
            citation_oracle_cases += 1
            if not existing_citations.intersection(expected_citations_for_case):
                missing_citation_cases += 1
                incompatible_cases.add(idx)
        elif expected_convs:
            # Non-message sources can legitimately carry a conversation-shaped
            # source id that does not exist in the WeChat conversations table.
            # A valid citation anchor is therefore authoritative; conversation
            # ids are a staleness signal only for conversation-only cases.
            conversation_only_oracle_cases += 1
            if not existing_conversations.intersection(expected_convs):
                missing_conversation_only_cases += 1
                incompatible_cases.add(idx)

    compatibility = {
        'state': 'compatible' if not incompatible_cases else 'incompatible',
        'selected_cases': len(cases),
        'incompatible_cases': len(incompatible_cases),
        'citation_oracle_cases': citation_oracle_cases,
        'missing_citation_oracle_cases': missing_citation_cases,
        'citation_refs': {
            'total': len(citation_refs),
            'found': len(existing_citations),
            'missing': len(citation_refs - existing_citations),
        },
        'conversation_oracle_cases': conversation_oracle_cases,
        'conversation_only_oracle_cases': conversation_only_oracle_cases,
        'missing_conversation_only_oracle_cases': missing_conversation_only_cases,
        'conversation_refs': {
            'total': len(conversation_refs),
            'found': len(existing_conversations),
            'missing': len(conversation_refs - existing_conversations),
        },
        'raw_values_included': False,
    }
    if incompatible_cases:
        raise EvalCasePackCompatibilityError(compatibility)
    return compatibility


def _load_resume_index(path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    validate_redacted_artifact(data)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, payload in (data.get('modes') or {}).items():
        out[str(mode)] = {str(case.get('case_hash')): case for case in (payload.get('cases') or []) if case.get('case_hash')}
    return out


def write_redacted_report_atomic(report: dict[str, Any], path: str | Path) -> None:
    validate_redacted_artifact(report)
    write_evidence_artifact(report, Path(path).expanduser())


def _refresh_partial_mode(report: dict[str, Any], mode: str, case_results: list[dict[str, Any]], notes: set[str], vector_states: Counter[str], *, k: int, implemented: bool) -> None:
    report['modes'][mode] = {
        'implemented': implemented,
        'notes': sorted(notes),
        'vector_states': dict(sorted(vector_states.items())),
        'metrics': _summarize_mode(case_results, k=k),
        'cases': case_results,
    }


def run_eval_matrix(
    vault: str | Path | None,
    cases_path: str | Path,
    *,
    modes: list[str] | None = None,
    k: int = 3,
    model_path: str | None = None,
    reranker_model_path: str | None = None,
    category_filters: list[str] | None = None,
    case_hash_filters: set[str] | None = None,
    max_cases: int | None = None,
    sample_seed: int = 0,
    fixture_id: str = 'synthetic_or_redacted',
    resume_path: str | Path | None = None,
    partial_out: str | Path | None = None,
    retrieval_candidate_limit: int = 200,
    fusion_candidate_limit: int = 200,
    reranker_candidate_limit: int = 50,
    cloud: bool = False,
) -> dict[str, Any]:
    retrieval_candidate_limit = int(BoundedLimit(
        retrieval_candidate_limit,
        field='retrieval_candidate_limit',
        spec=RETRIEVAL_CANDIDATES,
    ))
    fusion_candidate_limit = int(BoundedLimit(
        fusion_candidate_limit,
        field='fusion_candidate_limit',
        spec=FUSION_CANDIDATES,
    ))
    reranker_candidate_limit = int(BoundedLimit(
        reranker_candidate_limit,
        field='reranker_candidate_limit',
        spec=RERANK_CANDIDATES,
    ))
    cfg = VaultConfig.resolve(str(vault) if vault else None)
    store = open_store(cfg.paths.sqlite_path, readonly=True)
    case_pack_path = Path(cases_path)
    all_cases = load_case_pack(case_pack_path)
    cases = _select_cases(
        all_cases,
        category_filters=category_filters,
        case_hash_filters=case_hash_filters,
        max_cases=max_cases,
        sample_seed=sample_seed,
    )
    case_pack_compatibility = check_case_pack_compatibility(store, cases)
    selected_modes = modes or ['hybrid-weighted']
    for mode in selected_modes:
        if mode not in EVAL_MODES:
            raise ValueError(f'unsupported eval mode: {mode}')
    resume_index = _load_resume_index(Path(resume_path).expanduser() if resume_path else None)
    report: dict[str, Any] = {
        'schema_version': 2,
        'artifact_type': 'retrieval_eval_matrix_redacted',
        'created_at': now_iso(),
        'complete': False,
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
            'token_values_included': False,
        },
        'case_count_total_loaded': len(all_cases),
        'case_count': len(cases),
        'case_pack_anchor': {
            'sha256_prefix': hashlib.sha256(case_pack_path.expanduser().read_bytes()).hexdigest()[:32],
            'loaded_cases': len(all_cases),
            'selected_cases': len(cases),
            'path_included': False,
        },
        'case_quality': case_pack_quality_stats(cases),
        'case_pack_compatibility': case_pack_compatibility,
        'k': k,
        'controls': {
            'category_filters': sorted([c for c in (category_filters or []) if c]),
            'max_cases': max_cases,
            'sample_seed': sample_seed,
            'case_hash_filter_enabled': bool(case_hash_filters),
            'case_hash_filter_count': len(case_hash_filters or set()),
            'resume_enabled': bool(resume_index),
            'partial_write_enabled': bool(partial_out),
            'candidate_budgets': {
                'retrieval': retrieval_candidate_limit,
                'fusion': fusion_candidate_limit,
                'rerank': reranker_candidate_limit,
            },
            'reranker_model_configured': bool(reranker_model_path),
            'cloud_selected_vault': bool(cloud),
        },
        'modes': {},
        'comparisons': {},
    }
    needs_vector_context = any(mode in {'vector', 'hybrid-weighted', 'hybrid-rrf', 'feature-rerank', 'local-reranker', 'cloud-reranker'} for mode in selected_modes)
    vector_context = _build_vector_context(cfg, model_path, cloud=cloud) if needs_vector_context else None
    hybrid_search = None
    cloud_queries = None
    if needs_vector_context and vector_context is not None:
        provider, vector, status = vector_context
        episode_store = None
        selector = None
        if cloud and provider is not None:
            from trove_core.search.episodes import BoundedEvidenceSelector, EpisodeZVecStore, episode_collection_path

            candidate = EpisodeZVecStore(episode_collection_path(cfg.paths.vector_dir), store=store)
            if candidate.status(provider).get('state') == 'available':
                episode_store = candidate
                selector = BoundedEvidenceSelector(cfg.root)
            from trove_core.application.queries import TroveQueries

            cloud_queries = TroveQueries(cfg)
        hybrid_search = HyperSearch(
            store,
            vector_store=vector,
            embedding_provider=provider,
            vector_status=status,
            episode_store=episode_store,
            evidence_selector=selector,
        )
        warm_search_engine(hybrid_search)
    reranker_warmup = None
    if 'local-reranker' in selected_modes and reranker_model_path:
        reranker_warmup = warm_local_reranker(reranker_model_path)
    report['reranker_warmup'] = reranker_warmup
    report['provenance'] = build_artifact_provenance(
        repo_root=Path(__file__).resolve().parents[4],
        sqlite_path=cfg.paths.sqlite_path,
        case_pack_path=case_pack_path,
        seed=sample_seed,
        fixture_id=fixture_id,
        provider=vector_context[0] if vector_context is not None else None,
        temperature='warm' if hybrid_search is not None else 'cold',
        warmups=1 if hybrid_search is not None else 0,
        rounds=1,
    )
    mode_case_results: dict[str, list[dict[str, Any]]] = {}
    for mode in selected_modes:
        case_results: list[dict[str, Any]] = []
        notes = set()
        vector_states = Counter()
        resumed = 0
        for case in cases:
            case_hash = stable_hash(case.get('case_id'))
            existing = resume_index.get(mode, {}).get(str(case_hash))
            if existing:
                case_results.append(existing)
                resumed += 1
                vector_states[str(existing.get('vector_state') or 'unknown')] += 1
            else:
                mode_result = run_mode(
                    cfg,
                    store,
                    case,
                    mode,
                    k=k,
                    model_path=model_path,
                    reranker_model_path=reranker_model_path,
                    vector_context=vector_context,
                    hybrid_search=hybrid_search,
                    cloud_queries=cloud_queries,
                    retrieval_candidate_limit=retrieval_candidate_limit,
                    fusion_candidate_limit=fusion_candidate_limit,
                    reranker_candidate_limit=reranker_candidate_limit,
                )
                if mode_result.get('mode_note'):
                    notes.add(mode_result['mode_note'])
                vector_states[str((mode_result.get('vector_status') or {}).get('state') or 'unknown')] += 1
                case_results.append(_evaluate_case(store, case, mode_result, k=k))
            if partial_out and len(case_results) % 10 == 0:
                report['complete'] = False
                _refresh_partial_mode(report, mode, case_results, notes, vector_states, k=k, implemented=True)
                write_redacted_report_atomic(report, partial_out)
        mode_case_results[mode] = case_results
        if resumed:
            notes.add(f'resumed_cases:{resumed}')
        _refresh_partial_mode(report, mode, case_results, notes, vector_states, k=k, implemented=True)
        if partial_out:
            write_redacted_report_atomic(report, partial_out)
    if 'vector' in mode_case_results and 'exact' in mode_case_results:
        by_mode = {mode: {c['case_hash']: c for c in rows} for mode, rows in mode_case_results.items()}
        vector_only = [case_hash for case_hash, c in by_mode['vector'].items() if c.get('hit') and not by_mode['exact'].get(case_hash, {}).get('hit')]
        exact_only = [case_hash for case_hash, c in by_mode['exact'].items() if c.get('hit') and not by_mode['vector'].get(case_hash, {}).get('hit')]
        report['comparisons']['vector_vs_exact'] = {
            'vector_only_recovered': len(vector_only),
            'exact_only_recovered': len(exact_only),
            'vector_only_case_hashes': vector_only[:50],
            'exact_only_case_hashes': exact_only[:50],
        }
    if cloud and vector_context is not None and vector_context[0] is not None:
        cloud_provider = vector_context[0]
        input_tokens = int(getattr(cloud_provider, 'input_tokens', 0) or 0)
        from trove_core.search.episodes import EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION

        report['cloud_embedding_usage'] = {
            'provider_calls': int(getattr(cloud_provider, 'provider_calls', 0) or 0),
            'input_tokens': input_tokens,
            'estimated_cost_rmb': round(input_tokens * EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION / 1_000_000, 6),
        }
    report['complete'] = True
    validate_redacted_artifact(report)
    return report
