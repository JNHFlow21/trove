from __future__ import annotations
from dataclasses import dataclass, replace
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import threading
import time
from typing import Any

from trove_core.store.sqlite_store import SQLiteStore, vector_document_text
from .cloud_reranker import rerank_with_cloud_model
from .evidence import row_to_evidence
from .fusion import fuse_ranked_rows, fuse_ranked_rows_rrf, route_counts
from .local_reranker import rerank_with_local_model
from .query_understanding import MULTI_HOP_MARKERS, analyze_query
from .query import SearchRequest, SearchResponse
from .rerank import rerank_with_features

EXACT_ROUTE_WEIGHT = 2.0
EVIDENCE_ROUTE_WEIGHT = 10.0
# A vector-only hit is additive evidence, not a stronger claim than an exact
# lexical match. Cross-route agreement still accumulates the vector score.
VECTOR_ROUTE_WEIGHT = 1.0
FTS_ROUTE_WEIGHT = 4.0
CONVERSATION_CONTEXT_ROUTE_WEIGHT = 8.0
EPISODE_EVIDENCE_ROUTE_WEIGHT = 12.0
# The fixed-dev Episode collection has complete evidence in top-10 even when
# rank-only dense+sparse fusion leaves one complete bundle below top-3. Keep
# retrieval broad enough for the bounded default reranker, which ranks complete
# episode bundles without a separate online selector call.
EPISODE_RUNTIME_CANDIDATE_LIMIT = 10


class _QueryEmbeddingCacheProvider:
    """Search-only provider wrapper that caches one query embedding per text."""

    def __init__(self, owner: 'HyperSearch', provider: object):
        self._owner = owner
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def embed_query(self, text: str):
        return self._owner._cached_query_embedding(self._provider, text, prefer_query=True)

    def embed(self, text: str):
        return self._owner._cached_query_embedding(self._provider, text, prefer_query=False)

    def embed_many(self, texts):
        return self._provider.embed_many(texts)  # type: ignore[attr-defined]

    def embed_hybrid_many(self, texts, *, text_type='document', instruct=None):
        values = list(texts)
        if text_type != 'query':
            return self._provider.embed_hybrid_many(  # type: ignore[attr-defined]
                values, text_type=text_type, instruct=instruct
            )
        return [
            self._owner._cached_query_hybrid_embedding(
                self._provider,
                text,
                method_name='embed_hybrid_many',
                text_type=text_type,
                instruct=instruct,
            )
            for text in values
        ]

    def embed_query_hybrid(self, text: str):
        return self._owner._cached_query_hybrid_embedding(
            self._provider,
            text,
            method_name='embed_query_hybrid',
            text_type='query',
            instruct=None,
        )


@dataclass
class HyperSearch:
    store: SQLiteStore
    vector_store: object | None = None
    embedding_provider: object | None = None
    vector_status: dict | None = None
    episode_store: object | None = None
    evidence_selector: object | None = None
    query_embedding_cache_max: int = 128

    def __post_init__(self) -> None:
        # Dense and hybrid query vectors share one hard LRU bound.  Sparse
        # vectors can be materially larger than dense vectors, so separate
        # caches would silently double resident memory.
        self._query_embedding_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._query_embedding_inflight: dict[tuple[Any, ...], Future] = {}
        self._query_embedding_cache_lock = threading.RLock()
        self._query_embedding_cache_hits = 0
        self._query_embedding_cache_misses = 0
        self._query_embedding_singleflight_followers = 0

    def search(self, request: SearchRequest) -> SearchResponse:
        start = time.perf_counter()
        retrieval_start = start
        understanding = analyze_query(request.query, enabled=request.expand_query)
        multi_hop_requested = self._multi_hop_requested(understanding.normalized)
        multi_hop_expansion = bool(
            multi_hop_requested
            and not any(
                request.filters.get(key)
                for key in ('sender', 'since', 'until', 'source_type', 'source_family', 'scope_type')
            )
        )
        episode_future: Future | None = None
        if multi_hop_expansion and self.episode_store is not None and self.embedding_provider is not None:
            episode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='trove-episode-query')
            episode_future = episode_executor.submit(self._run_episode_route, request)
            episode_future.add_done_callback(
                lambda _completed, executor=episode_executor: executor.shutdown(wait=False)
            )
        route_queries = understanding.expanded_queries if request.expand_query else [request.query]
        route_limit = min(request.retrieval_candidate_limit, max(request.limit * 3, 10))
        multi_term_query = len(understanding.terms) >= 2 or len([p for p in request.query.split() if len(p) >= 2]) >= 2
        semantic_mode = 'off' if not request.include_vector else request.semantic
        structured_non_message_auto = bool(
            semantic_mode == 'auto'
            and any(
                request.filters.get(key) not in {None, '', 'all', 'message'}
                for key in ('source_type', 'source_family', 'scope_type')
            )
        )
        base_vector = dict(self.vector_status or {})
        zvec_status = base_vector.get('zvec') or {}
        vector_catchup_pending = bool(
            base_vector.get('catchup_pending')
            or base_vector.get('reason_code') == 'zvec_catchup_pending'
            or (isinstance(zvec_status, dict) and zvec_status.get('catchup_pending'))
        )
        vector_configured = bool(
            request.include_vector
            and self.vector_store
            and self.embedding_provider
            and base_vector.get('state') == 'available'
        )
        force_semantic = self._force_semantic_auto(request, understanding, vector_configured=vector_configured)
        # Auto semantic retrieval is additive. A calibrated score floor proves
        # score-domain separation on dev evidence; it does not prove that an
        # arbitrary non-empty vector result is better than lexical evidence.
        # Replacing lexical routes here caused false semantic positives to hide
        # exact/FTS candidates. Explicit route budgets keep the additive plan
        # bounded without turning vector availability into a recall regression.
        semantic_first = False
        # A catch-up generation is queryable but does not cover the full SQLite
        # corpus yet.  Keep its semantic candidates additive until the ledger
        # declares the generation complete, otherwise one partial vector hit
        # can hide the correct not-yet-indexed lexical result.
        if vector_catchup_pending:
            semantic_first = False
        rewrite_semantic_priority = semantic_mode == 'auto' and force_semantic
        fts_route_queries = [request.query] if rewrite_semantic_priority else route_queries[:3]
        # Deep retrieval is isolated to semantic rewrites.  Exact/lexical
        # requests keep their old shallow derived depth, while rewrite routes
        # can explicitly reach top-200 without coupling that work to fusion or
        # reranking windows.
        vector_route_limit = (
            request.retrieval_candidate_limit
            if rewrite_semantic_priority
            else min(request.retrieval_candidate_limit, max(request.limit * 2, 10))
        )
        vector_rows = []
        vector_status = {
            'enabled': False,
            'semantic_mode': semantic_mode,
            'semantic_first': bool(semantic_first),
            'semantic_priority': bool(rewrite_semantic_priority),
            'attempted': False,
            'available': False,
            'state': base_vector.get('state', 'unavailable_fallback'),
            'selected_backend': base_vector.get('selected_backend', 'none'),
            'reason_code': base_vector.get('reason_code'),
            'reason': base_vector.get('message') or base_vector.get('reason_code') or 'no vector store configured',
        }
        semantic_required = semantic_mode == 'on' or rewrite_semantic_priority
        run_vector = semantic_required
        vector_fallback_to_lexical = False
        if rewrite_semantic_priority:
            vector_rows, vector_status = self._run_vector_route(request, vector_route_limit, vector_status)
            # Semantic-first is an optimization, never an availability boundary.
            # A missing/degraded vector backend (or a valid empty vector result)
            # must fall back to the bounded lexical routes instead of returning
            # a false zero-result response.
            if not vector_rows or vector_status.get('state') != 'available':
                semantic_first = False
                vector_fallback_to_lexical = True
        if semantic_first:
            exact_rows = []
            evidence_rows = []
            exact_chunk_candidates = 0
            vector_candidates_sufficient = len(vector_rows) >= request.limit
            lexical_route_limit = 0
        else:
            vector_candidates_sufficient = rewrite_semantic_priority and len(vector_rows) >= request.limit
            lexical_route_limit = 10 if rewrite_semantic_priority else route_limit
            exact_rows = self._collect(
                lambda q, lim: self.store.exact_search(
                    q,
                    filters=request.filters,
                    limit=lim,
                    allow_like_fallback=True,
                    sender_prefilter=rewrite_semantic_priority,
                ),
                [request.query],
                lexical_route_limit,
            )
            evidence_rows = self._collect(
                lambda q, lim: self.store.chunk_search(
                    q,
                    filters=request.filters,
                    limit=lim,
                    allow_like_fallback=True,
                    actor_prefilter=rewrite_semantic_priority,
                ),
                [request.query],
                lexical_route_limit,
            )
            exact_chunk_candidates = self._unique_count(exact_rows, evidence_rows)
        if not semantic_required:
            semantic_required = bool(
                semantic_mode == 'on'
                or (
                    semantic_mode == 'auto'
                    and not structured_non_message_auto
                    and exact_chunk_candidates < request.limit
                )
            )
            run_vector = semantic_required
        if run_vector and not rewrite_semantic_priority:
            vector_rows, vector_status = self._run_vector_route(request, vector_route_limit, vector_status)
            vector_candidates_sufficient = len(vector_rows) >= request.limit
        skip_fts_expansion = (
            semantic_first
            or (exact_chunk_candidates >= request.limit and not semantic_required and not multi_term_query)
        )
        fts_route_limit = (
            min(request.retrieval_candidate_limit, 6, max(request.limit * 2, request.limit))
            if rewrite_semantic_priority
            else route_limit
        )
        fts_rows = [] if skip_fts_expansion else self._collect(
            lambda q, lim: self.store.fts_search_filtered(q, filters=request.filters, limit=lim, allow_like_fallback=False),
            fts_route_queries,
            fts_route_limit,
        )
        conversation_context_rows = self._conversation_context_rows(
            exact_rows,
            evidence_rows,
            vector_rows,
            fts_rows,
            max_anchors=10,
        ) if multi_hop_expansion else []
        episode_rows: list[Any] = []
        episode_status: dict[str, Any] = {
            'state': 'skipped',
            'reason_code': 'not_multi_hop' if not multi_hop_expansion else 'episode_collection_unavailable',
            'episode_count': 0,
            'candidate_count': 0,
            'selected_chain_count': 0,
        }
        selected_episode_citations: tuple[str, ...] = ()
        episode_bundles: tuple[Any, ...] = ()
        if episode_future is not None:
            try:
                episode_rows, episode_status, selected_episode_citations, episode_bundles = episode_future.result()
            except Exception as exc:
                episode_status = {
                    'state': 'degraded',
                    'reason_code': str(exc.args[0] if exc.args else exc.__class__.__name__),
                    'episode_count': 0,
                    'candidate_count': 0,
                    'selected_chain_count': 0,
                    'fallback_mode': 'conversation_context',
                    'parallel_execution': True,
                }
        lexical_first_stage_candidates = self._unique_count(exact_rows, evidence_rows, fts_rows)
        if not run_vector:
            vector_status['enabled'] = False
        elif not vector_status.get('attempted'):
            vector_status['enabled'] = True
        if not run_vector and semantic_mode == 'auto':
            vector_status.update({
                'available': base_vector.get('state') == 'available',
                'state': base_vector.get('state', 'unavailable_fallback'),
                'reason': (
                    'semantic auto skipped for a structured non-message source'
                    if structured_non_message_auto
                    else 'semantic auto skipped because lexical candidates satisfied the request'
                ),
                'reason_code': (
                    'semantic_auto_structured_source'
                    if structured_non_message_auto
                    else 'semantic_auto_satisfied'
                ),
            })
        elif run_vector and not vector_status.get('attempted'):
            vector_status.update({
                'available': False,
                'state': 'unavailable_fallback',
                'reason': base_vector.get('message') or base_vector.get('reason_code') or 'no vector store configured',
            })
        vector_status['cache'] = self._query_embedding_cache_status()
        vector_status['candidate_count'] = len(vector_rows)
        vector_status['candidate_sufficient'] = bool(vector_candidates_sufficient)
        groups = []
        if exact_rows or not semantic_first:
            groups.append(('exact', exact_rows, EXACT_ROUTE_WEIGHT))
        if evidence_rows or not semantic_first:
            groups.append(('evidence', evidence_rows, EVIDENCE_ROUTE_WEIGHT))
        if vector_rows:
            groups.append(('vector', vector_rows, VECTOR_ROUTE_WEIGHT))
        if not skip_fts_expansion:
            groups.append(('fts', fts_rows, FTS_ROUTE_WEIGHT))
        if conversation_context_rows:
            groups.append(('conversation-context', conversation_context_rows, CONVERSATION_CONTEXT_ROUTE_WEIGHT))
        if episode_rows:
            groups.append(('episode-evidence', episode_rows, EPISODE_EVIDENCE_ROUTE_WEIGHT))
        first_stage_routes = []
        if not semantic_first:
            first_stage_routes.extend(['exact', 'evidence'])
        if run_vector:
            first_stage_routes.append('vector')
        if not skip_fts_expansion:
            first_stage_routes.append('fts')
        if conversation_context_rows:
            first_stage_routes.append('conversation-context')
        if episode_rows:
            first_stage_routes.append('episode-evidence')
        first_stage_candidates = self._unique_count(exact_rows, evidence_rows, vector_rows, fts_rows, conversation_context_rows, episode_rows)
        candidate_citations = self._candidate_citations(exact_rows, evidence_rows, vector_rows, fts_rows, conversation_context_rows, episode_rows)
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000
        fusion_start = time.perf_counter()
        candidate_limit = request.fusion_candidate_limit
        ranking_mode = request.effective_ranking_mode
        if request.ranking_mode == 'rrf' or ranking_mode == 'feature':
            fused = fuse_ranked_rows_rrf(groups, candidate_limit)
            base_ranker = 'rrf'
        else:
            fused = fuse_ranked_rows(groups, candidate_limit)
            base_ranker = 'weighted'
        fusion_candidate_count = len(fused)
        fusion_elapsed_ms = (time.perf_counter() - fusion_start) * 1000
        rerank_start = time.perf_counter()
        reranker_status = {'state': 'off', 'mode': request.reranker_mode, 'candidate_count': 0}
        reranker_window_count = 0
        source_filtered_rerank = any(
            request.filters.get(key) not in {None, '', 'all', 'message'}
            for key in ('source_type', 'source_family', 'scope_type')
        )
        if ranking_mode == 'feature':
            reranker_window_count = min(len(fused), request.reranker_candidate_limit)
            reranked_head, reranker_status = rerank_with_features(
                fused[:reranker_window_count],
                request.query,
                filters=request.filters,
                limit=max(reranker_window_count, 1),
                understanding=understanding,
            )
            fused = reranked_head + fused[reranker_window_count:]
            semantic_rerank_eligible = bool(
                run_vector and (semantic_mode == 'on' or rewrite_semantic_priority)
            )
            if request.reranker_mode == 'local-bge':
                if semantic_rerank_eligible:
                    reranked_head, reranker_status = rerank_with_local_model(
                        fused[:reranker_window_count],
                        request.query,
                        model_path=request.reranker_model_path,
                        timeout_ms=request.reranker_timeout_ms,
                        limit=max(reranker_window_count, 1),
                    )
                    fused = reranked_head + fused[reranker_window_count:]
                else:
                    reranker_status = {
                        'state': 'skipped',
                        'mode': 'local-bge',
                        'reason_code': 'local_reranker_requires_semantic_route',
                        'fallback_mode': 'features',
                        'candidate_count': 0,
                        'invoked': False,
                    }
            elif request.reranker_mode == 'cloud-qwen3':
                if semantic_rerank_eligible:
                    reranked_head, reranker_status = rerank_with_cloud_model(
                        fused[:reranker_window_count],
                        request.query,
                        limit=max(reranker_window_count, 1),
                        candidate_limit=max(reranker_window_count, 1),
                        timeout_ms=request.reranker_timeout_ms,
                    )
                    fused = reranked_head + fused[reranker_window_count:]
                else:
                    # Preserve the U2 approval contract without invoking a
                    # reranker on the exact path.
                    reranker_status = {
                        'state': 'unavailable_fallback',
                        'mode': 'cloud-qwen3',
                        'reason_code': 'cloud_reranker_requires_exact_approval',
                        'fallback_mode': 'features',
                        'candidate_count': 0,
                        'invoked': False,
                    }
        elif multi_term_query or source_filtered_rerank:
            reranker_window_count = min(len(fused), request.reranker_candidate_limit)
            reranked_head, feature_status = rerank_with_features(
                fused[:reranker_window_count],
                request.query,
                filters=request.filters,
                limit=max(reranker_window_count, 1),
                understanding=understanding,
            )
            fused = reranked_head + fused[reranker_window_count:]
            reranker_status = {
                **feature_status,
                'state': 'available',
                'mode': 'bounded_source_features' if source_filtered_rerank and not multi_term_query else 'bounded_multiword_features',
            }
        rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000
        multi_hop_context_promoted = False
        if multi_hop_expansion and fused:
            fused, multi_hop_context_promoted = self._promote_multi_hop_context(
                fused,
                selected_episode_citations=selected_episode_citations,
            )
        fused = fused[:request.limit]
        hint_citations: list[str] = []
        if request.include_media_hints:
            for row, _, _ in fused:
                hint_citations.append(str(row['citation']))
                if hasattr(row, 'keys') and 'parent_citation' in row.keys():
                    hint_citations.append(str(row['parent_citation']))
        media_hints = self.store.media_hints_for_citations(hint_citations) if hint_citations and hasattr(self.store, 'media_hints_for_citations') else {}
        evidences = []
        for row, paths, score in fused:
            row_citation = str(row['citation'])
            parent_citation = str(row['parent_citation']) if hasattr(row, 'keys') and 'parent_citation' in row.keys() else row_citation
            hint = (media_hints.get(row_citation) or media_hints.get(parent_citation)) if request.include_media_hints else None
            evidences.append(row_to_evidence(row, request.query, paths, score, media_hint=hint))
        elapsed = (time.perf_counter() - start) * 1000
        return SearchResponse(
            query=request.query,
            results=evidences,
            total=len(evidences),
            retrieval_status={
                'exact': True,
                'fts': True,
                'multisource_evidence': True,
                'metadata_filters': request.filters,
                'vector': vector_status,
                'retrieval_plan': {
                    'route_policy': 'semantic_first_rewrite_auto' if semantic_first else 'trigram_evidence_first_vector_additive',
                    'first_stage_routes': first_stage_routes,
                    'first_stage_candidates': first_stage_candidates,
                    'lexical_first_stage_candidates': lexical_first_stage_candidates,
                    'exact_chunk_candidates': exact_chunk_candidates,
                    'exact_route_executed': not semantic_first,
                    'fts_route_executed': not skip_fts_expansion,
                    'fts_route_skipped_reason': 'semantic_first_rewrite' if semantic_first else (
                        'vector_candidates_satisfied'
                        if skip_fts_expansion and vector_candidates_sufficient and rewrite_semantic_priority
                        else ('exact_chunk_limit_satisfied' if skip_fts_expansion else None)
                    ),
                    'lexical_route_limit': lexical_route_limit,
                    'fts_route_query_count': len(fts_route_queries) if not skip_fts_expansion else 0,
                    'fts_route_limit': fts_route_limit if not skip_fts_expansion else 0,
                    'semantic_required': semantic_required,
                    'semantic_forced': bool(force_semantic),
                    'semantic_first': bool(semantic_first),
                    'semantic_priority': bool(rewrite_semantic_priority),
                    'vector_catchup_pending': vector_catchup_pending,
                    'vector_catchup_lexical_merge': bool(vector_catchup_pending and run_vector),
                    'vector_fallback_to_lexical': vector_fallback_to_lexical,
                    'vector_candidates_sufficient': bool(vector_candidates_sufficient),
                    'vector_route_limit': vector_route_limit if run_vector else 0,
                    'vector_replacement_enabled': False,
                    'multi_hop_expansion': bool(conversation_context_rows),
                    'multi_hop_context_candidates': len(conversation_context_rows),
                    'multi_hop_context_promoted': multi_hop_context_promoted,
                    'multi_hop_episode': episode_status,
                },
                'query_understanding': understanding.to_status(),
                'ranking': {
                    'requested_mode': request.ranking_mode,
                    'mode': ranking_mode,
                    'base_ranker': base_ranker,
                    'candidate_routes': route_counts(groups),
                    'candidate_count': sum(len(rows) for _, rows, _ in groups),
                    'fusion_candidate_count': fusion_candidate_count,
                },
                'candidate_budgets': {
                    'retrieval': {
                        'requested_limit': int(request.retrieval_candidate_limit),
                        'scope': 'per_route',
                        'max_route_candidates': max((len(rows) for _, rows, _ in groups), default=0),
                        'unique_candidates': first_stage_candidates,
                    },
                    'fusion': {
                        'requested_limit': int(request.fusion_candidate_limit),
                        'input_candidates': first_stage_candidates,
                        'output_candidates': fusion_candidate_count,
                    },
                    'rerank': {
                        'requested_limit': int(request.reranker_candidate_limit),
                        'input_candidates': reranker_window_count,
                        'model_candidates': int(reranker_status.get('candidate_count') or 0),
                    },
                },
                'reranker': reranker_status,
                'phase_latency_ms': {
                    'retrieval': round(retrieval_elapsed_ms, 3),
                    'fusion': round(fusion_elapsed_ms, 3),
                    'rerank': round(rerank_elapsed_ms, 3),
                },
            },
            elapsed_ms=round(elapsed, 3),
            candidate_citations=candidate_citations,
            episode_bundles=episode_bundles,
        )

    def warm_query_path(self, sample_text: str = 'trove search warmup') -> dict[str, Any]:
        """Preheat the vector query path outside user-visible query latency.

        This is intentionally read-only: it only opens the vector collection and
        asks the configured embedding provider for one synthetic query vector.
        The synthetic text is static and non-private, so no Vault content leaves
        the process during warmup.
        """
        base_vector = dict(self.vector_status or {})
        if not (self.vector_store and self.embedding_provider and base_vector.get('state') == 'available'):
            return {
                'ok': False,
                'state': base_vector.get('state', 'unavailable_fallback'),
                'reason_code': base_vector.get('reason_code') or 'vector_unavailable',
                'private_text_used': False,
            }
        status = {
            'enabled': True,
            'semantic_mode': 'on',
            'semantic_first': True,
            'semantic_priority': True,
            'attempted': False,
            'available': False,
            'state': base_vector.get('state', 'available'),
            'selected_backend': base_vector.get('selected_backend', 'none'),
            'reason_code': base_vector.get('reason_code'),
            'reason': None,
        }
        try:
            rows, status = self._run_vector_route(
                SearchRequest(sample_text, limit=1, semantic='on', include_vector=True, expand_query=False),
                1,
                status,
            )
        except Exception as exc:  # defensive; warmup must never break search startup
            return {
                'ok': False,
                'state': getattr(exc, 'vector_state', 'degraded'),
                'reason_code': getattr(exc, 'reason_code', exc.__class__.__name__),
                'private_text_used': False,
            }
        return {
            'ok': status.get('state') == 'available',
            'state': status.get('state'),
            'candidate_count': len(rows),
            'cache': status.get('cache') or self._query_embedding_cache_status(),
            'private_text_used': False,
        }

    def _run_vector_route(self, request: SearchRequest, vector_route_limit: int, vector_status: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        if self.vector_store and self.embedding_provider:
            try:
                vector_status['attempted'] = True
                cached_provider = _QueryEmbeddingCacheProvider(self, self.embedding_provider)
                search_method = self.vector_store.search  # type: ignore[attr-defined]
                try:
                    parameters = inspect.signature(search_method).parameters
                    accepts_provider = 'provider' in parameters or any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                except (TypeError, ValueError):
                    accepts_provider = True
                if accepts_provider:
                    vector_rows = self.vector_store.search(request.query, filters=request.filters, limit=vector_route_limit, provider=cached_provider)  # type: ignore[attr-defined]
                else:
                    vector_rows = self.vector_store.search(request.query, filters=request.filters, limit=vector_route_limit)  # type: ignore[attr-defined]
                vector_status.update({'enabled': True, 'available': True, 'state': 'available', 'reason': None, 'reason_code': None})
                if hasattr(self.vector_store, 'last_search_status'):
                    filter_status = self.vector_store.last_search_status()  # type: ignore[attr-defined]
                    if isinstance(filter_status, dict) and filter_status:
                        vector_status['filter_plan'] = filter_status
                return vector_rows, vector_status
            except Exception as exc:
                vector_status.update({
                    'enabled': True,
                    'available': False,
                    'state': getattr(exc, 'vector_state', 'degraded'),
                    'reason': str(exc),
                    'reason_code': getattr(exc, 'reason_code', exc.__class__.__name__),
                })
                return [], vector_status
        vector_status.update({
            'enabled': True,
            'available': False,
            'state': 'unavailable_fallback',
            'reason': (self.vector_status or {}).get('message') or (self.vector_status or {}).get('reason_code') or 'no vector store configured',
        })
        return [], vector_status

    def _provider_cache_identity(self, provider: object) -> tuple[Any, ...]:
        return (
            provider.__class__.__module__,
            provider.__class__.__qualname__,
            getattr(provider, 'name', None),
            getattr(provider, 'provider_name', None),
            getattr(provider, 'model_id', None),
            getattr(provider, 'model', None),
            getattr(provider, 'dimensions', None),
        )

    def _cached_query_embedding(self, provider: object, text: str, *, prefer_query: bool) -> list[float]:
        method_name = 'embed_query' if prefer_query and hasattr(provider, 'embed_query') else 'embed'
        key = (self._provider_cache_identity(provider), method_name, text)
        leader = False
        with self._query_embedding_cache_lock:
            cached = self._query_embedding_cache.get(key)
            if cached is not None:
                self._query_embedding_cache_hits += 1
                self._query_embedding_cache.move_to_end(key)
                return list(cached)
            in_flight = self._query_embedding_inflight.get(key)
            if in_flight is None:
                in_flight = Future()
                self._query_embedding_inflight[key] = in_flight
                self._query_embedding_cache_misses += 1
                leader = True
            else:
                self._query_embedding_singleflight_followers += 1
        if not leader:
            return list(in_flight.result())
        try:
            method = getattr(provider, method_name)
            vector = tuple(method(text))
        except BaseException as exc:
            with self._query_embedding_cache_lock:
                current = self._query_embedding_inflight.get(key)
                if current is in_flight:
                    in_flight.set_exception(exc)
                    self._query_embedding_inflight.pop(key, None)
            raise
        with self._query_embedding_cache_lock:
            self._query_embedding_cache[key] = vector
            self._query_embedding_cache.move_to_end(key)
            while len(self._query_embedding_cache) > self.query_embedding_cache_max:
                self._query_embedding_cache.popitem(last=False)
            current = self._query_embedding_inflight.get(key)
            if current is in_flight:
                in_flight.set_result(vector)
                self._query_embedding_inflight.pop(key, None)
        return list(vector)

    def _cached_query_hybrid_embedding(
        self,
        provider: object,
        text: str,
        *,
        method_name: str,
        text_type: str,
        instruct: str | None,
    ):
        from trove_core.embedding.base import HybridEmbedding

        key = (
            self._provider_cache_identity(provider),
            method_name,
            text_type,
            instruct,
            text,
        )
        leader = False
        with self._query_embedding_cache_lock:
            cached = self._query_embedding_cache.get(key)
            if cached is not None:
                self._query_embedding_cache_hits += 1
                self._query_embedding_cache.move_to_end(key)
                dense, sparse_items = cached
                return HybridEmbedding(dense=list(dense), sparse=dict(sparse_items))
            in_flight = self._query_embedding_inflight.get(key)
            if in_flight is None:
                in_flight = Future()
                self._query_embedding_inflight[key] = in_flight
                self._query_embedding_cache_misses += 1
                leader = True
            else:
                self._query_embedding_singleflight_followers += 1

        if not leader:
            dense, sparse_items = in_flight.result()
            return HybridEmbedding(dense=list(dense), sparse=dict(sparse_items))

        try:
            if method_name == 'embed_query_hybrid':
                method = getattr(provider, 'embed_query_hybrid', None)
                if callable(method):
                    result = method(text)
                else:
                    result = HybridEmbedding(
                        dense=self._cached_query_embedding(provider, text, prefer_query=True),
                        sparse={},
                    )
            else:
                method = getattr(provider, 'embed_hybrid_many', None)
                if callable(method):
                    values = method([text], text_type=text_type, instruct=instruct)
                    if not values:
                        raise RuntimeError('embedding_provider_returned_no_query_vectors')
                    result = values[0]
                else:
                    result = HybridEmbedding(
                        dense=self._cached_query_embedding(provider, text, prefer_query=True),
                        sparse={},
                    )
            cached_value = (
                tuple(result.dense),
                tuple(sorted(dict(result.sparse).items())),
            )
        except BaseException as exc:
            with self._query_embedding_cache_lock:
                current = self._query_embedding_inflight.get(key)
                if current is in_flight:
                    in_flight.set_exception(exc)
                    self._query_embedding_inflight.pop(key, None)
            raise
        with self._query_embedding_cache_lock:
            self._query_embedding_cache[key] = cached_value
            self._query_embedding_cache.move_to_end(key)
            while len(self._query_embedding_cache) > self.query_embedding_cache_max:
                self._query_embedding_cache.popitem(last=False)
            current = self._query_embedding_inflight.get(key)
            if current is in_flight:
                in_flight.set_result(cached_value)
                self._query_embedding_inflight.pop(key, None)
        return HybridEmbedding(dense=list(cached_value[0]), sparse=dict(cached_value[1]))

    def _query_embedding_cache_status(self) -> dict[str, int]:
        with self._query_embedding_cache_lock:
            return {
                'entries': len(self._query_embedding_cache),
                'max': self.query_embedding_cache_max,
                'hits': self._query_embedding_cache_hits,
                'misses': self._query_embedding_cache_misses,
                'inflight': len(self._query_embedding_inflight),
                'singleflight_followers': self._query_embedding_singleflight_followers,
            }

    def _semantic_first_auto(self, request: SearchRequest, understanding, *, vector_configured: bool) -> bool:
        if not self._force_semantic_auto(request, understanding, vector_configured=vector_configured):
            return False
        filters = request.filters
        # Filtered rewrites need lexical/evidence participation for path-aware
        # frozen oracles; make semantic additive instead of replacing routes.
        if any(filters.get(key) for key in ('sender', 'since', 'until')):
            return False
        return True

    @staticmethod
    def _multi_hop_requested(normalized_query: str) -> bool:
        lowered = str(normalized_query or '').lower()
        return any(marker in lowered for marker in MULTI_HOP_MARKERS)

    def _run_episode_route(
        self,
        request: SearchRequest,
    ) -> tuple[list[Any], dict[str, Any], tuple[str, ...], tuple[Any, ...]]:
        """Run cloud Episode retrieval independently from message retrieval."""

        started = time.perf_counter()
        try:
            hits = self.episode_store.search(  # type: ignore[attr-defined]
                request.query,
                provider=_QueryEmbeddingCacheProvider(self, self.embedding_provider),
                filters=request.filters,
                limit=EPISODE_RUNTIME_CANDIDATE_LIMIT,
            )
            ordered_citations: list[str] = []
            seen: set[str] = set()
            for hit in hits:
                for citation in getattr(hit, 'citations', ()):
                    citation = str(citation or '')
                    if not citation:
                        continue
                    if citation not in seen:
                        seen.add(citation)
                        ordered_citations.append(citation)
            by_citation = self.store.evidence_by_citations(ordered_citations)
            found_citations = [citation for citation in ordered_citations if citation in by_citation]
            episode_rows = [by_citation[citation] for citation in found_citations]
            bundles = []
            for hit in hits:
                citations = tuple(
                    str(citation) for citation in getattr(hit, 'citations', ())
                    if str(citation or '') in by_citation
                )
                rows = [by_citation[citation] for citation in citations]
                if not rows:
                    continue
                representative = rows[len(rows) // 2]
                evidence = row_to_evidence(
                    representative,
                    request.query,
                    ['episode-bundle', 'episode-evidence', 'evidence'],
                    float(getattr(hit, 'score', 0.0) or 0.0),
                )
                bundles.append(replace(
                    evidence,
                    snippet=self._episode_bundle_snippet(rows),
                    supporting_citations=citations,
                    evidence_kind='episode',
                    _rerank_text=self._episode_bundle_rerank_text(rows),
                ))
            selector_status = {
                'state': 'skipped',
                'reason_code': 'episode_bundle_rerank_deferred',
                'invoked': False,
            }
            return episode_rows, {
                'state': 'available' if hits else 'unavailable_fallback',
                'reason_code': None if hits else 'episode_candidates_missing',
                'episode_count': len(hits),
                'candidate_count': len(episode_rows),
                'bundle_count': len(bundles),
                'selected_chain_count': 0,
                'selector': selector_status,
                'parallel_execution': True,
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 3),
            }, (), tuple(bundles)
        except Exception as exc:
            return [], {
                'state': 'degraded',
                'reason_code': str(exc.args[0] if exc.args else exc.__class__.__name__),
                'episode_count': 0,
                'candidate_count': 0,
                'selected_chain_count': 0,
                'fallback_mode': 'conversation_context',
                'parallel_execution': True,
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 3),
            }, (), ()

    @staticmethod
    def _episode_bundle_snippet(rows: list[Any]) -> str:
        """Build one bounded chronological snippet for cloud rank and display."""

        lines: list[str] = []
        for row in rows:
            parts = []
            for key in ('timestamp', 'sender_name'):
                try:
                    value = str(row[key] or '')
                except Exception:
                    value = ''
                if value:
                    parts.append(value)
            try:
                content = str(row['content'] or '')[:220]
            except Exception:
                content = ''
            if content:
                parts.append(content)
            if parts:
                lines.append(' | '.join(parts))
        return ('会话证据片段:\n' + '\n'.join(lines))[:2400]

    @staticmethod
    def _episode_bundle_rerank_text(rows: list[Any]) -> str:
        """Match the validated dev pilot while keeping a hard 7x800 bound."""

        return '会话证据片段:\n' + '\n'.join(
            vector_document_text(row)[:800] for row in rows
        )

    @staticmethod
    def _row_citation(row: Any) -> str:
        try:
            return str(row['citation'] or '')
        except Exception:
            return ''

    def _conversation_context_rows(self, *row_groups, max_anchors: int) -> list[Any]:
        """Expand at most two top anchors by one message on either side."""

        anchors: list[str] = []
        seen_anchors: set[str] = set()
        for rows in row_groups:
            for row in rows:
                try:
                    citation = str(row['parent_citation'] or '') if 'parent_citation' in row.keys() else ''
                except Exception:
                    citation = ''
                if not citation:
                    try:
                        citation = str(row['citation'] or '')
                    except Exception:
                        citation = ''
                if citation and citation not in seen_anchors:
                    anchors.append(citation)
                    seen_anchors.add(citation)
                if len(anchors) >= max_anchors:
                    break
            if len(anchors) >= max_anchors:
                break
        expanded: list[Any] = []
        seen_rows: set[str] = set()
        for anchor in anchors:
            try:
                rows = self.store.context_window(anchor, before=3, after=3)
            except Exception:
                rows = []
            for row in rows:
                try:
                    citation = str(row['citation'] or '')
                except Exception:
                    continue
                if citation and citation not in seen_rows:
                    seen_rows.add(citation)
                    expanded.append(row)
        return expanded

    def _promote_multi_hop_context(
        self,
        fused: list[tuple[Any, list[str], float]],
        *,
        selected_episode_citations: tuple[str, ...] = (),
    ) -> tuple[list[tuple[Any, list[str], float]], bool]:
        """Keep a selected Episode chain intact, else promote local context."""

        if not fused:
            return fused, False
        if selected_episode_citations:
            selected_items = []
            selected_item_ids: set[int] = set()
            for citation in selected_episode_citations:
                for item in fused:
                    row, paths, _score = item
                    if id(item) in selected_item_ids or 'episode-evidence' not in paths:
                        continue
                    if self._row_citation(row) == citation:
                        selected_items.append(item)
                        selected_item_ids.add(id(item))
                        break
            if len(selected_items) >= 2:
                return [
                    *selected_items,
                    *(item for item in fused if id(item) not in selected_item_ids),
                ], True
        top_index = 0
        for idx, (_row, paths, _score) in enumerate(fused):
            if 'episode-evidence' in paths:
                top_index = idx
                break
        top = fused[top_index]
        fused_without_top = fused[:top_index] + fused[top_index + 1:]
        top_row = top[0]
        try:
            account_id = str(top_row['account_id'] or '')
            conversation_id = str(top_row['conversation_id'] or '')
        except Exception:
            return fused, False
        before = []
        after = []
        rest = []
        try:
            top_timestamp = str(top_row['timestamp'] or '')
        except Exception:
            top_timestamp = ''
        for item in fused_without_top:
            row, paths, _score = item
            try:
                same_conversation = (
                    str(row['account_id'] or '') == account_id
                    and str(row['conversation_id'] or '') == conversation_id
                )
            except Exception:
                same_conversation = False
            if same_conversation and 'conversation-context' in paths:
                try:
                    timestamp = str(row['timestamp'] or '')
                except Exception:
                    timestamp = ''
                if timestamp and top_timestamp and timestamp < top_timestamp:
                    before.append(item)
                else:
                    after.append(item)
            else:
                rest.append(item)
        related = [
            *(before[:1]),
            *(after[:1]),
        ]
        if len(related) < 2:
            remaining_directional = [*before[1:], *after[1:]]
            related.extend(remaining_directional[:2 - len(related)])
        if len(related) < 2:
            remaining = [item for item in [*before[1:], *after[1:]] if item not in related]
            related.extend(remaining[:2 - len(related)])
        if not related:
            return fused, False
        selected_ids = {id(item) for item in related}
        rest = [item for item in rest if id(item) not in selected_ids]
        for item in [*before, *after]:
            if id(item) not in selected_ids:
                rest.append(item)
        return [top, *related, *rest], True

    def _force_semantic_auto(self, request: SearchRequest, understanding, *, vector_configured: bool) -> bool:
        if not vector_configured or request.semantic != 'auto' or not request.include_vector:
            return False
        filters = request.filters
        # Source-family packs already have a strong structured anchor and were a
        # Q0 non-problem; do not trade their cheap lexical precision for vector.
        if filters.get('source_type') and filters.get('source_type') not in {'message', 'all'}:
            return False
        if filters.get('source_family') and filters.get('source_family') not in {'message', 'all'}:
            return False
        # Sender/time rewrites are high-confidence semantic queries once the
        # structured filter has narrowed the scope; broad trigram scans are both
        # slow and low-signal for them.
        if any(filters.get(key) for key in ('sender', 'since', 'until')):
            return True
        normalized = understanding.normalized
        if len(normalized) <= 4:
            return False
        lowered = normalized.lower()
        rewrite_markers = (
            '为什么', '哪些', '哪个', '哪条', '谁', '怎么', '如何', '之前', '最近',
            '需要', '跟进', '推进', '安排', '决定', '结论', '原因', '阻力', '卡点',
            'recall', 'decision', 'follow', 'blocker',
        )
        if any(marker in lowered for marker in rewrite_markers):
            return True
        # Space-separated task queries are usually operator rewrites, not exact
        # WeChat phrases.  Count raw tokens, not only known domain terms, so
        # fuzzy/multi-hop rewrites with no domain vocabulary still route
        # semantic-first.
        space_parts = [part for part in normalized.split() if len(part) >= 2]
        if ' ' in normalized and len(space_parts) >= 3:
            return True
        return False

    def _collect(self, fn, queries: list[str], limit: int):
        seen: set[str] = set()
        rows = []
        for query in queries:
            if not query:
                continue
            for row in fn(query, limit):
                citation = row['citation']
                if citation in seen:
                    continue
                seen.add(citation)
                rows.append(row)
                if len(rows) >= limit:
                    return rows
        return rows

    def _unique_count(self, *row_groups) -> int:
        seen: set[str] = set()
        for rows in row_groups:
            for row in rows:
                try:
                    citation = row['citation']
                except Exception:
                    continue
                seen.add(citation)
        return len(seen)

    def _candidate_citations(self, *row_groups) -> tuple[str, ...]:
        """Return the internal first-stage oracle in deterministic route order."""

        seen: set[str] = set()
        citations: list[str] = []
        for rows in row_groups:
            for row in rows:
                for key in ('citation', 'parent_citation'):
                    try:
                        citation = str(row[key] or '')
                    except Exception:
                        continue
                    if citation and citation not in seen:
                        seen.add(citation)
                        citations.append(citation)
        return tuple(citations)
