from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from trove_core.bounds import (
    BoundedLimit,
    FUSION_CANDIDATES,
    RERANK_CANDIDATES,
    RETRIEVAL_CANDIDATES,
    SEARCH_RESULTS,
)
from trove_core.domain.messages import Evidence

RANKING_MODES = {'weighted', 'rrf', 'feature'}
RERANKER_MODES = {'off', 'features', 'local-bge', 'cloud-qwen3'}
SEMANTIC_MODES = {'auto', 'on', 'off'}


@dataclass(frozen=True)
class SearchRequest:
    query: str
    limit: int = 10
    account_id: str | None = None
    conversation_id: str | None = None
    conversation_type: str | None = None
    sender: str | None = None
    source_type: str | None = None
    source_family: str | None = None
    scope_type: str | None = None
    since: str | None = None
    until: str | None = None
    include_vector: bool = True
    semantic: str = 'auto'
    ranking_mode: str = 'feature'
    reranker_mode: str = 'features'
    reranker_model_path: str | None = None
    reranker_timeout_ms: int = 200
    retrieval_candidate_limit: int = 200
    fusion_candidate_limit: int = 200
    reranker_candidate_limit: int = 50
    expand_query: bool = True
    include_media_hints: bool = False

    def __post_init__(self) -> None:
        if self.ranking_mode not in RANKING_MODES:
            raise ValueError(f'unsupported ranking_mode: {self.ranking_mode}')
        if self.reranker_mode not in RERANKER_MODES:
            raise ValueError(f'unsupported reranker_mode: {self.reranker_mode}')
        if self.semantic not in SEMANTIC_MODES:
            raise ValueError(f'unsupported semantic mode: {self.semantic}')
        object.__setattr__(self, 'limit', BoundedLimit(self.limit, field='limit', spec=SEARCH_RESULTS))
        if self.reranker_timeout_ms < 1:
            raise ValueError('reranker_timeout_ms must be >= 1')
        object.__setattr__(
            self,
            'retrieval_candidate_limit',
            BoundedLimit(
                self.retrieval_candidate_limit,
                field='retrieval_candidate_limit',
                spec=RETRIEVAL_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            'fusion_candidate_limit',
            BoundedLimit(
                self.fusion_candidate_limit,
                field='fusion_candidate_limit',
                spec=FUSION_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            'reranker_candidate_limit',
            BoundedLimit(
                self.reranker_candidate_limit,
                field='reranker_candidate_limit',
                spec=RERANK_CANDIDATES,
            ),
        )

    @property
    def effective_ranking_mode(self) -> str:
        if self.reranker_mode in {'features', 'local-bge', 'cloud-qwen3'}:
            return 'feature'
        return self.ranking_mode

    @property
    def filters(self) -> dict[str, str]:
        return {k: v for k, v in {
            'account_id': self.account_id,
            'conversation_id': self.conversation_id,
            'conversation_type': self.conversation_type,
            'sender': self.sender,
            'source_type': self.source_type,
            'source_family': self.source_family,
            'scope_type': self.scope_type,
            'since': self.since,
            'until': self.until,
        }.items() if v}

@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[Evidence]
    total: int
    retrieval_status: dict[str, Any]
    elapsed_ms: float
    # Internal-only candidate oracle.  It is deliberately excluded from
    # ``to_dict`` so API/CLI responses never widen their citation surface.
    candidate_citations: tuple[str, ...] = ()
    # Internal candidates for the application-level default cloud reranker.
    # Only bundles selected into ``results`` are serialized to the caller.
    episode_bundles: tuple[Evidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'query': self.query,
            'total': self.total,
            'retrieval_status': self.retrieval_status,
            'elapsed_ms': self.elapsed_ms,
            'results': [r.to_dict() for r in self.results],
        }
