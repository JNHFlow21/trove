"""Protocol-neutral TROVE query application service.

All adapters decode their wire format into these DTOs and encode the returned
``QueryResult``.  Contact resolution, bounds, time filters and result/error
contracts therefore cannot drift between CLI, HTTP, MCP and agent tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

from trove_core.bounds import (
    BoundedInputError,
    BoundedLimit,
    CONTEXT_WINDOW,
    FUSION_CANDIDATES,
    PRIVATE_LIST,
    RERANK_CANDIDATES,
    RETRIEVAL_CANDIDATES,
    SEARCH_RESULTS,
)
from trove_core.search.query import RANKING_MODES, RERANKER_MODES, SEMANTIC_MODES, SearchRequest, SearchResponse
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import vault_generation_read
from trove_core.domain.content import display_content_for_kind
from trove_core.domain.messages import ContextMessage

from .repositories import SQLiteUnitOfWork, UnitOfWork


class QueryInputError(ValueError):
    """A validation failure that is identical in every protocol adapter."""

    def __init__(self, code: str, message: str, *, field: str, **details: Any) -> None:
        self.code = code
        self.field = field
        self.details = details
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'field': self.field,
            'message': str(self),
            **self.details,
        }


@dataclass(frozen=True)
class QueryError:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {'code': self.code, 'message': self.message, **dict(self.details)}


@dataclass(frozen=True)
class QueryResult:
    """Typed result DTO; adapters should encode only ``to_dict``."""

    data: Mapping[str, Any] = field(default_factory=dict)
    error: QueryError | None = None
    code: str = 'ok'
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def success(
        cls,
        data: Mapping[str, Any],
        *,
        code: str = 'ok',
        metrics: Mapping[str, Any] | None = None,
    ) -> 'QueryResult':
        return cls(dict(data), code=code, metrics=dict(metrics or {}))

    @classmethod
    def failure(cls, code: str, message: str, **details: Any) -> 'QueryResult':
        return cls(error=QueryError(code, message, details), code=code)

    def to_dict(self) -> dict[str, Any]:
        if self.error is not None:
            return {
                'ok': False,
                'code': self.code,
                'error': self.error.to_dict(),
                'raw_content_included': False,
            }
        payload = dict(self.data)
        payload.setdefault('ok', True)
        payload.setdefault('code', self.code)
        return payload


def _complete_generation_read(method: Callable[..., QueryResult]) -> Callable[..., QueryResult]:
    """Bind every public query to one immutable Vault generation."""

    @wraps(method)
    def guarded(self: 'TroveQueries', *args: Any, **kwargs: Any) -> QueryResult:
        # A missing Vault has no publishable generation to lease.  Preserve
        # bounded empty-result helpers without creating the root; once the
        # root exists, every read must pass the shared generation barrier.
        if not self.config.root.exists():
            return method(self, *args, **kwargs)
        with vault_generation_read(self.config):
            return method(self, *args, **kwargs)

    return guarded


@dataclass(frozen=True)
class SearchQuery:
    query: str
    limit: int = 10
    contact: str | None = None
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
    reranker_timeout_ms: int = 750
    retrieval_candidate_limit: int = 200
    fusion_candidate_limit: int = 200
    reranker_candidate_limit: int = 50
    allow_cloud_rerank: bool = False
    cloud_rerank_approval_id: str | None = None
    cloud_rerank_one_step_approval: bool = False
    expand_query: bool = True
    include_media_hints: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'limit', int(BoundedLimit(self.limit, field='limit', spec=SEARCH_RESULTS)))
        object.__setattr__(
            self,
            'retrieval_candidate_limit',
            int(BoundedLimit(
                self.retrieval_candidate_limit,
                field='retrieval_candidate_limit',
                spec=RETRIEVAL_CANDIDATES,
            )),
        )
        object.__setattr__(
            self,
            'fusion_candidate_limit',
            int(BoundedLimit(
                self.fusion_candidate_limit,
                field='fusion_candidate_limit',
                spec=FUSION_CANDIDATES,
            )),
        )
        object.__setattr__(
            self,
            'reranker_candidate_limit',
            int(BoundedLimit(
                self.reranker_candidate_limit,
                field='reranker_candidate_limit',
                spec=RERANK_CANDIDATES,
            )),
        )
        _validate_time_range(self.since, self.until)
        for field_name, value, choices in (
            ('ranking_mode', self.ranking_mode, RANKING_MODES),
            ('reranker_mode', self.reranker_mode, RERANKER_MODES),
            ('semantic', self.semantic, SEMANTIC_MODES),
        ):
            if value not in choices:
                raise QueryInputError(
                    'invalid_choice',
                    f'{field_name} is unsupported.',
                    field=field_name,
                    allowed=sorted(choices),
                )
        if type(self.reranker_timeout_ms) is not int or self.reranker_timeout_ms < 1:
            raise QueryInputError(
                'invalid_timeout',
                'reranker_timeout_ms must be an integer greater than or equal to 1.',
                field='reranker_timeout_ms',
            )
        for field_name, value in (
            ('allow_cloud_rerank', self.allow_cloud_rerank),
            ('cloud_rerank_one_step_approval', self.cloud_rerank_one_step_approval),
        ):
            if type(value) is not bool:
                raise QueryInputError(
                    'invalid_boolean',
                    f'{field_name} must be a literal boolean.',
                    field=field_name,
                )
        if self.cloud_rerank_approval_id is not None and type(self.cloud_rerank_approval_id) is not str:
            raise QueryInputError(
                'invalid_approval_id',
                'cloud_rerank_approval_id must be text.',
                field='cloud_rerank_approval_id',
            )
        if self.allow_cloud_rerank and self.reranker_mode != 'cloud-qwen3':
            raise QueryInputError(
                'invalid_cloud_rerank_mode',
                'allow_cloud_rerank requires reranker_mode=cloud-qwen3.',
                field='allow_cloud_rerank',
            )

    def request(self, *, account_id: str | None, conversation_id: str | None) -> SearchRequest:
        return SearchRequest(
            str(self.query or ''),
            limit=self.limit,
            account_id=account_id,
            conversation_id=conversation_id,
            conversation_type=self.conversation_type,
            sender=self.sender,
            source_type=self.source_type,
            source_family=self.source_family,
            scope_type=self.scope_type,
            since=self.since,
            until=self.until,
            include_vector=self.include_vector,
            semantic=self.semantic,
            ranking_mode=self.ranking_mode,
            reranker_mode=self.reranker_mode,
            reranker_model_path=self.reranker_model_path,
            reranker_timeout_ms=self.reranker_timeout_ms,
            retrieval_candidate_limit=self.retrieval_candidate_limit,
            fusion_candidate_limit=self.fusion_candidate_limit,
            reranker_candidate_limit=self.reranker_candidate_limit,
            expand_query=self.expand_query,
            include_media_hints=self.include_media_hints,
        )


@dataclass(frozen=True)
class ContextQuery:
    citation: str
    before: int = 5
    after: int = 5
    contact: str | None = None
    account_id: str | None = None
    conversation_id: str | None = None
    since: str | None = None
    until: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'before', int(BoundedLimit(self.before, field='before', spec=CONTEXT_WINDOW)))
        object.__setattr__(self, 'after', int(BoundedLimit(self.after, field='after', spec=CONTEXT_WINDOW)))
        _validate_time_range(self.since, self.until)


@dataclass(frozen=True)
class ConversationContextQuery:
    conversation_id: str
    limit: int = 20
    account_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'limit', int(BoundedLimit(self.limit, field='limit', spec=PRIVATE_LIST)))


@dataclass(frozen=True)
class ListQuery:
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, 'limit', int(BoundedLimit(self.limit, field='limit', spec=PRIVATE_LIST)))


@dataclass(frozen=True)
class FilesQuery:
    contact: str | None = None
    account_id: str | None = None
    conversation_id: str | None = None
    file_name: str | None = None
    media_types: list[str] | str | None = None
    since: str | None = None
    until: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, 'limit', int(BoundedLimit(self.limit, field='limit', spec=PRIVATE_LIST)))
        _validate_time_range(self.since, self.until)


class TroveQueries:
    """The single application owner for search, context and private lists."""

    def __init__(
        self,
        config: VaultConfig | str | Path,
        *,
        runtime: Any | None = None,
        uow_factory: Callable[..., UnitOfWork] = SQLiteUnitOfWork,
        engine_factory: Callable[[VaultConfig], Any] | None = None,
    ) -> None:
        self.config = config if isinstance(config, VaultConfig) else VaultConfig.resolve(str(config))
        self.runtime = runtime
        self._uow_factory = uow_factory
        if engine_factory is None:
            # Resolve at construction time so runtime wiring remains patchable
            # for diagnostics/tests and ProviderFactory changes are not frozen
            # into an import-time alias.
            from trove_core.runtime import build_search_engine

            engine_factory = build_search_engine
        self._engine_factory = engine_factory

    @_complete_generation_read
    def resolve_contact(
        self,
        *,
        contact: str | None,
        conversation_id: str | None,
        account_id: str | None = None,
        uow: UnitOfWork | None = None,
    ) -> QueryResult:
        if conversation_id or not str(contact or '').strip():
            return QueryResult.success({
                'account_id': account_id,
                'conversation_id': conversation_id,
                'candidates': [],
            })
        if uow is None:
            with self._uow_factory(self.config, readonly=True) as owned:
                return self.resolve_contact(
                    contact=contact,
                    conversation_id=conversation_id,
                    account_id=account_id,
                    uow=owned,
                )
        candidates = uow.messages.conversation_candidates(str(contact), limit=10)
        if len(candidates) == 1:
            candidate = candidates[0]
            return QueryResult.success({
                'account_id': str(candidate['account_id']),
                'conversation_id': str(candidate['conversation_id']),
                'candidates': candidates,
            })
        if candidates:
            return QueryResult.failure(
                'ambiguous_contact',
                'Contact name matches multiple conversations; pass conversation_id explicitly.',
                candidates=candidates,
            )
        return QueryResult.failure(
            'no_results',
            'No conversation matched the requested contact.',
            candidates=[],
        )

    @_complete_generation_read
    def search(self, query: SearchQuery) -> QueryResult:
        # The overwhelmingly common search path already has explicit/no scope.
        # Do not open a second SQLiteStore before the U8 runtime/cache: contact
        # resolution is the only reason search needs a repository UoW here.
        if str(query.contact or '').strip() and not query.conversation_id:
            with self._uow_factory(self.config, readonly=True) as uow:
                resolved = self.resolve_contact(
                    contact=query.contact,
                    conversation_id=query.conversation_id,
                    account_id=query.account_id,
                    uow=uow,
                )
            if not resolved.ok:
                return resolved
            account_id = _optional_text(resolved.data.get('account_id'))
            conversation_id = _optional_text(resolved.data.get('conversation_id'))
        else:
            account_id = _optional_text(query.account_id)
            conversation_id = _optional_text(query.conversation_id)
        from trove_core.providers.cloud_policy import cloud_retrieval_policy

        policy_enabled = bool(cloud_retrieval_policy(self.config.root)['enabled'])
        if policy_enabled and query.reranker_mode == 'features':
            query = replace(query, reranker_mode='cloud-qwen3', allow_cloud_rerank=True)
        cloud_requested = query.reranker_mode == 'cloud-qwen3' and (query.allow_cloud_rerank or policy_enabled)
        if cloud_requested:
            from trove_core.providers.config import ProviderConfig

            cloud_cfg = ProviderConfig.resolve()
            configured_window = cloud_cfg.cloud_rerank_top_k
            cloud_window = min(query.reranker_candidate_limit, configured_window)
            local_query = replace(
                query,
                limit=max(query.limit, cloud_window),
                reranker_mode='features',
                allow_cloud_rerank=False,
                cloud_rerank_approval_id=None,
                cloud_rerank_one_step_approval=False,
            )
            request = local_query.request(account_id=account_id, conversation_id=conversation_id)
        else:
            cloud_cfg = None
            cloud_window = 0
            request = query.request(account_id=account_id, conversation_id=conversation_id)
        runtime = self.runtime
        owns_runtime = runtime is None
        runtime = runtime or self._engine_factory(self.config)
        rerank_cache_metrics: dict[str, Any] = {}
        try:
            if hasattr(runtime, 'search_with_metrics'):
                response, metrics = runtime.search_with_metrics(request)
            else:
                response = runtime.search(request)
                metrics = {
                    'cache_hit': False,
                    'candidate_count': len(getattr(response, 'results', ()) or ()),
                    'duration_ms': 0,
                    'resource_count': 0,
                }
            if cloud_requested:
                def run_cloud_rerank() -> SearchResponse:
                    return self._cloud_rerank_response(
                        query,
                        response,
                        result_limit=query.limit,
                        candidate_limit=cloud_window,
                        runtime=runtime,
                    )

                if policy_enabled and hasattr(runtime, 'memoize_generation'):
                    rerank_candidates = (
                        response.episode_bundles[:cloud_window]
                        if response.episode_bundles
                        else response.results[:cloud_window]
                    )
                    candidate_key = tuple(
                        (
                            str(getattr(item, 'citation', '')),
                            float(getattr(item, 'score', 0.0) or 0.0),
                            tuple(str(value) for value in (getattr(item, 'retrieval_paths', ()) or ())),
                            tuple(str(value) for value in (getattr(item, 'supporting_citations', ()) or ())),
                            str(getattr(item, 'evidence_kind', 'message') or 'message'),
                        )
                        for item in rerank_candidates
                    )
                    response, rerank_cache_metrics = runtime.memoize_generation(
                        'cloud-rerank',
                        (
                            request,
                            query.query,
                            query.limit,
                            cloud_window,
                            query.reranker_timeout_ms,
                            getattr(cloud_cfg, 'cloud_rerank_provider', None),
                            getattr(cloud_cfg, 'cloud_rerank_model', None),
                            getattr(cloud_cfg, 'cloud_rerank_endpoint', None),
                            candidate_key,
                        ),
                        run_cloud_rerank,
                        cache_if=lambda item: (
                            ((item.retrieval_status or {}).get('reranker') or {}).get('state')
                            == 'available'
                        ),
                    )
                else:
                    response = run_cloud_rerank()
                metrics = {
                    **metrics,
                    'duration_ms': response.elapsed_ms,
                    'candidate_count': len(response.results),
                    'cloud_rerank_cache_hit': bool(rerank_cache_metrics.get('cache_hit')),
                    'cloud_rerank_singleflight_shared': bool(
                        rerank_cache_metrics.get('singleflight_shared')
                    ),
                }
        finally:
            if owns_runtime and hasattr(runtime, 'close'):
                runtime.close()
        payload = response.to_dict()
        if int(payload.get('total') or 0) <= 0:
            return QueryResult.failure(
                'no_results',
                'No evidence matched the query and filters.',
            )
        code = 'ok'
        vector = ((payload.get('retrieval_status') or {}).get('vector') or {})
        if vector.get('state') == 'degraded':
            code = 'vector_degraded'
            payload.setdefault('warnings', []).append({
                'code': code,
                'message': vector.get('reason') or vector.get('reason_code') or 'Vector retrieval degraded; lexical retrieval was used.',
                'vector': vector,
            })
        return QueryResult.success(payload, code=code, metrics=metrics)

    def _cloud_rerank_response(
        self,
        query: SearchQuery,
        response: SearchResponse,
        *,
        result_limit: int,
        candidate_limit: int,
        runtime: Any | None = None,
    ) -> SearchResponse:
        """Apply one exact-approved cloud rerank to bounded local candidates."""

        import os
        import time

        from trove_core.approvals import ApprovalManager
        from trove_core.application.cloud_commands import execute_cloud_rerank, execute_policy_cloud_rerank
        from trove_core.providers.cloud_policy import cloud_retrieval_environment, cloud_retrieval_policy
        from trove_core.providers.config import ProviderConfig
        from trove_core.providers.factory import ProviderFactory, ProviderUnavailable
        from trove_core.search.cloud_reranker import rerank_document_text
        from trove_core.security.egress import cloud_rerank_payload

        bundle_candidates = list(response.episode_bundles[:min(10, candidate_limit)])
        # Multi-hop is a distinct retrieval object: rank complete bounded
        # Episodes against one another instead of letting isolated messages
        # displace a complete chain.  Any provider failure still falls back to
        # the original message result list below.
        candidates = (
            list(bundle_candidates)
            if bundle_candidates
            else list(response.results[:candidate_limit])
        )
        multi_hop_instruct = (
            'Given a multi-hop Chinese chat-history query, rank complete conversation episode '
            'bundles above isolated messages when the bundle jointly contains every requested '
            'fact, stage, entity, number, and time relation.'
            if bundle_candidates
            else None
        )
        retrieval_status = dict(response.retrieval_status or {})
        if len(candidates) < 2:
            fallback_results = list(candidates) if bundle_candidates else list(response.results)
            fallback_citations = {str(item.citation) for item in fallback_results}
            fallback_results.extend(
                item for item in response.results if str(item.citation) not in fallback_citations
            )
            retrieval_status['reranker'] = {
                'state': 'skipped',
                'mode': 'cloud-qwen3',
                'reason_code': 'single_episode_bundle' if bundle_candidates else 'insufficient_candidates',
                'candidate_count': len(candidates),
                'invoked': False,
                'fallback_mode': 'features',
                'episode_bundle_candidates': len(bundle_candidates),
                'episode_bundle_results': min(len(bundle_candidates), result_limit),
            }
            return SearchResponse(
                response.query,
                fallback_results[:result_limit],
                min(response.total, result_limit),
                retrieval_status,
                response.elapsed_ms,
                response.candidate_citations,
                response.episode_bundles,
            )

        # The request-level allow flag is itself an explicit feature enable;
        # the exact payload still requires a durable approval before transport.
        # This keeps direct CLI/API use functional without relying on a project
        # .env file while Agent Switch remains the only credential authority.
        policy_enabled = bool(cloud_retrieval_policy(self.config.root)['enabled'])
        provider_env = cloud_retrieval_environment(self.config.root, dict(os.environ))
        provider_env['TROVE_ENABLE_CLOUD_RERANK'] = '1'
        cfg = ProviderConfig.resolve(provider_env)
        provider_timeout = max(1.5, min(5.0, query.reranker_timeout_ms / 1000.0))
        factory = None
        provider = None
        provider_cache_metrics: dict[str, Any] = {}

        def create_provider():
            candidate_factory = ProviderFactory.resolve(provider_env)
            readiness = candidate_factory.readiness('rerank')
            if not readiness.ready:
                raise ProviderUnavailable(
                    'rerank', readiness.reason_code or 'cloud_reranker_unavailable'
                )
            provider_kwargs = {'timeout': provider_timeout}
            if multi_hop_instruct:
                provider_kwargs['instruct'] = multi_hop_instruct
            return candidate_factory.create_cloud_reranker(**provider_kwargs)

        if policy_enabled and runtime is not None and hasattr(runtime, 'memoize_generation'):
            try:
                provider, provider_cache_metrics = runtime.memoize_generation(
                    'cloud-reranker-provider',
                    (
                        cfg.cloud_rerank_provider,
                        cfg.cloud_rerank_model,
                        cfg.cloud_rerank_endpoint,
                        getattr(cfg, 'cloud_rerank_secret_name', None),
                        provider_timeout,
                        multi_hop_instruct,
                    ),
                    create_provider,
                )
            except ProviderUnavailable as exc:
                readiness_reason = exc.code
        else:
            factory = ProviderFactory.resolve(provider_env)
            readiness = factory.readiness('rerank')
            readiness_reason = readiness.reason_code or 'cloud_reranker_unavailable'
            if readiness.ready:
                readiness_reason = None
        if provider is None and (factory is None or readiness_reason is not None):
            retrieval_status['reranker'] = {
                'state': 'unavailable_fallback',
                'mode': 'cloud-qwen3',
                'reason_code': readiness_reason,
                'candidate_count': 0,
                'invoked': False,
                'fallback_mode': 'features',
            }
            return SearchResponse(
                response.query,
                list(response.results[:result_limit]),
                min(response.total, result_limit),
                retrieval_status,
                response.elapsed_ms,
                response.candidate_citations,
                response.episode_bundles,
            )

        documents = [rerank_document_text(item) for item in candidates]
        approval_payload = cloud_rerank_payload(
            query=query.query,
            documents=documents,
            top_n=len(documents),
            provider=cfg.cloud_rerank_provider,
            model=cfg.cloud_rerank_model,
            endpoint=cfg.cloud_rerank_endpoint,
        )
        grant = None
        if not policy_enabled:
            grant = ApprovalManager(self.config.root).require(
                'cloud_rerank',
                'cloud_rerank_upload',
                approval_payload,
                approval_id=query.cloud_rerank_approval_id,
                one_step_approval=query.cloud_rerank_one_step_approval,
            )
        started = time.perf_counter()
        try:
            if provider is None:
                provider_kwargs = {'timeout': provider_timeout}
                if multi_hop_instruct:
                    provider_kwargs['instruct'] = multi_hop_instruct
                provider = factory.create_cloud_reranker(**provider_kwargs)
            if policy_enabled:
                cloud_result = execute_policy_cloud_rerank(
                    self.config.root,
                    query=query.query,
                    documents=documents,
                    top_n=len(documents),
                    provider=provider,
                )
            else:
                cloud_result = execute_cloud_rerank(
                    self.config.root,
                    query=query.query,
                    documents=documents,
                    top_n=len(documents),
                    provider=provider,
                    approval_grant=grant,
                )
            scored = list(cloud_result.get('results') or [])
            if not scored:
                raise RuntimeError('cloud_rerank_empty_results')
        except ProviderUnavailable as exc:
            reason_code = exc.code
            scored = []
        except RuntimeError as exc:
            reason_code = str(exc.args[0] if exc.args else exc.__class__.__name__)
            scored = []
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if not scored:
            retrieval_status['reranker'] = {
                'state': 'degraded',
                'mode': 'cloud-qwen3',
                'reason_code': reason_code,
                'candidate_count': len(candidates),
                'elapsed_ms': elapsed_ms,
                'invoked': True,
                'fallback_mode': 'features',
            }
            return SearchResponse(
                response.query,
                list(response.results[:result_limit]),
                min(response.total, result_limit),
                retrieval_status,
                round(response.elapsed_ms + elapsed_ms, 3),
                response.candidate_citations,
                response.episode_bundles,
            )

        ordered: list[Any] = []
        seen: set[int] = set()
        for item in scored:
            try:
                index = int(item.get('index'))
            except Exception:
                continue
            if 0 <= index < len(candidates) and index not in seen:
                ordered.append(candidates[index])
                seen.add(index)
        ordered.extend(item for index, item in enumerate(candidates) if index not in seen)
        ordered_citations = {str(item.citation) for item in ordered}
        for item in response.results:
            citation = str(item.citation)
            if citation not in ordered_citations:
                ordered.append(item)
                ordered_citations.add(citation)
        retrieval_status['reranker'] = {
            'state': 'available',
            'mode': 'cloud-qwen3',
            'provider': cfg.cloud_rerank_provider,
            'model': cfg.cloud_rerank_model,
            'candidate_count': len(candidates),
            'returned_count': len(scored),
            'elapsed_ms': elapsed_ms,
            'input_tokens': (cloud_result.get('usage') or {}).get('input_tokens'),
            'estimated_cost_usd': (cloud_result.get('usage') or {}).get('estimated_cost_usd'),
            'invoked': True,
            'provider_cache_hit': bool(provider_cache_metrics.get('cache_hit')),
            'provider_singleflight_shared': bool(
                provider_cache_metrics.get('singleflight_shared')
            ),
            'episode_bundle_candidates': len(bundle_candidates),
            'episode_bundle_results': sum(
                getattr(item, 'evidence_kind', 'message') == 'episode'
                for item in ordered[:result_limit]
            ),
            'candidate_scope': 'episode-bundles-only' if bundle_candidates else 'messages',
            'authorization': 'vault-continuous-retrieval-v1' if policy_enabled else 'exact-approval',
            'approval_id': grant.approval_id if grant is not None else None,
        }
        phase_latency = dict(retrieval_status.get('phase_latency_ms') or {})
        phase_latency['rerank'] = round(float(phase_latency.get('rerank') or 0.0) + elapsed_ms, 3)
        retrieval_status['phase_latency_ms'] = phase_latency
        return SearchResponse(
            response.query,
            ordered[:result_limit],
            min(response.total, result_limit),
            retrieval_status,
            round(response.elapsed_ms + elapsed_ms, 3),
            response.candidate_citations,
            response.episode_bundles,
        )

    @_complete_generation_read
    def context(self, query: ContextQuery) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            resolved = self.resolve_contact(
                contact=query.contact,
                conversation_id=query.conversation_id,
                account_id=query.account_id,
                uow=uow,
            )
            if not resolved.ok:
                return resolved
            filters = {
                key: value for key, value in {
                    'account_id': _optional_text(resolved.data.get('account_id')),
                    'conversation_id': _optional_text(resolved.data.get('conversation_id')),
                    'since': query.since,
                    'until': query.until,
                }.items() if value
            }
            if filters and not uow.evidence.citation_matches(query.citation, filters):
                return QueryResult.failure(
                    'no_results',
                    'Citation does not match the requested context filters.',
                )
            payload = self._context_payload(uow, query)
        if not payload.get('messages') and not payload.get('evidence'):
            return QueryResult.failure('no_results', 'No context or evidence matched the citation.')
        return QueryResult.success(payload)

    @_complete_generation_read
    def conversation_context(self, query: ConversationContextQuery) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            rows = uow.messages.conversation_messages(
                query.conversation_id,
                account_id=query.account_id,
                limit=query.limit,
            )
            hints = uow.media.hints_for_citations([str(row['citation']) for row in rows]) if rows else {}
        messages = [_context_message(row, hints).to_dict() | (
            {'media_hint': hints[str(row['citation'])]} if str(row['citation']) in hints else {}
        ) for row in rows]
        return QueryResult.success({
            'conversation_id': query.conversation_id,
            'limit': query.limit,
            'messages': messages,
            'evidence': None,
            'raw_content_included': True,
        })

    @_complete_generation_read
    def list_contacts(self, query: ListQuery) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            contacts = uow.search.list_contacts(limit=query.limit)
        return QueryResult.success({'contacts': contacts, 'raw_content_included': False})

    @_complete_generation_read
    def list_moments(self, query: ListQuery) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            moments = uow.search.list_moments(limit=query.limit)
        return QueryResult.success({'moments': moments, 'raw_content_included': False})

    @_complete_generation_read
    def list_favorites(self, query: ListQuery) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            favorites = uow.search.list_favorites(limit=query.limit)
        return QueryResult.success({
            'favorites': favorites,
            'namespace': 'favorites_knowledge',
            'raw_content_included': False,
        })

    @_complete_generation_read
    def list_conversations(self, query: ListQuery) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            conversations = uow.messages.list_conversations(limit=query.limit)
        return QueryResult.success({'conversations': conversations, 'raw_content_included': False})

    @_complete_generation_read
    def list_files(self, query: FilesQuery) -> QueryResult:
        if not self.config.paths.sqlite_path.is_file():
            return QueryResult.success({
                'files': [],
                'count': 0,
                'total_candidates': 0,
                'raw_paths_included': False,
            })
        with self._uow_factory(self.config, readonly=True) as uow:
            resolved = self.resolve_contact(
                contact=query.contact,
                conversation_id=query.conversation_id,
                account_id=query.account_id,
                uow=uow,
            )
            if not resolved.ok:
                return resolved
            payload = uow.media.list_files(
                account_id=_optional_text(resolved.data.get('account_id')),
                contact=None if query.contact else query.contact,
                conversation_id=_optional_text(resolved.data.get('conversation_id')),
                file_name=query.file_name,
                media_types=query.media_types,
                since=query.since,
                until=query.until,
                limit=query.limit,
            )
        return QueryResult.success(payload)

    @_complete_generation_read
    def evidence(self, citation: str) -> QueryResult:
        with self._uow_factory(self.config, readonly=True) as uow:
            row = uow.evidence.by_citation(str(citation or ''))
        if row is None:
            return QueryResult.failure('no_results', 'No evidence matched the citation.')
        return QueryResult.success({'evidence': dict(row)})

    @staticmethod
    def _context_payload(uow: UnitOfWork, query: ContextQuery) -> dict[str, Any]:
        rows = uow.messages.context_window(query.citation, before=query.before, after=query.after)
        if rows:
            citations = [str(row['citation']) for row in rows]
            hints = uow.media.hints_for_citations(citations)
            messages: list[dict[str, Any]] = []
            for row in rows:
                item = _context_message(row, hints).to_dict()
                citation = str(row['citation'])
                if citation in hints:
                    item['media_hint'] = hints[citation]
                messages.append(item)
            return {
                'citation': query.citation,
                'before': query.before,
                'after': query.after,
                'messages': messages,
                'evidence': None,
            }
        evidence = uow.evidence.by_citation(query.citation)
        if evidence is None:
            return {
                'citation': query.citation,
                'before': query.before,
                'after': query.after,
                'messages': [],
                'evidence': None,
            }
        evidence_citation = str(_row_value(evidence, 'citation') or query.citation)
        hints = uow.media.hints_for_citations([query.citation, evidence_citation])
        content = str(_row_value(evidence, 'content') or '')
        content_kind = _row_value(evidence, 'content_kind')
        if content_kind:
            content = display_content_for_kind(content, str(content_kind))
        return {
            'citation': query.citation,
            'before': 0,
            'after': 0,
            'messages': [],
            'evidence': {
                'citation': evidence_citation,
                'source_type': _row_value(evidence, 'source_type'),
                'title': _row_value(evidence, 'conversation_title'),
                'actor': _row_value(evidence, 'sender_name'),
                'timestamp': _row_value(evidence, 'timestamp'),
                'content': content[:1200],
                'media_hint': hints.get(query.citation) or hints.get(evidence_citation),
            },
        }


def validation_error_payload(exc: BoundedInputError | QueryInputError) -> dict[str, Any]:
    error = exc.to_dict()
    return {
        'ok': False,
        'code': error['code'],
        'error': error,
        'raw_content_included': False,
    }


def _validate_time_range(since: str | None, until: str | None) -> None:
    since_dt = _parse_timestamp(since, field='since')
    until_dt = _parse_timestamp(until, field='until')
    if since_dt is not None and until_dt is not None:
        try:
            reversed_range = since_dt > until_dt
        except TypeError as exc:
            raise QueryInputError(
                'invalid_time_range',
                'since and until must use compatible timezone forms.',
                field='since',
            ) from exc
        if reversed_range:
            raise QueryInputError(
                'invalid_time_range',
                'since must be earlier than or equal to until.',
                field='since',
            )


def _parse_timestamp(value: str | None, *, field: str) -> datetime | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise QueryInputError('invalid_timestamp', f'{field} must be a non-empty ISO-8601 timestamp.', field=field)
    text = value.strip()
    try:
        return datetime.fromisoformat(text[:-1] + '+00:00' if text.endswith('Z') else text)
    except ValueError as exc:
        raise QueryInputError('invalid_timestamp', f'{field} must be an ISO-8601 timestamp.', field=field) from exc


def _context_message(row: Mapping[str, Any], hints: Mapping[str, Any]) -> ContextMessage:
    del hints
    content = str(_row_value(row, 'content') or '')
    content_kind = _row_value(row, 'content_kind') or 'text'
    content = display_content_for_kind(content, str(content_kind))
    return ContextMessage(
        citation=str(_row_value(row, 'citation') or ''),
        sender_name=str(_row_value(row, 'sender_name') or ''),
        timestamp=str(_row_value(row, 'timestamp') or ''),
        content=content,
        direction=str(_row_value(row, 'direction') or 'unknown'),
    )


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _optional_text(value: Any) -> str | None:
    text = str(value or '').strip()
    return text or None


__all__ = [
    'ContextQuery',
    'ConversationContextQuery',
    'FilesQuery',
    'ListQuery',
    'QueryError',
    'QueryInputError',
    'QueryResult',
    'SearchQuery',
    'TroveQueries',
    'validation_error_payload',
]
