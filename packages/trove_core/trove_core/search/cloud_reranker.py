from __future__ import annotations

import os
import time
from typing import Any, Callable

from trove_core.approvals import ApprovalGrant, ApprovalValidationError, require_claimed_approval_grant
from trove_core.security.egress import cloud_rerank_payload
from trove_core.store.sqlite_store import vector_document_text
from trove_core.bounds import BoundedLimit, RERANK_CANDIDATES

from .fusion import RankedRow

QWEN3_RERANK_USD_PER_MILLION_INPUT_TOKENS = 0.10


class CloudRerankScores(list[tuple[int, float]]):
    """List-compatible scores with content-free provider usage metadata."""

    def __init__(self, values: list[tuple[int, float]], *, total_tokens: int | None = None):
        super().__init__(values)
        self.total_tokens = total_tokens


def _httpx_post(*args: Any, **kwargs: Any) -> Any:
    from trove_core.providers.http_pool import post

    return post(*args, **kwargs)


class CloudRerankProvider:
    name = 'cloud-rerank'
    egress_kind = 'cloud_rerank_upload'

    def __init__(
        self,
        *,
        enabled: bool = False,
        endpoint: str | None = None,
        model: str | None = None,
        api_key_env: str = 'DASHSCOPE_API_KEY',
        api_key: str | None = None,
        api_key_name: str | None = None,
        provider_name: str = 'aliyun',
        timeout: float = 30.0,
        instruct: str = 'Retrieve semantically similar text.',
        post: Callable[..., Any] | None = None,
    ):
        if not enabled:
            raise RuntimeError('Cloud rerank is disabled by default; enable explicitly after accepting that selected snippets leave the machine.')
        if not endpoint or not model:
            raise RuntimeError('Cloud rerank requires explicit endpoint and model configuration.')
        if not api_key:
            raise RuntimeError('Cloud rerank credential must be supplied explicitly by ProviderFactory.')
        self.endpoint = endpoint
        self.model = model
        self._api_key = api_key
        self.api_key_name = api_key_name or api_key_env
        self.provider_name = provider_name
        self.timeout = timeout
        self.instruct = instruct
        self._post = post or _httpx_post
        self.name = f'{provider_name}:{model}'

    def _headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
        approval_grant: ApprovalGrant | None = None,
        approval_payload: dict[str, Any] | None = None,
        vault_root: str | os.PathLike[str] | None = None,
        continuous_policy: bool = False,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        top_n = BoundedLimit(top_n, field='reranker_candidate_limit', spec=RERANK_CANDIDATES)
        expected_approval = cloud_rerank_payload(
            query=query,
            documents=documents,
            top_n=int(top_n),
            provider=self.provider_name,
            model=self.model,
            endpoint=self.endpoint,
        )
        if vault_root is None:
            raise ApprovalValidationError('cloud rerank requires a claimed approval', code='invalid_grant')
        if continuous_policy:
            from trove_core.providers.cloud_policy import cloud_retrieval_policy

            if not cloud_retrieval_policy(vault_root)['enabled']:
                raise ApprovalValidationError('cloud rerank requires explicit opt-in', code='invalid_grant')
        else:
            if type(approval_payload) is not dict or approval_payload != expected_approval:
                raise ApprovalValidationError(
                    'cloud rerank approval payload does not match outbound content',
                    code='grant_payload_mismatch',
                )
            require_claimed_approval_grant(
                approval_grant,  # type: ignore[arg-type]
                vault_root,
                action='cloud_rerank',
                danger_class='cloud_rerank_upload',
                payload=expected_approval,
            )
        payload: dict[str, Any] = {
            'model': self.model,
            'query': query,
            'documents': documents,
            'top_n': min(max(1, top_n), len(documents)),
        }
        if self.instruct:
            payload['instruct'] = self.instruct
        response = self._post(self.endpoint, headers=self._headers(), json=payload, timeout=self.timeout)
        status_code = int(getattr(response, 'status_code', 0) or 0)
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f'cloud_rerank_http_{status_code}') from exc
        if status_code >= 400:
            code = ''
            if isinstance(data, dict):
                error = data.get('error') or {}
                if isinstance(error, dict):
                    code = str(error.get('code') or error.get('type') or '')
            raise RuntimeError(f'cloud_rerank_http_{status_code}{("_" + code) if code else ""}')
        total_tokens = None
        usage = data.get('usage') if isinstance(data, dict) else None
        if isinstance(usage, dict):
            raw_tokens = usage.get('total_tokens')
            if isinstance(raw_tokens, int) and raw_tokens >= 0:
                total_tokens = raw_tokens
        return CloudRerankScores(
            _parse_rerank_results(data, len(documents)),
            total_tokens=total_tokens,
        )


def rerank_document_text(row: Any) -> str:
    def value(key: str) -> str:
        if isinstance(row, dict):
            return str(row.get(key) or '')
        try:
            return str(getattr(row, key) or '')
        except Exception:
            return ''

    # SearchResponse carries bounded evidence snippets instead of the raw
    # SQLite row.  Preserve the same contextual fields for the explicit cloud
    # path without widening the ordinary response or re-reading source data.
    episode_rerank_text = value('_rerank_text')
    if value('evidence_kind') == 'episode' and episode_rerank_text:
        return episode_rerank_text
    snippet = value('snippet')
    if snippet:
        object_label = '微信会话证据包' if value('evidence_kind') == 'episode' else '微信聊天证据'
        parts = [
            f'检索对象: {object_label}',
            f"来源类型: {value('source_type')}",
            f"会话: {value('conversation_title')}",
            f"会话类型: {value('conversation_type')}",
            f"说话人: {value('sender_name')}",
            f"方向: {value('direction')}",
            f"时间: {value('timestamp')}",
            f"证据正文: {snippet}",
        ]
        return '\n'.join(part for part in parts if part.split(':', 1)[-1].strip())
    return vector_document_text(row)


def rerank_with_cloud_model(
    ranked: list[RankedRow],
    query: str,
    *,
    provider: CloudRerankProvider | None = None,
    limit: int,
    candidate_limit: int,
    timeout_ms: int | None = None,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict[str, Any] | None = None,
    vault_root: str | os.PathLike[str] | None = None,
) -> tuple[list[RankedRow], dict]:
    limit = BoundedLimit(limit, field='limit', spec=RERANK_CANDIDATES)
    candidate_limit = BoundedLimit(
        candidate_limit,
        field='reranker_candidate_limit',
        spec=RERANK_CANDIDATES,
    )
    if provider is None:
        return ranked[:limit], {
            'state': 'unavailable_fallback',
            'mode': 'cloud-qwen3',
            'candidate_count': 0,
            'returned_count': 0,
            'elapsed_ms': 0.0,
            'reason_code': 'cloud_reranker_requires_exact_approval',
            'fallback_mode': 'features',
        }
    candidates = ranked[:candidate_limit]
    documents = [rerank_document_text(row) for row, _paths, _score in candidates]
    start = time.perf_counter()
    try:
        scored_indexes = provider.rerank(
            query,
            documents,
            top_n=len(documents),
            approval_grant=approval_grant,
            approval_payload=approval_payload,
            vault_root=vault_root,
        )
    except Exception as exc:
        return ranked[:limit], {
            'state': 'degraded',
            'mode': 'cloud-qwen3',
            'provider': provider.provider_name,
            'model': provider.model,
            'candidate_count': len(candidates),
            'returned_count': 0,
            'elapsed_ms': round((time.perf_counter() - start) * 1000, 3),
            'reason_code': _reason_code(exc),
            'fallback_mode': 'features',
        }
    ordered: list[RankedRow] = []
    seen_indexes: set[int] = set()
    for index, score in scored_indexes:
        if index < 0 or index >= len(candidates) or index in seen_indexes:
            continue
        row, paths, _base_score = candidates[index]
        ordered.append((row, paths, float(score)))
        seen_indexes.add(index)
    for idx, item in enumerate(candidates):
        if idx not in seen_indexes:
            ordered.append(item)
    ordered.extend(ranked[len(candidates):])
    return ordered[:limit], {
        'state': 'available',
        'mode': 'cloud-qwen3',
        'provider': provider.provider_name,
        'model': provider.model,
        'candidate_count': len(candidates),
        'returned_count': len(scored_indexes),
        'elapsed_ms': round((time.perf_counter() - start) * 1000, 3),
        'input_tokens': getattr(scored_indexes, 'total_tokens', None),
        'estimated_cost_usd': (
            round(
                getattr(scored_indexes, 'total_tokens')
                * QWEN3_RERANK_USD_PER_MILLION_INPUT_TOKENS
                / 1_000_000,
                9,
            )
            if isinstance(getattr(scored_indexes, 'total_tokens', None), int)
            else None
        ),
    }


def _parse_rerank_results(data: Any, document_count: int) -> list[tuple[int, float]]:
    if not isinstance(data, dict):
        raise RuntimeError('cloud_rerank_invalid_response')
    results: Any = data.get('results')
    if results is None and isinstance(data.get('output'), dict):
        results = data['output'].get('results')
    if results is None and isinstance(data.get('data'), dict):
        results = data['data'].get('results')
    if results is None and isinstance(data.get('data'), list):
        results = data.get('data')
    if not isinstance(results, list):
        raise RuntimeError('cloud_rerank_missing_results')
    parsed: list[tuple[int, float]] = []
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError('cloud_rerank_invalid_result')
        index = item.get('index')
        if index is None:
            index = item.get('document_index')
        score = item.get('relevance_score')
        if score is None:
            score = item.get('score')
        if score is None:
            score = item.get('relevanceScore')
        try:
            idx = int(index)
            val = float(score)
        except Exception as exc:
            raise RuntimeError('cloud_rerank_invalid_score') from exc
        if idx < 0 or idx >= document_count:
            continue
        parsed.append((idx, val))
    if not parsed:
        raise RuntimeError('cloud_rerank_empty_results')
    parsed.sort(key=lambda item: -item[1])
    return parsed


def _reason_code(exc: Exception) -> str:
    code = getattr(exc, 'code', None)
    if type(code) is str and code:
        return code
    if isinstance(exc, RuntimeError) and exc.args:
        code = str(exc.args[0] or '').strip()
        if code:
            return code
    return exc.__class__.__name__
