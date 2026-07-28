from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any

from trove_core.approvals import ApprovalGrant, claim_approval_grant
from trove_core.bounds import BoundedLimit, RERANK_CANDIDATES
from trove_core.media_pipeline import (
    _cloud_asr_provider_from_runtime,
    ensure_voice_transcript,
)
from trove_core.providers.config import ProviderConfig
from trove_core.security.egress import cloud_asr_payload
from trove_core.security.egress import cloud_rerank_payload, cloud_vision_payload
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vision.jobs import run_image_observation_job
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation
from trove_core.runtime import index_vectors, rebuild_vectors_atomic, vector_cloud_approval_payload


def _environment_snapshot(env: dict[str, str] | None) -> dict[str, str]:
    if env is None:
        snapshot = dict(os.environ)
    elif type(env) is not dict:
        raise TypeError('cloud command environment must be an exact dictionary')
    else:
        snapshot = dict(env)
    if any(type(key) is not str or type(value) is not str for key, value in snapshot.items()):
        raise TypeError('cloud command environment keys and values must be exact strings')
    return snapshot


def _exact_citation(citation: object) -> str:
    if type(citation) is not str:
        raise TypeError('citation must be an exact string')
    if not citation or len(citation.encode('utf-8')) > 16 * 1024:
        raise ValueError('citation must be non-empty and bounded')
    return citation


def _require_cloud_provider_identity(provider: object, expected: dict[str, Any]) -> None:
    fields = {
        'name': expected['provider'],
        'model_name': expected['model'],
        'resource_id': expected['resource_id'],
    }
    for field, value in fields.items():
        actual = getattr(provider, field, None)
        if type(actual) is not str or actual != value:
            raise RuntimeError('cloud_asr_provider_identity_mismatch')
    endpoint = getattr(provider, 'endpoint', None)
    if type(endpoint) is not str:
        raise RuntimeError('cloud_asr_provider_identity_mismatch')
    expected_endpoint_hash = cloud_asr_payload(
        citation='identity-check',
        provider=expected['provider'],
        model=expected['model'],
        resource_id=expected['resource_id'],
        endpoint=endpoint,
    )['endpoint_hash']
    if expected_endpoint_hash != expected['endpoint_hash']:
        raise RuntimeError('cloud_asr_provider_identity_mismatch')


def cloud_voice_transcript_payload(
    citation: str,
    *,
    env: dict[str, str] | None = None,
    profile_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    citation = _exact_citation(citation)
    provider_config = ProviderConfig.resolve(_environment_snapshot(env))
    payload = cloud_asr_payload(
        citation=citation,
        provider='volcengine-asr-flash',
        model=provider_config.asr_model_name,
        resource_id=provider_config.asr_resource_id,
        endpoint=provider_config.asr_endpoint,
    )
    if profile_scope is not None:
        required = {
            'profile_run_hash', 'task_set_hash', 'citation_set_hash', 'source_revision_hash',
            'content_hash', 'actor_hash', 'session_hash', 'purpose', 'cost_ceiling_rmb',
        }
        if type(profile_scope) is not dict or set(profile_scope) != required:
            raise ValueError('profile_scope must contain the exact profile cloud-ASR scope')
        normalized: dict[str, Any] = {}
        for key in required - {'cost_ceiling_rmb'}:
            value = profile_scope[key]
            if type(value) is not str or len(value) > 256:
                raise ValueError(f'profile_scope {key} is invalid')
            normalized[key] = value
        ceiling = profile_scope['cost_ceiling_rmb']
        if ceiling is not None and (
            type(ceiling) not in {int, float}
            or not math.isfinite(float(ceiling))
            or float(ceiling) < 0
        ):
            raise ValueError('profile_scope cost_ceiling_rmb is invalid')
        normalized['cost_ceiling_rmb'] = None if ceiling is None else round(float(ceiling), 6)
        payload['profile_scope'] = normalized
    return payload


def execute_cloud_voice_transcript(
    vault_root: str | Path,
    *,
    citation: str,
    approval_grant: ApprovalGrant,
    env: dict[str, str] | None = None,
    profile_scope: dict[str, Any] | None = None,
    allow_group_voice: bool = False,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    """Run the only public cloud-ASR path after exact approval claim."""

    citation = _exact_citation(citation)
    env_snapshot = _environment_snapshot(env)
    cfg = VaultConfig.resolve(str(vault_root), env={})
    payload = cloud_voice_transcript_payload(citation, env=env_snapshot, profile_scope=profile_scope)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='voice_cloud_asr',
        danger_class='cloud_asr_upload',
        payload=payload,
    )

    # Provider construction and secret resolution happen only after the exact
    # capability has been claimed.  A replay or mismatched grant therefore has
    # zero provider/network side effects.
    provider, reason = _cloud_asr_provider_from_runtime(env_snapshot)
    if provider is None:
        return {
            'ok': False,
            'status': 'needs_provider',
            'reason': reason or 'provider_unavailable',
            'cloud_calls_made': False,
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    _require_cloud_provider_identity(provider, payload)
    return ensure_voice_transcript(
        cfg.root,
        citation=citation,
        allow_cloud_asr=True,
        provider=provider,
        env=env_snapshot,
        approval_grant=approval_grant,
        approval_payload=payload,
        cloud_cost_ceiling_rmb=(
            float(profile_scope['cost_ceiling_rmb'])
            if profile_scope is not None and profile_scope['cost_ceiling_rmb'] is not None else None
        ),
        allow_group_voice=allow_group_voice,
        write_session=write_session,
    )


def execute_cloud_vector_index(
    vault_root: str | Path,
    *,
    provider,
    approval_grant: ApprovalGrant,
    backend: str = 'zvec',
    batch_size: int = 256,
    max_messages: int | None = None,
    purge: bool = False,
    citations=None,
) -> dict[str, Any]:
    egress_kind = getattr(provider, 'egress_kind', None)
    if type(egress_kind) is not str or egress_kind != 'cloud_embedding_upload':
        raise RuntimeError('cloud_embedding_provider_identity_mismatch')
    cfg = VaultConfig.resolve(str(vault_root), env={})
    payload = vector_cloud_approval_payload(
        cfg,
        provider,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
        purge=purge,
        citations=citations,
    )
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='cloud_vector_index',
        danger_class='cloud_embedding_upload',
        payload=payload,
    )
    if purge:
        return rebuild_vectors_atomic(
            cfg,
            provider,
            backend=backend,
            batch_size=batch_size,
            max_messages=max_messages,
            approval_grant=approval_grant,
            approval_payload=payload,
        )
    return index_vectors(
        cfg,
        provider,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
        purge=False,
        citations=citations,
        approval_grant=approval_grant,
        approval_payload=payload,
    )


def execute_cloud_image_observation(
    vault_root: str | Path,
    *,
    asset_id: str,
    image_path: str | Path,
    citation: str,
    provider,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    if getattr(provider, 'egress_kind', None) != 'cloud_vision_upload':
        raise RuntimeError('cloud_vision_provider_identity_mismatch')
    if type(asset_id) is not str or not asset_id:
        raise TypeError('asset_id must be a non-empty exact string')
    citation = _exact_citation(citation)
    provider_name = getattr(provider, 'name', None)
    model = getattr(provider, 'model', None)
    endpoint = getattr(provider, 'endpoint', None)
    if any(type(value) is not str or not value for value in (provider_name, model, endpoint)):
        raise RuntimeError('cloud_vision_provider_identity_mismatch')
    cfg = VaultConfig.resolve(str(vault_root), env={})
    payload = cloud_vision_payload(
        citation=citation,
        provider=provider_name,
        model=model,
        endpoint=endpoint,
    )
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='image_cloud_vision',
        danger_class='cloud_vision_upload',
        payload=payload,
    )
    return run_image_observation_job(
        MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path)),
        asset_id=asset_id,
        image_path=Path(image_path),
        provider=provider,
        citation=citation,
        approval_grant=approval_grant,
        approval_payload=payload,
        mutation_context=lambda: coordinated_vault_mutation(cfg, operation='media_observe'),
    )


def execute_cloud_rerank(
    vault_root: str | Path,
    *,
    query: str,
    documents: list[str],
    top_n: int,
    provider,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    if getattr(provider, 'egress_kind', None) != 'cloud_rerank_upload':
        raise RuntimeError('cloud_rerank_provider_identity_mismatch')
    provider_name = getattr(provider, 'provider_name', None)
    model = getattr(provider, 'model', None)
    endpoint = getattr(provider, 'endpoint', None)
    if any(type(value) is not str or not value for value in (provider_name, model, endpoint)):
        raise RuntimeError('cloud_rerank_provider_identity_mismatch')
    top_n = int(BoundedLimit(top_n, field='reranker_candidate_limit', spec=RERANK_CANDIDATES))
    cfg = VaultConfig.resolve(str(vault_root), env={})
    payload = cloud_rerank_payload(
        query=query,
        documents=documents,
        top_n=top_n,
        provider=provider_name,
        model=model,
        endpoint=endpoint,
    )
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='cloud_rerank',
        danger_class='cloud_rerank_upload',
        payload=payload,
    )
    results = provider.rerank(
        query,
        documents,
        top_n=top_n,
        approval_grant=approval_grant,
        approval_payload=payload,
        vault_root=cfg.root,
    )
    input_tokens = getattr(results, 'total_tokens', None)
    from trove_core.search.cloud_reranker import QWEN3_RERANK_USD_PER_MILLION_INPUT_TOKENS

    return {
        'ok': True,
        'results': [{'index': index, 'score': score} for index, score in results],
        'usage': {
            'input_tokens': input_tokens,
            'estimated_cost_usd': (
                round(input_tokens * QWEN3_RERANK_USD_PER_MILLION_INPUT_TOKENS / 1_000_000, 9)
                if isinstance(input_tokens, int)
                else None
            ),
            'output_tokens_billed': 0,
        },
        'approval': approval_grant.to_dict(),
        'raw_content_included': False,
    }


def execute_policy_cloud_rerank(
    vault_root: str | Path,
    *,
    query: str,
    documents: list[str],
    top_n: int,
    provider,
) -> dict[str, Any]:
    """Rerank under one durable Vault-level continuous retrieval opt-in."""

    from trove_core.providers.cloud_policy import cloud_retrieval_policy

    cfg = VaultConfig.resolve(str(vault_root), env={})
    if not cloud_retrieval_policy(cfg.root)['enabled']:
        raise RuntimeError('cloud_retrieval_policy_disabled')
    if getattr(provider, 'egress_kind', None) != 'cloud_rerank_upload':
        raise RuntimeError('cloud_rerank_provider_identity_mismatch')
    top_n = int(BoundedLimit(top_n, field='reranker_candidate_limit', spec=RERANK_CANDIDATES))
    results = provider.rerank(
        query,
        documents,
        top_n=top_n,
        vault_root=cfg.root,
        continuous_policy=True,
    )
    input_tokens = getattr(results, 'total_tokens', None)
    from trove_core.search.cloud_reranker import QWEN3_RERANK_USD_PER_MILLION_INPUT_TOKENS

    return {
        'ok': True,
        'results': [{'index': index, 'score': score} for index, score in results],
        'usage': {
            'input_tokens': input_tokens,
            'estimated_cost_usd': (
                round(input_tokens * QWEN3_RERANK_USD_PER_MILLION_INPUT_TOKENS / 1_000_000, 9)
                if isinstance(input_tokens, int) else None
            ),
            'output_tokens_billed': 0,
        },
        'policy': 'vault-continuous-retrieval-v1',
        'raw_content_included': False,
    }
