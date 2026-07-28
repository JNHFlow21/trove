from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import hashlib

from trove_core.approvals import ApprovalGrant, claim_approval_grant
from trove_core.knowledge.observations import set_observation_status
from trove_core.media_pipeline import (
    run_image_observation_budget,
    voice_transcription_plan,
)
from trove_core.media_understanding import invalidate_media_understanding
from trove_core.runtime import index_vectors, rebuild_vectors_atomic
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.mutations import coordinated_vault_mutation
from trove_core.vault.operations import backfill_content_kind, purge_derived_data, rebuild_scope, reset_index_cache
from trove_core.wechat.files import archive_approval_payload, archive_files
from trove_core.wechat.import_job import run_import_job
from trove_core.wechat.appmsg_backfill import appmsg_source_identity, backfill_appmsg_payloads
from trove_core.wechat.media.backfill import backfill_message_media_references
from trove_core.media_fetch import fetch_media


def _cfg(vault_root: str | Path) -> VaultConfig:
    return VaultConfig.resolve(str(vault_root), env={})


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f'{field} must be an exact boolean')
    return value


def _exact_optional_limit(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise TypeError(f'{field} must be a positive exact integer or null')
    return value


def _grant_result(result: dict[str, Any], grant: ApprovalGrant) -> dict[str, Any]:
    out = dict(result)
    out['approval'] = grant.to_dict()
    return out


def full_import_payload(
    sources: Iterable[str | Path],
    *,
    reset_index_cache: bool,
    limit_per_sqlite: int | None,
    process_config: object | None,
    force_rescan: bool = False,
) -> dict[str, Any]:
    reset_index_cache = _exact_bool(reset_index_cache, field='reset_index_cache')
    force_rescan = _exact_bool(force_rescan, field='force_rescan')
    limit_per_sqlite = _exact_optional_limit(limit_per_sqlite, field='limit_per_sqlite')
    source_values = [str(Path(source).expanduser()) for source in sources]
    config_payload = None
    if process_config is not None:
        to_dict = getattr(process_config, 'to_dict', None)
        if not callable(to_dict):
            raise TypeError('process_config must expose to_dict')
        config_payload = to_dict()
        if type(config_payload) is not dict:
            raise TypeError('process_config.to_dict must return an exact dictionary')
    return {
        'sources': source_values,
        'sources_count': len(source_values),
        'reset_index_cache': reset_index_cache,
        'limit_per_sqlite': limit_per_sqlite,
        'process_config': config_payload,
        'force_rescan': force_rescan,
    }


def execute_full_import(
    vault_root: str | Path,
    sources: Iterable[str | Path],
    *,
    reset_index_cache: bool,
    limit_per_sqlite: int | None,
    process_config: object | None,
    force_rescan: bool = False,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    source_paths = [Path(source).expanduser() for source in sources]
    payload = full_import_payload(
        source_paths,
        reset_index_cache=reset_index_cache,
        limit_per_sqlite=limit_per_sqlite,
        process_config=process_config,
        force_rescan=force_rescan,
    )
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='full_import',
        danger_class='full_import',
        payload=payload,
    )
    result = run_import_job(
        cfg.root,
        source_paths,
        reset_index=reset_index_cache,
        limit_per_sqlite=limit_per_sqlite,
        process_config=process_config,
        force_rescan=force_rescan,
    )
    return _grant_result(result.to_dict(), approval_grant)


def reset_index_cache_payload() -> dict[str, Any]:
    return {'scope': 'index_cache'}


def execute_reset_index_cache(
    vault_root: str | Path,
    *,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = reset_index_cache_payload()
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='reset_index_cache',
        danger_class='destructive_rebuild',
        payload=payload,
    )
    return _grant_result(reset_index_cache(cfg.root), approval_grant)


def scope_rebuild_payload() -> dict[str, Any]:
    return {'scope': 'private_queryable'}


def execute_scope_rebuild(
    vault_root: str | Path,
    *,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = scope_rebuild_payload()
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='scope_rebuild',
        danger_class='destructive_rebuild',
        payload=payload,
    )
    report = rebuild_scope(cfg.root)
    return _grant_result(report, approval_grant)


def derived_data_purge_payload(
    *,
    scope_type: str,
    scope_id: str,
    audit_retention_days: int = 365,
) -> dict[str, Any]:
    if scope_type not in {'entity', 'source', 'run', 'task'}:
        raise ValueError('scope_type must be entity, source, run, or task')
    if type(scope_id) is not str or not scope_id or len(scope_id) > 1000:
        raise ValueError('scope_id must be non-empty bounded text')
    if type(audit_retention_days) is not int or not 1 <= audit_retention_days <= 3650:
        raise ValueError('audit_retention_days must be from 1 to 3650')
    return {
        'scope': 'derived_data',
        'scope_type': scope_type,
        'scope_hash': hashlib.sha256(f'{scope_type}:{scope_id}'.encode('utf-8')).hexdigest(),
        'audit_retention_days': audit_retention_days,
        'backup_policy': 'replace_all_pre_purge_backups_with_one_post_purge_backup',
    }


def execute_derived_data_purge(
    vault_root: str | Path,
    *,
    scope_type: str,
    scope_id: str,
    audit_retention_days: int = 365,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = derived_data_purge_payload(
        scope_type=scope_type,
        scope_id=scope_id,
        audit_retention_days=audit_retention_days,
    )
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='derived_data_purge',
        danger_class='delete_or_purge',
        payload=payload,
    )
    return _grant_result(
        purge_derived_data(
            cfg.root,
            scope_type=scope_type,
            scope_id=scope_id,
            audit_retention_days=audit_retention_days,
        ),
        approval_grant,
    )


def vector_mutation_payload(
    provider: object,
    *,
    backend: str,
    batch_size: int,
    max_messages: int | None,
    purge: bool,
) -> dict[str, Any]:
    if type(backend) is not str or backend not in {'zvec', 'sqlite'}:
        raise ValueError('unsupported vector backend')
    if type(batch_size) is not int or batch_size < 1:
        raise TypeError('batch_size must be a positive exact integer')
    max_messages = _exact_optional_limit(max_messages, field='max_messages')
    purge = _exact_bool(purge, field='purge')
    provider_name = getattr(provider, 'name', None)
    dimensions = getattr(provider, 'dimensions', None)
    if type(provider_name) is not str or not provider_name:
        raise TypeError('provider must expose a bounded name')
    if type(dimensions) is not int or dimensions < 1:
        raise TypeError('provider must expose positive dimensions')
    return {
        'backend': backend,
        'batch_size': batch_size,
        'max_messages': max_messages,
        'purge': purge,
        'provider': provider_name,
        'dimensions': dimensions,
    }


def execute_vector_mutation(
    vault_root: str | Path,
    provider: object,
    *,
    action: str,
    backend: str,
    batch_size: int,
    max_messages: int | None,
    purge: bool,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    if action not in {'vector_purge_rebuild', 'vector_rebuild'}:
        raise ValueError('unsupported vector approval action')
    cfg = _cfg(vault_root)
    payload = vector_mutation_payload(
        provider,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
        purge=purge,
    )
    if purge is not True:
        raise ValueError('approved vector mutation must purge or rebuild')
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action=action,
        danger_class='vector_purge_rebuild',
        payload=payload,
    )
    if action == 'vector_rebuild':
        report = rebuild_vectors_atomic(
            cfg,
            provider,
            backend=backend,
            batch_size=batch_size,
            max_messages=max_messages,
        )
    else:
        report = index_vectors(
            cfg,
            provider,
            backend=backend,
            batch_size=batch_size,
            max_messages=max_messages,
            purge=True,
        )
    return _grant_result(report, approval_grant)


def content_kind_backfill_payload(*, limit: int | None, backup_retention: int = 5) -> dict[str, Any]:
    if type(backup_retention) is not int or backup_retention < 1:
        raise TypeError('backup_retention must be a positive exact integer')
    return {
        'limit': _exact_optional_limit(limit, field='limit'),
        'backup_retention': backup_retention,
    }


def execute_content_kind_backfill(
    vault_root: str | Path,
    *,
    limit: int | None,
    backup_retention: int = 5,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = content_kind_backfill_payload(limit=limit, backup_retention=backup_retention)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='content_kind_backfill',
        danger_class='delete_or_purge',
        payload=payload,
    )
    report = backfill_content_kind(
        cfg.root,
        limit=limit,
        backup_retention=backup_retention,
    )
    return _grant_result(
        {
            'ok': True,
            'backup': report['backup'],
            'backfill': report['backfill'],
            'raw_content_included': False,
        },
        approval_grant,
    )


def appmsg_backfill_payload(
    source: str | Path,
    *,
    limit_per_sqlite: int | None,
    backup_retention: int = 5,
) -> dict[str, Any]:
    if type(backup_retention) is not int or backup_retention < 1:
        raise TypeError('backup_retention must be a positive exact integer')
    return {
        'source_identity': appmsg_source_identity(source),
        'limit_per_sqlite': _exact_optional_limit(limit_per_sqlite, field='limit_per_sqlite'),
        'backup_retention': backup_retention,
    }


def execute_appmsg_backfill(
    vault_root: str | Path,
    source: str | Path,
    *,
    limit_per_sqlite: int | None,
    backup_retention: int = 5,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = appmsg_backfill_payload(
        source,
        limit_per_sqlite=limit_per_sqlite,
        backup_retention=backup_retention,
    )
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='appmsg_backfill',
        danger_class='delete_or_purge',
        payload=payload,
    )
    report = backfill_appmsg_payloads(
        cfg.root,
        source,
        limit_per_sqlite=limit_per_sqlite,
        backup_retention=backup_retention,
    )
    return _grant_result(report, approval_grant)


def message_media_backfill_payload(*, limit: int | None, backup_retention: int = 5) -> dict[str, Any]:
    if type(backup_retention) is not int or backup_retention < 1:
        raise TypeError('backup_retention must be a positive exact integer')
    return {
        'limit': _exact_optional_limit(limit, field='limit'),
        'backup_retention': backup_retention,
    }


def execute_message_media_backfill(
    vault_root: str | Path,
    *,
    limit: int | None,
    backup_retention: int = 5,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = message_media_backfill_payload(limit=limit, backup_retention=backup_retention)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='message_media_backfill',
        danger_class='delete_or_purge',
        payload=payload,
    )
    report = backfill_message_media_references(
        cfg.root,
        limit=limit,
        backup_retention=backup_retention,
    )
    return _grant_result(report, approval_grant)


def execute_wechat_cdn_fetch(
    vault_root: str | Path,
    *,
    citation: str,
    approval_payload: dict[str, Any],
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='wechat_cdn_fetch',
        danger_class='remote_media_fetch',
        payload=approval_payload,
    )
    return _grant_result(
        fetch_media(
            cfg.root,
            citation,
            allow_remote=True,
            approval_grant=approval_grant,
            approval_payload=approval_payload,
        ),
        approval_grant,
    )


def media_invalidation_payload(
    *,
    content_sha256: str | None,
    model_id: str | None,
) -> dict[str, Any]:
    if content_sha256 is not None and type(content_sha256) is not str:
        raise TypeError('content_sha256 must be an exact string or null')
    if model_id is not None and type(model_id) is not str:
        raise TypeError('model_id must be an exact string or null')
    if not content_sha256 and not model_id:
        raise ValueError('content_sha256 or model_id is required')
    return {'content_sha256': content_sha256, 'model_id': model_id}


def execute_media_understanding_invalidate(
    vault_root: str | Path,
    *,
    content_sha256: str | None,
    model_id: str | None,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = media_invalidation_payload(content_sha256=content_sha256, model_id=model_id)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='media_understanding_invalidate',
        danger_class='delete_or_purge',
        payload=payload,
    )
    return _grant_result(
        invalidate_media_understanding(
            cfg.root,
            content_sha256=content_sha256,
            model_id=model_id,
        ),
        approval_grant,
    )


def execute_files_archive(
    vault_root: str | Path,
    *,
    selection: Any,
    dest_dir: str | Path,
    mode: str,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = archive_approval_payload(cfg, selection=selection, dest_dir=dest_dir, mode=mode)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='files_archive',
        danger_class='local-file-export',
        payload=payload,
    )
    report = archive_files(
        cfg,
        selection=selection,
        dest_dir=dest_dir,
        mode=mode,
        approval_grant=approval_grant,
        approval_payload=payload,
    )
    return _grant_result(report, approval_grant)


def observation_status_payload(*, observation_id: str) -> dict[str, Any]:
    if type(observation_id) is not str or not observation_id:
        raise TypeError('observation_id must be a non-empty exact string')
    return {'observation_id': observation_id}


def execute_observation_status(
    vault_root: str | Path,
    *,
    observation_id: str,
    action: str,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    if action not in {'observe_approve', 'observe_retire'}:
        raise ValueError('unsupported observation status action')
    cfg = _cfg(vault_root)
    payload = observation_status_payload(observation_id=observation_id)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action=action,
        danger_class='agent_sensitive_tool',
        payload=payload,
    )
    status = 'active' if action == 'observe_approve' else 'superseded'
    with coordinated_vault_mutation(cfg, operation='observation_write'):
        repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
        row = set_observation_status(repo, observation_id, status)
    return _grant_result(
        {'ok': True, 'observation': row, 'raw_content_included': False},
        approval_grant,
    )


def real_media_processing_payload(
    *,
    conversation_id: str,
    model_size: str,
    budget: int,
) -> dict[str, Any]:
    if type(conversation_id) is not str or not conversation_id:
        raise TypeError('conversation_id must be a non-empty exact string')
    if type(model_size) is not str or not model_size:
        raise TypeError('model_size must be a non-empty exact string')
    if type(budget) is not int or budget < 0:
        raise TypeError('budget must be a non-negative exact integer')
    return {'conversation_id': conversation_id, 'model': model_size, 'count': budget}


def prepare_real_voice_transcription(
    vault_root: str | Path,
    *,
    conversation_id: str,
    model_size: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = voice_transcription_plan(vault_root, conversation_id=conversation_id)
    budget = int(plan.get('pending') or 0)
    return plan, real_media_processing_payload(
        conversation_id=conversation_id,
        model_size=model_size,
        budget=budget,
    )


def cloud_asr_per_citation_required_payload(
    *,
    conversation_id: str | None = None,
    requested_budget: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'ok': False,
        'status': 'blocked',
        'error': {
            'code': 'cloud_asr_per_citation_required',
            'message': 'Batch voice transcription is disabled; run approved cloud ASR for each citation.',
            'action': 'use_voice_transcribe_lazy_with_cloud_approval',
        },
        'cloud_only': True,
    }
    if conversation_id:
        payload['conversation_id'] = conversation_id
    if requested_budget is not None:
        payload['requested_budget'] = int(requested_budget)
    return payload


def execute_real_voice_transcription(
    vault_root: str | Path,
    *,
    conversation_id: str,
    model_size: str,
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    _ = (vault_root, model_size, approval_grant)
    return cloud_asr_per_citation_required_payload(
        conversation_id=conversation_id,
    )


def real_media_budget_payload(
    *,
    operation: str,
    budget: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    if operation not in {'voice_transcription_budget', 'image_observation_budget'}:
        raise ValueError('unsupported real media operation')
    if type(budget) is not int or budget < 0:
        raise TypeError('budget must be a non-negative exact integer')
    if type(options) is not dict:
        raise TypeError('media options must be an exact dictionary')
    return {'operation': operation, 'count': budget, 'options': dict(options)}


def execute_real_media_budget(
    vault_root: str | Path,
    *,
    operation: str,
    budget: int,
    options: dict[str, Any],
    approval_grant: ApprovalGrant,
) -> dict[str, Any]:
    cfg = _cfg(vault_root)
    payload = real_media_budget_payload(operation=operation, budget=budget, options=options)
    if operation == 'voice_transcription_budget':
        return cloud_asr_per_citation_required_payload(requested_budget=budget)
    claim_approval_grant(
        approval_grant,
        cfg.root,
        action='real_media_processing',
        danger_class='real_media_processing',
        payload=payload,
    )
    report = run_image_observation_budget(cfg.root, budget=budget, **options)
    return _grant_result(report, approval_grant)
