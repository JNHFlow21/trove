from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import hashlib
import uuid

from trove_core.bounds import (
    BoundedLimit,
    PRIVATE_LIST,
    TRACE_EVENTS_APPROVALS,
)
from trove_core.embedding.model_registry import model_status as embedding_model_status
from trove_core.knowledge.conversation_card import build_conversation_card
from trove_core.knowledge.customer_card import build_customer_card
from trove_core.knowledge.customer_profile import build_customer_profile
from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.knowledge.entity_reconciliation import reconciliation_plan, reconcile_customer_entities
from trove_core.knowledge.person_profile import build_person_profile, propose_person_profile_claims as write_person_profile_claims
from trove_core.knowledge.profile_enrichment import ProfileEnrichmentService
from trove_core.knowledge.profile_automation import ProfileAutomationService, process_profile_refresh_queue
from trove_core.knowledge.profile_snapshots import (
    FINAL_STATES,
    diff_profile_snapshots as read_profile_snapshot_diff,
    finalize_profile_snapshot,
    get_profile_snapshot as read_profile_snapshot,
    list_profile_snapshots as read_profile_snapshot_list,
    profile_snapshot_status as read_profile_snapshot_status,
)
from trove_core.knowledge.report import build_cited_report
from trove_core.providers.config import provider_status_payload
from trove_core.providers.pricing import pricing_payload
from trove_core.providers.readiness import CloudReadinessInput, check_cloud_processing_readiness
from trove_core.runtime import build_search_engine, configured_embedding_provider, vector_status_payload
from trove_core.approvals import ApprovalManager, ApprovalRequired, canonical_payload_digest
from trove_core.application.sensitive_commands import (
    appmsg_backfill_payload,
    cloud_asr_per_citation_required_payload,
    execute_appmsg_backfill,
    execute_files_archive,
    execute_message_media_backfill,
    execute_media_understanding_invalidate,
    execute_observation_status,
    media_invalidation_payload,
    message_media_backfill_payload,
    observation_status_payload,
    prepare_real_voice_transcription,
)
from trove_core.application.commands import (
    AuxiliaryImportCommand,
    FullImportCommand,
    MaintainCommand,
    SyncCommand,
    TroveCommands,
    VectorCommand,
)
from trove_core.application.cloud_commands import (
    cloud_voice_transcript_payload,
    execute_cloud_voice_transcript,
)
from trove_core.application.queries import (
    ContextQuery,
    ConversationContextQuery,
    FilesQuery,
    ListQuery,
    SearchQuery,
    TroveQueries,
)
from trove_core.wechat.process_config import process_config_from_payload, write_process_config, read_latest_process_config
from trove_core.vault.tracing import TraceTimeline
from trove_core.search.evaluation import evaluate_golden
from trove_core.knowledge.wiki import build_wiki_page, write_wiki_page
from trove_core.store.repositories import EntityRecord, MultimodalRepository, ObservationRecord
from trove_core.store.sqlite_store import SQLiteStore, open_store
from trove_core.media_fetch import fetch_media
from trove_core.media_understanding import annotate_media_understanding, media_understanding_status
from trove_core.media_pipeline import ensure_voice_transcript, media_status_payload, run_voice_transcription_budget
from trove_core.schedule import ScheduleInstallOptions, install_schedule
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import vault_generation_read
from trove_core.vault.operations import (
    read_last_import_status,
    rebuild_chunks as rebuild_vault_chunks,
    rebuild_fts as rebuild_vault_fts,
    rebuild_scope as rebuild_vault_scope,
    reset_index_cache as reset_vault_index_cache,
)
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint, record_vault_mutation_noop
from trove_core.vision.fake import FakeVisionProvider
from trove_core.vision.jobs import run_image_observation_job
from trove_core.wechat.decrypt import DecryptConfig, build_decrypt_plan, decrypt_status as _decrypt_status
from trove_core.wechat.decrypt.config import selected_accounts_from_strings
from trove_core.wechat.files import archive_approval_payload
from trove_core.wechat.appmsg_backfill import appmsg_backfill_plan as build_appmsg_backfill_plan, recover_appmsg_payload
from trove_core.wechat.media.backfill import message_media_backfill_plan as build_message_media_backfill_plan
from trove_core.wechat.media.resources import discover_media_assets, summarize_media_references
from trove_core.wechat.source_inventory import inventory
from trove_core.wechat.source_manifest import build_manifest
from trove_core.wechat.scope import public_scope_contract


_STABLE_ID_NAMESPACE = 'trove' + '-wechat'


def _cfg(vault_root: VaultConfig | str | Path | None) -> VaultConfig:
    if isinstance(vault_root, VaultConfig):
        return vault_root
    return VaultConfig.resolve(str(vault_root) if vault_root is not None else None)


def _store(vault_root: str | Path) -> SQLiteStore:
    return SQLiteStore(_cfg(vault_root).paths.sqlite_path)


def _finalize_terminal_profile(
    store: SQLiteStore,
    run_id: str,
    *,
    actor: str,
    session: str,
    result: dict,
) -> dict:
    """Attach an idempotent saved snapshot whenever a run reaches a final state."""

    state = str(result.get('run_state') or result.get('state') or '')
    if state not in FINAL_STATES:
        state = str(ProfileEnrichmentService(store).manifest(
            run_id, actor=actor, session=session, limit=1,
        )['state'])
    if state not in FINAL_STATES:
        return result
    snapshot = finalize_profile_snapshot(
        store, run_id, actor=actor, session=session,
    )
    return result | {'state': state, 'profile_snapshot': snapshot}


def _voice_provider_cost(store: SQLiteStore, result: dict) -> float:
    transcript = result.get('transcript') if isinstance(result.get('transcript'), dict) else {}
    job_id = result.get('job_id') or transcript.get('job_id')
    with store.connect() as conn:
        if job_id:
            row = conn.execute(
                'SELECT COALESCE(cost_rmb,0) FROM provider_jobs WHERE job_id=?',
                (job_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COALESCE(pj.cost_rmb,0)
                     FROM transcripts tr JOIN provider_jobs pj ON pj.job_id=tr.job_id
                    WHERE tr.asset_id=? AND tr.status='active'
                    ORDER BY tr.created_at DESC LIMIT 1""",
                (result.get('asset_id'),),
            ).fetchone() if result.get('asset_id') else None
    return float(row[0]) if row is not None else 0.0


class RepairInputError(ValueError):
    pass


def _repair_limit(value: int | None, *, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 1_000_000:
        raise RepairInputError(f'{field} must be an exact integer from 1 to 1000000')
    return value


def _repair_retention(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 100:
        raise RepairInputError('backup_retention must be an exact integer from 1 to 100')
    return value


def _repair_source(value: str | Path) -> str | Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RepairInputError('source must be a non-empty local path')
    return value


@contextmanager
def _generation_read_if_present(cfg: VaultConfig) -> Iterator[None]:
    """Do not create a missing Vault merely to acquire a read lease."""

    if not cfg.root.exists():
        yield
        return
    with vault_generation_read(cfg):
        yield


@contextmanager
def _read_store(vault_root: VaultConfig | str | Path) -> Iterator[SQLiteStore]:
    """Open one validated read-only store and deterministically close it.

    Agent query helpers must never turn a missing or historical Vault into an
    implicit schema migration.  Mutation helpers continue to use ``_store``
    while this boundary opens ``mode=ro`` only when an index already exists.
    """

    cfg = _cfg(vault_root)
    path = cfg.paths.sqlite_path
    with _generation_read_if_present(cfg):
        store = open_store(path, readonly=True) if path.is_file() else SQLiteStore(path, readonly=True)
        try:
            yield store
        finally:
            store.close()


def _resolve_entity_id(store: SQLiteStore, entity: str, *, materialize: bool = False) -> str:
    value = str(entity or '').strip()
    if not value:
        raise ValueError('entity is required')
    store.initialize()
    with store.connect() as conn:
        row = conn.execute('SELECT entity_id FROM entities WHERE entity_id=? LIMIT 1', (value,)).fetchone()
        if row is not None:
            return row['entity_id']
    resolution = resolve_customer(store, value)
    resolved = resolution.get('resolved') or {}
    entity_id = str(resolved.get('entity_id') or '').strip()
    if entity_id and not entity_id.startswith('unresolved:'):
        return entity_id
    if entity_id.startswith('unresolved:') and resolved.get('primary_user_id'):
        # Preserve pre-1.0 identifiers while keeping the public product name TROVE.
        materialized = 'customer-' + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{_STABLE_ID_NAMESPACE}:{resolved.get('primary_user_id')}",
        ).hex[:16]
        if materialize:
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id=materialized,
                entity_type=resolved.get('entity_type') or 'Customer',
                display_name=resolved.get('display_name') or value,
                identifiers={
                    'primary_user_id': resolved.get('primary_user_id'),
                    'aliases': resolved.get('aliases') or [value],
                    'resolution_source': 'observe_materialized_unresolved_customer',
                    'source_entity_ref': entity_id,
                },
                status='needs_review',
                confidence=float(resolved.get('confidence') or 0.7),
            ))
        return materialized
    if materialize:
        # An explicit operator/agent intake may legitimately precede any
        # imported contact or private conversation. Materialize a low-trust
        # reviewable entity instead of making observation intake impossible.
        materialized = 'customer-' + uuid.uuid5(
            uuid.NAMESPACE_URL, f'{_STABLE_ID_NAMESPACE}:profile-intake:{value.casefold()}',
        ).hex[:16]
        MultimodalRepository(store).upsert_entity(EntityRecord(
            entity_id=materialized,
            entity_type='Customer',
            display_name=value,
            identifiers={
                'aliases': [value],
                'resolution_source': 'observe_materialized_unresolved_customer',
                'source_entity_ref': f'unresolved:{value}',
            },
            status='needs_review',
            confidence=0.5,
        ))
        return materialized
    raise ValueError(f'entity not found: {value}')


def _redacted_status(data: dict) -> dict:
    data = dict(data)
    data['root'] = 'configured-vault' if data.get('available') else 'unavailable-vault'
    data['index_path'] = 'redacted'
    return data


def vault_status(
    vault_root: VaultConfig | str | Path,
    *,
    include_coverage: bool = False,
    redact_paths: bool = True,
) -> dict:
    cfg = _cfg(vault_root)
    with _generation_read_if_present(cfg):
        if cfg.paths.sqlite_path.is_file():
            with _read_store(cfg.root) as store:
                counts = store.counts()
        else:
            counts = None
        data = cfg.status(counts).to_dict()
        if include_coverage:
            data['coverage'] = read_last_import_status(cfg.root)['coverage']
    return _redacted_status(data) if redact_paths else data


def source_inventory(sources: list[str]) -> dict:
    return build_manifest(inventory(sources)).to_dict()



def media_inventory(sources: list[str], *, limit_per_table: int | None = None) -> dict:
    refs = []
    for source in sources:
        refs.extend(discover_media_assets(Path(source), limit_per_table=limit_per_table))
    return {
        'counts': summarize_media_references(refs),
        'assets': [
            {
                'asset_id': r.asset_id,
                'account_id': r.account_id,
                'source_type': r.source_type,
                'source_id': r.source_id,
                'modality': r.modality,
                'media_type': r.media_type,
                'cache_state': r.cache_state,
                'citation': r.citation,
            } for r in refs
        ],
        'raw_paths_included': False,
    }


@mutation_entrypoint('auxiliary_import')
def import_contacts(vault_root: str | Path, contact_db: str | Path, *, account_id: str, limit: int | None = None) -> dict:
    return TroveCommands(_cfg(vault_root)).auxiliary_import(
        AuxiliaryImportCommand('contacts', contact_db, account_id, limit),
    )


@mutation_entrypoint('auxiliary_import')
def import_moments(vault_root: str | Path, sns_db: str | Path, *, account_id: str, limit: int | None = None) -> dict:
    return TroveCommands(_cfg(vault_root)).auxiliary_import(
        AuxiliaryImportCommand('moments', sns_db, account_id, limit),
    )


@mutation_entrypoint('auxiliary_import')
def import_favorites(vault_root: str | Path, favorite_db: str | Path, *, account_id: str, limit: int | None = None) -> dict:
    return TroveCommands(_cfg(vault_root)).auxiliary_import(
        AuxiliaryImportCommand('favorites', favorite_db, account_id, limit),
    )


def provider_jobs(vault_root: str | Path, *, limit: int = 50) -> dict:
    limit = int(BoundedLimit(limit, field='limit', spec=TRACE_EVENTS_APPROVALS))
    if not _cfg(vault_root).paths.sqlite_path.is_file():
        return {'jobs': []}
    with _read_store(vault_root) as store:
        jobs = MultimodalRepository(store).provider_job_status(limit=limit)
    return {'jobs': jobs}


@mutation_entrypoint('media_transcribe')
def voice_transcribe_fixture(
    vault_root: str | Path,
    audio_path: str | Path,
    *,
    asset_id: str,
    citation: str,
    transcript: str = 'fixture transcript',
) -> dict:
    _ = (vault_root, audio_path, asset_id, citation, transcript)
    record_vault_mutation_noop(operation='media_transcribe')
    return {
        'ok': False,
        'status': 'rejected',
        'error': {
            'code': 'fake_asr_test_only',
            'message': 'Caller-supplied ASR transcripts are test-only; use per-citation cloud ASR.',
            'action': 'use_voice_transcribe_lazy_with_cloud_approval',
        },
        'cloud_only': True,
    }


def voice_transcribe_conversation(
    vault_root: str | Path,
    *,
    conversation_id: str,
    dry_run: bool = True,
    yes: bool = False,
    model_size: str = 'small',
    approval_id: str | None = None,
) -> dict:
    if dry_run or (not yes and not approval_id):
        plan, _payload = prepare_real_voice_transcription(
            vault_root,
            conversation_id=conversation_id,
            model_size=model_size,
        )
        return {'ok': True, 'dry_run': True, **plan}
    return cloud_asr_per_citation_required_payload(
        conversation_id=conversation_id,
    )


@mutation_entrypoint('media_transcribe')
def voice_transcribe_lazy(
    vault_root: str | Path,
    *,
    citation: str,
    allow_cloud_asr: bool = False,
    approval_id: str | None = None,
) -> dict:
    if not allow_cloud_asr:
        return ensure_voice_transcript(vault_root, citation=citation, allow_cloud_asr=False)
    cfg = _cfg(vault_root)
    payload = cloud_voice_transcript_payload(citation)
    grant = ApprovalManager(cfg.root).require(
        'voice_cloud_asr',
        'cloud_asr_upload',
        payload,
        approval_id=approval_id,
        one_step_approval=False,
    )
    return execute_cloud_voice_transcript(
        cfg.root,
        citation=citation,
        approval_grant=grant,
    )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_voice_execute(
    vault_root: str | Path,
    task_id: str,
    *,
    actor: str,
    session: str,
    worker: str,
    claim_token: str,
    approval_id: str | None = None,
) -> dict:
    """Advance one voice task without holding the writer during ASR/provider work."""
    cfg = _cfg(vault_root)
    store = _store(vault_root)
    service = ProfileEnrichmentService(store)
    with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
        run_id = _enrichment_run_id_for_task(store, task_id)
        service.heartbeat(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            claim_token=claim_token, lease_seconds=1800,
        )
        citation, allow_group_voice = _profile_enrichment_voice_task_scope(store, run_id, task_id)

    # Materialization, normalization, and cost estimation happen without the
    # writer. This probe never constructs or calls a local ASR provider.
    prepared = ensure_voice_transcript(
        cfg.root,
        citation=citation,
        allow_local_asr=False,
        allow_cloud_asr=False,
        estimate_cloud_asr_cost=True,
        allow_group_voice=allow_group_voice,
    )
    if prepared.get('status') == 'cached':
        actual_cost = _voice_provider_cost(store, prepared)
        identity = 'voice-cloud-cache-' + hashlib.sha256(
            f"{task_id}:{(prepared.get('transcript') or {}).get('transcript_id') or 'cached'}".encode('utf-8')
        ).hexdigest()[:24]
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            completed = service.complete(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, completion_key=identity, actual_cost_rmb=actual_cost,
            )
            completed = _finalize_terminal_profile(
                store, run_id, actor=actor, session=session, result=completed,
            )
        return prepared | {'profile_enrichment': completed, 'execution_path': 'cloud_asr_cache'}
    if prepared.get('status') in {'media_unavailable', 'snapshot_unavailable'} or prepared.get('reason') in {
        'voice_asset_not_found', 'source_snapshot_unavailable', 'locator_routes_exhausted',
        'unsupported_format', 'decode_failed',
    }:
        terminal = prepared.get('reason') in {'voice_asset_not_found', 'locator_routes_exhausted', 'unsupported_format', 'decode_failed'}
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            failed = service.fail(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, reason=str(prepared.get('reason') or prepared.get('status') or 'media_unavailable'),
                terminal=terminal,
            )
            failed = _finalize_terminal_profile(
                store, run_id, actor=actor, session=session, result=failed,
            )
        return prepared | {'profile_enrichment': failed, 'execution_path': 'no_provider'}
    estimated_cost = prepared.get('estimated_cost_rmb')
    try:
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            scope = service.voice_cloud_scope(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token,
                estimated_cost_rmb=(
                    float(estimated_cost)
                    if type(estimated_cost) in {int, float} and float(estimated_cost) > 0
                    else None
                ),
            )
    except Exception as exc:
        if getattr(exc, 'code', '') != 'enrichment_cost_budget_exhausted':
            raise
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            paused = service.pause_budget(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token,
            )
        return prepared | {'ok': False, 'status': 'paused_budget', 'profile_enrichment': paused, 'execution_path': 'no_provider'}

    payload = cloud_voice_transcript_payload(citation, profile_scope=scope)
    try:
        grant = ApprovalManager(cfg.root).require(
            'voice_cloud_asr', 'cloud_asr_upload', payload,
            approval_id=approval_id, one_step_approval=False,
        )
    except ApprovalRequired as exc:
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            service.awaiting_approval(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, approval_id=exc.record.approval_id,
                approval_scope_hash=canonical_payload_digest(payload),
            )
        raise

    result = execute_cloud_voice_transcript(
        cfg.root,
        citation=citation,
        approval_grant=grant,
        profile_scope=scope,
        allow_group_voice=allow_group_voice,
    )
    if result.get('status') == 'paused_budget':
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            paused = service.pause_budget(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, reason=str(result.get('reason') or 'cloud_cost_budget_required'),
            )
        return result | {'profile_enrichment': paused, 'execution_path': 'no_provider'}
    if result.get('status') in {'cached', 'completed'}:
        actual_cost = _voice_provider_cost(store, result)
        identity = 'voice-cloud-' + hashlib.sha256(
            f"{task_id}:{result.get('job_id') or 'cached'}:{grant.request_hash}".encode('utf-8')
        ).hexdigest()[:24]
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            completed = service.complete(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, completion_key=identity, actual_cost_rmb=actual_cost,
            )
            completed = _finalize_terminal_profile(
                store, run_id, actor=actor, session=session, result=completed,
            )
        return result | {'profile_enrichment': completed, 'execution_path': 'approved_cloud_asr'}
    if result.get('status') == 'in_progress':
        return result | {'execution_path': 'approved_cloud_asr_in_progress'}
    reason = str(
        result.get('error_code') or result.get('reason') or result.get('status')
        or 'cloud_asr_failed'
    )
    retryable = result.get('status') in {'retryable_failure', 'needs_provider', 'superseded'} or reason in {
        'provider_timeout', 'provider_transport_error', 'voice_source_changed',
        'voice_scope_changed', 'provider_credential_resolution_failed',
    }
    with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
        failed = service.fail(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            claim_token=claim_token, reason=reason, terminal=not retryable,
        )
        failed = _finalize_terminal_profile(
            store, run_id, actor=actor, session=session, result=failed,
        )
    return result | {
        'profile_enrichment': failed,
        'execution_path': 'approved_cloud_asr_retryable' if retryable else 'approved_cloud_asr_terminal_gap',
    }


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_appmsg_execute(
    vault_root: str | Path,
    task_id: str,
    *,
    actor: str,
    session: str,
    worker: str,
    claim_token: str,
) -> dict:
    """Recover one exact AppMsg source payload and complete its claimed task."""
    cfg = _cfg(vault_root)
    store = _store(vault_root)
    with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
        service = ProfileEnrichmentService(store)
        run_id = _enrichment_run_id_for_task(store, task_id)
        service.heartbeat(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            claim_token=claim_token, lease_seconds=300,
        )
        citation = _profile_enrichment_task_citation(store, run_id, task_id)

    result = recover_appmsg_payload(cfg.root, citation)
    with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
        if result.get('parse_status') == 'parsed':
            identity = 'appmsg-local-' + hashlib.sha256(
                f"{task_id}:{result.get('source_hash')}:{result.get('parser_version')}".encode('utf-8')
            ).hexdigest()[:24]
            completed = service.complete(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, completion_key=identity,
            )
            completed = _finalize_terminal_profile(
                store, run_id, actor=actor, session=session, result=completed,
            )
            return result | {
                'ok': True,
                'profile_enrichment': completed,
                'execution_path': 'local_appmsg_recovery',
            }
        reason = str(result.get('reason') or result.get('status') or 'appmsg_recovery_failed')
        terminal = reason in {
            'appmsg_message_not_found', 'source_appmsg_not_found', 'malformed_xml', 'unsupported_appmsg',
        } or result.get('status') in {'unsupported', 'reclassified'}
        failed = service.fail(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            claim_token=claim_token, reason=reason, terminal=terminal,
        )
        failed = _finalize_terminal_profile(
            store, run_id, actor=actor, session=session, result=failed,
        )
        return result | {
            'profile_enrichment': failed,
            'execution_path': 'local_appmsg_recovery',
        }


@mutation_entrypoint('media_transcribe')
def media_transcribe_budget(
    vault_root: str | Path,
    *,
    budget: int,
    model_size: str = 'small',
    model_cache: str | Path | None = None,
    device: str = 'auto',
    compute_type: str = 'auto',
    language: str = 'zh',
) -> dict:
    # The pipeline owns two short DB phases and loads the local model between
    # them.  Wrapping it here would accidentally keep the outer writer locked
    # during model download.
    return run_voice_transcription_budget(
        vault_root,
        budget=budget,
        model_size=model_size,
        model_cache=model_cache,
        device=device,
        compute_type=compute_type,
        language=language,
    )


@mutation_entrypoint('media_observe')
def media_observe_budget(
    vault_root: str | Path,
    *,
    budget: int,
    languages: list[str] | None = None,
    caption: bool = False,
    caption_budget: int = 100,
    caption_model_id: str | None = None,
    caption_model_cache: str | Path | None = None,
    include_images: bool = False,
) -> dict:
    from trove_core.media_pipeline import run_image_observation_budget

    return run_image_observation_budget(
        vault_root,
        budget=budget,
        languages=languages,
        caption=caption,
        caption_budget=caption_budget,
        caption_model_id=caption_model_id,
        caption_model_cache=caption_model_cache,
        include_images=include_images,
    )


@mutation_entrypoint('media_observe')
def image_observe_fixture(vault_root: str | Path, image_path: str | Path, *, asset_id: str, citation: str, caption: str = 'fixture image observation') -> dict:
    cfg = _cfg(vault_root)
    return run_image_observation_job(
        MultimodalRepository(_store(vault_root)),
        asset_id=asset_id,
        image_path=Path(image_path),
        provider=FakeVisionProvider(caption=caption),
        citation=citation,
        mutation_context=lambda: coordinated_vault_mutation(cfg, operation='media_observe'),
    )


def _customer_from_conversation(store: SQLiteStore, conversation_id: str) -> str:
    with store.connect() as conn:
        row = conn.execute(
            'SELECT conversation_id,title,type FROM conversations WHERE conversation_id=? ORDER BY account_id LIMIT 1',
            (conversation_id,),
        ).fetchone()
    if row is None:
        return conversation_id
    return row['title'] or row['conversation_id']


def customer_profile(vault_root: str | Path, customer: str | None = None, *, limit: int = 5, conversation_id: str | None = None) -> dict:
    with _read_store(vault_root) as store:
        target = _customer_from_conversation(store, conversation_id) if conversation_id else (customer or '')
        profile = build_customer_profile(store, target, limit=limit)
        try:
            profile['snapshot_status'] = read_profile_snapshot_status(
                store, target, resolved_entity=profile.get('resolved_entity'),
            )
        except Exception as exc:
            profile['snapshot_status'] = {
                'ok': False,
                'completeness_state': 'missing',
                'reason': getattr(exc, 'code', 'customer_not_resolved'),
                'raw_content_included': False,
                'raw_paths_included': False,
            }
    if conversation_id:
        profile['conversation_id'] = conversation_id
    return profile


def person_profile(
    vault_root: str | Path,
    person: str | None = None,
    *,
    evidence_limit: int = 12,
    conversation_id: str | None = None,
) -> dict:
    """Build a comprehensive cited person-and-relationship profile.

    This is a read-only projection.  It never creates enrichment work, starts
    decryption, uploads media, or promotes agent hypotheses automatically.
    """
    with _read_store(vault_root) as store:
        target = _customer_from_conversation(store, conversation_id) if conversation_id else (person or '')
        profile = build_person_profile(store, target, evidence_limit=evidence_limit)
    if conversation_id:
        profile['conversation_id'] = conversation_id
    return profile


@mutation_entrypoint('observation_write')
def person_profile_claims_propose(
    vault_root: str | Path,
    *,
    person: str,
    claims: list[dict],
) -> dict:
    """Persist cited scientific profile hypotheses for explicit human review."""
    with coordinated_vault_mutation(_cfg(vault_root), operation='observation_write'):
        return write_person_profile_claims(_store(vault_root), person, claims)


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_plan(
    vault_root: str | Path,
    customer: str,
    *,
    mode: str = 'complete',
    actor: str,
    session: str,
    item_budget: int | None = None,
    cost_budget_rmb: float = 0.0,
    execution_location: str = 'local',
    processor_identity: str = 'local-agent/default',
    prompt_version: str = 'profile-enrichment/v1',
    purpose: str = 'customer_profile_enrichment',
) -> dict:
    """Create/reuse an explicit, owner-bound enrichment run.

    This is deliberately separate from ``customer_profile`` so reads never
    create work, materialize media, or cause provider egress.
    """
    cfg = _cfg(vault_root)
    # Discovery can scan the complete scoped corpus; keep it read-only and
    # outside the writer. The final manifest insert remains one short commit.
    with _read_store(vault_root) as read_store:
        prepared = ProfileEnrichmentService(read_store).discover(customer, purpose=purpose)
    with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
        store = _store(vault_root)
        manifest = ProfileEnrichmentService(store).plan(
            customer, mode=mode, actor=actor, session=session,
            item_budget=item_budget if item_budget is not None else (500 if mode == 'complete' else 50),
            cost_budget_rmb=cost_budget_rmb, execution_location=execution_location,
            processor_identity=processor_identity, prompt_version=prompt_version,
            purpose=purpose, prepared_discovery=prepared,
        )
        return _finalize_terminal_profile(
            store, str(manifest['run_id']), actor=actor, session=session,
            result=manifest,
        )


def profile_enrichment_status(vault_root: str | Path, run_id: str, *, actor: str, session: str) -> dict:
    with _read_store(vault_root) as store:
        return ProfileEnrichmentService(store).manifest(run_id, actor=actor, session=session)


def _enrichment_run_id_for_task(store: SQLiteStore, task_id: str) -> str:
    with store.connect() as conn:
        row = conn.execute('SELECT run_id FROM profile_enrichment_tasks WHERE task_id=?', (task_id,)).fetchone()
    if row is None:
        from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
        raise ProfileEnrichmentError('enrichment task not found', code='enrichment_task_not_found')
    return str(row['run_id'])


def _profile_enrichment_task_citation(store: SQLiteStore, run_id: str, task_id: str) -> str:
    with store.connect() as conn:
        row = conn.execute(
            'SELECT citation FROM profile_enrichment_tasks WHERE run_id=? AND task_id=?',
            (run_id, task_id),
        ).fetchone()
    if row is None:
        from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
        raise ProfileEnrichmentError('enrichment task not found', code='enrichment_task_not_found')
    return str(row['citation'])


def _profile_enrichment_task_payload(
    store: SQLiteStore,
    run_id: str,
    task_id: str,
) -> dict[str, Any]:
    with store.connect() as conn:
        row = conn.execute(
            'SELECT * FROM profile_enrichment_tasks WHERE run_id=? AND task_id=?',
            (run_id, task_id),
        ).fetchone()
    if row is None:
        from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
        raise ProfileEnrichmentError('enrichment task not found', code='enrichment_task_not_found')
    return ProfileEnrichmentService._task_payload(row)


def _profile_enrichment_next_claimable_task_id(
    store: SQLiteStore,
    run_id: str,
) -> str | None:
    with store.connect() as conn:
        row = conn.execute(
            """SELECT task_id FROM profile_enrichment_tasks
                 WHERE run_id=? AND state IN ('pending','retryable_failure')
                 ORDER BY CASE relevance_reason
                              WHEN 'direct_private_chat' THEN 0
                              WHEN 'direct_private_chat_appmsg' THEN 0
                              WHEN 'contact_authored_moment' THEN 1
                              ELSE 2
                          END,
                          created_at,task_id
                 LIMIT 1""",
            (run_id,),
        ).fetchone()
    return str(row['task_id']) if row is not None else None


def _profile_enrichment_voice_task_scope(
    store: SQLiteStore,
    run_id: str,
    task_id: str,
) -> tuple[str, bool]:
    with store.connect() as conn:
        row = conn.execute(
            """SELECT citation,modality,relevance_reason
                 FROM profile_enrichment_tasks
                WHERE run_id=? AND task_id=?""",
            (run_id, task_id),
        ).fetchone()
    if row is None:
        from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
        raise ProfileEnrichmentError('enrichment task not found', code='enrichment_task_not_found')
    if str(row['modality']) != 'voice':
        from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
        raise ProfileEnrichmentError('task is not a voice task', code='enrichment_task_modality_mismatch')
    return str(row['citation']), str(row['relevance_reason']) == 'contact_group_speech'


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_claim(
    vault_root: str | Path, run_id: str, *, actor: str, session: str, worker: str,
    task_id: str | None = None, lease_seconds: int = 120, execution_location: str = 'local',
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        service = ProfileEnrichmentService(_store(vault_root))
        if task_id is None:
            # Validate run ownership before using the exact indexed task query.
            # A manifest page is a projection, not the claimable task set: runs
            # can contain more than the maximum 500 returned items.
            manifest = service.manifest(run_id, actor=actor, session=session, limit=1)
            task_id = _profile_enrichment_next_claimable_task_id(service.store, run_id)
            if task_id is None:
                return _finalize_terminal_profile(
                    service.store, run_id, actor=actor, session=session,
                    result={
                        'ok': True,
                        'run_id': run_id,
                        'task': None,
                        'state': manifest['state'],
                        'raw_content_included': False,
                    },
                )
        return service.claim(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            lease_seconds=lease_seconds, execution_location=execution_location,
        )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_redeem_media(
    vault_root: str | Path, task_id: str, *, actor: str, session: str,
    worker: str, claim_token: str, media_capability: str,
) -> dict:
    cfg = _cfg(vault_root)
    store = _store(vault_root)
    with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
        run_id = _enrichment_run_id_for_task(store, task_id)
        service = ProfileEnrichmentService(store)
        redeemed = service.redeem_local_media(
            run_id, task_id, actor=actor, session=session, lease_owner=worker, claim_token=claim_token,
            media_capability=media_capability,
        )

    prepared = fetch_media(vault_root, redeemed['citation'])
    if not prepared.get('ok'):
        reason = str(prepared.get('reason') or prepared.get('status') or 'media_unavailable')
        terminal = reason in {
            'image_asset_not_found', 'locator_routes_exhausted', 'local_video_cache_missing', 'unsupported_format',
            'decode_failed', 'media_content_type_invalid', 'materialization_failed',
            'materialized_cache_unavailable', 'outside_vault_media_path',
        }
        with coordinated_vault_mutation(cfg, operation='profile_enrichment'):
            failed = service.fail(
                run_id, task_id, actor=actor, session=session, lease_owner=worker,
                claim_token=claim_token, reason=reason, terminal=terminal,
            )
            manifest = service.manifest(run_id, actor=actor, session=session, limit=1)
            failed_task = _profile_enrichment_task_payload(store, run_id, task_id)
            progress = {
                **failed,
                'run_id': run_id,
                'state': manifest['state'],
                'task': failed_task,
            }
            progress = _finalize_terminal_profile(
                store, run_id, actor=actor, session=session, result=progress,
            )
            return prepared | {
                'run_id': run_id,
                'task_id': task_id,
                'profile_enrichment': progress,
                'next_tool': failed_task['next_tool'],
                'agent_instruction': 'Media resolution failed; the manifest records a terminal gap or retryable repair state.',
            }
    return prepared | {
        'run_id': run_id,
        'task_id': task_id,
        'claim_token': claim_token,
        'next_tool': 'trove_profile_enrichment_image_annotate' if prepared.get('ok') else redeemed['next_tool'],
        'agent_instruction': redeemed['agent_instruction'],
    }


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_image_annotate(
    vault_root: str | Path,
    task_id: str,
    *,
    actor: str,
    session: str,
    worker: str,
    claim_token: str,
    model_id: str,
    prompt_version: str,
    caption: str | None = None,
    visible_text: str | None = None,
    objects=None,
    business_signals=None,
    keyframes=None,
    audio_transcript: str | None = None,
    confidence: float | None = None,
) -> dict:
    cfg = _cfg(vault_root)
    with coordinated_vault_mutation(cfg, operation='profile_enrichment') as active_session:
        store = _store(vault_root)
        service = ProfileEnrichmentService(store)
        run_id = _enrichment_run_id_for_task(store, task_id)
        scope = service.image_annotation_scope(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            claim_token=claim_token,
        )
        if str(model_id) != scope['model_id'] or str(prompt_version) != scope['prompt_version']:
            from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
            raise ProfileEnrichmentError(
                'annotation processor identity does not match the immutable task',
                code='annotation_processor_mismatch',
            )
        annotated = annotate_media_understanding(
            cfg.root,
            citation=scope['citation'],
            caption=caption,
            visible_text=visible_text,
            objects=objects,
            business_signals=business_signals,
            keyframes=keyframes,
            audio_transcript=audio_transcript,
            confidence=confidence,
            model_id=model_id,
            prompt_version=prompt_version,
            expected_content_sha256=scope['content_sha256'],
            expected_asset_id=scope['asset_id'],
            execution_location='local',
            write_session=active_session,
        )
        if not annotated.get('ok'):
            return annotated | {'run_id': run_id, 'task_id': task_id}
        completion_identity = 'image-' + hashlib.sha256(
            f"{task_id}:{scope['content_sha256']}:{model_id}:{prompt_version}".encode('utf-8')
        ).hexdigest()[:24]
        completed = service.complete(
            run_id, task_id, actor=actor, session=session, lease_owner=worker,
            claim_token=claim_token, completion_key=completion_identity,
        )
        completed = _finalize_terminal_profile(
            store, run_id, actor=actor, session=session, result=completed,
        )
        return annotated | {
            'profile_enrichment': completed,
            'execution_path': 'local_agent_vision',
        }


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_heartbeat(
    vault_root: str | Path, task_id: str, *, actor: str, session: str, worker: str,
    claim_token: str, lease_seconds: int = 120,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        store = _store(vault_root)
        run_id = _enrichment_run_id_for_task(store, task_id)
        return ProfileEnrichmentService(store).heartbeat(
            run_id, task_id, actor=actor, session=session, lease_owner=worker, claim_token=claim_token,
            lease_seconds=lease_seconds,
        )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_complete(
    vault_root: str | Path, task_id: str, *, actor: str, session: str, worker: str,
    claim_token: str, processor_identity: str, completion_identity: str,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        store = _store(vault_root)
        run_id = _enrichment_run_id_for_task(store, task_id)
        completed = ProfileEnrichmentService(store).complete(
            run_id, task_id, actor=actor, session=session, lease_owner=worker, claim_token=claim_token,
            completion_key=completion_identity,
        )
        return _finalize_terminal_profile(
            store, run_id, actor=actor, session=session, result=completed,
        )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_fail(
    vault_root: str | Path, task_id: str, *, actor: str, session: str, worker: str,
    claim_token: str, reason: str, retryable: bool = True,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        store = _store(vault_root)
        run_id = _enrichment_run_id_for_task(store, task_id)
        failed = ProfileEnrichmentService(store).fail(
            run_id, task_id, actor=actor, session=session, lease_owner=worker, claim_token=claim_token,
            reason=reason, terminal=not retryable,
        )
        return _finalize_terminal_profile(
            store, run_id, actor=actor, session=session, result=failed,
        )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_resume(
    vault_root: str | Path, run_id: str, *, actor: str, session: str,
    additional_items: int = 500, additional_cost_rmb: float = 0.0,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        return ProfileEnrichmentService(_store(vault_root)).resume_budget(
            run_id, actor=actor, session=session, additional_items=additional_items,
            additional_cost_rmb=additional_cost_rmb,
        )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_revoke(
    vault_root: str | Path, run_id: str, *, actor: str, session: str,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        return ProfileEnrichmentService(_store(vault_root)).revoke(
            run_id, actor=actor, session=session,
        )


@mutation_entrypoint('profile_enrichment')
def profile_enrichment_finalize(
    vault_root: str | Path, run_id: str, *, actor: str, session: str,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='profile_enrichment'):
        return finalize_profile_snapshot(
            _store(vault_root), run_id, actor=actor, session=session,
        )


def profile_snapshot_status(
    vault_root: str | Path, customer: str,
) -> dict:
    with _read_store(vault_root) as store:
        return read_profile_snapshot_status(store, customer)


def profile_snapshot_list(
    vault_root: str | Path,
    customer: str,
    *,
    limit: int = 20,
) -> dict:
    with _read_store(vault_root) as store:
        return read_profile_snapshot_list(store, customer, limit=limit)


def profile_snapshot_get(
    vault_root: str | Path,
    customer: str,
    *,
    version: int | None = None,
) -> dict:
    with _read_store(vault_root) as store:
        return read_profile_snapshot(store, customer, version=version)


def profile_snapshot_diff(
    vault_root: str | Path,
    customer: str,
    *,
    from_version: int,
    to_version: int | None = None,
    max_changes: int = 200,
) -> dict:
    with _read_store(vault_root) as store:
        return read_profile_snapshot_diff(
            store,
            customer,
            from_version=from_version,
            to_version=to_version,
            max_changes=max_changes,
        )


@mutation_entrypoint('profile_automation')
def profile_automation_enable(
    vault_root: str | Path,
    customer: str,
    *,
    debounce_seconds: int = 180,
) -> dict:
    cfg = _cfg(vault_root)
    with coordinated_vault_mutation(cfg, operation='profile_automation'):
        store = _store(vault_root)
        try:
            return ProfileAutomationService(store).enable(
                customer, debounce_seconds=debounce_seconds,
            )
        finally:
            store.close()


@mutation_entrypoint('profile_automation')
def profile_automation_disable(vault_root: str | Path, customer: str) -> dict:
    cfg = _cfg(vault_root)
    with coordinated_vault_mutation(cfg, operation='profile_automation'):
        store = _store(vault_root)
        try:
            return ProfileAutomationService(store).disable(customer)
        finally:
            store.close()


def profile_automation_status(
    vault_root: str | Path,
    customer: str | None = None,
    *,
    limit: int = 100,
) -> dict:
    with _read_store(vault_root) as store:
        return ProfileAutomationService(store).status(customer, limit=limit)


@mutation_entrypoint('profile_automation')
def profile_automation_refresh_now(
    vault_root: str | Path,
    customer: str,
) -> dict:
    cfg = _cfg(vault_root)
    with coordinated_vault_mutation(cfg, operation='profile_automation'):
        store = _store(vault_root)
        try:
            queued = ProfileAutomationService(store).enqueue_customer(customer)
        finally:
            store.close()
    refreshed = process_profile_refresh_queue(
        cfg, limit=1, entity_id=str(queued['entity_id']),
    )
    return {
        'ok': bool(refreshed.get('ok')),
        'type': 'profile_automation_refresh',
        'queue': queued,
        'refresh': refreshed,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


@mutation_entrypoint('profile_automation')
def profile_automation_run_due(
    vault_root: str | Path,
    *,
    limit: int = 10,
) -> dict:
    result = process_profile_refresh_queue(_cfg(vault_root), limit=limit)
    if not result.get('processed') and result.get('status') != 'locked':
        record_vault_mutation_noop(operation='profile_automation')
    return result


@mutation_entrypoint('observation_write')
def observe_add(
    vault_root: str | Path,
    *,
    entity: str,
    observation_type: str = 'operator_note',
    text: str,
    confidence: float = 0.9,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='observation_write'):
        store = _store(vault_root)
        repo = MultimodalRepository(store)
        entity_id = _resolve_entity_id(store, entity, materialize=True)
        obs_id = 'obs-' + uuid.uuid4().hex[:16]
        row = repo.add_observation(ObservationRecord(
            observation_id=obs_id,
            entity_id=entity_id,
            observation_type=observation_type or 'operator_note',
            value={'text': str(text or '')},
            status='active',
            confidence=float(confidence),
            citation=f'trove://operator/{obs_id}',
            source_type='operator',
        ))
    return {'ok': True, 'observation': row, 'raw_content_included': True}


@mutation_entrypoint('observation_write')
def observe_propose(
    vault_root: str | Path,
    *,
    entity: str,
    observation_type: str,
    text: str,
    confidence: float = 0.7,
    citation: str | None = None,
) -> dict:
    with coordinated_vault_mutation(_cfg(vault_root), operation='observation_write'):
        store = _store(vault_root)
        repo = MultimodalRepository(store)
        entity_id = _resolve_entity_id(store, entity, materialize=True)
        obs_id = 'obs-' + uuid.uuid4().hex[:16]
        row = repo.add_observation(ObservationRecord(
            observation_id=obs_id,
            entity_id=entity_id,
            observation_type=observation_type or 'agent_note',
            value={'text': str(text or '')},
            status='needs_review',
            confidence=float(confidence),
            citation=citation or f'trove://agent-proposal/{obs_id}',
            source_type='agent',
        ))
    return {'ok': True, 'observation': row, 'raw_content_included': True}


def observe_list(vault_root: str | Path, *, entity: str, include_retired: bool = False, limit: int = 50) -> dict:
    limit = int(BoundedLimit(limit, field='limit', spec=PRIVATE_LIST))
    statuses = ("'active','needs_review','merge_candidate'" if not include_retired else "'active','needs_review','merge_candidate','superseded','rejected'")
    with _read_store(vault_root) as store:
        entity_id = _resolve_entity_id(store, entity)
        with store.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                f"""SELECT observation_id,entity_id,observation_type,value_json,status,confidence,citation,source_type,updated_at
                    FROM observations
                    WHERE entity_id=? AND status IN ({statuses})
                    ORDER BY CASE WHEN source_type='operator' THEN 0 ELSE 1 END, confidence DESC, updated_at DESC
                    LIMIT ?""",
                (entity_id, limit),
            )]
    return {'ok': True, 'entity_id': entity_id, 'observations': rows, 'raw_content_included': True}


@mutation_entrypoint('observation_write')
def observe_retire(
    vault_root: str | Path,
    *,
    observation_id: str,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    cfg = _cfg(vault_root)
    payload = observation_status_payload(observation_id=observation_id)
    grant = ApprovalManager(cfg.root).require(
        'observe_retire',
        'agent_sensitive_tool',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return execute_observation_status(
        cfg.root,
        observation_id=observation_id,
        action='observe_retire',
        approval_grant=grant,
    )


@mutation_entrypoint('observation_write')
def observe_approve(vault_root: str | Path, *, observation_id: str, approval_id: str | None = None, yes: bool = False) -> dict:
    cfg = _cfg(vault_root)
    payload = observation_status_payload(observation_id=observation_id)
    grant = ApprovalManager(cfg.root).require(
        'observe_approve',
        'agent_sensitive_tool',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return execute_observation_status(
        cfg.root,
        observation_id=observation_id,
        action='observe_approve',
        approval_grant=grant,
    )


def identity_reconcile_plan(vault_root: str | Path, customer: str) -> dict:
    with _read_store(vault_root) as store:
        return reconciliation_plan(store, customer)


@mutation_entrypoint('entity_reconcile')
def identity_reconcile(
    vault_root: str | Path,
    *,
    customer: str,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    cfg = _cfg(vault_root)
    payload = {
        'scope': 'entity_reconcile',
        'input_digest': hashlib.sha256(str(customer or '').encode('utf-8')).hexdigest(),
    }
    ApprovalManager(cfg.root).require(
        'entity_reconcile',
        'destructive_rebuild',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    with coordinated_vault_mutation(cfg, operation='entity_reconcile'):
        return reconcile_customer_entities(_store(cfg.root), customer, apply=True)


def appmsg_repair_plan(
    vault_root: str | Path,
    *,
    source: str | Path,
    limit_per_sqlite: int | None = None,
) -> dict:
    source = _repair_source(source)
    limit_per_sqlite = _repair_limit(limit_per_sqlite, field='limit_per_sqlite')
    return build_appmsg_backfill_plan(
        _cfg(vault_root).root,
        source,
        limit_per_sqlite=limit_per_sqlite,
    )


@mutation_entrypoint('appmsg_backfill')
def appmsg_repair(
    vault_root: str | Path,
    *,
    source: str | Path,
    limit_per_sqlite: int | None = None,
    backup_retention: int = 5,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    source = _repair_source(source)
    limit_per_sqlite = _repair_limit(limit_per_sqlite, field='limit_per_sqlite')
    backup_retention = _repair_retention(backup_retention)
    cfg = _cfg(vault_root)
    payload = appmsg_backfill_payload(
        source,
        limit_per_sqlite=limit_per_sqlite,
        backup_retention=backup_retention,
    )
    grant = ApprovalManager(cfg.root).require(
        'appmsg_backfill',
        'delete_or_purge',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return execute_appmsg_backfill(
        cfg.root,
        source,
        limit_per_sqlite=limit_per_sqlite,
        backup_retention=backup_retention,
        approval_grant=grant,
    )


def message_media_repair_plan(
    vault_root: str | Path,
    *,
    limit: int | None = None,
) -> dict:
    return build_message_media_backfill_plan(
        _cfg(vault_root).root,
        limit=_repair_limit(limit, field='limit'),
    )


@mutation_entrypoint('message_media_backfill')
def message_media_repair(
    vault_root: str | Path,
    *,
    limit: int | None = None,
    backup_retention: int = 5,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    limit = _repair_limit(limit, field='limit')
    backup_retention = _repair_retention(backup_retention)
    cfg = _cfg(vault_root)
    payload = message_media_backfill_payload(limit=limit, backup_retention=backup_retention)
    grant = ApprovalManager(cfg.root).require(
        'message_media_backfill',
        'delete_or_purge',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return execute_message_media_backfill(
        cfg.root,
        limit=limit,
        backup_retention=backup_retention,
        approval_grant=grant,
    )


def scope_status(vault_root: str | Path) -> dict:
    with _read_store(vault_root) as store:
        status = store.scope_status()
    return {'scope_contract': public_scope_contract(), **status}


def list_contacts(vault_root: str | Path, *, limit: int = 100) -> dict:
    return TroveQueries(_cfg(vault_root)).list_contacts(ListQuery(limit=limit)).to_dict()


def list_moments(vault_root: str | Path, *, limit: int = 100) -> dict:
    return TroveQueries(_cfg(vault_root)).list_moments(ListQuery(limit=limit)).to_dict()


def list_favorites(vault_root: str | Path, *, limit: int = 100) -> dict:
    return TroveQueries(_cfg(vault_root)).list_favorites(ListQuery(limit=limit)).to_dict()


def files_list(
    vault_root: str | Path,
    *,
    contact: str | None = None,
    conversation_id: str | None = None,
    file_name: str | None = None,
    media_types: list[str] | str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
) -> dict:
    return TroveQueries(_cfg(vault_root)).list_files(FilesQuery(
            contact=contact,
            conversation_id=conversation_id,
            file_name=file_name,
            media_types=media_types,
            since=since,
            until=until,
            limit=limit,
        )).to_dict()


@mutation_entrypoint('files_archive')
def files_archive(
    vault_root: str | Path,
    *,
    selection: dict | list | str | None,
    dest_dir: str | Path,
    approval_id: str | None,
    mode: str = 'copy',
) -> dict:
    cfg = _cfg(vault_root)
    payload = archive_approval_payload(cfg, selection=selection, dest_dir=dest_dir, mode=mode)
    grant = ApprovalManager(cfg.root).require(
        'files_archive',
        'local-file-export',
        payload,
        approval_id=approval_id,
        one_step_approval=False,
    )
    return execute_files_archive(
        cfg.root,
        selection=selection,
        dest_dir=dest_dir,
        mode=mode,
        approval_grant=grant,
    )


@mutation_entrypoint('media_fetch')
def media_fetch(
    vault_root: str | Path,
    citation: str,
    *,
    allow_remote: bool = False,
    approval_id: str | None = None,
) -> dict:
    prepared = fetch_media(vault_root, citation)
    if not allow_remote or prepared.get('status') != 'awaiting_approval':
        return prepared
    payload = prepared.get('approval_payload')
    if type(payload) is not dict:
        return prepared
    cfg = _cfg(vault_root)
    grant = ApprovalManager(cfg.root).require(
        'wechat_cdn_fetch',
        'remote_media_fetch',
        payload,
        approval_id=approval_id,
        one_step_approval=False,
    )
    from trove_core.application.sensitive_commands import execute_wechat_cdn_fetch
    return execute_wechat_cdn_fetch(
        cfg.root,
        citation=citation,
        approval_payload=payload,
        approval_grant=grant,
    )


@mutation_entrypoint('media_annotate')
def media_annotate(
    vault_root: str | Path,
    *,
    citation: str,
    caption: str | None = None,
    visible_text: str | None = None,
    objects=None,
    business_signals=None,
    keyframes=None,
    audio_transcript: str | None = None,
    confidence: float | None = None,
    model_id: str,
    prompt_version: str,
    replace: bool = False,
) -> dict:
    return annotate_media_understanding(
        vault_root,
        citation=citation,
        caption=caption,
        visible_text=visible_text,
        objects=objects,
        business_signals=business_signals,
        keyframes=keyframes,
        audio_transcript=audio_transcript,
        confidence=confidence,
        model_id=model_id,
        prompt_version=prompt_version,
        replace=replace,
    )


def media_understanding_status_tool(vault_root: str | Path) -> dict:
    # Validate the same schema under the generation lease before the narrow
    # status helper opens its own read-only sqlite3 connection.
    with _read_store(vault_root):
        return media_understanding_status(vault_root)


def media_status(vault_root: str | Path) -> dict:
    # ``media_status_payload`` is intentionally a narrow SQL projection.  The
    # facade supplies the schema preflight and complete-generation boundary.
    with _read_store(vault_root):
        return media_status_payload(vault_root)


@mutation_entrypoint('media_invalidate')
def media_understanding_invalidate(
    vault_root: str | Path,
    *,
    content_sha256: str | None = None,
    model_id: str | None = None,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    cfg = _cfg(vault_root)
    payload = media_invalidation_payload(content_sha256=content_sha256, model_id=model_id)
    grant = ApprovalManager(cfg.root).require(
        'media_understanding_invalidate',
        'delete_or_purge',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return execute_media_understanding_invalidate(
        cfg.root,
        content_sha256=content_sha256,
        model_id=model_id,
        approval_grant=grant,
    )


@mutation_entrypoint('scope_rebuild')
def scope_rebuild(
    vault_root: str | Path,
    *,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    cfg = _cfg(vault_root)
    commands = TroveCommands(cfg)
    grant = ApprovalManager(cfg.root).require(
        'scope_rebuild',
        'destructive_rebuild',
        commands.scope_rebuild_payload(),
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return commands.scope_rebuild(approval_grant=grant)

@mutation_entrypoint('source_manifest_write')
def source_manifest(vault_root: str | Path, sources: list[str]) -> dict:
    cfg = _cfg(vault_root)
    # Filesystem inventory can be slow on a large source tree and does not
    # mutate the Vault. Only publish the finished redacted manifest under the
    # short writer boundary.
    manifest = build_manifest(inventory(sources))
    with coordinated_vault_mutation(cfg, operation='source_manifest_write'):
        cfg.ensure()
        out_path = cfg.paths.manifests_dir / 'source_manifest.redacted.json'
        manifest.write(out_path)
    data = manifest.to_dict()
    data['written'] = 'source_manifest.redacted.json'
    return data


def import_status(vault_root: str | Path) -> dict:
    return read_last_import_status(_cfg(vault_root).root)


def decrypt_preflight(
    vault_root: str | Path,
    *,
    live_root: str | Path,
    selected_accounts: list[str],
    secret_name: str | None = None,
    key_store_path: str | Path | None = None,
    output_source_name: str = 'wechat-integrated-decrypted',
) -> dict:
    cfg = _cfg(vault_root)
    plan = build_decrypt_plan(DecryptConfig(
        live_root=Path(live_root).expanduser(),
        vault_root=cfg.root,
        selected_accounts=selected_accounts_from_strings(selected_accounts, secret_name=secret_name),
        secret_name=secret_name,
        key_store_path=Path(key_store_path).expanduser() if key_store_path else None,
        output_source_name=output_source_name,
    ))
    return plan.to_redacted_dict()


def decrypt_status(vault_root: str | Path, *, run_id: str | None = None, output_source_name: str = 'wechat-integrated-decrypted') -> dict:
    return _decrypt_status(_cfg(vault_root).root, run_id=run_id, output_source_name=output_source_name)


@mutation_entrypoint('sync')
def sync(vault_root: str | Path, *, full: bool = False, since: str | None = None, snapshot_dir: str | Path | None = None) -> dict:
    return TroveCommands(_cfg(vault_root)).sync(
        SyncCommand(full=full, since=since, snapshot_dir=snapshot_dir),
    )


def realtime_sync(vault_root: str | Path, *, config_path: str | Path) -> dict:
    from trove_core.wechat.realtime_bridge import RealtimeBridgeConfig, run_realtime_bridge_once

    return run_realtime_bridge_once(
        _cfg(vault_root).root,
        config=RealtimeBridgeConfig.from_path(config_path),
    )


@mutation_entrypoint('maintain')
def maintain(
    vault_root: str | Path,
    *,
    auto_rebuild: bool = False,
    backup_retention: int = 3,
    log_retention: int = 5,
) -> dict:
    return TroveCommands(_cfg(vault_root)).maintain(
        MaintainCommand(
            auto_rebuild=auto_rebuild,
            backup_retention=backup_retention,
            log_retention=log_retention,
        ),
    )


def schedule_install(
    vault_root: str | Path,
    *,
    sync_interval: str = '1h',
    maintain_at: str = '03:00',
    watch: bool = False,
    realtime_config: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    return install_schedule(
        vault_root,
        options=ScheduleInstallOptions(
            sync_interval=sync_interval,
            maintain_at=maintain_at,
            watch=bool(watch),
            realtime_config=Path(realtime_config).expanduser() if realtime_config is not None else None,
            dry_run=True,
            output_dir=Path(output_dir).expanduser() if output_dir is not None else None,
        ),
    )


@mutation_entrypoint('full_import')
def start_import(
    vault_root: str | Path,
    sources: list[str],
    *,
    reset_index_cache: bool = False,
    limit_per_sqlite: int | None = None,
    force_rescan: bool = False,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    cfg = _cfg(vault_root)
    commands = TroveCommands(cfg)
    command = FullImportCommand(
        tuple(sources),
        reset_index_cache=reset_index_cache,
        limit_per_sqlite=limit_per_sqlite,
        process_config=None,
        force_rescan=force_rescan,
    )
    payload = commands.full_import_payload(command)
    grant = ApprovalManager(cfg.root).require(
        'full_import',
        'full_import',
        payload,
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return commands.full_import(command, approval_grant=grant)


@mutation_entrypoint('reset_index_cache')
def reset_index_cache(
    vault_root: str | Path,
    *,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    cfg = _cfg(vault_root)
    commands = TroveCommands(cfg)
    grant = ApprovalManager(cfg.root).require(
        'reset_index_cache',
        'destructive_rebuild',
        commands.reset_index_cache_payload(),
        approval_id=approval_id,
        one_step_approval=yes,
    )
    return commands.reset_index_cache(approval_grant=grant)


def model_status(model_path: str | None = None) -> dict:
    return embedding_model_status(model_path=model_path)



def provider_status() -> dict:
    return provider_status_payload()


def provider_pricing() -> dict:
    return pricing_payload()


def cloud_readiness(
    vault_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    cost_cap_rmb: float | None = None,
    estimated_cost_rmb: float | None = None,
    doc_verification_date: str | None = None,
    provider_docs_ok: bool = False,
    selected_account_ids: list[str] | None = None,
    discovered_account_ids: list[str] | None = None,
    undecryptable_account_ids: list[str] | None = None,
    coverage_gap_account_ids: list[str] | None = None,
    redaction_probe: str = '',
    require_clean_git: bool = True,
    require_usage_store: bool = True,
) -> dict:
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    report = check_cloud_processing_readiness(CloudReadinessInput(
        repo_root=root,
        vault_root=_cfg(vault_root).root,
        cost_cap_rmb=cost_cap_rmb,
        estimated_cost_rmb=estimated_cost_rmb,
        doc_verification_date=doc_verification_date,
        provider_docs_ok=provider_docs_ok,
        selected_account_ids=selected_account_ids or [],
        discovered_account_ids=discovered_account_ids or [],
        undecryptable_account_ids=undecryptable_account_ids or [],
        coverage_gap_account_ids=coverage_gap_account_ids or [],
        redaction_probe=redaction_probe,
        require_clean_git=require_clean_git,
        require_usage_store=require_usage_store,
    ))
    return report.to_dict()

def vector_status(
    vault_root: str | Path,
    *,
    backend: str = 'zvec',
    provider=None,
) -> dict:
    cfg = _cfg(vault_root)
    if provider is None:
        try:
            # The persisted Vault policy selects the active vector score
            # domain. Agent/MCP status must inspect that selected cloud
            # collection instead of silently falling back to the local-model
            # identity and reporting a false missing-collection alarm.
            provider = configured_embedding_provider(vault_root=cfg.root)
        except Exception:
            provider = None
    with vault_generation_read(cfg):
        return vector_status_payload(cfg, backend=backend, provider=provider)


@mutation_entrypoint('vector_index')
def vector_index(
    vault_root: str | Path,
    model_path: str | None = None,
    *,
    backend: str = 'zvec',
    batch_size: int = 256,
    max_messages: int | None = None,
    approval_id: str | None = None,
    yes: bool = False,
) -> dict:
    commands = TroveCommands(_cfg(vault_root))
    prepared = commands.prepare_vector(VectorCommand(
        action='index',
        model_path=model_path,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
    ))
    grant = None
    if prepared.requires_approval:
        grant = ApprovalManager(commands.config.root).require(
            prepared.approval_action or '',
            prepared.approval_danger_class or '',
            prepared.approval_payload or {},
            approval_id=approval_id,
            one_step_approval=yes,
        )
    return commands.vector(prepared, approval_grant=grant)


def search(vault_root: str | Path, query: str, *, runtime=None, **filters) -> dict:
    return TroveQueries(_cfg(vault_root), runtime=runtime).search(SearchQuery(query, **filters)).to_dict()


def fetch_context(vault_root: str | Path, citation: str, before: int = 5, after: int = 5) -> dict:
    return TroveQueries(_cfg(vault_root)).context(
        ContextQuery(citation, before=before, after=after),
    ).to_dict()


def fetch_conversation_context(vault_root: str | Path, conversation_id: str, *, limit: int = 20) -> dict:
    return TroveQueries(_cfg(vault_root)).conversation_context(
        ConversationContextQuery(conversation_id, limit=limit),
    ).to_dict()


def list_conversations(vault_root: str | Path, *, limit: int = 100) -> list[dict]:
    result = TroveQueries(_cfg(vault_root)).list_conversations(ListQuery(limit=limit))
    return list(result.data.get('conversations', []))


def customer_card(vault_root: str | Path, customer: str) -> dict:
    with _read_store(vault_root) as store:
        return build_customer_card(store, customer)


def conversation_card(vault_root: str | Path, account_id: str, conversation_id: str) -> dict:
    with _read_store(vault_root) as store:
        return build_conversation_card(store, account_id, conversation_id)


def cited_report(vault_root: str | Path, query: str, limit: int = 5) -> dict:
    with _read_store(vault_root) as store:
        return build_cited_report(store, query, limit=limit)


def process_config_preview(vault_root: str | Path, payload: dict | None = None, *, write: bool = False) -> dict:
    cfg = _cfg(vault_root)
    pcfg = process_config_from_payload(payload or {})
    errors = pcfg.validate()
    data = {'ok': not errors, 'errors': errors, 'config': pcfg.to_dict(), 'written': None}
    if write and not errors:
        data['written'] = write_process_config(cfg.root, pcfg).name
    return data


def latest_process_config(vault_root: str | Path) -> dict:
    return read_latest_process_config(_cfg(vault_root).root)


def trace_timeline(vault_root: str | Path, *, limit: int = 100) -> dict:
    return {'events': TraceTimeline(_cfg(vault_root).root).list(limit=limit)}


def request_approval(vault_root: str | Path, action: str, danger_class: str, payload: dict | None = None) -> dict:
    return {'approval': ApprovalManager(_cfg(vault_root).root).request(action, danger_class, payload or {}).to_dict()}


def decide_approval(vault_root: str | Path, approval_id: str, status: str, note: str | None = None) -> dict:
    return {'approval': ApprovalManager(_cfg(vault_root).root).decide(approval_id, status, note=note).to_dict()}


def list_approvals(vault_root: str | Path, *, limit: int = 50) -> dict:
    return {'approvals': ApprovalManager(_cfg(vault_root).root).list(limit=limit)}


@mutation_entrypoint('rebuild_chunks')
def rebuild_chunks(vault_root: str | Path, *, max_chars: int = 900, overlap_chars: int = 120) -> dict:
    return rebuild_vault_chunks(
        _cfg(vault_root).root,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


@mutation_entrypoint('rebuild_fts')
def rebuild_fts(vault_root: str | Path) -> dict:
    return rebuild_vault_fts(_cfg(vault_root).root)


def evaluate(vault_root: str | Path, golden: str | Path = 'tests/golden/search_queries.jsonl', *, k: int = 3) -> dict:
    cfg = _cfg(vault_root)
    with vault_generation_read(cfg):
        report = evaluate_golden(build_search_engine(cfg), Path(golden), k=k)
    TraceTimeline(cfg.root).append('evaluation', 'complete', {'queries': report.get('queries')})
    return report


def wiki_page(vault_root: str | Path, title: str, *, limit: int = 8, write: bool = False) -> dict:
    cfg = _cfg(vault_root)
    with _read_store(vault_root) as store:
        page = build_wiki_page(store, title, limit=limit)
    if write:
        page['written'] = write_wiki_page(cfg.root, title, page).name
    return page
