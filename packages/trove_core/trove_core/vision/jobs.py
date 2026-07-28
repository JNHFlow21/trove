from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import hashlib
from typing import Callable, ContextManager

from trove_core.approvals import ApprovalGrant, ApprovalValidationError, require_claimed_approval_grant
from trove_core.security.egress import cloud_vision_payload
from trove_core.store.repositories import ImageObservationRecord, MultimodalRepository, ProviderJobRecord
from .base import VisionProvider, VisionRequest


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def run_image_observation_job(
    repo: MultimodalRepository,
    *,
    asset_id: str,
    image_path: Path,
    provider: VisionProvider,
    citation: str,
    min_active_confidence: float = 0.7,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict | None = None,
    mutation_context: Callable[[], ContextManager] | None = None,
) -> dict:
    def mutation():
        return mutation_context() if mutation_context is not None else nullcontext()

    egress_kind = getattr(provider, 'egress_kind', None)
    if egress_kind is not None and (type(egress_kind) is not str or egress_kind != 'cloud_vision_upload'):
        raise RuntimeError('unsupported_vision_egress_kind')
    if egress_kind == 'cloud_vision_upload':
        expected_payload = cloud_vision_payload(
            citation=citation,
            provider=provider.name,
            model=provider.model,
            endpoint=str(getattr(provider, 'endpoint', '')),
        )
        if type(approval_payload) is not dict or approval_payload != expected_payload:
            raise ApprovalValidationError(
                'cloud vision approval payload does not match the outbound request',
                code='grant_payload_mismatch',
            )
        vault_root = Path(repo.store.path).resolve().parent.parent
        require_claimed_approval_grant(
            approval_grant,  # type: ignore[arg-type]
            vault_root,
            action='image_cloud_vision',
            danger_class='cloud_vision_upload',
            payload=expected_payload,
        )
    job_id = _stable('job', f'vision:{provider.name}:{asset_id}')
    observation_id = _stable('imageobs', f'{asset_id}:{provider.name}')
    with repo.store.connect() as conn:
        existing = conn.execute('SELECT status FROM provider_jobs WHERE job_id=?', (job_id,)).fetchone()
        obs = conn.execute(
            "SELECT observation_id FROM image_observations WHERE asset_id=? AND (TRIM(COALESCE(caption, '')) <> '' OR TRIM(COALESCE(visible_text, '')) <> '') LIMIT 1",
            (asset_id,),
        ).fetchone()
        if existing and existing['status'] == 'completed' and obs:
            return {'status': 'completed', 'job_id': job_id, 'observation_id': obs['observation_id'], 'idempotent': True}
    with mutation():
        repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision', status='running', citation=citation))
    try:
        result = provider.observe(VisionRequest(asset_id=asset_id, image_path=Path(image_path), citation=citation))
    except ValueError:
        with mutation():
            repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision', status='needs_review', error_code='malformed_provider_json', citation=citation))
        return {'status': 'needs_review', 'job_id': job_id, 'error_code': 'malformed_provider_json'}
    except TimeoutError:
        with mutation():
            repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision', status='retryable_failure', error_code='provider_timeout', citation=citation))
        return {'status': 'retryable_failure', 'job_id': job_id, 'error_code': 'provider_timeout'}
    except Exception:
        with mutation():
            repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision', status='terminal_failure', error_code='provider_rejected_or_failed', citation=citation))
        return {'status': 'terminal_failure', 'job_id': job_id, 'error_code': 'provider_rejected_or_failed'}
    status = 'active' if result.confidence >= min_active_confidence else 'needs_review'
    with mutation():
        repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision', status='completed', usage=result.usage.to_dict(), cost_rmb=result.usage.estimated_cost_rmb, citation=citation))
        row = repo.merge_image_observation(ImageObservationRecord(
            observation_id=observation_id,
            asset_id=asset_id,
            job_id=job_id,
            citation=f'{citation}#image',
            caption=result.caption,
            visible_text=result.visible_text or '',
            objects=[{'label': value} for value in result.objects],
            business_signals=[{'text': value} for value in result.business_signals],
            confidence=result.confidence,
            status=status,
        ))
    return {'status': 'completed', 'job_id': job_id, 'observation_id': row['observation_id'], 'observation_status': row['status'], 'estimated_cost_rmb': result.usage.estimated_cost_rmb, 'idempotent': False}
