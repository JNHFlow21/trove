from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import hashlib
from typing import Callable, ContextManager

from trove_core.local_vlm.base import ImageCaptionRequest, LocalVLMCaptionProvider
from trove_core.store.repositories import MultimodalRepository, ProviderJobRecord


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def run_image_caption_job(
    repo: MultimodalRepository,
    *,
    asset_id: str,
    image_path: Path,
    provider: LocalVLMCaptionProvider,
    citation: str,
    min_active_confidence: float = 0.7,
    mutation_context: Callable[[], ContextManager] | None = None,
) -> dict:
    def mutation():
        return mutation_context() if mutation_context is not None else nullcontext()

    job_id = _stable('job', f'caption:{provider.name}:{asset_id}')
    with repo.store.connect() as conn:
        existing_caption = conn.execute(
            "SELECT observation_id FROM image_observations WHERE asset_id=? AND TRIM(COALESCE(caption, '')) <> '' LIMIT 1",
            (asset_id,),
        ).fetchone()
        if existing_caption:
            return {'status': 'completed', 'job_id': job_id, 'observation_id': existing_caption['observation_id'], 'idempotent': True}
    with mutation():
        repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision_caption', status='running', citation=citation))
    try:
        result = provider.caption(ImageCaptionRequest(asset_id=asset_id, image_path=Path(image_path), citation=citation))
    except ValueError:
        with mutation():
            repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision_caption', status='needs_review', error_code='malformed_caption_output', citation=citation))
        return {'status': 'needs_review', 'job_id': job_id, 'error_code': 'malformed_caption_output'}
    except TimeoutError:
        with mutation():
            repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision_caption', status='retryable_failure', error_code='provider_timeout', citation=citation))
        return {'status': 'retryable_failure', 'job_id': job_id, 'error_code': 'provider_timeout'}
    except Exception:
        with mutation():
            repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision_caption', status='terminal_failure', error_code='provider_rejected_or_failed', citation=citation))
        return {'status': 'terminal_failure', 'job_id': job_id, 'error_code': 'provider_rejected_or_failed'}
    status = 'active' if result.confidence >= min_active_confidence else 'needs_review'
    with mutation():
        repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=provider.model, job_type='vision_caption', status='completed', usage=result.usage.to_dict(), cost_rmb=result.usage.estimated_cost_rmb, citation=citation))
        row = repo.update_image_caption(
            asset_id=asset_id,
            citation=f'{citation}#image',
            caption=result.caption,
            labels=result.labels,
            confidence=result.confidence,
            status=status,
            job_id=job_id,
        )
    return {'status': 'completed', 'job_id': job_id, 'observation_id': row['observation_id'], 'observation_status': row['status'], 'estimated_cost_rmb': result.usage.estimated_cost_rmb, 'idempotent': False}
