from __future__ import annotations

from pathlib import Path
import hashlib

from trove_core.approvals import ApprovalGrant, ApprovalValidationError, require_claimed_approval_grant
from trove_core.security.egress import cloud_asr_payload
from trove_core.store.repositories import MultimodalRepository, ProviderJobRecord, TranscriptRecord
from .base import ASRProvider, ASRRequest


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def run_voice_transcript_job(
    repo: MultimodalRepository,
    *,
    asset_id: str,
    audio_path: Path,
    provider: ASRProvider,
    citation: str,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict | None = None,
) -> dict:
    egress_kind = getattr(provider, 'egress_kind', None)
    if egress_kind is not None and (type(egress_kind) is not str or egress_kind != 'cloud_asr_upload'):
        raise RuntimeError('unsupported_asr_egress_kind')
    if egress_kind == 'cloud_asr_upload':
        expected_payload = cloud_asr_payload(
            citation=citation,
            provider=provider.name,
            model=provider.model_name,
            resource_id=provider.resource_id,
            endpoint=str(getattr(provider, 'endpoint', '')),
        )
        if type(approval_payload) is not dict or any(approval_payload.get(key) != value for key, value in expected_payload.items()):
            raise ApprovalValidationError(
                'cloud ASR approval payload does not match the outbound request',
                code='grant_payload_mismatch',
            )
        vault_root = Path(repo.store.path).resolve().parent.parent
        require_claimed_approval_grant(
            approval_grant,  # type: ignore[arg-type]
            vault_root,
            action='voice_cloud_asr',
            danger_class='cloud_asr_upload',
            payload=approval_payload,
        )
    job_id = _stable('job', f'asr:{provider.name}:{asset_id}')
    transcript_id = _stable('transcript', f'{asset_id}:{provider.name}')
    with repo.store.connect() as conn:
        existing = conn.execute('SELECT status FROM provider_jobs WHERE job_id=?', (job_id,)).fetchone()
        transcript = conn.execute('SELECT transcript_id FROM transcripts WHERE transcript_id=?', (transcript_id,)).fetchone()
        if existing and existing['status'] == 'completed' and transcript:
            return {'status': 'completed', 'job_id': job_id, 'transcript_id': transcript_id, 'idempotent': True}
    repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=f'{provider.model_name}:{provider.resource_id}', job_type='asr', status='running', citation=citation))
    try:
        result = provider.transcribe(ASRRequest(asset_id=asset_id, audio_path=Path(audio_path), citation=citation))
    except TimeoutError:
        repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=f'{provider.model_name}:{provider.resource_id}', job_type='asr', status='retryable_failure', error_code='provider_timeout', citation=citation))
        return {'status': 'retryable_failure', 'job_id': job_id, 'error_code': 'provider_timeout'}
    except Exception:
        repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=f'{provider.model_name}:{provider.resource_id}', job_type='asr', status='terminal_failure', error_code='provider_rejected_or_failed', citation=citation))
        return {'status': 'terminal_failure', 'job_id': job_id, 'error_code': 'provider_rejected_or_failed'}
    repo.record_provider_job(ProviderJobRecord(job_id=job_id, asset_id=asset_id, provider=provider.name, model=f'{provider.model_name}:{provider.resource_id}', job_type='asr', status='completed', usage=result.usage.to_dict(), cost_rmb=result.usage.estimated_cost_rmb, citation=citation))
    repo.insert_transcript(TranscriptRecord(transcript_id=transcript_id, asset_id=asset_id, job_id=job_id, citation=f'{citation}#voice', text=result.text, language=result.language, confidence=result.confidence or 0.0, duration_seconds=result.usage.duration_seconds))
    return {'status': 'completed', 'job_id': job_id, 'transcript_id': transcript_id, 'idempotent': False, 'estimated_cost_rmb': result.usage.estimated_cost_rmb}
