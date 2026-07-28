from __future__ import annotations

from pathlib import Path
import base64
import json
import uuid
import urllib.request
from typing import Any, Callable

from trove_core.providers.config import DEFAULT_ASR_ENDPOINT, DEFAULT_ASR_MODEL_NAME, DEFAULT_ASR_RESOURCE_ID
from trove_core.providers.pricing import estimate_asr_flash_rmb
from .base import ASRProvider, ASRRequest, ASRResult, ASRUsage


PROVIDER_REQUEST_RESPONSE_LOGGING = False
PROVIDER_RETENTION_POLICY = 'external provider retention is not controlled by TROVE; TROVE stores no request or response body'


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = [
        payload.get('text'),
        (payload.get('result') or {}).get('text') if isinstance(payload.get('result'), dict) else None,
        (payload.get('result') or {}).get('utterances') if isinstance(payload.get('result'), dict) else None,
        payload.get('utterances'),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get('text'), str):
                    parts.append(item['text'])
            if parts:
                return '\n'.join(parts).strip()
    return ''


def _extract_duration(payload: dict[str, Any]) -> float:
    for path in [('audio_info', 'duration'), ('result', 'additions', 'duration'), ('additions', 'duration'), ('usage', 'duration_seconds')]:
        cur: Any = payload
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None; break
            cur = cur[key]
        if cur is not None:
            try:
                value = float(cur)
                return value / 1000.0 if value > 600 else value
            except Exception:
                pass
    return 0.0


class VolcengineASRFlashProvider(ASRProvider):
    name = 'volcengine-asr-flash'
    egress_kind = 'cloud_asr_upload'
    model_name = DEFAULT_ASR_MODEL_NAME
    resource_id = DEFAULT_ASR_RESOURCE_ID
    request_response_logging = PROVIDER_REQUEST_RESPONSE_LOGGING
    retention_policy = PROVIDER_RETENTION_POLICY

    def __init__(self, *, api_key: str | None = None, endpoint: str = DEFAULT_ASR_ENDPOINT, urlopen: Callable | None = None, timeout: int = 60):
        if not api_key:
            raise RuntimeError('ASR credential must be supplied explicitly by ProviderFactory.')
        self.api_key = api_key
        self.endpoint = endpoint
        self.urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout

    def build_payload(self, request: ASRRequest) -> dict[str, Any]:
        audio: dict[str, Any] = {}
        if request.audio_url:
            audio['url'] = request.audio_url
        elif request.audio_path:
            data = Path(request.audio_path).read_bytes()
            audio['data'] = base64.b64encode(data).decode('ascii')
            audio['format'] = Path(request.audio_path).suffix.lower().lstrip('.') or 'wav'
        else:
            raise ValueError('ASR request requires audio_path or audio_url')
        return {'user': {'uid': 'trove'}, 'audio': audio, 'request': {'model_name': self.model_name}}

    def build_headers(self) -> dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key,
            'X-Api-Resource-Id': self.resource_id,
            'X-Api-Request-Id': str(uuid.uuid4()),
            'X-Api-Sequence': '-1',
        }

    def transcribe(self, request: ASRRequest) -> ASRResult:
        payload = self.build_payload(request)
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode('utf-8'), headers=self.build_headers(), method='POST')
        with self.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode('utf-8')
        parsed = json.loads(body) if body else {}
        text = _extract_text(parsed)
        status = str(parsed.get('status') or parsed.get('code') or 'completed')
        normalized_status = status.strip().lower()
        if not text or any(token in normalized_status for token in ('error', 'fail', 'reject')):
            raise RuntimeError('cloud_asr_invalid_response')
        duration = _extract_duration(parsed)
        usage = ASRUsage(duration_seconds=duration, estimated_cost_rmb=estimate_asr_flash_rmb(duration))
        return ASRResult(text=text, language=parsed.get('language'), confidence=None, usage=usage, citations=[request.citation] if request.citation else [], provider_status=status)
