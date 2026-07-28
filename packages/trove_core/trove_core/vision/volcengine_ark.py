from __future__ import annotations

from pathlib import Path
import base64
import json
import mimetypes
import urllib.request
from typing import Any, Callable

from trove_core.providers.config import DEFAULT_ARK_BASE_URL, DEFAULT_ARK_RESPONSES_PATH, DEFAULT_ARK_VISION_MODEL
from trove_core.providers.pricing import ArkVisionLitePricing
from .base import ImageObservationResult, VisionProvider, VisionRequest, VisionUsage
from .structured_output import parse_structured_vision_text


PROVIDER_REQUEST_RESPONSE_LOGGING = False
PROVIDER_RETENTION_POLICY = 'external provider retention is not controlled by TROVE; TROVE stores no request or response body'


def _usage_from(payload: dict[str, Any]) -> VisionUsage:
    usage = payload.get('usage') or {}
    details = usage.get('input_tokens_details') or {}
    input_tokens = int(usage.get('input_tokens') or 0)
    cached = int(details.get('cached_tokens') or 0)
    output_tokens = int(usage.get('output_tokens') or 0)
    cost = ArkVisionLitePricing().estimate_rmb(input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens)
    return VisionUsage(input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens, estimated_cost_rmb=cost)


def _output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get('output_text'), str):
        return payload['output_text']
    parts: list[str] = []
    for item in payload.get('output') or []:
        if not isinstance(item, dict):
            continue
        for content in item.get('content') or []:
            if isinstance(content, dict) and isinstance(content.get('text'), str):
                parts.append(content['text'])
    return '\n'.join(parts)


class VolcengineArkVisionProvider(VisionProvider):
    name = 'volcengine-ark-vision-lite'
    egress_kind = 'cloud_vision_upload'
    model = DEFAULT_ARK_VISION_MODEL
    request_response_logging = PROVIDER_REQUEST_RESPONSE_LOGGING
    retention_policy = PROVIDER_RETENTION_POLICY

    def __init__(self, *, api_key: str | None = None, base_url: str = DEFAULT_ARK_BASE_URL, responses_path: str = DEFAULT_ARK_RESPONSES_PATH, urlopen: Callable | None = None, timeout: int = 60):
        if not api_key:
            raise RuntimeError('Vision credential must be supplied explicitly by ProviderFactory.')
        self.api_key = api_key
        self.endpoint = base_url.rstrip('/') + responses_path
        self.urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout

    def _image_url(self, request: VisionRequest) -> str:
        if request.image_url:
            return request.image_url
        if not request.image_path:
            raise ValueError('Vision request requires image_path or image_url')
        data = Path(request.image_path).read_bytes()
        mime = mimetypes.guess_type(str(request.image_path))[0] or 'image/jpeg'
        return f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')

    def build_payload(self, request: VisionRequest) -> dict[str, Any]:
        prompt = request.prompt + '\nReturn JSON with keys: caption, visible_text, objects, business_signals, entity_mentions, confidence.'
        return {
            'model': self.model,
            'input': [{
                'role': 'user',
                'content': [
                    {'type': 'input_image', 'image_url': self._image_url(request)},
                    {'type': 'input_text', 'text': prompt},
                ],
            }],
        }

    def build_headers(self) -> dict[str, str]:
        return {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}

    def observe(self, request: VisionRequest) -> ImageObservationResult:
        payload = self.build_payload(request)
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode('utf-8'), headers=self.build_headers(), method='POST')
        with self.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode('utf-8')
        parsed = json.loads(body) if body else {}
        structured = parse_structured_vision_text(_output_text(parsed))
        usage = _usage_from(parsed)
        return ImageObservationResult(
            caption=structured['caption'],
            visible_text=structured['visible_text'],
            objects=[str(x) for x in structured['objects']],
            business_signals=[str(x) for x in structured['business_signals']],
            entity_mentions=[str(x) for x in structured['entity_mentions']],
            confidence=structured['confidence'],
            usage=usage,
            citations=[request.citation] if request.citation else [],
            raw_provider_payload_stored=False,
            metadata={'provider_payload_included': False},
        )
