from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import urllib.parse
import urllib.request
from typing import Any

OFFICIAL_DOCS = {
    'asr_flash_api': ('https://www.volcengine.com/docs/6561/1631584?lang=zh', 6561, 1631584),
    'asr_billing': ('https://www.volcengine.com/docs/6561/1359370?lang=zh', 6561, 1359370),
    'ark_api_key': ('https://www.volcengine.com/docs/82379/1541594?lang=zh', 82379, 1541594),
    'ark_image_understanding': ('https://www.volcengine.com/docs/82379/1362931?lang=zh', 82379, 1362931),
    'ark_model_list': ('https://www.volcengine.com/docs/82379/1330310?lang=zh', 82379, 1330310),
    'ark_pricing': ('https://www.volcengine.com/docs/82379/1544106?lang=zh', 82379, 1544106),
    'ark_response_object': ('https://www.volcengine.com/docs/82379/1783703?lang=zh', 82379, 1783703),
}


@dataclass(frozen=True)
class ProviderDocVerification:
    verified_at: str
    verification_date: str
    official_sources: list[str]
    doc_updates: dict[str, str | None]
    confirmed: dict[str, bool]
    ambiguities: list[str]

    @property
    def ok(self) -> bool:
        return all(self.confirmed.values())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['ok'] = self.ok
        return data


def _flatten_content(content: str) -> str:
    if not content:
        return ''
    if not content.lstrip().startswith('{'):
        return content
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return content
    out: list[str] = []

    def walk(value):
        if isinstance(value, dict):
            insert = value.get('insert')
            if isinstance(insert, str):
                out.append(insert)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return ''.join(out)


def _fetch_doc(lib_id: int, doc_id: int, timeout: int = 20) -> tuple[dict[str, Any], str]:
    query = urllib.parse.urlencode({'LibraryID': lib_id, 'DocumentID': doc_id, 'type': 'online', 'lang': 'zh'})
    url = f'https://www.volcengine.com/api/doc/getDocDetail?{query}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    result = payload.get('Result') or {}
    return result, _flatten_content(result.get('Content') or '')


def verify_volcengine_official_docs(timeout: int = 20) -> ProviderDocVerification:
    texts: dict[str, str] = {}
    updates: dict[str, str | None] = {}
    sources: list[str] = []
    for key, (source_url, lib_id, doc_id) in OFFICIAL_DOCS.items():
        result, text = _fetch_doc(lib_id, doc_id, timeout=timeout)
        texts[key] = text
        sources.append(source_url)
        updates[key] = result.get('UpdatedTime') or result.get('LastModifyTime') or result.get('FirstPublishedTime')
    asr_api = texts['asr_flash_api']
    asr_billing = texts['asr_billing']
    ark_image = texts['ark_image_understanding']
    ark_key = texts['ark_api_key']
    ark_models = texts['ark_model_list']
    ark_pricing = texts['ark_pricing']
    ark_response = texts['ark_response_object']
    confirmed = {
        'asr_endpoint': 'https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash' in asr_api,
        'asr_model_name_bigmodel': 'model_name' in asr_api and 'bigmodel' in asr_api,
        'asr_resource_id': 'volc.bigasr.auc_turbo' in asr_api,
        'asr_auth_header': 'X-Api-Key' in asr_api and 'X-Api-Resource-Id' in asr_api,
        'asr_duration_usage': 'audio_info' in asr_api and 'duration' in asr_api,
        'asr_postpaid_rate': '大模型录音文件识别（极速版） | 不限 | 4.5元/小时' in asr_billing or '大模型录音文件识别（极速版）' in asr_billing and '4.5元/小时' in asr_billing,
        'ark_api_key_env_guidance': 'API Key' in ark_key and '环境变量' in ark_key,
        'ark_responses_endpoint': '/responses' in ark_image or 'Responses API' in ark_image,
        'ark_bearer_auth': 'Authorization: Bearer' in ark_image,
        'ark_image_input_shape': 'input_image' in ark_image and 'input_text' in ark_image,
        'ark_model_id': 'doubao-seed-2-0-lite-260215' in ark_image and 'doubao-seed-2-0-lite-260215' in ark_models,
        'ark_usage_fields': ('input_tokens' in ark_image or 'input_tokens' in ark_response) and ('output_tokens' in ark_image or 'output_tokens' in ark_response) and ('total_tokens' in ark_image or 'total_tokens' in ark_response),
        'ark_pricing_family': 'doubao-seed-2.0-lite' in ark_pricing and '3.6' in ark_pricing and '0.6' in ark_pricing and '0.12' in ark_pricing,
    }
    ambiguities = [
        'ASR Flash official API docs expose duration fields, not an explicit usage/cost object.',
        'Ark pricing is documented at doubao-seed-2.0-lite family level; image inputs are billed as tokens.',
        'doubao-seed-2-0-lite-260215 is confirmed but newer lite model IDs may also be listed.',
    ]
    now = datetime.now(timezone.utc)
    return ProviderDocVerification(
        verified_at=now.isoformat().replace('+00:00', 'Z'),
        verification_date=now.date().isoformat(),
        official_sources=sources,
        doc_updates=updates,
        confirmed=confirmed,
        ambiguities=ambiguities,
    )
