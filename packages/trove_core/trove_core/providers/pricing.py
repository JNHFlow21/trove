from __future__ import annotations

from dataclasses import asdict, dataclass

ASR_FLASH_RMB_PER_HOUR = 4.5
ALIYUN_TEXT_EMBEDDING_V4_RMB_PER_MILLION_TOKENS = 0.5
ALIYUN_QWEN3_RERANK_RMB_PER_MILLION_TOKENS = 0.5
VOLCENGINE_TEXT_EMBEDDING_RMB_PER_MILLION_TOKENS = 0.5
VOLCENGINE_RERANK_RMB_PER_MILLION_TOKENS = 0.5


@dataclass(frozen=True)
class ArkVisionLitePricing:
    input_rmb_per_million_tokens: float = 0.6
    cached_input_rmb_per_million_tokens: float = 0.12
    output_rmb_per_million_tokens: float = 3.6

    def estimate_rmb(self, *, input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0) -> float:
        cost = 0.0
        cost += max(input_tokens, 0) / 1_000_000 * self.input_rmb_per_million_tokens
        cost += max(cached_input_tokens, 0) / 1_000_000 * self.cached_input_rmb_per_million_tokens
        cost += max(output_tokens, 0) / 1_000_000 * self.output_rmb_per_million_tokens
        return round(cost, 6)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_asr_flash_rmb(duration_seconds: float, *, rmb_per_hour: float = ASR_FLASH_RMB_PER_HOUR) -> float:
    seconds = max(float(duration_seconds), 0.0)
    return round(seconds / 3600.0 * rmb_per_hour, 6)


def estimate_token_rmb(tokens: int, *, rmb_per_million_tokens: float) -> float:
    return round(max(int(tokens), 0) / 1_000_000 * rmb_per_million_tokens, 6)


def pricing_payload() -> dict:
    ark = ArkVisionLitePricing()
    return {
        'asr_flash': {
            'currency': 'RMB',
            'rmb_per_audio_hour': ASR_FLASH_RMB_PER_HOUR,
            'model_name': 'bigmodel',
            'resource_id': 'volc.bigasr.auc_turbo',
        },
        'ark_vision_lite': ark.to_dict() | {'currency': 'RMB', 'model': 'doubao-seed-2-0-lite-260215'},
        'retrieval': {
            'currency': 'RMB',
            'disabled_by_default': True,
            'aliyun_text_embedding_v4': {
                'rmb_per_million_tokens': ALIYUN_TEXT_EMBEDDING_V4_RMB_PER_MILLION_TOKENS,
                'default_dimensions': 1024,
                'source': 'https://help.aliyun.com/zh/model-studio/model-pricing',
            },
            'aliyun_qwen3_rerank': {
                'rmb_per_million_tokens': ALIYUN_QWEN3_RERANK_RMB_PER_MILLION_TOKENS,
                'default_candidate_window': 50,
                'source': 'https://help.aliyun.com/zh/model-studio/model-pricing',
            },
            'volcengine_text_embedding': {
                'rmb_per_million_tokens': VOLCENGINE_TEXT_EMBEDDING_RMB_PER_MILLION_TOKENS,
                'source': 'https://www.volcengine.com/docs/82379/1544106',
            },
            'volcengine_rerank': {
                'rmb_per_million_tokens': VOLCENGINE_RERANK_RMB_PER_MILLION_TOKENS,
                'source': 'https://www.volcengine.com/docs/82379/1544106',
            },
        },
    }
