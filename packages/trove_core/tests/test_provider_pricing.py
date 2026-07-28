from __future__ import annotations

import unittest

from trove_core.providers.pricing import ArkVisionLitePricing, estimate_asr_flash_rmb, estimate_token_rmb, pricing_payload


class ProviderPricingTests(unittest.TestCase):
    def test_asr_flash_estimate_uses_verified_hourly_rate(self):
        self.assertEqual(estimate_asr_flash_rmb(3600), 4.5)
        self.assertEqual(estimate_asr_flash_rmb(60), 0.075)
        self.assertEqual(estimate_asr_flash_rmb(-10), 0.0)

    def test_ark_vision_pricing_estimate_is_token_bounded(self):
        pricing = ArkVisionLitePricing()
        self.assertEqual(pricing.estimate_rmb(input_tokens=1_000_000, output_tokens=1_000_000), 4.2)
        payload = pricing_payload()
        self.assertEqual(payload['asr_flash']['resource_id'], 'volc.bigasr.auc_turbo')
        self.assertEqual(payload['ark_vision_lite']['model'], 'doubao-seed-2-0-lite-260215')

    def test_cloud_retrieval_pricing_is_token_bounded_and_disabled_by_default(self):
        self.assertEqual(estimate_token_rmb(1_000_000, rmb_per_million_tokens=0.5), 0.5)
        self.assertEqual(estimate_token_rmb(-1, rmb_per_million_tokens=0.5), 0.0)
        payload = pricing_payload()
        self.assertTrue(payload['retrieval']['disabled_by_default'])
        self.assertEqual(payload['retrieval']['aliyun_text_embedding_v4']['rmb_per_million_tokens'], 0.5)
        self.assertEqual(payload['retrieval']['aliyun_qwen3_rerank']['default_candidate_window'], 50)


if __name__ == '__main__':
    unittest.main()
