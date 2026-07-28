from __future__ import annotations

import unittest

from trove_core.providers.config import (
    ASR_SECRET_NAME,
    DEFAULT_ARK_VISION_MODEL,
    DEFAULT_ALIYUN_EMBEDDING_MODEL,
    DEFAULT_ALIYUN_RERANK_MODEL,
    CLOUD_RETRIEVAL_SECRET_NAME,
    VISION_SECRET_NAME,
    ProviderConfig,
    configured_cost_cap_rmb,
    provider_status_payload,
)


class ProviderConfigTests(unittest.TestCase):
    def test_provider_status_is_redacted_and_agent_switch_aware(self):
        env = {
            'TROVE_ENABLE_CLOUD_ASR': 'true',
            'TROVE_ENABLE_CLOUD_VISION': '1',
            'TROVE_CLOUD_COST_CAP_RMB': '12.5',
        }
        cfg = ProviderConfig.resolve(env, check_agent_switch=False)
        status = cfg.to_redacted_dict(env, agent_switch_names={ASR_SECRET_NAME, VISION_SECRET_NAME}, check_agent_switch=False)
        self.assertTrue(status['providers']['asr']['configured'])
        self.assertTrue(status['providers']['vision']['configured'])
        self.assertTrue(status['providers']['asr']['enabled'])
        self.assertEqual(status['providers']['vision']['model'], DEFAULT_ARK_VISION_MODEL)
        self.assertFalse(status['providers']['asr']['secret_value_included'])
        self.assertNotIn('Bearer ', str(status))
        self.assertNotIn('sk-', str(status))
        self.assertEqual(configured_cost_cap_rmb(env), 12.5)

    def test_provider_status_payload_does_not_require_cloud_keys(self):
        data = provider_status_payload(env={}, check_agent_switch=False)
        self.assertFalse(data['cloud_embedding_enabled'])
        self.assertFalse(data['cloud_chat_enabled'])
        self.assertFalse(data['providers']['asr']['configured'])
        self.assertFalse(data['providers']['vision']['configured'])
        self.assertFalse(data['providers']['embedding']['enabled'])
        self.assertFalse(data['providers']['rerank']['enabled'])
        self.assertFalse(data['providers']['embedding']['secret_value_included'])
        self.assertTrue(data['cloud_retrieval_requires_explicit_opt_in'])
        self.assertFalse(data['cost_cap']['value_included'])

    def test_cloud_retrieval_status_is_opt_in_and_redacted(self):
        env = {
            'TROVE_ENABLE_CLOUD_EMBEDDING': 'true',
            'TROVE_ENABLE_CLOUD_RERANK': '1',
            'TROVE_CLOUD_EMBEDDING_DIMENSIONS': '1024',
        }
        cfg = ProviderConfig.resolve(env, check_agent_switch=False)
        status = cfg.to_redacted_dict(env, agent_switch_names={CLOUD_RETRIEVAL_SECRET_NAME}, check_agent_switch=False)
        self.assertTrue(status['cloud_embedding_enabled'])
        self.assertTrue(status['cloud_rerank_enabled'])
        self.assertTrue(status['providers']['embedding']['configured'])
        self.assertTrue(status['providers']['rerank']['configured'])
        self.assertIn(DEFAULT_ALIYUN_EMBEDDING_MODEL, status['providers']['embedding']['model'])
        self.assertEqual(status['providers']['rerank']['model'], DEFAULT_ALIYUN_RERANK_MODEL)
        self.assertIn('aliyun-embedding', status['cloud_exception_only'])
        self.assertIn('aliyun-rerank', status['cloud_exception_only'])
        self.assertNotIn('test-key', str(status))

    def test_volcengine_embedding_can_be_selected_without_becoming_default(self):
        cfg = ProviderConfig.resolve({'TROVE_CLOUD_EMBEDDING_PROVIDER': 'volcengine'}, check_agent_switch=False)
        status = cfg.embedding_status(env={}, agent_switch_names=set(), check_agent_switch=False).to_dict()
        self.assertEqual(status['provider'], 'volcengine-embedding')
        self.assertFalse(status['enabled'])
        self.assertIn('provider override', ' '.join(status['notes']))


if __name__ == '__main__':
    unittest.main()
