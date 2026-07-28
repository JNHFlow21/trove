from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.providers.cloud_policy import cloud_retrieval_environment, cloud_retrieval_policy
from trove_core.vault.config import VaultConfig
from trove_core.wechat.process_config import process_config_from_payload, write_process_config


class CloudRetrievalPolicyTests(unittest.TestCase):
    def test_new_vault_is_local_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = cloud_retrieval_policy(Path(tmp) / 'vault')
        self.assertFalse(policy['enabled'])
        self.assertFalse(policy['secret_value_included'])

    def test_explicit_vault_policy_enables_provider_flags_without_secret_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            write_process_config(vault, process_config_from_payload({
                'config_id': 'pcfg-cloud-test',
                'vector_index': 'incremental',
                'cloud_retrieval': 'enabled',
            }))
            policy = cloud_retrieval_policy(vault)
            env = cloud_retrieval_environment(vault, {})
        self.assertTrue(policy['enabled'])
        self.assertEqual(policy['consent_scope'], 'vault-continuous-retrieval-v1')
        self.assertEqual(env['TROVE_ENABLE_CLOUD_EMBEDDING'], '1')
        self.assertEqual(env['TROVE_ENABLE_CLOUD_RERANK'], '1')
        self.assertNotIn('DASHSCOPE_API_KEY', env)

    def test_agent_vector_status_resolves_the_vault_selected_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            vault.mkdir()
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = object()
            with (
                patch.object(agent_tools, 'configured_embedding_provider', return_value=provider) as select,
                patch.object(agent_tools, 'vector_status_payload', return_value={'state': 'available'}) as status,
            ):
                result = agent_tools.vector_status(vault)
        self.assertEqual(result, {'state': 'available'})
        select.assert_called_once_with(vault_root=cfg.root)
        status.assert_called_once_with(cfg, backend='zvec', provider=provider)


if __name__ == '__main__':
    unittest.main()
