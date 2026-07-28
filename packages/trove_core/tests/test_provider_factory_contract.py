from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trove_core.providers.config import ASR_SECRET_NAME, CLOUD_RETRIEVAL_SECRET_NAME, VISION_SECRET_NAME, ProviderConfig, SecretPresence, agent_switch_secret_names, provider_status_payload
from trove_core.providers.factory import EnvironmentAgentSwitchSecretResolver, ProviderFactory, ProviderUnavailable
from trove_core.wechat.decrypt.secrets import SecretResolutionError


class FakeSecretResolver:
    def __init__(self, values: dict[str, str], *, source: str = 'agent-switch') -> None:
        self._values = dict(values)
        self.source = source
        self.resolved: list[str] = []

    def __repr__(self) -> str:
        return '<FakeSecretResolver redacted>'

    def presence(self, name: str) -> SecretPresence:
        return SecretPresence(name=name, present=name in self._values, source=self.source if name in self._values else None)

    def resolve(self, name: str) -> str:
        self.resolved.append(name)
        if name not in self._values:
            raise RuntimeError('missing_key')
        return self._values[name]


class FakeAgentSwitch:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)
        self.help_checks = 0
        self.gets: list[str] = []

    def _supports_fd_secret_get(self) -> bool:
        self.help_checks += 1
        return True

    def get_secret(self, name: str) -> str:
        self.gets.append(name)
        return self._values[name]


class ProviderFactoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._home = tempfile.TemporaryDirectory()
        self._environment = patch.dict(os.environ, {'HOME': self._home.name}, clear=True)
        self._environment.start()
        self.addCleanup(self._environment.stop)
        self.addCleanup(self._home.cleanup)

    def test_concrete_agent_switch_resolver_uses_names_for_status_and_fd_for_invocation(self):
        canary = 'CANARY_AGENT_SWITCH_FD_VALUE_1192'
        ambient_canary = 'CANARY_AMBIENT_VALUE_MUST_BE_IGNORED_9931'
        switch = FakeAgentSwitch({ASR_SECRET_NAME: canary})
        resolver = EnvironmentAgentSwitchSecretResolver(
            {ASR_SECRET_NAME: ambient_canary},
            agent_switch_names={ASR_SECRET_NAME},
            agent_switch=switch,  # type: ignore[arg-type]
        )
        factory = ProviderFactory.resolve(
            {'TROVE_ENABLE_CLOUD_ASR': '1'},
            secret_resolver=resolver,
            dependency_probe=lambda _name: True,
        )
        status = factory.status_payload()
        self.assertTrue(status['providers']['asr']['ready'])
        self.assertEqual(switch.gets, [])
        provider = factory.create_asr()
        self.assertEqual(provider.api_key, canary)
        self.assertEqual(switch.gets, [ASR_SECRET_NAME])
        self.assertEqual(switch.help_checks, 1)
        self.assertNotIn(canary, str(status))
        self.assertNotEqual(provider.api_key, ambient_canary)
        self.assertNotIn(ambient_canary, str(status) + repr(resolver))
        self.assertNotIn(canary, repr(resolver))

    def test_agent_switch_name_probe_does_not_inherit_provider_secret_values(self):
        canary = 'CANARY_AGENT_SWITCH_LIST_ENV_VALUE_2376'
        with (
            patch.dict(os.environ, {ASR_SECRET_NAME: canary}),
            patch(
                'trove_core.providers.config.subprocess.run',
                return_value=SimpleNamespace(returncode=0, stdout=f'{ASR_SECRET_NAME}\n'),
            ) as run,
        ):
            self.assertEqual(agent_switch_secret_names(), {ASR_SECRET_NAME})
        child_env = run.call_args.kwargs['env']
        self.assertNotIn(ASR_SECRET_NAME, child_env)
        self.assertNotIn(canary, repr(child_env))

    def test_environment_values_are_never_provider_credentials(self):
        from trove_core.asr.volcengine_flash import VolcengineASRFlashProvider
        from trove_core.embedding.openai_compatible_provider import OpenAICompatibleEmbeddingProvider
        from trove_core.search.cloud_reranker import CloudRerankProvider
        from trove_core.vision.volcengine_ark import VolcengineArkVisionProvider

        canary = 'CANARY_ENVIRONMENT_PROVIDER_VALUE_7843'
        custom_names = {
            'asr': 'CUSTOM_ASR_SLOT',
            'vision': 'CUSTOM_VISION_SLOT',
            'embedding': 'CUSTOM_EMBEDDING_SLOT',
            'rerank': 'CUSTOM_RERANK_SLOT',
        }
        env = {
            ASR_SECRET_NAME: canary,
            VISION_SECRET_NAME: canary,
            CLOUD_RETRIEVAL_SECRET_NAME: canary,
            'TROVE_CLOUD_EMBEDDING_KEY': canary,
            'TROVE_ASR_SECRET_NAME': custom_names['asr'],
            'TROVE_VISION_SECRET_NAME': custom_names['vision'],
            'TROVE_CLOUD_EMBEDDING_KEY_ENV': custom_names['embedding'],
            'TROVE_CLOUD_RERANK_KEY_ENV': custom_names['rerank'],
            custom_names['asr']: canary,
            custom_names['vision']: canary,
            custom_names['embedding']: canary,
            custom_names['rerank']: canary,
            'TROVE_ENABLE_CLOUD_ASR': '1',
            'TROVE_ENABLE_CLOUD_VISION': '1',
            'TROVE_ENABLE_CLOUD_EMBEDDING': '1',
            'TROVE_ENABLE_CLOUD_RERANK': '1',
        }
        resolver = EnvironmentAgentSwitchSecretResolver(
            env,
            agent_switch_names=set(),
            assume_agent_switch_transport=True,
        )
        for name in {
            ASR_SECRET_NAME,
            VISION_SECRET_NAME,
            CLOUD_RETRIEVAL_SECRET_NAME,
            'TROVE_CLOUD_EMBEDDING_KEY',
            *custom_names.values(),
        }:
            self.assertFalse(resolver.presence(name).present)
            with self.assertRaises(SecretResolutionError):
                resolver.resolve(name)

        factory = ProviderFactory.resolve(env, secret_resolver=resolver, dependency_probe=lambda _name: True)
        status = factory.status_payload()
        self.assertEqual(status['secret_transport'], 'agent-switch-fd-only')
        self.assertFalse(status['environment_secret_values_accepted'])
        self.assertTrue(all(not provider['ready'] for provider in status['providers'].values()))
        self.assertTrue(all(provider['secret']['source'] is None for provider in status['providers'].values()))
        self.assertNotIn(canary, str(status) + repr(resolver) + repr(factory.env))

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as asr_error:
                VolcengineASRFlashProvider()
            with self.assertRaises(RuntimeError) as vision_error:
                VolcengineArkVisionProvider()
            with self.assertRaises(RuntimeError) as embedding_error:
                OpenAICompatibleEmbeddingProvider(
                    enabled=True,
                    endpoint='https://example.invalid/embeddings',
                    model='fixture-embedding',
                    api_key_env=custom_names['embedding'],
                )
            with self.assertRaises(RuntimeError) as rerank_error:
                CloudRerankProvider(
                    enabled=True,
                    endpoint='https://example.invalid/rerank',
                    model='fixture-rerank',
                    api_key_env=custom_names['rerank'],
                )
        errors = ' '.join(str(ctx.exception) for ctx in (asr_error, vision_error, embedding_error, rerank_error))
        self.assertNotIn(canary, errors)

    def test_agent_switch_only_status_and_invocation_share_one_resolver(self):
        canary = 'CANARY_PROVIDER_SECRET_VALUE_4791'
        resolver = FakeSecretResolver({ASR_SECRET_NAME: canary})
        env = {'TROVE_ENABLE_CLOUD_ASR': '1'}
        factory = ProviderFactory.resolve(env, secret_resolver=resolver, dependency_probe=lambda _name: True)

        status = factory.status_payload()
        self.assertTrue(status['providers']['asr']['ready'])
        self.assertEqual(status['providers']['asr']['secret']['source'], 'agent-switch')
        self.assertNotIn(canary, str(status))
        self.assertFalse(status['secret_values_included'])

        provider = factory.create_asr()
        self.assertEqual(provider.api_key, canary)
        self.assertEqual(resolver.resolved, [ASR_SECRET_NAME])
        self.assertNotIn(canary, repr(provider))
        self.assertNotIn(canary, str(provider.__dict__.keys()))
        self.assertNotIn(ASR_SECRET_NAME, os.environ)

    def test_status_payload_uses_injected_resolver_without_secret_value(self):
        canary = 'CANARY_STATUS_SECRET_VALUE_7348'
        resolver = FakeSecretResolver({CLOUD_RETRIEVAL_SECRET_NAME: canary})
        env = {
            'TROVE_ENABLE_CLOUD_EMBEDDING': 'true',
            'TROVE_ENABLE_CLOUD_RERANK': 'true',
        }
        status = provider_status_payload(
            env,
            secret_resolver=resolver,
            dependency_probe=lambda _name: True,
        )
        self.assertTrue(status['providers']['embedding']['ready'])
        self.assertTrue(status['providers']['rerank']['ready'])
        self.assertEqual(resolver.resolved, [])
        self.assertNotIn(canary, str(status))

    def test_invalid_environment_secret_has_same_status_and_invocation_failure(self):
        canary = 'CANARY_INVALID_SECRET_6321\nheader-injection'
        resolver = EnvironmentAgentSwitchSecretResolver(
            {ASR_SECRET_NAME: canary},
            agent_switch_names=set(),
            assume_agent_switch_transport=True,
        )
        factory = ProviderFactory.resolve(
            {'TROVE_ENABLE_CLOUD_ASR': '1', ASR_SECRET_NAME: canary},
            secret_resolver=resolver,
            dependency_probe=lambda _name: True,
        )
        status = factory.status_payload()
        self.assertFalse(status['providers']['asr']['ready'])
        self.assertEqual(status['providers']['asr']['reason_code'], 'provider_credential_missing')
        with self.assertRaises(ProviderUnavailable) as ctx:
            factory.create_asr()
        self.assertEqual(ctx.exception.code, status['providers']['asr']['reason_code'])
        self.assertNotIn('CANARY_INVALID_SECRET', str(status) + str(ctx.exception.to_dict()))

    def test_status_redacts_unsafe_model_and_endpoint_configuration(self):
        canary = 'CANARY_ENDPOINT_VALUE_5274'
        env = {
            'TROVE_ENABLE_CLOUD_EMBEDDING': '1',
            'TROVE_CLOUD_EMBEDDING_MODEL': f'/Users/{canary}/model',
            'TROVE_CLOUD_EMBEDDING_ENDPOINT': f'https://user:{canary}@example.invalid/private?token={canary}',
        }
        factory = ProviderFactory.resolve(
            env,
            secret_resolver=FakeSecretResolver({CLOUD_RETRIEVAL_SECRET_NAME: 'synthetic-value'}),
            dependency_probe=lambda _name: True,
        )
        status = factory.status_payload()['providers']['embedding']
        self.assertTrue(status['ready'])
        self.assertEqual(status['endpoint'], 'https://example.invalid')
        self.assertTrue(status['model'].startswith('model-'))
        self.assertNotIn(canary, str(status))

    def test_missing_credential_and_dependency_fail_typed_and_visible(self):
        missing = ProviderFactory.resolve(
            {'TROVE_ENABLE_CLOUD_ASR': '1'},
            secret_resolver=FakeSecretResolver({}),
            dependency_probe=lambda _name: True,
        )
        self.assertEqual(missing.readiness('asr').reason_code, 'provider_credential_missing')
        with self.assertRaises(ProviderUnavailable) as ctx:
            missing.create_asr()
        self.assertEqual(ctx.exception.code, 'provider_credential_missing')
        self.assertFalse(ctx.exception.to_dict()['secret_value_included'])

        canary = 'CANARY_DEPENDENCY_SECRET_VALUE_8830'
        unavailable = ProviderFactory.resolve(
            {'TROVE_ENABLE_CLOUD_EMBEDDING': '1'},
            secret_resolver=FakeSecretResolver({CLOUD_RETRIEVAL_SECRET_NAME: canary}),
            dependency_probe=lambda name: name != 'httpx',
        )
        status = unavailable.readiness('embedding')
        self.assertFalse(status.ready)
        self.assertEqual(status.reason_code, 'provider_dependency_missing')
        with self.assertRaises(ProviderUnavailable) as ctx:
            unavailable.create_cloud_embedding()
        self.assertEqual(ctx.exception.code, 'provider_dependency_missing')
        self.assertNotIn(canary, str(ctx.exception.to_dict()))

    def test_cloud_provider_headers_use_resolved_value_without_environment_mutation(self):
        canary = 'CANARY_DIRECT_PROVIDER_VALUE_2046'
        resolver = FakeSecretResolver({CLOUD_RETRIEVAL_SECRET_NAME: canary})
        env = {
            'TROVE_ENABLE_CLOUD_EMBEDDING': '1',
            'TROVE_ENABLE_CLOUD_RERANK': '1',
        }
        with patch.dict(os.environ, {}, clear=True):
            factory = ProviderFactory.resolve(env, secret_resolver=resolver, dependency_probe=lambda _name: True)
            embedding = factory.create_cloud_embedding(post=lambda *_a, **_k: None)
            reranker = factory.create_cloud_reranker(post=lambda *_a, **_k: None)
            self.assertEqual(embedding._headers()['Authorization'], f'Bearer {canary}')
            self.assertEqual(reranker._headers()['Authorization'], f'Bearer {canary}')
            self.assertNotIn(CLOUD_RETRIEVAL_SECRET_NAME, os.environ)
        self.assertNotIn(canary, repr(embedding))
        self.assertNotIn(canary, repr(reranker))

    def test_ordinary_local_absence_is_typed_fallback_but_explicit_path_fails(self):
        factory = ProviderFactory.resolve(
            {'TROVE_DISABLE_LOCAL_EMBEDDING': '1'},
            secret_resolver=FakeSecretResolver({}),
            dependency_probe=lambda _name: True,
        )
        selection = factory.select_local_embedding()
        self.assertIsNone(selection.provider)
        self.assertEqual(selection.status()['state'], 'unavailable_fallback')
        self.assertEqual(selection.reason_code, 'local_embedding_disabled')

        with tempfile.TemporaryDirectory() as home:
            explicit = ProviderFactory.resolve(
                {},
                secret_resolver=FakeSecretResolver({}),
                dependency_probe=lambda _name: True,
            )
            missing = Path(home) / 'missing-model'
            with self.assertRaises(ProviderUnavailable) as ctx:
                explicit.select_local_embedding(missing)
            self.assertEqual(ctx.exception.code, 'local_embedding_model_missing')
            self.assertNotIn(home, str(ctx.exception.to_dict()))


if __name__ == '__main__':
    unittest.main()
