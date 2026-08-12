from __future__ import annotations
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.embedding.openai_compatible_provider import OpenAICompatibleEmbeddingProvider
from trove_core.runtime import configured_embedding_provider
from trove_core.security.subprocess_env import agent_switch_subprocess_environment
from trove_core.wechat.process_config import process_config_from_payload, write_process_config


class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data

class EmbeddingProviderContractTests(unittest.TestCase):
    def test_fake_embeddings_are_deterministic(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        self.assertEqual(provider.embed('客户价格'), provider.embed('客户价格'))
        self.assertEqual(len(provider.embed('客户价格')), 16)

    def test_cloud_embeddings_fail_closed_without_opt_in(self):
        with self.assertRaises(RuntimeError):
            OpenAICompatibleEmbeddingProvider()

    def test_cloud_environment_cannot_select_provider_for_ordinary_paths(self):
        env = {
            'TROVE_ENABLE_CLOUD_EMBEDDING': '1',
            'TROVE_CLOUD_EMBEDDING_KEY': 'synthetic-test-key',
            'TROVE_DISABLE_LOCAL_EMBEDDING': '1',
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertIsNone(configured_embedding_provider())
            with self.assertRaisesRegex(RuntimeError, '^cloud_embedding_requires_exact_approval$'):
                configured_embedding_provider(strict=True)

    def test_persisted_cloud_policy_never_silently_falls_back_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            write_process_config(vault, process_config_from_payload({
                'config_id': 'pcfg-cloud-provider-required',
                'vector_index': 'incremental',
                'cloud_retrieval': 'enabled',
            }))
            factory = SimpleNamespace(
                readiness=lambda _kind: SimpleNamespace(
                    ready=False,
                    reason_code='provider_secret_missing',
                ),
            )
            with patch('trove_core.runtime.ProviderFactory.resolve', return_value=factory):
                with self.assertRaisesRegex(RuntimeError, '^provider_secret_missing$'):
                    configured_embedding_provider(vault_root=vault)

    def test_agent_switch_environment_adds_user_local_bin_for_launch_agents(self):
        fixture_home = '/tmp/trove-fixture-home'
        env = agent_switch_subprocess_environment({
            'HOME': fixture_home,
            'PATH': '/usr/bin:/bin',
            'DASHSCOPE_API_KEY': 'must-not-propagate',
        })
        self.assertEqual(env['PATH'].split(os.pathsep)[0], f'{fixture_home}/.local/bin')
        self.assertEqual(env['PATH'].split(os.pathsep)[1], '/opt/homebrew/bin')
        self.assertNotIn('DASHSCOPE_API_KEY', env)

    def test_cloud_openai_batch_embeddings_parse_vectors(self):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(200, {'data': [{'embedding': [1, 2]}, {'embedding': [3, 4]}]})

        provider = OpenAICompatibleEmbeddingProvider(
            enabled=True,
            endpoint='https://example.invalid/embeddings',
            model='test-embedding',
            api_key='test-key',
            dimensions=2,
            provider_name='aliyun',
            post=post,
        )
        self.assertEqual(provider.embed_many(['a', 'b']), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(provider.dimensions, 2)
        self.assertEqual(provider.name, 'aliyun:test-embedding')
        self.assertEqual(calls[0][1]['json']['input'], ['a', 'b'])
        self.assertEqual(calls[0][1]['json']['dimensions'], 2)

    def test_dashscope_native_embeddings_preserve_order_roles_and_sparse_vectors(self):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            texts = kwargs['json']['input']['texts']
            rows = [
                {
                    'text_index': index,
                    'embedding': [float(index), 1.0],
                    'sparse_embedding': [{'index': index + 10, 'value': 0.5}],
                }
                for index, _text in reversed(list(enumerate(texts)))
            ]
            return FakeResponse(200, {'output': {'embeddings': rows}, 'usage': {'total_tokens': len(texts)}})

        provider = OpenAICompatibleEmbeddingProvider(
            enabled=True,
            endpoint='https://example.invalid/native',
            model='text-embedding-v4',
            api_key='test-key',
            dimensions=2,
            request_format='dashscope-native',
            provider_name='aliyun',
            max_workers=1,
            post=post,
        )
        documents = provider.embed_hybrid_many(['a', 'b'], text_type='document')
        query = provider.embed_query_hybrid('q')
        self.assertEqual([item.dense for item in documents], [[0.0, 1.0], [1.0, 1.0]])
        self.assertEqual([item.sparse for item in documents], [{10: 0.5}, {11: 0.5}])
        self.assertEqual(query.sparse, {10: 0.5})
        self.assertEqual(calls[0][1]['json']['parameters']['text_type'], 'document')
        self.assertNotIn('instruct', calls[0][1]['json']['parameters'])
        self.assertEqual(calls[1][1]['json']['parameters']['text_type'], 'query')
        self.assertTrue(calls[1][1]['json']['parameters']['instruct'])
        self.assertEqual(provider.provider_calls, 2)
        self.assertEqual(provider.input_tokens, 3)

    def test_dashscope_native_splits_aggregate_400_without_skipping_documents(self):
        calls = []

        def post(_url, **kwargs):
            texts = kwargs['json']['input']['texts']
            calls.append(len(texts))
            if len(texts) > 1:
                return FakeResponse(400, {'code': 'InvalidParameter'})
            return FakeResponse(200, {
                'output': {'embeddings': [{
                    'text_index': 0,
                    'embedding': [1.0, 0.0],
                    'sparse_embedding': [{'index': 1, 'value': 1.0}],
                }]},
                'usage': {'total_tokens': 1},
            })

        provider = OpenAICompatibleEmbeddingProvider(
            enabled=True,
            endpoint='https://example.invalid/native',
            model='text-embedding-v4',
            api_key='test-key',
            dimensions=2,
            request_format='dashscope-native',
            provider_name='aliyun',
            max_workers=1,
            post=post,
        )
        result = provider.embed_hybrid_many(['a', 'b', 'c'], text_type='document')
        self.assertEqual(len(result), 3)
        self.assertEqual(calls, [3, 1, 2, 1, 1])

    def test_dashscope_native_does_not_amplify_arrearage_400(self):
        calls = []

        def post(_url, **kwargs):
            calls.append(len(kwargs['json']['input']['texts']))
            return FakeResponse(400, {'code': 'Arrearage'})

        provider = OpenAICompatibleEmbeddingProvider(
            enabled=True,
            endpoint='https://example.invalid/native',
            model='text-embedding-v4',
            api_key='test-key',
            dimensions=2,
            request_format='dashscope-native',
            provider_name='aliyun',
            max_workers=1,
            post=post,
        )
        with self.assertRaisesRegex(RuntimeError, '^cloud_embedding_http_400_Arrearage$'):
            provider.embed_hybrid_many(['a', 'b', 'c'], text_type='document')
        self.assertEqual(calls, [3])

    def test_cloud_volcengine_multimodal_embeddings_parse_vector(self):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(200, {'data': {'object': 'embedding', 'embedding': [0.5, -0.5]}})

        provider = OpenAICompatibleEmbeddingProvider(
            enabled=True,
            endpoint='https://example.invalid/embeddings/multimodal',
            model='doubao-embedding-vision-251215',
            api_key='test-key',
            api_key_env='VOLCENGINE_ARK_API_KEY',
            dimensions=1024,
            request_format='volcengine-multimodal',
            provider_name='volcengine',
            post=post,
        )
        self.assertEqual(provider.embed('query'), [0.5, -0.5])
        self.assertEqual(provider.dimensions, 1024)
        self.assertEqual(provider.name, 'volcengine:doubao-embedding-vision-251215')
        self.assertEqual(calls[0][1]['json']['dimensions'], 1024)

    def test_cloud_provider_import_is_local_safe_without_httpx(self):
        code = r'''
import builtins
import os
import sys

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "httpx" or name.startswith("httpx."):
        raise ModuleNotFoundError("No module named 'httpx'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from trove_core.runtime import configured_embedding_provider
from trove_core.embedding.openai_compatible_provider import OpenAICompatibleEmbeddingProvider

for key in list(os.environ):
    if key.startswith("TROVE_ENABLE_CLOUD_") or key.startswith("TROVE_CLOUD_EMBEDDING_"):
        os.environ.pop(key, None)
os.environ["TROVE_DISABLE_LOCAL_EMBEDDING"] = "1"
assert configured_embedding_provider() is None
provider = OpenAICompatibleEmbeddingProvider(
    enabled=True,
    endpoint="https://example.invalid/embeddings",
    model="test-embedding",
    api_key="test-key",
)
try:
    provider.embed_many(["local import should already have succeeded"])
except RuntimeError as exc:
    if "httpx" in str(exc):
        raise SystemExit(0)
raise SystemExit(1)
'''
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
