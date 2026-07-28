from __future__ import annotations
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from trove_core.embedding.daemon_client import EmbeddingDaemonClient
from trove_core.embedding.daemon_protocol import DaemonProtocolError
from trove_core.embedding.local_provider import LocalEmbeddingProvider
from trove_core.embedding.model_registry import model_status, registry_snapshot


class FakeSentenceTransformer:
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class LocalEmbeddingProviderTests(unittest.TestCase):
    def test_requires_existing_explicit_path(self):
        with self.assertRaises(RuntimeError):
            LocalEmbeddingProvider('/definitely/missing/trove-model')

    def test_uses_local_files_only_model_factory(self):
        calls = []
        def factory(path: str, **kwargs):
            calls.append((path, kwargs))
            return FakeSentenceTransformer(path, **kwargs)

        with tempfile.TemporaryDirectory() as d:
            provider = LocalEmbeddingProvider(Path(d), model_factory=factory)
            self.assertEqual(provider.dimensions, 3)
            self.assertEqual(provider.embed('客户')[0], 2.0)
            self.assertTrue(calls[0][1]['local_files_only'])

    def test_model_status_has_registry_without_private_path(self):
        data = model_status(model='bge-small-zh-v1.5')
        self.assertEqual(data['expected_dimensions'], 512)
        self.assertIn('models', registry_snapshot())
        self.assertNotIn('/Users/', str(data))

    def test_download_manifest_dimensions_are_ready_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'trove_model_manifest.json').write_text(json.dumps({
                'model_id': 'synthetic/local-embedding',
                'provider': 'sentence-transformers',
                'expected_dimensions': 512,
            }), encoding='utf-8')
            provider = LocalEmbeddingProvider(path, use_daemon=False)
            self.assertEqual(provider.dimensions, 512)
            self.assertEqual(provider._daemon_identity.dimensions, 512)

    def test_daemon_failure_uses_local_model_with_typed_redacted_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = LocalEmbeddingProvider(directory, dimensions=3, use_daemon=True)
            provider._model_factory = lambda path, **kwargs: FakeSentenceTransformer(path, **kwargs)
            with patch.object(EmbeddingDaemonClient, 'embed', side_effect=DaemonProtocolError('daemon_queue_saturated')):
                self.assertEqual(provider.embed('CANARY_PRIVATE_TEXT'), [19.0, 1.0, 0.0])
            telemetry = provider.daemon_telemetry()
            self.assertEqual(telemetry['last_reason_code'], 'daemon_queue_saturated')
            self.assertEqual(telemetry['fallback_mode'], 'in_process_local')
            self.assertEqual(telemetry['fallback_count'], 1)
            self.assertNotIn('CANARY_PRIVATE_TEXT', str(telemetry))
            self.assertFalse(telemetry['raw_content_included'])

    def test_in_process_model_loading_is_singleflight(self):
        calls = 0
        lock = threading.Lock()

        def factory(path: str, **kwargs):
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.02)
            return FakeSentenceTransformer(path, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            provider = LocalEmbeddingProvider(directory, dimensions=3, use_daemon=False)
            provider._model_factory = factory
            outputs: list[list[float]] = []
            threads = [threading.Thread(target=lambda: outputs.append(provider.embed('并发'))) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=1)
            self.assertEqual(calls, 1)
            self.assertEqual(len(outputs), 8)
