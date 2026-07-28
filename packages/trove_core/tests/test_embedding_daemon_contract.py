from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from trove_core.embedding.daemon import EmbeddingDaemonRuntime, _daemon_environment, serve, status_payload
from trove_core.embedding.daemon_client import EmbeddingDaemonClient
from trove_core.embedding.daemon_protocol import (
    DaemonIdentityMismatch,
    DaemonProtocolError,
    DaemonQueueSaturated,
    DaemonRequestTimeout,
    identity_for_model,
)


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 3, *, started: threading.Event | None = None, release: threading.Event | None = None) -> None:
        self.dimensions = dimensions
        self.started = started
        self.release = release
        self.calls = 0

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        return [[float(len(text)), 1.0, 0.0][:self.dimensions] for text in texts]


class EmbeddingDaemonContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._home = tempfile.TemporaryDirectory()
        self._environment = patch.dict(os.environ, {'HOME': self._home.name}, clear=True)
        self._environment.start()
        self.addCleanup(self._environment.stop)
        self.addCleanup(self._home.cleanup)

    def _model(self, root: str, *, dimensions: int = 3) -> Path:
        path = Path(root) / 'synthetic-model'
        path.mkdir()
        (path / 'trove_model_manifest.json').write_text(
            '{"model_id":"synthetic/embedding","provider":"sentence-transformers","dimensions":%d}' % dimensions,
            encoding='utf-8',
        )
        (path / 'config.json').write_text('{"synthetic":true}', encoding='utf-8')
        return path

    def test_handshake_returns_strict_redacted_identity_and_rejects_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = self._model(tmp)
            socket_path = str(Path(tmp) / 'CANARY_PRIVATE_SOCKET_NAME.sock')
            stop = threading.Event()
            provider = FakeEmbeddingProvider()
            thread = threading.Thread(
                target=serve,
                kwargs={
                    'model_path': str(model_path),
                    'socket_path': socket_path,
                    'stop_event': stop,
                    'provider_factory': lambda: provider,
                    'queue_size': 4,
                },
                daemon=True,
            )
            thread.start()
            for _ in range(100):
                if Path(socket_path).exists():
                    break
                time.sleep(0.01)
            identity = identity_for_model(model_path)
            client = EmbeddingDaemonClient(socket_path, identity=identity, timeout_ms=1000)
            handshake = client.handshake()
            self.assertEqual(handshake['identity']['provider'], 'sentence-transformers')
            self.assertEqual(handshake['identity']['model_id'], 'synthetic/embedding')
            self.assertEqual(handshake['identity']['dimensions'], 3)
            self.assertEqual(handshake['identity']['protocol_version'], 2)
            self.assertEqual(len(handshake['identity']['model_hash']), 64)
            self.assertNotIn(tmp, str(handshake))

            vectors, telemetry = client.embed(['synthetic request'])
            self.assertEqual(vectors, [[17.0, 1.0, 0.0]])
            self.assertEqual(telemetry['load_count'], 1)
            wrong = replace(identity, model_hash='0' * 64)
            with self.assertRaises(DaemonProtocolError) as ctx:
                EmbeddingDaemonClient(socket_path, identity=wrong, timeout_ms=1000).embed(['must not run'])
            self.assertEqual(ctx.exception.code, 'daemon_identity_mismatch')
            self.assertEqual(provider.calls, 1)
            self.assertEqual(os.stat(socket_path).st_mode & 0o777, 0o600)
            public_status = status_payload(socket_path, timeout=1)
            self.assertTrue(public_status['responsive'])
            self.assertNotIn(tmp, str(public_status))
            self.assertNotIn('CANARY_PRIVATE_SOCKET_NAME', str(public_status))
            stop.set()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    def test_daemon_environment_drops_known_and_custom_credentials(self):
        canary = 'CANARY_DAEMON_ENV_SECRET_9137'
        with patch.dict(os.environ, {
            'HOME': self._home.name,
            'DASHSCOPE_API_KEY': canary,
            'TROVE_CLOUD_EMBEDDING_KEY_ENV': 'ODD_PROVIDER_VALUE',
            'ODD_PROVIDER_VALUE': canary,
            'SAFE_RUNTIME_SETTING': 'synthetic-safe',
        }, clear=True):
            environment = _daemon_environment()
        self.assertNotIn('DASHSCOPE_API_KEY', environment)
        self.assertNotIn('ODD_PROVIDER_VALUE', environment)
        self.assertEqual(environment['SAFE_RUNTIME_SETTING'], 'synthetic-safe')
        self.assertNotIn(canary, str(environment))

    def test_manifest_identity_labels_are_redacted_before_handshake(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'private-model-directory'
            path.mkdir()
            canary = 'CANARY_PRIVATE_MANIFEST_4482'
            (path / 'trove_model_manifest.json').write_text(
                '{"model_id":"/Users/%s/private","provider":"private?%s","dimensions":3}' % (canary, canary),
                encoding='utf-8',
            )
            identity = identity_for_model(path)
            self.assertTrue(identity.model_id.startswith('custom-local-'))
            self.assertTrue(identity.provider.startswith('provider-'))
            self.assertNotIn(canary, str(identity.to_dict()))
            self.assertNotIn(tmp, str(identity.to_dict()))

    def test_concurrent_requests_singleflight_load_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = identity_for_model(self._model(tmp))
            provider = FakeEmbeddingProvider()
            loads = 0
            load_lock = threading.Lock()

            def load():
                nonlocal loads
                with load_lock:
                    loads += 1
                time.sleep(0.02)
                return provider

            runtime = EmbeddingDaemonRuntime(
                load,
                identity,
                queue_size=32,
                max_batch_requests=16,
                max_batch_texts=64,
                batch_wait_ms=25,
            )
            barrier = threading.Barrier(12)
            outputs: list[list[list[float]]] = []
            errors: list[Exception] = []

            def request(index: int) -> None:
                try:
                    barrier.wait(timeout=2)
                    outputs.append(runtime.submit([f'item-{index}'], expected_identity=identity, timeout_ms=2000))
                except Exception as exc:  # pragma: no cover - assertion captures
                    errors.append(exc)

            threads = [threading.Thread(target=request, args=(index,)) for index in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
            telemetry = runtime.telemetry()
            runtime.close()
            self.assertEqual(errors, [])
            self.assertEqual(len(outputs), 12)
            self.assertEqual(loads, 1)
            self.assertEqual(telemetry['load_count'], 1)
            self.assertGreaterEqual(telemetry['batched_requests'], 2)
            self.assertLess(provider.calls, 12)

    def test_queue_saturation_and_timeout_are_typed_and_content_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = identity_for_model(self._model(tmp))
            started = threading.Event()
            release = threading.Event()
            runtime = EmbeddingDaemonRuntime(
                lambda: FakeEmbeddingProvider(started=started, release=release),
                identity,
                queue_size=1,
                max_batch_requests=1,
                batch_wait_ms=0,
            )
            errors: list[DaemonProtocolError] = []

            def first() -> None:
                try:
                    runtime.submit(['CANARY_PRIVATE_FIRST'], expected_identity=identity, timeout_ms=2000)
                except DaemonProtocolError as exc:
                    errors.append(exc)

            def waiting() -> None:
                try:
                    runtime.submit(['CANARY_PRIVATE_WAITING'], expected_identity=identity, timeout_ms=30)
                except DaemonProtocolError as exc:
                    errors.append(exc)

            one = threading.Thread(target=first)
            one.start()
            self.assertTrue(started.wait(timeout=1))
            two = threading.Thread(target=waiting)
            two.start()
            for _ in range(100):
                if runtime.telemetry()['queue_depth'] == 1:
                    break
                time.sleep(0.001)
            self.assertEqual(runtime.telemetry()['queue_depth'], 1)
            with self.assertRaises(DaemonQueueSaturated) as ctx:
                runtime.submit(['CANARY_PRIVATE_SATURATED'], expected_identity=identity, timeout_ms=50)
            self.assertEqual(ctx.exception.code, 'daemon_queue_saturated')
            two.join(timeout=1)
            self.assertTrue(any(isinstance(error, DaemonRequestTimeout) for error in errors))
            release.set()
            one.join(timeout=2)
            telemetry = runtime.telemetry()
            runtime.close()
            rendered = str(telemetry) + str([error.code for error in errors]) + str(ctx.exception)
            self.assertNotIn('CANARY_PRIVATE', rendered)
            self.assertGreaterEqual(telemetry['saturated_requests'], 1)
            self.assertGreaterEqual(telemetry['timed_out_requests'], 1)
            self.assertFalse(telemetry['raw_content_included'])
            self.assertFalse(telemetry['raw_paths_included'])

    def test_dimension_mismatch_fails_before_any_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = identity_for_model(self._model(tmp, dimensions=3))
            runtime = EmbeddingDaemonRuntime(lambda: FakeEmbeddingProvider(dimensions=2), identity)
            with self.assertRaises(DaemonIdentityMismatch):
                runtime.submit(['synthetic'], expected_identity=identity, timeout_ms=1000)
            runtime.close()


if __name__ == '__main__':
    unittest.main()
