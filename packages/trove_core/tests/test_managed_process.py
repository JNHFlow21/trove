from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.managed_process import (
    ManagedLogGuard,
    ManagedProcessError,
    ManagedProcessManager,
    ManagedProcessRecord,
)


class FakeInspector:
    def __init__(self, *, running=True, birth='birth-a', command_hash='a' * 64):
        self.is_running = running
        self.birth = birth
        self.hash = command_hash

    def running(self, _pid):
        return self.is_running

    def birth_time(self, _pid):
        return self.birth

    def command_hash(self, _pid):
        return self.hash


class FakeProcess:
    def __init__(self, pid=4242, *, exit_code=None):
        self.pid = pid
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        self.exit_code = 0

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout=None):
        return self.exit_code


class ManagedProcessTests(unittest.TestCase):
    @staticmethod
    def _record(*, birth='birth-a', command_hash='a' * 64):
        return ManagedProcessRecord(
            name='api',
            pid=4242,
            birth_time=birth,
            command_hash=command_hash,
            health_endpoint='http://127.0.0.1:8765/health',
            nonce='b' * 32,
            created_at='2026-07-10T00:00:00Z',
        )

    def test_record_contains_complete_identity_and_success_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            inspector = FakeInspector()
            process = FakeProcess()
            manager = ManagedProcessManager(
                Path(directory),
                inspector=inspector,
                probe=lambda endpoint, nonce: endpoint.endswith('/health') and len(nonce) == 32,
                popen=lambda *args, **kwargs: process,
            )
            result = manager.start(
                'api',
                ['python', '-m', 'trove_api.server'],
                health_endpoint='http://127.0.0.1:8765/health',
                cwd=Path(directory),
                env={},
                readiness_timeout=0.2,
            )
            self.assertTrue(result['started'])
            payload = json.loads((Path(directory) / 'trove-api.pid').read_text(encoding='utf-8'))
            self.assertEqual(payload['pid'], 4242)
            self.assertEqual(payload['birth_time'], 'birth-a')
            self.assertEqual(payload['command_hash'], 'a' * 64)
            self.assertRegex(payload['nonce'], r'^[0-9a-f]{32}$')
            self.assertEqual(payload['health_endpoint'], 'http://127.0.0.1:8765/health')

    def test_managed_log_guard_truncates_one_shared_oversized_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'managed.log'
            with path.open('w+b') as stream:
                stream.write(b'x' * 2048)
                stream.flush()
                guard = ManagedLogGuard(
                    file_descriptors=(stream.fileno(), stream.fileno()),
                    max_bytes=1024,
                    poll_seconds=0.01,
                )

                self.assertEqual(guard.enforce_once(), 1)
                self.assertEqual(path.stat().st_size, 0)

    def test_new_managed_generation_discards_unbounded_old_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'trove-api.log'
            log.write_bytes(b'x' * 2048)
            process = FakeProcess()
            manager = ManagedProcessManager(
                root,
                inspector=FakeInspector(),
                probe=lambda *_args: True,
                popen=lambda *args, **kwargs: process,
            )

            manager.start(
                'api',
                ['python', '-m', 'trove_api.server'],
                health_endpoint='http://127.0.0.1:8765/health',
                cwd=root,
                env={},
                readiness_timeout=0.2,
            )

            self.assertEqual(log.read_bytes(), b'')

    def test_child_environment_is_minimal_and_excludes_default_and_custom_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            process = FakeProcess()
            captured: dict = {}

            def popen(*args, **kwargs):
                captured.update(kwargs)
                return process

            manager = ManagedProcessManager(
                Path(directory),
                inspector=FakeInspector(),
                probe=lambda *_args: True,
                popen=popen,
            )
            canary = 'CANARY_MANAGED_CHILD_SECRET_VALUE_9184'
            env = {
                'PATH': '/usr/bin:/bin',
                'HOME': '/tmp/synthetic-home',
                'LANG': 'C.UTF-8',
                'PYTHONPATH': '/tmp/synthetic-packages',
                'TROVE_CONSOLE_PORT': '4173',
                'TROVE_ENABLE_CLOUD_ASR': '1',
                'TROVE_EMBEDDING_DAEMON_QUEUE_SIZE': '64',
                'TROVE_EMBEDDING_SOCKET': '/tmp/synthetic.sock',
                'VOLCENGINE_ASR_API_KEY': canary,
                'VOLCENGINE_ARK_API_KEY': canary,
                'DASHSCOPE_API_KEY': canary,
                'TROVE_CLOUD_EMBEDDING_KEY': canary,
                'UNRELATED_PASSWORD': canary,
                'TROVE_ASR_SECRET_NAME': 'TROVE_EMBEDDING_SOCKET',
                'TROVE_VISION_SECRET_NAME': 'CUSTOM_VISION_SLOT',
                'TROVE_CLOUD_EMBEDDING_KEY_ENV': 'CUSTOM_EMBEDDING_SLOT',
                'TROVE_CLOUD_RERANK_KEY_ENV': 'CUSTOM_RERANK_SLOT',
                'CUSTOM_VISION_SLOT': canary,
                'CUSTOM_EMBEDDING_SLOT': canary,
                'CUSTOM_RERANK_SLOT': canary,
            }
            manager.start(
                'api',
                ['python', '-m', 'trove_api.server'],
                health_endpoint='http://127.0.0.1:8765/health',
                cwd=Path(directory),
                env=env,
                readiness_timeout=0.2,
            )

            child = captured['env']
            self.assertEqual(child['PATH'], env['PATH'])
            self.assertEqual(child['HOME'], env['HOME'])
            self.assertEqual(child['LANG'], env['LANG'])
            self.assertEqual(child['PYTHONPATH'], env['PYTHONPATH'])
            self.assertEqual(child['TROVE_CONSOLE_PORT'], '4173')
            self.assertEqual(child['TROVE_ENABLE_CLOUD_ASR'], '1')
            self.assertEqual(child['TROVE_EMBEDDING_DAEMON_QUEUE_SIZE'], '64')
            self.assertRegex(child['TROVE_MANAGED_NONCE'], r'^[0-9a-f]{32}$')
            for selector in (
                'TROVE_ASR_SECRET_NAME',
                'TROVE_VISION_SECRET_NAME',
                'TROVE_CLOUD_EMBEDDING_KEY_ENV',
                'TROVE_CLOUD_RERANK_KEY_ENV',
            ):
                self.assertEqual(child[selector], env[selector])
            self.assertNotIn('TROVE_EMBEDDING_SOCKET', child)
            self.assertFalse(set(child) & {
                'VOLCENGINE_ASR_API_KEY',
                'VOLCENGINE_ARK_API_KEY',
                'DASHSCOPE_API_KEY',
                'TROVE_CLOUD_EMBEDDING_KEY',
                'UNRELATED_PASSWORD',
                'CUSTOM_VISION_SLOT',
                'CUSTOM_EMBEDDING_SLOT',
                'CUSTOM_RERANK_SLOT',
            })
            self.assertNotIn(canary, repr(child))

    def test_pid_reuse_identity_mismatch_is_never_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = ManagedProcessManager(
                Path(directory),
                inspector=FakeInspector(birth='birth-reused'),
                probe=lambda *_args: True,
            )
            manager._write(self._record())
            with patch('trove_core.managed_process.os.kill', side_effect=AssertionError('must not signal')):
                result = manager.stop('api')
            self.assertFalse(result['identity_verified'])
            self.assertTrue(result['running'])
            self.assertEqual(result['error']['code'], 'managed_process_identity_unverified')

    def test_nonce_is_rechecked_before_signal_even_when_birth_and_command_match(self):
        with tempfile.TemporaryDirectory() as directory:
            probes = iter((True, False))
            manager = ManagedProcessManager(
                Path(directory),
                inspector=FakeInspector(),
                probe=lambda *_args: next(probes),
            )
            manager._write(self._record())
            with patch('trove_core.managed_process.os.kill', side_effect=AssertionError('must not signal reused pid')):
                result = manager.stop('api')
            self.assertFalse(result['identity_verified'])
            self.assertEqual(result['error']['code'], 'managed_process_identity_changed')

    def test_live_legacy_or_stale_pidfile_is_never_signalled_or_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trove-api.pid'
            path.write_text('4242\n', encoding='utf-8')
            path.chmod(0o600)
            manager = ManagedProcessManager(Path(directory), inspector=FakeInspector(), probe=lambda *_args: True)
            with patch('trove_core.managed_process.os.kill', side_effect=AssertionError('must not signal')):
                stopped = manager.stop('api')
            self.assertFalse(stopped['identity_verified'])
            with self.assertRaises(ManagedProcessError) as raised:
                manager.start(
                    'api',
                    ['python', '-m', 'trove_api.server'],
                    health_endpoint='http://127.0.0.1:8765/health',
                    cwd=Path(directory),
                    readiness_timeout=0.1,
                )
            self.assertEqual(raised.exception.code, 'managed_process_identity_unverified')

    def test_startup_failure_never_signals_pidfile_derived_process(self):
        with tempfile.TemporaryDirectory() as directory:
            process = FakeProcess(exit_code=7)
            manager = ManagedProcessManager(
                Path(directory),
                inspector=FakeInspector(running=False),
                probe=lambda *_args: False,
                popen=lambda *args, **kwargs: process,
            )
            with patch('trove_core.managed_process.os.kill', side_effect=AssertionError('must not signal')):
                with self.assertRaises(ManagedProcessError) as raised:
                    manager.start(
                        'api',
                        ['python', '-m', 'trove_api.server'],
                        health_endpoint='http://127.0.0.1:8765/health',
                        cwd=Path(directory),
                        readiness_timeout=0.1,
                    )
            self.assertEqual(raised.exception.code, 'managed_process_startup_failed')
            self.assertFalse((Path(directory) / 'trove-api.pid').exists())

    def test_readiness_failure_terminates_only_new_popen_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            process = FakeProcess()
            manager = ManagedProcessManager(
                Path(directory),
                inspector=FakeInspector(),
                probe=lambda *_args: False,
                popen=lambda *args, **kwargs: process,
            )
            with patch('trove_core.managed_process.os.kill', side_effect=AssertionError('must not signal by pid')):
                with self.assertRaises(ManagedProcessError) as raised:
                    manager.start(
                        'api',
                        ['python', '-m', 'trove_api.server'],
                        health_endpoint='http://127.0.0.1:8765/health',
                        cwd=Path(directory),
                        readiness_timeout=0.1,
                    )
            self.assertEqual(raised.exception.code, 'managed_process_readiness_failed')
            self.assertTrue(process.terminated)
            self.assertFalse((Path(directory) / 'trove-api.pid').exists())


if __name__ == '__main__':
    unittest.main()
