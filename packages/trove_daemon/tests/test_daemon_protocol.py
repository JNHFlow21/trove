from __future__ import annotations

from functools import partial
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_client.client import TroveClient
from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.vault.config import VaultConfig
from trove_daemon.cursors import DaemonCursorStore
from trove_daemon.lifecycle import (
    LifecycleError,
    RuntimeIdentity,
    build_identity,
    catalog_identity,
    require_macos,
)
from trove_daemon.server import DaemonServer as _DaemonServer
from trove_daemon.session import SessionContract, SessionError
from trove_protocol.codec import FrameDecoder, MAX_FRAME_BYTES, encode_frame


DaemonServer = (
    _DaemonServer if sys.platform == 'darwin'
    else partial(_DaemonServer, peer_uid=lambda _connection: os.getuid())
)


class _Dispatcher:
    def dispatch(self, capability, payload, *, request_id, response_budget=None):
        return {
            'ok': True,
            'request_id': request_id,
            'data': {'capability': capability, 'payload': dict(payload)},
        }


def _read_frame(stream: socket.socket) -> dict:
    header = stream.recv(4)
    if len(header) != 4:
        raise AssertionError('missing frame header')
    length = struct.unpack('>I', header)[0]
    body = bytearray()
    while len(body) < length:
        chunk = stream.recv(length - len(body))
        if not chunk:
            raise AssertionError('truncated frame')
        body.extend(chunk)
    return json.loads(body)


class DaemonPlatformBoundaryTests(unittest.TestCase):
    def test_production_guard_rejects_non_macos(self):
        with patch('trove_daemon.lifecycle.sys.platform', 'linux'):
            with self.assertRaises(LifecycleError) as caught:
                require_macos()
        self.assertEqual(caught.exception.code, 'platform_unsupported')


class DaemonProtocolTests(unittest.TestCase):
    def test_build_identity_is_independent_of_current_working_directory(self):
        build_identity.cache_clear()
        expected = build_identity()
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                build_identity.cache_clear()
                actual = build_identity()
            finally:
                os.chdir(previous)
                build_identity.cache_clear()
        self.assertEqual(actual, expected)

    def setUp(self):
        # Exercise the portable Unix-socket protocol in Linux CI while the
        # production lifecycle continues to reject unsupported platforms.
        if sys.platform != 'darwin':
            for target in ('trove_daemon.lifecycle.require_macos', 'trove_daemon.server.require_macos'):
                platform_guard = patch(target)
                platform_guard.start()
                self.addCleanup(platform_guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault-a'
        self.vault.mkdir()
        (self.vault / 'index').mkdir()
        self.identity = RuntimeIdentity.for_vault(
            self.vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_vault_runtime_paths_are_isolated_owner_only_and_uds_only(self):
        other = Path(self.temp.name) / 'vault-b'
        other.mkdir()
        second = RuntimeIdentity.for_vault(
            other, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )
        self.identity.prepare()
        second.prepare()
        self.assertNotEqual(self.identity.vault_identity, second.vault_identity)
        self.assertNotEqual(self.identity.socket_path, second.socket_path)
        self.assertEqual(os.stat(self.identity.runtime_dir).st_mode & 0o777, 0o700)
        self.assertFalse(hasattr(self.identity, 'tcp_port'))

    def test_long_vault_fallback_is_stable_and_not_ambient_tmpdir(self):
        long_vault = Path(self.temp.name) / ('long-' + 'x' * 80)
        long_vault.mkdir()
        first = RuntimeIdentity.for_vault(
            long_vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )
        previous = os.environ.get('TMPDIR')
        os.environ['TMPDIR'] = str(Path(self.temp.name) / 'different-temp')
        try:
            second = RuntimeIdentity.for_vault(
                long_vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
            )
        finally:
            if previous is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = previous
        self.assertEqual(first.runtime_dir, second.runtime_dir)
        self.assertEqual(first.runtime_dir.parent, Path('/tmp') / f'trove-{os.getuid()}')

    def test_two_vaults_run_isolated_daemons_simultaneously(self):
        other = Path(self.temp.name) / 'vault-b-live'
        other.mkdir()
        (other / 'index').mkdir()
        second_identity = RuntimeIdentity.for_vault(
            other, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )
        first = DaemonServer(self.identity, _Dispatcher(), idle_timeout=None)
        second = DaemonServer(second_identity, _Dispatcher(), idle_timeout=None)
        first.start()
        second.start()
        self.addCleanup(first.stop)
        self.addCleanup(second.stop)
        with TroveClient(self.identity, autostart=None) as first_client, TroveClient(second_identity, autostart=None) as second_client:
            self.assertTrue(first_client.call('trove.capabilities', {}, request_id='vault-a')['ok'])
            self.assertTrue(second_client.call('trove.capabilities', {}, request_id='vault-b')['ok'])
        self.assertNotEqual(self.identity.cache_dir, second_identity.cache_dir)

    def test_hello_rejects_wrong_identity_build_catalog_and_protocol(self):
        contract = SessionContract(self.identity)
        base = contract.client_hello(client_id='fixture', role='cli')
        for field, value in (
            ('vault_identity', 'wrong'),
            ('build_hash', '0' * 64),
            ('catalog_hash', '1' * 64),
            ('protocol_min', 'trove/2'),
        ):
            with self.subTest(field=field):
                with self.assertRaises(SessionError):
                    contract.accept_hello(base | {field: value}, peer_uid=os.getuid())

    def test_non_current_peer_and_invalid_frames_fail_closed(self):
        contract = SessionContract(self.identity)
        hello = contract.client_hello(client_id='fixture', role='mcp')
        with self.assertRaises(SessionError) as caught:
            contract.accept_hello(hello, peer_uid=os.getuid() + 1)
        self.assertEqual(caught.exception.code, 'peer_unauthorized')
        decoder = FrameDecoder()
        with self.assertRaises(Exception):
            decoder.feed(struct.pack('>I', MAX_FRAME_BYTES + 1))
        decoder = FrameDecoder()
        decoder.feed(b'\x00\x00\x00\x08{"half"')
        with self.assertRaises(Exception):
            decoder.finish()

    def test_non_current_peer_receives_typed_rejection(self):
        server = DaemonServer(
            self.identity, _Dispatcher(), idle_timeout=None,
            peer_uid=lambda _connection: os.getuid() + 1,
        )
        server.start()
        self.addCleanup(server.stop)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.identity.socket_path))
        client.sendall(encode_frame(SessionContract(self.identity).client_hello(client_id='wrong-peer', role='cli')))
        response = _read_frame(client)
        client.close()
        self.assertEqual(response['error']['code'], 'peer_unauthorized')

    def test_signed_operator_session_has_separate_exact_control_contract(self):
        calls = []
        server = DaemonServer(
            self.identity,
            _Dispatcher(),
            idle_timeout=None,
            peer_pid=lambda _connection: 4242,
            operator_authorizer=lambda pid: pid == 4242,
            operator_control=lambda action, payload: (
                calls.append((action, dict(payload)))
                or {'state': 'approved'}
            ),
        )
        server.start()
        self.addCleanup(server.stop)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.identity.socket_path))
        self.addCleanup(client.close)
        client.sendall(encode_frame(
            SessionContract(self.identity).client_hello(
                client_id='signed-companion',
                role='operator',
                persistent=False,
            )
        ))
        self.assertTrue(_read_frame(client)['ok'])
        client.sendall(encode_frame({
            'type': 'operator_request',
            'request_id': 'operator-review-fixture',
            'action': 'reply.approve',
            'review_id': 'review_fixture_0001',
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
        }))
        response = _read_frame(client)
        self.assertTrue(response['ok'])
        self.assertEqual(calls, [(
            'reply.approve', {'review_id': 'review_fixture_0001'},
        )])
        client.sendall(encode_frame({
            'type': 'operator_request',
            'request_id': 'operator-retry-fixture',
            'action': 'reply.retry',
            'review_id': 'review_fixture_0001',
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
        }))
        response = _read_frame(client)
        self.assertTrue(response['ok'])
        self.assertEqual(calls[-1], (
            'reply.retry', {'review_id': 'review_fixture_0001'},
        ))
        client.sendall(encode_frame({
            'type': 'operator_request',
            'request_id': 'operator-mode-fixture',
            'action': 'reply.set_mode',
            'mode': 'review_queue',
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
        }))
        response = _read_frame(client)
        self.assertTrue(response['ok'])
        self.assertEqual(calls[-1], (
            'reply.set_mode', {'mode': 'review_queue'},
        ))

    def test_signed_operator_mode_rejects_unbounded_value(self):
        calls = []
        server = DaemonServer(
            self.identity,
            _Dispatcher(),
            idle_timeout=None,
            peer_pid=lambda _connection: 4242,
            operator_authorizer=lambda pid: pid == 4242,
            operator_control=lambda action, payload: calls.append(
                (action, dict(payload))
            ) or {},
        )
        server.start()
        self.addCleanup(server.stop)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.identity.socket_path))
        self.addCleanup(client.close)
        client.sendall(encode_frame(
            SessionContract(self.identity).client_hello(
                client_id='signed-companion-mode',
                role='operator',
                persistent=False,
            )
        ))
        self.assertTrue(_read_frame(client)['ok'])
        client.sendall(encode_frame({
            'type': 'operator_request',
            'request_id': 'operator-mode-invalid',
            'action': 'reply.set_mode',
            'mode': 'unsafe',
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
        }))

        response = _read_frame(client)

        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'invalid_request')
        self.assertEqual(calls, [])

    def test_untrusted_operator_process_is_rejected_before_request(self):
        server = DaemonServer(
            self.identity,
            _Dispatcher(),
            idle_timeout=None,
            peer_pid=lambda _connection: 99,
            operator_authorizer=lambda _pid: False,
            operator_control=lambda _action, _payload: {},
        )
        server.start()
        self.addCleanup(server.stop)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.identity.socket_path))
        client.sendall(encode_frame(
            SessionContract(self.identity).client_hello(
                client_id='untrusted-companion',
                role='operator',
                persistent=False,
            )
        ))
        response = _read_frame(client)
        client.close()
        self.assertFalse(response['ok'])
        self.assertEqual(
            response['error']['code'], 'operator_unauthorized',
        )

    def test_deadline_expiry_is_typed_over_real_unix_socket(self):
        server = DaemonServer(self.identity, _Dispatcher(), idle_timeout=None)
        server.start()
        self.addCleanup(server.stop)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.identity.socket_path))
        self.addCleanup(client.close)
        client.sendall(encode_frame(SessionContract(self.identity).client_hello(client_id='fixture', role='cli')))
        self.assertTrue(_read_frame(client)['ok'])
        client.sendall(encode_frame({
            'protocol': 'trove/1', 'request_id': 'expired',
            'capability': 'trove.capabilities', 'input': {},
            'deadline_ms': int(time.time() * 1000) - 1,
            'response_budget': 8192,
            'vault_identity': self.identity.vault_identity,
            'build_hash': self.identity.build_hash,
            'catalog_hash': self.identity.catalog_hash,
        }))
        response = _read_frame(client)
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'deadline_expired')
        self.assertEqual(os.stat(self.identity.socket_path).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.identity.pid_path).st_mode & 0o777, 0o600)
        metadata = self.identity.pid_path.read_text(encoding='utf-8')
        self.assertNotIn(str(self.vault), metadata)

    def test_oversize_and_half_frame_disconnect_do_not_wedge_daemon(self):
        server = DaemonServer(self.identity, _Dispatcher(), idle_timeout=None)
        server.start()
        self.addCleanup(server.stop)
        oversized = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        oversized.connect(str(self.identity.socket_path))
        oversized.sendall(struct.pack('>I', MAX_FRAME_BYTES + 1))
        self.assertEqual(_read_frame(oversized)['error']['code'], 'frame_too_large')
        oversized.close()
        partial = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        partial.connect(str(self.identity.socket_path))
        partial.sendall(b'\x00\x00\x00\x20{"half"')
        partial.close()
        healthy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        healthy.connect(str(self.identity.socket_path))
        healthy.sendall(encode_frame(SessionContract(self.identity).client_hello(client_id='after-half', role='cli')))
        self.assertTrue(_read_frame(healthy)['ok'])
        healthy.close()

    def test_stale_unix_socket_is_replaced_safely(self):
        self.identity.prepare()
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(self.identity.socket_path))
        stale.close()
        server = DaemonServer(self.identity, _Dispatcher(), idle_timeout=None)
        server.start()
        self.addCleanup(server.stop)
        self.assertTrue(self.identity.socket_path.exists())

    def test_persistent_mcp_session_suppresses_idle_exit(self):
        server = DaemonServer(self.identity, _Dispatcher(), idle_timeout=0.1)
        runner = threading.Thread(target=server.serve_forever, daemon=True)
        runner.start()
        deadline = time.time() + 1
        while not self.identity.socket_path.exists() and time.time() < deadline:
            time.sleep(0.01)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(self.identity.socket_path))
        client.sendall(encode_frame(SessionContract(self.identity).client_hello(client_id='mcp-fixture', role='mcp')))
        self.assertTrue(_read_frame(client)['ok'])
        time.sleep(0.3)
        self.assertTrue(runner.is_alive())
        client.close()
        runner.join(timeout=1.5)
        self.assertFalse(runner.is_alive())

    def test_armed_reply_keepalive_suppresses_idle_exit_until_disarmed(self):
        armed = True
        server = DaemonServer(
            self.identity,
            _Dispatcher(),
            idle_timeout=0.1,
            keepalive=lambda: armed,
        )
        runner = threading.Thread(
            target=server.serve_forever, daemon=True,
        )
        runner.start()
        deadline = time.time() + 1
        while not self.identity.socket_path.exists() and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.3)
        self.assertTrue(runner.is_alive())
        armed = False
        runner.join(timeout=1.5)
        self.assertFalse(runner.is_alive())

    def test_pending_operation_is_recoverable_after_bounded_daemon_stop(self):
        config = VaultConfig.resolve(str(self.vault), env={})
        first = DaemonServer(self.identity, build_default_dispatcher(config), idle_timeout=None)
        first.start()
        with TroveClient(self.identity, autostart=None) as client:
            started = client.call(
                'trove.media_enrich',
                {'citation': 'trove://fixture/account-a/message-1', 'kind': 'transcribe'},
                request_id='durable-operation',
            )
        operation_id = started['data']['operation']['operation_id']
        self.assertTrue(first.stop(timeout=1.0))
        replacement = DaemonServer(self.identity, build_default_dispatcher(config), idle_timeout=None)
        replacement.start()
        self.addCleanup(replacement.stop)
        with TroveClient(self.identity, autostart=None) as client:
            status = client.call(
                'trove.operation_status', {'operation_id': operation_id},
                request_id='recover-operation',
            )
        self.assertEqual(status['data']['operation']['operation_id'], operation_id)
        self.assertEqual(status['data']['operation']['state'], 'pending')

    def test_cursor_restart_expiry_generation_and_tamper_are_typed(self):
        clock = [100.0]
        store = DaemonCursorStore(
            vault_identity=self.identity.vault_identity,
            restart_id='restart-a', ttl_seconds=2, clock=lambda: clock[0],
        )
        handle = store.issue(
            capability='trove.search', filters={'q': 'fixture'}, keyset={'row': 1},
            high_water='1', generation='generation-a',
        )
        self.assertGreaterEqual(len(handle), 40)
        self.assertEqual(store.resolve(
            handle, capability='trove.search', filters={'q': 'fixture'},
            generation='generation-a',
        ).keyset, {'row': 1})
        for candidate, code, generation in (
            (handle + 'tamper', 'cursor_invalid', 'generation-a'),
            (handle, 'cursor_stale', 'generation-b'),
            ('restart-b.' + handle.split('.', 1)[1], 'cursor_stale', 'generation-a'),
        ):
            with self.subTest(code=code):
                with self.assertRaises(Exception) as caught:
                    store.resolve(candidate, capability='trove.search', filters={'q': 'fixture'}, generation=generation)
                self.assertEqual(caught.exception.code, code)
        clock[0] = 103.0
        with self.assertRaises(Exception) as caught:
            store.resolve(handle, capability='trove.search', filters={'q': 'fixture'}, generation='generation-a')
        self.assertEqual(caught.exception.code, 'cursor_expired')


if __name__ == '__main__':
    unittest.main()
