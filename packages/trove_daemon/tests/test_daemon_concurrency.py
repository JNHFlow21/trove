from __future__ import annotations

import concurrent.futures
from functools import partial
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_client.client import TroveClient, TroveClientError
from trove_client.autostart import AutostartCoordinator
from trove_daemon.lifecycle import RuntimeIdentity, catalog_identity
from trove_daemon.server import DaemonServer as _DaemonServer


DaemonServer = (
    _DaemonServer if sys.platform == 'darwin'
    else partial(_DaemonServer, peer_uid=lambda _connection: os.getuid())
)


class _BlockingDispatcher:
    def __init__(self, gate=None):
        self.gate = gate
        self.calls = 0
        self.lock = threading.Lock()

    def dispatch(self, capability, payload, *, request_id, response_budget=None):
        with self.lock:
            self.calls += 1
        if self.gate is not None:
            self.gate.wait(2)
        return {'ok': True, 'request_id': request_id, 'data': {'generation': 'stable-a'}}


class DaemonConcurrencyTests(unittest.TestCase):
    def setUp(self):
        # Keep concurrency coverage portable without weakening the macOS-only
        # production guard enforced by lifecycle.require_macos.
        if sys.platform != 'darwin':
            for target in ('trove_daemon.lifecycle.require_macos', 'trove_daemon.server.require_macos'):
                platform_guard = patch(target)
                platform_guard.start()
                self.addCleanup(platform_guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        self.vault.mkdir()
        (self.vault / 'index').mkdir()
        self.identity = RuntimeIdentity.for_vault(
            self.vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_eight_concurrent_reads_share_one_generation(self):
        dispatcher = _BlockingDispatcher()
        server = DaemonServer(self.identity, dispatcher, max_workers=8, max_pending=8, idle_timeout=None)
        server.start()
        self.addCleanup(server.stop)

        def invoke(index):
            with TroveClient(self.identity, pool_size=1, autostart=None) as client:
                return client.call('trove.capabilities', {}, request_id=f'read-{index}')

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(invoke, range(8)))
        self.assertEqual(dispatcher.calls, 8)
        self.assertEqual({item['data']['generation'] for item in results}, {'stable-a'})

    def test_queue_saturation_returns_busy_without_wedging_server(self):
        gate = threading.Event()
        dispatcher = _BlockingDispatcher(gate)
        server = DaemonServer(self.identity, dispatcher, max_workers=1, max_pending=1, idle_timeout=None)
        server.start()
        self.addCleanup(server.stop)
        clients = [TroveClient(self.identity, pool_size=1, autostart=None) for _ in range(2)]
        self.addCleanup(lambda: [client.close() for client in clients])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(clients[0].call, 'trove.capabilities', {}, request_id='one')
            deadline = time.time() + 1
            while dispatcher.calls < 1 and time.time() < deadline:
                time.sleep(0.01)
            second = pool.submit(clients[1].call, 'trove.capabilities', {}, request_id='two')
            with self.assertRaises(TroveClientError) as caught:
                second.result(timeout=1)
            self.assertEqual(caught.exception.code, 'busy')
            gate.set()
            self.assertTrue(first.result(timeout=2)['ok'])
        self.assertTrue(clients[0].call('trove.capabilities', {}, request_id='after')['ok'])

    def test_stale_pooled_connection_recovers_once_after_daemon_crash(self):
        first = DaemonServer(self.identity, _BlockingDispatcher(), idle_timeout=None)
        first.start()
        coordinator = AutostartCoordinator(
            self.identity, probe=lambda: 'compatible',
            start=lambda: self.fail('replacement daemon is already running'),
        )
        client = TroveClient(self.identity, pool_size=1, autostart=coordinator)
        self.addCleanup(client.close)
        self.assertTrue(client.call('trove.capabilities', {}, request_id='before')['ok'])
        first.stop()
        replacement = DaemonServer(self.identity, _BlockingDispatcher(), idle_timeout=None)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.assertTrue(client.call('trove.capabilities', {}, request_id='after')['ok'])

    def test_stop_has_a_finite_drain_timeout(self):
        gate = threading.Event()
        dispatcher = _BlockingDispatcher(gate)
        server = DaemonServer(self.identity, dispatcher, max_workers=1, max_pending=1, idle_timeout=None)
        server.start()
        client = TroveClient(self.identity, pool_size=1, autostart=None)
        failures = []

        def invoke():
            try:
                client.call('trove.capabilities', {}, request_id='long-read')
            except TroveClientError as exc:
                failures.append(exc.code)

        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        deadline = time.time() + 1
        while dispatcher.calls < 1 and time.time() < deadline:
            time.sleep(0.01)
        started = time.monotonic()
        self.assertFalse(server.stop(timeout=0.1))
        self.assertLess(time.monotonic() - started, 0.75)
        gate.set()
        thread.join(timeout=1)
        client.close()
        self.assertEqual(failures, ['daemon_unavailable'])


if __name__ == '__main__':
    unittest.main()
