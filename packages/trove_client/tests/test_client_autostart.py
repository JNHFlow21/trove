from __future__ import annotations

import concurrent.futures
import os
import tempfile
import threading
import unittest
from pathlib import Path

from trove_client.autostart import AutostartCoordinator, AutostartError
from trove_daemon.lifecycle import RuntimeIdentity, catalog_identity
from trove_core.managed_process import ManagedProcessManager


class ClientAutostartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        self.vault.mkdir()
        self.identity = RuntimeIdentity.for_vault(
            self.vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_two_clients_autostart_exactly_one_daemon(self):
        lock = threading.Lock()
        starts = []
        running = threading.Event()

        def probe():
            return 'compatible' if running.is_set() else 'unavailable'

        def start():
            with lock:
                starts.append('start')
                running.set()

        coordinator = AutostartCoordinator(self.identity, probe=probe, start=start)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: coordinator.ensure_running(), range(2)))
        self.assertEqual(starts, ['start'])
        self.assertEqual(os.stat(self.identity.lock_path).st_mode & 0o777, 0o600)

    def test_old_build_is_bounded_replace_not_silent_reuse(self):
        states = iter(['incompatible', 'compatible'])
        events = []
        coordinator = AutostartCoordinator(
            self.identity,
            probe=lambda: next(states),
            start=lambda: events.append('start'),
            stop=lambda timeout: events.append(('stop', timeout)),
            replace_timeout=3.0,
        )
        coordinator.ensure_running()
        self.assertEqual(events, [('stop', 3.0), 'start'])

    def test_failed_restart_is_attempted_once(self):
        starts = []
        coordinator = AutostartCoordinator(
            self.identity, probe=lambda: 'unavailable',
            start=lambda: starts.append('start'),
        )
        with self.assertRaises(AutostartError):
            coordinator.ensure_running()
        self.assertEqual(starts, ['start'])

    def test_system_autostart_uses_unix_health_and_stops_with_verified_identity(self):
        identity = RuntimeIdentity.for_vault(self.vault)
        coordinator = AutostartCoordinator.system(identity)
        manager = ManagedProcessManager(identity.runtime_dir)
        try:
            coordinator.ensure_running()
            status = manager.status('daemon')
            self.assertTrue(status['identity_verified'])
            self.assertTrue(status['health_endpoint'].startswith('unix://'))
        finally:
            stopped = coordinator.stop_callback(5.0)
        self.assertTrue(stopped['ok'])


if __name__ == '__main__':
    unittest.main()
