from __future__ import annotations

import socket
import unittest

from scripts.public_surface_lint import repository_snapshot, scan_public_surface


class NoWebApiSurfaceTests(unittest.TestCase):
    def test_tracked_product_surface_is_local_agent_runtime_only(self):
        self.assertEqual(scan_public_surface(), [])

    def test_daemon_transport_is_unix_socket_only(self):
        files = repository_snapshot()
        source = files['packages/trove_daemon/trove_daemon/server.py']
        self.assertIn('socket.AF_UNIX', source)
        self.assertNotIn('socket.AF_INET', source)
        self.assertEqual(socket.AF_UNIX, 1)

    def test_removed_surface_is_absent_from_repository_index(self):
        paths = set(repository_snapshot())
        self.assertFalse(any(path.startswith('apps/web_console/') for path in paths))
        self.assertFalse(any(path.startswith('packages/trove_api/') for path in paths))
        self.assertNotIn('package.json', paths)
        self.assertNotIn('package-lock.json', paths)


if __name__ == '__main__':
    unittest.main()
