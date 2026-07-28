from __future__ import annotations

import inspect
import unittest

import trove_mcp.server as server
from trove_mcp.catalog_adapter import descriptors_for_pack


class MCPServerTests(unittest.TestCase):
    def test_entry_point_is_thin_protocol_client_adapter(self):
        source = inspect.getsource(server)
        self.assertIn('v1_server', source)
        self.assertNotIn('trove_core.', source)
        self.assertNotIn('trove_api', source)

    def test_standard_surface_is_catalog_generated(self):
        descriptors = descriptors_for_pack('standard')
        self.assertEqual(len(descriptors), 12)
        self.assertEqual(len({item.name for item in descriptors}), len(descriptors))


if __name__ == '__main__':
    unittest.main()
