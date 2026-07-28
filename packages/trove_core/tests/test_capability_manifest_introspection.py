from __future__ import annotations

import unittest

from trove_cli.parser import public_routes
from trove_mcp.catalog_adapter import descriptors_for_pack
from trove_protocol.capabilities import CATALOG, catalog_snapshot, validate_catalog


class CapabilityManifestIntrospectionTests(unittest.TestCase):
    def test_protocol_catalog_is_the_single_typed_manifest(self):
        validate_catalog(CATALOG)
        self.assertEqual(catalog_snapshot()['protocol'], 'trove/1')
        self.assertEqual(len(CATALOG), len({spec.capability_id for spec in CATALOG}))

    def test_real_cli_and_mcp_surfaces_are_catalog_projections(self):
        cli = {(route.path, route.spec.capability_id) for route in public_routes() if route.spec}
        mcp = {(item.name, item.capability_id) for item in descriptors_for_pack('admin')}
        for spec in CATALOG:
            self.assertIn((spec.cli_route, spec.capability_id), cli)
            self.assertIn((spec.mcp_name, spec.capability_id), mcp)


if __name__ == '__main__':
    unittest.main()
