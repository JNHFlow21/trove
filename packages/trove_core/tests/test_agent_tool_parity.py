from __future__ import annotations

import unittest

from trove_cli.parser import public_routes
from trove_mcp.catalog_adapter import descriptors_for_pack
from trove_protocol.capabilities import CATALOG


class AgentToolParityTests(unittest.TestCase):
    def test_cli_and_admin_mcp_are_complete_projections_of_catalog(self):
        cli = {(route.path, route.spec.capability_id) for route in public_routes() if route.spec}
        mcp = {(item.name, item.capability_id) for item in descriptors_for_pack('admin')}
        self.assertTrue({(spec.cli_route, spec.capability_id) for spec in CATALOG} <= cli)
        self.assertEqual({(spec.mcp_name, spec.capability_id) for spec in CATALOG}, mcp)

    def test_catalog_has_no_source_specific_generic_surface(self):
        generic = ' '.join(
            value
            for spec in CATALOG
            for value in (spec.capability_id, spec.mcp_name, *spec.cli_route, spec.description)
        )
        self.assertNotIn('wechat', generic.lower())


if __name__ == '__main__':
    unittest.main()
