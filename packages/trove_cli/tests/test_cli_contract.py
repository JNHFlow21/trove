from __future__ import annotations

import inspect
import unittest

import trove_cli.main as cli_main
from trove_cli.parser import public_routes
from trove_protocol.capabilities import CATALOG


class CliContractTests(unittest.TestCase):
    def test_entry_point_is_thin_v1_adapter(self):
        source = inspect.getsource(cli_main)
        self.assertIn('v1_main', source)
        self.assertNotIn('trove_core.', source)
        self.assertNotIn('trove_api', source)

    def test_catalog_is_the_only_business_route_registry(self):
        mapped = {
            (route.path, route.spec.capability_id)
            for route in public_routes() if route.spec is not None
        }
        self.assertTrue({(spec.cli_route, spec.capability_id) for spec in CATALOG} <= mapped)


if __name__ == '__main__':
    unittest.main()
