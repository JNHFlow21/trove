from __future__ import annotations

import json
from pathlib import Path
import unittest

from trove_mcp.catalog_adapter import schema_size


class MCPSchemaBudgetTests(unittest.TestCase):
    def test_standard_tools_list_stays_within_locked_release_budget(self):
        root = Path(__file__).resolve().parents[3]
        budgets = json.loads((root / 'docs/perf/agent-runtime-budgets.json').read_text())['targets']
        measured = schema_size('standard')
        self.assertLessEqual(measured['bytes'], budgets['standard_tools_list_bytes_max'])
        self.assertLessEqual(measured['estimated_tokens'], budgets['standard_tools_list_tokens_max'])

    def test_schema_measurement_is_deterministic_and_pack_monotonic(self):
        self.assertEqual(schema_size('standard'), schema_size('standard'))
        self.assertLess(schema_size('standard')['bytes'], schema_size('operations')['bytes'])
        self.assertLess(schema_size('operations')['bytes'], schema_size('admin')['bytes'])


if __name__ == '__main__':
    unittest.main()
