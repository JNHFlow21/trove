from __future__ import annotations

import unittest

from trove_protocol.capabilities import CATALOG, catalog_snapshot


class CapabilityMapDriftTests(unittest.TestCase):
    def test_catalog_snapshot_is_complete_and_hash_bound(self):
        snapshot = catalog_snapshot()
        self.assertEqual(len(snapshot['capabilities']), len(CATALOG))
        self.assertEqual(len(snapshot['catalog_sha256']), 64)
        self.assertEqual(set(snapshot['packs']), {'standard', 'operations', 'admin'})

    def test_scoped_capabilities_expose_account_filter_and_citation_account(self):
        for spec in CATALOG:
            if not spec.scoped:
                continue
            self.assertIn('account_id', spec.input_schema['properties'])
            citation = spec.output_schema['$defs']['citation']
            self.assertIn('account_id', citation['required'])


if __name__ == '__main__':
    unittest.main()
