from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.measure_agent_surface import ALLOWED_DISPOSITIONS, validate_inventory


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'tests' / 'golden' / 'trove_legacy_surface_inventory.json'


class LegacySurfaceInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(INVENTORY.read_text(encoding='utf-8'))

    def test_inventory_is_valid_and_complete(self):
        validate_inventory(self.payload)
        self.assertEqual(self.payload['schema_version'], 1)
        self.assertRegex(self.payload['captured_git_sha'], r'^[0-9a-f]{40}$')
        self.assertEqual(set(self.payload['dispositions']), ALLOWED_DISPOSITIONS)

    def test_every_legacy_item_has_one_disposition_and_unique_identity(self):
        identities: set[tuple[str, str]] = set()
        for item in self.payload['items']:
            identity = (item['surface'], item['name'])
            self.assertNotIn(identity, identities)
            identities.add(identity)
            self.assertIn(item['disposition'], ALLOWED_DISPOSITIONS)
            self.assertTrue(item['replacement'] or item['disposition'] in {'internal', 'intentional_delete'})

    def test_frozen_surface_counts_match_the_pre_cutover_product(self):
        counts: dict[str, int] = {}
        for item in self.payload['items']:
            counts[item['surface']] = counts.get(item['surface'], 0) + 1
        self.assertEqual(counts['cli'], 117)
        self.assertEqual(counts['mcp'], 50)
        self.assertEqual(counts['entry_point'], 3)
        self.assertGreaterEqual(counts['skill'], 1)


if __name__ == '__main__':
    unittest.main()
