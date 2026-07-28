from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.store.repositories import EntityRecord, MultimodalRepository, ObservationRecord
from trove_core.store.sqlite_store import SQLiteStore


class CustomerOntologyTests(unittest.TestCase):
    def test_wechat_id_lookup_resolves_customer_with_citation(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育', identifiers={'wechat_id': 'wxid-example_edu'}))
            repo.add_observation(ObservationRecord(observation_id='obs-1', entity_id='customer-1', observation_type='wechat_username', value={'text': 'wxid-example_edu'}, status='active', confidence=0.95, citation='trove://wechat/acct/contact/1', source_type='contact'))
            result = resolve_customer(store, 'wxid-example_edu')
            self.assertFalse(result['ambiguous'])
            self.assertEqual(result['resolved']['entity_id'], 'customer-1')
            self.assertTrue(result['resolved']['citations'])


if __name__ == '__main__':
    unittest.main()
