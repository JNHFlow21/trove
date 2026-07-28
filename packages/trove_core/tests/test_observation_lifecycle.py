from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.knowledge.observations import set_observation_status, supersede_observation
from trove_core.store.repositories import EntityRecord, MultimodalRepository, ObservationRecord
from trove_core.store.sqlite_store import SQLiteStore


class ObservationLifecycleTests(unittest.TestCase):
    def test_status_transitions_preserve_audit_history(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育'))
            repo.add_observation(ObservationRecord(observation_id='obs-old', entity_id='customer-1', observation_type='Need', value={'text': '旧需求'}, status='active', confidence=0.7, citation='trove://old', source_type='message'))
            new = supersede_observation(repo, 'obs-old', ObservationRecord(observation_id='obs-new', entity_id='customer-1', observation_type='Need', value={'text': '新需求'}, status='active', confidence=0.9, citation='trove://new', source_type='message'))
            self.assertEqual(new['supersedes_observation_id'], 'obs-old')
            with repo.store.connect() as conn:
                self.assertEqual(conn.execute('SELECT status FROM observations WHERE observation_id=?', ('obs-old',)).fetchone()[0], 'superseded')
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 2)
            rejected = set_observation_status(repo, 'obs-new', 'rejected')
            self.assertEqual(rejected['status'], 'rejected')


if __name__ == '__main__':
    unittest.main()
