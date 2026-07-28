from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.application.operation_journal import OperationConflict, OperationJournal
from trove_core.application.operations import OperationService
from trove_core.store.sqlite_store import SQLiteStore


class OperationReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'trove.sqlite'
        SQLiteStore(self.db).initialize()
        self.journal = OperationJournal(self.db)
        self.service = OperationService(self.journal)

    def tearDown(self):
        self.temp.cleanup()

    def test_same_idempotency_key_replays_one_operation(self):
        first, replayed = self.service.start(
            'trove.observe_add', {'text': 'fixture'},
            idempotency_key='fixture-key-00000003', replay_policy='journaled',
        )
        second, replayed_second = self.service.start(
            'trove.observe_add', {'text': 'fixture'},
            idempotency_key='fixture-key-00000003', replay_policy='journaled',
        )
        self.assertFalse(replayed)
        self.assertTrue(replayed_second)
        self.assertEqual(first.operation_id, second.operation_id)
        with self.assertRaises(OperationConflict):
            self.service.start(
                'trove.observe_add', {'text': 'different'},
                idempotency_key='fixture-key-00000003', replay_policy='journaled',
            )

    def test_domain_commit_and_terminal_journal_transition_share_one_transaction(self):
        operation, _ = self.service.start(
            'trove.observe_add', {'text': 'fixture'},
            idempotency_key='fixture-key-00000004', replay_policy='journaled',
        )
        with self.journal.transaction() as conn:
            conn.execute(
                """INSERT INTO observations(
                       observation_id,entity_id,observation_type,value_json,status,
                       confidence,citation,source_type,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    'observation-fixture', 'entity-fixture', 'operator_note',
                    '{"text":"fixture"}', 'active', 1.0, 'trove://fixture',
                    'operator', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                ),
            )
            self.journal.transition(
                operation.operation_id, expected_states={'pending'}, state='completed',
                stage='terminal', owner='none', result={'observation_id': 'observation-fixture'},
                connection=conn,
            )
        # Simulate losing the response after COMMIT.  The retry returns the
        # durable result instead of inserting a second domain row.
        replay, replayed = self.service.start(
            'trove.observe_add', {'text': 'fixture'},
            idempotency_key='fixture-key-00000004', replay_policy='journaled',
        )
        self.assertTrue(replayed)
        self.assertEqual(replay.result, {'observation_id': 'observation-fixture'})
        with self.journal.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM observations WHERE observation_id='observation-fixture'").fetchone()[0]
        self.assertEqual(count, 1)

    def test_external_side_effect_stays_reconciling_until_provider_status_is_terminal(self):
        operation, _ = self.service.start(
            'trove.media_enrich', {'citation': 'fixture'},
            idempotency_key='fixture-key-00000005', replay_policy='journaled',
        )
        dispatched = self.service.mark_external_dispatched(
            operation.operation_id, external_ref='provider-job-fixture',
        )
        self.assertEqual((dispatched.state, dispatched.owner), ('reconciling', 'provider'))
        pending = self.service.reconcile_external(dispatched.operation_id, terminal=False)
        self.assertEqual(pending.state, 'reconciling')
        completed = self.service.reconcile_external(
            dispatched.operation_id, terminal=True, result={'charge_count': 1},
        )
        self.assertEqual(completed.result, {'charge_count': 1})


if __name__ == '__main__':
    unittest.main()
