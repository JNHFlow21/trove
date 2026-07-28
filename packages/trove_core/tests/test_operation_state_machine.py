from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.application.operation_journal import OperationConflict, OperationJournal
from trove_core.application.operations import OperationService
from trove_core.store.sqlite_store import SQLiteStore


class OperationStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'trove.sqlite'
        SQLiteStore(self.db).initialize()
        self.journal = OperationJournal(self.db)
        self.service = OperationService(self.journal)

    def tearDown(self):
        self.temp.cleanup()

    def test_awaiting_agent_continuation_is_opaque_and_single_use(self):
        operation, _ = self.service.start(
            'trove.media_enrich', {'citation': 'fixture', 'kind': 'annotate'},
            idempotency_key='fixture-key-00000001', replay_policy='journaled',
        )
        waiting, token = self.service.await_agent(operation.operation_id, stage='image_understanding')
        self.assertNotIn(operation.operation_id, token)
        completed = self.service.continue_operation(
            waiting.operation_id, token=token, payload={'observation': 'fixture'},
            continuation=lambda _record, payload: {'accepted': payload['observation']},
        )
        self.assertEqual(completed.state, 'completed')
        self.assertEqual(completed.result, {'accepted': 'fixture'})
        with self.assertRaises(OperationConflict):
            self.service.continue_operation(
                waiting.operation_id, token=token, payload={'observation': 'again'},
                continuation=lambda _record, payload: payload,
            )

    def test_terminal_state_cannot_regress_and_write_window_cannot_cancel(self):
        operation, _ = self.service.start(
            'trove.observe_add', {'text': 'fixture'},
            idempotency_key='fixture-key-00000002', replay_policy='journaled',
        )
        committing = self.journal.transition(
            operation.operation_id, expected_states={'pending'}, state='running',
            stage='write_committing', owner='daemon',
        )
        with self.assertRaises(OperationConflict):
            self.service.cancel(committing.operation_id)
        completed = self.journal.transition(
            operation.operation_id, expected_states={'running'}, state='completed',
            stage='terminal', owner='none', result={'observation_id': 'fixture-id'},
        )
        with self.assertRaises(OperationConflict):
            self.journal.transition(
                completed.operation_id, expected_states={'completed'}, state='running',
                stage='retry', owner='daemon',
            )


if __name__ == '__main__':
    unittest.main()
