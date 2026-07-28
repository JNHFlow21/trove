from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.operation_journal import OperationJournal, OperationNotFound
from trove_core.vault.config import VaultConfig


IDEMPOTENCY_KEY = 'crash-fixture-idempotency'


def _crash_worker(root: str, phase: str) -> None:
    config = VaultConfig.resolve(root, env={})
    config.ensure()
    journal = OperationJournal(config.paths.sqlite_path)
    if phase == 'before_request':
        os._exit(23)
    record, _ = journal.start(
        'trove.observe_add', {'target': 'fixture', 'text': 'fixture'},
        idempotency_key=IDEMPOTENCY_KEY, replay_policy='journaled',
    )
    if phase == 'during_request':
        os._exit(23)
    connection = sqlite3.connect(config.paths.sqlite_path)
    connection.execute('BEGIN IMMEDIATE')
    connection.execute(
        'CREATE TABLE IF NOT EXISTS fault_side_effects(id TEXT PRIMARY KEY,value TEXT NOT NULL)'
    )
    if phase == 'before_write':
        os._exit(23)
    connection.execute(
        'INSERT OR IGNORE INTO fault_side_effects(id,value) VALUES(?,?)',
        (record.operation_id, 'committed'),
    )
    connection.commit()
    if phase == 'after_write':
        os._exit(23)
    journal.transition(
        record.operation_id, expected_states={'pending'},
        state='completed', stage='terminal', owner='none',
        result={'side_effect_committed': True},
    )


class OperationCrashRecoveryTests(unittest.TestCase):
    def test_kill_boundaries_recover_to_complete_old_or_new_state(self):
        # TROVE v1 is macOS-only and the crash boundary itself is the subject
        # under test. ``fork`` avoids importing the test package in a fresh
        # interpreter after deliberately terminating the child with os._exit.
        context = multiprocessing.get_context('fork')
        for phase in ('before_request', 'during_request', 'before_write', 'after_write'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                vault = Path(directory) / 'vault'
                process = context.Process(target=_crash_worker, args=(str(vault), phase))
                process.start()
                process.join(timeout=10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 23)

                config = VaultConfig.resolve(str(vault), env={})
                journal = OperationJournal(config.paths.sqlite_path)
                with journal.connection() as connection:
                    exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fault_side_effects'"
                    ).fetchone()
                    side_effects = int(connection.execute(
                        'SELECT COUNT(*) FROM fault_side_effects'
                    ).fetchone()[0]) if exists else 0
                    operation_count = int(connection.execute(
                        'SELECT COUNT(*) FROM operation_journal'
                    ).fetchone()[0])
                self.assertIn(side_effects, (0, 1))
                self.assertIn(operation_count, (0, 1))

                if operation_count:
                    with journal.connection() as connection:
                        operation_id = str(connection.execute(
                            'SELECT operation_id FROM operation_journal'
                        ).fetchone()[0])
                    record = journal.get(operation_id)
                    if side_effects:
                        record = journal.transition(
                            operation_id, expected_states={'pending'},
                            state='completed', stage='reconciled_after_restart', owner='none',
                            result={'side_effect_committed': True},
                        )
                    self.assertEqual(
                        (record.state, side_effects),
                        ('completed', 1) if side_effects else ('pending', 0),
                    )
                else:
                    self.assertEqual((phase, side_effects), ('before_request', 0))

    def test_disk_full_simulation_returns_typed_failure_without_journal_row(self):
        with tempfile.TemporaryDirectory() as directory:
            config = VaultConfig.resolve(str(Path(directory) / 'vault'), env={})
            config.ensure()
            dispatcher = build_default_dispatcher(config)
            journal = dispatcher.context.operations.journal
            with mock.patch.object(
                journal, 'start', side_effect=sqlite3.OperationalError('database or disk is full'),
            ):
                response = dispatcher.dispatch(
                    'trove.media_enrich',
                    {'citation': 'trove://fixture/acct/media', 'kind': 'annotate'},
                    request_id='disk-full',
                )
            self.assertFalse(response['ok'])
            self.assertEqual(response['error']['code'], 'capability_unavailable')
            with journal.connection() as connection:
                count = connection.execute('SELECT COUNT(*) FROM operation_journal').fetchone()[0]
            self.assertEqual(count, 0)

    def test_lost_continuation_token_cannot_be_replayed_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config = VaultConfig.resolve(str(Path(directory) / 'vault'), env={})
            config.ensure()
            journal = OperationJournal(config.paths.sqlite_path)
            record, _ = journal.start(
                'trove.media_enrich', {'citation': 'fixture'},
                idempotency_key='continuation-crash-fixture', replay_policy='journaled',
            )
            from trove_core.application.operations import OperationService

            service = OperationService(journal)
            _waiting, token = service.await_agent(record.operation_id, stage='awaiting_agent')
            completed = service.continue_operation(
                record.operation_id, token=token, payload={'ok': True},
                continuation=lambda _record, payload: dict(payload),
            )
            self.assertEqual(completed.state, 'completed')
            restarted = OperationService(OperationJournal(config.paths.sqlite_path))
            with self.assertRaises(Exception):
                restarted.continue_operation(
                    record.operation_id, token=token, payload={'ok': True},
                    continuation=lambda _record, payload: dict(payload),
                )


if __name__ == '__main__':
    unittest.main()
