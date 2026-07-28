from __future__ import annotations

import ast
import importlib
from pathlib import Path
import tempfile
import unittest

from trove_core.store.change_journal import (
    clear_dirty_citations,
    read_aux_fingerprints,
    read_dirty_citations,
    read_waterlines,
    record_dirty_citations,
    write_aux_fingerprints,
    write_waterlines,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.source_discovery import iter_importable_files


CORE = Path(__file__).resolve().parents[1] / 'trove_core'


class ImportArchitectureTests(unittest.TestCase):
    def test_sync_and_import_job_depend_on_neutral_modules_not_each_other(self) -> None:
        sync = (CORE / 'sync.py').read_text(encoding='utf-8')
        import_job = (CORE / 'wechat' / 'import_job.py').read_text(encoding='utf-8')
        self.assertNotIn('trove_core.wechat.import_job import', sync)
        self.assertNotIn('trove_core.sync import', import_job)
        self.assertIn('trove_core.wechat.source_discovery import', sync)
        self.assertIn('from trove_core.store import change_journal', import_job)

        # Import order must be irrelevant; this used to rely on sync being
        # mostly initialized before import_job's dynamic callback ran.
        for first, second in (
            ('trove_core.sync', 'trove_core.wechat.import_job'),
            ('trove_core.wechat.import_job', 'trove_core.sync'),
        ):
            with self.subTest(first=first):
                self.assertIsNotNone(importlib.import_module(first))
                self.assertIsNotNone(importlib.import_module(second))

    def test_discovery_yields_account_units_and_skips_sqlite_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            account = root / 'account-a'
            account.mkdir()
            (account / 'contact.db').touch()
            (account / 'message_0.db').touch()
            (root / 'other.db').touch()
            (root / 'other.db-wal').touch()
            self.assertEqual(list(iter_importable_files(root)), [account])

            account.unlink(missing_ok=True) if account.is_file() else None
            for child in account.iterdir():
                child.unlink()
            account.rmdir()
            self.assertEqual(list(iter_importable_files(root)), [root / 'other.db'])

    def test_change_journal_round_trip_is_independent_of_sync(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = VaultConfig.resolve(str(Path(directory) / 'vault'))
            cfg.paths.index_dir.mkdir(parents=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()
            try:
                updates = {('a', 'c', 's'): {
                    'max_local_id': 7,
                    'max_create_time': 8,
                    'max_timestamp': '2026-01-01T00:00:00Z',
                }}
                self.assertEqual(write_waterlines(store, updates), 1)
                self.assertEqual(read_waterlines(store)[('a', 'c', 's')]['max_local_id'], 7)
                self.assertEqual(write_aux_fingerprints(store, {'contacts:a': 'digest'}), 1)
                self.assertEqual(read_aux_fingerprints(store), {'contacts:a': 'digest'})
                refs = [{'citation': 'trove://x', 'account_id': 'a', 'conversation_id': 'c', 'source_type': 'message'}]
                self.assertEqual(record_dirty_citations(store, refs), 1)
                self.assertEqual(read_dirty_citations(store), ['trove://x'])
                self.assertEqual(clear_dirty_citations(store, ['trove://x']), 1)
                self.assertEqual(read_dirty_citations(store), [])
            finally:
                store.close_all()


if __name__ == '__main__':
    unittest.main()
