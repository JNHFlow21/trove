from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class IncrementalIndexTests(unittest.TestCase):
    def test_duplicate_local_ids_across_shards_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            rows = store.messages_for_conversation('acct-work', 'conv-example_edu-private')
            local_one = [r for r in rows if r['local_id'] == 1]
            self.assertEqual(len(local_one), 2)
            self.assertEqual({r['shard_id'] for r in local_one}, {'message_0', 'message_1'})

    def test_reindex_does_not_duplicate_counts(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            before = store.counts()
            # Atomic fixture publication refuses to unlink a live WAL/SHM.
            store.close()
            index_fixture_vault(Path(d), reset=False)
            refreshed = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            try:
                self.assertEqual(before, refreshed.counts())
            finally:
                refreshed.close()
