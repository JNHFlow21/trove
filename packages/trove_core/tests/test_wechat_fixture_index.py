from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class WeChatFixtureIndexTests(unittest.TestCase):
    def test_fixture_indexes_two_accounts_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            first = index_fixture_vault(Path(d), reset=True)
            second = index_fixture_vault(Path(d), reset=False)
            self.assertEqual(first['counts'], second['counts'])
            self.assertEqual(first['counts']['accounts'], 2)
            self.assertGreaterEqual(first['counts']['conversations'], 4)
            self.assertGreaterEqual(first['counts']['messages'], 12)

    def test_chinese_fts_has_customer_blocker(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            rows = store.fts_search('价格太高', limit=5)
            self.assertTrue(any('预算审批' in row['content'] for row in rows), rows)
