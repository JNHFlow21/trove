from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.zvec_store import ZVecStore
from trove_core.wechat.indexer import index_fixture_vault

class ZvecOptionalDependencyTests(unittest.TestCase):
    def test_zvec_absence_does_not_break_search(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            zvec = ZVecStore(str(Path(d) / 'vectors' / 'zvec'))
            resp = HyperSearch(SQLiteStore(Path(d) / 'index' / 'trove.sqlite')).search(SearchRequest('价格太高'))
            self.assertTrue(resp.results)
            if not zvec.available:
                self.assertIn('ZVEC', zvec.unavailable_reason)
