from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore
from trove_core.wechat.indexer import index_fixture_vault

class VectorStoreContractTests(unittest.TestCase):
    def test_fake_vector_index_and_filtered_search(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            provider = FakeEmbeddingProvider(dimensions=24)
            vector = SQLiteVectorStore(store)
            self.assertGreaterEqual(vector.index_all_messages(provider), 12)
            rows = vector.search('预算审批', filters={'conversation_id': 'conv-sales-review'}, limit=3, provider=provider)
            self.assertTrue(rows)
            self.assertEqual({r['conversation_id'] for r in rows}, {'conv-sales-review'})
