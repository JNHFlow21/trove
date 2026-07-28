from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault


class TrigramFTSTests(unittest.TestCase):
    def test_three_plus_character_chinese_substrings_hit_indexes(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            self.assertTrue(store.exact_search('示例教育', limit=5))
            rows = store.chunk_search('客户卡在哪', filters={'conversation_id': 'conv-sales-review'}, limit=5)
            self.assertTrue(rows)
            self.assertTrue(any(row['parent_citation'].endswith('/message_0/10') for row in rows))

    def test_two_character_query_uses_bounded_like_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            rows = store.fts_search_filtered('预算', filters={'conversation_id': 'conv-sales-review'}, limit=5)
            self.assertTrue(rows)
            self.assertTrue(any('预算审批' in row['content'] for row in rows))

    def test_phrase_priority_and_filters(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            rows = store.exact_search('evidence-first', filters={'conversation_id': 'conv-trove-team'}, limit=5)
            self.assertTrue(rows)
            self.assertEqual(rows[0]['citation'], 'trove://wechat/acct-work/conv-trove-team/message_0/20')
            sender_rows = store.fts_search_filtered('local token', filters={'sender': '工程-小陈'}, limit=5)
            self.assertEqual([row['sender_name'] for row in sender_rows], ['工程-小陈'])
            chunk_rows = store.chunk_search('三个月试点', filters={'source_type': 'message'}, limit=5)
            self.assertTrue(all(row['source_type'] == 'message' for row in chunk_rows))

    def test_fts_special_characters_are_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            for query in ['客户"预算', 'token*', '-预算', '(审批)', '客户😀卡']:
                with self.subTest(query=query):
                    store.exact_search(query, limit=3)
                    store.fts_search_filtered(query, limit=3, allow_like_fallback=False)
                    store.chunk_search(query, limit=3)


if __name__ == '__main__':
    unittest.main()
