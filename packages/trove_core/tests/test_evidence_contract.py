from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class EvidenceContractTests(unittest.TestCase):
    def test_message_rows_have_evidence_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            row = store.get_message_by_citation('trove://wechat/acct-work/conv-sales-review/message_0/10')
            self.assertIsNotNone(row)
            self.assertEqual(row['account_label'], 'Work-WeChat')
            self.assertEqual(row['conversation_title'], '私域成交复盘群')
            self.assertEqual(row['direction'], 'incoming')
            self.assertIn('价格太高', row['content'])

    def test_outgoing_message_direction_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            row = store.get_message_by_citation('trove://wechat/acct-work/conv-sales-review/message_0/12')
            self.assertEqual(row['direction'], 'outgoing')

    def test_batch_evidence_lookup_returns_message_rows_by_citation(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            citations = [
                'trove://wechat/acct-work/conv-sales-review/message_0/10',
                'trove://wechat/acct-work/conv-sales-review/message_0/12',
                'trove://wechat/acct-work/conv-sales-review/message_0/10',
                'trove://wechat/missing',
            ]
            rows = store.evidence_by_citations(citations)
            self.assertEqual(set(rows), set(citations[:2]))
            self.assertIn('价格太高', rows[citations[0]]['content'])
            self.assertEqual(rows[citations[1]]['direction'], 'outgoing')
