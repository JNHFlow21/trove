from __future__ import annotations
import unittest
import tempfile
from pathlib import Path

from trove_core.search.evaluation import evaluate_golden
from trove_core.search.hyper_search import HyperSearch
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class RetrievalEvalTests(unittest.TestCase):
    def test_golden_eval_reports_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            search = HyperSearch(SQLiteStore(Path(d) / 'index' / 'trove.sqlite'))
            metrics = evaluate_golden(search, Path('tests/golden/search_queries.jsonl'), k=3)
            self.assertEqual(metrics['queries'], 3)
            self.assertGreaterEqual(metrics['recall_at_3'], 2/3)
            self.assertGreater(metrics['mrr'], 0)
            self.assertEqual(metrics['evidence_completeness'], 1.0)
            rendered = str(metrics)
            self.assertNotIn('top_citations', rendered)
            self.assertNotIn('trove://wechat/acct-work', rendered)
            self.assertFalse(metrics['privacy']['raw_citations_included'])
