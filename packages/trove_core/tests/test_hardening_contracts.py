from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.approvals import ApprovalManager, ApprovalRequired
from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.knowledge.wiki import build_wiki_page
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.change_journal import clear_all_dirty_citations, dirty_citation_count
from trove_core.vault.tracing import TraceTimeline
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.process_config import process_config_from_payload


class HardeningContractTests(unittest.TestCase):
    def test_process_config_validation_blocks_cloud_without_gate(self):
        cfg = process_config_from_payload({'allow_cloud_asr': True, 'multimodal': 'metadata_only'})
        self.assertTrue(cfg.validate())
        self.assertIn('redacted_hash', cfg.to_dict())

    def test_trace_redacts_private_paths(self):
        with tempfile.TemporaryDirectory() as d:
            trace = TraceTimeline(d)
            tid = trace.start('import', {'path': '/' + 'Users' + '/example/private/vault'})
            trace.complete(tid, {'ok': True})
            text = str(trace.list())
            self.assertNotIn('/' + 'Users' + '/example/private/vault', text)
            self.assertIn('redacted-path', text)

    def test_approval_workflow_records_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            mgr = ApprovalManager(d)
            rec = mgr.request('vector_purge_rebuild', 'vector_purge_rebuild', {'backend': 'zvec'})
            with self.assertRaises(ApprovalRequired):
                mgr.require('vector_purge_rebuild', 'vector_purge_rebuild', {'backend': 'zvec'}, approval_id=rec.approval_id)
            approved = mgr.decide(rec.approval_id, 'approved')
            gate = mgr.require('vector_purge_rebuild', 'vector_purge_rebuild', {'backend': 'zvec'}, approval_id=approved.approval_id)
            self.assertEqual(gate['approval_status'], 'consumed')

    def test_evidence_chunks_and_wiki_are_cited(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            clear_all_dirty_citations(store)
            chunks = store.rebuild_evidence_chunks(max_chars=40, overlap_chars=5)
            self.assertGreater(chunks['chunks'], chunks['parents'])
            self.assertGreater(chunks['dirty_recorded'], 0)
            self.assertGreater(dirty_citation_count(store), 0)
            resp = HyperSearch(store).search(SearchRequest('价格太高', limit=5))
            self.assertTrue(any('evidence' in r.retrieval_paths for r in resp.results))
            page = build_wiki_page(store, '示例教育', limit=3)
            self.assertGreaterEqual(page['citation_count'], 1)
            self.assertEqual(page['uncited_claims'], 0)

    def test_vector_failure_reports_degraded_while_exact_returns(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            class BrokenVector:
                def search(self, *args, **kwargs):
                    raise RuntimeError('vector probe failed')
            resp = HyperSearch(store, vector_store=BrokenVector(), embedding_provider=FakeEmbeddingProvider()).search(SearchRequest('价格太高', limit=3, semantic='on'))
            self.assertTrue(resp.results)
            self.assertEqual(resp.retrieval_status['vector']['state'], 'degraded')


if __name__ == '__main__':
    unittest.main()
