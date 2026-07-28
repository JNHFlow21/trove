from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.search.hyper_search import HyperSearch
from trove_core.search.fusion import fuse_ranked_rows
from trove_core.search.query import SearchRequest
from trove_core.search.query_understanding import analyze_query
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message
from trove_core.wechat.indexer import index_fixture_vault


class SearchRankingContractTests(unittest.TestCase):
    def make_search(self, tmp):
        index_fixture_vault(Path(tmp), reset=True)
        store = SQLiteStore(Path(tmp) / 'index' / 'trove.sqlite')
        store.rebuild_evidence_chunks()
        return HyperSearch(store)

    def test_query_understanding_expands_domain_terms(self):
        q = analyze_query('客户预算审批')
        self.assertIn('客户', q.terms)
        self.assertIn('commercial_blocker', q.intents)
        self.assertGreaterEqual(q.expansion_count if hasattr(q, 'expansion_count') else len(q.expansions), 1)

    def test_default_search_uses_deterministic_feature_ranking(self):
        request = SearchRequest('默认排序')

        self.assertEqual(request.ranking_mode, 'feature')
        self.assertEqual(request.reranker_mode, 'features')
        self.assertEqual(request.effective_ranking_mode, 'feature')

    def test_multi_hop_understanding_preserves_space_separated_anchor_terms(self):
        q = analyze_query('预算 试点 关联脉络')

        self.assertIn('预算', q.terms)
        self.assertIn('试点', q.terms)
        self.assertIn('multi_hop', q.intents)

    def test_rrf_ranking_status_and_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            resp = self.make_search(d).search(SearchRequest('客户预算', limit=5, ranking_mode='rrf'))
            self.assertEqual(resp.retrieval_status['ranking']['base_ranker'], 'rrf')
            citations = [r.citation for r in resp.results]
            self.assertEqual(len(citations), len(set(citations)))

    def test_fusion_merges_chunk_child_with_parent_message_routes(self):
        parent = {'citation': 'parent-1', 'timestamp': '2026-01-01T00:00:00Z'}
        child = {'citation': 'parent-1#chunk-0', 'parent_citation': 'parent-1', 'timestamp': '2026-01-01T00:00:00Z'}

        fused = fuse_ranked_rows([
            ('exact', [parent], 2.0),
            ('evidence', [child], 10.0),
            ('vector', [child], 10.0),
        ], limit=5)

        self.assertEqual(len(fused), 1)
        self.assertEqual(set(fused[0][1]), {'exact', 'evidence', 'vector'})

    def test_feature_and_local_reranker_contracts(self):
        with tempfile.TemporaryDirectory() as d:
            search = self.make_search(d)
            feature = search.search(SearchRequest('价格太高', limit=5, ranking_mode='feature', reranker_mode='features'))
            self.assertEqual(feature.retrieval_status['reranker']['state'], 'available')
            self.assertIn('exact_phrase', feature.retrieval_status['reranker']['reason_codes'])

            local = search.search(SearchRequest('价格太高', limit=5, ranking_mode='feature', reranker_mode='local-bge'))
            self.assertEqual(local.retrieval_status['reranker']['state'], 'skipped')
            self.assertEqual(local.retrieval_status['reranker']['fallback_mode'], 'features')
            self.assertEqual(local.retrieval_status['reranker']['reason_code'], 'local_reranker_requires_semantic_route')
            self.assertTrue(local.results)

            cloud = search.search(SearchRequest('价格太高', limit=5, ranking_mode='feature', reranker_mode='cloud-qwen3'))
            self.assertEqual(cloud.retrieval_status['reranker']['state'], 'unavailable_fallback')
            self.assertEqual(cloud.retrieval_status['reranker']['fallback_mode'], 'features')
            self.assertEqual(cloud.retrieval_status['reranker']['reason_code'], 'cloud_reranker_requires_exact_approval')
            self.assertTrue(cloud.results)

    def test_query_expansion_only_fans_out_fts_route(self):
        class CountingStore:
            def __init__(self):
                self.calls = {'exact': [], 'fts': [], 'metadata': [], 'evidence': [], 'chunk': []}

            def exact_search(self, query, **_kwargs):
                self.calls['exact'].append(query)
                return []

            def fts_search_filtered(self, query, **_kwargs):
                self.calls['fts'].append(query)
                return []

            def metadata_search(self, query, **_kwargs):
                self.calls['metadata'].append(query)
                return []

            def multisource_search(self, query, **_kwargs):
                self.calls['evidence'].append(query)
                return []

            def chunk_search(self, query, **_kwargs):
                self.calls['chunk'].append(query)
                return []

        store = CountingStore()
        HyperSearch(store).search(SearchRequest('客户预算审批', conversation_id='target', limit=5, ranking_mode='feature'))

        self.assertEqual(store.calls['exact'], ['客户预算审批'])
        self.assertLessEqual(len(store.calls['fts']), 3)
        self.assertEqual(store.calls['metadata'], [])
        self.assertEqual(store.calls['evidence'], [])
        self.assertEqual(store.calls['chunk'], ['客户预算审批'])

    def test_multiword_query_keeps_fts_route_and_ranks_term_cooccurrence(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            store.upsert_accounts([Account('acct-a', 'A', 'A')])
            store.upsert_conversations([Conversation('conv-a', 'acct-a', 'A', 'private')])
            store.upsert_messages([
                Message('acct-a', 'A', 'conv-a', 'A', 'private', 'u', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '只谈预算', 's', 1),
                Message('acct-a', 'A', 'conv-a', 'A', 'private', 'u', '客户', datetime(2026, 1, 2, tzinfo=timezone.utc), '只谈审批', 's', 2),
                Message('acct-a', 'A', 'conv-a', 'A', 'private', 'u', '客户', datetime(2026, 1, 3, tzinfo=timezone.utc), '预算审批一起推进', 's', 3),
            ])
            resp = HyperSearch(store).search(SearchRequest('预算 审批', limit=1, semantic='off'))

            self.assertTrue(resp.retrieval_status['retrieval_plan']['fts_route_executed'])
            self.assertEqual(resp.retrieval_status['reranker']['mode'], 'features')
            self.assertIn('预算审批一起推进', resp.results[0].snippet)


if __name__ == '__main__':
    unittest.main()
