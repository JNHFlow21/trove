from __future__ import annotations
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.embedding.base import HybridEmbedding
from trove_core.search.episodes import EpisodeHit
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class HyperSearchTests(unittest.TestCase):
    def make_search(self, tmp):
        index_fixture_vault(Path(tmp), reset=True)
        return HyperSearch(SQLiteStore(Path(tmp) / 'index' / 'trove.sqlite'))

    def test_exact_phrase_ranks_before_loose_matches(self):
        with tempfile.TemporaryDirectory() as d:
            resp = self.make_search(d).search(SearchRequest('价格太高', limit=5))
            self.assertGreater(resp.total, 0)
            top = resp.results[0]
            self.assertIn('exact', top.retrieval_paths)
            self.assertIn('价格太高', top.snippet)

    def test_group_filter_excludes_other_conversations(self):
        with tempfile.TemporaryDirectory() as d:
            resp = self.make_search(d).search(SearchRequest('决定', conversation_id='conv-trove-team', limit=10))
            self.assertTrue(resp.results)
            self.assertEqual({r.conversation_id for r in resp.results}, {'conv-trove-team'})

    def test_sender_filter_respects_person_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            resp = self.make_search(d).search(SearchRequest('报价', sender='我', limit=10))
            self.assertTrue(resp.results)
            self.assertEqual({r.sender_name for r in resp.results}, {'我'})

    def test_no_vector_index_still_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            resp = self.make_search(d).search(SearchRequest('ZVEC', limit=3))
            self.assertIn('vector', resp.retrieval_status)
            self.assertFalse(resp.retrieval_status['vector']['available'])

    def test_ambiguous_terms_do_not_duplicate_citations(self):
        with tempfile.TemporaryDirectory() as d:
            resp = self.make_search(d).search(SearchRequest('客户', limit=10))
            citations = [r.citation for r in resp.results]
            self.assertEqual(len(citations), len(set(citations)))

    def test_media_hints_are_lazy_and_batched(self):
        class MediaHintSpyStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.media_hint_calls = 0
                self.media_hint_batches: list[list[str]] = []

            def chunk_search(self, *_args, **_kwargs):
                return []

            def fts_search_filtered(self, *_args, **_kwargs):
                return []

            def media_hints_for_citations(self, citations):
                batch = list(citations)
                self.media_hint_calls += 1
                self.media_hint_batches.append(batch)
                return {citation: {'type': 'image', 'raw_paths_included': False} for citation in batch}

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = MediaHintSpyStore(Path(d) / 'index' / 'trove.sqlite')

            without_hints = HyperSearch(store).search(SearchRequest('价格太高', limit=2, semantic='off'))
            self.assertTrue(without_hints.results)
            self.assertEqual(store.media_hint_calls, 0)
            self.assertIsNone(without_hints.results[0].media_hint)

            with_hints = HyperSearch(store).search(SearchRequest('价格太高', limit=2, semantic='off', include_media_hints=True))
            self.assertTrue(with_hints.results)
            self.assertEqual(store.media_hint_calls, 1)
            self.assertGreaterEqual(len(store.media_hint_batches[0]), len(with_hints.results))
            self.assertEqual(with_hints.results[0].media_hint['type'], 'image')

    def test_available_vector_is_additive_and_preserves_evidence_routes(self):
        class SpyStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.exact_calls = 0
                self.supplemental_calls: list[str] = []

            def exact_search(self, *_args, **_kwargs):
                self.exact_calls += 1
                return super().exact_search(*_args, **_kwargs)

            def metadata_search(self, *_args, **_kwargs):
                self.supplemental_calls.append('metadata')
                return super().metadata_search(*_args, **_kwargs)

            def multisource_search(self, *_args, **_kwargs):
                self.supplemental_calls.append('evidence')
                return super().multisource_search(*_args, **_kwargs)

            def chunk_search(self, *_args, **_kwargs):
                self.supplemental_calls.append('chunk')
                return super().chunk_search(*_args, **_kwargs)

        class FixedVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                return self.rows[:limit]

        class DummyProvider:
            def embed(self, _text):
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SpyStore(Path(d) / 'index' / 'trove.sqlite')
            vector_rows = store.all_messages()
            resp = HyperSearch(
                store,
                vector_store=FixedVector(vector_rows),
                embedding_provider=DummyProvider(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('客户预算', limit=3, semantic='on'))

            self.assertTrue(resp.results)
            self.assertEqual(store.exact_calls, 1)
            self.assertEqual(set(store.supplemental_calls), {'chunk'})
            self.assertEqual(resp.retrieval_status['vector']['state'], 'available')
            self.assertEqual(resp.retrieval_status['retrieval_plan']['route_policy'], 'trigram_evidence_first_vector_additive')
            self.assertTrue(resp.retrieval_status['retrieval_plan']['exact_route_executed'])
            self.assertFalse(resp.retrieval_status['retrieval_plan']['vector_replacement_enabled'])
            self.assertEqual(resp.retrieval_status['retrieval_plan']['first_stage_routes'], ['exact', 'evidence', 'vector', 'fts'])
            self.assertIn('vector', resp.retrieval_status['ranking']['candidate_routes'])

    def test_vector_only_candidate_cannot_displace_exact_lexical_anchor(self):
        class ExactOnlyStore(SQLiteStore):
            def __init__(self, path, target):
                super().__init__(path)
                self.target = target

            def exact_search(self, *_args, **_kwargs):
                return [self.target]

            def chunk_search(self, *_args, **_kwargs):
                return []

            def fts_search_filtered(self, *_args, **_kwargs):
                return []

        class FixedVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                return self.rows[:limit]

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            base = SQLiteStore(vault / 'index' / 'trove.sqlite')
            rows = base.all_messages()
            target = rows[-1]
            wrong = [row for row in rows if row['citation'] != target['citation']]
            store = ExactOnlyStore(vault / 'index' / 'trove.sqlite', target)

            response = HyperSearch(
                store,
                vector_store=FixedVector(wrong),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('synthetic rewrite query', limit=3, semantic='on'))

            self.assertEqual(response.results[0].citation, target['citation'])

    def test_additive_vector_still_returns_when_lexical_stage_is_empty(self):
        class EmptyLexicalStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.supplemental_calls: list[str] = []

            def exact_search(self, *_args, **_kwargs):
                return []

            def fts_search_filtered(self, *_args, **_kwargs):
                return []

            def metadata_search(self, *_args, **_kwargs):
                self.supplemental_calls.append('metadata')
                return []

            def multisource_search(self, *_args, **_kwargs):
                self.supplemental_calls.append('evidence')
                return []

            def chunk_search(self, *_args, **_kwargs):
                self.supplemental_calls.append('chunk')
                return []

        class FixedVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                return self.rows[:limit]

        class DummyProvider:
            def embed(self, _text):
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = EmptyLexicalStore(Path(d) / 'index' / 'trove.sqlite')
            vector_rows = store.all_messages()
            resp = HyperSearch(
                store,
                vector_store=FixedVector(vector_rows),
                embedding_provider=DummyProvider(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('semantic only', limit=3, semantic='on'))

            self.assertTrue(resp.results)
            self.assertEqual(set(store.supplemental_calls), {'chunk'})
            self.assertEqual(resp.retrieval_status['retrieval_plan']['first_stage_routes'], ['exact', 'evidence', 'vector', 'fts'])

    def test_semantic_auto_skips_vector_when_lexical_candidates_suffice(self):
        class FailingVector:
            def search(self, *_args, **_kwargs):
                raise AssertionError('vector should be gated off')

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            resp = HyperSearch(
                store,
                vector_store=FailingVector(),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('价格太高', limit=1, semantic='auto'))

            self.assertTrue(resp.results)
            self.assertEqual(resp.retrieval_status['vector']['reason_code'], 'semantic_auto_satisfied')
            self.assertNotIn('vector', resp.retrieval_status['ranking']['candidate_routes'])

    def test_semantic_auto_keeps_vector_additive_for_rewrite_queries(self):
        class RewriteStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.calls = []

            def chunk_search(self, *_args, **_kwargs):
                self.calls.append('chunk')
                return super().chunk_search(*_args, **_kwargs)

            def exact_search(self, *_args, **_kwargs):
                self.calls.append('exact')
                return super().exact_search(*_args, **_kwargs)

            def fts_search_filtered(self, *_args, **_kwargs):
                self.calls.append('fts')
                return super().fts_search_filtered(*_args, **_kwargs)

        class FixedVector:
            def __init__(self, rows):
                self.rows = rows
                self.calls = 0

            def search(self, _query, filters=None, limit=10, provider=None):
                self.calls += 1
                return self.rows[:limit]

        class DummyProvider:
            def embed(self, _text):
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = RewriteStore(Path(d) / 'index' / 'trove.sqlite')
            vector = FixedVector(store.all_messages())
            resp = HyperSearch(
                store,
                vector_store=vector,
                embedding_provider=DummyProvider(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('最近需要跟进哪些事情', limit=3, semantic='auto'))

            self.assertTrue(resp.results)
            self.assertEqual(vector.calls, 1)
            self.assertEqual(store.calls, ['exact', 'chunk', 'fts'])
            self.assertEqual(resp.retrieval_status['retrieval_plan']['route_policy'], 'trigram_evidence_first_vector_additive')
            self.assertEqual(resp.retrieval_status['retrieval_plan']['first_stage_routes'], ['exact', 'evidence', 'vector', 'fts'])
            self.assertTrue(resp.retrieval_status['retrieval_plan']['exact_route_executed'])
            self.assertEqual(resp.retrieval_status['vector']['state'], 'available')
            self.assertTrue(resp.retrieval_status['retrieval_plan']['vector_candidates_sufficient'])
            self.assertFalse(resp.retrieval_status['retrieval_plan']['vector_catchup_pending'])

    def test_semantic_auto_skips_vector_for_structured_non_message_source(self):
        class ForbiddenVector:
            def search(self, *_args, **_kwargs):
                raise AssertionError('structured non-message auto search must not scan ZVEC residuals')

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            response = HyperSearch(
                store,
                vector_store=ForbiddenVector(),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('音频纪要线索', source_type='transcript', semantic='auto'))

            self.assertFalse(response.retrieval_status['vector']['attempted'])
            self.assertEqual(
                response.retrieval_status['vector']['reason_code'],
                'semantic_auto_structured_source',
            )

    def test_explicit_multi_hop_query_adds_bounded_conversation_neighbors(self):
        class MultiHopStore(SQLiteStore):
            def __init__(self, path, anchor, neighbor):
                super().__init__(path)
                self.anchor = anchor
                self.neighbor = neighbor
                self.context_calls = 0

            def exact_search(self, *_args, **_kwargs):
                return [self.anchor]

            def chunk_search(self, *_args, **_kwargs):
                return []

            def fts_search_filtered(self, *_args, **_kwargs):
                return []

            def context_window(self, citation, before=5, after=5):
                self.context_calls += 1
                self.asserted_citation = citation
                return [self.anchor, self.neighbor]

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            base = SQLiteStore(vault / 'index' / 'trove.sqlite')
            rows = base.all_messages()
            anchor = rows[0]
            neighbor = next(
                row
                for row in rows[1:]
                if row['account_id'] == anchor['account_id']
                and row['conversation_id'] == anchor['conversation_id']
            )
            store = MultiHopStore(vault / 'index' / 'trove.sqlite', anchor, neighbor)

            response = HyperSearch(store).search(
                SearchRequest('关联脉络 前后因果', limit=3, semantic='off')
            )

            self.assertEqual(store.context_calls, 1)
            self.assertEqual(
                {result.citation for result in response.results[:2]},
                {anchor['citation'], neighbor['citation']},
            )
            self.assertIn('conversation-context', response.results[1].retrieval_paths)
            self.assertTrue(response.retrieval_status['retrieval_plan']['multi_hop_expansion'])

    def test_semantic_rewrite_merges_lexical_routes_while_vector_catchup_is_pending(self):
        class PartialCatchupStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.target_rows = []
                self.exact_calls = 0
                self.chunk_calls = 0
                self.fts_calls = 0

            def exact_search(self, *_args, limit=10, **_kwargs):
                self.exact_calls += 1
                return self.target_rows[:limit]

            def chunk_search(self, *_args, limit=10, **_kwargs):
                self.chunk_calls += 1
                return self.target_rows[:limit]

            def fts_search_filtered(self, *_args, limit=10, **_kwargs):
                self.fts_calls += 1
                return self.target_rows[:limit]

        class PartialVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                return self.rows[:limit]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = PartialCatchupStore(Path(d) / 'index' / 'trove.sqlite')
            target = SQLiteStore.exact_search(store, '价格太高', limit=1)[0]
            store.target_rows = [target]
            wrong_rows = [row for row in store.all_messages() if row['citation'] != target['citation']]
            self.assertGreaterEqual(len(wrong_rows), 2)

            response = HyperSearch(
                store,
                vector_store=PartialVector(wrong_rows),
                embedding_provider=object(),
                vector_status={
                    'state': 'available',
                    'selected_backend': 'zvec',
                    'zvec': {'catchup_pending': True},
                },
            ).search(SearchRequest('最近需要跟进哪些事情', limit=2, semantic='auto'))

            self.assertIn(target['citation'], {row.citation for row in response.results})
            self.assertEqual((store.exact_calls, store.chunk_calls, store.fts_calls), (1, 1, 1))
            plan = response.retrieval_status['retrieval_plan']
            self.assertFalse(plan['semantic_first'])
            self.assertTrue(plan['vector_catchup_pending'])
            self.assertTrue(plan['vector_catchup_lexical_merge'])
            self.assertEqual(plan['first_stage_routes'], ['exact', 'evidence', 'vector', 'fts'])
            self.assertTrue(plan['fts_route_executed'])
            self.assertIsNone(plan['fts_route_skipped_reason'])

    def test_semantic_first_vector_failure_falls_back_to_lexical_routes(self):
        class LexicalFallbackStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.exact_calls = 0

            def exact_search(self, *_args, **_kwargs):
                self.exact_calls += 1
                return self.all_messages()[:1]

            def chunk_search(self, *_args, **_kwargs):
                return []

            def fts_search_filtered(self, *_args, **_kwargs):
                return []

        class FailingVector:
            def search(self, *_args, **_kwargs):
                raise RuntimeError('synthetic vector outage')

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = LexicalFallbackStore(Path(d) / 'index' / 'trove.sqlite')
            response = HyperSearch(
                store,
                vector_store=FailingVector(),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('最近需要跟进哪些事情', limit=3, semantic='auto'))

            self.assertTrue(response.results)
            self.assertEqual(store.exact_calls, 1)
            self.assertEqual(response.retrieval_status['vector']['state'], 'degraded')
            plan = response.retrieval_status['retrieval_plan']
            self.assertFalse(plan['semantic_first'])
            self.assertTrue(plan['vector_fallback_to_lexical'])
            self.assertTrue(plan['exact_route_executed'])

    def test_rewrite_depth_and_stage_budgets_are_independent(self):
        class RecordingVector:
            def __init__(self, rows):
                self.rows = rows
                self.limits: list[int] = []

            def search(self, _query, filters=None, limit=10, provider=None):
                self.limits.append(limit)
                return self.rows[:limit]

        class DummyProvider:
            def embed(self, _text):
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            vector = RecordingVector(store.all_messages())
            search = HyperSearch(
                store,
                vector_store=vector,
                embedding_provider=DummyProvider(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            )

            deep = search.search(SearchRequest('最近需要跟进哪些事情', limit=3, semantic='auto'))
            self.assertEqual(vector.limits[-1], 200)
            self.assertEqual(deep.retrieval_status['candidate_budgets']['retrieval']['requested_limit'], 200)

            bounded = search.search(SearchRequest(
                '最近需要跟进哪些事情',
                limit=3,
                semantic='auto',
                ranking_mode='feature',
                reranker_mode='features',
                retrieval_candidate_limit=7,
                fusion_candidate_limit=5,
                reranker_candidate_limit=3,
            ))
            budgets = bounded.retrieval_status['candidate_budgets']
            self.assertEqual(vector.limits[-1], 7)
            self.assertLessEqual(budgets['retrieval']['max_route_candidates'], 7)
            self.assertLessEqual(budgets['fusion']['output_candidates'], 5)
            self.assertLessEqual(budgets['rerank']['input_candidates'], 3)
            self.assertTrue(bounded.candidate_citations)
            self.assertNotIn('candidate_citations', bounded.to_dict())

    def test_exact_route_never_invokes_local_model_reranker(self):
        with tempfile.TemporaryDirectory() as d:
            search = self.make_search(d)
            with patch(
                'trove_core.search.hyper_search.rerank_with_local_model',
                side_effect=AssertionError('local model reranker must stay off exact path'),
            ):
                response = search.search(SearchRequest(
                    '价格太高',
                    semantic='off',
                    ranking_mode='feature',
                    reranker_mode='local-bge',
                ))
            self.assertTrue(response.results)
            self.assertEqual(response.retrieval_status['reranker']['state'], 'skipped')
            self.assertFalse(response.retrieval_status['reranker']['invoked'])
            phases = response.retrieval_status['phase_latency_ms']
            self.assertEqual(set(phases), {'retrieval', 'fusion', 'rerank'})
            self.assertTrue(all(isinstance(value, float) and value >= 0.0 for value in phases.values()))

    def test_semantic_route_invokes_local_reranker_window_only(self):
        class FixedVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                return self.rows[:limit]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            search = HyperSearch(
                store,
                vector_store=FixedVector(store.all_messages()),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            )
            calls: list[int] = []

            def invoked(ranked, _query, **kwargs):
                calls.append(len(ranked))
                return ranked, {
                    'state': 'available',
                    'mode': 'local-bge',
                    'candidate_count': len(ranked),
                    'invoked': True,
                }

            with patch('trove_core.search.hyper_search.rerank_with_local_model', side_effect=invoked):
                response = search.search(SearchRequest(
                    '最近需要跟进哪些事情',
                    semantic='auto',
                    ranking_mode='feature',
                    reranker_mode='local-bge',
                    reranker_model_path='/synthetic/local/model',
                    reranker_candidate_limit=3,
                ))

            self.assertEqual(calls, [3])
            self.assertTrue(response.retrieval_status['reranker']['invoked'])
            self.assertLessEqual(response.retrieval_status['reranker']['candidate_count'], 3)
            self.assertGreaterEqual(response.retrieval_status['phase_latency_ms']['rerank'], 0.0)

    def test_query_embedding_cache_reuses_rewrite_embedding(self):
        class EmbeddingVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, query, filters=None, limit=10, provider=None):
                provider.embed_query(query)
                return self.rows[:limit]

        class CountingProvider:
            dimensions = 1

            def __init__(self):
                self.calls = 0

            def embed_query(self, _text):
                self.calls += 1
                return [1.0]

            def embed(self, _text):
                self.calls += 1
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            provider = CountingProvider()
            search = HyperSearch(
                store,
                vector_store=EmbeddingVector(store.all_messages()),
                embedding_provider=provider,
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            )

            first = search.search(SearchRequest('最近需要跟进哪些事情', limit=3, semantic='auto'))
            second = search.search(SearchRequest('最近需要跟进哪些事情', limit=3, semantic='auto'))

            self.assertTrue(first.results)
            self.assertTrue(second.results)
            self.assertEqual(provider.calls, 1)
            self.assertEqual(second.retrieval_status['vector']['cache']['hits'], 1)
            self.assertEqual(second.retrieval_status['vector']['cache']['misses'], 1)

    def test_hybrid_query_cache_keeps_message_and_episode_instructions_distinct(self):
        class HybridVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, query, filters=None, limit=10, provider=None):
                provider.embed_query_hybrid(query)
                return self.rows[:limit]

        class EmptyEpisodeStore:
            def search(self, query, *, provider, filters=None, limit=3):
                provider.embed_hybrid_many(
                    [query], text_type='query', instruct='fixture-episode-instruction'
                )
                return []

        class CountingHybridProvider:
            dimensions = 2

            def __init__(self):
                self.calls: list[tuple[str, str | None]] = []

            def embed_query_hybrid(self, _text):
                self.calls.append(('query', None))
                return HybridEmbedding([1.0, 0.0], {1: 1.0})

            def embed_hybrid_many(self, texts, *, text_type='document', instruct=None):
                self.calls.append((text_type, instruct))
                return [HybridEmbedding([1.0, 0.0], {1: 1.0}) for _ in texts]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            provider = CountingHybridProvider()
            search = HyperSearch(
                store,
                vector_store=HybridVector(store.all_messages()),
                embedding_provider=provider,
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
                episode_store=EmptyEpisodeStore(),
            )
            request = SearchRequest('价格变化的前因后果', limit=3, semantic='on')

            first = search.search(request)
            second = search.search(request)

            self.assertTrue(first.results)
            self.assertTrue(second.results)
            self.assertCountEqual(
                provider.calls,
                [('query', None), ('query', 'fixture-episode-instruction')],
            )
            self.assertEqual(second.retrieval_status['vector']['cache']['hits'], 2)
            self.assertEqual(second.retrieval_status['vector']['cache']['misses'], 2)

    def test_multi_hop_episode_route_overlaps_message_route_and_builds_bounded_bundles(self):
        episode_entered = threading.Event()
        vector_entered = threading.Event()

        class OverlapVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                vector_entered.set()
                episode_entered.wait(2.0)
                return self.rows[:limit]

        class OverlapEpisodeStore:
            overlapped = False
            requested_limit = None

            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, *, provider, filters=None, limit=3):
                self.requested_limit = limit
                episode_entered.set()
                self.overlapped = vector_entered.wait(2.0)
                return [
                    EpisodeHit('episode-a', tuple(row['citation'] for row in self.rows[:2]), 1.0),
                    EpisodeHit('episode-b', (self.rows[1]['citation'], self.rows[2]['citation']), 0.9),
                ]

        class FixedSelector:
            calls = 0

            def select(self, _query, rows, *, candidate_metadata=None):
                self.calls += 1
                raise AssertionError('episode bundle path must not invoke the generative selector')

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            rows = store.all_messages()
            episode_store = OverlapEpisodeStore(rows)
            selector = FixedSelector()
            search = HyperSearch(
                store,
                vector_store=OverlapVector(rows),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
                episode_store=episode_store,
                evidence_selector=selector,
            )

            response = search.search(
                SearchRequest('价格变化的前因后果', limit=3, semantic='on')
            )

            self.assertTrue(episode_store.overlapped)
            self.assertEqual(episode_store.requested_limit, 10)
            self.assertTrue(response.results)
            self.assertEqual(len(response.episode_bundles), 2)
            self.assertEqual(response.episode_bundles[0].evidence_kind, 'episode')
            self.assertIn('evidence', response.episode_bundles[0].retrieval_paths)
            self.assertTrue(response.episode_bundles[0]._rerank_text)
            self.assertNotIn('_rerank_text', response.episode_bundles[0].to_dict())
            self.assertEqual(
                response.episode_bundles[0].supporting_citations,
                tuple(row['citation'] for row in rows[:2]),
            )
            self.assertEqual(selector.calls, 0)
            episode_status = response.retrieval_status['retrieval_plan']['multi_hop_episode']
            self.assertTrue(episode_status['parallel_execution'])
            self.assertEqual(episode_status['bundle_count'], 2)
            self.assertEqual(episode_status['selected_chain_count'], 0)
            self.assertEqual(
                episode_status['selector']['reason_code'],
                'episode_bundle_rerank_deferred',
            )

    def test_warm_query_path_preheats_vector_without_private_text(self):
        class EmbeddingVector:
            def __init__(self):
                self.calls: list[tuple[str, int]] = []

            def search(self, query, filters=None, limit=10, provider=None):
                provider.embed_query(query)
                self.calls.append((query, limit))
                return []

        class CountingProvider:
            dimensions = 1

            def __init__(self):
                self.calls = 0

            def embed_query(self, _text):
                self.calls += 1
                return [1.0]

            def embed(self, _text):
                self.calls += 1
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            provider = CountingProvider()
            vector = EmbeddingVector()
            search = HyperSearch(
                SQLiteStore(Path(d) / 'index' / 'trove.sqlite'),
                vector_store=vector,
                embedding_provider=provider,
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            )

            status = search.warm_query_path()

            self.assertTrue(status['ok'])
            self.assertFalse(status['private_text_used'])
            self.assertEqual(provider.calls, 1)
            self.assertEqual(vector.calls, [('trove search warmup', 1)])
            self.assertEqual(status['cache']['misses'], 1)

    def test_filtered_rewrite_caps_fts_and_prefilters_sender_exact(self):
        class RewriteStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.exact_like_flags: list[bool] = []
                self.exact_strict_sender_flags: list[bool] = []
                self.exact_sender_prefilter_flags: list[bool] = []
                self.chunk_like_flags: list[bool] = []
                self.chunk_actor_prefilter_flags: list[bool] = []
                self.fts_queries: list[tuple[str, int | None]] = []

            def exact_search(self, query, filters=None, limit=10, *, allow_like_fallback=True, strict_sender_match=False, sender_prefilter=False):
                self.exact_like_flags.append(allow_like_fallback)
                self.exact_strict_sender_flags.append(strict_sender_match)
                self.exact_sender_prefilter_flags.append(sender_prefilter)
                return []

            def chunk_search(self, query, filters=None, limit=10, *, allow_like_fallback=True, actor_prefilter=False):
                self.chunk_like_flags.append(allow_like_fallback)
                self.chunk_actor_prefilter_flags.append(actor_prefilter)
                return []

            def fts_search_filtered(self, query, filters=None, limit=10, **_kwargs):
                self.fts_queries.append((query, limit))
                return []

        class EmptyVector:
            def search(self, _query, filters=None, limit=10, provider=None):
                return []

        class DummyProvider:
            def embed(self, _text):
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            store = RewriteStore(Path(d) / 'index' / 'trove.sqlite')
            resp = HyperSearch(
                store,
                vector_store=EmptyVector(),
                embedding_provider=DummyProvider(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('最近需要谁跟进', sender='我', limit=3, semantic='auto'))

            self.assertEqual(store.exact_like_flags, [True])
            self.assertEqual(store.exact_strict_sender_flags, [False])
            self.assertEqual(store.exact_sender_prefilter_flags, [True])
            self.assertEqual(store.chunk_like_flags, [True])
            self.assertEqual(store.chunk_actor_prefilter_flags, [True])
            self.assertEqual(len(store.fts_queries), 1)
            self.assertLessEqual(store.fts_queries[0][1] or 0, 6)
            self.assertEqual(resp.retrieval_status['retrieval_plan']['fts_route_query_count'], 1)

    def test_filtered_rewrite_forces_semantic_without_replacing_lexical_paths(self):
        class CountingStore(SQLiteStore):
            def __init__(self, path):
                super().__init__(path)
                self.calls: list[str] = []
                self.limits: dict[str, list[int]] = {'exact': [], 'chunk': [], 'fts': []}

            def exact_search(self, *_args, **kwargs):
                self.calls.append('exact')
                self.limits['exact'].append(kwargs.get('limit'))
                return super().exact_search(*_args, **kwargs)

            def chunk_search(self, *_args, **kwargs):
                self.calls.append('chunk')
                self.limits['chunk'].append(kwargs.get('limit'))
                return super().chunk_search(*_args, **kwargs)

            def fts_search_filtered(self, *_args, **kwargs):
                self.calls.append('fts')
                self.limits['fts'].append(kwargs.get('limit'))
                return super().fts_search_filtered(*_args, **kwargs)

        class FixedVector:
            def __init__(self, rows):
                self.rows = rows

            def search(self, _query, filters=None, limit=10, provider=None):
                return self.rows[:limit]

        class DummyProvider:
            def embed(self, _text):
                return [1.0]

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = CountingStore(Path(d) / 'index' / 'trove.sqlite')
            resp = HyperSearch(
                store,
                vector_store=FixedVector(store.all_messages()),
                embedding_provider=DummyProvider(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('最近需要谁跟进', sender='我', limit=3, semantic='auto'))

            self.assertTrue(resp.results)
            self.assertIn('exact', store.calls)
            self.assertIn('chunk', store.calls)
            self.assertIn('fts', store.calls)
            self.assertLessEqual(max(store.limits['exact']), 10)
            self.assertLessEqual(max(store.limits['chunk']), 10)
            self.assertTrue(resp.retrieval_status['retrieval_plan']['semantic_forced'])
            self.assertTrue(resp.retrieval_status['retrieval_plan']['semantic_priority'])
            self.assertFalse(resp.retrieval_status['retrieval_plan']['semantic_first'])
            self.assertIn('vector', resp.retrieval_status['retrieval_plan']['first_stage_routes'])
            self.assertIsNone(resp.retrieval_status['retrieval_plan']['fts_route_skipped_reason'])
