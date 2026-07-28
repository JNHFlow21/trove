from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

from trove_core.bounds import (
    BoundedInputError,
    BoundedLimit,
    CONTEXT_WINDOW,
    FUSION_CANDIDATES,
    PRIVATE_LIST,
    RERANK_CANDIDATES,
    RETRIEVAL_CANDIDATES,
    SEARCH_RESULTS,
)
from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.agent_tools import tools as agent_tools
from trove_core.approvals import ApprovalManager
from trove_core.knowledge.report import build_cited_report
from trove_core.knowledge.wiki import build_wiki_page
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.tracing import TraceTimeline
from trove_core.wechat.files import list_conversation_files
from trove_core.vector.sqlite_vector_store import (
    SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT,
    SQLiteVectorStore,
    SQLiteVectorUnavailable,
)
from trove_core.wechat.indexer import index_fixture_vault


class BoundedLimitTests(unittest.TestCase):
    def test_declared_bounds_accept_endpoints_and_reject_coercion(self):
        self.assertEqual(BoundedLimit(1, spec=SEARCH_RESULTS), 1)
        self.assertEqual(BoundedLimit(50, spec=SEARCH_RESULTS), 50)
        self.assertEqual(BoundedLimit(spec=PRIVATE_LIST), 100)
        self.assertEqual(BoundedLimit(0, field='before', spec=CONTEXT_WINDOW), 0)
        for value in (0, -1, 51, True, 3.0, '3', None):
            with self.subTest(value=value), self.assertRaises(BoundedInputError) as raised:
                BoundedLimit(value, field='limit', spec=SEARCH_RESULTS)
            self.assertEqual(raised.exception.code, 'invalid_limit')
            self.assertEqual(raised.exception.to_dict()['field'], 'limit')

    def test_search_request_enforces_result_and_reranker_budgets(self):
        self.assertEqual(SearchRequest('fixture').limit, 10)
        request = SearchRequest(
            'fixture',
            limit=50,
            retrieval_candidate_limit=200,
            fusion_candidate_limit=200,
            reranker_candidate_limit=200,
        )
        self.assertEqual(request.limit, 50)
        self.assertEqual(request.retrieval_candidate_limit, RETRIEVAL_CANDIDATES.maximum)
        self.assertEqual(request.fusion_candidate_limit, FUSION_CANDIDATES.maximum)
        for kwargs, field in [
            ({'limit': 0}, 'limit'),
            ({'limit': 51}, 'limit'),
            ({'limit': True}, 'limit'),
            ({'retrieval_candidate_limit': 201}, 'retrieval_candidate_limit'),
            ({'fusion_candidate_limit': 201}, 'fusion_candidate_limit'),
            ({'reranker_candidate_limit': 201}, 'reranker_candidate_limit'),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(BoundedInputError) as raised:
                SearchRequest('fixture', **kwargs)
            self.assertEqual(raised.exception.field, field)

    def test_sqlite_vector_bulk_hydrates_without_n_plus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            provider = FakeEmbeddingProvider(dimensions=8)
            vectors = SQLiteVectorStore(store)
            self.assertGreater(vectors.index_all_messages(provider), 1)
            single_calls = 0
            vector_query_calls = 0
            original_single = store.evidence_by_citation
            original_bulk = store.evidence_by_citations
            original_vector_query = store.vector_entries_for_search

            def counted_single(citation):
                nonlocal single_calls
                single_calls += 1
                return original_single(citation)

            def counted_bulk(citations):
                return original_bulk(citations)

            def counted_vector_query(filters=None, *, limit):
                nonlocal vector_query_calls
                vector_query_calls += 1
                return original_vector_query(filters, limit=limit)

            store.evidence_by_citation = counted_single  # type: ignore[method-assign]
            store.evidence_by_citations = counted_bulk  # type: ignore[method-assign]
            store.vector_entries_for_search = counted_vector_query  # type: ignore[method-assign]
            statements: list[str] = []
            store.connect().set_trace_callback(statements.append)
            rows = vectors.search('预算审批', limit=3, provider=provider)
            store.connect().set_trace_callback(None)
            self.assertTrue(rows)
            self.assertEqual(single_calls, 0)
            self.assertEqual(vector_query_calls, 1)
            selects = [sql for sql in statements if sql.lstrip().upper().startswith(('SELECT', 'WITH'))]
            self.assertLessEqual(len(selects), 2, selects)
            with self.assertRaises(BoundedInputError):
                vectors.search('预算审批', limit=201, provider=provider)

    def test_sqlite_vector_pushes_filters_to_sql_and_disables_large_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            provider = FakeEmbeddingProvider(dimensions=8)
            vectors = SQLiteVectorStore(store)
            self.assertGreater(vectors.index_all_messages(provider), 1)

            original_filter = store._filter_row
            store._filter_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('post-filter used'))  # type: ignore[method-assign]
            try:
                rows = vectors.search(
                    '预算审批',
                    filters={'conversation_id': 'conv-example_edu-private'},
                    limit=3,
                    provider=provider,
                )
            finally:
                store._filter_row = original_filter  # type: ignore[method-assign]
            self.assertTrue(rows)
            self.assertEqual({row['conversation_id'] for row in rows}, {'conv-example_edu-private'})

            provider_calls = 0
            original_embed = provider.embed

            def counted_embed(text):
                nonlocal provider_calls
                provider_calls += 1
                return original_embed(text)

            provider.embed = counted_embed  # type: ignore[method-assign]
            original_entries = store.vector_entries_for_search
            store.vector_entries_for_search = lambda *_args, **_kwargs: [object()] * (SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT + 1)  # type: ignore[method-assign]
            try:
                with self.assertRaises(SQLiteVectorUnavailable):
                    vectors.search('fixture', provider=provider)
            finally:
                store.vector_entries_for_search = original_entries  # type: ignore[method-assign]
                provider.embed = original_embed  # type: ignore[method-assign]
            self.assertEqual(provider_calls, 0)

    def test_like_fallback_is_cardinality_guarded_and_cache_is_generation_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            store.initialize()
            self.assertLessEqual(len(store.exact_search('价', limit=2)), 2)
            self.assertLessEqual(len(store.exact_search('价格 太高', limit=2)), 2)

            connection = store.connect()
            initial = int(connection.execute('SELECT COUNT(*) FROM messages').fetchone()[0])
            self.assertTrue(store._allow_message_like_fallback(connection, {}, threshold=initial))
            with sqlite3.connect(store.path) as external:
                external.execute(
                    """INSERT INTO messages(
                        citation,account_id,account_label,conversation_id,conversation_title,
                        conversation_type,sender_id,sender_name,timestamp,content,content_kind,
                        shard_id,local_id,sent_by_me,source_type,direction
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'trove://synthetic/cache-generation', 'acct-work', 'work', 'conv-example_edu-private',
                        'Synthetic', 'private', 'sender', 'Sender', '2026-07-10T00:00:00Z',
                        'cache generation token', 'text', 'cache-shard', 999999, 0, 'message', 'incoming',
                    ),
                )
                external.commit()
            self.assertFalse(store._allow_message_like_fallback(connection, {}, threshold=initial))
            self.assertTrue(
                store._allow_message_like_fallback(
                    connection,
                    {'conversation_id': 'conv-trove-team'},
                    threshold=10,
                )
            )

            token = store._cardinality_cache_token(connection)
            store._count_cache['messages'] = (token, 50001)
            self.assertFalse(store._allow_message_like_fallback(connection, {}, threshold=50000))

    def test_list_profile_trace_event_and_approval_budgets_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            calls = [
                lambda value: store.list_conversations(value),
                lambda value: store.list_contacts(value),
                lambda value: store.list_moments(value),
                lambda value: store.list_favorites(value),
                lambda value: build_wiki_page(store, 'fixture', limit=value),
                lambda value: build_cited_report(store, 'fixture', limit=value),
                lambda value: TraceTimeline(vault).list(limit=value),
                lambda value: ApprovalManager(vault).list(limit=value),
            ]
            for value in (0, -1, True, '3'):
                for call in calls:
                    with self.subTest(value=value, call=call), self.assertRaises(BoundedInputError) as raised:
                        call(value)
                    self.assertEqual(raised.exception.code, 'invalid_limit')
            with self.assertRaises(BoundedInputError):
                store.list_contacts(501)
            with self.assertRaises(BoundedInputError):
                build_wiki_page(store, 'fixture', limit=51)
            with self.assertRaises(BoundedInputError):
                TraceTimeline(vault).list(limit=201)

    def test_list_queries_push_limit_into_sql(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            store.initialize()
            statements: list[str] = []
            store.connect().set_trace_callback(statements.append)
            self.assertLessEqual(len(store.list_conversations(2)), 2)
            self.assertLessEqual(len(store.list_contacts(2)), 2)
            self.assertLessEqual(len(store.list_moments(2)), 2)
            self.assertLessEqual(len(store.list_favorites(2)), 2)
            files = list_conversation_files(store, limit=2)
            store.connect().set_trace_callback(None)
            self.assertLessEqual(len(files['files']), 2)
            list_selects = [
                sql for sql in statements
                if any(marker in sql for marker in (
                    'FROM conversations ORDER BY',
                    'FROM entities WHERE',
                    'FROM moment_items ORDER BY',
                    'FROM favorites ORDER BY',
                    'FROM media_assets ma',
                    'SELECT * FROM messages ',
                ))
            ]
            self.assertTrue(list_selects)
            self.assertTrue(all('LIMIT' in sql.upper() for sql in list_selects), list_selects)

    def test_agent_boundary_propagates_the_same_typed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            index_fixture_vault(vault, reset=True)
            calls = [
                lambda: agent_tools.search(vault, 'fixture', limit=True),
                lambda: agent_tools.search(vault, 'fixture', limit='3'),
                lambda: agent_tools.search(vault, 'fixture', limit=None),
                lambda: agent_tools.list_contacts(vault, limit=501),
                lambda: agent_tools.fetch_context(vault, 'trove://fixture', before=201),
                lambda: agent_tools.customer_profile(vault, 'fixture', limit=51),
                lambda: agent_tools.trace_timeline(vault, limit=201),
                lambda: agent_tools.list_approvals(vault, limit=201),
            ]
            for call in calls:
                with self.subTest(call=call), self.assertRaises(BoundedInputError) as raised:
                    call()
                self.assertEqual(raised.exception.code, 'invalid_limit')


if __name__ == '__main__':
    unittest.main()
