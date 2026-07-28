from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from trove_core.search.hyper_search import HyperSearch
from trove_core.runtime import SearchRuntimeCache
from trove_core.search.query import SearchRequest, SearchResponse
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore, SQLiteVectorUnavailable, SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT
from trove_core.vector.zvec_store import ZVecStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.models import Account, Conversation, Message


class P2ReadPathTests(unittest.TestCase):
    def _store(self) -> SQLiteStore:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        store = SQLiteStore(Path(d.name) / 'trove.sqlite')
        store.initialize()
        return store

    def _seed_mixed_evidence(self, store: SQLiteStore) -> list[str]:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.upsert_accounts([Account('acct-p2', 'work', 'Work')])
        store.upsert_conversations([Conversation('conv-p2', 'acct-p2', 'P2 fixture', 'private')])
        messages = [
            Message(
                account_id='acct-p2',
                account_label='work',
                conversation_id='conv-p2',
                conversation_title='P2 fixture',
                conversation_type='private',
                sender_id='u1',
                sender_name='Tester',
                timestamp=base + timedelta(seconds=i),
                content=f'message evidence {i}',
                shard_id='message_0',
                local_id=i,
            )
            for i in range(7)
        ]
        store.upsert_messages(messages)
        citations = [m.citation for m in messages]
        now = base.isoformat().replace('+00:00', 'Z')
        with store.connect() as conn:
            for i in range(7):
                citation = f'trove://chunk/{i}'
                citations.append(citation)
                conn.execute(
                    """INSERT INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (citation, citation, messages[i % len(messages)].citation, 'acct-p2', 'work', 'message', 'conv-p2', 'Chunk title', 'Tester', now, f'chunk evidence {i}', i, '{}', 'active', now),
                )
            for i in range(6):
                citation = f'trove://moment/item/{i}'
                citations.append(citation)
                conn.execute(
                    "INSERT INTO moment_items(moment_id,account_id,author_id,citation,timestamp,text) VALUES(?,?,?,?,?,?)",
                    (f'moment-{i}', 'acct-p2', 'author', citation, now, f'moment evidence {i}'),
                )
            for i in range(6):
                citation = f'trove://moment/interaction/{i}'
                citations.append(citation)
                conn.execute(
                    "INSERT INTO moment_interactions(interaction_id,moment_id,account_id,citation,interaction_type,actor_id,text,timestamp) VALUES(?,?,?,?,?,?,?,?)",
                    (f'interaction-{i}', f'moment-{i}', 'acct-p2', citation, 'comment', 'actor', f'interaction evidence {i}', now),
                )
            for i in range(6):
                citation = f'trove://favorite/{i}'
                citations.append(citation)
                conn.execute(
                    "INSERT INTO favorites(favorite_id,account_id,citation,timestamp,title,text) VALUES(?,?,?,?,?,?)",
                    (f'favorite-{i}', 'acct-p2', citation, now, 'Favorite title', f'favorite evidence {i}'),
                )
            for i in range(6):
                citation = f'trove://transcript/{i}'
                citations.append(citation)
                content_hash = f'{i + 1:064x}'
                conn.execute(
                    """INSERT INTO media_assets(
                           asset_id,account_id,source_type,source_id,modality,media_type,
                           citation,content_hash,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (f'asset-t-{i}', 'acct-p2', 'message', f'voice-{i}', 'voice', 'voice',
                     citation, content_hash, now, now),
                )
                conn.execute(
                    """INSERT INTO provider_jobs(
                           job_id,asset_id,provider,model,job_type,status,request_hash,
                           citation,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (f'job-t-{i}', f'asset-t-{i}', 'volcengine-asr-flash',
                     'bigmodel:volc.bigasr.auc_turbo', 'asr', 'completed', content_hash,
                     citation, now, now),
                )
                conn.execute(
                    "INSERT INTO transcripts(transcript_id,asset_id,job_id,citation,text,created_at) VALUES(?,?,?,?,?,?)",
                    (f'transcript-{i}', f'asset-t-{i}', f'job-t-{i}', citation,
                     f'transcript evidence {i}', now),
                )
            for i in range(6):
                citation = f'trove://image/{i}'
                citations.append(citation)
                conn.execute(
                    "INSERT INTO image_observations(observation_id,asset_id,citation,caption,created_at) VALUES(?,?,?,?,?)",
                    (f'image-{i}', f'asset-i-{i}', citation, f'image evidence {i}', now),
                )
            for i in range(6):
                citation = f'trove://observation/{i}'
                citations.append(citation)
                conn.execute(
                    "INSERT INTO observations(observation_id,entity_id,observation_type,value_json,status,confidence,citation,source_type,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (f'observation-{i}', 'entity-p2', 'note', json.dumps({'text': f'observation evidence {i}'}), 'active', 1.0, citation, 'contact', now, now),
                )
            conn.commit()
        self.assertEqual(len(citations), 50)
        return citations

    def test_evidence_by_citations_hydrates_50_mixed_rows_in_at_most_5_selects(self):
        store = self._store()
        citations = self._seed_mixed_evidence(store)
        conn = store.connect()
        statements: list[str] = []
        conn.set_trace_callback(lambda sql: statements.append(sql) if sql.strip().upper().startswith('SELECT') else None)
        try:
            rows = store.evidence_by_citations(citations)
        finally:
            conn.set_trace_callback(None)
        self.assertEqual(set(rows), set(citations))
        self.assertLessEqual(len(statements), 5, statements)
        self.assertEqual(rows['trove://chunk/0']['parent_citation'], citations[0])
        self.assertEqual(rows['trove://favorite/0']['source_type'], 'favorite')
        self.assertEqual(rows['trove://observation/0']['source_type'], 'contact')

    def test_context_window_uses_bounded_index_queries_for_long_conversation(self):
        store = self._store()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.upsert_accounts([Account('acct-context', 'work', 'Work')])
        store.upsert_conversations([Conversation('conv-long', 'acct-context', 'Long fixture', 'private')])
        messages = [
            Message(
                account_id='acct-context',
                account_label='work',
                conversation_id='conv-long',
                conversation_title='Long fixture',
                conversation_type='private',
                sender_id='u1',
                sender_name='Tester',
                timestamp=base + timedelta(seconds=i),
                content=f'context row {i}',
                shard_id='message_0',
                local_id=i,
            )
            for i in range(10001)
        ]
        store.upsert_messages(messages)
        anchor = messages[5000].citation

        def fail_full_load(*_args, **_kwargs):
            raise AssertionError('context_window must not load the full conversation')

        store.messages_for_conversation = fail_full_load  # type: ignore[method-assign]
        rows = store.context_window(anchor, before=3, after=3)
        self.assertEqual([row['local_id'] for row in rows], list(range(4997, 5004)))

        with store.connect() as conn:
            anchor_row = conn.execute('SELECT timestamp,shard_id,local_id FROM messages WHERE citation=?', (anchor,)).fetchone()
            plan_rows = conn.execute(
                """EXPLAIN QUERY PLAN SELECT * FROM messages
                   WHERE account_id=? AND conversation_id=?
                     AND (timestamp, shard_id, local_id) < (?, ?, ?)
                   ORDER BY timestamp DESC, shard_id DESC, local_id DESC
                   LIMIT ?""",
                ('acct-context', 'conv-long', anchor_row['timestamp'], anchor_row['shard_id'], anchor_row['local_id'], 3),
            ).fetchall()
        plan = ' | '.join(str(row['detail']) for row in plan_rows)
        self.assertIn('idx_messages_context_window', plan)
        self.assertNotIn('SCAN messages', plan)

    def test_hyper_search_skips_fts_expansion_when_exact_chunk_satisfy_auto_semantics(self):
        class FailingFtsStore(SQLiteStore):
            def fts_search_filtered(self, *_args, **_kwargs):
                raise AssertionError('FTS expansion route should be skipped')

        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = FailingFtsStore(Path(d) / 'index' / 'trove.sqlite')
            resp = HyperSearch(
                store,
                vector_store=object(),
                embedding_provider=object(),
                vector_status={'state': 'available', 'selected_backend': 'zvec'},
            ).search(SearchRequest('价格太高', limit=1, semantic='auto'))
        self.assertTrue(resp.results)
        plan = resp.retrieval_status['retrieval_plan']
        self.assertFalse(plan['fts_route_executed'])
        self.assertEqual(plan['fts_route_skipped_reason'], 'exact_chunk_limit_satisfied')
        self.assertNotIn('fts', resp.retrieval_status['ranking']['candidate_routes'])
        self.assertEqual(resp.retrieval_status['vector']['reason_code'], 'semantic_auto_satisfied')

    def test_zvec_reuses_open_collection_until_reset(self):
        class FakeDoc:
            def __init__(self, citation: str):
                self.id = citation
                self.fields = {'citation': citation}

        class FakeCollection:
            def query(self, *_args, **_kwargs):
                return [FakeDoc('trove://wechat/acct-p2/conv-p2/message_0/1')]

            def flush(self):
                pass

        class FakeZvec:
            open_calls = 0

            class HnswQueryParam:
                def __init__(self, *_args, **_kwargs):
                    pass

            class Query:
                def __init__(self, *_args, **_kwargs):
                    pass

            @classmethod
            def open(cls, _path):
                cls.open_calls += 1
                return FakeCollection()

        store = self._store()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.upsert_accounts([Account('acct-p2', 'work', 'Work')])
        store.upsert_conversations([Conversation('conv-p2', 'acct-p2', 'P2 fixture', 'private')])
        store.upsert_messages([
            Message(
                account_id='acct-p2',
                account_label='work',
                conversation_id='conv-p2',
                conversation_title='P2 fixture',
                conversation_type='private',
                sender_id='u1',
                sender_name='Tester',
                timestamp=base,
                content='cached zvec evidence',
                shard_id='message_0',
                local_id=1,
            )
        ])
        collection_path = store.path.parent / 'zvec-cache-test'
        collection_path.mkdir(parents=True)
        zvec = ZVecStore(collection_path, store=store)
        zvec._zvec = FakeZvec  # type: ignore[assignment]
        zvec._error = None

        # Collection-handle reuse is independent of score calibration. Product
        # search now fails closed until a generation-bound floor is installed.
        self.assertIs(zvec._open_existing(), zvec._open_existing())
        self.assertEqual(FakeZvec.open_calls, 1)
        zvec.reset_collection()
        self.assertIsNone(zvec._collection)

    def test_sqlite_vector_search_is_unavailable_above_diagnostic_limit(self):
        store = self._store()
        vector = SQLiteVectorStore(store)
        vector.initialize()
        with store.connect() as conn:
            conn.executemany(
                'INSERT INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)',
                ((f'trove://vector/{i}', 'fake', 1, '[1.0]', str(i)) for i in range(SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT + 1)),
            )
            conn.commit()

        class Provider:
            def embed(self, _text):
                raise AssertionError('search must fail before embedding or loading all vectors')

        with self.assertRaises(SQLiteVectorUnavailable) as ctx:
            vector.search('diagnostic cap', provider=Provider())
        self.assertEqual(ctx.exception.reason_code, 'sqlite_vector_diagnostic_limit_exceeded')
        self.assertEqual(ctx.exception.vector_state, 'unavailable_fallback')

    def test_hyper_search_preserves_vector_unavailable_state(self):
        class UnavailableVector:
            def search(self, *_args, **_kwargs):
                raise SQLiteVectorUnavailable('diagnostic fallback unavailable')

        store = self._store()
        resp = HyperSearch(
            store,
            vector_store=UnavailableVector(),
            embedding_provider=object(),
            vector_status={'state': 'available', 'selected_backend': 'sqlite'},
        ).search(SearchRequest('semantic probe', limit=1, semantic='on'))
        vector_status = resp.retrieval_status['vector']
        self.assertEqual(vector_status['state'], 'unavailable_fallback')
        self.assertEqual(vector_status['reason_code'], 'sqlite_vector_diagnostic_limit_exceeded')

    def test_search_runtime_result_cache_hits_and_invalidates_by_generation(self):
        class CountingEngine:
            def __init__(self):
                self.calls = 0

            def search(self, request):
                self.calls += 1
                return SearchResponse(
                    query=request.query,
                    results=[],
                    total=0,
                    retrieval_status={'call': self.calls},
                    elapsed_ms=float(self.calls),
                )

        class FakeCache(SearchRuntimeCache):
            def __init__(self, cfg):
                super().__init__(cfg, provider_factory=lambda: None)
                self.engine = CountingEngine()

            def get(self):
                return self.engine

        with tempfile.TemporaryDirectory() as d:
            cache = FakeCache(VaultConfig.resolve(d, env={}))
            req = SearchRequest('cache me', limit=3)
            first = cache.search(req)
            second = cache.search(req)
            self.assertIs(first, second)
            self.assertEqual(cache.engine.calls, 1)
            self.assertEqual(cache.status()['result_cache_entries'], 1)
            cache.invalidate('import_mutation')
            third = cache.search(req)
            self.assertIsNot(first, third)
            self.assertEqual(cache.engine.calls, 2)
            self.assertEqual(cache.status()['generation'], 1)

    def test_search_runtime_result_cache_bypasses_after_external_sqlite_commit(self):
        class CountingEngine:
            def __init__(self):
                self.calls = 0

            def search(self, request):
                self.calls += 1
                return SearchResponse(
                    query=request.query,
                    results=[],
                    total=0,
                    retrieval_status={'call': self.calls},
                    elapsed_ms=float(self.calls),
                )

        class FakeCache(SearchRuntimeCache):
            def __init__(self, cfg):
                super().__init__(cfg, provider_factory=lambda: None)
                self.engine = CountingEngine()

            def get(self):
                return self.engine

        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(d, env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()
            self.addCleanup(store.close)
            cache = FakeCache(cfg)
            req = SearchRequest('cache me', limit=3)

            first = cache.search(req)
            second = cache.search(req)
            self.assertIs(first, second)
            self.assertEqual(cache.engine.calls, 1)

            with sqlite3.connect(cfg.paths.sqlite_path) as conn:
                conn.execute(
                    """INSERT INTO messages(citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
                       sender_id,sender_name,timestamp,content,shard_id,local_id,sent_by_me,source_type,direction)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'trove://wechat/acct-cache/conv-cache/message_0/1',
                        'acct-cache',
                        'work',
                        'conv-cache',
                        'Cache fixture',
                        'private',
                        'u1',
                        'External Writer',
                        '2026-01-01T00:00:00Z',
                        'external cache invalidation message',
                        'message_0',
                        1,
                        0,
                        'message',
                        'incoming',
                    ),
                )
                conn.commit()

            third = cache.search(req)
            self.assertIsNot(first, third)
            self.assertEqual(cache.engine.calls, 2)
            fourth = cache.search(req)
            self.assertIs(third, fourth)
            self.assertEqual(cache.engine.calls, 2)


if __name__ == '__main__':
    unittest.main()
