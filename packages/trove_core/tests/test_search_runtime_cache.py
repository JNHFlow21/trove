from __future__ import annotations

from dataclasses import replace
import tempfile
import threading
import time
import unittest
from pathlib import Path
import sqlite3
from unittest.mock import patch

from trove_core.runtime import SearchRuntimeCache
from trove_core.search.query import SearchRequest
from trove_core.store.migrations import SchemaMigrationRequired
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import vault_generation_publish
from trove_core.wechat.fixture_factory import FixtureData, generate_fixture
from trove_core.wechat.fixture_guard import FixtureVaultGuardError
from trove_core.wechat.indexer import index_fixture_data, index_fixture_vault


class SearchRuntimeCacheTests(unittest.TestCase):
    @staticmethod
    def _generation_fixture(token: str, local_id: int) -> FixtureData:
        data = generate_fixture()
        message = replace(
            data.messages[0],
            content=token,
            shard_id="generation",
            local_id=local_id,
        )
        return FixtureData(data.accounts, data.conversations, [*data.messages, message])

    def test_cache_reuses_engine_until_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)

            first = cache.get()
            second = cache.get()
            self.assertIs(first, second)
            self.assertTrue(first.search(SearchRequest('价格太高', limit=2)).results)
            before = cache.generation

            status = cache.invalidate('fixture_rebuild')
            self.assertEqual(status['generation'], before + 1)
            third = cache.get()
            self.assertIsNot(first, third)
            self.assertTrue(third.search(SearchRequest('价格太高', limit=2)).results)

    def test_result_correctness_does_not_depend_on_cache_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cache = SearchRuntimeCache(
                VaultConfig.resolve(directory, env={}),
                provider_factory=lambda: None,
                result_cache_max_entries=1,
                result_cache_max_bytes=1,
            )
            try:
                request = SearchRequest('价格太高', limit=2, semantic='off')
                first = cache.search(request)
                second = cache.search(request)
                self.assertEqual(
                    [item.citation for item in first.results],
                    [item.citation for item in second.results],
                )
                status = cache.status()
                self.assertEqual(status['result_cache_entries'], 0)
                self.assertGreaterEqual(status['result_cache_evictions'], 2)
                self.assertLessEqual(status['result_cache_bytes'], status['result_cache_max_bytes'])
            finally:
                cache.close()

    def test_runtime_does_not_eagerly_warm_vector_path_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)

            with patch.dict('os.environ', {'TROVE_SEARCH_EAGER_WARMUP': '0'}), patch(
                'trove_core.runtime.warm_search_engine',
                side_effect=AssertionError('lexical runtime construction must stay lazy'),
            ):
                engine = cache.get()

            self.assertEqual(
                engine.vector_status['warmup']['reason_code'],
                'lazy_until_semantic_query',
            )
            self.assertTrue(engine.vector_status['warmup']['ok'])
            cache.close()

    def test_cache_requires_explicit_migration_for_legacy_schema(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            db = vault / 'index' / 'trove.sqlite'
            with sqlite3.connect(db) as conn:
                for name in ['message_fts_ai', 'message_fts_ad', 'message_fts_au', 'chunk_fts_ai', 'chunk_fts_ad', 'chunk_fts_au']:
                    conn.execute(f'DROP TRIGGER IF EXISTS {name}')
                conn.execute('DROP TABLE IF EXISTS message_fts')
                conn.execute('DROP TABLE IF EXISTS chunk_fts')
                conn.execute("DELETE FROM schema_meta WHERE key='fts_tokenizer'")
                conn.commit()

            cfg = VaultConfig.resolve(str(vault), env={})
            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)
            before = db.read_bytes()
            with self.assertRaises(SchemaMigrationRequired) as error:
                cache.get()
            self.assertEqual(error.exception.code, 'schema_migration_required')
            self.assertEqual(db.read_bytes(), before)
            with sqlite3.connect(db) as conn:
                row = conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()
            self.assertIsNone(row)

    def test_external_generation_and_explicit_invalidation_each_advance_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cache = SearchRuntimeCache(VaultConfig.resolve(directory, env={}), provider_factory=lambda: None)
            request = SearchRequest("价格太高", limit=2, semantic="off")
            self.assertTrue(cache.search(request).results)
            initial = cache.generation

            index_fixture_data(root, self._generation_fixture("external generation sentinel alpha", 9001), reset=True)
            alpha = cache.search(SearchRequest("sentinel alpha", limit=2, semantic="off"))
            self.assertTrue(alpha.results)
            self.assertEqual(cache.generation, initial + 1)
            cache.search(SearchRequest("sentinel alpha", limit=2, semantic="off"))
            self.assertEqual(cache.generation, initial + 1)

            index_fixture_data(root, self._generation_fixture("explicit generation sentinel beta", 9002), reset=True)
            status = cache.invalidate("fixture_rebuild")
            self.assertEqual(status["generation"], initial + 2)
            self.assertTrue(cache.search(SearchRequest("sentinel beta", limit=2, semantic="off")).results)
            self.assertEqual(cache.generation, initial + 2)
            cache.close()

    def test_in_place_sqlite_generation_invalidates_once_with_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(directory, env={})
            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)
            self.assertTrue(cache.search(SearchRequest("价格太高", limit=2, semantic="off")).results)
            before = cache.generation

            with vault_generation_publish(cfg):
                with sqlite3.connect(cfg.paths.sqlite_path) as connection:
                    connection.execute(
                        "UPDATE messages SET content=? WHERE citation=?",
                        ("wal generation sentinel delta", "trove://wechat/acct-work/conv-example_edu-private/message_0/1"),
                    )
                    connection.commit()
                    self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

            self.assertTrue(cache.search(SearchRequest("sentinel delta", limit=2, semantic="off")).results)
            self.assertEqual(cache.generation, before + 1)
            cache.search(SearchRequest("sentinel delta", limit=2, semantic="off"))
            self.assertEqual(cache.generation, before + 1)
            cache.close()

    def test_active_search_finishes_old_generation_before_fixture_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(directory, env={})
            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)
            engine = cache.get()
            original_search = engine.search
            reader_entered = threading.Event()
            allow_reader_to_finish = threading.Event()
            reader_result = []
            errors: list[BaseException] = []

            def slow_search(request):
                reader_entered.set()
                if not allow_reader_to_finish.wait(3.0):
                    raise TimeoutError("reader release was not signalled")
                return original_search(request)

            engine.search = slow_search  # type: ignore[method-assign]

            def read_old() -> None:
                try:
                    reader_result.append(cache.search(SearchRequest("价格太高", limit=2, semantic="off")))
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            old_inode = cfg.paths.sqlite_path.stat().st_ino
            reader = threading.Thread(target=read_old, daemon=True)
            reader.start()
            self.assertTrue(reader_entered.wait(2.0))
            with self.assertRaises(FixtureVaultGuardError) as blocked:
                index_fixture_data(
                    root,
                    self._generation_fixture("new complete generation sentinel gamma", 9003),
                    reset=True,
                )
            self.assertEqual(blocked.exception.reason_code, "fixture_claim_locked")
            self.assertEqual(cfg.paths.sqlite_path.stat().st_ino, old_inode)
            allow_reader_to_finish.set()
            reader.join(3.0)

            self.assertFalse(errors)
            self.assertTrue(reader_result and reader_result[0].results)
            index_fixture_data(
                root,
                self._generation_fixture("new complete generation sentinel gamma", 9003),
                reset=True,
            )
            self.assertNotEqual(cfg.paths.sqlite_path.stat().st_ino, old_inode)
            engine.search = original_search  # type: ignore[method-assign]
            current = cache.search(SearchRequest("sentinel gamma", limit=2, semantic="off"))
            self.assertTrue(current.results)
            self.assertEqual(cache.generation, 1)
            cache.close()

    def test_concurrent_identical_searches_share_one_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cache = SearchRuntimeCache(
                VaultConfig.resolve(directory, env={}),
                provider_factory=lambda: None,
            )
            engine = cache.get()
            original_search = engine.search
            entered = threading.Event()
            release = threading.Event()
            calls = 0
            call_lock = threading.Lock()
            responses = []
            metrics = []
            errors: list[BaseException] = []

            def slow_search(request):
                nonlocal calls
                with call_lock:
                    calls += 1
                entered.set()
                if not release.wait(3.0):
                    raise TimeoutError('singleflight leader was not released')
                return original_search(request)

            engine.search = slow_search  # type: ignore[method-assign]
            request = SearchRequest('价格太高', limit=2, semantic='off')

            def run() -> None:
                try:
                    response, runtime_metrics = cache.search_with_metrics(request)
                    responses.append(response)
                    metrics.append(runtime_metrics)
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            first = threading.Thread(target=run, daemon=True)
            second = threading.Thread(target=run, daemon=True)
            first.start()
            self.assertTrue(entered.wait(2.0))
            second.start()
            deadline = time.monotonic() + 2.0
            while cache.status()['singleflight_followers'] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            release.set()
            first.join(3.0)
            second.join(3.0)

            self.assertFalse(errors)
            self.assertEqual(calls, 1)
            self.assertEqual(len(responses), 2)
            self.assertEqual(sum(bool(item['singleflight_shared']) for item in metrics), 1)
            self.assertEqual(cache.status()['result_inflight'], 0)
            cache.close()

    def test_generation_memo_retains_only_accepted_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cache = SearchRuntimeCache(
                VaultConfig.resolve(directory, env={}),
                provider_factory=lambda: None,
            )
            calls = 0

            def load():
                nonlocal calls
                calls += 1
                return {'state': 'available' if calls > 1 else 'degraded'}

            first, first_metrics = cache.memoize_generation(
                'fixture', ('same',), load, cache_if=lambda value: value['state'] == 'available'
            )
            second, second_metrics = cache.memoize_generation(
                'fixture', ('same',), load, cache_if=lambda value: value['state'] == 'available'
            )
            third, third_metrics = cache.memoize_generation(
                'fixture', ('same',), load, cache_if=lambda value: value['state'] == 'available'
            )

            self.assertEqual(first['state'], 'degraded')
            self.assertEqual(second['state'], 'available')
            self.assertEqual(third['state'], 'available')
            self.assertFalse(first_metrics['cache_hit'])
            self.assertFalse(second_metrics['cache_hit'])
            self.assertTrue(third_metrics['cache_hit'])
            self.assertEqual(calls, 2)
            cache.close()


if __name__ == '__main__':
    unittest.main()
