from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.vector.registry import VectorBackendRegistry
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore
from trove_core.vector.zvec_store import ZVecStore
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault


class VectorBackendRegistryTests(unittest.TestCase):
    def test_missing_collection_status_skips_unused_full_corpus_counts(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            registry = VectorBackendRegistry(
                store=store,
                zvec_path=cfg.paths.vector_dir / 'zvec' / 'messages',
            )
            with patch.object(store, 'counts', side_effect=AssertionError('multi-table counts are not needed')), \
                 patch.object(registry._zvec, '_expected_document_count', side_effect=AssertionError('no collection to compare')):
                status = registry.status('zvec').to_dict()

            self.assertEqual(status['reason_code'], 'local_embedding_model_missing')
            self.assertGreater(status['message_count'], 0)

    def test_unavailable_fallback_without_model_or_zvec(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            status = VectorBackendRegistry(store=SQLiteStore(cfg.paths.sqlite_path), zvec_path=cfg.paths.vector_dir / 'zvec' / 'messages').status('zvec').to_dict()
            self.assertEqual(status['state'], 'unavailable_fallback')
            self.assertEqual(status['selected_backend'], 'none')
            self.assertTrue(status['baseline_search_available'])

    def test_available_with_sqlite_local_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            provider = FakeEmbeddingProvider(dimensions=8)
            SQLiteVectorStore(store).index_all_messages(provider, max_messages=3)
            status = VectorBackendRegistry(store=store, zvec_path=cfg.paths.vector_dir / 'zvec' / 'messages', provider=provider).status('sqlite').to_dict()
            self.assertEqual(status['state'], 'available')
            self.assertEqual(status['selected_backend'], 'sqlite')
            self.assertGreaterEqual(status['sqlite']['entries'], 3)

    def test_degraded_zvec_collection_does_not_disable_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            path = cfg.paths.vector_dir / 'zvec' / 'messages'
            path.mkdir(parents=True)
            reg = VectorBackendRegistry(store=SQLiteStore(cfg.paths.sqlite_path), zvec_path=path, provider=FakeEmbeddingProvider())
            class BrokenZvec:
                def status(self, provider=None):
                    return {'backend': 'zvec', 'available': True, 'collection_exists': True, 'health': 'degraded', 'reason_code': 'zvec_collection_degraded', 'unavailable_reason': 'probe failed'}
            reg._zvec = BrokenZvec()
            status = reg.status('zvec').to_dict()
            self.assertEqual(status['state'], 'degraded')
            self.assertEqual(status['reason_code'], 'zvec_collection_degraded')
            self.assertTrue(status['baseline_search_available'])

    def test_zvec_status_marks_provider_mismatch_rebuild_required(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'vectors' / 'zvec' / 'messages'
            path.mkdir(parents=True)
            path.with_name(path.name + '.trove-meta.json').write_text(
                '{"vector_text_version": 3, "indexed_count": 1, "embedding_provider": "local", "embedding_model": "old", "embedding_dimensions": 512}',
                encoding='utf-8',
            )

            class Provider:
                provider_name = 'aliyun'
                model = 'text-embedding-v4'
                dimensions = 1024
                request_format = 'openai'

            status = ZVecStore(path).status(provider=Provider())
            self.assertTrue(status['provider_mismatch'])
            self.assertTrue(status['rebuild_required'])
            self.assertEqual(status['expected_embedding']['embedding_provider'], 'aliyun')

    def test_message_count_is_a_max_rowid_upper_bound(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            registry = VectorBackendRegistry(store=store, zvec_path=cfg.paths.vector_dir / 'zvec' / 'messages')
            with store.connect() as conn:
                max_rowid = int(conn.execute('SELECT MAX(rowid) FROM messages').fetchone()[0])
                exact = int(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0])
                conn.execute('DELETE FROM messages WHERE rowid = (SELECT MIN(rowid) FROM messages)')
                conn.commit()
            # An exact COUNT(*) would now return ``exact - 1``; the status
            # read path deliberately reports the instant MAX(rowid) bound.
            self.assertEqual(registry.message_count(), max_rowid)
            self.assertGreater(registry.message_count(), exact - 1)

    def test_sqlite_entries_count_is_bounded_by_the_diagnostic_limit(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            cfg = VaultConfig.resolve(d, env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            provider = FakeEmbeddingProvider(dimensions=8)
            SQLiteVectorStore(store).index_all_messages(provider, max_messages=3)
            registry = VectorBackendRegistry(
                store=store,
                zvec_path=cfg.paths.vector_dir / 'zvec' / 'messages',
                provider=provider,
            )
            actual = registry.sqlite_entries()
            self.assertGreaterEqual(actual, 3)
            with patch('trove_core.vector.registry.SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT', actual - 1):
                status = registry.status('sqlite').to_dict()
            # The bounded count caps at ``limit + 1`` and still drives the
            # exact availability decision.
            self.assertEqual(status['sqlite']['entries'], actual)
            self.assertEqual(status['sqlite']['reason_code'], 'sqlite_vector_diagnostic_limit_exceeded')
            self.assertEqual(status['state'], 'unavailable_fallback')
            self.assertEqual(status['selected_backend'], 'none')


if __name__ == '__main__':
    unittest.main()
