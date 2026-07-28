from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trove_core.embedding.base import HybridEmbedding
from trove_core.runtime import VectorIndexSourceChanged, _vector_source_snapshot
from trove_core.search.episodes import (
    BoundedEvidenceSelector,
    EpisodeCloudBudgetExceeded,
    EpisodeZVecStore,
    _episode_rows,
    _incremental_episode_rows,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.process_config import process_config_from_payload, write_process_config


class _Response:
    status_code = 200

    def json(self):
        return {
            'choices': [{'message': {'content': json.dumps({'selected_indices': [1, 0]})}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 2},
        }


class _EpisodeCollection:
    def __init__(self):
        self.docs_by_id = {}

    @property
    def stats(self):
        return SimpleNamespace(doc_count=len(self.docs_by_id))

    def upsert(self, docs):
        for doc in docs:
            self.docs_by_id[doc.id] = doc

    def flush(self):
        pass


class _EpisodeZvec:
    class DataType:
        STRING = 'string'
        VECTOR_FP32 = 'vector'
        SPARSE_VECTOR_FP32 = 'sparse'

    class MetricType:
        IP = 'ip'

    class FieldSchema:
        def __init__(self, *_args, **_kwargs):
            pass

    class VectorSchema:
        def __init__(self, *_args, **_kwargs):
            pass

    class CollectionSchema:
        def __init__(self, *_args, **_kwargs):
            pass

    class HnswIndexParam:
        def __init__(self, *_args, **_kwargs):
            pass

    class Doc:
        def __init__(self, id, fields, vectors):
            self.id = id
            self.fields = fields
            self.vectors = vectors

    collections = {}

    @classmethod
    def create_and_open(cls, path, _schema):
        Path(path).mkdir(parents=True, exist_ok=True)
        collection = _EpisodeCollection()
        cls.collections[str(path)] = collection
        return collection

    @classmethod
    def open(cls, path):
        return cls.collections[str(path)]


class _EpisodeProvider:
    name = 'fixture-cloud:hybrid'
    provider_name = 'fixture-cloud'
    model = 'fixture-hybrid'
    request_format = 'dashscope-native'
    supports_sparse = True
    dimensions = 2

    def __init__(self, *, fail_call=None, after_embed=None):
        self.fail_call = fail_call
        self.after_embed = after_embed
        self.provider_calls = 0
        self.input_tokens = 0
        self.document_count = 0

    def embed_hybrid_many(self, texts, *, text_type='document', instruct=None):
        _ = (text_type, instruct)
        self.provider_calls += 1
        if self.fail_call == self.provider_calls:
            raise RuntimeError('synthetic_provider_pause')
        self.document_count += len(texts)
        self.input_tokens += sum(max(1, len(text) // 2) for text in texts)
        if self.after_embed is not None:
            callback, self.after_embed = self.after_embed, None
            callback()
        return [HybridEmbedding([1.0, 0.0], {1: 1.0}) for _ in texts]


def _synthetic_episodes(count):
    for index in range(count):
        yield {
            'episode_id': f'episode-{index:04d}',
            'account_id': 'fixture-account',
            'conversation_id': f'fixture-conversation-{index // 9}',
            'timestamp': f'2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z',
            'citations': (f'fixture-citation-{index}',),
            'text': f'synthetic episode {index} ' + ('x' * 80),
        }


class CloudEpisodeRetrievalTests(unittest.TestCase):
    def setUp(self):
        _EpisodeZvec.collections = {}

    def _episode_store(self, cfg):
        store = EpisodeZVecStore(cfg.paths.vector_dir / 'zvec' / 'episodes-cloud')
        store._zvec = _EpisodeZvec
        store._error = None
        return store

    def test_episode_windows_are_bounded_and_conversation_local(self):
        rows = [
            {
                'citation': f'c-{index}',
                'account_id': 'a',
                'conversation_id': 'conversation',
                'timestamp': f'2026-01-01T00:00:0{index}Z',
                'sender_name': 'sender',
                'content': f'message {index} ' + ('x' * 900),
            }
            for index in range(8)
        ]
        episodes = list(_episode_rows(rows))
        self.assertEqual(len(episodes), 3)
        self.assertTrue(all(len(item['citations']) <= 7 for item in episodes))
        self.assertTrue(all(item['conversation_id'] == 'conversation' for item in episodes))
        self.assertEqual(len({item['episode_id'] for item in episodes}), 3)
        self.assertTrue(all(len(item['text']) < 2048 for item in episodes))

    def test_append_only_episode_delta_embeds_only_overlapping_tail_windows(self):
        rows = [
            {
                'citation': f'c-{index}',
                'account_id': 'a',
                'conversation_id': 'conversation',
                'timestamp': f'2026-01-01T00:{index // 60:02d}:{index % 60:02d}Z',
                'sender_name': 'sender',
                'content': f'message {index}',
            }
            for index in range(300)
        ]
        mode, episodes = _incremental_episode_rows(
            rows,
            {f'c-{index}' for index in range(290, 300)},
            has_tombstone=False,
        )
        self.assertEqual(mode, 'tail')
        self.assertLessEqual(len(episodes), 5)
        self.assertLess(len(episodes), len(list(_episode_rows(rows))))

    def test_mid_conversation_or_deleted_episode_delta_keeps_full_fallback(self):
        rows = [
            {
                'citation': f'c-{index}',
                'account_id': 'a',
                'conversation_id': 'conversation',
                'timestamp': f'2026-01-01T00:00:{index:02d}Z',
                'sender_name': 'sender',
                'content': f'message {index}',
            }
            for index in range(30)
        ]
        mid_mode, mid_episodes = _incremental_episode_rows(rows, {'c-10'}, has_tombstone=False)
        deleted_mode, deleted_episodes = _incremental_episode_rows(rows, set(), has_tombstone=True)
        self.assertEqual(mid_mode, 'full')
        self.assertEqual(deleted_mode, 'full')
        self.assertEqual(len(mid_episodes), len(list(_episode_rows(rows))))
        self.assertEqual(len(deleted_episodes), len(list(_episode_rows(rows))))

    def test_selector_accepts_only_bounded_existing_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            write_process_config(vault, process_config_from_payload({
                'config_id': 'pcfg-selector-test',
                'cloud_retrieval': 'enabled',
            }))
            rows = [
                {'citation': f'c-{index}', 'content': f'document {index}'}
                for index in range(3)
            ]
            selector = BoundedEvidenceSelector(vault)
            with (
                patch('trove_core.providers.factory.EnvironmentAgentSwitchSecretResolver.resolve', return_value='test-key'),
                patch('trove_core.search.episodes._httpx_post', return_value=_Response()) as post,
            ):
                ordered, status = selector.select(
                    'query',
                    rows,
                    candidate_metadata=[
                        {'episode_rank': 0, 'episode_position': index}
                        for index in range(len(rows))
                    ],
                )
        self.assertEqual([row['citation'] for row in ordered[:2]], ['c-1', 'c-0'])
        self.assertEqual(status['state'], 'available')
        self.assertEqual(status['selected_count'], 2)
        self.assertFalse(status['raw_content_included'])
        request = post.call_args.kwargs['json']
        self.assertIn('all facts, stages, entities, numbers, and time relations', request['messages'][0]['content'])
        user = json.loads(request['messages'][1]['content'])
        self.assertEqual(user['candidates'][1]['episode_rank'], 0)
        self.assertEqual(user['candidates'][1]['episode_position'], 1)

    def test_episode_build_resumes_completed_batches_and_preserves_spend_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            snapshot = _vector_source_snapshot(cfg)
            episode_store = self._episode_store(cfg)
            with patch('trove_core.search.episodes.iter_episode_documents', side_effect=lambda _path: _synthetic_episodes(250)):
                first = _EpisodeProvider(fail_call=2)
                with self.assertRaisesRegex(RuntimeError, 'synthetic_provider_pause'):
                    episode_store.rebuild(first, cfg=cfg, source_snapshot=snapshot)
                progress = json.loads(episode_store.staging_metadata_path.read_text())
                self.assertEqual(progress['indexed_count'], 120)
                self.assertEqual(progress['state'], 'paused_provider_error')
                self.assertGreater(progress['budget']['consumed_input_tokens'], first.input_tokens)

                # Simulate a process death after ZVec committed a batch but
                # before the progress JSON caught up.  Collection count is the
                # recovery authority for the contiguous staged prefix.
                progress['indexed_count'] = 0
                episode_store.staging_metadata_path.write_text(json.dumps(progress))
                second = _EpisodeProvider()
                result = episode_store.rebuild(second, cfg=cfg, source_snapshot=snapshot)

            self.assertTrue(result['ok'])
            self.assertEqual(result['indexed_count'], 250)
            self.assertEqual(result['expected_document_count'], 250)
            self.assertEqual(second.document_count, 130)
            self.assertFalse(episode_store.staging_path.exists())

    def test_episode_budget_pause_is_resumable_and_never_publishes_partial_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            snapshot = _vector_source_snapshot(cfg)
            episode_store = self._episode_store(cfg)
            provider = _EpisodeProvider()
            with (
                patch('trove_core.search.episodes.iter_episode_documents', side_effect=lambda _path: _synthetic_episodes(121)),
                self.assertRaises(EpisodeCloudBudgetExceeded),
            ):
                episode_store.rebuild(provider, cfg=cfg, source_snapshot=snapshot, cost_cap_rmb=0.000001)

            progress = json.loads(episode_store.staging_metadata_path.read_text())
            self.assertEqual(progress['state'], 'paused_budget')
            self.assertEqual(progress['indexed_count'], 0)
            self.assertEqual(provider.provider_calls, 0)
            self.assertTrue(episode_store.staging_path.exists())
            self.assertFalse(episode_store.path.exists())

    def test_episode_source_cas_rejects_publish_but_keeps_complete_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            snapshot = _vector_source_snapshot(cfg)
            episode_store = self._episode_store(cfg)

            def mutate_vector_source():
                store = SQLiteStore(cfg.paths.sqlite_path)
                with store.connect() as conn:
                    conn.execute(
                        "UPDATE messages SET content=content || ' synthetic-cas-change' WHERE id=(SELECT MIN(id) FROM messages)"
                    )
                    conn.commit()

            provider = _EpisodeProvider(after_embed=mutate_vector_source)
            with (
                patch('trove_core.search.episodes.iter_episode_documents', side_effect=lambda _path: _synthetic_episodes(10)),
                self.assertRaises(VectorIndexSourceChanged),
            ):
                episode_store.rebuild(provider, cfg=cfg, source_snapshot=snapshot)

            progress = json.loads(episode_store.staging_metadata_path.read_text())
            self.assertTrue(progress['complete'])
            self.assertEqual(progress['indexed_count'], 10)
            self.assertTrue(episode_store.staging_path.exists())
            self.assertFalse(episode_store.path.exists())


if __name__ == '__main__':
    unittest.main()
