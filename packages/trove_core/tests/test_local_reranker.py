from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from trove_core.search import local_reranker as local_reranker_module
from trove_core.search.local_reranker import rerank_with_local_model, warm_local_reranker
from trove_core.store.sqlite_store import vector_document_text


def _ranked_rows():
    return [
        ({
            'citation': f'fixture-{index}',
            'content': content,
            'content_kind': 'text',
            'source_type': 'message',
            'conversation_title': 'Synthetic',
            'conversation_type': 'private',
            'sender_name': 'Fixture',
            'direction': 'incoming',
            'timestamp': f'2026-01-0{index + 1}T00:00:00Z',
        }, ['vector'], 1.0 - index * 0.1)
        for index, content in enumerate(('hello world', 'document query'))
    ]


def _tiny_cross_encoder(path: Path) -> None:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    torch.manual_seed(0)
    (path / 'vocab.txt').write_text(
        '\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'world', 'query', 'document']) + '\n',
        encoding='utf-8',
    )
    tokenizer = BertTokenizerFast(vocab_file=str(path / 'vocab.txt'), do_lower_case=True)
    tokenizer.save_pretrained(path)
    model = BertForSequenceClassification(BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
        max_position_embeddings=64,
    ))
    model.save_pretrained(path)
    (path / 'trove_model_manifest.json').write_text(json.dumps({
        'model_id': 'synthetic/tiny-cross-encoder',
        'provider': 'sentence-transformers-cross-encoder',
    }), encoding='utf-8')


class LocalRerankerTests(unittest.TestCase):
    def test_identity_cache_singleflights_walk_and_invalidates_on_fingerprint_change(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / 'identity-cache-reranker'
            model_path.mkdir()
            (model_path / 'trove_model_manifest.json').write_text(json.dumps({
                'model_id': 'synthetic/identity-cache-reranker',
                'provider': 'sentence-transformers-cross-encoder',
            }), encoding='utf-8')
            config_path = model_path / 'config.json'
            config_path.write_text('{"revision":1}', encoding='utf-8')

            class ImmediateModel:
                def __init__(self, _path: Path):
                    pass

                def predict(self, pairs, **_kwargs):
                    return [float(index) for index, _pair in enumerate(pairs)]

            original_identity = local_reranker_module.identity_for_model
            calls = 0

            def counted_identity(path):
                nonlocal calls
                calls += 1
                return original_identity(path)

            with patch.object(local_reranker_module, 'identity_for_model', side_effect=counted_identity):
                first = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=500, limit=2, model_factory=ImmediateModel,
                )[1]
                second = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=500, limit=2, model_factory=ImmediateModel,
                )[1]
                self.assertEqual(calls, 1)
                self.assertEqual(first['identity']['model_hash'], second['identity']['model_hash'])

                config_path.write_text('{"revision":2,"changed":true}', encoding='utf-8')
                third = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=500, limit=2, model_factory=ImmediateModel,
                )[1]

            self.assertEqual(calls, 2)
            self.assertNotEqual(first['identity']['model_hash'], third['identity']['model_hash'])

    def test_identity_work_is_inside_end_to_end_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / 'slow-identity-reranker'
            model_path.mkdir()
            (model_path / 'trove_model_manifest.json').write_text(json.dumps({
                'model_id': 'synthetic/slow-identity-reranker',
                'provider': 'sentence-transformers-cross-encoder',
            }), encoding='utf-8')
            original_identity = local_reranker_module.identity_for_model
            model_calls: list[int] = []

            def slow_identity(path):
                time.sleep(0.5)
                return original_identity(path)

            class MustNotLoad:
                def __init__(self, _path: Path):
                    model_calls.append(1)

            start = time.perf_counter()
            with patch.object(local_reranker_module, 'identity_for_model', side_effect=slow_identity):
                _rows, status = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=10, limit=2, model_factory=MustNotLoad,
                )
            wall_ms = (time.perf_counter() - start) * 1000

            self.assertEqual(status['reason_code'], 'local_reranker_timeout')
            self.assertEqual(status['phase'], 'identity')
            self.assertFalse(status['invoked'])
            self.assertTrue(status['task_submitted'])
            self.assertEqual(model_calls, [])
            self.assertLess(wall_ms, 250.0)
            time.sleep(0.51)  # let the bounded identity task finish and release its slot

    @unittest.skipUnless(
        importlib.util.find_spec('sentence_transformers') is not None,
        'requires the optional local-embedding runtime extra',
    )
    def test_real_cross_encoder_identity_and_invocation_are_proven(self):
        from sentence_transformers import CrossEncoder

        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / 'tiny-reranker'
            model_path.mkdir()
            _tiny_cross_encoder(model_path)
            ranked = _ranked_rows()
            pairs = [('hello query', vector_document_text(row)) for row, _paths, _score in ranked]
            proof_model = CrossEncoder(
                str(model_path),
                device='cpu',
                trust_remote_code=False,
                local_files_only=True,
            )
            expected_scores = [float(score) for score in proof_model.predict(pairs, show_progress_bar=False)]
            expected = sorted(
                ((expected_scores[index], index, ranked[index][0]['citation']) for index in range(len(ranked))),
                key=lambda value: (-value[0], value[1]),
            )

            reranked, status = rerank_with_local_model(
                ranked,
                'hello query',
                model_path=str(model_path),
                timeout_ms=10_000,
                limit=2,
            )

            self.assertEqual(len(reranked), 2)
            self.assertEqual([item[0]['citation'] for item in reranked], [item[2] for item in expected])
            for returned, expected_item in zip(reranked, expected):
                self.assertAlmostEqual(returned[2], expected_item[0], places=6)
            self.assertEqual(status['state'], 'available')
            self.assertTrue(status['invoked'])
            self.assertTrue(status['completed'])
            self.assertEqual(status['invocation_count'], 1)
            self.assertEqual(status['candidate_count'], 2)
            self.assertEqual(status['identity']['provider'], 'sentence-transformers-cross-encoder')
            self.assertEqual(status['identity']['model_id'], 'synthetic/tiny-cross-encoder')
            self.assertEqual(len(status['identity']['model_hash']), 64)
            rendered = str(status)
            self.assertNotIn(d, rendered)
            self.assertNotIn('hello query', rendered)
            self.assertFalse(status['raw_content_included'])
            self.assertFalse(status['raw_paths_included'])
            warmup = warm_local_reranker(str(model_path))
            self.assertTrue(warmup['ok'])
            self.assertTrue(warmup['invoked'])
            self.assertFalse(warmup['private_text_used'])

    def test_timed_out_jobs_keep_executor_queue_hard_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / 'blocked-reranker'
            model_path.mkdir()
            (model_path / 'trove_model_manifest.json').write_text(json.dumps({
                'model_id': 'synthetic/blocked-reranker',
                'provider': 'sentence-transformers-cross-encoder',
            }), encoding='utf-8')
            release = threading.Event()

            class BlockingModel:
                def __init__(self, _path: Path):
                    pass

                def predict(self, pairs, **_kwargs):
                    release.wait(timeout=2)
                    return [0.0] * len(pairs)

            try:
                first = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=50, limit=2, model_factory=BlockingModel,
                )[1]
                second = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=50, limit=2, model_factory=BlockingModel,
                )[1]
                saturated = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=50, limit=2, model_factory=BlockingModel,
                )[1]
                self.assertEqual(first['reason_code'], 'local_reranker_timeout')
                self.assertEqual(second['reason_code'], 'local_reranker_timeout')
                self.assertEqual(saturated['reason_code'], 'local_reranker_saturated')
                self.assertFalse(saturated['invoked'])
            finally:
                release.set()

            class ImmediateModel:
                def __init__(self, _path: Path):
                    pass

                def predict(self, pairs, **_kwargs):
                    return [float(index) for index, _pair in enumerate(pairs)]

            recovered = None
            for _ in range(100):
                recovered = rerank_with_local_model(
                    _ranked_rows(), 'synthetic query', model_path=str(model_path),
                    timeout_ms=100, limit=2, model_factory=ImmediateModel,
                )[1]
                if recovered['state'] == 'available':
                    break
                time.sleep(0.01)
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered['state'], 'available')

    def test_mock_success_hook_is_forbidden(self):
        _rows, status = rerank_with_local_model(
            _ranked_rows(),
            'synthetic query',
            model_path='__mock__',
            timeout_ms=200,
            limit=2,
        )
        self.assertEqual(status['state'], 'unavailable_fallback')
        self.assertEqual(status['reason_code'], 'local_reranker_mock_forbidden')
        self.assertFalse(status['invoked'])

    def test_timeout_is_a_real_bounded_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d) / 'slow-reranker'
            model_path.mkdir()
            (model_path / 'trove_model_manifest.json').write_text(json.dumps({
                'model_id': 'synthetic/slow-reranker',
                'provider': 'sentence-transformers-cross-encoder',
            }), encoding='utf-8')

            class SlowModel:
                def __init__(self, _path: Path):
                    pass

                def predict(self, pairs, **_kwargs):
                    time.sleep(0.5)
                    return [0.0] * len(pairs)

            start = time.perf_counter()
            reranked, status = rerank_with_local_model(
                _ranked_rows(),
                'synthetic query',
                model_path=str(model_path),
                timeout_ms=50,
                limit=2,
                model_factory=SlowModel,
            )
            wall_ms = (time.perf_counter() - start) * 1000

            self.assertEqual(len(reranked), 2)
            self.assertEqual(status['state'], 'degraded')
            self.assertEqual(status['reason_code'], 'local_reranker_timeout')
            self.assertTrue(status['invoked'])
            self.assertLess(wall_ms, 250.0)
            time.sleep(0.51)  # release the bounded executor slot before cleanup


if __name__ == '__main__':
    unittest.main()
