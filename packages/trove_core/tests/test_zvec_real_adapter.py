from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.embedding.base import HybridEmbedding
from trove_core.search.evidence_provenance import write_evidence_artifact
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.registry import VectorBackendRegistry
from trove_core.vector.score_calibration import (
    VectorScoreCalibrationError,
    build_score_calibration_artifact,
    embedding_identity,
)
from trove_core.vector.zvec_store import (
    ZVEC_ADAPTIVE_OVERFETCH_MAX,
    ZVEC_COLLECTION_CONTRACT_VERSION,
    ZVecStore,
    _embedding_text,
)
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import vault_generation_publish, vault_generation_read
from trove_core.vault.generation import VaultGenerationUnavailable
from trove_core.wechat.auxiliary_import import import_auxiliary_sources
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.models import Message


class _FakeCollection:
    def __init__(self):
        self.docs = []
        self.docs_by_id = {}
        self.deleted_ids = []

    def upsert(self, docs):
        self.docs.extend(docs)
        for doc in docs:
            self.docs_by_id[doc.id] = doc

    def delete(self, ids):
        if isinstance(ids, str):
            self.deleted_ids.append(ids)
            self.docs_by_id.pop(ids, None)
        else:
            self.deleted_ids.extend(ids)
            for doc_id in ids:
                self.docs_by_id.pop(doc_id, None)

    def flush(self):
        pass

    def query(self, *_args, **_kwargs):
        return []


class _SearchDoc:
    def __init__(self, citation: str, score: float | None):
        self.id = citation
        self.fields = {'citation': citation}
        self.score = score


class _SearchCollection:
    def __init__(self, citations: list[str], scores: list[float | None] | None = None):
        resolved_scores = scores or [1.0 - (index * 0.0001) for index in range(len(citations))]
        self.docs = [
            _SearchDoc(citation, score)
            for citation, score in zip(citations, resolved_scores)
        ]
        self.calls: list[dict[str, Any]] = []

    def query(self, *_args, **kwargs):
        self.calls.append(dict(kwargs))
        return self.docs[:int(kwargs['topk'])]


class _SearchStore(SQLiteStore):
    def __init__(self, path: Path, rows: dict[str, dict[str, str]]):
        super().__init__(path)
        self.initialize()
        self.rows = rows

    def evidence_by_citations(self, citations):
        return {citation: self.rows[citation] for citation in citations if citation in self.rows}

    def _filter_row(self, row, filters):
        return all(
            value == 'all' if key in {'source_family', 'scope_type'} else row.get(key) == value
            for key, value in filters.items()
        )


class _FakeZvecModule:
    class DataType:
        STRING = 'string'
        VECTOR_FP32 = 'vector'

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

    class HnswQueryParam:
        def __init__(self, *_args, **_kwargs):
            pass

    class Query:
        def __init__(self, *_args, **_kwargs):
            pass

    class Doc:
        def __init__(self, id: str, fields: dict[str, Any], vectors: dict[str, Any]):
            self.id = id
            self.fields = fields
            self.vectors = vectors

    collections: dict[str, _FakeCollection] = {}

    @classmethod
    def init(cls, *_args, **_kwargs):
        pass

    @classmethod
    def create_and_open(cls, path, _schema):
        Path(path).mkdir(parents=True, exist_ok=True)
        collection = _FakeCollection()
        cls.collections[str(path)] = collection
        return collection

    @classmethod
    def open(cls, path):
        return cls.collections.setdefault(str(path), _FakeCollection())


class _OtherFakeEmbeddingProvider(FakeEmbeddingProvider):
    name = 'fake-other'


class _HybridFakeEmbeddingProvider(FakeEmbeddingProvider):
    name = 'hybrid-fake'
    provider_name = 'fixture-cloud'
    model = 'fixture-dense-sparse'
    request_format = 'dashscope-native'
    supports_sparse = True
    query_instruct = 'fixture query instruction'

    def embed_hybrid_many(self, texts, *, text_type='document', instruct=None):
        dense = self.embed_many(list(texts))
        return [
            HybridEmbedding(vector, {index + 1: 0.5})
            for index, vector in enumerate(dense)
        ]


def _calibration_provenance() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'git': {'commit_sha': 'a' * 40, 'dirty': False},
        'platform': {
            'system': 'test', 'release': 'test', 'machine': 'test',
            'python_implementation': 'CPython', 'python_version': '3.13.0',
            'cpu_count': 1, 'processor_sha256': 'b' * 64, 'memory_bytes': None,
        },
        'fixture': {'kind': 'synthetic_or_redacted', 'sha256': 'c' * 64},
        'seed': 42,
        'case_pack_sha256': 'd' * 64,
        'store': {
            'schema_version': 1,
            'schema_manifest_sha256': 'e' * 64,
            'content_identity_sha256': 'f' * 64,
            'index_generation_sha256': '1' * 64,
            'document_count': 2,
        },
        'provider': {'provider_sha256': '2' * 64, 'model_sha256': '3' * 64, 'dimensions': 16},
        'execution': {'temperature': 'warm', 'warmups': 0, 'rounds': 2, 'includes_engine_build': False},
        'privacy': {
            'raw_fixture_identity_included': False,
            'raw_case_pack_included': False,
            'private_paths_included': False,
            'provider_names_included': False,
            'model_names_included': False,
        },
    }


class ZvecRealAdapterTests(unittest.TestCase):
    def _fake_zvec(self, path: Path, store: SQLiteStore) -> ZVecStore:
        zvec = ZVecStore(path, store=store, memory_limit_mb=256)
        zvec._zvec = _FakeZvecModule
        zvec._error = None
        return zvec

    def _assert_atomic_generation_consistent(self, zvec: ZVecStore, provider: FakeEmbeddingProvider) -> None:
        metadata = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
        progress = json.loads(zvec.progress_path.read_text(encoding='utf-8'))
        self.assertTrue(Path(zvec.collection_path).exists())
        self.assertTrue(metadata['complete'])
        self.assertTrue(progress['complete'])
        self.assertEqual(progress['indexed_count'], metadata['indexed_count'])
        self.assertFalse(zvec.swap_marker_path.exists())
        self.assertFalse(Path(str(zvec.collection_path) + '.trove-backup').exists())
        self.assertFalse(Path(str(zvec.metadata_path) + '.trove-backup').exists())
        self.assertFalse(Path(str(zvec.progress_path) + '.trove-backup').exists())
        self.assertFalse(list(Path(zvec.collection_path).parent.glob(Path(zvec.collection_path).name + '.trove-tmp*')))
        status = zvec.status(provider=provider)
        self.assertIn(status['state'], {'degraded', 'available', 'unavailable_fallback'})
        if status['state'] == 'degraded':
            self.assertEqual(status['reason_code'], 'vector_score_calibration_missing')
        self.assertFalse(status['incomplete'])
        with zvec.store.connect() as conn:
            self.assertEqual(int(conn.execute(
                """SELECT COUNT(*) FROM vector_index_ledger l
                   LEFT JOIN vector_index_generations g
                     ON g.backend=l.backend AND g.generation_id=l.generation_id
                   WHERE g.generation_id IS NULL"""
            ).fetchone()[0]), 0)
            self.assertEqual(int(conn.execute(
                "SELECT COUNT(*) FROM vector_index_generations WHERE backend='zvec' AND status='active'"
            ).fetchone()[0]), 1)
            self.assertEqual(int(conn.execute(
                "SELECT COUNT(*) FROM vector_index_generations WHERE backend='zvec' AND status<>'active'"
            ).fetchone()[0]), 0)

    def _bind_score_calibration(
        self,
        zvec: ZVecStore,
        provider: FakeEmbeddingProvider,
        *,
        query: str = 'synthetic calibration query',
        score_bounds: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        metadata = zvec._read_metadata()
        if not metadata:
            citations = sorted(getattr(zvec.store, 'rows', {}))
            identity = embedding_identity(provider)
            generation_id = 'a' * 32
            if zvec.ledger is None:
                self.fail('fixture ZVEC must have an authoritative ledger')
            zvec.ledger.begin_generation(
                generation_id,
                vector_text_version=3,
                embedding_provider=provider.name,
                embedding_model='',
                dimensions=provider.dimensions,
                expected_count=len(citations),
            )
            zvec.ledger.apply_delta(
                generation_id,
                upserts=((citation, f'hash-{index}') for index, citation in enumerate(citations)),
                expected_count=len(citations),
            )
            zvec.ledger.mark_ready(generation_id, expected_count=len(citations))
            zvec.ledger.activate(generation_id)
            generation = zvec.ledger.generation(generation_id)
            if generation is None:
                self.fail('fixture vector generation disappeared')
            metadata = {
                'schema_version': 4,
                'generation_id': generation_id,
                'generation_revision': generation.revision,
                'collection_contract_version': ZVEC_COLLECTION_CONTRACT_VERSION,
                'vector_text_version': 3,
                'dimensions': provider.dimensions,
                'embedding_provider': provider.name,
                'embedding_model': '',
                'embedding_dimensions': provider.dimensions,
                'embedding_request_format': '',
                'embedding_identity_sha256': identity['sha256'],
                'indexed_count': len(citations),
                'expected_document_count': len(citations),
                'complete': True,
                'backend': 'zvec',
            }
            zvec._write_metadata(metadata)
        metadata = zvec._authoritative_score_metadata(metadata)
        if score_bounds is None:
            candidates = zvec.calibration_candidates(
                query,
                limit=ZVEC_ADAPTIVE_OVERFETCH_MAX,
                provider=provider,
            )
            self.assertTrue(candidates)
            minimum = min(score for _row, score in candidates)
            score_bounds = (minimum - 0.02, minimum)
        artifact = build_score_calibration_artifact(
            metadata=metadata,
            provider=provider,
            max_negative_top_score=score_bounds[0],
            min_positive_target_score=score_bounds[1],
            positive_case_count=1,
            negative_case_count=1,
            case_pack_sha256='d' * 64,
            split_manifest_sha256='5' * 64,
            provenance=_calibration_provenance(),
        )
        artifact_path = Path(zvec.collection_path).parent / 'score-calibration.redacted.json'
        write_evidence_artifact(artifact, artifact_path)
        return zvec.apply_score_calibration_file(artifact_path, provider=provider)

    def _fork_crashing_atomic_rebuild(self, path: Path, sqlite_path: Path, provider: FakeEmbeddingProvider, crashpoint: str) -> int:
        if not hasattr(os, 'fork'):
            self.skipTest('real crash injection requires fork')
        pid = os.fork()
        if pid == 0:
            try:
                os.environ['TROVE_ZVEC_ATOMIC_CRASHPOINT'] = crashpoint
                child_store = SQLiteStore(sqlite_path)
                child_zvec = self._fake_zvec(path, child_store)
                child_zvec.atomic_rebuild(provider, batch_size=4)
            except BaseException:
                os._exit(98)
            os._exit(0)
        _pid, status = os.waitpid(pid, 0)
        self.assertTrue(os.WIFEXITED(status), status)
        self.assertEqual(os.WEXITSTATUS(status), 99, crashpoint)
        return os.WEXITSTATUS(status)

    def test_zvec_sidecar_replace_failure_preserves_previous_complete_json(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            zvec = ZVecStore(Path(d) / 'vectors' / 'messages', store=store)
            zvec.metadata_path.parent.mkdir(parents=True)
            previous = '{"schema_version":4,"complete":true}\n'
            zvec.metadata_path.write_text(previous, encoding='utf-8')

            with patch('trove_core.vector.zvec_store.os.replace', side_effect=OSError('fixture replace failure')):
                with self.assertRaises(OSError):
                    zvec._write_metadata({'schema_version': 4, 'complete': False})

            self.assertEqual(zvec.metadata_path.read_text(encoding='utf-8'), previous)
            self.assertEqual(list(zvec.metadata_path.parent.glob('.*.tmp')), [])

    def test_zvec_adapter_is_optional_or_searches_when_installed(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            zvec = ZVecStore(Path(d) / 'vectors' / 'zvec', store=store, memory_limit_mb=256)
            if not zvec.available:
                self.assertIn('ZVEC', zvec.unavailable_reason)
                return
            provider = FakeEmbeddingProvider(dimensions=16)
            self.assertGreater(zvec.index_all_messages(provider, batch_size=4), 0)
            self.assertEqual(zvec.status(provider=provider)['state'], 'degraded')
            blocked = VectorBackendRegistry(store=store, zvec_path=zvec.collection_path, provider=provider).status('zvec')
            self.assertEqual(blocked.state, 'degraded')
            self.assertEqual(blocked.selected_backend, 'none')
            self.assertEqual(blocked.reason_code, 'vector_score_calibration_missing')
            self._bind_score_calibration(zvec, provider, query='预算审批')
            rows = zvec.search('预算审批', limit=3, provider=provider)
            self.assertTrue(rows)
            self.assertIn('citation', rows[0].keys())
            target_conversation = str(rows[0]['conversation_id'])
            filtered = zvec.search(
                '预算审批',
                filters={'conversation_id': target_conversation},
                limit=6,
                provider=provider,
            )
            self.assertTrue(filtered)
            self.assertEqual({str(row['conversation_id']) for row in filtered}, {target_conversation})
            filter_status = zvec.last_search_status()
            self.assertEqual(filter_status['pushdown_keys'], ['conversation_id'])
            self.assertEqual(filter_status['residual_keys'], [])
            self.assertTrue(filter_status['complete'])

    def test_zvec_native_hybrid_schema_persists_dense_and_sparse_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            zvec = ZVecStore(Path(d) / 'vectors' / 'zvec-hybrid', store=store, memory_limit_mb=256)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            provider = _HybridFakeEmbeddingProvider(dimensions=16)
            self.assertGreater(zvec.index_all_messages(provider, batch_size=4), 0)
            generation = zvec.ledger.active_generation()
            self.assertIsNotNone(generation)
            citation = next(zvec.ledger.iter_citations(generation.generation_id))
            doc = zvec._open_existing().fetch(zvec._doc_id(citation), include_vector=True)[zvec._doc_id(citation)]
            self.assertTrue(doc.has_vector('embedding'))
            self.assertTrue(doc.has_vector('sparse_embedding'))
            metadata = zvec._read_metadata()
            self.assertTrue(metadata['embedding_sparse'])
            self.assertEqual(metadata['embedding_request_format'], 'dashscope-native')

    def test_zvec_calibrated_floor_rejects_low_similarity_without_score_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            citations = ['fixture-high', 'fixture-low']
            rows = {citation: {'citation': citation, 'source_type': 'message'} for citation in citations}
            path = Path(d) / 'zvec-score-floor'
            path.mkdir()
            collection = _SearchCollection(citations, scores=[0.80, 0.20])
            zvec = ZVecStore(path, store=_SearchStore(Path(d) / 'search.sqlite', rows), memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = collection
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))

            result = zvec.search('synthetic query', limit=2, provider=provider)

            self.assertEqual([row['citation'] for row in result], ['fixture-high'])
            status = zvec.last_search_status()
            self.assertEqual(status['score_rejected_count'], 1)
            self.assertTrue(status['score_floor_exhausted'])
            self.assertEqual(status['score_calibration']['state'], 'available')
            self.assertNotIn('fixture-', str(status))

    def test_zvec_missing_or_corrupt_revision_mirror_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            citation = 'redacted-citation'
            rows = {citation: {'citation': citation, 'source_type': 'message'}}
            path = Path(d) / 'zvec-revision-mirror'
            path.mkdir()
            zvec = ZVecStore(
                path,
                store=_SearchStore(Path(d) / 'search.sqlite', rows),
                memory_limit_mb=256,
            )
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = _SearchCollection([citation], scores=[0.80])
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))
            original = zvec._read_metadata()

            for label, value in (('missing', None), ('corrupt', 'not-an-integer')):
                with self.subTest(label=label):
                    broken = dict(original)
                    if value is None:
                        broken.pop('generation_revision', None)
                    else:
                        broken['generation_revision'] = value
                    zvec._write_metadata(broken)
                    status = zvec.status(provider=provider)
                    self.assertTrue(status['generation_revision_mismatch'])
                    self.assertEqual(status['reason_code'], 'vector_generation_revision_mismatch')
                    with self.assertRaisesRegex(
                        VectorScoreCalibrationError,
                        'vector_generation_revision_mismatch',
                    ):
                        zvec.search('synthetic query', limit=1, provider=provider)

            zvec._write_metadata(original)
            self.assertEqual(zvec.status(provider=provider)['state'], 'available')

    def test_zvec_live_tmp_build_keeps_old_active_generation_readable(self):
        with tempfile.TemporaryDirectory() as d:
            citation = 'redacted-citation'
            rows = {citation: {'citation': citation, 'source_type': 'message'}}
            path = Path(d) / 'zvec-live-build'
            path.mkdir()
            zvec = ZVecStore(
                path,
                store=_SearchStore(Path(d) / 'search.sqlite', rows),
                memory_limit_mb=256,
            )
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = _SearchCollection([citation], scores=[0.80])
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))

            Path(str(path) + '.trove-tmp-live').mkdir()

            self.assertTrue(zvec._atomic_recovery_required())
            self.assertIsNone(zvec._atomic_recovery_reason())
            self.assertEqual(zvec.status(provider=provider)['state'], 'available')
            self.assertEqual(
                [row['citation'] for row in zvec.search('synthetic query', limit=1, provider=provider)],
                [citation],
            )

    def test_zvec_missing_document_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            citations = ['fixture-scored', 'fixture-unscored']
            rows = {citation: {'citation': citation, 'source_type': 'message'} for citation in citations}
            path = Path(d) / 'zvec-unscored'
            path.mkdir()
            collection = _SearchCollection(citations, scores=[0.80, None])
            zvec = ZVecStore(path, store=_SearchStore(Path(d) / 'search.sqlite', rows), memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = collection
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))

            with self.assertRaisesRegex(VectorScoreCalibrationError, 'vector_unscored'):
                zvec.search('synthetic query', limit=2, provider=provider)
            status = zvec.last_search_status()
            self.assertEqual(status['returned_count'], 0)
            self.assertEqual(status['unscored_count'], 1)

    def test_zvec_residual_filter_uses_bounded_adaptive_overfetch(self):
        with tempfile.TemporaryDirectory() as d:
            citations = [f'fixture-{index:04d}' for index in range(120)]
            rows = {
                citation: {
                    'citation': citation,
                    'source_type': 'favorite' if index in {70, 90} else 'message',
                }
                for index, citation in enumerate(citations)
            }
            path = Path(d) / 'zvec-adaptive'
            path.mkdir()
            collection = _SearchCollection(citations)
            zvec = ZVecStore(path, store=_SearchStore(Path(d) / 'search.sqlite', rows), memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = collection
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))

            result = zvec.search(
                'synthetic query',
                filters={'source_type': 'favorite'},
                limit=2,
                provider=provider,
            )

            self.assertEqual([row['citation'] for row in result], ['fixture-0070', 'fixture-0090'])
            status = zvec.last_search_status()
            self.assertEqual(status['attempt_depths'], [50, 100])
            self.assertEqual(status['pushdown_keys'], [])
            self.assertEqual(status['residual_keys'], ['source_type'])
            self.assertTrue(status['adaptive_overfetch'])
            self.assertTrue(status['bounded'])
            self.assertTrue(status['complete'])

    def test_zvec_residual_filter_stops_at_hard_overfetch_bound(self):
        with tempfile.TemporaryDirectory() as d:
            citations = [f'fixture-{index:04d}' for index in range(2000)]
            rows = {
                citation: {'citation': citation, 'source_type': 'message'}
                for citation in citations
            }
            path = Path(d) / 'zvec-bounded'
            path.mkdir()
            collection = _SearchCollection(citations)
            zvec = ZVecStore(path, store=_SearchStore(Path(d) / 'search.sqlite', rows), memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = collection
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))

            result = zvec.search(
                'synthetic query',
                filters={'source_type': 'favorite'},
                limit=2,
                provider=provider,
            )

            self.assertEqual(result, [])
            status = zvec.last_search_status()
            self.assertEqual(status['attempt_depths'][-1], ZVEC_ADAPTIVE_OVERFETCH_MAX)
            self.assertLessEqual(len(status['attempt_depths']), 6)
            self.assertFalse(status['complete'])
            self.assertFalse(status['exhausted'])

    def test_zvec_pushdown_expression_is_redacted_and_injection_safe(self):
        with tempfile.TemporaryDirectory() as d:
            citation = 'fixture-target'
            rows = {citation: {'citation': citation, 'account_id': 'acct" OR account_id = "other'}}
            path = Path(d) / 'zvec-pushdown'
            path.mkdir()
            collection = _SearchCollection([citation])
            zvec = ZVecStore(path, store=_SearchStore(Path(d) / 'search.sqlite', rows), memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            zvec._collection = collection
            provider = FakeEmbeddingProvider(dimensions=16)
            self._bind_score_calibration(zvec, provider)

            result = zvec.search(
                'synthetic query',
                filters={'account_id': 'acct" OR account_id = "other'},
                limit=1,
                provider=provider,
            )

            self.assertEqual(len(result), 1)
            expression = collection.calls[-1]['filter']
            self.assertIn('\\" OR account_id = \\"other', expression)
            status = zvec.last_search_status()
            self.assertEqual(status['pushdown_keys'], ['account_id'])
            self.assertNotIn('acct', str(status))

    def test_partial_zvec_index_requires_rebuild_before_registry_selects_it(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            path = Path(d) / 'vectors' / 'zvec-partial'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            provider = FakeEmbeddingProvider(dimensions=16)
            self.assertEqual(zvec.index_all_messages(provider, batch_size=4, max_messages=3), 3)
            zvec_status = zvec.status()
            self.assertTrue(zvec_status['rebuild_required'])
            self.assertTrue(zvec_status['incomplete'])
            self.assertGreater(zvec_status['expected_document_count'], zvec_status['indexed_count'])

            reg = VectorBackendRegistry(store=store, zvec_path=path, provider=provider)
            status = reg.status('zvec').to_dict()
            self.assertEqual(status['state'], 'unavailable_fallback')
            self.assertEqual(status['selected_backend'], 'none')
            self.assertEqual(status['reason_code'], 'zvec_rebuild_required')

    def test_zvec_incremental_index_rebuilds_when_embedding_contract_changes(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            path = Path(d) / 'vectors' / 'zvec-contract'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()

            first = zvec.index_all_messages(FakeEmbeddingProvider(dimensions=16), batch_size=4)
            self.assertGreater(first, 0)
            unchanged = zvec.index_all_messages(FakeEmbeddingProvider(dimensions=16), batch_size=4)
            self.assertEqual(unchanged, 0)
            rebuilt = zvec.index_all_messages(_OtherFakeEmbeddingProvider(dimensions=16), batch_size=4)
            self.assertEqual(rebuilt, first)
            self.assertEqual(zvec.status(provider=_OtherFakeEmbeddingProvider(dimensions=16))['provider_mismatch'], False)

    def test_zvec_collection_contract_change_requires_full_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            path = Path(d) / 'vectors' / 'zvec-collection-contract'
            zvec = self._fake_zvec(path, store)
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            expected = zvec.index_all_messages(provider, batch_size=4)
            metadata = zvec._read_metadata()
            metadata['collection_contract_version'] = ZVEC_COLLECTION_CONTRACT_VERSION - 1
            zvec._write_metadata(metadata)

            status = zvec.status(provider=provider)

            self.assertTrue(status['collection_contract_mismatch'])
            self.assertTrue(status['rebuild_required'])
            self.assertEqual(status['reason_code'], 'zvec_rebuild_required')
            self.assertEqual(zvec.index_all_messages(provider, batch_size=4), expected)
            self.assertEqual(
                zvec._read_metadata()['collection_contract_version'],
                ZVEC_COLLECTION_CONTRACT_VERSION,
            )

    def test_zvec_atomic_rebuild_switches_collection_metadata_and_progress_together(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            sqlite_path = vault / 'index' / 'trove.sqlite'
            store = SQLiteStore(sqlite_path)
            path = Path(d) / 'vectors' / 'zvec-atomic-success'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)

            self.assertEqual(zvec.atomic_rebuild(provider, batch_size=4), sum(1 for _ in store.iter_vector_documents()))
            before = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
            anchor = store.all_messages()[0]
            new_message = Message(
                account_id=anchor['account_id'],
                account_label=anchor['account_label'],
                conversation_id=anchor['conversation_id'],
                conversation_title=anchor['conversation_title'],
                conversation_type=anchor['conversation_type'],
                sender_id='fixture-sender',
                sender_name='Fixture Sender',
                timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
                content='fixture zvec atomic rebuild new generation token',
                shard_id='message_9',
                local_id=9003,
            )
            store.upsert_messages([new_message])
            store.rebuild_message_chunks_for_conversations({(new_message.account_id, new_message.conversation_id)})

            indexed = zvec.atomic_rebuild(provider, batch_size=4)

            after = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
            progress = json.loads(zvec.progress_path.read_text(encoding='utf-8'))
            self.assertGreater(indexed, 0)
            self.assertGreater(after['indexed_count'], before['indexed_count'])
            self.assertEqual(progress['state'], 'complete')
            self.assertEqual(progress['indexed_count'], after['indexed_count'])
            self.assertTrue(after['complete'])
            self.assertFalse(zvec.swap_marker_path.exists())
            self.assertFalse(Path(str(path) + '.trove-backup').exists())

    def test_zvec_atomic_rebuild_retains_old_generation_until_reader_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            path = cfg.paths.vector_dir / 'zvec' / 'messages'
            zvec = self._fake_zvec(path, store)
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            zvec.atomic_rebuild(provider, batch_size=4)
            old_metadata = zvec.metadata_path.read_bytes()

            anchor = store.all_messages()[0]
            message = Message(
                account_id=anchor['account_id'],
                account_label=anchor['account_label'],
                conversation_id=anchor['conversation_id'],
                conversation_title=anchor['conversation_title'],
                conversation_type=anchor['conversation_type'],
                sender_id='generation-reader',
                sender_name='Generation Reader',
                timestamp=datetime(2026, 1, 4, tzinfo=timezone.utc),
                content='vector generation reader lease sentinel',
                shard_id='message_lease',
                local_id=9010,
            )
            store.upsert_messages([message])
            store.rebuild_message_chunks_for_conversations({(message.account_id, message.conversation_id)})
            finished = threading.Event()
            errors: list[BaseException] = []

            def rebuild() -> None:
                try:
                    zvec.atomic_rebuild(
                        provider,
                        batch_size=4,
                        generation_publish=lambda: vault_generation_publish(
                            cfg,
                            operation='vector-rebuild',
                        ),
                    )
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)
                finally:
                    finished.set()

            with vault_generation_read(cfg):
                worker = threading.Thread(target=rebuild, daemon=True)
                worker.start()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and not list(path.parent.glob('messages.trove-tmp-*.trove-meta.json')):
                    time.sleep(0.01)
                self.assertFalse(finished.is_set())
                self.assertEqual(zvec.metadata_path.read_bytes(), old_metadata)

            self.assertTrue(finished.wait(3.0))
            worker.join(3.0)
            self.assertFalse(errors)
            self.assertNotEqual(zvec.metadata_path.read_bytes(), old_metadata)
            self._assert_atomic_generation_consistent(zvec, provider)

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires fork semantics')
    def test_zvec_swap_crash_blocks_read_until_atomic_retry_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            sqlite_path = cfg.paths.sqlite_path
            path = cfg.paths.vector_dir / 'zvec' / 'messages'
            provider = FakeEmbeddingProvider(dimensions=16)
            store = SQLiteStore(sqlite_path)
            self._fake_zvec(path, store).atomic_rebuild(provider, batch_size=4)

            pid = os.fork()
            if pid == 0:  # pragma: no cover - child process
                try:
                    os.environ['TROVE_ZVEC_ATOMIC_CRASHPOINT'] = 'after_final_collection_to_backup'
                    child = self._fake_zvec(path, SQLiteStore(sqlite_path))
                    child.atomic_rebuild(
                        provider,
                        batch_size=4,
                        generation_publish=lambda: vault_generation_publish(
                            cfg,
                            operation='vector-rebuild',
                        ),
                    )
                except BaseException:
                    os._exit(98)
                os._exit(0)

            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 99)
            with self.assertRaises(VaultGenerationUnavailable) as blocked:
                with vault_generation_read(cfg):
                    pass
            self.assertEqual(blocked.exception.code, 'vault_generation_recovery_required')

            reopened = self._fake_zvec(path, SQLiteStore(sqlite_path))
            reopened.atomic_rebuild(
                provider,
                batch_size=4,
                generation_publish=lambda: vault_generation_publish(
                    cfg,
                    operation='vector-rebuild',
                ),
            )
            with vault_generation_read(cfg):
                pass
            self.assertFalse((vault / '.trove-generation-publish.json').exists())
            self._assert_atomic_generation_consistent(reopened, provider)

    def test_zvec_atomic_rebuild_recovers_from_each_swap_failpoint(self):
        failpoints = [
            'after_prepare',
            'after_final_collection_to_backup',
            'after_final_metadata_to_backup',
            'after_final_progress_to_backup',
            'after_tmp_collection_to_final',
            'after_tmp_metadata_to_final',
            'after_tmp_progress_to_final',
            'after_ledger_activation_before_marker',
            'after_ledger_activation',
        ]
        for failpoint in failpoints:
            with self.subTest(failpoint=failpoint), tempfile.TemporaryDirectory() as d:
                vault = Path(d) / 'vault'
                index_fixture_vault(vault, reset=True)
                store = SQLiteStore(vault / 'index' / 'trove.sqlite')
                path = Path(d) / 'vectors' / f'zvec-atomic-{failpoint}'
                zvec = ZVecStore(path, store=store, memory_limit_mb=256)
                zvec._zvec = _FakeZvecModule
                zvec._error = None
                _FakeZvecModule.collections.clear()
                provider = FakeEmbeddingProvider(dimensions=16)
                zvec.index_all_messages(provider, batch_size=4)
                before = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))

                os.environ['TROVE_ZVEC_ATOMIC_FAILPOINT'] = failpoint
                try:
                    with self.assertRaisesRegex(RuntimeError, 'simulated_zvec_atomic_failpoint'):
                        zvec.atomic_rebuild(provider, batch_size=4)
                finally:
                    os.environ.pop('TROVE_ZVEC_ATOMIC_FAILPOINT', None)

                recovery = zvec.recover_atomic_rebuild()
                after = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
                progress = json.loads(zvec.progress_path.read_text(encoding='utf-8'))
                self.assertIn(recovery['status'], {'noop', 'finalized', 'cleaned_stale_backup'})
                self.assertEqual(after['indexed_count'], before['indexed_count'])
                self.assertEqual(progress['indexed_count'], after['indexed_count'])
                self.assertTrue(path.exists())
                self.assertFalse(zvec.swap_marker_path.exists())
                self.assertFalse(Path(str(path) + '.trove-backup').exists())

    def test_zvec_atomic_rebuild_recovers_from_real_child_crashes(self):
        crash_matrix = [
            ('no_old_final_after_prepare', 'after_prepare', False, 0),
            ('old_final_after_final_collection_to_backup', 'after_final_collection_to_backup', True, 0),
            ('old_final_after_final_metadata_to_backup', 'after_final_metadata_to_backup', True, 0),
            ('old_final_before_tmp_to_final', 'after_final_progress_to_backup', True, 0),
            ('old_final_before_metadata_commit', 'after_tmp_collection_to_final', True, 0),
            ('old_final_after_metadata_commit', 'after_tmp_metadata_to_final', True, 0),
            ('old_final_after_tmp_to_final', 'after_tmp_progress_to_final', True, 0),
            ('old_final_after_ledger_commit', 'after_ledger_activation_before_marker', True, 0),
            ('old_final_after_ready_marker', 'after_ledger_activation', True, 0),
            ('orphan_tmp_half_finished', 'during_tmp_generation_progress', True, 1),
        ]
        provider = FakeEmbeddingProvider(dimensions=16)
        for label, crashpoint, old_final, min_removed_tmp in crash_matrix:
            with self.subTest(crashpoint=crashpoint), tempfile.TemporaryDirectory() as d:
                vault = Path(d) / 'vault'
                index_fixture_vault(vault, reset=True)
                sqlite_path = vault / 'index' / 'trove.sqlite'
                store = SQLiteStore(sqlite_path)
                path = Path(d) / 'vectors' / 'messages'
                _FakeZvecModule.collections.clear()
                if old_final:
                    self._fake_zvec(path, store).atomic_rebuild(provider, batch_size=4)

                self._fork_crashing_atomic_rebuild(path, sqlite_path, provider, crashpoint)

                reopened = self._fake_zvec(path, SQLiteStore(sqlite_path))
                recovery = reopened.recover_atomic_rebuild()
                self.assertIn(recovery['status'], {
                    'finalized',
                    'finalized_initial',
                    'restored_backup',
                    'discarded_incomplete_tmp',
                    'noop',
                }, label)
                if min_removed_tmp:
                    self.assertGreaterEqual(recovery.get('tmp_generations_removed', 0), min_removed_tmp, label)
                self._assert_atomic_generation_consistent(reopened, provider)

    def test_zvec_incremental_index_reads_only_dirty_citations(self):
        class CountingStore(SQLiteStore):
            def __init__(self, path: Path):
                super().__init__(path)
                self.rows_read = 0
                self.last_citations = None

            def iter_vector_documents(self, batch_size: int = 500, citations=None):
                self.last_citations = None if citations is None else list(citations)
                for row in super().iter_vector_documents(batch_size=batch_size, citations=self.last_citations):
                    self.rows_read += 1
                    yield row

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            sqlite_path = vault / 'index' / 'trove.sqlite'
            base_store = SQLiteStore(sqlite_path)
            dirty_parent = base_store.all_messages()[0]['citation']
            dirty_doc_count = sum(1 for _ in base_store.iter_vector_documents(citations=[dirty_parent]))
            full_doc_count = sum(1 for _ in base_store.iter_vector_documents())
            self.assertGreater(dirty_doc_count, 0)
            self.assertLess(dirty_doc_count, full_doc_count)

            store = CountingStore(sqlite_path)
            path = Path(d) / 'vectors' / 'zvec-dirty'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()

            provider = FakeEmbeddingProvider(dimensions=16)
            self.assertEqual(zvec.index_all_messages(provider, batch_size=4), full_doc_count)
            self.assertTrue(zvec.status(provider=provider)['complete'])
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))
            baseline_metadata = zvec._read_metadata()
            baseline_revision = zvec.ledger.generation(baseline_metadata['generation_id']).revision
            self.assertEqual(zvec.status(provider=provider)['state'], 'available')
            store.rows_read = 0
            store.last_citations = None

            indexed = zvec.index_all_messages(provider, batch_size=2, citations=[dirty_parent])

            self.assertEqual(indexed, 0)
            self.assertEqual(store.rows_read, dirty_doc_count)
            self.assertEqual(store.last_citations, [dirty_parent])
            after = zvec._read_metadata()
            self.assertEqual(after['generation_revision'], baseline_revision)
            self.assertEqual(zvec.ledger.generation(after['generation_id']).revision, baseline_revision)
            self.assertEqual(zvec.status(provider=provider)['state'], 'available')

    def test_zvec_incremental_accepts_complete_baseline_with_new_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            sqlite_path = vault / 'index' / 'trove.sqlite'
            store = SQLiteStore(sqlite_path)
            anchor = store.all_messages()[0]
            path = Path(d) / 'vectors' / 'zvec-new-chunk'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            full_count = sum(1 for _ in store.iter_vector_documents())

            self.assertEqual(zvec.index_all_messages(provider, batch_size=4), full_count)

            new_message = Message(
                account_id=anchor['account_id'],
                account_label=anchor['account_label'],
                conversation_id=anchor['conversation_id'],
                conversation_title=anchor['conversation_title'],
                conversation_type=anchor['conversation_type'],
                sender_id='fixture-sender',
                sender_name='Fixture Sender',
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                content='fixture zvec incremental new chunk token',
                shard_id='message_9',
                local_id=9001,
            )
            store.upsert_messages([new_message])
            store.rebuild_message_chunks_for_conversations({(new_message.account_id, new_message.conversation_id)})
            with store.connect() as conn:
                new_chunk = conn.execute(
                    'SELECT chunk_citation FROM evidence_chunks WHERE parent_citation=?',
                    (new_message.citation,),
                ).fetchone()['chunk_citation']

            catchup_status = zvec.status(provider=provider)
            self.assertFalse(catchup_status['rebuild_required'])
            self.assertTrue(catchup_status['catchup_pending'])
            self.assertEqual(catchup_status['reason_code'], 'zvec_catchup_pending')

            indexed = zvec.index_all_messages(provider, batch_size=2, citations=[new_message.citation])

            self.assertEqual(indexed, 1)
            metadata = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
            self.assertNotIn('content_hashes', metadata)
            generation = zvec.ledger.generation(metadata['generation_id'])
            self.assertIsNotNone(generation)
            self.assertIn(new_chunk, zvec.ledger.hashes(metadata['generation_id'], [new_chunk]))
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['indexed_count'], generation.indexed_count)
            self.assertLess(zvec.metadata_path.stat().st_size, 4096)

    def test_incremental_catchup_remains_queryable_and_accepts_next_delta(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            anchor = store.all_messages()[0]
            path = Path(d) / 'vectors' / 'zvec-partial-catchup'
            zvec = self._fake_zvec(path, store)
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            zvec.index_all_messages(provider, batch_size=4)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))
            calibration_sha = zvec.status(provider=provider)['score_calibration']['artifact_sha256']

            messages = [
                Message(
                    account_id=anchor['account_id'],
                    account_label=anchor['account_label'],
                    conversation_id=anchor['conversation_id'],
                    conversation_title=anchor['conversation_title'],
                    conversation_type=anchor['conversation_type'],
                    sender_id=f'fixture-catchup-sender-{index}',
                    sender_name=f'Fixture Catchup Sender {index}',
                    timestamp=datetime(2026, 1, 2 + index, tzinfo=timezone.utc),
                    content=f'fixture partial catchup calibration token {index}',
                    shard_id='message_9',
                    local_id=9200 + index,
                )
                for index in range(2)
            ]
            store.upsert_messages(messages)
            store.rebuild_message_chunks_for_conversations({
                (messages[0].account_id, messages[0].conversation_id),
            })

            self.assertEqual(
                zvec.index_all_messages(provider, batch_size=2, citations=[messages[0].citation]),
                1,
            )
            partial = zvec._read_metadata()
            self.assertFalse(partial['complete'])
            self.assertTrue(partial['catchup_pending'])
            partial_status = zvec.status(provider=provider)
            self.assertEqual(partial_status['state'], 'available')
            self.assertTrue(partial_status['catchup_pending'])
            self.assertEqual(
                partial_status['score_calibration']['artifact_sha256'],
                calibration_sha,
            )
            self.assertIsInstance(
                zvec.search('fixture partial catchup', limit=2, provider=provider),
                list,
            )

            self.assertEqual(
                zvec.index_all_messages(provider, batch_size=2, citations=[messages[1].citation]),
                1,
            )
            complete = zvec.status(provider=provider)
            self.assertTrue(complete['complete'])
            self.assertFalse(complete['catchup_pending'])
            self.assertEqual(complete['state'], 'available')
            self.assertEqual(
                complete['score_calibration']['artifact_sha256'],
                calibration_sha,
            )

    def test_precomputed_incremental_delta_preserves_score_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            anchor = store.all_messages()[0]
            path = Path(d) / 'vectors' / 'zvec-precomputed-delta'
            zvec = self._fake_zvec(path, store)
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            zvec.index_all_messages(provider, batch_size=4)
            self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))
            before = zvec.status(provider=provider)

            new_message = Message(
                account_id=anchor['account_id'],
                account_label=anchor['account_label'],
                conversation_id=anchor['conversation_id'],
                conversation_title=anchor['conversation_title'],
                conversation_type=anchor['conversation_type'],
                sender_id='fixture-precomputed-sender',
                sender_name='Fixture Precomputed Sender',
                timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
                content='fixture precomputed incremental calibration token',
                shard_id='message_9',
                local_id=9002,
            )
            store.upsert_messages([new_message])
            store.rebuild_message_chunks_for_conversations({
                (new_message.account_id, new_message.conversation_id),
            })
            row = dict(next(store.iter_vector_documents(citations=[new_message.citation])))
            text = _embedding_text(row)
            row['content_hash'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
            row['vector'] = provider.embed(text)

            delta = zvec.apply_precomputed_delta(
                provider,
                rows=[row],
                deletes=[],
                expected_count=sum(1 for _ in store.iter_vector_documents()),
            )

            self.assertEqual(delta, {'indexed': 1, 'deleted': 0})
            after = zvec.status(provider=provider)
            self.assertEqual(after['state'], 'available')
            self.assertEqual(
                after['score_calibration']['artifact_sha256'],
                before['score_calibration']['artifact_sha256'],
            )

    def test_zvec_incremental_replays_crashes_before_and_after_ledger_commit(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        cases = (
            ('after_incremental_collection_flush', False, 1),
            ('after_incremental_ledger', True, 0),
        )
        for failpoint, ledger_committed, retry_indexed in cases:
            with self.subTest(failpoint=failpoint), tempfile.TemporaryDirectory() as d:
                vault = Path(d) / 'vault'
                index_fixture_vault(vault, reset=True)
                store = SQLiteStore(vault / 'index' / 'trove.sqlite')
                anchor = store.all_messages()[0]
                path = Path(d) / 'vectors' / failpoint
                zvec = self._fake_zvec(path, store)
                _FakeZvecModule.collections.clear()
                zvec.index_all_messages(provider, batch_size=4)
                self._bind_score_calibration(zvec, provider, score_bounds=(0.20, 0.80))
                before_crash = zvec._read_metadata()
                before_revision = zvec.ledger.generation(before_crash['generation_id']).revision

                new_message = Message(
                    account_id=anchor['account_id'],
                    account_label=anchor['account_label'],
                    conversation_id=anchor['conversation_id'],
                    conversation_title=anchor['conversation_title'],
                    conversation_type=anchor['conversation_type'],
                    sender_id='fixture-crash-sender',
                    sender_name='Fixture Crash Sender',
                    timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
                    content=f'fixture incremental crash replay {failpoint}',
                    shard_id='message_9',
                    local_id=9100,
                )
                store.upsert_messages([new_message])
                store.rebuild_message_chunks_for_conversations({
                    (new_message.account_id, new_message.conversation_id),
                })
                with store.connect() as conn:
                    new_chunk = conn.execute(
                        'SELECT chunk_citation FROM evidence_chunks WHERE parent_citation=?',
                        (new_message.citation,),
                    ).fetchone()['chunk_citation']
                generation_id = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))['generation_id']

                os.environ['TROVE_ZVEC_ATOMIC_FAILPOINT'] = failpoint
                try:
                    with self.assertRaisesRegex(RuntimeError, 'simulated_zvec_atomic_failpoint'):
                        zvec.index_all_messages(provider, batch_size=2, citations=[new_message.citation])
                finally:
                    os.environ.pop('TROVE_ZVEC_ATOMIC_FAILPOINT', None)

                self.assertTrue(zvec.swap_marker_path.exists())
                hashes_after_crash = zvec.ledger.hashes(generation_id, [new_chunk])
                self.assertEqual(new_chunk in hashes_after_crash, ledger_committed)
                crash_generation = zvec.ledger.generation(generation_id)
                self.assertEqual(
                    crash_generation.revision,
                    before_revision + int(ledger_committed),
                )
                crash_status = zvec.status(provider=provider)
                self.assertTrue(crash_status['recovery_required'])
                self.assertEqual(crash_status['state'], 'degraded')
                self.assertEqual(crash_status['reason_code'], 'vector_incremental_replay_required')
                if ledger_committed:
                    self.assertTrue(crash_status['generation_revision_mismatch'])
                else:
                    self.assertFalse(crash_status['generation_revision_mismatch'])
                with self.assertRaisesRegex(
                    VectorScoreCalibrationError,
                    'vector_incremental_replay_required',
                ):
                    zvec.search('synthetic query', limit=2, provider=provider)
                self.assertEqual(
                    zvec.recover_atomic_rebuild()['status'],
                    'incremental_replay_required',
                )
                self.assertEqual(
                    zvec.status(provider=provider)['reason_code'],
                    'vector_incremental_replay_required',
                )

                self.assertEqual(
                    zvec.index_all_messages(provider, batch_size=2, citations=[new_message.citation]),
                    retry_indexed,
                )
                metadata = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
                generation = zvec.ledger.generation(generation_id)
                self.assertFalse(zvec.swap_marker_path.exists())
                self.assertIn(new_chunk, zvec.ledger.hashes(generation_id, [new_chunk]))
                self.assertEqual(metadata['indexed_count'], generation.indexed_count)
                self.assertEqual(metadata['generation_revision'], generation.revision)
                self.assertEqual(generation.revision, before_revision + 1)
                self.assertTrue(metadata['complete'])
                replayed_status = zvec.status(provider=provider)
                self.assertFalse(replayed_status['generation_revision_mismatch'])
                self.assertEqual(zvec.status(provider=provider)['state'], 'available')

    def test_zvec_status_distinguishes_catchup_from_incomplete_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            anchor = store.all_messages()[0]
            path = Path(d) / 'vectors' / 'zvec-status'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            zvec.index_all_messages(provider, batch_size=4)
            new_message = Message(
                account_id=anchor['account_id'],
                account_label=anchor['account_label'],
                conversation_id=anchor['conversation_id'],
                conversation_title=anchor['conversation_title'],
                conversation_type=anchor['conversation_type'],
                sender_id='fixture-sender',
                sender_name='Fixture Sender',
                timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
                content='fixture zvec catchup status token',
                shard_id='message_9',
                local_id=9002,
            )
            store.upsert_messages([new_message])
            store.rebuild_message_chunks_for_conversations({(new_message.account_id, new_message.conversation_id)})

            status = zvec.status(provider=provider)
            self.assertFalse(status['rebuild_required'])
            self.assertTrue(status['catchup_pending'])
            self.assertEqual(status['reason_code'], 'zvec_catchup_pending')

            metadata = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
            metadata['complete'] = False
            zvec.metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            interrupted = zvec.status(provider=provider)

            self.assertTrue(interrupted['rebuild_required'])
            self.assertTrue(interrupted['incomplete'])
            self.assertFalse(interrupted['catchup_pending'])
            self.assertEqual(interrupted['reason_code'], 'zvec_rebuild_required')

            with self.assertRaisesRegex(RuntimeError, 'complete existing collection'):
                zvec.index_all_messages(provider, batch_size=2, citations=[new_message.citation])

    def test_zvec_incremental_deletes_removed_favorite_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            acct = root / 'com.tencent.xinWeChat__wxid_vector_delete_fixture'
            acct.mkdir()
            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('CREATE TABLE favorite_item(fav_id TEXT, update_time TEXT, title TEXT, content TEXT)')
                conn.execute('INSERT INTO favorite_item VALUES(?,?,?,?)', ('f1', '2026-01-01', 'Fixture favorite', 'fixture favorite delete token'))
                conn.commit()
            store = SQLiteStore(root / 'vault' / 'index' / 'trove.sqlite')
            repo = MultimodalRepository(store)

            report = import_auxiliary_sources(acct, account_id='acct-delete', store=store, repo=repo)
            self.assertEqual(report.favorites_imported, 1)
            store.rebuild_evidence_chunks_for_source_types(report.changed_families().keys())
            with store.connect() as conn:
                favorite = conn.execute('SELECT citation FROM favorites').fetchone()
                self.assertIsNotNone(favorite)
                parent_citation = favorite['citation']
                chunk_citation = conn.execute(
                    'SELECT chunk_citation FROM evidence_chunks WHERE parent_citation=?',
                    (parent_citation,),
                ).fetchone()['chunk_citation']

            path = root / 'vectors' / 'zvec-favorite-delete'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            self.assertEqual(zvec.index_all_messages(provider, batch_size=2), 1)
            collection = _FakeZvecModule.collections[str(path)]
            doc_id = zvec._doc_id(chunk_citation)
            self.assertIn(doc_id, collection.docs_by_id)

            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('DELETE FROM favorite_item')
                conn.commit()
            removed_report = import_auxiliary_sources(acct, account_id='acct-delete', store=store, repo=repo)
            self.assertEqual(removed_report.removed_counts.get('favorite'), 1)
            self.assertTrue(any(ref['citation'] == parent_citation for ref in removed_report.dirty_refs()))
            store.rebuild_evidence_chunks_for_source_types(removed_report.changed_families().keys())

            indexed = zvec.index_all_messages(provider, batch_size=2, citations=[parent_citation])

            metadata = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(indexed, 0)
            self.assertNotIn('content_hashes', metadata)
            self.assertNotIn(chunk_citation, zvec.ledger.hashes(metadata['generation_id'], [chunk_citation]))
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['indexed_count'], zvec.ledger.generation(metadata['generation_id']).indexed_count)
            self.assertNotIn(doc_id, collection.docs_by_id)

    def test_zvec_full_scan_reconciles_removed_favorite_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            acct = root / 'com.tencent.xinWeChat__wxid_vector_full_delete_fixture'
            acct.mkdir()
            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('CREATE TABLE favorite_item(fav_id TEXT, update_time TEXT, title TEXT, content TEXT)')
                conn.execute('INSERT INTO favorite_item VALUES(?,?,?,?)', ('f1', '2026-01-01', 'Fixture favorite', 'fixture favorite full delete token'))
                conn.commit()
            store = SQLiteStore(root / 'vault' / 'index' / 'trove.sqlite')
            repo = MultimodalRepository(store)

            report = import_auxiliary_sources(acct, account_id='acct-full-delete', store=store, repo=repo)
            self.assertEqual(report.favorites_imported, 1)
            store.rebuild_evidence_chunks_for_source_types(report.changed_families().keys())
            with store.connect() as conn:
                parent_citation = conn.execute('SELECT citation FROM favorites').fetchone()['citation']
                chunk_citation = conn.execute(
                    'SELECT chunk_citation FROM evidence_chunks WHERE parent_citation=?',
                    (parent_citation,),
                ).fetchone()['chunk_citation']

            path = root / 'vectors' / 'zvec-favorite-full-delete'
            zvec = ZVecStore(path, store=store, memory_limit_mb=256)
            zvec._zvec = _FakeZvecModule
            zvec._error = None
            _FakeZvecModule.collections.clear()
            provider = FakeEmbeddingProvider(dimensions=16)
            self.assertEqual(zvec.index_all_messages(provider, batch_size=2), 1)
            collection = _FakeZvecModule.collections[str(path)]
            doc_id = zvec._doc_id(chunk_citation)
            self.assertIn(doc_id, collection.docs_by_id)

            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('DELETE FROM favorite_item')
                conn.commit()
            removed_report = import_auxiliary_sources(acct, account_id='acct-full-delete', store=store, repo=repo)
            self.assertEqual(removed_report.removed_counts.get('favorite'), 1)
            store.rebuild_evidence_chunks_for_source_types(removed_report.changed_families().keys())

            indexed = zvec.index_all_messages(provider, batch_size=2)

            metadata = json.loads(zvec.metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(indexed, 0)
            self.assertNotIn('content_hashes', metadata)
            self.assertNotIn(chunk_citation, zvec.ledger.hashes(metadata['generation_id'], [chunk_citation]))
            self.assertEqual(metadata['deleted_count'], 1)
            self.assertTrue(metadata['complete'])
            self.assertEqual(metadata['indexed_count'], zvec.ledger.generation(metadata['generation_id']).indexed_count)
            self.assertNotIn(doc_id, collection.docs_by_id)
