from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import benchmark_search as benchmark_module
from scripts.gate_retrieval_eval import _load_release_config, build_gate_report
from trove_core.search.evidence_provenance import (
    EvidenceProvenanceError,
    stable_payload_sha256,
    verify_evidence_manifest,
    write_evidence_artifact,
)
from trove_core.search.eval_schema import validate_redacted_artifact


def _provenance(commit: str, *, temperature: str = 'cold', warmups: int = 0) -> dict:
    same = stable_payload_sha256('same')
    return {
        'schema_version': 1,
        'git': {'commit_sha': commit, 'dirty': False},
        'platform': {
            'system': 'TestOS', 'release': '1', 'machine': 'test',
            'python_implementation': 'CPython', 'python_version': '3.11.0',
            'cpu_count': 4, 'processor_sha256': same, 'memory_bytes': 1024,
        },
        'fixture': {'kind': 'synthetic_or_redacted', 'sha256': same},
        'seed': 7,
        'case_pack_sha256': same,
        'store': {
            'schema_version': 11, 'schema_manifest_sha256': same,
            'content_identity_sha256': same,
            'safe_metadata': {}, 'document_counts': {'messages': 2, 'chunks': 2, 'vectors': 0},
            'index_generation_sha256': same, 'document_count': 2,
        },
        'provider': {'provider_sha256': same, 'model_sha256': same, 'dimensions': 0},
        'execution': {
            'temperature': temperature,
            'warmups': warmups,
            'rounds': 3,
            'includes_engine_build': temperature == 'cold',
        },
        'privacy': {
            'raw_fixture_identity_included': False, 'raw_case_pack_included': False,
            'private_paths_included': False, 'provider_names_included': False,
            'model_names_included': False,
        },
    }


def _quality_report(commit: str) -> dict:
    metrics = {
        'recall_at_3': 1.0,
        'mrr': 1.0,
        'negative_pass_rate': 1.0,
        'case_success_rate': 1.0,
        'positive_queries': 1,
        'negative_only_queries': 1,
        'per_category': {
            'exact_sparse': {'positive_queries': 1, 'negative_only_queries': 0, 'recall_at_3': 1.0, 'negative_pass_rate': None},
            'negative_scope': {'positive_queries': 0, 'negative_only_queries': 1, 'recall_at_3': 0.0, 'negative_pass_rate': 1.0},
        },
    }
    return {
        'schema_version': 2,
        'artifact_type': 'retrieval_eval_matrix_redacted',
        'complete': True,
        'k': 3,
        'case_pack_anchor': {'sha256_prefix': 'same-pack'},
        'provenance': _provenance(commit),
        'modes': {'hybrid-weighted': {'metrics': metrics, 'cases': [
            {'case_hash': 'positive', 'positive_expected': True, 'negative_only': False, 'hit': True, 'reciprocal_rank': 1.0},
            {'case_hash': 'negative', 'positive_expected': False, 'negative_only': True, 'hit': True, 'reciprocal_rank': 0.0},
        ]}},
    }


def _benchmark(commit: str, *, temperature: str, semantic: str, p50: float = 10.0, p95: float = 20.0) -> dict:
    return {
        'schema_version': 2,
        'artifact_type': 'search_benchmark_redacted',
        'semantic_mode': semantic,
        'latency_ms': {'p50': p50, 'p95': p95},
        'provenance': _provenance(commit, temperature=temperature, warmups=1 if temperature == 'warm' else 0),
        'privacy': {'raw_queries_included': False, 'private_paths_included': False},
    }


class EvidenceProvenanceTests(unittest.TestCase):
    def test_sidecar_hash_is_independent_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            artifact = Path(d) / 'artifact.redacted.json'
            report = _quality_report('a' * 40)
            manifest = write_evidence_artifact(report, artifact)
            validate_redacted_artifact(json.loads(manifest.read_text(encoding='utf-8')))
            self.assertIsNotNone(verify_evidence_manifest(artifact, required=True))
            artifact.write_text(artifact.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaises(EvidenceProvenanceError):
                verify_evidence_manifest(artifact, required=True)

    def test_release_gate_requires_full_quality_and_four_latency_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'quality-base.json'
            cand = root / 'quality-candidate.json'
            write_evidence_artifact(_quality_report('a' * 40), base)
            write_evidence_artifact(_quality_report('b' * 40), cand)
            latency = {}
            for profile, temperature, semantic in (
                ('exact_cold', 'cold', 'off'), ('exact_warm', 'warm', 'off'),
                ('rewrite_cold', 'cold', 'on'), ('rewrite_warm', 'warm', 'on'),
            ):
                base_path = root / f'{profile}-base.json'
                cand_path = root / f'{profile}-candidate.json'
                write_evidence_artifact(_benchmark('a' * 40, temperature=temperature, semantic=semantic), base_path)
                write_evidence_artifact(_benchmark('b' * 40, temperature=temperature, semantic=semantic), cand_path)
                latency[profile] = {
                    'baseline_artifact': str(base_path), 'candidate_artifact': str(cand_path),
                    'semantic_mode': semantic, 'temperature': temperature,
                    'max_p50_ms': 50.0, 'max_p95_ms': 100.0,
                    'max_p50_regression_ratio': 1.2, 'max_p95_regression_ratio': 1.2,
                }
            config = {
                'schema_version': 1,
                '_config_root': str(root),
                'quality': {
                    'min_recall_at_k': 0.9, 'min_negative_pass_rate': 1.0,
                    'max_recall_drop': 0.0, 'max_mrr_drop': 0.0,
                    'category_floors': {
                        'exact_sparse': {'min_recall_at_k': 0.9},
                        'negative_scope': {'min_negative_pass_rate': 1.0},
                    },
                },
                'latency': latency,
            }
            passed = build_gate_report(base, cand, mode='hybrid-weighted', release_config=config)
            self.assertTrue(passed['ok'])
            self.assertTrue(passed['latency_gate']['ok'])
            self.assertTrue(passed['category_quality_gate']['ok'])

            broken = copy.deepcopy(_quality_report('b' * 40))
            broken.pop('provenance')
            write_evidence_artifact(broken, cand)
            failed = build_gate_report(base, cand, mode='hybrid-weighted', release_config=config)
            self.assertFalse(failed['ok'])
            self.assertIn('candidate_invalid_or_missing_provenance', failed['evidence_gate']['failures'])

    def test_release_gate_fails_when_configured_category_metrics_are_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'base.json'
            cand = root / 'candidate.json'
            base_report = _quality_report('a' * 40)
            candidate_report = _quality_report('b' * 40)
            del candidate_report['modes']['hybrid-weighted']['metrics']['per_category']['exact_sparse']
            write_evidence_artifact(base_report, base)
            write_evidence_artifact(candidate_report, cand)
            config = self._release_config_with_latency(root)

            report = build_gate_report(base, cand, mode='hybrid-weighted', release_config=config)

            self.assertFalse(report['ok'])
            self.assertIn('exact_sparse', report['category_quality_gate']['failed_categories'])
            self.assertIn(
                'missing_category_metrics',
                report['category_quality_gate']['categories']['exact_sparse']['failures'],
            )

    def test_release_config_rejects_non_finite_and_mislabeled_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self._release_config_with_latency(root)
            path = root / 'release.json'
            config.pop('_config_root', None)
            config['quality']['min_recall_at_k'] = float('nan')
            path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaises(SystemExit):
                _load_release_config(path)

            config['quality']['min_recall_at_k'] = 0.9
            config['latency']['exact_cold']['semantic_mode'] = 'on'
            path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaises(SystemExit):
                _load_release_config(path)

    def test_source_tree_does_not_commit_machine_specific_benchmark_bundle(self):
        bundle = (
            Path(__file__).resolve().parents[3]
            / 'docs'
            / 'perf'
            / '2026-07-09-u4-threshold-lock'
        )
        self.assertFalse(bundle.exists())

    def test_cold_benchmark_rebuilds_engine_inside_each_timed_request(self):
        class Engine:
            embedding_provider = None

            def __init__(self):
                self.closed = False

            def search(self, _request):
                return SimpleNamespace(
                    total=1,
                    retrieval_status={
                        'ranking': {'candidate_routes': {'exact': 1}, 'candidate_count': 1},
                        'vector': {'state': 'disabled'},
                    },
                )

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries = root / 'queries.jsonl'
            queries.write_text('{"query":"synthetic one"}\n{"query":"synthetic two"}\n', encoding='utf-8')
            vault = root / 'vault'
            cold_engines = []

            def build_cold(_cfg):
                engine = Engine()
                cold_engines.append(engine)
                return engine

            with patch.object(benchmark_module, 'build_search_engine', side_effect=build_cold) as build:
                cold = benchmark_module.run_benchmark(
                    str(vault), queries, rounds=2, limit=3, temperature='cold', warmups=0,
                )
            self.assertEqual(build.call_count, 4)
            self.assertTrue(all(engine.closed for engine in cold_engines))
            self.assertTrue(cold['provenance']['execution']['includes_engine_build'])

            warm_engines = []

            def build_warm(_cfg):
                engine = Engine()
                warm_engines.append(engine)
                return engine

            with patch.object(benchmark_module, 'build_search_engine', side_effect=build_warm) as build:
                warm = benchmark_module.run_benchmark(
                    str(vault), queries, rounds=2, limit=3, temperature='warm', warmups=1,
                )
            self.assertEqual(build.call_count, 1)
            self.assertTrue(warm_engines[0].closed)
            self.assertFalse(warm['provenance']['execution']['includes_engine_build'])

    def test_benchmark_closes_engine_when_search_raises(self):
        class FailingEngine:
            embedding_provider = None

            def __init__(self):
                self.closed = False

            def search(self, _request):
                raise RuntimeError('synthetic search failure')

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            queries = root / 'queries.jsonl'
            queries.write_text('{"query":"synthetic"}\n', encoding='utf-8')
            engine = FailingEngine()
            with patch.object(benchmark_module, 'build_search_engine', return_value=engine):
                with self.assertRaisesRegex(RuntimeError, 'synthetic search failure'):
                    benchmark_module.run_benchmark(
                        str(root / 'vault'), queries, rounds=1, limit=3,
                        temperature='cold', warmups=0,
                    )
            self.assertTrue(engine.closed)

    @staticmethod
    def _release_config_with_latency(root: Path) -> dict:
        latency = {}
        for profile, temperature, semantic in (
            ('exact_cold', 'cold', 'off'), ('exact_warm', 'warm', 'off'),
            ('rewrite_cold', 'cold', 'on'), ('rewrite_warm', 'warm', 'on'),
        ):
            base_path = root / f'{profile}-base.json'
            cand_path = root / f'{profile}-candidate.json'
            write_evidence_artifact(_benchmark('a' * 40, temperature=temperature, semantic=semantic), base_path)
            write_evidence_artifact(_benchmark('b' * 40, temperature=temperature, semantic=semantic), cand_path)
            latency[profile] = {
                'baseline_artifact': str(base_path), 'candidate_artifact': str(cand_path),
                'semantic_mode': semantic, 'temperature': temperature,
                'max_p50_ms': 50.0, 'max_p95_ms': 100.0,
                'max_p50_regression_ratio': 1.2, 'max_p95_regression_ratio': 1.2,
            }
        return {
            'schema_version': 1,
            '_config_root': str(root),
            'quality': {
                'min_recall_at_k': 0.9, 'min_negative_pass_rate': 1.0,
                'max_recall_drop': 0.0, 'max_mrr_drop': 0.0,
                'category_floors': {
                    'exact_sparse': {'min_recall_at_k': 0.9},
                    'negative_scope': {'min_negative_pass_rate': 1.0},
                },
            },
            'latency': latency,
        }


if __name__ == '__main__':
    unittest.main()
