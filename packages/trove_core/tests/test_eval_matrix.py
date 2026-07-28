from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from trove_core.search.eval_matrix import EvalCasePackCompatibilityError, _case_relevance, _evaluate_case, _failure_class, _matched_expected_citations, _summarize_mode, run_eval_matrix, run_mode
from trove_core.search.eval_schema import stable_hash, validate_redacted_artifact
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault
from scripts.gate_retrieval_eval import build_gate_report
from scripts.generate_real_eval_cases import add_balanced


class EvalMatrixTests(unittest.TestCase):
    def test_episode_bundle_counts_every_returned_supporting_citation(self):
        case = {
            'case_id': 'episode-bundle',
            'query': 'synthetic multi-hop',
            'oracle': {'expected_all_citations': ['support-a', 'support-b']},
        }
        results = [{
            'citation': 'representative',
            'context_anchor': 'representative',
            'supporting_citations': ['support-a', 'support-b'],
            'retrieval_paths': ['episode-bundle'],
        }]
        relevant, expected = _case_relevance(case, results)
        self.assertEqual(relevant, [1])
        self.assertEqual(_matched_expected_citations(case, results), expected)
        self.assertIsNone(_failure_class(case, results, relevant, k=3, context_ok=None))

    def test_cloud_matrix_mode_uses_bounded_production_rerank_path(self):
        class _Response:
            candidate_citations = ('candidate-a', 'candidate-b')
            retrieval_status = {
                'vector': {'state': 'available'},
                'reranker': {'state': 'available', 'invoked': True},
                'phase_latency_ms': {'retrieval': 1, 'fusion': 2, 'rerank': 3},
            }

            def to_dict(self):
                return {'results': []}

        class _Search:
            def __init__(self):
                self.request = None

            def search(self, request):
                self.request = request
                return _Response()

        class _Queries:
            def __init__(self):
                self.query = None
                self.candidate_limit = None

            def _cloud_rerank_response(self, query, response, *, result_limit, candidate_limit):
                self.query = query
                self.candidate_limit = candidate_limit
                self.result_limit = result_limit
                return response

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            search = _Search()
            queries = _Queries()
            result = run_mode(
                VaultConfig.resolve(str(vault), env={}),
                SQLiteStore(vault / 'index' / 'trove.sqlite'),
                {'case_id': 'cloud-production', 'query': 'synthetic query', 'filters': {}},
                'cloud-reranker',
                k=3,
                hybrid_search=search,
                cloud_queries=queries,
                reranker_candidate_limit=50,
            )

        self.assertEqual(search.request.limit, 20)
        self.assertEqual(search.request.reranker_mode, 'features')
        self.assertTrue(queries.query.allow_cloud_rerank)
        self.assertEqual(queries.candidate_limit, 20)
        self.assertEqual(queries.result_limit, 10)
        self.assertEqual(len(result['candidate_citation_hashes']), 2)

    def test_candidate_recall_is_measured_with_hashes_only(self):
        case = {
            'case_id': 'candidate-oracle',
            'query': 'needle',
            'oracle': {'expected_any_citation': ['expected-citation']},
        }
        with tempfile.TemporaryDirectory() as d:
            evaluated = _evaluate_case(
                SQLiteStore(Path(d) / 'trove.sqlite'),
                case,
                {
                    'results': [],
                    'elapsed_ms': 0.0,
                    'candidate_citation_hashes': [stable_hash('expected-citation')],
                },
                k=3,
            )
        self.assertTrue(evaluated['candidate_hit'])
        self.assertEqual(evaluated['candidate_recall'], 1.0)
        self.assertNotIn('expected-citation', str(evaluated))

    def test_matrix_runs_fixture_without_raw_query_leak(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            report = run_eval_matrix(vault, Path('tests/golden/retrieval_core.jsonl'), modes=['exact', 'fts', 'hybrid-weighted', 'cloud-reranker', 'vector-degraded'], k=3)
            validate_redacted_artifact(report)
            self.assertEqual(report['case_count'], 9)
            self.assertIn('hybrid-weighted', report['modes'])
            self.assertIn('cloud-reranker', report['modes'])
            self.assertIn('cloud_reranker_requires_exact_approval', report['modes']['cloud-reranker']['notes'])
            self.assertGreaterEqual(report['modes']['hybrid-weighted']['metrics']['recall_at_3'], 0.75)
            rendered = str(report)
            self.assertNotIn('价格太高', rendered)
            self.assertNotIn('trove://wechat/acct-work', rendered)
            self.assertEqual(report['modes']['vector-degraded']['vector_states'].get('degraded'), 9)
            self.assertEqual(report['case_pack_compatibility']['state'], 'compatible')

    def test_matrix_rejects_stale_oracles_with_redacted_typed_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            index_fixture_vault(vault, reset=True)
            cases = root / 'stale-cases.json'
            private_query = 'sensitive query must not be returned'
            stale_citation = 'trove://wechat/stale-account/stale-conversation/message_0/999'
            stale_conversation = 'stale-conversation'
            cases.write_text(json.dumps({'cases': [{
                'case_id': 'stale-case',
                'query': private_query,
                'category': 'exact_sparse',
                'oracle': {
                    'expected_any_citation': [stale_citation],
                    'expected_any_conversation_id': [stale_conversation],
                },
            }]}), encoding='utf-8')

            with self.assertRaises(EvalCasePackCompatibilityError) as raised:
                run_eval_matrix(vault, cases, modes=['exact'])

            failure = raised.exception.to_redacted_dict()
            validate_redacted_artifact(failure)
            self.assertEqual(failure['error_code'], 'case_pack_incompatible_with_index')
            compatibility = failure['compatibility']
            self.assertEqual(compatibility['state'], 'incompatible')
            self.assertEqual(compatibility['incompatible_cases'], 1)
            self.assertEqual(compatibility['missing_citation_oracle_cases'], 1)
            self.assertEqual(compatibility['citation_refs'], {'total': 1, 'found': 0, 'missing': 1})
            self.assertEqual(compatibility['conversation_only_oracle_cases'], 0)
            self.assertEqual(compatibility['missing_conversation_only_oracle_cases'], 0)
            self.assertEqual(compatibility['conversation_refs'], {'total': 0, 'found': 0, 'missing': 0})
            rendered = json.dumps(failure, ensure_ascii=False)
            self.assertNotIn(private_query, rendered)
            self.assertNotIn(stale_citation, rendered)
            self.assertNotIn(stale_conversation, rendered)
            self.assertNotIn(str(root), rendered)

    def test_matrix_accepts_non_message_citation_when_conversation_ref_is_not_message_backed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            index_fixture_vault(vault, reset=True)
            cases = root / 'mixed-oracles.json'
            cases.write_text(json.dumps({'cases': [{
                'case_id': 'mixed-any-case',
                'query': '校长下周三 试点评审',
                'category': 'voice_transcript',
                'filters': {'source_type': 'transcript'},
                'oracle': {
                    'expected_any_citation': [
                        'trove://wechat/stale-account/stale-conversation/message_0/999',
                        'trove://wechat/acct-work/conv-example_edu-private/message_0/1#voice-fixture-1',
                    ],
                    'expected_any_conversation_id': ['transcript-source-not-in-messages'],
                },
            }]}), encoding='utf-8')

            report = run_eval_matrix(vault, cases, modes=['exact'])

            self.assertEqual(report['case_pack_compatibility']['state'], 'compatible')
            self.assertEqual(report['case_pack_compatibility']['incompatible_cases'], 0)
            self.assertEqual(report['case_pack_compatibility']['citation_refs']['found'], 1)
            self.assertEqual(report['case_pack_compatibility']['conversation_refs']['found'], 0)
            self.assertEqual(report['case_pack_compatibility']['conversation_refs']['total'], 0)
            self.assertEqual(report['case_pack_compatibility']['conversation_only_oracle_cases'], 0)

    def test_matrix_rejects_stale_conversation_only_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            index_fixture_vault(vault, reset=True)
            cases = root / 'conversation-only.json'
            cases.write_text(json.dumps({'cases': [{
                'case_id': 'conversation-only-case',
                'query': 'semantic query',
                'category': 'semantic_paraphrase',
                'oracle': {'expected_any_conversation_id': ['stale-conversation-only']},
            }]}), encoding='utf-8')

            with self.assertRaises(EvalCasePackCompatibilityError) as raised:
                run_eval_matrix(vault, cases, modes=['exact'])

            compatibility = raised.exception.compatibility
            self.assertEqual(compatibility['citation_oracle_cases'], 0)
            self.assertEqual(compatibility['conversation_only_oracle_cases'], 1)
            self.assertEqual(compatibility['missing_conversation_only_oracle_cases'], 1)

    def test_matrix_controls_partial_and_resume_are_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            partial = vault / 'proof' / 'retrieval-eval' / 'redacted' / 'partial.redacted.json'
            report = run_eval_matrix(
                vault,
                Path('tests/golden/retrieval_core.jsonl'),
                modes=['hybrid-rrf', 'feature-rerank'],
                k=3,
                max_cases=2,
                sample_seed=42,
                partial_out=partial,
            )
            validate_redacted_artifact(report)
            self.assertTrue(report['complete'])
            self.assertEqual(report['case_count'], 2)
            self.assertTrue(partial.exists())
            partial_data = json.loads(partial.read_text(encoding='utf-8'))
            validate_redacted_artifact(partial_data)
            self.assertIn('case_quality', report)
            self.assertIn('case_pack_anchor', report)

            resumed = run_eval_matrix(
                vault,
                Path('tests/golden/retrieval_core.jsonl'),
                modes=['hybrid-rrf'],
                k=3,
                max_cases=2,
                sample_seed=42,
                resume_path=partial,
            )
            self.assertIn('resumed_cases:2', resumed['modes']['hybrid-rrf']['notes'])

    def test_matrix_runs_multimodal_fixture_categories(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            report = run_eval_matrix(vault, Path('tests/golden/retrieval_multimodal.jsonl'), modes=['parent_child', 'hybrid-weighted'], k=3)
            validate_redacted_artifact(report)
            self.assertEqual(report['case_count'], 22)
            per_category = report['modes']['parent_child']['metrics']['per_category']
            self.assertEqual(per_category['voice_transcript']['queries'], 10)
            self.assertEqual(per_category['image_observation']['queries'], 12)
            self.assertEqual(per_category['voice_transcript']['recall_at_3'], 1.0)
            self.assertEqual(per_category['image_observation']['recall_at_3'], 1.0)

    def test_relevance_accepts_chunk_parent_but_not_expected_conversation_for_recall(self):
        parent_case = {
            'query': 'needle',
            'oracle': {'expected_any_citation': ['parent-citation']},
        }
        relevant, _ = _case_relevance(parent_case, [
            {'citation': 'parent-citation#chunk-0', 'context_anchor': 'parent-citation', 'conversation_id': 'source-1'}
        ])
        self.assertEqual(relevant, [1])

        conversation_case = {
            'query': 'needle',
            'oracle': {'expected_any_citation': ['anchor-citation'], 'expected_any_conversation_id': ['conv-1']},
        }
        relevant, _ = _case_relevance(conversation_case, [
            {'citation': 'nearby-citation', 'conversation_id': 'conv-1'}
        ])
        self.assertEqual(relevant, [0])

    def test_recall_at_k_does_not_fail_when_hit_is_not_top_rank(self):
        case = {'oracle': {'expected_any_citation': ['c2']}}
        results = [{'citation': 'c1'}, {'citation': 'c2'}, {'citation': 'c3'}]
        self.assertIsNone(_failure_class(case, results, [0, 1, 0], k=3, context_ok=None))

    def test_negative_excluded_oracle_passes_only_when_excluded_citation_absent(self):
        case = {
            'case_id': 'negative-excluded-only',
            'query': 'needle',
            'oracle': {'negative_excluded_citations': ['bad-citation']},
        }
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ok = _evaluate_case(
                store,
                case,
                {'results': [{'citation': 'other-citation', 'retrieval_paths': ['fts']}], 'elapsed_ms': 0.0},
                k=3,
            )
            self.assertTrue(ok['hit'])

            failed = _evaluate_case(
                store,
                case,
                {'results': [{'citation': 'bad-citation', 'retrieval_paths': ['fts']}], 'elapsed_ms': 0.0},
                k=3,
            )
            self.assertFalse(failed['hit'])
            self.assertEqual(failed['failure_class'], 'negative_excluded_citation_in_topk')

    def test_summary_separates_positive_recall_from_negative_pass_rate(self):
        summary = _summarize_mode([
            {
                'category': 'positive',
                'positive_expected': True,
                'negative_only': False,
                'hit': False,
                'precision': 0.0,
                'reciprocal_rank': 0.0,
                'average_precision': 0.0,
                'ndcg_at_3': 0.0,
                'ndcg_at_10': 0.0,
                'latency_ms': 1.0,
                'phase_latency_ms': {'retrieval': 0.5, 'fusion': 0.1, 'rerank': 0.2},
                'reranker_elapsed_ms': 0.15,
                'retrieval_paths': ['fts'],
            },
            {
                'category': 'negative_scope',
                'positive_expected': False,
                'negative_only': True,
                'hit': True,
                'precision': 0.0,
                'reciprocal_rank': 0.0,
                'average_precision': 0.0,
                'ndcg_at_3': 0.0,
                'ndcg_at_10': 0.0,
                'latency_ms': 2.0,
                'phase_latency_ms': {'retrieval': 1.0, 'fusion': 0.2, 'rerank': 0.4},
                'reranker_elapsed_ms': None,
                'retrieval_paths': ['fts'],
            },
        ], k=3)

        self.assertEqual(summary['positive_queries'], 1)
        self.assertEqual(summary['negative_only_queries'], 1)
        self.assertEqual(summary['recall_at_3'], 0.0)
        self.assertEqual(summary['negative_pass_rate'], 1.0)
        self.assertEqual(summary['case_success_rate'], 0.5)
        self.assertEqual(summary['phase_latency_ms']['retrieval']['p95'], 1.0)
        self.assertEqual(summary['phase_latency_ms']['rerank']['samples'], 2)
        self.assertEqual(summary['reranker_latency_ms']['samples'], 1)
        self.assertEqual(summary['reranker_latency_ms']['p95'], 0.15)

    def test_relaxed_conversation_relevance_metrics_stay_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            case = {
                'case_id': 'relaxed-conversation-many-hits',
                'query': 'needle',
                'oracle': {
                    'expected_any_citation': ['anchor-citation'],
                    'expected_any_conversation_id': ['conv-1'],
                },
            }
            results = [
                {'citation': f'nearby-citation-{idx}', 'conversation_id': 'conv-1', 'retrieval_paths': ['fts']}
                for idx in range(10)
            ]

            evaluated = _evaluate_case(
                SQLiteStore(Path(d) / 'trove.sqlite'),
                case,
                {'results': results, 'elapsed_ms': 0.0},
                k=10,
            )

            self.assertGreaterEqual(evaluated['average_precision'], 0.0)
            self.assertLessEqual(evaluated['average_precision'], 1.0)
            self.assertGreaterEqual(evaluated['ndcg_at_3'], 0.0)
            self.assertLessEqual(evaluated['ndcg_at_3'], 1.0)
            self.assertGreaterEqual(evaluated['ndcg_at_10'], 0.0)
            self.assertLessEqual(evaluated['ndcg_at_10'], 1.0)

    def test_real_eval_gate_detects_and_ignores_hash_regressions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'base.redacted.json'
            cand = root / 'cand.redacted.json'
            base.write_text(json.dumps({
                'artifact_type': 'retrieval_eval_matrix_redacted',
                'complete': True,
                'k': 3,
                'case_pack_anchor': {'sha256_prefix': 'same-frozen-pack'},
                'modes': {'hybrid-weighted': {'metrics': {'recall_at_3': 1.0, 'mrr': 1.0}, 'cases': [
                    {'case_hash': 'stable-hit', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                    {'case_hash': 'known-drift', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                ]}},
            }), encoding='utf-8')
            cand.write_text(json.dumps({
                'artifact_type': 'retrieval_eval_matrix_redacted',
                'complete': True,
                'k': 3,
                'case_pack_anchor': {'sha256_prefix': 'same-frozen-pack'},
                'modes': {'hybrid-weighted': {'metrics': {'recall_at_3': 0.5, 'mrr': 0.5}, 'cases': [
                    {'case_hash': 'stable-hit', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                    {'case_hash': 'known-drift', 'positive_expected': True, 'hit': False, 'reciprocal_rank': 0.0},
                ]}},
            }), encoding='utf-8')

            failed = build_gate_report(base, cand, mode='hybrid-weighted')
            self.assertFalse(failed['ok'])
            self.assertEqual(failed['hit_regressions'], 1)

            ignored = build_gate_report(base, cand, mode='hybrid-weighted', ignore_case_hashes={'known-drift'})
            validate_redacted_artifact(ignored)
            self.assertTrue(ignored['ok'])
            self.assertEqual(ignored['case_counts']['common_compared'], 1)
            self.assertTrue(ignored['frozen_pack_gate']['ok'])

    def test_real_eval_gate_fails_without_same_frozen_pack_and_case_set(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'base.redacted.json'
            cand = root / 'cand.redacted.json'
            base.write_text(json.dumps({
                'artifact_type': 'retrieval_eval_matrix_redacted',
                'complete': True,
                'k': 3,
                'case_pack_anchor': {'sha256_prefix': 'pack-a'},
                'modes': {'hybrid-weighted': {'metrics': {'recall_at_3': 1.0, 'mrr': 1.0}, 'cases': [
                    {'case_hash': 'baseline-only', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                    {'case_hash': 'shared', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                ]}},
            }), encoding='utf-8')
            cand.write_text(json.dumps({
                'artifact_type': 'retrieval_eval_matrix_redacted',
                'complete': True,
                'k': 10,
                'case_pack_anchor': {'sha256_prefix': 'pack-b'},
                'modes': {'hybrid-weighted': {'metrics': {'recall_at_10': 1.0, 'mrr': 1.0}, 'cases': [
                    {'case_hash': 'shared', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                    {'case_hash': 'candidate-only', 'positive_expected': True, 'hit': True, 'reciprocal_rank': 1.0},
                ]}},
            }), encoding='utf-8')

            failed = build_gate_report(base, cand, mode='hybrid-weighted', ignore_case_hashes={'baseline-only'})

            self.assertFalse(failed['ok'])
            self.assertFalse(failed['frozen_pack_gate']['ok'])
            self.assertIn('case_pack_anchor_mismatch', failed['frozen_pack_gate']['failures'])
            self.assertIn('k_mismatch', failed['frozen_pack_gate']['failures'])
            self.assertIn('case_set_mismatch', failed['frozen_pack_gate']['failures'])
            self.assertEqual(failed['case_counts']['baseline_only'], 1)
            self.assertEqual(failed['case_counts']['candidate_only'], 1)

    def test_real_eval_gate_separates_positive_and_negative_regressions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'base.redacted.json'
            cand = root / 'cand.redacted.json'
            common = {
                'artifact_type': 'retrieval_eval_matrix_redacted',
                'complete': True,
                'k': 3,
                'case_pack_anchor': {'sha256_prefix': 'same-frozen-pack'},
                'modes': {'hybrid-weighted': {'metrics': {
                    'recall_at_3': 1.0,
                    'mrr': 1.0,
                    'negative_pass_rate': 1.0,
                    'case_success_rate': 1.0,
                }, 'cases': [
                    {'case_hash': 'positive-hit', 'positive_expected': True, 'negative_only': False, 'hit': True, 'reciprocal_rank': 1.0},
                    {'case_hash': 'negative-pass', 'positive_expected': False, 'negative_only': True, 'hit': True, 'reciprocal_rank': 0.0},
                ]}},
            }
            base.write_text(json.dumps(common), encoding='utf-8')
            candidate = json.loads(json.dumps(common))
            candidate['modes']['hybrid-weighted']['metrics']['negative_pass_rate'] = 0.0
            candidate['modes']['hybrid-weighted']['metrics']['case_success_rate'] = 0.5
            candidate['modes']['hybrid-weighted']['cases'][1]['hit'] = False
            cand.write_text(json.dumps(candidate), encoding='utf-8')

            failed = build_gate_report(base, cand, mode='hybrid-weighted')

            self.assertFalse(failed['ok'])
            self.assertEqual(failed['case_counts']['positive_compared'], 1)
            self.assertEqual(failed['case_counts']['negative_only_compared'], 1)
            self.assertEqual(failed['hit_regressions'], 0)
            self.assertEqual(failed['negative_regressions'], 1)
            self.assertTrue(failed['negative_pass_rate_floor_miss'])

    def test_real_eval_gate_enforces_v2_case_quality_when_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'base.redacted.json'
            cand = root / 'cand.redacted.json'
            common = {
                'artifact_type': 'retrieval_eval_matrix_redacted',
                'complete': True,
                'k': 3,
                'case_pack_anchor': {'sha256_prefix': 'same-frozen-pack'},
                'case_quality': {
                    'cases': 77,
                    'literal_substring_rate': 0.0,
                    'avg_word_overlap_ratio': 0.5,
                },
                'modes': {'hybrid-weighted': {'metrics': {'recall_at_3': 1.0, 'mrr': 1.0}, 'cases': [
                    {'case_hash': 'stable-hit', 'hit': True, 'reciprocal_rank': 1.0},
                ]}},
            }
            base.write_text(json.dumps(common), encoding='utf-8')
            candidate = json.loads(json.dumps(common))
            candidate['case_quality']['literal_substring_rate'] = 0.01
            cand.write_text(json.dumps(candidate), encoding='utf-8')

            failed = build_gate_report(base, cand, mode='hybrid-weighted', require_case_quality_v2=True, min_case_count=77)
            self.assertFalse(failed['ok'])
            self.assertIn('literal_substring_rate_above_max', failed['case_quality_gate']['failures'])

    def test_generated_cases_dedupe_ambiguous_query_filter_pairs(self):
        cases = []
        seen: set[str] = set()
        first = {'category': 'cross_source_family', 'query': 'same phrase', 'filters': {'source_type': 'favorite'}, 'private': {'anchor_citation': 'a'}}
        second = {'category': 'cross_source_family', 'query': 'same phrase', 'filters': {'source_type': 'favorite'}, 'private': {'anchor_citation': 'b'}}
        add_balanced(cases, seen, first, max_cases=10, per_category_limit=10)
        add_balanced(cases, seen, second, max_cases=10, per_category_limit=10)
        self.assertEqual(len(cases), 1)


if __name__ == '__main__':
    unittest.main()
