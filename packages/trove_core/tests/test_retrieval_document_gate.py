from __future__ import annotations

import copy
import unittest

from trove_core.search.retrieval_document_gate import evaluate_retrieval_document_migration


def _eval_report() -> dict:
    metrics = {
        'recall_at_3': 1.0,
        'mrr': 1.0,
        'negative_pass_rate': 1.0,
        'per_category': {
            'rewrite': {
                'positive_queries': 1,
                'negative_only_queries': 0,
                'recall_at_3': 1.0,
                'negative_pass_rate': None,
            },
            'negative_scope': {
                'positive_queries': 0,
                'negative_only_queries': 1,
                'recall_at_3': 0.0,
                'negative_pass_rate': 1.0,
            },
        },
    }
    return {
        'artifact_type': 'retrieval_eval_matrix_redacted',
        'complete': True,
        'k': 3,
        'case_pack_anchor': {'sha256_prefix': 'synthetic-frozen-pack'},
        'modes': {'hybrid-weighted': {'metrics': metrics, 'cases': [
            {'case_hash': 'positive', 'hit': True, 'negative_only': False, 'reciprocal_rank': 1.0},
            {'case_hash': 'negative', 'hit': True, 'negative_only': True, 'reciprocal_rank': 0.0},
        ]}},
    }


def _write_cost() -> dict:
    return {
        'artifact_type': 'retrieval_write_cost_redacted',
        'complete': True,
        'metrics': {
            'elapsed_ms': 100.0,
            'bytes_written': 1000,
            'fts_rows_written': 100,
            'vector_rows_written': 100,
        },
    }


class RetrievalDocumentGateTests(unittest.TestCase):
    def test_missing_ab_evidence_keeps_current_documents(self):
        report = evaluate_retrieval_document_migration({}, {}, {}, {})
        self.assertFalse(report['ok'])
        self.assertEqual(report['decision'], 'retain_current_retrieval_documents')
        self.assertFalse(report['automatic_migration'])

    def test_quality_regression_blocks_migration(self):
        baseline = _eval_report()
        candidate = copy.deepcopy(baseline)
        candidate['modes']['hybrid-weighted']['cases'][0]['hit'] = False
        candidate['modes']['hybrid-weighted']['metrics']['recall_at_3'] = 0.0

        report = evaluate_retrieval_document_migration(baseline, candidate, _write_cost(), _write_cost())

        self.assertFalse(report['ok'])
        self.assertIn('positive_hit_regression', report['failures'])
        self.assertEqual(report['decision'], 'retain_current_retrieval_documents')

    def test_write_cost_regression_blocks_migration(self):
        baseline = _eval_report()
        candidate_cost = _write_cost()
        candidate_cost['metrics']['fts_rows_written'] = 101

        report = evaluate_retrieval_document_migration(baseline, copy.deepcopy(baseline), _write_cost(), candidate_cost)

        self.assertFalse(report['ok'])
        self.assertIn('write_cost_regression', report['failures'])
        self.assertFalse(report['write_cost_gate']['checks']['fts_rows_written'])

    def test_green_ab_is_manual_eligibility_not_automatic_mutation(self):
        baseline = _eval_report()
        report = evaluate_retrieval_document_migration(
            baseline,
            copy.deepcopy(baseline),
            _write_cost(),
            _write_cost(),
        )

        self.assertTrue(report['ok'])
        self.assertEqual(report['decision'], 'eligible_for_manual_migration')
        self.assertFalse(report['automatic_migration'])


if __name__ == '__main__':
    unittest.main()
