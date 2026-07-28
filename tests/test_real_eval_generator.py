from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import generate_real_eval_cases
from trove_core.wechat.indexer import index_fixture_vault


class RealEvalGeneratorTests(unittest.TestCase):
    def test_scaled_plan_allows_tiny_case_counts(self):
        plan = generate_real_eval_cases.scaled_plan(2)
        self.assertEqual(sum(count for _category, count in plan), 2)
        self.assertTrue(any(count == 0 for _category, count in plan))

    def test_customer_profile_is_in_regular_eval_plan(self):
        plan = dict(generate_real_eval_cases.scaled_plan(90))
        self.assertEqual(plan['customer_profile'], 10)
        self.assertEqual(plan['multi_hop'], 8)
        self.assertEqual(plan['negative_scope'], 8)
        self.assertEqual(plan['voice_transcript'], 8)
        self.assertEqual(plan['image_observation'], 8)

    def test_max_cases_two_completes_without_looping(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = generate_real_eval_cases.main(['--vault', str(vault), '--max-cases', '2'])
            self.assertEqual(code, 0)
            summary = json.loads(buf.getvalue())
            self.assertTrue(summary['ok'])
            self.assertEqual(summary['case_count'], 2)
            self.assertEqual(summary['case_quality']['literal_substring_rate'], 0.0)
            self.assertLessEqual(summary['case_quality']['avg_word_overlap_ratio'], 0.5)
            self.assertFalse(summary['raw_queries_printed'])

    def test_rewrite_query_is_not_literal_substring(self):
        query, quality = generate_real_eval_cases.rewrite_query(
            '我们看了方案，功能认可，但是价格太高，预算审批也没过。',
            category='semantic_paraphrase',
            strategy='task_question',
        )
        self.assertNotIn(query.replace(' ', ''), '我们看了方案，功能认可，但是价格太高，预算审批也没过。')
        self.assertFalse(quality['literal_substring'])
        self.assertLessEqual(quality['word_overlap_ratio'], 0.5)

    def test_negative_excluded_case_has_no_positive_expected_citation(self):
        case = generate_real_eval_cases.make_case(
            'negative_scope',
            {
                'citation': 'positive-citation',
                'conversation_id': 'conv-1',
                'source_type': 'message',
                'content': '价格预算推进阻力',
            },
            '价格 预算 推进阻力 之外线索',
            query_type='negative_excluded_citation',
            negative_excluded_citations=['distractor-citation'],
            expect_citation=False,
        )
        self.assertNotIn('expected_any_citation', case['oracle'])
        self.assertEqual(case['oracle']['negative_excluded_citations'], ['distractor-citation'])
        self.assertFalse(case['context'])

    def test_generated_pack_contains_true_no_result_calibration_negatives(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            cases, _inventory = generate_real_eval_cases.generate_cases(vault, max_cases=24)

        negatives = [
            case
            for case in cases
            if (case.get('oracle') or {}).get('negative_no_results') is True
        ]
        self.assertTrue(negatives)
        self.assertTrue(all(case.get('filters', {}).get('account_id') for case in negatives))
        self.assertTrue(all(case.get('filters', {}).get('conversation_id') for case in negatives))
        self.assertTrue(all(not (case.get('oracle') or {}).get('expected_any_citation') for case in negatives))


if __name__ == '__main__':
    unittest.main()
