from __future__ import annotations

import unittest

from trove_core.search.eval_schema import (
    RedactionError,
    case_pack_quality_stats,
    expected_citations,
    query_expected_quality,
    redact_case,
    redact_case_pack,
    validate_case,
    validate_redacted_artifact,
)


class EvalSchemaTests(unittest.TestCase):
    def test_private_case_supports_real_query_and_oracles(self):
        case = validate_case({
            'schema_version': 2,
            'case_id': 'private-1',
            'category': 'blocker_diagnosis',
            'query': '真实查询可以只留在 private case pack',
            'oracle': {'expected_any_citation': ['trove://wechat/acct/conv/message_0/1']},
            'private': {'bounded_context_note': 'private local note'},
        })
        self.assertEqual(case['category'], 'blocker_diagnosis')
        self.assertEqual(expected_citations(case), ['trove://wechat/acct/conv/message_0/1'])

    def test_redact_case_hashes_sensitive_fields(self):
        redacted = redact_case({
            'schema_version': 2,
            'case_id': 'private-1',
            'category': 'exact_sparse',
            'query': 'local private query',
            'oracle': {'expected_any_citation': ['trove://wechat/acct/conv/message_0/1']},
        })
        self.assertNotIn('query', redacted)
        self.assertIn('query_hash', redacted)
        self.assertEqual(redacted['query_length'], len('local private query'))
        self.assertEqual(len(redacted['expected_citation_hashes']), 1)

    def test_redacted_guard_rejects_raw_query_path_and_token(self):
        with self.assertRaises(RedactionError):
            validate_redacted_artifact({'query': 'raw private query'})
        with self.assertRaises(RedactionError):
            validate_redacted_artifact({'safe': '/Users/' + 'somebody/private/vault/file.txt'})
        with self.assertRaises(RedactionError):
            validate_redacted_artifact({'safe': 'Bearer trove-local-abcdef1234567890'})

    def test_redact_case_pack_summary_is_safe(self):
        payload = redact_case_pack([{
            'schema_version': 2,
            'case_id': 'case-1',
            'category': 'semantic_paraphrase',
            'query': 'private semantic query',
            'source_family': 'message',
            'oracle': {'expected_any_citation': ['citation-1']},
        }], created_at='2026-06-22T00:00:00Z')
        validate_redacted_artifact(payload)
        self.assertEqual(payload['stats']['cases'], 1)
        self.assertIn('case_quality', payload)
        text = str(payload)
        self.assertNotIn('private semantic query', text)
        self.assertNotIn('citation-1', text)

    def test_case_quality_detects_literal_substring_and_overlap(self):
        copied = query_expected_quality('价格太高', '我们看了方案，但是价格太高。')
        self.assertTrue(copied['literal_substring'])
        rewritten = query_expected_quality('预算 推进阻力', '价格太高，预算审批也没过。')
        self.assertFalse(rewritten['literal_substring'])
        self.assertLessEqual(rewritten['word_overlap_ratio'], 0.5)
        stats = case_pack_quality_stats([{
            'case_id': 'case-1',
            'category': 'semantic_paraphrase',
            'query_type': 'task_question',
            'query': '预算 推进阻力',
            'private': {'bounded_context_note': '价格太高，预算审批也没过。'},
            'oracle': {'expected_any_citation': ['citation-1']},
        }])
        self.assertEqual(stats['literal_substring_rate'], 0.0)
        self.assertLessEqual(stats['avg_word_overlap_ratio'], 0.5)


if __name__ == '__main__':
    unittest.main()
