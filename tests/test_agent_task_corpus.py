from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.measure_agent_surface import PROPOSED_STANDARD_TOOLS, load_task_corpus


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'tests' / 'golden' / 'agent_task_corpus.jsonl'


class AgentTaskCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_task_corpus(CORPUS)

    def test_every_standard_tool_has_a_representative_task(self):
        self.assertEqual({case['capability'] for case in self.cases}, PROPOSED_STANDARD_TOOLS)

    def test_cases_are_synthetic_bounded_and_assert_cited_outcomes(self):
        ids: set[str] = set()
        for case in self.cases:
            self.assertNotIn(case['id'], ids)
            ids.add(case['id'])
            self.assertLessEqual(case['max_calls'], 4)
            self.assertTrue(case['legacy_adapter'])
            self.assertTrue(case['expected']['outcome'])
            if case['capability'] not in {'trove_capabilities', 'trove_operation_status', 'trove_operation_continue'}:
                self.assertTrue(case['expected']['requires_citation'])
            encoded = json.dumps(case, ensure_ascii=False)
            self.assertNotIn('/Users/', encoded)
            self.assertNotIn('wxid_', encoded)

    def test_corpus_contains_a_multi_account_task(self):
        multi = [case for case in self.cases if case.get('requires_multi_account')]
        self.assertTrue(multi)
        self.assertTrue(any('account_id' in json.dumps(case['input']) for case in multi))


if __name__ == '__main__':
    unittest.main()
