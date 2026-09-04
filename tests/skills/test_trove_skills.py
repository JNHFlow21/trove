from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from trove_protocol.capabilities import CATALOG_BY_ID, catalog_snapshot


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / 'skills'
EXPECTED = {
    'trove-recall', 'trove-group-summary', 'trove-search', 'trove-profile',
    'trove-file-recall', 'trove-media-enrichment', 'trove-moments',
    'trove-triage',
}
FORBIDDEN = (
    '/Users/', 'python -m', 'sqlite', 'message_fts', 'trove_chat_recall',
    'trove_customer_profile', 'profile_enrichment_claim', 'profile_enrichment_heartbeat',
    'profile_enrichment_finalize', 'VOLCENGINE_ASR_API_KEY', 'TROVE_WECHAT_KEY_STORE',
)


class TroveSkillsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((SKILLS / 'manifest.json').read_text())
        cls.text = {
            name: (SKILLS / name / 'SKILL.md').read_text(encoding='utf-8')
            for name in EXPECTED
        }

    def test_manifest_is_exactly_eight_version_bound_catalog_skills(self):
        snapshot = catalog_snapshot()
        self.assertEqual(self.manifest['protocol'], 'trove/1')
        self.assertEqual(self.manifest['catalog_sha256'], snapshot['catalog_sha256'])
        self.assertEqual({item['name'] for item in self.manifest['skills']}, EXPECTED)
        for item in self.manifest['skills']:
            self.assertEqual(item['version'], '1.0.0')
            self.assertTrue(item['mcp_first'])
            self.assertTrue(item['cli_fallback'])
            for capability in item['capabilities']:
                self.assertIn(capability, CATALOG_BY_ID)

    def test_skills_are_short_and_contain_no_legacy_or_source_checkout_detail(self):
        for name, text in self.text.items():
            with self.subTest(name=name):
                self.assertLessEqual(len(text.splitlines()), 70)
                self.assertLessEqual(len(text.encode('utf-8')), 5000)
                self.assertTrue(text.startswith('---\nname: ' + name + '\n'))
                for forbidden in FORBIDDEN:
                    self.assertNotIn(forbidden.lower(), text.lower())
                self.assertNotRegex(text, r'\b(accounts|messages|conversations)\s+table\b')

    def test_every_skill_is_mcp_first_cli_fallback_and_untrusted_safe(self):
        for name, text in self.text.items():
            lowered = text.lower()
            with self.subTest(name=name):
                self.assertLess(lowered.index('mcp'), lowered.index('cli'))
                self.assertIn('untrusted evidence', lowered)
                self.assertIn('never decide an approval', lowered)

    def test_recall_search_and_summary_call_budgets_are_explicit(self):
        recall = self.text['trove-recall'].lower()
        search = self.text['trove-search'].lower()
        summary = self.text['trove-group-summary'].lower()
        self.assertIn('call once', recall)
        self.assertIn('no more than these two calls', search)
        self.assertIn('until coverage is `complete`', summary)
        self.assertIn('do not give a complete conclusion from a partial page', summary)

    def test_multi_account_and_typed_stop_paths_are_bounded(self):
        joined = '\n'.join(self.text.values()).lower()
        for phrase in ('ambiguous_target', 'account_id', 'no_results', 'provider unavailable', 'approval'):
            self.assertIn(phrase, joined)
        self.assertNotIn('try each account', joined)
        self.assertNotIn('loop forever', joined)

    def test_operation_skills_use_only_opaque_public_continuation(self):
        for name in ('trove-profile', 'trove-media-enrichment'):
            lowered = self.text[name].lower()
            self.assertIn('opaque token', lowered)
            self.assertIn('trove_operation_continue', lowered)
            for internal in ('heartbeat', 'finalize', 'claim'):
                self.assertNotIn('trove_' + internal, lowered)


if __name__ == '__main__':
    unittest.main()
