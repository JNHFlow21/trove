from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from trove_core.wechat.indexer import index_fixture_vault

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'scripts' / 'trove-python'


class RealEvalCaseGenerationTests(unittest.TestCase):
    def test_generator_writes_private_and_redacted_under_vault_proof_only(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            proc = subprocess.run([str(PYTHON), 'scripts/generate_real_eval_cases.py', '--vault', str(vault), '--max-cases', '24', '--min-cases', '8'], cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads(proc.stdout)
            self.assertTrue(summary['ok'])
            self.assertGreaterEqual(summary['case_count'], 8)
            self.assertEqual(summary['case_quality']['literal_substring_rate'], 0.0)
            self.assertLessEqual(summary['case_quality']['avg_word_overlap_ratio'], 0.5)
            self.assertFalse(summary['raw_queries_printed'])
            self.assertFalse(summary['private_paths_printed'])
            private_path = vault / 'proof' / 'retrieval-eval' / 'private' / 'cases.local.jsonl'
            redacted_path = vault / 'proof' / 'retrieval-eval' / 'redacted' / 'cases.redacted.json'
            self.assertTrue(private_path.exists())
            self.assertTrue(redacted_path.exists())
            private_text = private_path.read_text(encoding='utf-8')
            redacted_text = redacted_path.read_text(encoding='utf-8')
            self.assertIn('query', private_text)
            # Fixture text may exist in the private file, but not stdout or redacted output.
            self.assertNotIn('价格太高', proc.stdout)
            self.assertNotIn('价格太高', redacted_text)
            self.assertNotIn(str(vault), proc.stdout)
            self.assertNotIn(str(vault), redacted_text)

    def test_generator_refuses_source_repo_output(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            proc = subprocess.run([str(PYTHON), 'scripts/generate_real_eval_cases.py', '--vault', str(vault), '--out-root', str(ROOT / 'tmp-real-eval-output'), '--max-cases', '2'], cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('refusing to write', proc.stderr + proc.stdout)


if __name__ == '__main__':
    unittest.main()
