from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from trove_cli.main import main
from trove_core.wechat.indexer import index_fixture_vault


class CliEvalMatrixTests(unittest.TestCase):
    def test_cli_eval_matrix_writes_redacted_report(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            out = vault / 'proof' / 'retrieval-eval' / 'redacted' / 'eval-matrix.redacted.json'
            stdout = StringIO()
            with redirect_stdout(stdout):
                rc = main(['--vault', str(vault), 'eval-matrix', '--cases', 'tests/golden/retrieval_core.jsonl', '--modes', 'exact,hybrid-weighted', '--k', '3', '--out', str(out), '--json'])
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertIn('hybrid-weighted', data['modes'])
            combined = stdout.getvalue() + out.read_text(encoding='utf-8')
            self.assertNotIn('价格太高', combined)
            self.assertNotIn('trove://wechat/acct-work', combined)

    def test_cli_eval_matrix_controls(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            out = vault / 'proof' / 'retrieval-eval' / 'redacted' / 'controlled.redacted.json'
            partial = vault / 'proof' / 'retrieval-eval' / 'redacted' / 'controlled.partial.redacted.json'
            stdout = StringIO()
            with redirect_stdout(stdout):
                rc = main([
                    '--vault', str(vault),
                    'eval-matrix',
                    '--cases', 'tests/golden/retrieval_core.jsonl',
                    '--modes', 'hybrid-rrf,feature-rerank,local-reranker,cloud-reranker',
                    '--max-cases', '2',
                    '--sample-seed', '7',
                    '--partial-out', str(partial),
                    '--out', str(out),
                    '--json',
                ])
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(data['case_count'], 2)
            self.assertTrue(data['complete'])
            self.assertIn('local-reranker', data['modes'])
            self.assertIn('cloud-reranker', data['modes'])
            self.assertIn('cloud_reranker_requires_exact_approval', data['modes']['cloud-reranker']['notes'])
            self.assertTrue(partial.exists())


if __name__ == '__main__':
    unittest.main()
