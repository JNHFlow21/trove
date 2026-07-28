from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'scripts' / 'trove-python'
ENV = {**os.environ, 'TROVE_DISABLE_AUTO_MODEL_DISCOVERY': '1'}


def _citations(value):
    if isinstance(value, str) and value.startswith('trove://'):
        return [value]
    if isinstance(value, dict):
        return [citation for item in value.values() for citation in _citations(item)]
    if isinstance(value, list):
        return [citation for item in value for citation in _citations(item)]
    return []


class CliFixtureE2ETests(unittest.TestCase):
    def run_json(self, args):
        completed = subprocess.run(
            args, cwd=ROOT, env=ENV, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        return json.loads(completed.stdout)

    def test_v1_cli_fixture_search_and_context_over_daemon(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_json([str(PYTHON), 'scripts/generate_fixture_vault.py', '--vault', directory, '--reset'])
            base = [str(PYTHON), '-m', 'trove_cli.v1_main', '--vault', directory]
            try:
                self.assertTrue(self.run_json([*base, 'start'])['ok'])
                search = self.run_json([*base, 'search', '--query', '客户卡在哪', '--limit', '10'])
                self.assertTrue(search['ok'])
                citations = _citations(search.get('data'))
                self.assertTrue(citations)
                context = self.run_json([*base, 'context', '--citation', citations[0]])
                self.assertTrue(context['ok'])
                self.assertTrue(_citations(context.get('data')))
            finally:
                stopped = self.run_json([*base, 'stop'])
                self.assertTrue(stopped['ok'])


if __name__ == '__main__':
    unittest.main()
