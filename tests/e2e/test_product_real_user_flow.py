from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from trove_daemon.lifecycle import RuntimeIdentity


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'scripts' / 'trove-python'
ENV = {**os.environ, 'TROVE_DISABLE_AUTO_MODEL_DISCOVERY': '1'}


class ProductRealUserFlowTests(unittest.TestCase):
    def run_json(self, argv, *, allowed=(0,)):
        completed = subprocess.run(
            argv, cwd=ROOT, env=ENV, text=True, capture_output=True, timeout=60,
        )
        self.assertIn(completed.returncode, allowed, completed.stderr or completed.stdout)
        return json.loads(completed.stdout)

    def test_fixture_user_can_start_resolve_recall_and_stop_v1_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_json([str(PYTHON), 'scripts/generate_fixture_vault.py', '--vault', directory, '--reset'])
            base = [str(PYTHON), '-m', 'trove_cli.v1_main', '--vault', directory]
            try:
                start = self.run_json([*base, 'start'])
                self.assertTrue(start['ok'])
                status = self.run_json([*base, 'status'])
                self.assertTrue(status['ok'])
                self.assertTrue(status['data']['running'])
                metadata = json.loads(
                    RuntimeIdentity.for_vault(directory).pid_path.read_text(encoding='utf-8')
                )
                listeners = subprocess.run(
                    ['lsof', '-nP', '-a', '-p', str(metadata['pid']), '-iTCP', '-sTCP:LISTEN'],
                    cwd=ROOT, text=True, capture_output=True, timeout=10,
                )
                self.assertIn(listeners.returncode, (0, 1), listeners.stderr)
                self.assertEqual(listeners.stdout.strip(), '')
                accounts = self.run_json([*base, 'accounts', '--kind', 'account'])
                self.assertTrue(accounts['ok'])
                recall = self.run_json([
                    *base, 'recall', '--conversation-id', 'conv-sales-review', '--limit', '5',
                ])
                self.assertTrue(recall['ok'])
                self.assertIn('coverage', recall)
            finally:
                stop = self.run_json([*base, 'stop'])
                self.assertTrue(stop['ok'])

    def test_legacy_route_is_unknown_without_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_json(
                [str(PYTHON), '-m', 'trove_cli.v1_main', '--vault', directory, 'chat-recall'],
                allowed=(2,),
            )
            self.assertFalse(result['ok'])
            self.assertEqual(result['error']['code'], 'invalid_input')

    def test_six_skill_product_acceptance_passes_for_cli_and_mcp(self):
        completed = subprocess.run(
            [str(PYTHON), 'scripts/run_agent_product_acceptance.py'],
            cwd=ROOT, env=ENV, text=True, capture_output=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        report = json.loads(completed.stdout)
        self.assertTrue(report['ok'])
        self.assertEqual(report['artifact_type'], 'agent_product_acceptance_redacted')
        self.assertEqual(report['summary']['clients'], 2)
        self.assertEqual(report['summary']['tasks'], 12)
        self.assertEqual(report['summary']['tasks_succeeded'], 12)
        self.assertEqual(report['summary']['wrong_tool_calls'], 0)
        self.assertEqual(report['summary']['operator_interventions'], 0)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn('trove://', serialized)
        self.assertNotIn('/Users/', serialized)

    def test_real_vault_acceptance_reads_legacy_citation_and_emits_metrics_only(self):
        def citations(value):
            if isinstance(value, str) and value.startswith('trove://'):
                return [value]
            if isinstance(value, dict):
                return [item for nested in value.values() for item in citations(nested)]
            if isinstance(value, list):
                return [item for nested in value for item in citations(nested)]
            return []

        with tempfile.TemporaryDirectory() as directory:
            self.run_json([str(PYTHON), 'scripts/generate_fixture_vault.py', '--vault', directory, '--reset'])
            base = [str(PYTHON), '-m', 'trove_cli.v1_main', '--vault', directory]
            try:
                self.run_json([*base, 'start'])
                recall = self.run_json([
                    *base, 'recall', '--conversation-id', 'conv-sales-review', '--limit', '20',
                ])
                citation = citations(recall['data'])[0]
                private = {
                    'recall': {'conversation_id': 'conv-sales-review', 'limit': 20},
                    'search': {'query': '客户卡在哪', 'semantic': 'off', 'limit': 5},
                    'legacy_citations': [citation],
                    'minimums': {
                        'accounts': {'result_count_min': 2},
                        'recall': {'result_count_min': 1, 'citation_count_min': 1, 'warm_latency_ms_max': 1000},
                        'search': {'result_count_min': 1, 'citation_count_min': 1, 'warm_latency_ms_max': 1000},
                    },
                }
                read_fd, write_fd = os.pipe()
                try:
                    os.write(write_fd, json.dumps(private, ensure_ascii=False).encode('utf-8'))
                finally:
                    os.close(write_fd)
                try:
                    completed = subprocess.run(
                        [
                            str(PYTHON), 'scripts/run_real_vault_acceptance.py',
                            '--vault', directory, '--input-fd', str(read_fd),
                        ],
                        cwd=ROOT, env=ENV, text=True, capture_output=True,
                        pass_fds=(read_fd,), timeout=60,
                    )
                finally:
                    os.close(read_fd)
                self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
                report = json.loads(completed.stdout)
                self.assertTrue(report['ok'])
                self.assertTrue(report['legacy_citations']['ok'])
                self.assertTrue(report['quality']['ok'])
                self.assertTrue(report['baseline']['ok'])
                serialized = json.dumps(report, ensure_ascii=False)
                self.assertNotIn(citation, serialized)
                self.assertNotIn('客户卡在哪', serialized)
                self.assertNotIn(directory, serialized)
            finally:
                self.run_json([*base, 'stop'])


if __name__ == '__main__':
    unittest.main()
