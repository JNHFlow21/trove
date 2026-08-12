from __future__ import annotations

import unittest
import contextlib
import io
import json
from pathlib import Path

from scripts.benchmark_agent_runtime import (
    REQUIRED_MEASUREMENTS,
    evaluate_absolute_budgets,
    evaluate_relative_regressions,
    main,
    validate_benchmark_artifact,
)
from scripts.measure_agent_surface import estimate_json_tokens
from scripts.generate_fixture_vault import redacted_fixture_metadata
from scripts.release_gate_contracts import agent_runtime_budget_contract_valid


class AgentRuntimeBenchmarkContractTests(unittest.TestCase):
    def test_cli_help_renders_on_supported_python_versions(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('10% p95', output.getvalue())
        self.assertIn('regression gate', output.getvalue())

    def _artifact(self) -> dict:
        measurements = {
            name: {
                'cold': {'samples_ms': [10.0, 20.0], 'p50_ms': 15.0, 'p95_ms': 19.5},
                'warm': {
                    'warmup_samples_ms': [100.0],
                    'samples_ms': [1.0, 2.0],
                    'p50_ms': 1.5,
                    'p95_ms': 1.95,
                },
            }
            for name in REQUIRED_MEASUREMENTS
        }
        return {
            'schema_version': 1,
            'artifact_type': 'agent_runtime_baseline_redacted',
            'git_sha': 'a' * 40,
            'fixture_sha256': 'b' * 64,
            'seed': 20260621,
            'rounds': 2,
            'hardware': {
                'system': 'Darwin',
                'machine': 'arm64',
                'cpu_count': 8,
                'python': '3.11.0',
            },
            'measurements': measurements,
            'resources': {
                'daemon_idle_rss_mib': 0.0,
                'mcp_idle_rss_mib': 50.0,
                'idle_cpu_percent': 0.0,
                'store_builds': 1,
                'engine_builds': 1,
            },
            'surface': {'tools_list_bytes': 1000, 'tools_list_estimated_tokens': 250},
            'task_calls': {'sample_count': 1, 'p50': 1.0, 'p95': 1.0},
            'privacy': {
                'content_included': False,
                'contacts_included': False,
                'citations_included': False,
                'absolute_paths_included': False,
                'secret_values_included': False,
            },
        }

    def test_valid_artifact_requires_reproducibility_and_privacy_metadata(self):
        artifact = self._artifact()
        validate_benchmark_artifact(artifact)
        for field in ('git_sha', 'fixture_sha256', 'rounds', 'hardware'):
            with self.subTest(field=field):
                invalid = self._artifact()
                invalid.pop(field)
                with self.assertRaises(ValueError):
                    validate_benchmark_artifact(invalid)

    def test_warmups_are_separate_from_measured_samples(self):
        artifact = self._artifact()
        artifact['measurements']['exact_recall']['warm']['samples_ms'] = [100.0, 1.0, 2.0]
        with self.assertRaisesRegex(ValueError, 'percentile'):
            validate_benchmark_artifact(artifact)

    def test_privacy_flags_must_all_be_false(self):
        artifact = self._artifact()
        artifact['privacy']['contacts_included'] = True
        with self.assertRaisesRegex(ValueError, 'privacy'):
            validate_benchmark_artifact(artifact)

    def test_tools_list_token_estimator_has_a_frozen_golden_case(self):
        payload = {'tools': [{'name': 'trove_recall', 'description': 'Recall cited evidence.'}]}
        self.assertEqual(estimate_json_tokens(payload), 21)

    def test_absolute_budget_evaluation_rejects_a_measured_overage(self):
        artifact = self._artifact()
        self.assertTrue(evaluate_absolute_budgets(artifact)['ok'])
        artifact['resources']['daemon_idle_rss_mib'] = 97.0
        evaluation = evaluate_absolute_budgets(artifact)
        self.assertFalse(evaluation['ok'])
        self.assertEqual(evaluation['failures'][0]['metric'], 'daemon_idle_rss_mib_max')

    def test_relative_gate_requires_comparable_runs_and_rejects_ten_percent_regression(self):
        baseline = self._artifact()
        current = self._artifact()
        current['measurements']['exact_recall']['warm'].update({
            'samples_ms': [2.0, 3.0], 'p50_ms': 2.5, 'p95_ms': 2.95,
        })
        evaluation = evaluate_relative_regressions(current, baseline)
        self.assertFalse(evaluation['ok'])
        self.assertTrue(evaluation['comparable'])
        incompatible = self._artifact()
        incompatible['rounds'] = 3
        self.assertFalse(evaluate_relative_regressions(incompatible, baseline)['comparable'])

    def test_fixture_metadata_is_deterministic_and_path_free(self):
        report_a = {'vault': '/private/a', 'sqlite': '/private/a/index.db', 'changed': 2, 'chunks': {'chunks': 2}, 'counts': {'messages': 2}}
        report_b = {'vault': '/different/b', 'sqlite': '/different/b/index.db', 'changed': 2, 'chunks': {'chunks': 2}, 'counts': {'messages': 2}}
        first = redacted_fixture_metadata(report_a, seed=7)
        second = redacted_fixture_metadata(report_b, seed=7)
        self.assertEqual(first, second)
        self.assertNotIn('/private', json.dumps(first))

    def test_checked_in_budget_artifact_satisfies_the_release_contract(self):
        payload = json.loads(Path('docs/perf/agent-runtime-budgets.json').read_text(encoding='utf-8'))
        self.assertTrue(agent_runtime_budget_contract_valid(payload))


if __name__ == '__main__':
    unittest.main()
