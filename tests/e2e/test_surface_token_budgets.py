from __future__ import annotations

import json
from pathlib import Path
import unittest

import tomllib

from trove_core.application.handlers.base import HandlerOutcome
from trove_mcp.catalog_adapter import schema_size
from trove_protocol.envelope import Envelope
from trove_protocol.errors import ErrorDetail

from tests.e2e.runtime_harness import RuntimeHarness


ROOT = Path(__file__).resolve().parents[2]


def encoded_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


class SurfaceTokenBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = json.loads(
            (ROOT / 'docs/perf/agent-runtime-budgets.json').read_text(encoding='utf-8')
        )['targets']

    def test_standard_tools_list_and_all_skill_files_fit_reviewed_budgets(self):
        measured = schema_size('standard')
        self.assertLessEqual(measured['bytes'], self.targets['standard_tools_list_bytes_max'])
        self.assertLessEqual(
            measured['estimated_tokens'], self.targets['standard_tools_list_tokens_max'],
        )
        skill_files = sorted((ROOT / 'skills').glob('*/SKILL.md'))
        self.assertEqual(len(skill_files), 8)
        self.assertTrue(all(path.stat().st_size <= 5_000 for path in skill_files))

    def test_compact_success_and_error_envelopes_fit_soft_budget(self):
        success = Envelope.success(
            {'items': [{'citation': {'uri': 'trove://fixture/a/c/m/1'}}]},
            request_id='budget-success',
            page={'has_more': False}, coverage={'state': 'complete'},
        ).to_dict()
        failure = Envelope.failure(
            ErrorDetail(
                'capability_unavailable', retryable=False,
                details={'capability': 'trove.media_enrich'},
                message='Capability is unavailable.',
            ),
            request_id='budget-failure',
            next={'capability': 'trove.provider_status', 'action': 'inspect_provider'},
        ).to_dict()
        self.assertLess(encoded_size(success), self.targets['compact_response_soft_bytes_max'])
        self.assertLess(encoded_size(failure), self.targets['compact_response_soft_bytes_max'])
        self.assertLess(encoded_size(success), self.targets['response_hard_bytes_max'])
        self.assertLess(encoded_size(failure), self.targets['response_hard_bytes_max'])

    def test_dispatcher_replaces_over_budget_evidence_with_small_typed_error(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(
            Path(directory) / 'vault', with_media=False,
        ) as runtime:
            runtime.dispatcher.handlers['trove.search'] = lambda _ctx, _payload, _request: (
                HandlerOutcome.success({'results': [{'snippet': 'x' * 20_000}]})
            )
            result = runtime.dispatcher.dispatch(
                'trove.search', {'query': 'fixture', 'semantic': 'off'},
                request_id='over-budget-search', response_budget=512,
            )
            self.assertFalse(result['ok'])
            self.assertEqual(result['error']['code'], 'response_too_large')
            self.assertLess(encoded_size(result), 512)
            self.assertNotIn('data', result)

    def test_distribution_exposes_exactly_three_public_executables(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(
            project['project']['scripts'],
            {
                'trove': 'trove_cli.main:main',
                'trove-mcp': 'trove_mcp.server:main',
                'troved': 'trove_daemon.main:main',
            },
        )


if __name__ == '__main__':
    unittest.main()
