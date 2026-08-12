#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'scripts' / 'trove-python'
AGENT_RUNTIME_OUTPUT = Path(tempfile.gettempdir()) / f'trove-agent-runtime-budgets-{os.getpid()}.json'


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]


CHECKS = {
    'unit': (
        Check('protocol', (str(PYTHON), '-m', 'unittest', 'discover', '-s', 'packages/trove_protocol/tests')),
        Check('daemon-client', (str(PYTHON), '-m', 'unittest', 'discover', '-s', 'packages/trove_daemon/tests')),
        Check('reply-core', (
            str(PYTHON), '-m', 'unittest',
            'packages.trove_core.tests.test_reply_domain',
            'packages.trove_core.tests.test_reply_store',
            'packages.trove_core.tests.test_reply_rounds',
            'packages.trove_core.tests.test_reply_context',
            'packages.trove_core.tests.test_reply_generator',
            'packages.trove_core.tests.test_reply_media',
            'packages.trove_core.tests.test_reply_migration',
            'packages.trove_core.tests.test_reply_service',
        )),
        Check('provider-wechat', (
            str(PYTHON), '-m', 'unittest', 'discover',
            '-s', 'packages/trove_provider_wechat/tests', '-t', '.',
        )),
    ),
    'contract': (
        Check('cli-v1', (str(PYTHON), '-m', 'unittest', 'packages.trove_cli.tests.test_cli_surface_v1', 'packages.trove_cli.tests.test_cli_daemon_contract')),
        Check('mcp-v1', (str(PYTHON), '-m', 'unittest', 'packages.trove_mcp.tests.test_mcp_surface_v1', 'packages.trove_mcp.tests.test_mcp_daemon_contract', 'packages.trove_mcp.tests.test_mcp_schema_budget')),
        Check('skills-v1', (str(PYTHON), '-m', 'unittest', 'tests.skills.test_trove_skills')),
        Check('docs-v1', (str(PYTHON), '-m', 'unittest', 'tests.test_documentation_contract')),
        Check('public-surface', (str(PYTHON), '-m', 'unittest', 'tests.test_public_surface_lint', 'tests.test_installed_consumer_migration', 'tests.e2e.test_no_web_api_surface')),
    ),
    'package': (
        Check('config', (str(PYTHON), '-m', 'unittest', 'packages.trove_core.tests.test_product_config', 'tests.test_bootstrap_runtime_extras')),
        Check('privacy', (str(PYTHON), 'scripts/privacy_scan.py', '.')),
        Check('packaging-smoke', (str(PYTHON), '-m', 'unittest', 'tests.e2e.test_packaging_smoke')),
    ),
    'e2e': (
        Check('e2e', (str(PYTHON), '-m', 'unittest', 'discover', '-s', 'tests/e2e')),
        Check('agent-product-acceptance', (str(PYTHON), 'scripts/run_agent_product_acceptance.py')),
    ),
    'perf': (
        Check('agent-runtime', (
            str(PYTHON), 'scripts/benchmark_agent_runtime.py',
            '--rounds', '3', '--warmups', '1', '--out', str(AGENT_RUNTIME_OUTPUT),
        )),
    ),
}
RELEASE_ORDER = ('unit', 'contract', 'package', 'e2e', 'perf')


def selected_checks(tier: str) -> tuple[Check, ...]:
    if tier == 'release':
        return tuple(check for name in RELEASE_ORDER for check in CHECKS[name])
    return CHECKS[tier]


def run(tier: str, *, list_only: bool = False) -> int:
    for check in selected_checks(tier):
        if list_only:
            print(check.name)
            continue
        print(f'==> {check.name}', flush=True)
        completed = subprocess.run(check.command, cwd=ROOT)
        if completed.returncode:
            return completed.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='check.py')
    parser.add_argument('tier', choices=(*CHECKS, 'release'))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    return run(args.tier, list_only=args.list)


if __name__ == '__main__':
    raise SystemExit(main())
