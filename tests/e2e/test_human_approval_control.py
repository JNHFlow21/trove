from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from trove_client import TroveClient
from trove_mcp.v1_server import create_server

from tests.e2e.runtime_harness import RuntimeHarness


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'scripts' / 'trove-python'


class HumanApprovalControlTests(unittest.TestCase):
    def test_agent_mcp_can_request_and_read_but_cannot_decide(self):
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(
            Path(directory) / 'vault', with_media=False,
        ) as runtime:
            class BoundClient(TroveClient):
                def __init__(self, _identity, *, role='sdk'):
                    super().__init__(runtime.identity, role=role, autostart=None)

            server = create_server(
                pack='operations', vault=runtime.config.root,
                client_factory=BoundClient,
            )
            try:
                names = {tool.name for tool in server._tool_manager.list_tools()}
                self.assertIn('trove_approval_request', names)
                self.assertIn('trove_approval_status', names)
                self.assertFalse(
                    any(term in name for name in names for term in ('approve', 'reject', 'decision'))
                )
                requested = asyncio.run(server._tool_manager.call_tool(
                    'trove_approval_request', {
                        'action': 'fixture_export',
                        'danger_class': 'local-file-export',
                        'payload': {'selection_id': 'synthetic-selection'},
                        'idempotency_key': 'fixture-mcp-approval-request',
                    },
                ))
                approval_id = requested['data']['approval']['approval_id']
                status = asyncio.run(server._tool_manager.call_tool(
                    'trove_approval_status', {'approval_id': approval_id},
                ))
                self.assertEqual(status['data']['approval']['status'], 'pending')
            finally:
                server._trove_runtime.close()

    def test_detached_noninteractive_cli_cannot_approve_its_own_request(self):
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(
            Path(directory) / 'vault', with_media=False,
        ) as runtime:
            requested = runtime.call('trove.approval_request', {
                'action': 'fixture_export',
                'danger_class': 'local-file-export',
                'payload': {'selection_id': 'synthetic-selection'},
                'idempotency_key': 'fixture-cli-approval-request',
            }, 'approval-request-detached-cli')
            approval_id = requested['data']['approval']['approval_id']
            completed = subprocess.run(
                [
                    str(PYTHON), '-m', 'trove_cli.v1_main',
                    '--vault', str(runtime.config.root),
                    'operator', 'approve', approval_id,
                ],
                cwd=ROOT,
                env={**os.environ, 'TROVE_DISABLE_AUTO_MODEL_DISCOVERY': '1'},
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                start_new_session=True,
                timeout=15,
            )
            self.assertEqual(completed.returncode, 7, completed.stderr or completed.stdout)
            response = json.loads(completed.stdout)
            self.assertEqual(response['error']['code'], 'operator_confirmation_required')
            status = runtime.call(
                'trove.approval_status', {'approval_id': approval_id},
                'approval-status-after-detached-cli',
            )
            self.assertEqual(status['data']['approval']['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
