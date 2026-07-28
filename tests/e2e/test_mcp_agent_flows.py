from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from trove_client import TroveClient
from trove_mcp.v1_server import create_server

from tests.e2e.runtime_harness import RuntimeHarness, citations


class MCPAgentFlowTests(unittest.TestCase):
    @staticmethod
    def call(server, name: str, arguments: dict):
        return asyncio.run(server._tool_manager.call_tool(name, arguments))

    def test_mcp_recall_search_context_media_and_operations_use_one_daemon(self):
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(Path(directory) / 'vault') as runtime:
            class BoundClient(TroveClient):
                def __init__(self, _identity, *, role='sdk'):
                    super().__init__(runtime.identity, role=role, autostart=None)

            server = create_server(
                pack='operations', vault=runtime.config.root,
                client_factory=BoundClient,
            )
            try:
                recall = self.call(server, 'trove_recall', {
                    'account_id': 'acct-work',
                    'conversation_id': 'conv-sales-review',
                    'limit': 20,
                })
                self.assertEqual(recall['coverage']['state'], 'complete')
                self.assertTrue(citations(recall['data']))

                search = self.call(server, 'trove_search', {
                    'query': '客户卡在哪', 'account_id': 'acct-work',
                    'semantic': 'off', 'limit': 5,
                })
                context = self.call(server, 'trove_context', {
                    'citation': citations(search['data'])[0], 'before': 2, 'after': 2,
                })
                self.assertTrue(citations(context['data']))

                files = self.call(server, 'trove_files_list', {
                    'account_id': 'acct-work',
                    'conversation_id': 'conv-sales-review',
                    'media_types': ['image'], 'limit': 20,
                })
                selected = next(
                    item['citation'] for item in files['data']['files']
                    if item['asset_id'] == 'asset-agent-flow-image'
                )
                fetched = self.call(server, 'trove_media_fetch', {
                    'citation': selected, 'allow_remote': False,
                })
                self.assertTrue(fetched['data']['evidence_ok'])
                self.assertEqual(fetched['data']['status'], 'available')

                enrichment = self.call(server, 'trove_media_enrich', {
                    'citation': selected, 'kind': 'annotate',
                })
                status = self.call(server, 'trove_operation_status', {
                    'operation_id': enrichment['data']['operation']['operation_id'],
                })
                self.assertEqual(status['data']['operation']['state'], 'pending')

                added = self.call(server, 'trove_observe_add', {
                    'target': '示例教育', 'text': 'synthetic MCP note',
                    'idempotency_key': 'fixture-mcp-observation',
                })
                listed = self.call(server, 'trove_observe_list', {
                    'target': '示例教育', 'limit': 10,
                })
                self.assertEqual(listed['data']['count'], 1)
                self.assertEqual(
                    listed['data']['observations'][0]['observation_id'],
                    added['data']['observation']['observation_id'],
                )
            finally:
                server._trove_runtime.close()

    def test_operations_pack_has_request_status_but_no_decision_tool(self):
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(Path(directory) / 'vault') as runtime:
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
                self.assertFalse(any('approve' in name or 'reject' in name or 'decision' in name for name in names))
            finally:
                server._trove_runtime.close()


if __name__ == '__main__':
    unittest.main()
