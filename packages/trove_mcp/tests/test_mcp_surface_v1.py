from __future__ import annotations

import tempfile
import unittest

from trove_mcp.catalog_adapter import UNTRUSTED_RULE, descriptors_for_pack
from trove_mcp.packs import MCPPackError, tool_names
from trove_mcp.v1_server import SERVER_NAME, create_server
from trove_protocol.capabilities import STANDARD_MCP_TOOLS, capabilities_for_pack


class _Client:
    def __init__(self, *_args, **_kwargs):
        pass

    def call(self, *_args, **_kwargs):
        return {'protocol': 'trove/1', 'request_id': 'test', 'ok': True, 'data': {}}

    def close(self):
        pass


class MCPSurfaceV1Tests(unittest.TestCase):
    def test_server_identity_and_standard_pack_are_exact(self):
        self.assertEqual(SERVER_NAME, 'trove')
        self.assertEqual(tool_names('standard'), STANDARD_MCP_TOOLS)
        self.assertLessEqual(len(STANDARD_MCP_TOOLS), 12)

    def test_packs_are_cumulative_reviewed_catalog_sets(self):
        standard = tool_names('standard')
        operations = tool_names('operations')
        admin = tool_names('admin')
        self.assertTrue(standard < operations < admin)
        for pack in ('standard', 'operations', 'admin'):
            self.assertEqual(tool_names(pack), {spec.mcp_name for spec in capabilities_for_pack(pack)})

    def test_tools_list_uses_exact_catalog_schema_and_untrusted_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=_Client)
        by_name = {tool.name: tool for tool in server._tool_manager.list_tools()}
        for item in descriptors_for_pack('standard'):
            self.assertEqual(by_name[item.name].parameters, item.input_schema)
            self.assertEqual(by_name[item.name].description, item.description)
            if item.name not in {'trove_capabilities', 'trove_operation_status', 'trove_operation_continue'}:
                self.assertIn(UNTRUSTED_RULE, item.description)
        server._trove_runtime.close()

    def test_legacy_and_approval_decision_tools_are_absent(self):
        names = tool_names('admin')
        forbidden = {
            'trove_chat_recall', 'trove_customer_profile', 'trove_list_contacts',
            'trove_profile_enrichment_claim', 'trove_approval_decide',
            'trove_approval_approve', 'trove_approval_reject',
        }
        self.assertFalse(names & forbidden)
        reply = {name for name in names if name.startswith('trove_reply_')}
        self.assertEqual(reply, {
            'trove_reply_status',
            'trove_reply_reviews',
            'trove_reply_activity',
        })
        self.assertFalse(any(
            token in name
            for name in reply
            for token in ('approve', 'reject', 'send', 'arm', 'disarm')
        ))

    def test_invalid_pack_fails_before_server_start(self):
        with self.assertRaises(MCPPackError):
            tool_names('full')


if __name__ == '__main__':
    unittest.main()
