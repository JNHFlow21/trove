from __future__ import annotations

import asyncio
import tempfile
import unittest

from trove_mcp.v1_server import create_server


class _NoConnect:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError('invalid bounded input reached daemon transport')


class BoundedMCPContractTests(unittest.TestCase):
    def test_catalog_bounds_are_rejected_before_daemon_io(self):
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=_NoConnect)
            cases = (
                ('trove_search', {'query': 'x', 'limit': 0}),
                ('trove_search', {'query': 'x', 'limit': 1001}),
                ('trove_context', {'citation': 'trove://source/a/c/s/1', 'before': 201}),
                ('trove_context', {'citation': 'trove://source/a/c/s/1', 'after': -1}),
            )
            for name, arguments in cases:
                with self.subTest(name=name, arguments=arguments):
                    result = asyncio.run(server._tool_manager.call_tool(name, arguments))
                    self.assertFalse(result['ok'])
                    self.assertEqual(result['error']['code'], 'invalid_request')
            server._trove_runtime.close()


if __name__ == '__main__':
    unittest.main()
