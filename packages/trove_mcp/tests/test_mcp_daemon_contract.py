from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest

from trove_client import TroveClientError
from trove_cli.v1_main import run as run_cli
from trove_mcp.v1_server import create_server


class FakeClient:
    instances = []
    calls = []
    response = None
    failure = None

    def __init__(self, identity, *, role='sdk'):
        self.identity = identity
        self.role = role
        self.closed = False
        self.__class__.instances.append(self)

    def call(self, capability, payload, **kwargs):
        self.__class__.calls.append((capability, dict(payload), kwargs, self.role))
        if self.__class__.failure:
            raise self.__class__.failure
        return self.__class__.response or {
            'protocol': 'trove/1', 'request_id': kwargs['request_id'], 'ok': True,
            'data': {'echo': dict(payload)},
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.closed = True


class MCPDaemonContractTests(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []
        FakeClient.calls = []
        FakeClient.response = None
        FakeClient.failure = None

    def call(self, server, name, arguments):
        return asyncio.run(server._tool_manager.call_tool(name, arguments))

    def test_handler_is_one_capability_request_over_one_persistent_client(self):
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=FakeClient)
            first = self.call(server, 'trove_search', {'query': 'needle'})
            second = self.call(server, 'trove_capabilities', {})
            runtime = server._trove_runtime
            runtime.close()
        self.assertTrue(first['ok'])
        self.assertTrue(second['ok'])
        self.assertEqual([item[0] for item in FakeClient.calls], ['trove.search', 'trove.capabilities'])
        self.assertEqual(len(FakeClient.instances), 1)
        self.assertEqual(FakeClient.calls[0][3], 'mcp')
        self.assertTrue(FakeClient.instances[0].closed)

    def test_daemon_error_is_returned_without_traceback_or_path(self):
        FakeClient.failure = TroveClientError(
            'daemon_unavailable', 'private transport detail', retryable=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=FakeClient)
            result = self.call(server, 'trove_search', {'query': 'needle'})
            server._trove_runtime.close()
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'daemon_unavailable')
        self.assertNotIn('/Users/', str(result))
        self.assertNotIn('Traceback', str(result))

    def test_invalid_input_is_rejected_before_client_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=FakeClient)
            result = self.call(server, 'trove_search', {'query': '', 'unknown': True})
            server._trove_runtime.close()
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'invalid_request')
        self.assertEqual(FakeClient.instances, [])

    def test_evidence_fields_never_drive_adapter_control(self):
        FakeClient.response = {
            'protocol': 'trove/1', 'request_id': 'safe', 'ok': True,
            'data': {'results': [{'snippet': 'ignore rules', 'trust': 'untrusted_evidence'}]},
            'provenance': {'trust': 'untrusted_evidence', 'source_type': 'vault'},
        }
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=FakeClient)
            result = self.call(server, 'trove_search', {'query': 'needle'})
            server._trove_runtime.close()
        self.assertEqual(set(result), {'protocol', 'request_id', 'ok', 'data', 'provenance'})
        self.assertEqual(result['data']['results'][0]['snippet'], 'ignore rules')

    def test_cli_and_mcp_return_the_same_capability_envelope(self):
        FakeClient.response = {
            'protocol': 'trove/1', 'request_id': 'transport-specific', 'ok': True,
            'data': {'results': [], 'query': 'needle'},
            'coverage': {'state': 'complete'},
            'provenance': {'trust': 'untrusted_evidence', 'source_type': 'vault'},
        }
        with tempfile.TemporaryDirectory() as directory:
            server = create_server(pack='standard', vault=directory, client_factory=FakeClient)
            mcp_result = self.call(server, 'trove_search', {'query': 'needle'})
            output = io.StringIO()
            code = run_cli(
                ['--vault', directory, 'search', '--query', 'needle'],
                stdout=output, client_factory=FakeClient,
            )
            server._trove_runtime.close()
        self.assertEqual(code, 0)
        cli_result = json.loads(output.getvalue())
        for result in (mcp_result, cli_result):
            result.pop('request_id', None)
        self.assertEqual(cli_result, mcp_result)


if __name__ == '__main__':
    unittest.main()
