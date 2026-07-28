from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from trove_client import TroveClientError
from trove_cli.v1_main import run


class FakeClient:
    calls = []
    response = None
    failure = None

    def __init__(self, identity, *, role='sdk'):
        self.identity = identity
        self.role = role

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def call(self, capability, payload, **kwargs):
        self.__class__.calls.append((capability, dict(payload), kwargs, self.role))
        if self.__class__.failure:
            raise self.__class__.failure
        return self.__class__.response or {
            'protocol': 'trove/1', 'request_id': kwargs['request_id'], 'ok': True,
            'data': {'echo': dict(payload)},
        }


class CLIDaemonContractTests(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []
        FakeClient.response = None
        FakeClient.failure = None

    def invoke(self, vault: Path, *args: str, pretty: bool = False):
        stream = io.StringIO()
        argv = ['--vault', str(vault), '--request-id', 'cli-test-request-0001']
        if pretty:
            argv.append('--pretty')
        argv.extend(args)
        code = run(argv, stdout=stream, client_factory=FakeClient)
        return code, stream.getvalue(), json.loads(stream.getvalue())

    def test_business_command_is_one_catalog_request_and_one_compact_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            code, raw, result = self.invoke(Path(directory), 'search', '--query', 'needle', '--account', 'a')
        self.assertEqual(code, 0)
        self.assertEqual(raw.count('\n'), 1)
        self.assertEqual(result['data']['echo']['query'], 'needle')
        self.assertEqual(FakeClient.calls[0][0], 'trove.search')
        self.assertEqual(FakeClient.calls[0][1]['account_id'], 'a')
        self.assertEqual(FakeClient.calls[0][3], 'cli')

    def test_pretty_changes_only_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            _, compact, compact_value = self.invoke(Path(directory), 'accounts')
            _, pretty, pretty_value = self.invoke(Path(directory), 'accounts', pretty=True)
        self.assertEqual(compact_value, pretty_value)
        self.assertNotEqual(compact, pretty)

    def test_typed_daemon_error_has_stable_exit_and_no_traceback_or_private_path(self):
        FakeClient.failure = TroveClientError(
            'ambiguous_target', 'private path must not be rendered',
            response={
                'protocol': 'trove/1', 'request_id': 'cli-test-request-0001', 'ok': False,
                'error': {'code': 'ambiguous_target', 'retryable': False, 'details': {'accounts': ['a', 'b']}},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            code, raw, result = self.invoke(Path(directory), 'recall', '--target', 'same-name')
        self.assertEqual(code, 3)
        self.assertEqual(result['error']['code'], 'ambiguous_target')
        self.assertNotIn('Traceback', raw)
        self.assertNotIn('/Users/', raw)

    def test_invalid_input_never_connects(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _raw, result = self.invoke(Path(directory), 'search', '--query', '')
        self.assertEqual(code, 2)
        self.assertEqual(result['error']['code'], 'invalid_request')
        self.assertEqual(FakeClient.calls, [])


if __name__ == '__main__':
    unittest.main()
