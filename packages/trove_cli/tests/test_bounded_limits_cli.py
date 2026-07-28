from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from trove_cli.v1_main import run


class _NoConnect:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError('bounded invalid input reached daemon transport')


class BoundedCliContractTests(unittest.TestCase):
    def assert_invalid(self, command: list[str]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            code = run(
                ['--vault', directory, *command], stdout=output,
                client_factory=_NoConnect,
            )
        self.assertEqual(code, 2, command)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload['error']['code'], 'invalid_request')
        self.assertNotIn(str(Path(directory)), output.getvalue())

    def test_catalog_bounds_are_rejected_before_vault_or_daemon_io(self):
        for command in (
            ['search', '--query', 'x', '--limit', '0'],
            ['search', '--query', 'x', '--limit', '1001'],
            ['context', '--citation', 'trove://source/a/c/s/1', '--before', '201'],
            ['context', '--citation', 'trove://source/a/c/s/1', '--after', '-1'],
            ['files', 'list', '--limit', '1001'],
        ):
            with self.subTest(command=command):
                self.assert_invalid(command)


if __name__ == '__main__':
    unittest.main()
