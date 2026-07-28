from __future__ import annotations

import inspect
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from trove_cli import operator_approval
from trove_cli.parser import CLIInputError, build_parser
from trove_cli.v1_main import run
from trove_core.security.operator_confirm import OperatorConfirmationError


class CliHardeningContractTests(unittest.TestCase):
    def test_agent_routes_cannot_decide_approval(self):
        parser = build_parser()
        for command in (['approval', 'approve'], ['approval', 'reject'], ['approval-decision']):
            with self.subTest(command=command), self.assertRaises(CLIInputError):
                parser.parse_args(command)

    def test_operator_path_has_no_yes_stdin_or_environment_bypass(self):
        signature = inspect.signature(operator_approval.decide)
        self.assertNotIn('yes', signature.parameters)
        self.assertNotIn('stdin', signature.parameters)
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch(
                'trove_cli.v1_main.operator_decide',
                side_effect=OperatorConfirmationError('controlling terminal required'),
            ):
                code = run([
                    '--vault', directory, 'operator', 'approve',
                    'approval-0000000000000001',
                ], stdout=output)
        self.assertEqual(code, 7)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload['error']['code'], 'operator_confirmation_required')
        self.assertNotIn(str(directory), output.getvalue())

    def test_removed_web_and_process_routes_are_unknown(self):
        parser = build_parser()
        for command in ('up', 'down', 'ps', 'wiki', 'schedule', 'embed-daemon'):
            with self.subTest(command=command), self.assertRaises(CLIInputError):
                parser.parse_args([command])


if __name__ == '__main__':
    unittest.main()
