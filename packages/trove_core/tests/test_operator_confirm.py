from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from trove_core.security.operator_confirm import (
    OperatorConfirmationError,
    _open_controlling_terminal,
    decide_from_controlling_terminal,
)


class _TTY:
    def __init__(self, input_text):
        self.input = io.StringIO(input_text)
        self.output = io.StringIO()

    def isatty(self):
        return True

    def write(self, value):
        return self.output.write(value)

    def flush(self):
        return None

    def readline(self):
        return self.input.readline()

    def getvalue(self):
        return self.output.getvalue()


class _Record:
    def __init__(self):
        self.payload = {'destination': 'fixture-destination', 'selection': ['fixture-citation']}
        self.action = 'files_export'
        self.danger_class = 'private_export'

    def to_dict(self):
        return {'approval_id': 'approval-fixture', 'status': 'approved'}


class _Manager:
    def __init__(self):
        self.decisions = []
        self.record = _Record()

    def load(self, approval_id):
        return self.record

    def decide(self, approval_id, status, note=None):
        self.decisions.append((approval_id, status, note))
        return self.record


class OperatorConfirmTests(unittest.TestCase):
    def test_controlling_terminal_uses_non_seekable_safe_duplex_stream(self):
        reader = io.BytesIO(b'APPROVE approval-fixture\n')
        writer = io.BytesIO()
        with patch(
            'trove_core.security.operator_confirm.io.FileIO',
            side_effect=(reader, writer),
        ) as file_io:
            terminal = _open_controlling_terminal()
        self.assertEqual(terminal.readline(), 'APPROVE approval-fixture\n')
        terminal.write('prompt')
        terminal.flush()
        self.assertEqual(writer.getvalue(), b'prompt')
        self.assertEqual(
            [call.args for call in file_io.call_args_list],
            [('/dev/tty', 'r'), ('/dev/tty', 'w')],
        )
        terminal.close()

    def test_exact_payload_is_displayed_before_live_confirmation(self):
        manager = _Manager()
        terminal = _TTY('APPROVE approval-fixture\n')
        result = decide_from_controlling_terminal(
            manager, 'approval-fixture', 'approved', terminal=terminal,
        )
        self.assertEqual(result['status'], 'approved')
        self.assertEqual(manager.decisions, [('approval-fixture', 'approved', None)])
        output = terminal.getvalue()
        self.assertIn('fixture-destination', output)
        self.assertIn('fixture-citation', output)

    def test_wrong_confirmation_never_calls_decide(self):
        manager = _Manager()
        terminal = _TTY('yes\n')
        with self.assertRaises(OperatorConfirmationError):
            decide_from_controlling_terminal(
                manager, 'approval-fixture', 'approved', terminal=terminal,
            )
        self.assertEqual(manager.decisions, [])


if __name__ == '__main__':
    unittest.main()
