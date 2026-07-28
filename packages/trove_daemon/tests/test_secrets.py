from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from trove_daemon.secrets import (
    SecretUnavailable,
    load_key_store_secret,
    read_agent_switch_secret,
)


def executable(path: Path, body: str) -> Path:
    path.write_text('#!/usr/bin/env python3\n' + body, encoding='utf-8')
    path.chmod(0o700)
    return path


class SecretBoundaryTests(unittest.TestCase):
    def test_secret_is_read_only_from_inherited_non_tty_fd(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = executable(
                Path(directory) / 'agent-switch',
                """import os,sys
fd=int(sys.argv[sys.argv.index('--fd')+1])
os.write(fd,b'fd-only-value')
print('stdout-decoy')
""",
            )
            self.assertEqual(
                read_agent_switch_secret('VALID_NAME', executable=tool),
                b'fd-only-value',
            )

    def test_secret_command_output_is_never_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = executable(
                Path(directory) / 'agent-switch',
                """import sys
sys.stderr.write('must-not-leak')
print('also-must-not-leak')
raise SystemExit(2)
""",
            )
            with self.assertRaises(SecretUnavailable) as caught:
                read_agent_switch_secret(
                    'VALID_NAME', executable=tool,
                )
            self.assertNotIn('must-not-leak', str(caught.exception))

    def test_key_store_wrapper_is_parsed_without_returning_raw_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = executable(
                Path(directory) / 'agent-switch',
                """import os,sys
fd=int(sys.argv[sys.argv.index('--fd')+1])
os.write(fd,b'{"keys":{"abcd":{"dk":"1234","rounds":2}}}')
""",
            )
            from unittest.mock import patch

            with patch(
                'trove_daemon.secrets.AGENT_SWITCH', tool,
            ), patch(
                'trove_daemon.secrets.read_agent_switch_secret',
                lambda _name: (
                    b'{"keys":{"abcd":{"dk":"1234","rounds":2}}}'
                ),
            ):
                self.assertEqual(
                    load_key_store_secret('VALID_NAME'),
                    {'abcd': {'dk': '1234', 'rounds': 2}},
                )

    def test_invalid_secret_name_fails_before_execution(self):
        with self.assertRaises(SecretUnavailable):
            read_agent_switch_secret('BAD NAME', executable='/missing')


if __name__ == '__main__':
    unittest.main()
