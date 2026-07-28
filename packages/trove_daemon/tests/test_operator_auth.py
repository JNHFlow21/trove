from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from trove_daemon.operator_auth import (
    OperatorAppIdentity,
    SignedOperatorAuthorizer,
    load_operator_trust,
    save_operator_trust,
    trust_path,
)


class OperatorAuthorizationTests(unittest.TestCase):
    def test_trust_is_owner_only_and_exact_identity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / 'Companion.app'
            executable = app / 'Contents/MacOS/Companion'
            executable.parent.mkdir(parents=True)
            executable.write_text('fixture', encoding='utf-8')
            identity = OperatorAppIdentity(
                bundle_identifier='com.trove.Companion',
                app_path=str(app),
                executable_path=str(executable),
                cdhash='a' * 40,
            )
            save_operator_trust(root, identity)
            self.assertEqual(load_operator_trust(root), identity)
            self.assertEqual(
                trust_path(root).stat().st_mode & 0o777, 0o600,
            )

            authorizer = SignedOperatorAuthorizer(
                root,
                process_path=lambda _pid: executable.resolve(),
                inspect_app=lambda _path: identity,
            )
            self.assertTrue(authorizer.authorize(42))
            wrong = SignedOperatorAuthorizer(
                root,
                process_path=lambda _pid: root / 'other',
                inspect_app=lambda _path: identity,
            )
            self.assertFalse(wrong.authorize(42))

    def test_missing_trust_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            authorizer = SignedOperatorAuthorizer(
                directory,
                process_path=lambda _pid: Path('/unused'),
            )
            self.assertFalse(authorizer.authorize(os.getpid()))


if __name__ == '__main__':
    unittest.main()
