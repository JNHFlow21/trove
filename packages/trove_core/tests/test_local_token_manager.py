from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.security.local_token import LocalTokenManager


class LocalTokenManagerTests(unittest.TestCase):
    def test_token_is_created_under_existing_vault_and_verifies_bearer(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'api' / 'local_token'
            manager = LocalTokenManager(path)

            token = manager.get_or_create()

            self.assertTrue(token.startswith('trove-local-'))
            self.assertEqual(path.read_text(encoding='utf-8').strip(), token)
            self.assertTrue(manager.verify(f'Bearer {token}'))
            self.assertFalse(manager.verify('Bearer wrong'))

    def test_missing_vault_uses_ephemeral_token_without_creating_parent(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'missing' / 'api' / 'local_token'
            manager = LocalTokenManager(path)

            token = manager.get_or_create()

            self.assertTrue(token.startswith('trove-local-'))
            self.assertFalse(path.exists())
            self.assertEqual(manager.get_or_create(), token)


if __name__ == '__main__':
    unittest.main()
