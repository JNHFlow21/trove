from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from trove_core.product_config import (
    ProductConfig, ProductConfigError, load_product_config, write_product_config,
)
from trove_client.product_config import resolve_vault_root


class ProductConfigTests(unittest.TestCase):
    def test_missing_config_allows_read_discovery_but_write_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            vault = home / 'existing-vault'
            vault.mkdir()
            config = load_product_config(home / 'missing.json', env={'TROVE_VAULT_ROOT': str(vault)}, home=home)
            self.assertFalse(config.explicit)
            self.assertEqual(config.vault_root, vault)
            with self.assertRaisesRegex(ProductConfigError, 'write operations require') as raised:
                load_product_config(home / 'missing.json', env={}, home=home, for_write=True)
            self.assertEqual(raised.exception.code, 'explicit_config_required')

    def test_atomic_owner_only_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'config/config.json'
            expected = ProductConfig(
                vault_root=root / 'vault', runtime_root=root / 'runtime',
                secret_names=('TROVE_PROVIDER_TOKEN',), explicit=True,
            )
            write_product_config(path, expected)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(path.parent).st_mode & 0o777, 0o700)
            self.assertEqual(load_product_config(path).to_dict(), expected.to_dict())

    def test_secret_values_and_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            base = ProductConfig().to_dict()
            for payload in (
                {**base, 'secret_values': {'TOKEN': 'private'}},
                {**base, 'secret_names': ['not-a-secret-name']},
            ):
                path.write_text(json.dumps(payload), encoding='utf-8')
                path.chmod(0o600)
                with self.assertRaises(ProductConfigError):
                    load_product_config(path)

    def test_explicit_write_path_does_not_require_default_config(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'new-vault'
            self.assertEqual(resolve_vault_root(str(vault), create=True), vault.resolve())
            self.assertEqual(vault.stat().st_mode & 0o777, 0o700)


if __name__ == '__main__':
    unittest.main()
