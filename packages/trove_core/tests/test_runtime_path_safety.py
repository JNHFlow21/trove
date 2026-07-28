from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.vault.config import VaultConfig

class RuntimePathSafetyTests(unittest.TestCase):
    def test_rejects_product_repo_as_vault(self):
        repo = Path(__file__).resolve().parents[3]
        cfg = VaultConfig.resolve(str(repo), env={})
        with self.assertRaises(ValueError):
            cfg.validate_runtime_path()

    def test_allows_temp_vault(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(d, env={})
            cfg.validate_runtime_path()
