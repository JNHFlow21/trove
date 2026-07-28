from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.vault.config import VaultConfig, default_vault_root, path_is_under, product_vault_root

class VaultConfigTests(unittest.TestCase):
    def test_missing_default_reports_guidance_without_project_vault(self):
        cfg = VaultConfig.resolve(env={})
        self.assertEqual(cfg.source, 'unconfigured')
        repo_root = Path(__file__).resolve().parents[3]
        self.assertFalse(path_is_under(default_vault_root(), repo_root))
        self.assertFalse(path_is_under(product_vault_root(env={}, allow_default_home=True), repo_root))  # type: ignore[arg-type]
        self.assertFalse(cfg.status().root.endswith('.'))
        self.assertFalse(cfg.status().available)
        self.assertIn('TROVE_VAULT_ROOT', cfg.status().problem)

    def test_auto_discovers_product_vault_when_structure_exists(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'Trove' / 'trove-vault'
            (root / 'index').mkdir(parents=True)
            cfg = VaultConfig.resolve(env={'HOME': d})
            status = cfg.status().to_dict()
            self.assertEqual(cfg.root, root)
            self.assertEqual(status['source'], 'auto-discovered')
            self.assertTrue(status['available'])

    def test_missing_configured_external_path_reports_problem_without_create(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / 'not-mounted' / 'vault'
            cfg = VaultConfig.resolve(str(missing), env={})
            status = cfg.status().to_dict()
            self.assertFalse(missing.exists())
            self.assertFalse(status['available'])
            self.assertIn('unavailable', status['problem'])

    def test_legacy_config_readonly_guidance(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(env={'WECHAT_KOS_VAULT_ROOT': d})
            status = cfg.status().to_dict()
            self.assertEqual(status['source'], 'legacy-env-readonly')
            self.assertTrue(status['legacy_config_detected'])
            self.assertIn('compatibility', status['migration_guidance'])

class VaultEnsureRuntimeDirsTests(unittest.TestCase):
    def test_ensure_creates_runtime_metadata_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(d, env={})
            cfg.ensure()
            self.assertTrue((Path(d) / 'manifests').is_dir())
            self.assertTrue((Path(d) / 'jobs').is_dir())
            self.assertTrue((Path(d) / 'sources').is_dir())
            self.assertTrue((Path(d) / 'proof').is_dir())
