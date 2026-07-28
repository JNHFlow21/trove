from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from trove_core.providers.registry import ProviderAllowlistEntry, ProviderRegistry
from trove_daemon.provider_loader import (
    ProviderLoader, discover_provider_distributions,
)
from trove_protocol.provider import ProviderManifest


PACKAGE = Path(__file__).resolve().parents[2] / 'trove_provider_wechat' / 'trove_provider_wechat'


def _registry(manifest: ProviderManifest, *, package_hash: str | None = None) -> ProviderRegistry:
    allow = ProviderAllowlistEntry(
        provider_id=manifest.provider_id,
        package_sha256=package_hash or manifest.package_sha256,
        owner_uid=os.getuid(), capabilities=frozenset(manifest.capabilities),
        source_types=frozenset(manifest.source_types),
        secret_names=frozenset(manifest.secret_names),
    )
    return ProviderRegistry({manifest.provider_id: allow})


class ProviderLoaderTests(unittest.TestCase):
    def test_distribution_metadata_is_discovered_without_import(self):
        entry = SimpleNamespace(
            group='trove.providers', name='wechat-source',
            value='trove_provider_wechat:create_provider',
        )

        class Distribution:
            entry_points = (entry,)
            metadata = {'Name': 'trove-provider-wechat'}
            version = '1.0.0'

            @staticmethod
            def locate_file(_path):
                return PACKAGE

        with mock.patch('importlib.import_module') as imported:
            discovered = discover_provider_distributions([Distribution()])
        imported.assert_not_called()
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].provider_id, 'wechat-source')
        self.assertEqual(discovered[0].package_root, PACKAGE)

    def test_editable_distribution_resolves_direct_url_and_deduplicates(self):
        entry = SimpleNamespace(
            group='trove.providers', name='wechat-source',
            value='trove_provider_wechat:create_provider',
        )

        class EditableDistribution:
            entry_points = (entry,)
            metadata = {'Name': 'trove-provider-wechat'}
            version = '1.0.0'

            @staticmethod
            def locate_file(_path):
                return Path('/definitely/missing/trove_provider_wechat')

            @staticmethod
            def read_text(name):
                if name != 'direct_url.json':
                    return None
                return json.dumps({
                    'url': PACKAGE.parent.as_uri(),
                    'dir_info': {'editable': True},
                })

        discovered = discover_provider_distributions([
            EditableDistribution(),
            EditableDistribution(),
        ])

        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0].package_root, PACKAGE.resolve())

    def test_failed_preimport_verification_never_imports_distribution(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied = root / 'package'
            shutil.copytree(PACKAGE, copied)
            (copied / 'main.py').write_text((copied / 'main.py').read_text() + '\n# tampered\n')
            distribution = SimpleNamespace(
                provider_id='wechat-source', distribution_name='trove-provider-wechat',
                version='1.0.0', module_name='trove_provider_wechat',
                factory_name='create_provider', package_root=copied,
            )
            loader = ProviderLoader(_registry(manifest), runtime_dir=root / 'runtime')
            with mock.patch('importlib.import_module') as imported:
                result = loader.load_distribution(distribution)
            imported.assert_not_called()
            self.assertFalse(result.ok)
            self.assertEqual(result.error_code, 'provider_package_hash_mismatch')

    def test_verified_distribution_loads_only_through_contract_registry(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            loader = ProviderLoader(_registry(manifest), runtime_dir=Path(directory) / 'runtime')
            result = loader.load(PACKAGE, module_name='trove_provider_wechat')
            self.assertTrue(result.ok)
            self.assertTrue(result.to_dict()['pure_vault_read_available'])

    def test_hash_or_allowlist_failure_keeps_pure_vault_read_available(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied = root / 'package'
            shutil.copytree(PACKAGE, copied)
            (copied / 'main.py').write_text((copied / 'main.py').read_text() + '\n# tampered\n')
            cases = [
                (_registry(manifest), copied),
                (_registry(manifest, package_hash='0' * 64), PACKAGE),
            ]
            for index, (registry, package) in enumerate(cases):
                loader = ProviderLoader(registry, runtime_dir=root / f'runtime-{index}')
                result = loader.load(package, module_name='trove_provider_wechat')
                self.assertFalse(result.ok)
                self.assertTrue(result.to_dict()['pure_vault_read_available'])

    def test_core_and_daemon_have_no_direct_provider_internal_import(self):
        repo = Path(__file__).resolve().parents[3]
        findings = []
        for base in (repo / 'packages' / 'trove_core' / 'trove_core', repo / 'packages' / 'trove_daemon' / 'trove_daemon'):
            for path in base.rglob('*.py'):
                text = path.read_text(encoding='utf-8')
                if 'trove_provider_wechat.' in text:
                    findings.append(path.relative_to(repo).as_posix())
        self.assertEqual(findings, [])


if __name__ == '__main__':
    unittest.main()
