from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest

from tests.distribution_support import clean_env, create_venv, distribution_dir
from scripts.verify_distribution import (
    DistributionVerificationError, verify_distribution,
)


class ProviderPackageInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.venv = create_venv(provider=False)

    def entry_points(self) -> int:
        completed = subprocess.run([
            str(self.venv / 'bin/python'), '-c',
            'from importlib.metadata import entry_points; print(len(entry_points(group="trove.providers")))',
        ], cwd=self.venv, env=clean_env(), text=True, capture_output=True, check=True)
        return int(completed.stdout)

    def test_provider_can_install_load_and_uninstall_independently(self):
        self.assertEqual(self.entry_points(), 0)
        wheel = next(distribution_dir().glob('trove_provider_wechat-*.whl'))
        subprocess.run([
            str(self.venv / 'bin/python'), '-m', 'pip', 'install', '--no-deps', str(wheel),
        ], cwd=self.venv, env=clean_env(), check=True, stdout=subprocess.DEVNULL)
        self.assertEqual(self.entry_points(), 1)
        with tempfile.TemporaryDirectory() as directory:
            script = textwrap.dedent('''
                import json, os
                from pathlib import Path
                from trove_daemon.provider_loader import ProviderLoader, discover_provider_distributions, official_provider_registry
                found = discover_provider_distributions()
                loader = ProviderLoader(official_provider_registry(), runtime_dir=Path(os.environ['RUNTIME']))
                result = loader.load_distribution(found[0])
                print(json.dumps({'count': len(found), 'ok': result.ok, 'provider': result.provider_id}))
            ''')
            completed = subprocess.run(
                [str(self.venv / 'bin/python'), '-c', script], cwd=self.venv,
                env=clean_env() | {'RUNTIME': directory},
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout), {
                'count': 1, 'ok': True, 'provider': 'wechat-source',
            })
        subprocess.run([
            str(self.venv / 'bin/python'), '-m', 'pip', 'uninstall', '-y', 'trove-provider-wechat',
        ], cwd=self.venv, env=clean_env(), check=True, stdout=subprocess.DEVNULL)
        self.assertEqual(self.entry_points(), 0)
        version = subprocess.run(
            [str(self.venv / 'bin/trove'), 'version'], cwd=self.venv,
            env=clean_env(), text=True, capture_output=True,
        )
        self.assertEqual(version.returncode, 0, version.stderr)
        self.assertEqual(json.loads(version.stdout)['data']['version'], '1.0.0')

    def test_tampered_provider_artifact_is_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'distribution'
            shutil.copytree(distribution_dir(), copied)
            provider = next(copied.glob('trove_provider_wechat-*.whl'))
            with provider.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(DistributionVerificationError, 'artifact hash mismatch'):
                verify_distribution(copied / 'distribution-manifest.json')


if __name__ == '__main__':
    unittest.main()
