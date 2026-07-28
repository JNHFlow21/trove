from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

from scripts.public_surface_lint import repository_snapshot


ROOT = Path(__file__).resolve().parents[2]


class PackagingSmokeTests(unittest.TestCase):
    def test_source_package_declares_only_v1_runtime_entrypoints(self):
        pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(pyproject['project']['name'], 'trove-runtime')
        self.assertEqual(pyproject['project']['version'], '1.0.0')
        self.assertEqual(
            set(pyproject['project']['scripts']),
            {'trove', 'trove-mcp', 'troved'},
        )
        from trove_core import __version__
        self.assertEqual(pyproject['project']['version'], __version__)
        self.assertNotIn('packages/trove_provider_wechat', pyproject['tool']['setuptools']['packages']['find']['where'])
        release = pyproject['tool']['trove']['release']
        self.assertEqual(release, {
            'protocol': 'trove/1',
            'artifact_set': ['trove-runtime', 'trove-provider-wechat'],
            'public_surface': ['mcp', 'cli', 'skills'],
            'official_skills': 6,
            'promotion_policy': 'automated-gates',
            'final_manifest': 'dist/v1.0.0-release-manifest.json',
            'legacy_aliases': False,
        })

    def test_provider_is_an_independent_version_locked_distribution(self):
        provider = tomllib.loads((
            ROOT / 'packages/trove_provider_wechat/pyproject.toml'
        ).read_text(encoding='utf-8'))
        self.assertEqual(provider['project']['name'], 'trove-provider-wechat')
        self.assertEqual(provider['project']['dependencies'], [
            'trove-runtime==1.0.0',
            "pyobjc-framework-Cocoa>=10,<12; sys_platform == 'darwin'",
            "pyobjc-framework-Quartz>=10,<12; sys_platform == 'darwin'",
        ])
        self.assertEqual(
            provider['project']['entry-points']['trove.providers'],
            {'wechat-source': 'trove_provider_wechat:create_provider'},
        )

    def test_source_tree_has_no_web_api_or_node_packaging(self):
        paths = set(repository_snapshot())
        self.assertFalse(any(path.startswith('apps/web_console/') for path in paths))
        self.assertFalse(any(path.startswith('packages/trove_api/') for path in paths))
        self.assertNotIn('package.json', paths)
        self.assertNotIn('package-lock.json', paths)

    def test_python_orchestration_and_runtime_packages_exist(self):
        self.assertTrue((ROOT / 'scripts/check.py').is_file())
        for script in ('build_distribution.py', 'verify_distribution.py', 'activate_distribution.py'):
            self.assertTrue((ROOT / 'scripts' / script).is_file())
        self.assertTrue((ROOT / 'scripts/trove-python').is_file())
        for package in (
            'trove_protocol', 'trove_core', 'trove_client', 'trove_daemon',
            'trove_provider_wechat', 'trove_cli', 'trove_mcp',
        ):
            self.assertTrue((ROOT / 'packages' / package / package).is_dir(), package)
        self.assertIn('ci.yml', {path.name for path in (ROOT / '.github/workflows').glob('*.yml')})

    def test_v1_release_note_describes_only_the_current_product_surface(self):
        note = (ROOT / 'docs/release-notes/v1.0.0.md').read_text(encoding='utf-8').lower()
        for current in ('trove-mcp --pack standard', 'trove-recall', 'trove-search'):
            self.assertIn(current, note)
        for removed in (
            'trove-api', 'trove_api', 'web console', 'npm ', 'chat-recall',
            'person-profile', 'files-list', 'python -m',
        ):
            self.assertNotIn(removed, note)


if __name__ == '__main__':
    unittest.main()
