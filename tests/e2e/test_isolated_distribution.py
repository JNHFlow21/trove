from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from scripts.activate_distribution import install_release
from tests.distribution_support import ROOT, clean_env, distribution_dir


class IsolatedDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.install_root = Path(tempfile.mkdtemp(prefix='trove-isolated-activation-'))
        cls.release = install_release(
            distribution_dir() / 'distribution-manifest.json', cls.install_root,
        )
        cls.venv = cls.release / 'venv'

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.install_root, ignore_errors=True)

    def run_json(self, argv, *, cwd: Path, timeout=60):
        completed = subprocess.run(
            argv, cwd=cwd, env=clean_env(), text=True, capture_output=True, timeout=timeout,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        return json.loads(completed.stdout)

    def test_installed_artifact_runs_without_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            subprocess.run([
                str(ROOT / 'scripts/trove-python'), 'scripts/generate_fixture_vault.py',
                '--vault', str(vault), '--reset',
            ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            cwd = root / 'empty-cwd'
            cwd.mkdir()
            trove = self.venv / 'bin/trove'
            try:
                self.assertTrue(self.run_json([str(trove), '--vault', str(vault), 'start'], cwd=cwd)['ok'])
                version = self.run_json([str(trove), 'version'], cwd=cwd)
                self.assertEqual(version['data']['version'], '1.0.0')
                doctor = self.run_json([str(trove), '--vault', str(vault), 'doctor'], cwd=cwd)
                self.assertTrue(doctor['ok'])
                self.assertTrue(doctor['data']['provider']['ok'], doctor)
                recall = self.run_json([
                    str(trove), '--vault', str(vault), 'recall',
                    '--conversation-id', 'conv-sales-review', '--limit', '3',
                ], cwd=cwd)
                self.assertTrue(recall['ok'])
                self.assertIn('coverage', recall)
                help_result = subprocess.run(
                    [str(self.venv / 'bin/troved'), '--help'], cwd=cwd,
                    env=clean_env(), text=True, capture_output=True, timeout=20,
                )
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn('--vault', help_result.stdout)

                script = textwrap.dedent('''
                    import asyncio, json, os
                    from mcp import ClientSession, StdioServerParameters
                    from mcp.client.stdio import stdio_client
                    async def run():
                        params = StdioServerParameters(
                            command=os.environ['TROVE_MCP'],
                            args=['--pack', 'standard', '--vault', os.environ['TROVE_VAULT']],
                        )
                        async with stdio_client(params) as streams:
                            async with ClientSession(*streams) as session:
                                await session.initialize()
                                tools = await session.list_tools()
                                result = await session.call_tool('trove_capabilities', {})
                                print(json.dumps({'tools': len(tools.tools), 'error': result.isError}))
                    asyncio.run(run())
                ''')
                environment = clean_env() | {
                    'TROVE_MCP': str(self.venv / 'bin/trove-mcp'),
                    'TROVE_VAULT': str(vault),
                }
                mcp = subprocess.run(
                    [str(self.venv / 'bin/python'), '-c', script], cwd=cwd,
                    env=environment, text=True, capture_output=True, timeout=30,
                )
                self.assertEqual(mcp.returncode, 0, mcp.stderr)
                self.assertEqual(json.loads(mcp.stdout), {'tools': 19, 'error': False})
            finally:
                stopped = self.run_json([str(trove), '--vault', str(vault), 'stop'], cwd=cwd)
                self.assertTrue(stopped['data']['drained'], stopped)
                self.assertFalse(stopped['data']['running'], stopped)

    def test_isolated_environment_has_no_source_path_or_removed_packages(self):
        completed = subprocess.run([
            str(self.venv / 'bin/python'), '-c',
            'import importlib.util,json,sys; print(json.dumps({"path":sys.path,"api":importlib.util.find_spec("trove_api"),"provider":importlib.util.find_spec("trove_provider_wechat") is not None}))',
        ], cwd=self.venv, env=clean_env(), text=True, capture_output=True, check=True)
        payload = json.loads(completed.stdout)
        self.assertIsNone(payload['api'])
        self.assertTrue(payload['provider'])
        self.assertFalse(any(str(ROOT) in path for path in payload['path']))

        identity_script = (
            'import json; from pathlib import Path; '
            'from importlib.metadata import distributions; '
            'from trove_daemon.lifecycle import build_identity,catalog_identity; '
            'd=[d for d in distributions() if any(e.group=="trove.providers" for e in d.entry_points)]; '
            'p=json.loads(Path(d[0].locate_file("trove_provider_wechat/manifest.json")).read_text()); '
            'print(json.dumps({"runtime_build_hash":build_identity(),"catalog_hash":catalog_identity(),"provider_package_hash":p["package_sha256"]},sort_keys=True))'
        )
        identity = subprocess.run(
            [str(self.venv / 'bin/python'), '-c', identity_script], cwd=self.venv,
            env=clean_env(), text=True, capture_output=True, check=True,
        )
        manifest = json.loads((distribution_dir() / 'distribution-manifest.json').read_text())
        self.assertEqual(json.loads(identity.stdout), {
            key: manifest[key]
            for key in ('runtime_build_hash', 'catalog_hash', 'provider_package_hash')
        })

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS sender dependencies')
    def test_installed_provider_includes_macos_sender_dependencies(self):
        completed = subprocess.run(
            [
                str(self.venv / 'bin/python'), '-c',
                'import AppKit, Quartz; print("ready")',
            ],
            cwd=self.venv,
            env=clean_env(),
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode, 0, completed.stderr or completed.stdout,
        )
        self.assertEqual(completed.stdout.strip(), 'ready')


if __name__ == '__main__':
    unittest.main()
