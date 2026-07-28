from __future__ import annotations

import unittest

from scripts.public_surface_lint import scan_public_surface, scan_snapshot


class PublicSurfaceLintTests(unittest.TestCase):
    def test_repository_index_has_only_v1_public_surface(self):
        self.assertEqual(scan_public_surface(), [])

    def test_rejects_web_api_node_and_unreviewed_entrypoint(self):
        findings = scan_snapshot({
            'pyproject.toml': '[project.scripts]\ntrove = "x:y"\ntrove-api = "x:y"\n',
            'apps/web_console/server.mjs': 'serve()',
            'packages/trove_api/trove_api/server.py': '',
        })
        self.assertTrue(any('forbidden legacy product surface' in item for item in findings))
        self.assertTrue(any('Node/JavaScript' in item for item in findings))
        self.assertTrue(any('public executables' in item for item in findings))

    def test_rejects_core_and_provider_imports_from_public_adapters(self):
        findings = scan_snapshot({
            'pyproject.toml': (
                '[project.scripts]\n'
                'trove = "x:y"\n'
                'trove-mcp = "x:y"\n'
                'troved = "x:y"\n'
            ),
            'packages/trove_cli/trove_cli/example.py': (
                'from trove_core.store import sqlite_store\n'
                'import trove_provider_wechat\n'
            ),
        })
        self.assertTrue(any('imports trove_core' in item for item in findings))
        self.assertTrue(any('imports a source provider' in item for item in findings))


if __name__ == '__main__':
    unittest.main()
