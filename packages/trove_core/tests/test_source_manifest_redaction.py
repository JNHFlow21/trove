from __future__ import annotations
from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.wechat.source_inventory import inventory
from trove_core.wechat.source_manifest import RedactedSourceManifest, build_manifest

class SourceManifestRedactionTests(unittest.TestCase):
    def test_source_scan_runs_before_short_manifest_publish_writer(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'decrypted' / 'current'
            source.mkdir(parents=True)
            (source / 'message.db').write_bytes(b'x')
            vault = root / 'vault'
            lock_depth = 0
            scan_states: list[bool] = []
            write_states: list[bool] = []
            original_coordinated = agent_tools.coordinated_vault_mutation
            original_inventory = agent_tools.inventory
            original_write = RedactedSourceManifest.write

            @contextmanager
            def tracked_coordinated(*args, **kwargs):
                nonlocal lock_depth
                with original_coordinated(*args, **kwargs) as session:
                    lock_depth += 1
                    try:
                        yield session
                    finally:
                        lock_depth -= 1

            def tracked_inventory(*args, **kwargs):
                scan_states.append(lock_depth > 0)
                return original_inventory(*args, **kwargs)

            def tracked_write(manifest, path):
                write_states.append(lock_depth > 0)
                return original_write(manifest, path)

            with patch.object(
                agent_tools, 'coordinated_vault_mutation', tracked_coordinated,
            ), patch.object(
                agent_tools, 'inventory', side_effect=tracked_inventory,
            ), patch.object(
                RedactedSourceManifest, 'write', autospec=True, side_effect=tracked_write,
            ):
                report = agent_tools.source_manifest(vault, [str(source)])

            self.assertEqual(scan_states, [False])
            self.assertEqual(write_states, [True])
            self.assertEqual(report['written'], 'source_manifest.redacted.json')

    def test_manifest_contains_redacted_source_ids_not_full_private_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'decrypted' / 'current'
            root.mkdir(parents=True)
            (root / 'message.db').write_bytes(b'x')
            manifest = build_manifest(inventory([root])).to_dict()
            self.assertEqual(len(manifest['canonical_source_ids']), 1)
            body = str(manifest)
            self.assertNotIn('/Users/', body)
            self.assertIn('redacted_path', body)
