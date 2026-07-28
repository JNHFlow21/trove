from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.import_job import run_import_job
from trove_core.wechat.media.locator import locate_media_asset
from packages.trove_core.tests.test_message_media_registration import create_multimodal_message_source


class SourceSnapshotRegistryTests(unittest.TestCase):
    def test_import_binds_assets_to_opaque_vault_relative_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-a'
            create_multimodal_message_source(snapshot)

            result = run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                source_row = conn.execute('SELECT * FROM source_snapshots').fetchone()
                bindings = list(conn.execute('SELECT * FROM media_source_bindings ORDER BY asset_id'))

            self.assertEqual(result.status, 'completed')
            self.assertEqual(source_row['state'], 'available')
            self.assertFalse(Path(source_row['root_ref']).is_absolute())
            self.assertEqual(len(source_row['snapshot_revision']), 64)
            self.assertEqual(len(bindings), 5)
            self.assertTrue(all(len(row['account_dir_hash']) == 64 for row in bindings))
            self.assertNotIn('wxid_ownerfixture', str([dict(row) for row in bindings]))
            self.assertNotIn(str(snapshot), cfg.paths.sqlite_path.read_bytes().decode('utf-8', errors='ignore'))

    def test_snapshot_mutation_is_detected_before_locator_reads_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-b'
            account = create_multimodal_message_source(snapshot)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                asset = conn.execute("SELECT * FROM media_assets WHERE modality='image' AND source_type='private_chat'").fetchone()
            with (account / 'message_0.db').open('ab') as handle:
                handle.write(b'changed')

            located = locate_media_asset(cfg, store, asset)

            self.assertEqual(located.status, 'unavailable')
            self.assertEqual(located.reason, 'source_snapshot_changed')


if __name__ == '__main__':
    unittest.main()
