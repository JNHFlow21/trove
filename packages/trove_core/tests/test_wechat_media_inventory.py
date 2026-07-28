from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.wechat.media.resources import discover_media_assets
from trove_core.wechat.source_inventory import inventory


class WeChatMediaInventoryTests(unittest.TestCase):
    def test_message_resources_create_distinct_media_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'decrypted' / 'current' / 'acct'
            (root / 'cache').mkdir(parents=True)
            (root / 'cache' / 'photo.jpg').write_bytes(b'\xff\xd8\xfffixture')
            db = root / 'message_resource.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE message_resource(local_id INTEGER, local_type TEXT, path TEXT)')
            conn.execute('INSERT INTO message_resource VALUES(?,?,?)', (1, '3', 'cache/photo.jpg'))
            conn.execute('INSERT INTO message_resource VALUES(?,?,?)', (2, '34', 'voice_missing.amr'))
            conn.commit(); conn.close()

            refs = discover_media_assets(root, account_id='acct-a')
            modalities = {r.modality for r in refs}
            self.assertIn('image', modalities)
            self.assertIn('voice', modalities)
            image = next(r for r in refs if r.modality == 'image')
            self.assertEqual(image.cache_state, 'source_available')
            self.assertIsNone(image.path_ref)
            voice = next(r for r in refs if r.modality == 'voice')
            self.assertEqual(voice.cache_state, 'missing_local_cache')
            self.assertIsNone(voice.path_ref)
            self.assertTrue(image.citation.startswith('trove://wechat/acct-a/media/'))

    def test_source_inventory_reports_redacted_media_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'decrypted' / 'current' / 'acct'
            root.mkdir(parents=True)
            db = root / 'message_resource.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE resource(local_id INTEGER, local_type TEXT, file_path TEXT)')
            conn.execute('INSERT INTO resource VALUES(?,?,?)', (1, '3', 'missing.jpg'))
            conn.commit(); conn.close()
            candidate = inventory([root])[0]
            self.assertTrue(candidate.importable)
            self.assertGreaterEqual(candidate.media_counts.get('image', 0), 1)
            self.assertNotIn(str(root), candidate.to_dict()['redacted_path'])


if __name__ == '__main__':
    unittest.main()
