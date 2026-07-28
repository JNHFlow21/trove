from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.wechat.media.resources import discover_media_assets


class MediaResourceLinkingTests(unittest.TestCase):
    def test_duplicate_resource_rows_with_same_path_resolve_to_one_asset(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'message_resource.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE resource_detail(local_id INTEGER, local_type TEXT, path TEXT)')
            conn.execute('INSERT INTO resource_detail VALUES(?,?,?)', (1, '3', 'same.dat'))
            conn.execute('INSERT INTO resource_detail VALUES(?,?,?)', (2, '3', 'same.dat'))
            conn.commit(); conn.close()
            refs = discover_media_assets(root, account_id='acct-a')
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].media_type, 'image')
        self.assertEqual(refs[0].cache_state, 'missing_local_cache')


if __name__ == '__main__':
    unittest.main()
