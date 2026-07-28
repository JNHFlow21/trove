from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.importers.favorites import FavoritesImporter


class FavoritesImporterTests(unittest.TestCase):
    def test_favorite_mixed_text_and_media_imports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'favorite.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE favorite_item(fav_id TEXT, update_time TEXT, title TEXT, content TEXT, media_path TEXT)')
            conn.execute('INSERT INTO favorite_item VALUES(?,?,?,?,?)', ('fav1', '2026-01-01', '报价资料', '客户收藏的方案', 'missing.png'))
            conn.commit(); conn.close()
            importer = FavoritesImporter(db, account_id='acct-a')
            rows = importer.load()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].title, '报价资料')
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            self.assertEqual(importer.import_to_store(repo), 1)
            with repo.store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM favorites').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
