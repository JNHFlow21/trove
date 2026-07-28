from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from trove_cli.main import main


class ScopeCliContractTests(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, json.loads(buf.getvalue())

    def test_scope_list_and_scoped_search_commands(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            contact_db = root / 'contact.db'
            with sqlite3.connect(contact_db) as conn:
                conn.execute('CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, signature TEXT, big_head_url TEXT)')
                conn.execute('INSERT INTO contact VALUES(?,?,?,?,?,?)', ('wxid-a', '示例教育', '示例', 'example_edu', '预算负责人', 'avatar'))
            self.assertEqual(self.run_cli(['--vault', str(vault), 'import-contacts', str(contact_db), '--account-id', 'acct-a', '--json'])[1]['imported_contacts'], 1)

            sns_db = root / 'sns.db'
            with sqlite3.connect(sns_db) as conn:
                conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
                conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-m1', 'wxid-a', '<TimelineObject><id>m1</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>朋友圈预算</contentDesc></TimelineObject>', b''))
            self.assertEqual(self.run_cli(['--vault', str(vault), 'import-moments', str(sns_db), '--account-id', 'acct-a', '--json'])[1]['imported_moments'], 1)

            fav_db = root / 'favorite.db'
            with sqlite3.connect(fav_db) as conn:
                conn.execute('CREATE TABLE favorite_item(fav_id TEXT, update_time TEXT, title TEXT, content TEXT)')
                conn.execute('INSERT INTO favorite_item VALUES(?,?,?,?)', ('f1', '2026-01-02', '收藏预算', '收藏夹知识'))
            self.assertEqual(self.run_cli(['--vault', str(vault), 'import-favorites', str(fav_db), '--account-id', 'acct-a', '--json'])[1]['imported_favorites'], 1)

            code, scope = self.run_cli(['--vault', str(vault), 'scope-status', '--json'])
            self.assertEqual(code, 0)
            self.assertGreaterEqual(scope['families']['contact'], 1)
            self.assertGreaterEqual(scope['families']['moment'], 1)
            self.assertGreaterEqual(scope['families']['favorite'], 1)
            self.assertTrue(self.run_cli(['--vault', str(vault), 'list-contacts', '--json'])[1]['contacts'])
            self.assertTrue(self.run_cli(['--vault', str(vault), 'list-moments', '--json'])[1]['moments'])
            self.assertTrue(self.run_cli(['--vault', str(vault), 'list-favorites', '--json'])[1]['favorites'])
            fav_search = self.run_cli(['--vault', str(vault), 'search', '收藏夹', '--source-type', 'favorite', '--json'])[1]
            self.assertTrue(fav_search['results'])
            self.assertTrue(all(r['source_type'] == 'favorite' for r in fav_search['results']))


if __name__ == '__main__':
    unittest.main()
