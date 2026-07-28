from __future__ import annotations
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.wechat.importers import JsonlExportImporter, SQLiteArchiveImporter

class RealImportAdapterTests(unittest.TestCase):
    def test_jsonl_export_importer_maps_messages(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'messages.jsonl'
            p.write_text(json.dumps({'account_id':'a','account_label':'A','conversation_id':'c','conversation_title':'C','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'hello','local_id':1}, ensure_ascii=False) + '\n', encoding='utf-8')
            accounts, conversations, messages = JsonlExportImporter(p).load()
            self.assertEqual(len(accounts), 1)
            self.assertEqual(len(conversations), 1)
            self.assertEqual(messages[0].content, 'hello')

    def test_sqlite_archive_importer_discovers_message_table(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'archive.db'
            with sqlite3.connect(p) as conn:
                conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, create_time INTEGER, talker TEXT, sender TEXT)')
                conn.execute('INSERT INTO messages(content,create_time,talker,sender) VALUES (?,?,?,?)', ('预算审批没过', 1710000000, 'room@chatroom', 'alice'))
                conn.commit()
            accounts, conversations, messages = SQLiteArchiveImporter(p, account_id='acct').load()
            self.assertEqual(accounts[0].account_id, 'acct')
            self.assertEqual(conversations[0].type, 'group')
            self.assertIn('预算', messages[0].content)
