from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.store.schema import SCHEMA_VERSION, TABLES
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class MultimodalSchemaTests(unittest.TestCase):
    def test_empty_vault_creates_v2_multimodal_tables(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.initialize()
            with store.connect() as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')")}
            for name in ['media_assets', 'provider_jobs', 'transcripts', 'image_observations', 'sns_cache_mappings', 'media_understanding', 'moment_items', 'favorites', 'entities', 'observations', 'relationships']:
                self.assertIn(TABLES[name], tables)
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)

    def test_v1_vault_migrates_without_losing_searchable_messages(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trove.sqlite'
            conn = sqlite3.connect(path)
            conn.executescript('''
                CREATE TABLE accounts(account_id TEXT PRIMARY KEY,label TEXT NOT NULL,display_name TEXT NOT NULL);
                CREATE TABLE conversations(conversation_id TEXT NOT NULL,account_id TEXT NOT NULL,title TEXT NOT NULL,type TEXT NOT NULL,member_count INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(account_id,conversation_id));
                CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,citation TEXT NOT NULL UNIQUE,account_id TEXT NOT NULL,account_label TEXT NOT NULL,conversation_id TEXT NOT NULL,conversation_title TEXT NOT NULL,conversation_type TEXT NOT NULL,sender_id TEXT NOT NULL,sender_name TEXT NOT NULL,timestamp TEXT NOT NULL,content TEXT NOT NULL,shard_id TEXT NOT NULL,local_id INTEGER NOT NULL,sent_by_me INTEGER NOT NULL,source_type TEXT NOT NULL,direction TEXT NOT NULL,UNIQUE(account_id,conversation_id,shard_id,local_id));
                CREATE VIRTUAL TABLE message_fts USING fts5(citation UNINDEXED,content,sender_name,conversation_title,tokenize='unicode61');
            ''')
            conn.execute('INSERT INTO accounts VALUES(?,?,?)', ('acct-a', 'work', 'Work'))
            conn.execute('INSERT INTO conversations VALUES(?,?,?,?,?)', ('conv-a', 'acct-a', '示例教育', 'private', 1))
            conn.execute('''INSERT INTO messages(citation,account_id,account_label,conversation_id,conversation_title,conversation_type,sender_id,sender_name,timestamp,content,shard_id,local_id,sent_by_me,source_type,direction)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', ('trove://wechat/acct-a/conv-a/s1/1', 'acct-a', 'work', 'conv-a', '示例教育', 'private', 'u1', '客户', '2026-01-01T00:00:00Z', '价格太高，需要审批', 's1', 1, 0, 'message', 'incoming'))
            conn.execute('INSERT INTO message_fts(rowid,citation,content,sender_name,conversation_title) VALUES(?,?,?,?,?)', (1, 'trove://wechat/acct-a/conv-a/s1/1', '价格太高，需要审批', '客户', '示例教育'))
            conn.commit(); conn.close()

            store = SQLiteStore(path)
            store.initialize()
            self.assertEqual(store.counts()['messages'], 1)
            self.assertTrue(store.exact_search('价格太高'))
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            with store.connect() as conn:
                self.assertIsNotNone(conn.execute("SELECT name FROM sqlite_master WHERE name='provider_jobs'").fetchone())
                self.assertEqual(conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()[0], 'trigram/v1')
                self.assertIsNotNone(conn.execute("SELECT name FROM sqlite_master WHERE name='chunk_fts'").fetchone())

    def test_initialize_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            store.initialize()
            store.initialize()
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            self.assertEqual(store.counts()['messages'], 0)


if __name__ == '__main__':
    unittest.main()
