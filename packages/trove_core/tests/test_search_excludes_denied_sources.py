from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone

from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter, msg_table_for


class SearchExcludesDeniedSourcesTests(unittest.TestCase):
    def make_account(self, root: Path) -> Path:
        acct = root / 'acct'
        acct.mkdir()
        with sqlite3.connect(acct / 'contact.db') as conn:
            conn.execute('CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
            conn.execute('INSERT INTO contact VALUES(?,?,?,?)', ('wxid-human', 'Human', '', ''))
            conn.execute('INSERT INTO contact VALUES(?,?,?,?)', ('gh_public', 'Public Account', '', ''))
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
            for rowid, username in [(1, 'wxid-human'), (2, 'gh_public'), (3, 'notifymessage'), (4, 'filehelper')]:
                conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (rowid, username, 1))
                table = msg_table_for(username)
                conn.execute(f'CREATE TABLE {table} (local_id INTEGER, real_sender_id INTEGER, create_time INTEGER, message_content TEXT, compress_content BLOB, WCDB_CT_message_content BLOB)')
                text = '允许的人类关系内容' if username == 'wxid-human' else f'排除源内容 {username}'
                conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (1, rowid, 1710000000, text))
        return acct

    def test_decrypted_importer_excludes_public_system_filehelper(self):
        with tempfile.TemporaryDirectory() as d:
            acct = self.make_account(Path(d))
            accounts, conversations, messages = WeChatDecryptedAccountImporter(acct).load()
            self.assertEqual(len(conversations), 1)
            self.assertEqual(len(messages), 1)
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            store.upsert_accounts(accounts); store.upsert_conversations(conversations); store.upsert_messages(messages)
            self.assertTrue(HyperSearch(store).search(SearchRequest('允许的人类')).results)
            self.assertFalse(HyperSearch(store).search(SearchRequest('排除源内容')).results)

    def test_scope_rebuild_is_conservative_for_user_titles(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            store.upsert_accounts([Account('acct-a', 'A', 'A')])
            store.upsert_conversations([
                Conversation('wxid-human', 'acct-a', '客户 Service 讨论', 'private'),
                Conversation('gh_public', 'acct-a', '公众号标题', 'private'),
            ])
            store.upsert_messages([
                Message('acct-a', 'A', 'wxid-human', '客户 Service 讨论', 'private', 'wxid-human', 'Human', datetime(2026, 1, 1, tzinfo=timezone.utc), '真实关系内容', 's', 1),
                Message('acct-a', 'A', 'gh_public', '公众号标题', 'private', 'gh_public', 'Public', datetime(2026, 1, 1, tzinfo=timezone.utc), '公众号内容', 's', 1),
            ])
            with store.connect() as conn:
                for row in conn.execute('SELECT citation FROM messages'):
                    conn.execute(
                        'INSERT INTO vector_entries(citation,provider,dimensions,vector_json) VALUES(?,?,?,?)',
                        (row['citation'], 'test', 1, '[1]'),
                    )
                conn.commit()
            report = store.purge_excluded_scope()
            self.assertEqual(report['purged_conversations'], 1)
            self.assertTrue(HyperSearch(store).search(SearchRequest('真实关系内容')).results)
            self.assertFalse(HyperSearch(store).search(SearchRequest('公众号内容')).results)
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM vector_entries').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
