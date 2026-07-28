from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_provider_wechat.source.current_importer import WeChatDecryptedAccountImporter, decode_content, msg_table_for
from trove_protocol.provider import ProviderManifest
from trove_provider_wechat import create_provider
from trove_core.wechat.import_job import run_import_job
from trove_core.store.sqlite_store import SQLiteStore

class WeChatDecryptedImporterTests(unittest.TestCase):
    def test_provider_discovers_and_reads_current_account_through_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            account = self.make_account_dir(root)
            package = Path(__file__).resolve().parents[1] / 'trove_provider_wechat'
            manifest = ProviderManifest.from_dict(json.loads((package / 'manifest.json').read_text()))
            provider = create_provider(manifest, source_root=root)
            account_id = WeChatDecryptedAccountImporter(account).account_id

            listed = provider.accounts()
            first = provider.invoke('read', {
                'operation': 'changes', 'account_id': account_id, 'cursor': None,
            })
            replay = provider.invoke('read', {
                'operation': 'changes', 'account_id': account_id,
                'cursor': first['change_cursor'],
            })

            self.assertEqual([item['account_id'] for item in listed], [account_id])
            self.assertEqual(len(first['records']), 2)
            self.assertTrue(all(item['citation'].startswith(f'trove://wechat/{account_id}/') for item in first['records']))
            self.assertEqual(replay['records'], [])
            self.assertTrue(replay['replayed'])

    def test_zstd_message_content_is_decompressed_before_appmsg_parsing(self):
        compressed = base64.b64decode(
            'KLUv/QRYzQEAtAI8bXNnPjxhcHB0eXBlPjU8Lzx0aXRsZT5ac3Rk5Y2h54mHPC88Ly9tc2c+BABTEbzmBHBxbZEDNLw5sg=='
        )

        decoded = decode_content(compressed)

        self.assertTrue(decoded.startswith('<msg><appmsg>'))
        self.assertIn('Zstd卡片', decoded)

    def test_appmsg_filtered_load_includes_zstd_rows_with_extended_local_type(self):
        with tempfile.TemporaryDirectory() as directory:
            account = self.make_account_dir(Path(directory))
            compressed = base64.b64decode(
                'KLUv/QRYzQEAtAI8bXNnPjxhcHB0eXBlPjU8Lzx0aXRsZT5ac3Rk5Y2h54mHPC88Ly9tc2c+BABTEbzmBHBxbZEDNLw5sg=='
            )
            table = msg_table_for('room@chatroom')
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
                    (99, 244813135921, 2, 1710000999, compressed),
                )
                conn.commit()

            _, _, messages = WeChatDecryptedAccountImporter(account).load(content_kinds={'appmsg'})

            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].content_kind, 'appmsg')
            self.assertIn('Zstd卡片', messages[0].content)

    def make_account_dir(self, root: Path, *, dirname: str = 'com.tencent.xinWeChat__wxid_demo', include_self_sender: bool = False) -> Path:
        acct = root / dirname
        acct.mkdir()
        with sqlite3.connect(acct / 'contact.db') as conn:
            conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
            conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
            conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('room@chatroom', '项目群', '', ''))
            conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('alice', 'Alice', '', ''))
            if include_self_sender:
                conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_demo', '我', '', ''))
            conn.execute('INSERT INTO chatroom_member(chatroom,member) VALUES (?,?)', ('room@chatroom', 'alice'))
            conn.commit()
        table = msg_table_for('room@chatroom')
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
            conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'room@chatroom', 1))
            conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (2, 'alice', 0))
            if include_self_sender:
                conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (3, 'wxid_demo', 0))
            conn.execute(f'''CREATE TABLE {table} (
                local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
                real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
                download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
                message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
            )''')
            conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (1, 2, 1710000000, '真实微信导入测试'))
            conn.execute(f'INSERT INTO {table}(local_id,local_type,real_sender_id,create_time,message_content,compress_content) VALUES (?,?,?,?,?,?)', (2, 3, 2, 1710000060, '', b'\\x00\\xff\\x01'))
            if include_self_sender:
                conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (3, 3, 1710000120, '我发送的消息'))
            conn.commit()
        return acct

    def test_importer_maps_name2id_msg_table(self):
        with tempfile.TemporaryDirectory() as d:
            acct = self.make_account_dir(Path(d))
            importer = WeChatDecryptedAccountImporter(acct)
            accounts, conversations, messages = importer.load()
            self.assertEqual(len(accounts), 1)
            self.assertEqual(conversations[0].title, '项目群')
            self.assertEqual(conversations[0].member_count, 1)
            self.assertEqual(messages[0].sender_name, 'Alice')
            self.assertEqual(messages[0].content, '真实微信导入测试')
            media = next(m for m in messages if m.local_id == 2)
            self.assertEqual(media.content_kind, 'image')
            self.assertEqual(media.content, '[image]')
            self.assertEqual(messages[0].direction, 'incoming')
            self.assertEqual(importer.waterline_snapshot(), importer.last_waterline_updates)

    def test_importer_marks_self_only_when_real_sender_matches_own_wxid(self):
        with tempfile.TemporaryDirectory() as d:
            acct = self.make_account_dir(Path(d), include_self_sender=True)
            _, _, messages = WeChatDecryptedAccountImporter(acct).load()
            by_local = {m.local_id: m for m in messages}
            self.assertFalse(by_local[1].sent_by_me)
            self.assertEqual(by_local[1].direction, 'incoming')
            self.assertTrue(by_local[3].sent_by_me)
            self.assertEqual(by_local[3].direction, 'outgoing')

    def test_importer_uses_unknown_direction_when_own_wxid_is_not_known(self):
        with tempfile.TemporaryDirectory() as d:
            acct = self.make_account_dir(Path(d), dirname='com.tencent.xinWeChat__fixture_account')
            _, _, messages = WeChatDecryptedAccountImporter(acct).load()
            self.assertEqual({m.direction for m in messages}, {'unknown'})
            self.assertFalse(any(m.sent_by_me for m in messages))

    def test_import_job_persists_unknown_direction_without_own_wxid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'current'
            root.mkdir()
            self.make_account_dir(root, dirname='com.tencent.xinWeChat__fixture_account')
            vault = Path(d) / 'vault'
            result = run_import_job(vault, [root], reset_index=True)
            self.assertEqual(result.status, 'completed')
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                directions = {row['direction'] for row in conn.execute('SELECT direction FROM messages')}
                sent_by_me = {row['sent_by_me'] for row in conn.execute('SELECT sent_by_me FROM messages')}
                chunk = conn.execute("SELECT chunk_citation FROM evidence_chunks WHERE source_type='message' LIMIT 1").fetchone()
            self.assertEqual(directions, {'unknown'})
            self.assertEqual(sent_by_me, {0})
            self.assertIsNotNone(chunk)
            self.assertEqual(store.evidence_by_citation(chunk['chunk_citation'])['direction'], 'unknown')

    def test_import_job_accepts_decrypted_current_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'current'
            root.mkdir()
            self.make_account_dir(root)
            vault = Path(d) / 'vault'
            result = run_import_job(vault, [root], reset_index=True)
            self.assertEqual(result.status, 'completed')
            self.assertEqual(result.waterlines_updated, 1)
            self.assertEqual(SQLiteStore(vault / 'index' / 'trove.sqlite').counts()['messages'], 2)
            with SQLiteStore(vault / 'index' / 'trove.sqlite').connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM sync_state').fetchone()[0], 1)

    def test_import_job_persists_safe_appmsg_payload_without_raw_xml(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'current'
            root.mkdir()
            account = self.make_account_dir(root)
            table = msg_table_for('room@chatroom')
            raw = (
                '<msg><appmsg><type>5</type><title>安全应用标题</title><des>可搜索摘要</des>'
                '<url>https://example.com/private/item?token=do-not-store</url></appmsg></msg>'
            )
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
                    (9, 49, 2, 1710000180, raw),
                )
                conn.commit()

            importer = WeChatDecryptedAccountImporter(account)
            _, _, messages = importer.load()
            appmsg = next(message for message in messages if message.local_id == 9)
            self.assertEqual(appmsg.content_kind, 'appmsg')
            self.assertIn('安全应用标题', appmsg.content)
            self.assertNotIn('<appmsg', appmsg.content)
            self.assertIsNotNone(appmsg.normalized_payload)

            vault = Path(d) / 'vault'
            result = run_import_job(vault, [root], reset_index=True)
            self.assertEqual(result.status, 'completed')
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                row = conn.execute(
                    """SELECT m.content,p.normalized_type,p.parse_status,p.normalized_json,p.source_hash
                         FROM messages m JOIN message_payloads p ON p.citation=m.citation
                        WHERE m.local_id=9"""
                ).fetchone()
            self.assertEqual((row['normalized_type'], row['parse_status']), ('link', 'parsed'))
            self.assertIn('安全应用标题', row['content'])
            persisted = row['content'] + row['normalized_json']
            self.assertNotIn('do-not-store', persisted)
            self.assertNotIn('/private/item', persisted)
            self.assertEqual(len(row['source_hash']), 64)
            self.assertTrue(store.exact_search('安全应用标题'))

            _, _, app_messages_only = importer.load(content_kinds={'appmsg'})
            self.assertEqual([message.local_id for message in app_messages_only], [9])

            second = run_import_job(vault, [root], reset_index=False)
            self.assertEqual(second.status, 'completed')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM message_payloads').fetchone()[0], 1)
