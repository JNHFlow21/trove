from __future__ import annotations

import json
import base64
import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.import_job import run_import_job
from trove_core.wechat.importers.wechat_decrypted import msg_table_for
from trove_core.wechat.media.resources import message_media_asset_id
from trove_core.wechat.media.backfill import backfill_message_media_references, message_media_backfill_plan


def create_multimodal_message_source(root: Path) -> Path:
    account = root / 'com.tencent.xinWeChat__wxid_ownerfixture'
    account.mkdir(parents=True)
    media = account / 'media'
    media.mkdir()
    (media / 'private.jpg').write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8z8AARQABywH6f6n6AAAAAElFTkSuQmCC'
    ))
    (media / 'voice.wav').write_bytes(b'RIFF\x10\x00\x00\x00WAVEfixture')
    with sqlite3.connect(account / 'contact.db') as conn:
        conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
        conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
        conn.execute('INSERT INTO contact VALUES (?,?,?,?)', ('wxid_personmedia', 'Private Media', '', ''))
        conn.execute('INSERT INTO contact VALUES (?,?,?,?)', ('room@chatroom', 'Group Media', '', ''))
        conn.execute('INSERT INTO contact VALUES (?,?,?,?)', ('wxid_groupmember', 'Member', '', ''))
        conn.execute('INSERT INTO chatroom_member VALUES (?,?)', ('room@chatroom', 'wxid_groupmember'))
        conn.commit()
    private_table = msg_table_for('wxid_personmedia')
    group_table = msg_table_for('room@chatroom')
    with sqlite3.connect(account / 'message_0.db') as conn:
        conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_personmedia', 1))
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (2, 'room@chatroom', 1))
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (3, 'wxid_groupmember', 0))
        for table in (private_table, group_table):
            conn.execute(f'''CREATE TABLE {table} (
                local_id INTEGER, local_type INTEGER, real_sender_id INTEGER, create_time INTEGER,
                message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                WCDB_CT_message_content BLOB
            )''')
        conn.execute(
            f'INSERT INTO {private_table}(local_id,local_type,real_sender_id,create_time,message_content,packed_info_data) VALUES (?,?,?,?,?,?)',
            (1, 3, 1, 1710000000, '', json.dumps({'file_path': 'media/private.jpg', 'type': 'image'}).encode()),
        )
        conn.execute(
            f'INSERT INTO {private_table}(local_id,local_type,real_sender_id,create_time,message_content,packed_info_data) VALUES (?,?,?,?,?,?)',
            (2, 34, 1, 1710000060, '', json.dumps({'file_path': 'media/voice.wav', 'type': 'voice'}).encode()),
        )
        conn.execute(
            f'INSERT INTO {private_table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
            (3, 43, 1, 1710000120, ''),
        )
        conn.execute(
            f'INSERT INTO {private_table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
            (
                4, 49, 1, 1710000180,
                '<msg><appmsg><type>6</type><title>方案.pdf</title>'
                '<appattach><totallen>42</totallen><fileext>pdf</fileext></appattach></appmsg></msg>',
            ),
        )
        conn.execute(
            f'INSERT INTO {group_table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
            (5, 3, 3, 1710000240, ''),
        )
        conn.commit()
    return account


class MessageMediaRegistrationTests(unittest.TestCase):
    def test_import_registers_all_message_media_and_resource_upgrade_reuses_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'current'
            vault = root / 'vault'
            create_multimodal_message_source(source)

            first = run_import_job(vault, [source], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                assets = list(conn.execute(
                    """SELECT ma.asset_id,ma.modality,ma.media_type,ma.cache_state,m.local_id,m.conversation_type
                         FROM media_assets ma JOIN messages m ON m.citation=ma.citation
                        WHERE ma.asset_id LIKE 'message-asset-%' OR ma.asset_id LIKE 'voice-asset-%'
                        ORDER BY m.local_id"""
                ))
                accepted = conn.execute('SELECT COUNT(*) FROM media_asset_links WHERE accepted=1').fetchone()[0]
                excluded = conn.execute('SELECT COUNT(*) FROM media_asset_links WHERE accepted=0').fetchone()[0]
                image_message = conn.execute('SELECT citation FROM messages WHERE local_id=1').fetchone()
                file_message = conn.execute('SELECT citation FROM messages WHERE local_id=4').fetchone()
                group_message = conn.execute('SELECT citation FROM messages WHERE local_id=5').fetchone()
            second = run_import_job(vault, [source], reset_index=False)

            self.assertEqual(first.status, 'completed')
            self.assertEqual(second.status, 'completed')
            self.assertEqual(len(assets), 5)
            self.assertEqual([(row['modality'], row['media_type']) for row in assets], [
                ('image', 'image'), ('voice', 'voice'), ('video', 'video'), ('file', 'document'), ('image', 'image'),
            ])
            self.assertEqual(assets[0]['asset_id'], message_media_asset_id(image_message['citation'], 'image', 'image'))
            self.assertEqual(assets[0]['cache_state'], 'source_available')
            self.assertEqual((accepted, excluded), (4, 1))
            file_hint = store.media_hints_for_citations([file_message['citation']])[file_message['citation']]
            self.assertEqual((file_hint['type'], file_hint['file_count']), ('file', 1))
            self.assertEqual(store.media_hints_for_citations([group_message['citation']]), {})

            backfill_message_media_references(vault)
            backfilled = backfill_message_media_references(vault)
            self.assertEqual(backfilled['link_result']['assets_upserted'], 0)
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT cache_state FROM media_assets WHERE asset_id=?', (assets[0]['asset_id'],)).fetchone()[0], 'source_available')
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM media_assets WHERE asset_id LIKE 'message-asset-%' OR asset_id LIKE 'voice-asset-%'"
                ).fetchone()[0], 5)

    def test_backfill_repairs_preexisting_unregistered_placeholders_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'current'
            vault = root / 'vault'
            create_multimodal_message_source(source)
            run_import_job(vault, [source], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                conn.execute('DELETE FROM media_asset_links')
                conn.execute('DELETE FROM media_assets')
                conn.commit()

            plan = message_media_backfill_plan(vault)
            first = backfill_message_media_references(vault)
            second = backfill_message_media_references(vault)

            self.assertEqual(plan['eligible_messages'], 5)
            self.assertEqual(plan['missing_assets'], 5)
            self.assertEqual(first['link_result']['assets_upserted'], 5)
            self.assertEqual(second['link_result']['assets_upserted'], 0)
            self.assertEqual(second['link_result']['links_upserted'], 0)
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_assets').fetchone()[0], 5)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_asset_links WHERE accepted=1').fetchone()[0], 4)


if __name__ == '__main__':
    unittest.main()
