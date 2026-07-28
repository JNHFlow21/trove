from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trove_core.media_pipeline import enqueue_media_jobs
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.import_job import _voice_media_references_for_messages, run_import_job
from trove_core.wechat.importers.wechat_decrypted import msg_table_for
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.resources import MediaReference
from trove_core.wechat.models import Message


def _create_voice_account(root: Path) -> Path:
    account = root / 'com.tencent.xinWeChat__wxid_ownerfixture'
    account.mkdir(parents=True)
    with sqlite3.connect(account / 'contact.db') as conn:
        conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
        conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
        conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_privatefixture', 'Private Fixture', '', ''))
        conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('room@chatroom', 'Group Fixture', '', ''))
        conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_groupmember', 'Group Member', '', ''))
        conn.execute('INSERT INTO chatroom_member(chatroom,member) VALUES (?,?)', ('room@chatroom', 'wxid_groupmember'))
        conn.commit()
    private_table = msg_table_for('wxid_privatefixture')
    group_table = msg_table_for('room@chatroom')
    with sqlite3.connect(account / 'message_0.db') as conn:
        conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_privatefixture', 1))
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (2, 'room@chatroom', 1))
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (3, 'wxid_groupmember', 0))
        for table in (private_table, group_table):
            conn.execute(f'''CREATE TABLE {table} (
                local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
                real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
                download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
                message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
            )''')
        conn.execute(f'INSERT INTO {private_table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)', (1, 34, 1, 1710000000, ''))
        conn.execute(f'INSERT INTO {group_table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)', (2, 34, 3, 1710000060, ''))
        conn.commit()
    return account


class WeChatImportVoiceAssetTests(unittest.TestCase):
    def test_import_registers_private_voice_asset_idempotently_and_skips_group_link(self):
        with tempfile.TemporaryDirectory() as d:
            source_root = Path(d) / 'current'
            vault = Path(d) / 'vault'
            _create_voice_account(source_root)

            first = run_import_job(vault, [source_root], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                first_asset_ids = [
                    row['asset_id']
                    for row in conn.execute("SELECT asset_id FROM media_assets WHERE asset_id LIKE 'voice-asset-%' ORDER BY asset_id")
                ]
                first_jobs = list(conn.execute(
                    """SELECT mj.asset_id,mj.job_type,mj.status,m.conversation_type
                       FROM media_jobs mj
                       JOIN media_assets ma ON ma.asset_id=mj.asset_id
                       JOIN messages m ON m.citation=ma.citation
                       WHERE mj.job_type='voice_transcribe'
                       ORDER BY mj.asset_id"""
                ))

            second = run_import_job(vault, [source_root], reset_index=False)

            self.assertEqual(first.status, 'completed')
            self.assertEqual(second.status, 'completed')
            with store.connect() as conn:
                private_msg = conn.execute("SELECT citation FROM messages WHERE content_kind='voice' AND conversation_type='private'").fetchone()
                group_msg = conn.execute("SELECT citation FROM messages WHERE content_kind='voice' AND conversation_type='group'").fetchone()
                self.assertIsNotNone(private_msg)
                self.assertIsNotNone(group_msg)
                phase_b_assets = list(conn.execute(
                    """SELECT asset_id,citation,source_id,modality,media_type,cache_state
                       FROM media_assets
                       WHERE asset_id LIKE 'voice-asset-%'
                       ORDER BY asset_id"""
                ))
                private_asset = conn.execute(
                    """SELECT ma.asset_id,ma.citation,ma.source_id,ma.modality,ma.media_type,ma.cache_state,mal.accepted
                       FROM media_assets ma
                       JOIN media_asset_links mal ON mal.asset_id=ma.asset_id
                       WHERE ma.asset_id LIKE 'voice-asset-%' AND ma.citation=?""",
                    (private_msg['citation'],),
                ).fetchone()
                accepted_group_links = conn.execute(
                    'SELECT COUNT(*) AS n FROM media_asset_links WHERE source_citation=? AND accepted=1',
                    (group_msg['citation'],),
                ).fetchone()['n']

            self.assertEqual(len(first_asset_ids), 2)
            self.assertEqual(len(first_jobs), 1)
            self.assertEqual(first_jobs[0]['job_type'], 'voice_transcribe')
            self.assertEqual(first_jobs[0]['status'], 'pending')
            self.assertEqual(first_jobs[0]['conversation_type'], 'private')
            self.assertEqual([row['asset_id'] for row in phase_b_assets], first_asset_ids)
            self.assertIsNotNone(private_asset)
            self.assertEqual(private_asset['citation'], private_msg['citation'])
            self.assertEqual(private_asset['source_id'], private_msg['citation'])
            self.assertEqual(private_asset['modality'], 'voice')
            self.assertEqual(private_asset['media_type'], 'voice')
            self.assertEqual(private_asset['cache_state'], 'metadata_only')
            self.assertEqual(private_asset['accepted'], 1)
            self.assertEqual(accepted_group_links, 0)

    def test_import_empty_media_delta_passes_empty_ids_and_does_no_queue_work(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            reports: list[dict[str, object]] = []

            def capture_enqueue(store: SQLiteStore, **kwargs):
                report = enqueue_media_jobs(store, **kwargs)
                reports.append(report)
                return report

            with patch('trove_core.wechat.import_job.enqueue_media_jobs', side_effect=capture_enqueue) as enqueue:
                result = run_import_job(vault, [], reset_index=False)

            self.assertEqual(result.status, 'completed')
            enqueue.assert_called_once()
            self.assertEqual(enqueue.call_args.kwargs['asset_ids'], ())
            self.assertIsNotNone(enqueue.call_args.kwargs['asset_ids'])
            self.assertEqual(reports[0]['seen'], 0)
            self.assertEqual(reports[0]['queued'], 0)
            self.assertEqual(reports[0]['metrics']['sql_statements'], 0)
            self.assertEqual(reports[0]['metrics']['commits'], 0)

    def test_changed_private_voice_asset_is_requeued_durably(self):
        with tempfile.TemporaryDirectory() as d:
            source_root = Path(d) / 'current'
            vault = Path(d) / 'vault'
            account = _create_voice_account(source_root)
            private_table = msg_table_for('wxid_privatefixture')

            first = run_import_job(vault, [source_root], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                private_asset_id = conn.execute(
                    """SELECT ma.asset_id
                       FROM media_assets ma JOIN messages m ON m.citation=ma.citation
                       WHERE ma.modality='voice' AND m.conversation_type='private'"""
                ).fetchone()['asset_id']
                conn.execute('DELETE FROM media_jobs WHERE asset_id=?', (private_asset_id,))
                conn.execute(
                    "UPDATE media_assets SET cache_state='missing_local_cache' WHERE asset_id=?",
                    (private_asset_id,),
                )
                conn.commit()
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(
                    f'UPDATE {private_table} SET create_time=? WHERE local_id=?',
                    (1711000000, 1),
                )
                conn.commit()

            queued_asset_ids: list[tuple[str, ...]] = []

            def capture_enqueue(store: SQLiteStore, **kwargs):
                queued_asset_ids.append(tuple(kwargs['asset_ids']))
                return enqueue_media_jobs(store, **kwargs)

            with patch('trove_core.wechat.import_job.enqueue_media_jobs', side_effect=capture_enqueue):
                changed = run_import_job(vault, [source_root], reset_index=False)

            self.assertEqual(first.status, 'completed')
            self.assertEqual(changed.status, 'completed')
            self.assertEqual(len(queued_asset_ids), 1)
            self.assertIn(private_asset_id, queued_asset_ids[0])
            with store.connect() as conn:
                job = conn.execute(
                    'SELECT asset_id,job_type,status FROM media_jobs WHERE asset_id=?',
                    (private_asset_id,),
                ).fetchone()
            self.assertIsNotNone(job)
            self.assertEqual(job['job_type'], 'voice_transcribe')
            self.assertEqual(job['status'], 'pending')

    def test_voice_reference_marks_known_absent_resource_as_missing_local_cache(self):
        message = Message(
            'acct-a',
            'A',
            'conv-a',
            'A private',
            'private',
            'sender-a',
            'Sender',
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            'fixture voice reference media/missing.amr',
            'message_0',
            7,
            content_kind='voice',
        )

        refs = _voice_media_references_for_messages([message])

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].citation, message.citation)
        self.assertEqual(refs[0].modality, 'voice')
        self.assertEqual(refs[0].media_type, 'voice')
        self.assertEqual(refs[0].cache_state, 'missing_local_cache')
        self.assertTrue(refs[0].asset_id.startswith('voice-asset-'))

    def test_linker_rejects_orphan_voice_reference(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            report = MediaLinker(MultimodalRepository(store)).link_references([
                MediaReference(
                    asset_id='voice-asset-orphan',
                    account_id='acct-a',
                    source_type='message',
                    source_id='orphan',
                    modality='voice',
                    media_type='voice',
                    citation='trove://wechat/acct-a/media/orphan',
                )
            ])

            self.assertEqual(report.accepted_links, 0)
            self.assertEqual(report.excluded_links, 1)
            with store.connect() as conn:
                accepted = conn.execute('SELECT accepted FROM media_asset_links WHERE asset_id=?', ('voice-asset-orphan',)).fetchone()
            self.assertIsNotNone(accepted)
            self.assertEqual(accepted['accepted'], 0)


if __name__ == '__main__':
    unittest.main()
