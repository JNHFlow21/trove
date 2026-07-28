from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.media_pipeline import enqueue_media_jobs
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class MediaQueueScopeTests(unittest.TestCase):
    def test_voice_queue_only_accepts_private_message_citations(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('conv-private', 'acct-a', '私聊', 'private'),
                    Conversation('conv-group', 'acct-a', '群聊', 'group', member_count=3),
                ],
                [
                    Message('acct-a', 'A', 'conv-private', '私聊', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '私聊语音', 'message_0', 1),
                    Message('acct-a', 'A', 'conv-group', '群聊', 'group', 'u2', '群友', datetime(2026, 1, 1, tzinfo=timezone.utc), '群语音', 'message_0', 2),
                ],
            )
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord('voice-private', 'acct-a', 'message', 'p', 'voice', 'voice', 'trove://wechat/acct-a/conv-private/message_0/1'))
            repo.upsert_media_asset(MediaAssetRecord('voice-group', 'acct-a', 'message', 'g', 'voice', 'voice', 'trove://wechat/acct-a/conv-group/message_0/2'))
            repo.upsert_media_asset(MediaAssetRecord('voice-channel', 'acct-a', 'channel', 'c', 'voice', 'voice', 'trove://wechat/acct-a/media/channel/1'))
            repo.upsert_media_asset(MediaAssetRecord('voice-orphan', 'acct-a', 'message', 'o', 'voice', 'voice', 'trove://wechat/acct-a/media/orphan/1'))

            report = enqueue_media_jobs(store, modalities={'voice'})

            self.assertEqual(report['queued'], 1)
            with store.connect() as conn:
                rows = list(conn.execute('SELECT ma.asset_id,mj.job_type,mj.status FROM media_jobs mj JOIN media_assets ma ON ma.asset_id=mj.asset_id'))
            self.assertEqual([(r['asset_id'], r['job_type'], r['status']) for r in rows], [('voice-private', 'voice_transcribe', 'pending')])

    def test_existing_non_private_voice_jobs_are_marked_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord('voice-orphan', 'acct-a', 'message', 'o', 'voice', 'voice', 'trove://wechat/acct-a/media/orphan/1'))
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO media_jobs(job_id,asset_id,job_type,status,retry_count,error_code,last_duration_ms,created_at,updated_at)
                       VALUES('job-orphan','voice-orphan','voice_transcribe','pending',0,NULL,0,datetime('now'),datetime('now'))"""
                )
                conn.commit()

            report = enqueue_media_jobs(store, modalities={'voice'})

            self.assertEqual(report['queued'], 0)
            with store.connect() as conn:
                row = conn.execute('SELECT status,error_code FROM media_jobs WHERE job_id=?', ('job-orphan',)).fetchone()
            self.assertEqual((row['status'], row['error_code']), ('skipped', 'out_of_scope'))

    def test_local_id_without_full_coordinates_cannot_cross_match_private_message(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('conv-private', 'acct-a', 'private', 'private'),
                    Conversation('conv-group', 'acct-a', 'group', 'group', member_count=3),
                ],
                [
                    Message('acct-a', 'A', 'conv-private', 'private', 'private', 'u1', 'one', datetime(2026, 1, 1, tzinfo=timezone.utc), 'private', 'message_0', 7),
                    Message('acct-a', 'A', 'conv-group', 'group', 'group', 'u2', 'two', datetime(2026, 1, 1, tzinfo=timezone.utc), 'group', 'message_0', 7),
                ],
            )
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                'voice-ambiguous', 'acct-a', 'message', 'legacy', 'voice', 'voice',
                'trove://wechat/acct-a/media/legacy/7',
                metadata={'message_local_id': 7},
            ))

            report = enqueue_media_jobs(store, modalities={'voice'}, asset_ids=['voice-ambiguous'])

            self.assertEqual(report['queued'], 0)
            with store.connect() as conn:
                self.assertIsNone(conn.execute(
                    "SELECT 1 FROM media_jobs WHERE asset_id='voice-ambiguous'",
                ).fetchone())


if __name__ == '__main__':
    unittest.main()
