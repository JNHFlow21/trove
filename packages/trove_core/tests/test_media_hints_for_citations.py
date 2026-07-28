from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.store.repositories import (
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MultimodalRepository,
    ProviderJobRecord,
    TranscriptRecord,
    WeChatRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class MediaHintsForCitationsTests(unittest.TestCase):
    def _store_and_repo(self, tmp: str) -> tuple[SQLiteStore, MultimodalRepository]:
        store = SQLiteStore(Path(tmp) / 'index' / 'trove.sqlite')
        repo = MultimodalRepository(store)
        return store, repo

    def _seed_private_voice_message(self, store: SQLiteStore, citation: str) -> None:
        parts = citation.split('/')
        account_id = parts[3]
        conversation_id = parts[4]
        shard_id = parts[5]
        local_id = int(parts[6])
        WeChatRepository(store).replace_fixture(
            [Account(account_id, 'A', 'A')],
            [Conversation(conversation_id, account_id, 'Private', 'private')],
            [Message(
                account_id,
                'A',
                conversation_id,
                'Private',
                'private',
                'sender-a',
                'Sender',
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                '[voice]',
                shard_id,
                local_id,
                content_kind='voice',
            )],
        )

    def _voice_asset(
        self,
        repo: MultimodalRepository,
        *,
        asset_id: str,
        citation: str,
        cache_state: str,
        content_hash: str | None = None,
    ) -> None:
        repo.upsert_media_asset(MediaAssetRecord(
            asset_id=asset_id,
            account_id='acct-a',
            source_type='message',
            source_id=asset_id,
            modality='voice',
            media_type='voice',
            citation=citation,
            cache_state=cache_state,
            content_hash=content_hash,
            metadata={'fixture': True},
        ))

    def test_voice_hint_cached_when_active_transcript_exists(self):
        with tempfile.TemporaryDirectory() as d:
            store, repo = self._store_and_repo(d)
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            self._seed_private_voice_message(store, citation)
            self._voice_asset(
                repo, asset_id='asset-voice-cached', citation=citation,
                cache_state='metadata_only', content_hash='a' * 64,
            )
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-voice-cached', asset_id='asset-voice-cached',
                provider='volcengine-asr-flash', model='bigmodel:volc.bigasr.auc_turbo',
                job_type='asr', status='completed', request_hash='a' * 64, citation=citation,
            ))
            repo.insert_transcript(TranscriptRecord(
                transcript_id='transcript-a',
                asset_id='asset-voice-cached',
                citation=citation,
                text='fixture transcript',
                job_id='job-voice-cached',
                status='active',
            ))

            hint = store.media_hints_for_citations([citation])[citation]

            self.assertEqual(hint, {
                'citation': citation,
                'asset_id': 'asset-voice-cached',
                'modality': 'voice',
                'media_type': 'voice',
                'available': False,
                'cache_state': 'metadata_only',
                'transcript_state': 'cached',
                'transcribe_tool': 'trove_voice_transcribe_lazy',
                'raw_paths_included': False,
            })

    def test_voice_hint_pending_uses_parent_and_link_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            store, repo = self._store_and_repo(d)
            parent = 'trove://wechat/acct-a/conv-a/message_0/2'
            chunk = f'{parent}#chunk-0'
            self._seed_private_voice_message(store, parent)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-voice-pending',
                account_id='acct-a',
                source_type='message',
                source_id='asset-voice-pending',
                modality='voice',
                media_type='voice',
                citation='trove://media/asset-voice-pending',
                cache_state='normalized',
                metadata={'fixture': True},
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                link_id='link-voice-pending',
                asset_id='asset-voice-pending',
                account_id='acct-a',
                source_type='message',
                source_citation=parent,
                scope_type='private_chat',
                accepted=True,
                reason='fixture',
            ))
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO transcripts(transcript_id,asset_id,job_id,citation,text,language,confidence,duration_seconds,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    ('transcript-inactive', 'asset-voice-pending', None, parent, 'old transcript', None, 0.0, 0.0, 'superseded', '2026-01-01T00:00:00Z'),
                )
                conn.execute(
                    """INSERT INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    ('chunk-a', chunk, parent, 'acct-a', 'A', 'message', 'conv-a', 'Voice', '客户', '2026-01-01T00:00:00Z', 'voice chunk', 0, '{}', 'active', '2026-01-01T00:00:00Z'),
                )
                conn.commit()

            hint = store.media_hints_for_citations([chunk])[chunk]

            self.assertEqual(hint['citation'], parent)
            self.assertEqual(hint['asset_id'], 'asset-voice-pending')
            self.assertEqual(hint['modality'], 'voice')
            self.assertTrue(hint['available'])
            self.assertEqual(hint['cache_state'], 'normalized')
            self.assertEqual(hint['transcript_state'], 'pending')
            self.assertEqual(hint['transcribe_tool'], 'trove_voice_transcribe_lazy')
            self.assertFalse(hint['raw_paths_included'])

    def test_voice_hint_unavailable_for_metadata_only_without_transcript(self):
        with tempfile.TemporaryDirectory() as d:
            store, repo = self._store_and_repo(d)
            citation = 'trove://wechat/acct-a/conv-a/message_0/3'
            self._seed_private_voice_message(store, citation)
            self._voice_asset(repo, asset_id='asset-voice-unavailable', citation=citation, cache_state='metadata_only')

            hint = store.media_hints_for_citations([citation])[citation]

            self.assertEqual(hint['transcript_state'], 'unavailable')
            self.assertFalse(hint['available'])
            self.assertEqual(hint['transcribe_tool'], 'trove_voice_transcribe_lazy')
            self.assertFalse(hint['raw_paths_included'])

    def test_voice_hint_is_not_emitted_for_group_voice_even_with_asset(self):
        with tempfile.TemporaryDirectory() as d:
            store, repo = self._store_and_repo(d)
            citation = 'trove://wechat/acct-a/room@chatroom/message_0/4'
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('room@chatroom', 'acct-a', 'Group', 'group')],
                [Message(
                    'acct-a',
                    'A',
                    'room@chatroom',
                    'Group',
                    'group',
                    'sender-a',
                    'Sender',
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    '[voice]',
                    'message_0',
                    4,
                    content_kind='voice',
                )],
            )
            self._voice_asset(repo, asset_id='asset-voice-group', citation=citation, cache_state='cached')

            self.assertEqual(store.media_hints_for_citations([citation]), {})

    def test_image_video_hint_shape_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            store, repo = self._store_and_repo(d)
            parent = 'trove://wechat/acct-a/moment/moment-1'
            image_citation = f'{parent}#image-0'
            video_citation = f'{parent}#video-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image',
                account_id='acct-a',
                source_type='moment',
                source_id='image-source',
                modality='image',
                media_type='image',
                citation=image_citation,
                cache_state='cached',
                metadata={'fixture': True},
            ))
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-video',
                account_id='acct-a',
                source_type='moment',
                source_id='video-source',
                modality='video',
                media_type='video',
                citation=video_citation,
                cache_state='metadata_only',
                metadata={'fixture': True},
            ))

            hint = store.media_hints_for_citations([parent])[parent]

            self.assertEqual(hint, {
                'type': 'media',
                'image_count': 1,
                'video_count': 1,
                'media_count': 2,
                'available_count': 1,
                'items': [
                    {
                        'citation': image_citation,
                        'asset_id': 'asset-image',
                        'modality': 'image',
                        'media_type': 'image',
                        'available': True,
                        'cache_state': 'cached',
                        'fetch_tool': 'trove_media_fetch',
                        'raw_paths_included': False,
                    },
                    {
                        'citation': video_citation,
                        'asset_id': 'asset-video',
                        'modality': 'video',
                        'media_type': 'video',
                        'available': False,
                        'cache_state': 'metadata_only',
                        'fetch_tool': 'trove_media_fetch',
                        'raw_paths_included': False,
                    },
                ],
                'fetch_tool': 'trove_media_fetch',
                'raw_paths_included': False,
            })


if __name__ == '__main__':
    unittest.main()
