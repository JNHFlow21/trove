from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest

from trove_core.reply.context import ReplyContextMessage
from trove_core.reply.generation import ReplyAgentWorkspace
from trove_core.reply.media import ReplyMediaResolver
from trove_core.store.repositories import (
    MediaAssetRecord,
    MediaUnderstandingRecord,
    MultimodalRepository,
    ProviderJobRecord,
    TranscriptRecord,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg'
    'YAAAAAMAASsJTYQAAAAASUVORK5CYII='
)


class ReplyMediaResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = VaultConfig.resolve(
            str(Path(self.temp.name) / 'vault'), env={},
        )
        self.config.ensure()
        self.store = SQLiteStore(self.config.paths.sqlite_path)
        self.store.initialize()
        self.account_id = 'acct-fixture'
        self.conversation_id = 'conv-fixture'
        self.store.upsert_accounts([
            Account(self.account_id, 'Fixture', 'Fixture'),
        ])
        self.store.upsert_conversations([
            Conversation(
                self.conversation_id,
                self.account_id,
                'Private',
                'private',
            ),
        ])
        self.workspace = ReplyAgentWorkspace.for_vault(
            self.config.root, agent_id='fixture',
        )
        self.workspace.ensure_layout()

    def tearDown(self):
        self.temp.cleanup()

    def _message(self, position: int, kind: str) -> tuple[str, ReplyContextMessage]:
        message = Message(
            self.account_id,
            'Fixture',
            self.conversation_id,
            'Private',
            'private',
            'peer',
            'Peer',
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            f'[{kind}]',
            'message_0',
            position,
            direction_hint='incoming',
            content_kind=kind,
        )
        self.store.upsert_messages([message])
        return message.citation, ReplyContextMessage(
            citation=message.citation,
            source_position=position,
            timestamp='2026-01-01T00:00:00Z',
            direction='incoming',
            kind=kind,
            text=f'[{kind}]',
            live_delta=False,
        )

    def test_image_and_sticker_use_bounded_sandbox_attachments(self):
        citation, message = self._message(1, 'sticker')
        source = self.config.root / 'media' / 'fixture.png'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(PNG_1X1)
        digest = hashlib.sha256(PNG_1X1).hexdigest()
        repo = MultimodalRepository(self.store)
        repo.upsert_media_asset(MediaAssetRecord(
            'asset-sticker',
            self.account_id,
            'private_chat',
            citation,
            'image',
            'image',
            citation,
            local_type='sticker',
            content_hash=digest,
            path_ref=str(source.relative_to(self.config.root)),
            cache_state='cached',
        ))
        repo.upsert_media_understanding(MediaUnderstandingRecord(
            content_sha256=digest,
            modality='image',
            caption='一只挥手的小猫表情',
            visible_text='你好',
            model_id='fixture-vision',
            prompt_version='p1',
            confidence=0.9,
            source_citations=[citation],
        ))
        sentinel = self.config.root / 'vectors' / 'unchanged'
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('same', encoding='utf-8')
        before = sentinel.stat().st_mtime_ns

        hints = self.store.media_hints_for_citations([citation])
        result = ReplyMediaResolver(
            self.config, workspace=self.workspace,
        ).resolve(
            [message],
            new_message_citations=[citation],
            hints=hints,
        )

        self.assertEqual(result[0]['modality'], 'sticker')
        self.assertEqual(result[0]['state'], 'understood')
        self.assertEqual(
            result[0]['understanding']['caption'],
            '一只挥手的小猫表情',
        )
        attachment = result[0]['attachment']
        self.assertTrue(
            (self.workspace.workspace / attachment['workspace_path']).is_file(),
        )
        self.assertFalse(result[0]['raw_paths_included'])
        self.assertEqual(sentinel.stat().st_mtime_ns, before)

    def test_voice_uses_only_exact_cloud_transcript(self):
        citation, message = self._message(2, 'voice')
        content_hash = 'a' * 64
        repo = MultimodalRepository(self.store)
        repo.upsert_media_asset(MediaAssetRecord(
            'asset-voice',
            self.account_id,
            'private_chat',
            citation,
            'voice',
            'voice',
            citation,
            content_hash=content_hash,
            cache_state='cached',
        ))
        repo.record_provider_job(ProviderJobRecord(
            job_id='job-cloud',
            asset_id='asset-voice',
            provider='volcengine-asr-flash',
            model='bigmodel:volc.bigasr.auc_turbo',
            job_type='asr',
            status='completed',
            request_hash=content_hash,
            citation=citation,
        ))
        repo.insert_transcript(TranscriptRecord(
            'transcript-cloud',
            'asset-voice',
            citation,
            '明天下午三点见',
            job_id='job-cloud',
            language='zh',
            confidence=0.95,
        ))

        result = ReplyMediaResolver(self.config).resolve(
            [message],
            new_message_citations=[citation],
            hints=self.store.media_hints_for_citations([citation]),
        )

        self.assertEqual(result[0]['state'], 'understood')
        self.assertEqual(
            result[0]['understanding']['audio_transcript'],
            '明天下午三点见',
        )
        self.assertEqual(
            result[0]['understanding']['provider'],
            'volcengine-asr-flash',
        )

    def test_uncached_voice_is_cloud_approval_pending_never_local_asr(self):
        citation, message = self._message(3, 'voice')
        MultimodalRepository(self.store).upsert_media_asset(MediaAssetRecord(
            'asset-voice-pending',
            self.account_id,
            'private_chat',
            citation,
            'voice',
            'voice',
            citation,
            cache_state='cached',
        ))

        result = ReplyMediaResolver(self.config).resolve(
            [message],
            new_message_citations=[citation],
            hints=self.store.media_hints_for_citations([citation]),
        )

        self.assertEqual(result[0]['state'], 'pending_cloud_approval')
        self.assertEqual(result[0]['asr_policy'], 'cloud_only')
        self.assertFalse(result[0]['local_asr_used'])
        self.assertEqual(result[0]['reply_policy'], 'do_not_infer_content')

    def test_file_is_metadata_only_and_does_not_extract(self):
        citation, message = self._message(4, 'file')
        MultimodalRepository(self.store).upsert_media_asset(MediaAssetRecord(
            'asset-file',
            self.account_id,
            'private_chat',
            citation,
            'file',
            'document',
            citation,
            cache_state='metadata_only',
            metadata={
                'file_name': 'proposal.pdf',
                'file_extension': 'pdf',
                'file_size': 123,
            },
        ))

        result = ReplyMediaResolver(self.config).resolve(
            [message],
            new_message_citations=[citation],
            hints=self.store.media_hints_for_citations([citation]),
        )

        self.assertEqual(result[0]['state'], 'metadata_only')
        self.assertEqual(result[0]['document_extract'], 'disabled')
        self.assertEqual(result[0]['metadata']['file_name'], 'proposal.pdf')
        self.assertNotIn('understanding', result[0])

    def test_unindexed_live_media_is_explicit_and_never_guessed(self):
        message = ReplyContextMessage(
            citation='provider://wechat/live/fixture/9',
            source_position=9,
            timestamp='2026-01-01T00:00:00Z',
            direction='incoming',
            kind='image',
            text='[图片]',
            live_delta=True,
        )

        result = ReplyMediaResolver(self.config).resolve(
            [message],
            new_message_citations=[message.citation],
            hints={},
        )

        self.assertEqual(result[0]['state'], 'pending_index')
        self.assertEqual(result[0]['reply_policy'], 'do_not_infer_content')
        self.assertFalse(result[0]['raw_paths_included'])


if __name__ == '__main__':
    unittest.main()
