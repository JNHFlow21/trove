from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trove_core.runtime import build_search_engine
from trove_core.search.context import ContextService
from trove_core.search.query import SearchRequest
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository, ProviderJobRecord, TranscriptRecord, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


class VoiceMediaHintTests(unittest.TestCase):
    def test_voice_hint_reports_cached_pending_and_unavailable_states(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', '私聊', 'private')],
                [
                    Message('acct-a', 'A', 'conv-a', '私聊', 'private', 'u1', '客户', ts, 'voice pending needle', 'message_0', 1, content_kind='voice'),
                    Message('acct-a', 'A', 'conv-a', '私聊', 'private', 'u1', '客户', ts, 'voice unavailable needle', 'message_0', 2, content_kind='voice'),
                    Message('acct-a', 'A', 'conv-a', '私聊', 'private', 'u1', '客户', ts, 'voice cached needle', 'message_0', 3, content_kind='voice'),
                ],
            )
            repo = MultimodalRepository(store)
            citations = [f'trove://wechat/acct-a/conv-a/message_0/{i}' for i in (1, 2, 3)]
            repo.upsert_media_asset(MediaAssetRecord('asset-pending', 'acct-a', 'private_chat', citations[0], 'voice', 'voice', citations[0], cache_state='cached'))
            repo.upsert_media_asset(MediaAssetRecord('asset-unavailable', 'acct-a', 'private_chat', citations[1], 'voice', 'voice', citations[1], cache_state='metadata_only'))
            repo.upsert_media_asset(MediaAssetRecord('asset-cached', 'acct-a', 'private_chat', citations[2], 'voice', 'voice', citations[2], cache_state='cached', content_hash='a' * 64))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-cached-cloud', asset_id='asset-cached', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='a' * 64, citation=citations[2],
            ))
            repo.insert_transcript(TranscriptRecord('tr-cached', 'asset-cached', citations[2], 'already transcribed', job_id='job-cached-cloud', status='active'))

            hints = store.media_hints_for_citations(citations)

            self.assertEqual(hints[citations[0]]['modality'], 'voice')
            self.assertEqual(hints[citations[0]]['transcript_state'], 'pending')
            self.assertEqual(hints[citations[0]]['transcribe_tool'], 'trove_voice_transcribe_lazy')
            self.assertTrue(hints[citations[0]]['available'])
            self.assertEqual(hints[citations[1]]['transcript_state'], 'unavailable')
            self.assertFalse(hints[citations[1]]['available'])
            self.assertEqual(hints[citations[2]]['transcript_state'], 'cached')
            self.assertFalse(hints[citations[2]]['raw_paths_included'])

    def test_search_and_context_surface_voice_hint_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', '私聊', 'private')],
                [Message('acct-a', 'A', 'conv-a', '私聊', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), 'voicehintneedle', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-voice-search',
                'acct-a',
                'private_chat',
                citation,
                'voice',
                'voice',
                citation,
                cache_state='cached',
            ))

            with patch('trove_core.media_pipeline.ProviderFactory.create_asr', side_effect=AssertionError('provider must not be constructed')):
                response = build_search_engine(cfg).search(SearchRequest('voicehintneedle', limit=1, semantic='off', include_media_hints=True)).to_dict()
                context = ContextService(store).fetch(citation)

            self.assertEqual(response['results'][0]['media_hint']['modality'], 'voice')
            self.assertEqual(response['results'][0]['media_hint']['transcript_state'], 'pending')
            self.assertEqual(context['messages'][0]['media_hint']['modality'], 'voice')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM provider_jobs').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
