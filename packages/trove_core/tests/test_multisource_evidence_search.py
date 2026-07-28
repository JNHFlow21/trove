from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.search.context import ContextService
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.repositories import EntityRecord, ImageObservationRecord, MediaAssetRecord, MultimodalRepository, ObservationRecord, ProviderJobRecord, TranscriptRecord
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message
from datetime import datetime, timezone


class MultisourceEvidenceSearchTests(unittest.TestCase):
    def seed(self, root: Path) -> SQLiteStore:
        store = SQLiteStore(root / 'vault.sqlite')
        repo = MultimodalRepository(store)
        store.upsert_accounts([Account('acct-a', 'A', 'A')])
        store.upsert_conversations([Conversation('conv-private', 'acct-a', '客户私聊', 'private')])
        store.upsert_messages([Message('acct-a','A','conv-private','客户私聊','private','sender-a','Alice',datetime(2026,1,1,tzinfo=timezone.utc),'私聊预算审批','s',1)])
        repo.upsert_entity(EntityRecord('customer-1', 'Customer', '示例教育', {'wechat_id': 'wxid-a'}))
        repo.add_observation(ObservationRecord('obs-contact', 'customer-1', 'remark', {'text': '联系人预算负责人'}, 'active', 0.9, 'trove://wechat/acct-a/contact/contact-a', 'contact'))
        repo.insert_moment_item(moment_id='moment-1', account_id='acct-a', citation='trove://wechat/acct-a/moment/moment-1', author_id='wxid-a', timestamp='2026-01-02T00:00:00Z', text='朋友圈新校区预算消息')
        repo.insert_favorite(favorite_id='fav-1', account_id='acct-a', citation='trove://wechat/acct-a/favorite/fav-1', timestamp='2026-01-03T00:00:00Z', title='收藏方案', text='收藏夹知识库预算资料')
        repo.upsert_media_asset(MediaAssetRecord(
            'asset-v','acct-a','private_chat','msg-1','voice','voice',
            'trove://wechat/acct-a/chat/conv-private/s/2', content_hash='a' * 64,
        ))
        repo.record_provider_job(ProviderJobRecord(
            job_id='job-cloud-voice', asset_id='asset-v', provider='volcengine-asr-flash',
            model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
            request_hash='a' * 64, citation='trove://wechat/acct-a/chat/conv-private/s/2',
        ))
        repo.insert_transcript(TranscriptRecord(
            'tr-1','asset-v','trove://wechat/acct-a/chat/conv-private/s/2#voice',
            '语音里提到预算审批', job_id='job-cloud-voice',
        ))
        repo.upsert_media_asset(MediaAssetRecord('asset-i','acct-a','moment','moment-1','image','image','trove://wechat/acct-a/moment/moment-1'))
        repo.insert_image_observation(ImageObservationRecord('img-1','asset-i','trove://wechat/acct-a/moment/moment-1#image','图片显示预算表', visible_text='预算表'))
        store.rebuild_evidence_chunks()
        return store

    def test_search_and_context_cover_all_evidence_families(self):
        with tempfile.TemporaryDirectory() as d:
            store = self.seed(Path(d))
            resp = HyperSearch(store).search(SearchRequest('预算', limit=10))
            source_types = {row.source_type for row in resp.results}
            self.assertIn('message', source_types)
            self.assertIn('contact', source_types)
            self.assertIn('moment', source_types)
            self.assertIn('favorite', source_types)
            self.assertIn('transcript', source_types)
            self.assertIn('image_observation', source_types)
            fav = HyperSearch(store).search(SearchRequest('预算', source_type='favorite', limit=5))
            self.assertTrue(fav.results)
            self.assertTrue(all(r.source_type == 'favorite' for r in fav.results))
            ctx = ContextService(store).fetch(fav.results[0].citation)
            self.assertIsNotNone(ctx['evidence'])
            self.assertEqual(ctx['evidence']['source_type'], 'favorite')


if __name__ == '__main__':
    unittest.main()
