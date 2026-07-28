from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.knowledge.customer_profile import _evidence_claims, build_customer_profile
from trove_core.store.repositories import EntityRecord, ImageObservationRecord, MediaAssetRecord, MultimodalRepository, ObservationRecord, ProviderJobRecord, TranscriptRecord
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.repositories import WeChatRepository
from trove_core.wechat.models import Account, Conversation, Message


class CustomerProfileProjectionTests(unittest.TestCase):
    def test_evidence_media_hints_are_loaded_once_for_the_whole_batch(self):
        class BatchStore:
            def __init__(self):
                self.hint_calls: list[list[str]] = []

            def chunk_search(self, alias, *, filters, limit):
                return [
                    {'citation': f'trove://fixture/{index}', 'content': f'证据 {index}'}
                    for index in range(5)
                ]

            def media_hints_for_citations(self, citations):
                values = list(citations)
                self.hint_calls.append(values)
                return {citation: {'kind': 'fixture'} for citation in values}

        store = BatchStore()

        claims = _evidence_claims(store, ['批量证据'], source_type='moment', limit=5)

        self.assertEqual(len(claims), 5)
        self.assertEqual(len(store.hint_calls), 1)
        self.assertEqual(len(store.hint_calls[0]), 5)
        self.assertTrue(all(claim.get('media_hint') for claim in claims))

    def test_profile_uses_bidirectional_private_chat_authored_moments_and_interactions(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-sample_contact',
                entity_type='Customer',
                display_name='Sample Contact',
                identifiers={'wechat_id': 'wxid-sample_contact', 'nickname': 'Sample Contact'},
            ))
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('wxid-sample_contact', 'acct-a', 'Sample Contact', 'private'),
                    Conversation('group-1', 'acct-a', '业务群', 'group', member_count=3),
                ],
                [
                    Message('acct-a', 'A', 'wxid-sample_contact', 'Sample Contact', 'private', 'wxid-sample_contact', 'Sample Contact', datetime(2026, 1, 1, tzinfo=timezone.utc), '客户需要预算方案', 's', 1),
                    Message('acct-a', 'A', 'wxid-sample_contact', 'Sample Contact', 'private', 'self', '我', datetime(2026, 1, 2, tzinfo=timezone.utc), '我方确认下周跟进', 's', 2, sent_by_me=True),
                    Message('acct-a', 'A', 'group-1', '业务群', 'group', 'wxid-sample_contact', 'Sample Contact', datetime(2026, 1, 3, tzinfo=timezone.utc), '群里确认试点目标', 's', 3),
                    Message('acct-a', 'A', 'group-1', '业务群', 'group', 'wxid-other', 'Other', datetime(2026, 1, 4, tzinfo=timezone.utc), '其他人的消息不应进入画像', 's', 4),
                ],
            )
            repo.insert_moment_item(moment_id='moment-authored', account_id='acct-a', citation='trove://wechat/acct-a/moment/authored', author_id='wxid-sample_contact', timestamp='2026-01-05T00:00:00Z', text='Sample Contact 发布新校区朋友圈')
            repo.insert_moment_item(moment_id='moment-self', account_id='acct-a', citation='trove://wechat/acct-a/moment/self', author_id='self', timestamp='2026-01-06T00:00:00Z', text='我方朋友圈')
            repo.insert_moment_interaction(interaction_id='mi-1', moment_id='moment-self', account_id='acct-a', citation='trove://wechat/acct-a/moment/self/interaction/mi-1', interaction_type='comment', actor_id='wxid-sample_contact', actor_name='Sample Contact', text='朋友圈互动评论', timestamp='2026-01-06T01:00:00Z')
            store.rebuild_evidence_chunks_for_source_types(['moment'])

            profile = build_customer_profile(store, 'Sample Contact', limit=5)

            chat_values = [row['value'] for row in profile['sections']['chat_evidence']]
            self.assertTrue(any('客户需要预算方案' in value for value in chat_values))
            self.assertTrue(any('我方确认下周跟进' in value for value in chat_values))
            self.assertTrue(any('群里确认试点目标' in value for value in chat_values))
            self.assertFalse(any('其他人的消息' in value for value in chat_values))
            self.assertIn('self', {row.get('direction') for row in profile['sections']['chat_evidence']})
            self.assertIn('peer', {row.get('direction') for row in profile['sections']['chat_evidence']})
            self.assertTrue(any('新校区朋友圈' in row['value'] for row in profile['sections']['moments_authored']))
            self.assertTrue(any('朋友圈互动评论' in row['value'] for row in profile['sections']['interactions']))

    def test_profile_filters_short_chatter_from_timeline_and_business_sections_only(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-noise',
                entity_type='Customer',
                display_name='噪声客户',
                identifiers={'wechat_id': 'wxid-noise', 'nickname': '噪声客户'},
            ))
            repo.add_observation(ObservationRecord('obs-noise', 'customer-noise', 'Need', {'text': '好滴'}, 'active', 0.9, 'trove://operator/noise', 'operator'))
            repo.add_observation(ObservationRecord('obs-need', 'customer-noise', 'Need', {'text': '需要预算审批'}, 'active', 0.9, 'trove://operator/need', 'operator'))
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('wxid-noise', 'acct-a', '噪声客户', 'private')],
                [
                    Message('acct-a', 'A', 'wxid-noise', '噪声客户', 'private', 'wxid-noise', '噪声客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '好滴', 's', 1),
                    Message('acct-a', 'A', 'wxid-noise', '噪声客户', 'private', 'wxid-noise', '噪声客户', datetime(2026, 1, 2, tzinfo=timezone.utc), '需要预算审批', 's', 2),
                ],
            )

            profile = build_customer_profile(store, '噪声客户', limit=5)

            self.assertTrue(any(row['value'] == '好滴' for row in profile['sections']['chat_evidence']))
            self.assertFalse(any(row['value'] == '好滴' for row in profile['sections']['timeline_summary']))
            self.assertFalse(any(row['value'] == '好滴' for row in profile['sections']['needs']))
            self.assertTrue(any('预算审批' in row['value'] for row in profile['sections']['needs']))

    def test_profile_projects_cited_multimodal_sections_and_stays_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育', identifiers={'wechat_id': 'wxid-example_edu'}))
            repo.add_observation(ObservationRecord(observation_id='obs-need', entity_id='customer-1', observation_type='Need', value={'text': '需要预算审批'}, status='active', confidence=0.9, citation='trove://wechat/acct/conv/s1/1', source_type='message'))
            repo.add_observation(ObservationRecord(observation_id='obs-next', entity_id='customer-1', observation_type='NextAction', value={'text': '下周三复盘'}, status='active', confidence=0.85, citation='trove://wechat/acct/conv/s1/2', source_type='message'))
            repo.insert_moment_item(moment_id='moment-1', account_id='acct-a', citation='trove://wechat/acct-a/moment/1', text='示例教育发布新校区动态')
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-v', account_id='acct-a', source_type='message', source_id='m1', modality='voice', media_type='voice', citation='trove://wechat/acct/conv/s1/3', content_hash='a' * 64))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-profile-cloud', asset_id='asset-v', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='a' * 64, citation='trove://wechat/acct/conv/s1/3',
            ))
            repo.insert_transcript(TranscriptRecord(transcript_id='tr-1', asset_id='asset-v', citation='trove://wechat/acct/conv/s1/3#voice', text='示例教育说预算审批需要两周', job_id='job-profile-cloud', confidence=0.8))
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-i', account_id='acct-a', source_type='moment', source_id='sns1', modality='image', media_type='image', citation='trove://wechat/acct-a/moment/1'))
            repo.insert_image_observation(ImageObservationRecord(observation_id='img-1', asset_id='asset-i', citation='trove://wechat/acct-a/moment/1#image', caption='示例教育校区宣传图', confidence=0.8, status='active'))
            store.rebuild_evidence_chunks_for_source_types(['moment', 'transcript', 'image_observation'])
            profile = build_customer_profile(store, '示例教育', limit=3)
            self.assertEqual(profile['type'], 'customer_profile')
            self.assertFalse(profile['raw_content_included'])
            self.assertTrue(profile['sections']['needs'])
            self.assertTrue(profile['sections']['next_actions'])
            self.assertIn('file_exchanges', profile['sections'])
            self.assertIn('timeline_summary', profile['sections'])
            self.assertIn('文件往来', profile['sections'])
            self.assertIn('时间线摘要', profile['sections'])
            self.assertTrue(profile['sections']['moments'])
            self.assertIn('media_hint', profile['sections']['moments'][0])
            self.assertGreaterEqual(profile['sections']['moments'][0]['media_hint']['image_count'], 1)
            self.assertTrue(profile['sections']['voice_transcripts'])
            self.assertTrue(profile['sections']['image_observations'])
            for section, rows in profile['sections'].items():
                self.assertLessEqual(len(rows), 3, section)
                for row in rows:
                    if 'value' in row and section != 'ambiguities':
                        self.assertIn('citations', row)

    def test_profile_multimodal_lookup_uses_chunk_fts_not_source_table_scans(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.insert_moment_item(moment_id='moment-1', account_id='acct-a', citation='trove://wechat/acct-a/moment/1', text='示例教育新校区')
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-v', account_id='acct-a', source_type='message', source_id='m1', modality='voice', media_type='voice', citation='trove://wechat/acct/conv/s1/3'))
            repo.insert_transcript(TranscriptRecord(transcript_id='tr-1', asset_id='asset-v', citation='trove://wechat/acct/conv/s1/3#voice', text='示例教育预算审批'))
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-i', account_id='acct-a', source_type='moment', source_id='sns1', modality='image', media_type='image', citation='trove://wechat/acct-a/moment/1'))
            repo.insert_image_observation(ImageObservationRecord(observation_id='img-1', asset_id='asset-i', citation='trove://wechat/acct-a/moment/1#image', caption='示例教育宣传图', status='active'))
            store.rebuild_evidence_chunks_for_source_types(['moment', 'transcript', 'image_observation'])
            with store.connect() as conn:
                plan_rows = conn.execute(
                    """EXPLAIN QUERY PLAN
                       SELECT e.* FROM chunk_fts f JOIN evidence_chunks e ON e.rowid=f.rowid
                       WHERE chunk_fts MATCH ? AND e.status='active' AND e.source_type=?
                       ORDER BY rank, e.timestamp DESC LIMIT ?""",
                    ('"示例教育"', 'moment', 3),
                ).fetchall()
            plan = ' | '.join(str(row['detail']) for row in plan_rows)
            self.assertIn('VIRTUAL TABLE', plan)
            self.assertNotIn('SCAN moment_items', plan)
            self.assertNotIn('SCAN transcripts', plan)
            self.assertNotIn('SCAN image_observations', plan)

    def test_new_transcript_is_visible_in_profile_before_family_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育', identifiers={'wechat_id': 'wxid-example_edu'}))
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-v', account_id='acct-a', source_type='message', source_id='m1', modality='voice', media_type='voice', citation='trove://wechat/acct/conv/s1/3', content_hash='e' * 64))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-immediate-cloud', asset_id='asset-v', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation='trove://wechat/acct/conv/s1/3',
            ))
            repo.insert_transcript(TranscriptRecord(transcript_id='tr-immediate', asset_id='asset-v', citation='trove://wechat/acct/conv/s1/3#voice', text='示例教育刚刚确认预算审批需要两周', job_id='job-immediate-cloud', confidence=0.8))

            profile = build_customer_profile(store, '示例教育', limit=3)

            transcript_values = [row['value'] for row in profile['sections']['voice_transcripts']]
            self.assertTrue(any('预算审批需要两周' in value for value in transcript_values))

    def test_profile_surfaces_redacted_pending_voice_count_only(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-voice',
                entity_type='Customer',
                display_name='语音客户',
                identifiers={'wechat_id': 'wxid-voice', 'nickname': '语音客户'},
            ))
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('conv-voice', 'acct-a', '语音客户', 'private'),
                    Conversation('group-voice', 'acct-a', '语音群', 'group', member_count=3),
                ],
                [
                    Message('acct-a', 'A', 'conv-voice', '语音客户', 'private', 'wxid-voice', '语音客户', datetime(2026, 1, 1, tzinfo=timezone.utc), 'private incoming voice body', 's', 1, content_kind='voice'),
                    Message('acct-a', 'A', 'conv-voice', '语音客户', 'private', 'self', '我', datetime(2026, 1, 2, tzinfo=timezone.utc), 'private outgoing voice body', 's', 2, sent_by_me=True, content_kind='voice'),
                    Message('acct-a', 'A', 'conv-voice', '语音客户', 'private', 'wxid-voice', '语音客户', datetime(2026, 1, 3, tzinfo=timezone.utc), 'already transcribed voice body', 's', 3, content_kind='voice'),
                    Message('acct-a', 'A', 'group-voice', '语音群', 'group', 'wxid-voice', '语音客户', datetime(2026, 1, 4, tzinfo=timezone.utc), 'group voice body', 's', 4, content_kind='voice'),
                ],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-transcribed',
                account_id='acct-a',
                source_type='message',
                source_id='s-3',
                modality='voice',
                media_type='voice',
                citation='trove://wechat/acct-a/conv-voice/s/3',
                path_ref='/private/raw/voice.amr',
                cache_state='cached',
                content_hash='e' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-transcribed-cloud', asset_id='asset-transcribed', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation='trove://wechat/acct-a/conv-voice/s/3',
            ))
            repo.insert_transcript(TranscriptRecord(
                transcript_id='tr-transcribed',
                asset_id='asset-transcribed',
                job_id='job-transcribed-cloud',
                citation='trove://wechat/acct-a/conv-voice/s/3#voice',
                text='语音客户 已转写',
            ))

            profile = build_customer_profile(store, '语音客户', limit=5)

            pending = profile['sections']['pending_voice']
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]['count'], 2)
            self.assertEqual(pending[0]['scope'], 'private_chat')
            self.assertEqual(pending[0]['transcribe_tool'], 'trove_voice_transcribe_lazy')
            self.assertFalse(pending[0]['raw_content_included'])
            self.assertFalse(pending[0]['raw_paths_included'])
            redacted = json.dumps(pending, ensure_ascii=False)
            self.assertNotIn('private incoming voice body', redacted)
            self.assertNotIn('private outgoing voice body', redacted)
            self.assertNotIn('group voice body', redacted)
            self.assertNotIn('/private/raw/voice.amr', redacted)
            self.assertNotIn('trove://wechat/', redacted)

    def test_profile_observations_include_all_merged_entity_ids(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-account-a',
                entity_type='Customer',
                display_name='客户备注名',
                identifiers={'wechat_username': 'wxid-merge-1', 'nickname': '客户昵称'},
            ))
            repo.upsert_entity(EntityRecord(
                entity_id='customer-account-b',
                entity_type='Customer',
                display_name='群内名',
                identifiers={'wechat_username': 'wxid-merge-1', 'remark': '客户备注名'},
            ))
            repo.add_observation(ObservationRecord(observation_id='obs-need', entity_id='customer-account-a', observation_type='Need', value={'text': '需要校区预算方案'}, status='active', confidence=0.9, citation='trove://contact/a', source_type='contact'))
            repo.add_observation(ObservationRecord(observation_id='obs-pain', entity_id='customer-account-b', observation_type='Objection', value={'text': '担心审批周期太长'}, status='active', confidence=0.85, citation='trove://contact/b', source_type='contact'))

            profile = build_customer_profile(store, '客户备注名', limit=5)

            self.assertCountEqual(profile['resolved_entity']['entity_ids'], ['customer-account-a', 'customer-account-b'])
            self.assertTrue(any('预算方案' in row['value'] for row in profile['sections']['needs']))
            self.assertTrue(any('审批周期' in row['value'] for row in profile['sections']['objections']))


if __name__ == '__main__':
    unittest.main()
