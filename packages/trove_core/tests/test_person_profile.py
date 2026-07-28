from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

from trove_core.knowledge.person_profile import (
    PersonProfileClaimError,
    build_person_profile,
    propose_person_profile_claims,
)
from trove_core.store.repositories import (
    EntityRecord,
    MediaAssetRecord,
    MultimodalRepository,
    ProviderJobRecord,
    TranscriptRecord,
    WeChatRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message
from trove_core.wechat.decrypt.manifest import write_account_identity


def _fixture(store: SQLiteStore) -> list[str]:
    repo = MultimodalRepository(store)
    repo.upsert_entity(EntityRecord(
        entity_id='person-sample_contact',
        entity_type='Person',
        display_name='Sample Contact',
        identifiers={'wechat_id': 'wxid-sample_contact', 'nickname': 'Sample Contact'},
    ))
    citations = [f'trove://wechat/acct-a/conv-sample_contact/message_0/{index}' for index in range(1, 9)]
    WeChatRepository(store).replace_fixture(
        [Account('acct-a', 'A', 'A')],
        [
            Conversation('conv-sample_contact', 'acct-a', 'Sample Contact', 'private'),
            Conversation('group-friends', 'acct-a', '朋友群', 'group', member_count=8),
        ],
        [
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'wxid-sample_contact', 'Sample Contact', datetime(2025, 1, 1, 2, tzinfo=timezone.utc), '我从小在示例城市甲长大，生日是5月20日', 'message_0', 1),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'self', '我', datetime(2025, 1, 1, 3, tzinfo=timezone.utc), '谢谢你上次帮我介绍朋友', 'message_0', 2, sent_by_me=True),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'wxid-sample_contact', 'Sample Contact', datetime(2025, 1, 2, 2, tzinfo=timezone.utc), '我希望今年能换工作，更看重成长空间', 'message_0', 3),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'self', '我', datetime(2025, 1, 2, 4, tzinfo=timezone.utc), '我明天把资料发给你', 'message_0', 4, sent_by_me=True),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'wxid-sample_contact', 'Sample Contact', datetime(2026, 1, 1, 2, tzinfo=timezone.utc), '压力大的时候我一般先自己想清楚', 'message_0', 5),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'self', '我', datetime(2026, 1, 1, 3, tzinfo=timezone.utc), '需要我帮忙就直接说', 'message_0', 6, sent_by_me=True),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'wxid-sample_contact', 'Sample Contact', datetime(2026, 2, 1, 2, tzinfo=timezone.utc), '[voice]', 'message_0', 7, content_kind='voice'),
            Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'self', '我', datetime(2026, 2, 1, 3, tzinfo=timezone.utc), '[appmsg/location] 我发起了位置共享', 'message_0', 8, sent_by_me=True, content_kind='appmsg'),
            Message('acct-a', 'A', 'group-friends', '朋友群', 'group', 'wxid-sample_contact', 'Sample Contact', datetime(2026, 2, 2, 2, tzinfo=timezone.utc), '我来协调周末聚会', 'message_0', 9),
        ],
    )
    repo.insert_moment_item(
        moment_id='moment-sample_contact', account_id='acct-a', citation='trove://wechat/acct-a/moment/sample_contact',
        author_id='wxid-sample_contact', timestamp='2026-01-15T00:00:00Z', text='开始新的工作挑战',
    )
    repo.insert_moment_interaction(
        interaction_id='interaction-sample_contact', moment_id='moment-sample_contact', account_id='acct-a',
        citation='trove://wechat/acct-a/moment/sample_contact/interaction/1', interaction_type='comment',
        actor_id='self', actor_name='我', text='加油', timestamp='2026-01-15T01:00:00Z',
    )
    repo.upsert_media_asset(MediaAssetRecord(
        asset_id='voice-sample_contact', account_id='acct-a', source_type='message', source_id='voice-7',
        modality='voice', media_type='voice', citation=citations[6], cache_state='cached',
        content_hash='e' * 64,
    ))
    repo.record_provider_job(ProviderJobRecord(
        job_id='job-voice-sample_contact', asset_id='voice-sample_contact', provider='volcengine-asr-flash',
        model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
        request_hash='e' * 64, citation=citations[6],
    ))
    repo.insert_transcript(TranscriptRecord(
        transcript_id='transcript-sample_contact', asset_id='voice-sample_contact', citation=citations[6] + '#voice',
        text='我最近最需要的是稳定一点的生活节奏', job_id='job-voice-sample_contact', confidence=0.9,
    ))
    return citations


class PersonProfileTests(unittest.TestCase):
    def test_duplicate_archive_events_are_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'vault.sqlite')
            _fixture(store)
            duplicate = [
                Message(
                    'acct-b', 'B', 'conv-sample_contact-copy', 'Sample Contact', 'private',
                    'wxid-sample_contact-copy', 'Sample Contact', datetime(2025, 1, 1, 2, tzinfo=timezone.utc),
                    '新版解析文本不影响事件去重', 'message_0', 1,
                ),
                Message(
                    'acct-b', 'B', 'group-friends-copy', '朋友群', 'group',
                    'wxid-sample_contact-copy', 'Sample Contact', datetime(2026, 2, 2, 2, tzinfo=timezone.utc),
                    '新版解析文本不影响事件去重', 'message_0', 9,
                ),
            ]
            WeChatRepository(store).replace_fixture(
                [Account('acct-b', 'B', 'B')],
                [
                    Conversation('conv-sample_contact-copy', 'acct-b', 'Sample Contact', 'private'),
                    Conversation('group-friends-copy', 'acct-b', '朋友群', 'group', member_count=8),
                ],
                duplicate,
            )
            resolved = {
                'entity_id': 'person-sample_contact',
                'entity_ids': ['person-sample_contact', 'person-sample_contact-copy'],
                'display_name': 'Sample Contact',
                'primary_user_id': 'wxid-sample_contact',
                'conversation_ids': ['conv-sample_contact', 'conv-sample_contact-copy'],
                'sender_ids': ['wxid-sample_contact', 'wxid-sample_contact-copy'],
            }
            with patch(
                'trove_core.knowledge.person_profile.resolve_customer',
                return_value={'resolved': resolved, 'candidates': []},
            ):
                profile = build_person_profile(store, 'Sample Contact', evidence_limit=10)

            self.assertEqual(profile['data_coverage']['messages']['total'], 9)
            self.assertEqual(profile['data_coverage']['messages']['private'], 8)
            self.assertEqual(profile['data_coverage']['messages']['group'], 1)
            self.assertEqual(profile['data_coverage']['conversations']['total'], 2)

    def test_analysis_cap_is_reported_as_truncated_not_full_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'vault.sqlite')
            _fixture(store)

            with patch('trove_core.knowledge.person_profile.MAX_SCOPED_MESSAGES', 3):
                profile = build_person_profile(store, 'Sample Contact', evidence_limit=3)

            coverage = profile['data_coverage']['messages']
            self.assertEqual(coverage['analyzed'], 3)
            self.assertTrue(coverage['analysis_cap_applied'])
            self.assertFalse(coverage['scope_complete'])
            self.assertFalse(profile['evidence_projection']['full_scope_analyzed'])
            gap = next(item for item in profile['data_gaps'] if item['kind'] == 'message_analysis_truncated')
            self.assertEqual(gap['analyzed'], 3)
            self.assertTrue(gap['more_messages_exist'])

    def test_profile_uses_complete_scope_and_returns_person_relationship_action_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'vault.sqlite')
            _fixture(store)

            profile = build_person_profile(store, 'Sample Contact', evidence_limit=10)

            self.assertEqual(profile['type'], 'person_profile')
            self.assertEqual(profile['schema_version'], 'person-profile/v1')
            self.assertEqual(profile['data_coverage']['messages']['total'], 9)
            self.assertEqual(profile['data_coverage']['messages']['private'], 8)
            self.assertEqual(profile['data_coverage']['messages']['group'], 1)
            self.assertEqual(profile['data_coverage']['messages']['peer'], 5)
            self.assertEqual(profile['data_coverage']['messages']['self'], 4)
            self.assertEqual(profile['data_coverage']['media']['voice']['total'], 1)
            self.assertEqual(profile['data_coverage']['media']['voice']['understood'], 1)
            self.assertGreaterEqual(profile['relationship_model']['interaction_dynamics']['sessions'], 3)
            self.assertTrue(profile['relationship_model']['timeline'])
            self.assertTrue(profile['evidence_candidates']['identity_and_life_context'])
            self.assertTrue(profile['evidence_candidates']['values_and_tradeoffs'])
            self.assertTrue(profile['evidence_candidates']['goals_and_needs'])
            action_types = {item['action_type'] for item in profile['relationship_actions']}
            self.assertIn('remember_key_detail', action_types)
            self.assertIn('close_commitment_loop', action_types)
            self.assertIn('express_gratitude', action_types)
            self.assertIn('offer_help', action_types)
            self.assertIn('big_five', profile['scientific_framework']['lenses'])
            self.assertFalse(profile['raw_paths_included'])
            self.assertFalse(profile['clinical_diagnosis_included'])

    def test_agent_profile_claims_are_structured_cited_reviewable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'vault.sqlite')
            citations = _fixture(store)
            claims = [
                {
                    'dimension': 'life_context',
                    'evidence_class': 'fact',
                    'statement': '她明确表示自己从小在示例城市甲长大。',
                    'citations': [citations[0]],
                    'scope': 'self_disclosure',
                    'confidence': 0.95,
                },
                {
                    'dimension': 'situational_patterns',
                    'evidence_class': 'hypothesis',
                    'statement': '在压力和不确定情境下，她可能倾向先独立处理，再接受外部支持。',
                    'citations': [citations[2], citations[4], citations[5]],
                    'scope': 'private_chat',
                    'confidence': 0.68,
                    'counterevidence_reviewed': True,
                    'alternative_explanations': ['这些表达可能只反映当时的工作压力，而不是稳定倾向。'],
                },
            ]

            first = propose_person_profile_claims(store, 'Sample Contact', claims)
            second = propose_person_profile_claims(store, 'Sample Contact', claims)
            profile = build_person_profile(store, 'Sample Contact', evidence_limit=5)

            self.assertEqual(first['written'], 2)
            self.assertEqual(second['written'], 0)
            self.assertEqual(second['unchanged'], 2)
            self.assertEqual(len(profile['person_model']['life_context']), 1)
            self.assertEqual(len(profile['person_model']['situational_patterns']), 1)
            hypothesis = profile['person_model']['situational_patterns'][0]
            self.assertEqual(hypothesis['review_status'], 'needs_review')
            self.assertEqual(hypothesis['evidence_class'], 'hypothesis')
            self.assertEqual(len(hypothesis['citations']), 3)
            self.assertTrue(hypothesis['alternative_explanations'])
            self.assertTrue(hypothesis['counterevidence_reviewed'])

            transcript_items = [
                item
                for values in profile['evidence_candidates'].values()
                for item in values
                if item['source_type'] == 'transcript'
            ]
            self.assertTrue(transcript_items)
            self.assertEqual(profile['data_coverage']['understood_source_items']['transcript'], 1)
            self.assertEqual(profile['relationship_model']['group_contexts']['total_groups_spoken_in'], 1)
            self.assertIn('analysis_protocol', profile)
            self.assertIn('questions_for_user', profile)

    def test_hypothesis_requires_repeated_evidence_and_alternative_explanation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'vault.sqlite')
            citations = _fixture(store)

            with self.assertRaises(PersonProfileClaimError):
                propose_person_profile_claims(store, 'Sample Contact', [{
                    'dimension': 'attachment_related_patterns',
                    'evidence_class': 'hypothesis',
                    'statement': '她可能在亲密关系中需要较多确定性。',
                    'citations': [citations[0]],
                    'scope': 'private_chat',
                    'confidence': 0.7,
                    'counterevidence_reviewed': True,
                }])

            with self.assertRaisesRegex(PersonProfileClaimError, 'evidence citation was not found'):
                propose_person_profile_claims(store, 'Sample Contact', [{
                    'dimension': 'life_context',
                    'evidence_class': 'fact',
                    'statement': 'This fixture claim has no supporting row.',
                    'citations': ['trove://missing/evidence'],
                    'scope': 'fixture',
                    'confidence': 0.5,
                }])

    def test_unknown_appmsg_direction_is_never_attributed_to_the_person(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='person-unknown-direction', entity_type='Person', display_name='Unknown Direction',
                identifiers={'wechat_id': 'wxid-unknown-direction', 'nickname': 'Unknown Direction'},
            ))
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-unknown', 'acct-a', 'Unknown Direction', 'private')],
                [Message(
                    'acct-a', 'A', 'conv-unknown', 'Unknown Direction', 'private',
                    'unresolved-sender', 'Unknown', datetime(2026, 3, 1, tzinfo=timezone.utc),
                    '[appmsg]', 'message_0', 1, content_kind='appmsg', direction_hint='unknown',
                    normalized_payload={
                        'appmsg_type': 1,
                        'normalized_type': 'note',
                        'parse_status': 'parsed',
                        'fields': {'title': '我希望明年换工作'},
                        'display_text': '[appmsg/note] 我希望明年换工作',
                        'source_hash': hashlib.sha256(b'unknown-appmsg').hexdigest(),
                        'parser_version': 'fixture.v1',
                    },
                )],
            )

            profile = build_person_profile(store, 'Unknown Direction', evidence_limit=10)
            appmsg_items = [
                item
                for item in profile['evidence_candidates']['goals_and_needs']
                if item['source_type'] == 'appmsg_payload'
            ]

            self.assertEqual(len(appmsg_items), 1)
            self.assertEqual(appmsg_items[0]['direction'], 'unknown')
            self.assertEqual(appmsg_items[0]['subject_scope'], 'relationship_or_self')
            self.assertEqual(profile['relationship_model']['timeline'][0]['unknown_direction'], 1)

    def test_only_proven_operator_moment_interactions_are_labeled_self(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            _fixture(store)
            account_dir = vault / 'sources' / 'wechat-integrated-decrypted' / 'current' / 'account-0123456789abcdef'
            write_account_identity(
                account_dir, account_ref_hash='0123456789abcdef', own_wxid='wxid_ownerfixture',
            )
            repo = MultimodalRepository(store)
            repo.insert_moment_interaction(
                interaction_id='interaction-owner', moment_id='moment-sample_contact', account_id='acct-a',
                citation='trove://wechat/acct-a/moment/sample_contact/interaction/owner', interaction_type='comment',
                actor_id='wxid_ownerfixture', actor_name='我', text='谢谢你的分享',
                timestamp='2026-01-15T02:00:00Z',
            )
            repo.insert_moment_interaction(
                interaction_id='interaction-third-party', moment_id='moment-sample_contact', account_id='acct-a',
                citation='trove://wechat/acct-a/moment/sample_contact/interaction/third-party', interaction_type='comment',
                actor_id='wxid_thirdparty', actor_name='第三方', text='谢谢你的分享',
                timestamp='2026-01-15T03:00:00Z',
            )

            profile = build_person_profile(store, 'Sample Contact', evidence_limit=50)
            interaction_items = [
                item
                for values in profile['evidence_candidates'].values()
                for item in values
                if item['source_type'] == 'moment_interaction'
            ]

            self.assertEqual({item['citations'][0] for item in interaction_items}, {
                'trove://wechat/acct-a/moment/sample_contact/interaction/owner',
            })
            self.assertEqual({item['direction'] for item in interaction_items}, {'self'})
            self.assertEqual(profile['data_coverage']['operator_person_moment_interactions'], 1)
            self.assertEqual(profile['data_coverage']['unattributed_public_interactions_excluded'], 2)


if __name__ == '__main__':
    unittest.main()
