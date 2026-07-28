from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.store.repositories import EntityRecord, MultimodalRepository, ObservationRecord, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class EntityResolutionTests(unittest.TestCase):
    def test_private_conversation_exact_beats_group_alias(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('wxid-private', 'acct-a', 'Sample Contact示例联系人甲', 'private'),
                    Conversation('group-1', 'acct-a', '业务群', 'group', member_count=5),
                ],
                [
                    Message('acct-a', 'A', 'wxid-private', 'Sample Contact示例联系人甲', 'private', 'wxid-private', 'Sample Contact示例联系人甲', datetime(2026, 1, 1, tzinfo=timezone.utc), '私聊消息', 's', 1),
                    Message('acct-a', 'A', 'group-1', '业务群', 'group', 'wxid-group-sample_contact', 'Sample Contact示例联系人甲', datetime(2026, 1, 2, tzinfo=timezone.utc), '群聊同名消息', 's', 2),
                ],
            )

            result = resolve_customer(store, 'Sample Contact示例联系人甲')

            self.assertFalse(result['ambiguous'])
            self.assertEqual(result['resolved']['primary_user_id'], 'wxid-private')
            self.assertIn('private_conversation_title_exact', result['resolved']['match_reasons'])

    def test_ambiguous_nickname_returns_merge_candidates_not_auto_merge(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            for idx in [1, 2]:
                eid = f'customer-{idx}'
                repo.upsert_entity(EntityRecord(entity_id=eid, entity_type='Customer', display_name='示例人物丁'))
                repo.add_observation(ObservationRecord(observation_id=f'obs-{idx}', entity_id=eid, observation_type='nickname', value={'text': '示例人物丁'}, status='active', confidence=0.8, citation=f'trove://contact/{idx}', source_type='contact'))
            result = resolve_customer(store, '示例人物丁')
            self.assertTrue(result['ambiguous'])
            self.assertEqual(len(result['merge_candidates']), 2)
            self.assertIsNone(result['resolved'])

    def test_user_id_is_primary_key_for_wechat_aliases(self):
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
            repo.add_observation(ObservationRecord(observation_id='obs-a', entity_id='customer-account-a', observation_type='nickname', value={'text': '客户昵称'}, status='active', confidence=0.8, citation='trove://contact/a', source_type='contact'))
            repo.add_observation(ObservationRecord(observation_id='obs-b', entity_id='customer-account-b', observation_type='remark', value={'text': '客户备注名'}, status='active', confidence=0.8, citation='trove://contact/b', source_type='contact'))
            result = resolve_customer(store, '客户备注名')
            self.assertFalse(result['ambiguous'])
            self.assertEqual(result['resolved']['primary_user_id'], 'wxid-merge-1')
            self.assertCountEqual(result['resolved']['entity_ids'], ['customer-account-a', 'customer-account-b'])

    def test_user_id_sibling_is_merged_when_only_one_alias_matches(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-account-a',
                entity_type='Customer',
                display_name='客户备注名',
                identifiers={'wechat_username': 'wxid-merge-1'},
            ))
            repo.upsert_entity(EntityRecord(
                entity_id='customer-account-b',
                entity_type='Customer',
                display_name='群内名',
                identifiers={'wechat_username': 'wxid-merge-1', 'nickname': '不匹配昵称'},
            ))

            result = resolve_customer(store, '客户备注名')

            self.assertFalse(result['ambiguous'])
            self.assertEqual(result['resolved']['primary_user_id'], 'wxid-merge-1')
            self.assertIn('identifier_exact', result['resolved']['match_reasons'])
            self.assertIn('user_id_sibling', result['resolved']['match_reasons'])
            self.assertCountEqual(result['resolved']['entity_ids'], ['customer-account-a', 'customer-account-b'])

    def test_exact_identifier_lookup_is_not_limited_by_global_entity_rank(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            for idx in range(600):
                repo.upsert_entity(EntityRecord(
                    entity_id=f'noise-{idx}',
                    entity_type='Customer',
                    display_name=f'Noise {idx}',
                    identifiers={'nickname': f'noise-{idx}'},
                    confidence=1.0,
                ))
            repo.upsert_entity(EntityRecord(
                entity_id='customer-target',
                entity_type='Customer',
                display_name='目标客户',
                identifiers={'remark': '目标客户', 'wechat_username': 'wxid-target'},
                confidence=0.7,
            ))

            result = resolve_customer(store, '目标客户')

            self.assertEqual(result['resolved']['entity_id'], 'customer-target')
            self.assertEqual(result['resolved']['primary_user_id'], 'wxid-target')

    def test_contact_entity_wins_and_binds_private_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-contact',
                entity_type='Customer',
                display_name='示例客户甲',
                identifiers={'remark': '示例客户甲', 'wechat_username': 'wxid-wei'},
                confidence=0.95,
            ))
            repo.upsert_entity(EntityRecord(
                entity_id='customer-temporary',
                entity_type='Customer',
                display_name='示例客户甲',
                identifiers={
                    'aliases': ['示例客户甲', 'conv-wei'],
                    'primary_user_id': 'conv-wei',
                    'resolution_source': 'observe_materialized_unresolved_customer',
                    'source_entity_ref': 'unresolved:conv-wei',
                },
                status='needs_review',
                confidence=1.0,
            ))
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-wei', 'acct-a', '示例客户甲', 'private')],
                [Message('acct-a', 'A', 'conv-wei', '示例客户甲', 'private', 'sender-wei', '示例客户甲', datetime(2026, 1, 1, tzinfo=timezone.utc), '你好', 's', 1)],
            )

            result = resolve_customer(store, '示例客户甲')

            self.assertFalse(result['ambiguous'])
            self.assertEqual(result['resolved']['entity_id'], 'customer-contact')
            self.assertEqual(result['resolved']['primary_user_id'], 'wxid-wei')
            self.assertEqual(result['resolved']['conversation_ids'], ['conv-wei'])
            self.assertEqual(result['resolved']['sender_ids'], ['sender-wei'])

    def test_concatenated_remark_and_nickname_bind_the_unique_alias_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-sample_contact-peng',
                entity_type='Customer',
                display_name='示例联系人甲',
                identifiers={'remark': '示例联系人甲', 'nickname': 'Sample Contact', 'wechat_username': 'wxid-sample_contact-peng'},
                confidence=0.95,
            ))
            repo.upsert_entity(EntityRecord(
                entity_id='customer-other-sample_contact',
                entity_type='Customer',
                display_name='Sample Contact',
                identifiers={'nickname': 'Sample Contact', 'wechat_username': 'wxid-other-sample_contact'},
                confidence=0.95,
            ))
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-sample_contact-peng', 'acct-a', 'Sample Contact', 'private')],
                [Message(
                    'acct-a', 'A', 'conv-sample_contact-peng', 'Sample Contact', 'private', 'sender-sample_contact-peng', 'Sample Contact',
                    datetime(2026, 1, 1, tzinfo=timezone.utc), '私聊消息', 's', 1,
                )],
            )

            result = resolve_customer(store, '示例联系人甲sample_contact')

            self.assertFalse(result['ambiguous'])
            self.assertEqual(result['resolved']['entity_id'], 'customer-sample_contact-peng')
            self.assertEqual(result['resolved']['conversation_ids'], ['conv-sample_contact-peng'])
            self.assertEqual(result['resolved']['sender_ids'], ['sender-sample_contact-peng'])
            self.assertIn('multi_token_identifier_match', result['resolved']['match_reasons'])


if __name__ == '__main__':
    unittest.main()
