from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.knowledge.entity_reconciliation import reconciliation_plan, reconcile_customer_entities
from trove_core.store.repositories import (
    EntityRecord,
    MultimodalRepository,
    ObservationRecord,
    RelationshipRecord,
    WeChatRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class EntityReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteStore(Path(self.tmp.name) / 'vault.sqlite')
        repo = MultimodalRepository(self.store)
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
        repo.add_observation(ObservationRecord(
            observation_id='obs-temp',
            entity_id='customer-temporary',
            observation_type='relationship_signal',
            value={'text': 'fixture'},
            status='needs_review',
            confidence=0.6,
            citation='trove://fixture/message/1',
            source_type='agent',
        ))
        repo.add_relationship(RelationshipRecord(
            relationship_id='rel-temp',
            subject_entity_id='customer-temporary',
            predicate='mentioned_in',
            object_ref='trove://fixture/message/1',
        ))
        WeChatRepository(self.store).replace_fixture(
            [Account('acct-a', 'A', 'A')],
            [Conversation('conv-wei', 'acct-a', '示例客户甲', 'private')],
            [Message('acct-a', 'A', 'conv-wei', '示例客户甲', 'private', 'sender-wei', '示例客户甲', datetime(2026, 1, 1, tzinfo=timezone.utc), '你好', 's', 1)],
        )
        with self.store.connect() as conn:
            conn.execute(
                'INSERT INTO profile_snapshots(profile_id,entity_id,version,projection_json,created_at) VALUES(?,?,?,?,?)',
                ('profile-canonical', 'customer-contact', 1, '{}', '2025-12-31T00:00:00Z'),
            )
            conn.execute(
                'INSERT INTO profile_snapshots(profile_id,entity_id,version,projection_json,created_at) VALUES(?,?,?,?,?)',
                ('profile-temp', 'customer-temporary', 1, '{}', '2026-01-01T00:00:00Z'),
            )
            conn.execute(
                """INSERT INTO profile_automation_subscriptions(
                       entity_id,selector,enabled,debounce_seconds,consent_scope,last_profile_id,
                       created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    'customer-temporary', 'conv-wei', 1, 180,
                    'explicit-profile-auto-maintenance-v1', 'profile-temp',
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                ),
            )
            conn.execute(
                """INSERT INTO profile_refresh_queue(
                       entity_id,generation,state,reason,available_at,created_at,updated_at)
                   VALUES(?,1,'pending','fixture',?,?,?)""",
                (
                    'customer-temporary', '2026-01-01T00:03:00Z',
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                ),
            )
            conn.commit()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dry_run_is_read_only_and_counts_references(self):
        plan = reconciliation_plan(self.store, '示例客户甲')

        self.assertEqual(plan['canonical_entity_id'], 'customer-contact')
        self.assertEqual(plan['duplicate_entity_ids'], ['customer-temporary'])
        self.assertEqual(plan['counts']['observations'], 1)
        self.assertEqual(plan['counts']['profile_automation_subscriptions'], 1)
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT status FROM entities WHERE entity_id='customer-temporary'").fetchone()[0], 'needs_review')

    def test_reconciliation_reparents_evidence_and_is_idempotent(self):
        first = reconcile_customer_entities(self.store, '示例客户甲', apply=True)
        second = reconcile_customer_entities(self.store, '示例客户甲', apply=True)

        self.assertTrue(first['applied'])
        self.assertFalse(second['applied'])
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT entity_id FROM observations WHERE observation_id='obs-temp'").fetchone()[0], 'customer-contact')
            self.assertEqual(conn.execute("SELECT subject_entity_id FROM relationships WHERE relationship_id='rel-temp'").fetchone()[0], 'customer-contact')
            self.assertEqual(conn.execute("SELECT entity_id FROM profile_snapshots WHERE profile_id='profile-temp'").fetchone()[0], 'customer-contact')
            self.assertEqual(
                [tuple(row) for row in conn.execute(
                    "SELECT profile_id,version FROM profile_snapshots WHERE entity_id='customer-contact' ORDER BY version"
                )],
                [('profile-canonical', 1), ('profile-temp', 2)],
            )
            subscription = conn.execute(
                "SELECT entity_id,selector,last_profile_id FROM profile_automation_subscriptions"
            ).fetchone()
            self.assertEqual(subscription['entity_id'], 'customer-contact')
            self.assertEqual(subscription['last_profile_id'], 'profile-temp')
            self.assertEqual(
                conn.execute('SELECT entity_id FROM profile_refresh_queue').fetchone()[0],
                'customer-contact',
            )
            duplicate = conn.execute("SELECT status,identifiers_json FROM entities WHERE entity_id='customer-temporary'").fetchone()
            self.assertEqual(duplicate['status'], 'merged')
            self.assertEqual(json.loads(duplicate['identifiers_json'])['merged_into'], 'customer-contact')
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM entity_identifiers WHERE entity_id='customer-temporary'").fetchone()[0], 0)

    def test_reconciliation_always_queues_the_enabled_canonical_subscription(self):
        with self.store.connect() as conn:
            conn.execute('DELETE FROM profile_refresh_queue')
            conn.commit()

        reconcile_customer_entities(self.store, '示例客户甲', apply=True)

        with self.store.connect() as conn:
            row = conn.execute(
                'SELECT entity_id,state,reason,attempt_count FROM profile_refresh_queue'
            ).fetchone()
        self.assertEqual(row['entity_id'], 'customer-contact')
        self.assertEqual(row['state'], 'pending')
        self.assertEqual(row['reason'], 'entity_reconciled')
        self.assertEqual(row['attempt_count'], 0)


if __name__ == '__main__':
    unittest.main()
