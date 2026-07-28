from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from trove_core.reply.context import (
    ContextBridge,
    ContextBridgeError,
    ReplyContextEnvelope,
)
from trove_core.reply.models import EvidenceMessage, ReplyEvent, sha256_text
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


def build_vault(root: Path) -> VaultConfig:
    config = VaultConfig.resolve(str(root / 'vault'), env={})
    config.ensure()
    store = SQLiteStore(config.paths.sqlite_path)
    store.initialize()
    store.upsert_accounts([Account('account-fixture', 'Account', 'Account')])
    store.upsert_conversations([
        Conversation('conversation-fixture', 'account-fixture', 'Alice', 'private'),
    ])
    store.upsert_messages([
        Message(
            'account-fixture',
            'Account',
            'conversation-fixture',
            'Alice',
            'private',
            'peer-fixture',
            'Alice',
            datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            'history',
            'message-fixture',
            1,
            direction_hint='incoming',
        ),
    ])
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO entities(
                   entity_id,entity_type,display_name,identifiers_json,status,
                   confidence,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                'entity-fixture', 'Person', 'Alice', '{}', 'active', 1.0,
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
            ),
        )
        conn.execute(
            """INSERT INTO entity_identifiers(
                   entity_id,identifier_type,normalized_value,source,confidence,
                   citation,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                'entity-fixture', 'wechat_id', 'peer-fixture', 'fixture', 1.0,
                'trove://fixture/entity', '2026-01-01T00:00:00Z',
                '2026-01-01T00:00:00Z',
            ),
        )
        conn.execute(
            """INSERT INTO profile_snapshots(
                   profile_id,entity_id,version,projection_json,content_hash,
                   source_revision,schema_version,completeness_state,
                   evidence_citations_json,enrichment_summary_json,gaps_json,
                   created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                'profile-fixture', 'entity-fixture', 1,
                json.dumps({
                    'summary': 'prefers concise replies',
                    'evidence_citations': ['trove://fixture/entity'],
                }),
                'a' * 64, 'fixture', 'customer-profile/auto-v1', 'current',
                '["trove://fixture/entity"]', '{}', '[]',
                '2026-01-01T00:00:00Z',
            ),
        )
        conn.commit()
    store.close_all()
    return config


def live_event(position: int = 2, *, text: str = 'new message') -> ReplyEvent:
    return ReplyEvent(
        event_id=f'event-fixture-{position}',
        account_id='account-fixture',
        conversation_id='conversation-fixture',
        target_ref='b' * 64,
        source_position=position,
        latest_fingerprint=sha256_text(f'row-{position}'),
        messages=(
            EvidenceMessage(
                citation=f'provider://wechat/live/{"b" * 64}/{position}',
                source_position=position,
                observed_at=1_767_268_810.0,
                kind='text',
                text=text,
            ),
        ),
        observed_at=1_767_268_810.0,
    )


class ReplyContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = build_vault(Path(self.temp.name))
        self.bridge = ContextBridge(self.config, history_limit=20)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_scope_merges_live_delta_with_citations_and_profile(self):
        envelope = self.bridge.build(
            live_event(), round_id='round-fixture', round_revision=1,
        )
        self.assertEqual(envelope.account_id, 'account-fixture')
        self.assertEqual(envelope.conversation_id, 'conversation-fixture')
        self.assertEqual([item.text for item in envelope.messages], [
            'history', 'new message',
        ])
        self.assertTrue(envelope.messages[-1].live_delta)
        self.assertEqual(
            envelope.new_message_citations,
            (f'provider://wechat/live/{"b" * 64}/2',),
        )
        self.assertEqual(envelope.profile['summary'], 'prefers concise replies')
        self.assertEqual(envelope.coverage['state'], 'complete')
        self.assertEqual(len(envelope.digest), 64)

    def test_projection_catchup_deduplicates_same_source_position_and_text(self):
        store = SQLiteStore(self.config.paths.sqlite_path)
        store.upsert_messages([
            Message(
                'account-fixture',
                'Account',
                'conversation-fixture',
                'Alice',
                'private',
                'peer-fixture',
                'Alice',
                datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
                'new message',
                'message-fixture',
                2,
                direction_hint='incoming',
            ),
        ])
        store.close_all()
        envelope = self.bridge.build(
            live_event(), round_id='round-fixture', round_revision=1,
        )
        self.assertEqual([item.text for item in envelope.messages], [
            'history', 'new message',
        ])
        self.assertFalse(envelope.messages[-1].live_delta)
        self.assertEqual(
            envelope.new_message_citations,
            (envelope.messages[-1].citation,),
        )

    def test_projection_catchup_deduplicates_media_with_different_placeholder(self):
        store = SQLiteStore(self.config.paths.sqlite_path)
        store.upsert_messages([
            Message(
                'account-fixture',
                'Account',
                'conversation-fixture',
                'Alice',
                'private',
                'peer-fixture',
                'Alice',
                datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
                '[image]',
                'message-fixture',
                2,
                direction_hint='incoming',
                content_kind='image',
            ),
        ])
        store.close_all()
        source = live_event(position=2, text='[图片]')
        event = ReplyEvent(**{
            **source.__dict__,
            'messages': (
                EvidenceMessage(
                    citation=f'provider://wechat/live/{"b" * 64}/2',
                    source_position=2,
                    observed_at=1_767_268_810.0,
                    kind='image',
                    text='[图片]',
                ),
            ),
        })

        envelope = self.bridge.build(
            event, round_id='round-fixture', round_revision=1,
        )

        self.assertEqual(len(envelope.messages), 2)
        self.assertEqual(envelope.messages[-1].kind, 'image')
        self.assertFalse(envelope.messages[-1].live_delta)
        self.assertTrue(
            envelope.new_message_citations[0].startswith('trove://wechat/'),
        )
        self.assertEqual(envelope.media[0]['state'], 'pending_index')

    def test_cross_account_or_missing_conversation_fails_closed(self):
        event = ReplyEvent(**{
            **live_event().__dict__,
            'account_id': 'different-account',
        })
        with self.assertRaises(ContextBridgeError):
            self.bridge.build(
                event, round_id='round-fixture', round_revision=1,
            )

    def test_context_is_bounded(self):
        bridge = ContextBridge(self.config, history_limit=1)
        envelope = bridge.build(
            live_event(), round_id='round-fixture', round_revision=1,
        )
        self.assertEqual(len(envelope.messages), 1)
        self.assertTrue(envelope.coverage['truncated'])


if __name__ == '__main__':
    unittest.main()
