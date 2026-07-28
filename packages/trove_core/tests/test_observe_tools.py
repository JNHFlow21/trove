from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.agent_tools import tools as agent_tools
from trove_core.knowledge.customer_profile import build_customer_profile
from trove_core.store.repositories import EntityRecord, MultimodalRepository
from trove_core.store.repositories import WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


class ObserveToolsTests(unittest.TestCase):
    def test_operator_observation_add_profile_retire_and_agent_proposal_approval(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-1',
                entity_type='Customer',
                display_name='示例教育',
                identifiers={'nickname': '示例教育'},
            ))

            added = agent_tools.observe_add(
                vault,
                entity='示例教育',
                observation_type='Need',
                text='需要校区预算方案',
                confidence=0.95,
            )
            self.assertEqual(added['observation']['status'], 'active')
            profile = build_customer_profile(store, '示例教育', limit=5)
            self.assertTrue(any('校区预算方案' in row['value'] for row in profile['sections']['needs']))

            listed = agent_tools.observe_list(vault, entity='示例教育')
            self.assertEqual(len(listed['observations']), 1)
            agent_tools.observe_retire(vault, observation_id=added['observation']['observation_id'], yes=True)
            profile = build_customer_profile(store, '示例教育', limit=5)
            self.assertFalse(any('校区预算方案' in row['value'] for row in profile['sections']['needs']))

            proposed = agent_tools.observe_propose(
                vault,
                entity='示例教育',
                observation_type='NextAction',
                text='下周复盘报价',
                confidence=0.7,
            )
            self.assertEqual(proposed['observation']['status'], 'needs_review')
            approved = agent_tools.observe_approve(vault, observation_id=proposed['observation']['observation_id'], yes=True)
            self.assertEqual(approved['observation']['status'], 'active')
            profile = build_customer_profile(store, '示例教育', limit=5)
            self.assertTrue(any('下周复盘报价' in row['value'] for row in profile['sections']['next_actions']))

    def test_agent_proposal_materializes_unresolved_private_customer(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-sample_contact', 'acct-a', 'Sample Contact', 'private')],
                [Message('acct-a', 'A', 'conv-sample_contact', 'Sample Contact', 'private', 'conv-sample_contact', 'Sample Contact', datetime(2026, 1, 1, tzinfo=timezone.utc), '需要预算方案', 's', 1)],
            )

            proposed = agent_tools.observe_propose(
                vault,
                entity='Sample Contact',
                observation_type='Need',
                text='需要预算方案',
                confidence=0.7,
                citation='trove://wechat/acct-a/conv-sample_contact/s/1',
            )

            self.assertEqual(proposed['observation']['status'], 'needs_review')
            self.assertTrue(proposed['observation']['entity_id'].startswith('customer-'))
            profile = build_customer_profile(store, 'Sample Contact', limit=5)
            self.assertTrue(any('需要预算方案' in row['value'] for row in profile['sections']['needs']))

    def test_explicit_intake_materializes_reviewable_entity_without_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            VaultConfig.resolve(str(vault), env={}).ensure()

            added = agent_tools.observe_add(
                vault,
                entity='Synthetic Intake Only',
                observation_type='profile_intake',
                text='Explicit operator request; facts still require evidence.',
                confidence=1.0,
            )

            self.assertTrue(added['observation']['entity_id'].startswith('customer-'))
            store = SQLiteStore(VaultConfig.resolve(str(vault), env={}).paths.sqlite_path)
            with store.connect() as conn:
                entity = conn.execute(
                    'SELECT status,confidence FROM entities WHERE entity_id=?',
                    (added['observation']['entity_id'],),
                ).fetchone()
            self.assertEqual(entity['status'], 'needs_review')
            self.assertEqual(entity['confidence'], 0.5)


if __name__ == '__main__':
    unittest.main()
