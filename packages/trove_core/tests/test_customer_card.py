from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.knowledge.customer_card import build_customer_card
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class CustomerCardTests(unittest.TestCase):
    def test_customer_card_has_blocker_stage_next_action_and_citations(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            card = build_customer_card(SQLiteStore(Path(d) / 'index' / 'trove.sqlite'), '示例教育')
            self.assertIn('价格', card['blocker']['value'])
            self.assertTrue(card['blocker']['citations'])
            self.assertTrue(card['stage']['citations'])
            self.assertTrue(card['next_action']['citations'])
