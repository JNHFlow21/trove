from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.knowledge.conversation_card import build_conversation_card
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class ConversationCardTests(unittest.TestCase):
    def test_team_card_extracts_cited_decisions(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            card = build_conversation_card(SQLiteStore(Path(d) / 'index' / 'trove.sqlite'))
            self.assertEqual(card['conversation_title'], 'TROVE 产品小组')
            self.assertTrue(card['decisions'])
            self.assertTrue(all(item['citation'].startswith('trove://') for item in card['decisions']))
