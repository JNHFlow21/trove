from __future__ import annotations
import unittest
from datetime import datetime, timezone

from trove_core.wechat.models import Message
from trove_core.wechat.normalize import normalize_text, safe_label

class DomainModelTests(unittest.TestCase):
    def test_message_citation_and_direction(self):
        msg = Message(account_id='acct-a', account_label='Work', conversation_id='conv-1', conversation_title='客户群', conversation_type='group', sender_id='u1', sender_name='Alice', timestamp=datetime(2026,6,21,tzinfo=timezone.utc), content='价格是卡点', shard_id='s1', local_id=42)
        self.assertEqual(msg.citation, 'trove://wechat/acct-a/conv-1/s1/42')
        self.assertEqual(msg.direction, 'incoming')
        self.assertEqual(msg.safe_dict()['timestamp'], '2026-06-21T00:00:00Z')

    def test_text_normalization_and_safe_label(self):
        self.assertEqual(normalize_text('  A\n\tB　C  '), 'A B C')
        self.assertEqual(safe_label('客户/群\\A'), '客户-群-A')
