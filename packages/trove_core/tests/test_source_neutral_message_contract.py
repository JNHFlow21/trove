from __future__ import annotations

from datetime import datetime, timezone
import unittest

from trove_core.domain.content import classify_content_kind, display_content_for_kind
from trove_core.domain.messages import Message


class SourceNeutralMessageContractTests(unittest.TestCase):
    def test_canonical_message_identity_is_source_parameterized(self):
        message = Message(
            account_id='account-a', account_label='Fixture',
            conversation_id='conversation-a', conversation_title='Fixture conversation',
            conversation_type='private', sender_id='peer-a', sender_name='Fixture peer',
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), content='hello',
            shard_id='shard-a', local_id=1, citation_source='fixture',
        )
        self.assertEqual(
            message.citation,
            'trove://fixture/account-a/conversation-a/shard-a/1',
        )
        self.assertNotIn('citation_source', message.safe_dict())

    def test_content_classification_has_no_source_dependency(self):
        self.assertEqual(classify_content_kind('<voice/>'), 'voice')
        self.assertEqual(display_content_for_kind('raw', 'voice'), '[voice]')


if __name__ == '__main__':
    unittest.main()
