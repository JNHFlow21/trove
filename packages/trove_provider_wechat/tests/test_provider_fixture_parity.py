from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from trove_core.domain.messages import Message
from trove_protocol.provider import ProviderManifest
from trove_provider_wechat import create_provider

from .fixtures import write_account


PACKAGE = Path(__file__).resolve().parents[1] / 'trove_provider_wechat'


class ProviderFixtureParityTests(unittest.TestCase):
    def test_current_adapter_and_provider_emit_same_citations_counts_and_watermark(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            account = write_account(root, 'account-a', label='A', token='parity')
            provider = create_provider(manifest, source_root=root)
            result = provider.invoke('read', {'operation': 'changes', 'account_id': 'account-a'})
            raw_rows = [json.loads(line) for line in (account / 'messages.jsonl').read_text().splitlines()]
            legacy = [Message(
                account_id='account-a', account_label='A',
                conversation_id=row['conversation_id'], conversation_title=row['conversation_title'],
                conversation_type=row['conversation_type'], sender_id=row['sender_id'],
                sender_name=row['sender_name'], timestamp=datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')),
                content=row['content'], shard_id=row['shard_id'], local_id=row['local_id'],
                direction_hint=row['direction'],
            ) for row in raw_rows]
            self.assertEqual(
                [item.citation for item in legacy],
                [item['citation'] for item in result['records']],
            )
            account_status = provider.accounts()[0]
            self.assertEqual(account_status['message_count'], len(legacy))
            self.assertEqual(account_status['watermark'], result['watermark'])


if __name__ == '__main__':
    unittest.main()
