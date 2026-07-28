from __future__ import annotations

from datetime import datetime, timezone, timedelta
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class SQLiteSearchPushdownTests(unittest.TestCase):
    def test_exact_search_pushes_conversation_filter_before_limit(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.upsert_accounts([Account('acct', 'Work', 'Work')])
            conversations = [Conversation(f'conv-{idx}', 'acct', f'Conversation {idx}', 'group') for idx in range(20)]
            conversations.append(Conversation('target', 'acct', 'Target', 'group'))
            store.upsert_conversations(conversations)
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            messages = []
            for idx in range(20):
                messages.append(Message(
                    account_id='acct',
                    account_label='Work',
                    conversation_id=f'conv-{idx}',
                    conversation_title=f'Conversation {idx}',
                    conversation_type='group',
                    sender_id='u',
                    sender_name='User',
                    timestamp=base + timedelta(minutes=idx),
                    content='needle decoy',
                    shard_id='s',
                    local_id=idx,
                ))
            messages.append(Message(
                account_id='acct',
                account_label='Work',
                conversation_id='target',
                conversation_title='Target',
                conversation_type='group',
                sender_id='u',
                sender_name='User',
                timestamp=base + timedelta(hours=2),
                content='needle target',
                shard_id='s',
                local_id=999,
            ))
            store.upsert_messages(messages)

            rows = store.exact_search('needle', filters={'conversation_id': 'target'}, limit=1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['conversation_id'], 'target')

    def test_chunk_search_pushes_source_filter(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.upsert_accounts([Account('acct', 'Work', 'Work')])
            store.upsert_conversations([Conversation('target', 'acct', 'Target', 'group')])
            store.upsert_messages([Message(
                account_id='acct',
                account_label='Work',
                conversation_id='target',
                conversation_title='Target',
                conversation_type='group',
                sender_id='u',
                sender_name='User',
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                content='chunk needle',
                shard_id='s',
                local_id=1,
            )])
            store.rebuild_evidence_chunks(max_chars=50, overlap_chars=0)
            rows = store.chunk_search('needle', filters={'source_type': 'message'}, limit=5)
            self.assertTrue(rows)
            self.assertEqual({r['source_type'] for r in rows}, {'message'})

    def test_metadata_search_skips_broad_filters(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.upsert_accounts([Account('acct', 'Work', 'Work')])
            store.upsert_conversations([Conversation('target', 'acct', 'Target', 'group')])
            store.upsert_messages([Message(
                account_id='acct',
                account_label='Work',
                conversation_id='target',
                conversation_title='Target',
                conversation_type='group',
                sender_id='u',
                sender_name='User',
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                content='metadata needle',
                shard_id='s',
                local_id=1,
            )])

            self.assertEqual(store.metadata_search('needle', filters={'source_type': 'message'}, limit=5), [])
            self.assertEqual(store.metadata_search('needle', filters={'account_id': 'acct'}, limit=5), [])
            rows = store.metadata_search('needle', filters={'conversation_id': 'target'}, limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['conversation_id'], 'target')

    def test_fts_like_fallback_only_applies_to_short_queries(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.upsert_accounts([Account('acct', 'Work', 'Work')])
            store.upsert_conversations([Conversation('target', 'acct', 'Target', 'group')])
            store.upsert_messages([Message(
                account_id='acct',
                account_label='Work',
                conversation_id='target',
                conversation_title='Target',
                conversation_type='group',
                sender_id='u',
                sender_name='User',
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                content='预算',
                shard_id='s',
                local_id=1,
            )])

            self.assertTrue(store.fts_search_filtered('预算', filters={'source_type': 'message'}, limit=5))
            self.assertEqual(
                store.fts_search_filtered('预算', filters={'source_type': 'message'}, limit=5, allow_like_fallback=False),
                [],
            )

    def test_normalized_phrase_search_matches_line_breaks(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.upsert_accounts([Account('acct', 'Work', 'Work')])
            store.upsert_conversations([Conversation('target', 'acct', 'Target', 'group')])
            store.upsert_messages([Message(
                account_id='acct',
                account_label='Work',
                conversation_id='target',
                conversation_title='Target',
                conversation_type='group',
                sender_id='u',
                sender_name='User',
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                content='alpha\nbeta unique',
                shard_id='s',
                local_id=1,
            )])

            rows = store.exact_search('alpha beta', limit=3)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['citation'], 'trove://wechat/acct/target/s/1')

    def test_short_query_like_fallback_prefers_recent_messages(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            store.upsert_accounts([Account('acct', 'Work', 'Work')])
            store.upsert_conversations([Conversation('target', 'acct', 'Target', 'group')])
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            store.upsert_messages([
                Message(
                    account_id='acct',
                    account_label='Work',
                    conversation_id='target',
                    conversation_title='Target',
                    conversation_type='group',
                    sender_id='u',
                    sender_name='User',
                    timestamp=base,
                    content='预算',
                    shard_id='s',
                    local_id=1,
                ),
                Message(
                    account_id='acct',
                    account_label='Work',
                    conversation_id='target',
                    conversation_title='Target',
                    conversation_type='group',
                    sender_id='u',
                    sender_name='User',
                    timestamp=base + timedelta(days=1),
                    content='预算',
                    shard_id='s',
                    local_id=2,
                ),
            ])

            rows = store.exact_search('预算', limit=1)
            self.assertEqual(rows[0]['citation'], 'trove://wechat/acct/target/s/2')


if __name__ == '__main__':
    unittest.main()
