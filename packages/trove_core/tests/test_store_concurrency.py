from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from pathlib import Path

from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.fixture_factory import generate_fixture
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.models import Message


def _bulk_messages(count: int = 1200) -> list[Message]:
    base = datetime(2026, 7, 7, 9, 0, tzinfo=timezone.utc)
    out: list[Message] = []
    for idx in range(count):
        out.append(
            Message(
                account_id='acct-bulk',
                account_label='Bulk',
                conversation_id='conv-bulk',
                conversation_title='批量导入等价性',
                conversation_type='group',
                sender_id=f'sender-{idx % 7}',
                sender_name=f'成员{idx % 7}',
                timestamp=base + timedelta(seconds=idx),
                content=f'批量消息 {idx} 示例教育 预算审批',
                shard_id='message_0',
                local_id=idx,
                sent_by_me=False,
            )
        )
    return out


class StoreConcurrencyTests(unittest.TestCase):
    def test_thread_local_connections_support_parallel_search_and_write(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            engine = HyperSearch(store)
            writer_message = _bulk_messages(1)[0]

            def search_once() -> int:
                return engine.search(SearchRequest('价格太高')).total

            def write_once() -> int:
                return store.upsert_messages([writer_message])

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(search_once) for _ in range(4)] + [pool.submit(write_once)]
                results = [future.result(timeout=10) for future in futures]

            self.assertEqual(results[-1], 1)
            self.assertTrue(all(value >= 1 for value in results[:-1]))

    def test_batched_message_upsert_is_idempotent_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            fixture = generate_fixture()
            store.upsert_accounts(fixture.accounts)
            store.upsert_conversations(fixture.conversations)
            store.upsert_messages(_bulk_messages())
            before_counts = store.counts()
            before = store.get_message_by_citation('trove://wechat/acct-bulk/conv-bulk/message_0/777')
            self.assertIsNotNone(before)

            before_total_changes = store.connect().total_changes
            changed = store.upsert_messages(_bulk_messages())
            duplicate_total_changes = store.connect().total_changes - before_total_changes
            after_counts = store.counts()
            after = store.get_message_by_citation('trove://wechat/acct-bulk/conv-bulk/message_0/777')

            self.assertEqual(changed, 0)
            self.assertEqual(duplicate_total_changes, 0)
            self.assertEqual(before_counts, after_counts)
            self.assertEqual(dict(before), dict(after))
            self.assertEqual(len(store.fts_search_filtered('示例教育 预算审批', limit=5)), 5)


if __name__ == '__main__':
    unittest.main()
