from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from trove_core.bounds import BoundedInputError
from trove_core.search.context import ContextService
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.indexer import index_fixture_vault

class ContextWindowTests(unittest.TestCase):
    def test_context_returns_before_after_in_order_and_rejects_oversize(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            service = ContextService(SQLiteStore(Path(d) / 'index' / 'trove.sqlite'), max_window=5)
            with self.assertRaises(BoundedInputError) as raised:
                service.fetch('trove://wechat/acct-work/conv-sales-review/message_0/11', before=99, after=5)
            self.assertEqual(raised.exception.code, 'invalid_limit')
            self.assertEqual(raised.exception.field, 'before')
            ctx = service.fetch('trove://wechat/acct-work/conv-sales-review/message_0/11', before=5, after=5)
            self.assertEqual(ctx['before'], 5)
            self.assertEqual(ctx['after'], 5)
            self.assertLessEqual(len(ctx['messages']), 6)
            timestamps = [m['timestamp'] for m in ctx['messages']]
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertTrue(any(m['citation'].endswith('/11') for m in ctx['messages']))
