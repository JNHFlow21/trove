from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore, open_store
from trove_core.wechat.indexer import index_fixture_vault


class PreparedStatementReuseTests(unittest.TestCase):
    def test_repeated_repository_shape_reuses_connection_and_statement_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            store = open_store(
                root / 'index' / 'trove.sqlite', readonly=True,
                max_connections=3, prepared_statement_cache_size=32,
                page_cache_kib=4096,
            )
            baseline = store.connection_open_count
            for _ in range(20):
                with store.connect() as connection:
                    count = connection.execute(
                        'SELECT COUNT(*) FROM messages WHERE account_id=?', ('acct-work',),
                    ).fetchone()[0]
                self.assertGreater(count, 0)
            self.assertEqual(store.connection_open_count, baseline)
            self.assertEqual(store.prepared_statement_cache_size, 32)
            self.assertEqual(store.page_cache_kib, 4096)
            with store.connect() as connection:
                self.assertEqual(connection.execute('PRAGMA cache_size').fetchone()[0], -4096)
            store.close_all()

    def test_bounded_threads_never_share_live_cursor_or_exceed_pool_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            store = SQLiteStore(
                root / 'index' / 'trove.sqlite', readonly=True,
                max_connections=2, connection_wait_seconds=0.2,
            )
            store.initialize()
            # Initialization owns one reusable main-thread handle; retire it so
            # the two worker slots represent the complete read pool.
            store.close_all()
            barrier = threading.Barrier(3)
            release = threading.Event()
            cursors = []
            errors = []

            def read():
                try:
                    with store.connect() as connection:
                        cursor = connection.execute('SELECT citation FROM messages ORDER BY citation')
                        cursors.append(cursor)
                        barrier.wait(timeout=2)
                        release.wait(2)
                        cursor.fetchall()
                except BaseException as exc:
                    errors.append(exc)

            workers = [threading.Thread(target=read, daemon=True) for _ in range(2)]
            for worker in workers:
                worker.start()
            barrier.wait(timeout=2)
            self.assertEqual(store.active_connection_count, 2)
            self.assertIsNot(cursors[0], cursors[1])
            release.set()
            for worker in workers:
                worker.join(timeout=2)
            self.assertFalse(errors)
            store.close_all()


if __name__ == '__main__':
    unittest.main()
