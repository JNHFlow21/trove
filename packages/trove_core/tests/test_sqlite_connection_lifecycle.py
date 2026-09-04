from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore, open_store
from trove_core.wechat.indexer import index_fixture_vault


class SQLiteConnectionLifecycleTests(unittest.TestCase):
    def _fixture_path(self, root: Path) -> Path:
        index_fixture_vault(root, reset=True)
        path = root / 'index' / 'trove.sqlite'
        # The fixture publisher must leave a complete main database.  These
        # tests start without sidecars so any later sidecar is attributable to
        # a read connection created by SQLiteStore.
        self.assertFalse(Path(f'{path}-wal').exists())
        self.assertFalse(Path(f'{path}-shm').exists())
        return path

    def test_cross_thread_close_all_releases_every_idle_search_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture_path(Path(directory))
            store = open_store(path, readonly=True)
            engine = HyperSearch(store)

            worker_count = 6
            ready = threading.Barrier(worker_count + 1)
            release = threading.Event()

            def search_then_wait() -> int:
                result = engine.search(SearchRequest('价格太高', limit=2))
                ready.wait(timeout=10)
                release.wait(timeout=10)
                return result.total

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(search_then_wait) for _ in range(worker_count)]
                ready.wait(timeout=10)
                # ``open_store`` validated the schema on the main thread and
                # each worker owns one additional cached handle.
                self.assertEqual(store.active_connection_count, worker_count + 1)
                self.assertLessEqual(store.active_connection_count, store.max_connections)

                # Invalidation happens on a thread which owns none of the
                # connections.  It must nevertheless close the whole store
                # generation, not merely this thread's local slot.
                store.close_all()
                store.close_all()
                self.assertEqual(store.active_connection_count, 0)
                self.assertFalse(Path(f'{path}-wal').exists())
                self.assertFalse(Path(f'{path}-shm').exists())

                release.set()
                self.assertTrue(all(future.result(timeout=10) >= 1 for future in futures))

    def test_adhoc_threads_release_connection_slots_only_on_explicit_close(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture_path(Path(directory))
            store = open_store(path, readonly=True, max_connections=3)
            try:
                def ad_hoc_read(*, close: bool) -> None:
                    with store.connect() as conn:
                        conn.execute('SELECT COUNT(*) FROM messages').fetchone()
                    if close:
                        store.close_thread_connection()

                def run(*, close: bool) -> str | None:
                    errors: list[str] = []

                    def target() -> None:
                        try:
                            ad_hoc_read(close=close)
                        except Exception as exc:  # surfaced via return value
                            errors.append(exc.__class__.__name__)

                    thread = threading.Thread(target=target)
                    thread.start()
                    thread.join(10)
                    self.assertFalse(thread.is_alive())
                    return errors[0] if errors else None

                # open_store() validated the schema on the main thread, which
                # keeps one pooled handle; two more ad-hoc reads fit exactly.
                self.assertIsNone(run(close=False))
                self.assertIsNone(run(close=False))
                # Both ad-hoc slots are still held by the two dead threads'
                # handles; a third ad-hoc reader cannot acquire one.
                self.assertEqual(run(close=False), 'SQLiteConnectionLimit')
                self.assertEqual(store.active_connection_count, 3)
                store.close_all()
                self.assertEqual(store.active_connection_count, 0)
                # Closing the ad-hoc thread's handle on exit keeps the pool
                # stable across any number of short-lived readers.
                for _ in range(3):
                    self.assertIsNone(run(close=True))
                self.assertEqual(store.active_connection_count, 0)
            finally:
                store.close_all()
            self.assertEqual(store.active_connection_count, 0)
            self.assertFalse(Path(f'{path}-wal').exists())
            self.assertFalse(Path(f'{path}-shm').exists())

    def test_search_and_close_all_are_serialized_without_programming_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture_path(Path(directory))
            store = open_store(path, readonly=True)
            engine = HyperSearch(store)
            start = threading.Barrier(9)

            def search_many() -> list[int]:
                start.wait(timeout=10)
                return [engine.search(SearchRequest('价格太高', limit=2)).total for _ in range(30)]

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(search_many) for _ in range(8)]
                start.wait(timeout=10)
                for _ in range(30):
                    store.close_all()
                results = [future.result(timeout=30) for future in futures]

            self.assertTrue(all(total >= 1 for batch in results for total in batch))
            store.close_all()
            self.assertEqual(store.active_connection_count, 0)
            self.assertGreaterEqual(store.connection_open_count, 1)
            self.assertFalse(Path(f'{path}-wal').exists())
            self.assertFalse(Path(f'{path}-shm').exists())

    def test_retired_generation_is_never_reused_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture_path(Path(directory))
            store = open_store(path, readonly=True)
            with store.connect() as first:
                first_generation = first._trove_generation
                self.assertGreater(first.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)

            store.close()
            store.close()
            with store.connect() as second:
                self.assertNotEqual(first_generation, second._trove_generation)
                self.assertIsNot(first, second)
                self.assertGreater(second.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)

            store.close_all()
            self.assertEqual(store.active_connection_count, 0)

    @unittest.skipUnless(hasattr(os, 'fork'), 'fork is required')
    def test_forked_child_discards_inherited_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture_path(Path(directory))
            store = open_store(path, readonly=True)
            with store.connect() as parent_connection:
                parent_generation = parent_connection._trove_generation

            read_fd, write_fd = os.pipe()
            child = os.fork()
            if child == 0:  # pragma: no cover - assertions are reported by pipe
                os.close(read_fd)
                try:
                    with store.connect() as child_connection:
                        payload = (
                            f'ok:{child_connection._trove_generation}:'
                            f'{child_connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]}'
                        )
                except BaseException as exc:
                    payload = f'error:{type(exc).__name__}'
                os.write(write_fd, payload.encode('ascii'))
                os.close(write_fd)
                os._exit(0)

            os.close(write_fd)
            payload = os.read(read_fd, 256).decode('ascii')
            os.close(read_fd)
            _, status = os.waitpid(child, 0)
            self.assertEqual(status, 0)
            self.assertTrue(payload.startswith('ok:'), payload)
            _, generation, count = payload.split(':')
            self.assertNotEqual(int(generation), parent_generation)
            self.assertGreater(int(count), 0)

            # Closing the child's copy must not disturb the parent's handle.
            with store.connect() as connection:
                self.assertEqual(connection._trove_generation, parent_generation)
                self.assertGreater(connection.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)
            store.close_all()


if __name__ == '__main__':
    unittest.main()
