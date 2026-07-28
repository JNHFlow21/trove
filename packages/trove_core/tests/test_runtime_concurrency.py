from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from trove_core.runtime import BoundedExecutor, RuntimeOverloaded, RuntimeTimedOut, SearchRuntimeCache
from trove_core.search.query import SearchRequest
from trove_core.vault.config import VaultConfig
from trove_core.vault.mutations import coordinated_vault_mutation
from trove_core.wechat.indexer import index_fixture_vault


class RuntimeConcurrencyTests(unittest.TestCase):
    @staticmethod
    def _open_fd_count() -> int:
        count = 0
        for fd in range(512):
            try:
                fcntl.fcntl(fd, fcntl.F_GETFD)
            except OSError:
                continue
            count += 1
        return count

    def test_bounded_executor_rejects_overload_and_releases_cancelled_queue_slot(self):
        executor = BoundedExecutor(max_workers=1, max_queue=1, thread_name_prefix='bounded-test', submit_timeout_seconds=0)
        entered = threading.Event()
        release = threading.Event()

        def block():
            entered.set()
            release.wait(3)

        active = executor.submit(block)
        self.assertTrue(entered.wait(1))
        queued = executor.submit(lambda: None)
        with self.assertRaises(RuntimeOverloaded):
            executor.submit(lambda: None)
        self.assertTrue(queued.cancel())
        replacement = executor.submit(lambda: 'replacement')
        release.set()
        active.result(timeout=2)
        self.assertEqual(replacement.result(timeout=2), 'replacement')
        executor.shutdown()

    def test_search_runtime_timeout_is_typed_and_worker_remains_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(directory, env={})
            runtime = SearchRuntimeCache(
                cfg,
                provider_factory=lambda: None,
                max_workers=1,
                max_queue=0,
                timeout_seconds=0.02,
                submit_timeout_seconds=0,
            )
            engine = runtime.get()
            original = engine.search
            release = threading.Event()

            def slow(request):
                release.wait(1)
                return original(request)

            engine.search = slow  # type: ignore[method-assign]
            with self.assertRaises(RuntimeTimedOut):
                runtime.search(SearchRequest('价格太高', semantic='off'))
            with self.assertRaises(RuntimeOverloaded):
                runtime.search(SearchRequest('报价', semantic='off'))
            self.assertLessEqual(runtime.status()['workers']['active_workers'], 1)
            release.set()
            deadline = time.time() + 2
            while runtime.status()['workers']['active_workers'] and time.time() < deadline:
                time.sleep(0.01)
            runtime.close()

    def test_thirty_two_readers_and_one_writer_have_bounded_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(directory, env={})
            runtime = SearchRuntimeCache(
                cfg,
                provider_factory=lambda: None,
                max_workers=8,
                max_queue=32,
                timeout_seconds=10,
            )
            barrier = threading.Barrier(32)
            readers_entered = threading.Event()
            release_readers = threading.Event()
            entered_lock = threading.Lock()
            entered_count = 0
            engine = runtime.get()
            original_search = engine.search

            def held_search(search_request):
                nonlocal entered_count
                with entered_lock:
                    entered_count += 1
                    if entered_count == 8:
                        readers_entered.set()
                release_readers.wait(5)
                return original_search(search_request)

            engine.search = held_search  # type: ignore[method-assign]
            baseline_threads = threading.active_count()
            baseline_fds = self._open_fd_count()
            writer_started = threading.Event()
            writer_done = threading.Event()
            writer_errors: list[BaseException] = []

            def read(index: int):
                barrier.wait(5)
                # Distinct timeout values deliberately bypass identical-query
                # singleflight while preserving the same retrieval workload.
                # This test stresses eight simultaneous engine readers; the
                # dedicated runtime-cache test covers shared identical calls.
                return runtime.search(SearchRequest(
                    '价格太高',
                    limit=2,
                    semantic='off',
                    reranker_timeout_ms=200 + index,
                ))

            def write() -> None:
                writer_started.set()
                try:
                    with coordinated_vault_mutation(cfg, operation='scope_rebuild'):
                        connection = sqlite3.connect(cfg.paths.sqlite_path)
                        try:
                            connection.execute(
                                'UPDATE messages SET content=? WHERE citation=?',
                                ('runtime writer sentinel', 'trove://wechat/acct-work/conv-example_edu-private/message_0/1'),
                            )
                            connection.commit()
                        finally:
                            connection.close()
                except BaseException as exc:  # pragma: no cover - asserted below
                    writer_errors.append(exc)
                finally:
                    writer_done.set()

            with ThreadPoolExecutor(max_workers=32) as callers:
                reader_futures = [callers.submit(read, index) for index in range(32)]
                self.assertTrue(readers_entered.wait(3))
                writer = threading.Thread(target=write, daemon=True)
                writer.start()
                self.assertTrue(writer_started.wait(1))
                time.sleep(0.05)
                self.assertFalse(writer_done.is_set())
                release_readers.set()
                responses = [future.result(timeout=5) for future in reader_futures]
                writer.join(5)
            self.assertTrue(all(response.results for response in responses))
            self.assertFalse(writer_errors)
            self.assertTrue(writer_done.is_set())
            engine.search = original_search  # type: ignore[method-assign]
            updated = runtime.search(SearchRequest('writer sentinel', limit=2, semantic='off'))
            self.assertTrue(updated.results)
            status = runtime.status()
            self.assertEqual(runtime.generation, 1)
            self.assertLessEqual(status['resource_count'], status['workers']['max_workers'] + 3)
            self.assertEqual(status['workers']['active_workers'], 0)
            self.assertEqual(status['workers']['queued_workers'], 0)
            self.assertLessEqual(
                status['resource_counts']['engine_connections'],
                status['workers']['max_workers'] + 1,
            )
            self.assertLessEqual(status['result_cache_bytes'], status['result_cache_max_bytes'])
            runtime.close()
            self.assertLessEqual(threading.active_count(), baseline_threads + 1)
            self.assertLessEqual(self._open_fd_count(), baseline_fds + 2)


if __name__ == '__main__':
    unittest.main()
