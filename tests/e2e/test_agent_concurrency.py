from __future__ import annotations

import concurrent.futures
from pathlib import Path
import tempfile
import threading
import time
import unittest

from trove_client import TroveClient, TroveClientError
from trove_daemon.lifecycle import RuntimeIdentity, catalog_identity
from trove_daemon.server import DaemonServer

from tests.e2e.runtime_harness import RuntimeHarness


class AgentConcurrencyTests(unittest.TestCase):
    def test_eight_reads_and_one_writer_complete_without_corruption(self):
        baseline_threads = threading.active_count()
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(
            Path(directory) / 'vault', max_workers=8, max_pending=16,
        ) as runtime:
            def read(index: int):
                with TroveClient(runtime.identity, pool_size=1, autostart=None) as client:
                    return client.call(
                        'trove.recall',
                        {
                            'account_id': 'acct-work',
                            'conversation_id': 'conv-sales-review', 'limit': 20,
                        },
                        request_id=f'concurrent-read-{index}', timeout=5,
                    )

            def write():
                with TroveClient(runtime.identity, pool_size=1, autostart=None) as client:
                    return client.call(
                        'trove.observe_add',
                        {
                            'target': '示例教育', 'text': 'concurrency fixture',
                            'idempotency_key': ('concurrency-' + 'observation-1'),
                        },
                        request_id='concurrent-write', timeout=5,
                    )

            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
                futures = [pool.submit(read, index) for index in range(8)]
                futures.append(pool.submit(write))
                results = [future.result(timeout=10) for future in futures]
            self.assertTrue(all(item['ok'] for item in results))
            self.assertEqual(
                {item['coverage']['state'] for item in results[:8]}, {'complete'},
            )
            observations = runtime.call(
                'trove.observe_list', {'target': '示例教育', 'limit': 10}, 'after-concurrency',
            )
            self.assertEqual(observations['data']['count'], 1)
            status = runtime.owner.status()
            self.assertLessEqual(status['read_connections'], status['max_read_connections'])
            workers = status['workers']
            self.assertLessEqual(workers['active_workers'], workers['max_workers'])
            self.assertLessEqual(workers['queued_workers'], workers['max_queue'])
        deadline = time.time() + 2
        while threading.active_count() > baseline_threads + 1 and time.time() < deadline:
            time.sleep(0.02)
        self.assertLessEqual(threading.active_count(), baseline_threads + 1)

    def test_queue_saturation_and_timeout_are_typed_and_recoverable(self):
        class BlockingDispatcher:
            def __init__(self):
                self.gate = threading.Event()
                self.entered = threading.Event()

            def dispatch(self, _capability, _payload, *, request_id, response_budget=None):
                self.entered.set()
                self.gate.wait(2)
                return {
                    'protocol': 'trove/1', 'request_id': request_id,
                    'ok': True, 'data': {'complete': True},
                }

        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            vault.mkdir()
            identity = RuntimeIdentity.for_vault(
                vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
            )
            dispatcher = BlockingDispatcher()
            server = DaemonServer(
                identity, dispatcher, max_workers=1, max_pending=1, idle_timeout=None,
            )
            server.start()
            first = TroveClient(identity, autostart=None)
            second = TroveClient(identity, autostart=None)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    blocked = pool.submit(
                        first.call, 'trove.capabilities', {},
                        request_id='blocked', timeout=0.1,
                    )
                    self.assertTrue(dispatcher.entered.wait(1))
                    with self.assertRaises(TroveClientError) as saturated:
                        second.call(
                            'trove.capabilities', {}, request_id='saturated', timeout=0.2,
                        )
                    self.assertIn(saturated.exception.code, {'busy', 'timeout'})
                    with self.assertRaises(TroveClientError) as timed_out:
                        blocked.result(timeout=1)
                    self.assertEqual(timed_out.exception.code, 'timeout')
                    dispatcher.gate.set()
                time.sleep(0.1)
                self.assertTrue(first.call(
                    'trove.capabilities', {}, request_id='recovered', timeout=1,
                )['ok'])
            finally:
                dispatcher.gate.set()
                first.close()
                second.close()
                server.stop(timeout=1)


if __name__ == '__main__':
    unittest.main()
