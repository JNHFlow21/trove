from __future__ import annotations

from dataclasses import replace
import tempfile
import time
import unittest
from pathlib import Path
import subprocess
import sys

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.runtime import ByteBoundedLRU
from trove_core.search.query import SearchRequest
from trove_core.vault.config import VaultConfig
from trove_core.wechat.fixture_factory import FixtureData, generate_fixture
from trove_core.wechat.indexer import index_fixture_data, index_fixture_vault
from trove_daemon.runtime_owner import RuntimeOwner
from trove_daemon.main import build_parser
from trove_daemon.server import DEFAULT_IDLE_TIMEOUT_SECONDS


class RuntimeResourceBudgetTests(unittest.TestCase):
    def test_runtime_owner_accepts_exactly_one_reply_service_and_closes_it(self):
        class Service:
            keepalive_required = True

            def __init__(self):
                self.closed = False

            def status(self):
                return {'state': 'running'}

            def reviews(self, **_kwargs):
                return []

            def activity(self, **_kwargs):
                return []

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            config = VaultConfig.resolve(directory, env={})
            owner = RuntimeOwner(config)
            service = Service()
            owner.attach_reply_service(service)
            self.assertTrue(owner.keepalive_required)
            self.assertEqual(owner.reply_status()['state'], 'running')
            with self.assertRaises(RuntimeError):
                owner.attach_reply_service(Service())
            owner.close()
            self.assertTrue(service.closed)

    def test_default_daemon_idle_timeout_releases_transient_runtime_before_next_sync(self):
        args = build_parser().parse_args(['--vault', '/tmp/trove-fixture'])
        self.assertEqual(DEFAULT_IDLE_TIMEOUT_SECONDS, 60.0)
        self.assertEqual(args.idle_timeout, DEFAULT_IDLE_TIMEOUT_SECONDS)

    def test_daemon_read_pool_has_a_process_wide_page_cache_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            owner = RuntimeOwner(VaultConfig.resolve(directory, env={}))
            try:
                owner.read_store.initialize()
                with owner.read_store.connect() as connection:
                    self.assertEqual(
                        connection.execute('PRAGMA cache_size').fetchone()[0], -8 * 1024,
                    )
                status = owner.status()
                self.assertEqual(status['read_page_cache_kib'], 8 * 1024)
                self.assertEqual(status['max_read_page_cache_bytes'], 64 * 1024 * 1024)
            finally:
                owner.close()

    def test_status_path_does_not_import_search_provider_or_model_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            code = (
                'import sys\n'
                'from trove_daemon.runtime_owner import RuntimeOwner\n'
                'from trove_core.vault.config import VaultConfig\n'
                f'owner=RuntimeOwner(VaultConfig.resolve({directory!r}, env={{}}))\n'
                'owner.status()\n'
                "for name in ('trove_core.runtime','trove_core.search.hyper_search','trove_core.providers.factory'):\n"
                "    assert name not in sys.modules, name\n"
                'owner.close()\n'
            )
            completed = subprocess.run(
                [sys.executable, '-c', code], text=True, capture_output=True, timeout=5,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_lru_evicts_on_entry_or_byte_limit_without_affecting_loader_correctness(self):
        cache = ByteBoundedLRU(max_entries=2, max_bytes=12, sizeof=lambda value: len(value))
        cache['a'] = b'1234'
        cache['b'] = b'5678'
        cache['c'] = b'90ab'
        self.assertNotIn('a', cache)
        cache['large'] = b'x' * 20
        self.assertNotIn('large', cache)
        self.assertLessEqual(cache.current_bytes, 12)
        self.assertGreaterEqual(cache.evictions, 2)

    def test_exact_recall_does_not_build_search_provider_or_model_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            owner = RuntimeOwner(VaultConfig.resolve(directory, env={}), provider_factory=lambda: None)
            try:
                dispatcher = build_default_dispatcher(owner.config, runtime_owner=owner)
                result = dispatcher.dispatch(
                    'trove.recall',
                    {'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 5},
                    request_id='exact-recall-budget',
                )
                self.assertTrue(result['ok'])
                status = owner.status()
                self.assertEqual(status['search_runtime_builds'], 0)
                self.assertEqual(status['search_engine_builds'], 0)
                self.assertEqual(status['provider_builds'], 0)
                self.assertLessEqual(status['read_connections'], status['max_read_connections'])
            finally:
                owner.close()

    def test_same_query_builds_once_and_generation_change_rebuilds_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            owner = RuntimeOwner(VaultConfig.resolve(directory, env={}), provider_factory=lambda: None)
            try:
                request = SearchRequest('价格太高', limit=2, semantic='off')
                first = owner.search_with_metrics(request)[0]
                second = owner.search_with_metrics(request)[0]
                self.assertEqual(first.to_dict(), second.to_dict())
                self.assertEqual(owner.status()['search_engine_builds'], 1)

                data = generate_fixture()
                changed = replace(
                    data.messages[0], content='runtime owner generation sentinel',
                    shard_id='runtime-owner', local_id=9911,
                )
                index_fixture_data(
                    root, FixtureData(data.accounts, data.conversations, [*data.messages, changed]),
                    reset=True,
                )
                self.assertTrue(owner.search_with_metrics(SearchRequest(
                    'generation sentinel', limit=2, semantic='off',
                ))[0].results)
                owner.search_with_metrics(SearchRequest('generation sentinel', limit=2, semantic='off'))
                self.assertEqual(owner.status()['search_engine_builds'], 2)
            finally:
                owner.close()

    def test_idle_heavy_resources_release_and_restart_with_same_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            owner = RuntimeOwner(
                VaultConfig.resolve(directory, env={}), provider_factory=lambda: None,
                heavy_idle_seconds=0.05,
            )
            try:
                request = SearchRequest('价格太高', limit=2, semantic='off')
                before = owner.search_with_metrics(request)[0].to_dict()
                deadline = time.time() + 1
                while owner.status()['heavy_loaded'] and time.time() < deadline:
                    time.sleep(0.02)
                self.assertFalse(owner.status()['heavy_loaded'])
                after = owner.search_with_metrics(request)[0].to_dict()
                self.assertEqual(
                    [item['citation'] for item in before['results']],
                    [item['citation'] for item in after['results']],
                )
                self.assertEqual(owner.status()['search_runtime_builds'], 1)
                self.assertEqual(owner.status()['search_engine_builds'], 2)
            finally:
                owner.close()

    def test_mutation_marks_dirty_and_rebuilds_only_at_next_safe_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            owner = RuntimeOwner(VaultConfig.resolve(directory, env={}), provider_factory=lambda: None)
            try:
                request = SearchRequest('价格太高', limit=2, semantic='off')
                owner.search_with_metrics(request)
                owner.mark_generation_dirty()
                self.assertTrue(owner.status()['generation_dirty'])
                self.assertEqual(owner.status()['search_engine_builds'], 1)
                owner.search_with_metrics(request)
                self.assertFalse(owner.status()['generation_dirty'])
                self.assertEqual(owner.status()['search_engine_builds'], 2)
            finally:
                owner.close()

    def test_dispatcher_mutation_marks_owner_dirty_without_eager_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            config = VaultConfig.resolve(directory, env={})
            owner = RuntimeOwner(config, provider_factory=lambda: None)
            try:
                dispatcher = build_default_dispatcher(config, runtime_owner=owner)
                response = dispatcher.dispatch(
                    'trove.media_enrich',
                    {'citation': 'trove://fixture/account-a/message-1', 'kind': 'transcribe'},
                    request_id='mark-dirty',
                )
                self.assertTrue(response['ok'])
                self.assertTrue(owner.status()['generation_dirty'])
                self.assertEqual(owner.status()['search_runtime_builds'], 0)
            finally:
                owner.close()


if __name__ == '__main__':
    unittest.main()
