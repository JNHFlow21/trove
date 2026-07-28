from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers.system import _ACCOUNT_SUMMARY_SQL
from trove_core.approvals import ApprovalManager
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault
from trove_protocol.capabilities import CATALOG


class CapabilityDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_vault(self.vault, reset=True)
        self.dispatcher = build_default_dispatcher(self.vault)

    def tearDown(self):
        self.temp.cleanup()

    def test_catalog_drives_one_complete_handler_graph(self):
        self.assertEqual(set(self.dispatcher.handlers), {spec.capability_id for spec in CATALOG})

    def test_account_summary_preaggregates_each_large_table_before_joining(self):
        store = SQLiteStore(VaultConfig.resolve(str(self.vault)).paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as connection:
                plan = ' | '.join(
                    str(row['detail'])
                    for row in connection.execute('EXPLAIN QUERY PLAN ' + _ACCOUNT_SUMMARY_SQL)
                )
        finally:
            store.close()
        self.assertIn('MATERIALIZE message_summary', plan)
        self.assertIn('MATERIALIZE sync_summary', plan)
        self.assertNotIn('USE TEMP B-TREE FOR count(DISTINCT)', plan)

        response = self.dispatcher.dispatch('trove.capabilities', {}, request_id='req-catalog')
        self.assertTrue(response['ok'])
        self.assertTrue(response['data']['accounts'])

    def test_recall_returns_canonical_cited_envelope_without_starting_search(self):
        with mock.patch('trove_core.runtime.build_search_engine', side_effect=AssertionError('search runtime started')):
            payload = self.dispatcher.dispatch(
                'trove.recall',
                {'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 20},
                request_id='req-recall',
            )
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['protocol'], 'trove/1')
        self.assertEqual(payload['coverage']['state'], 'complete')
        self.assertTrue(payload['data']['messages'])
        self.assertTrue(all(item['account_id'] == 'acct-work' for item in payload['data']['messages']))
        self.assertTrue(all(item['citation'] for item in payload['data']['messages']))

    def test_resolve_returns_typed_cross_account_ambiguity(self):
        payload = self.dispatcher.dispatch(
            'trove.resolve', {'target': 'fixture-does-not-exist'}, request_id='req-resolve',
        )
        self.assertFalse(payload['ok'])
        self.assertEqual(payload['error']['code'], 'no_results')
        self.assertEqual(payload['error']['retryable'], False)

    def test_unknown_capability_and_input_fail_typed(self):
        unknown = self.dispatcher.dispatch('trove.unknown', {}, request_id='req-unknown')
        invalid = self.dispatcher.dispatch(
            'trove.recall', {'forged': True}, request_id='req-invalid',
        )
        self.assertEqual(unknown['error']['code'], 'unknown_capability')
        self.assertEqual(invalid['error']['code'], 'invalid_request')

    def test_capability_catalog_reports_only_real_handlers_as_available(self):
        response = self.dispatcher.dispatch('trove.capabilities', {}, request_id='req-availability')
        availability = {
            item['id']: item['available']
            for item in response['data']['capabilities']
        }
        self.assertTrue(availability['trove.files_export'])
        self.assertFalse(availability['trove.provider_reload'])
        self.assertFalse(availability['trove.repair'])
        self.assertFalse(availability['trove.sync'])

    def test_sync_is_journaled_replay_safe_and_scoped_to_selected_accounts(self):
        class ProviderRegistry:
            def accounts(self, provider_id):
                self.provider_id = provider_id
                return [
                    {'account_id': 'acct-one', 'label': 'one'},
                    {'account_id': 'acct-two', 'label': 'two'},
                ]

            def status(self):
                return [{
                    'provider_id': 'wechat-source',
                    'capabilities': ['read', 'media'],
                }]

        class Owner:
            def __init__(self, config):
                self.config = config
                self.provider_registry = ProviderRegistry()
                self.provider_load_status = {'ok': True, 'provider_id': 'wechat-source'}

            def status(self):
                return {'provider_load': dict(self.provider_load_status)}

            def mark_generation_dirty(self):
                pass

        owner = Owner(VaultConfig.resolve(str(self.vault)))
        dispatcher = build_default_dispatcher(self.vault, runtime_owner=owner)
        expected = {
            'ok': True,
            'status': 'completed',
            'sources_seen': 1,
            'messages_imported': 2,
        }
        with mock.patch(
            'trove_core.application.commands.TroveCommands.sync',
            return_value=expected,
        ) as sync:
            first = dispatcher.dispatch(
                'trove.sync',
                {
                    'account_ids': ['acct-one'],
                    'full': False,
                    'idempotency_key': 'sync-fixture-key-0001',
                },
                request_id='req-sync-one',
            )
            replay = dispatcher.dispatch(
                'trove.sync',
                {
                    'account_ids': ['acct-one'],
                    'full': False,
                    'idempotency_key': 'sync-fixture-key-0001',
                },
                request_id='req-sync-two',
            )
        self.assertTrue(first['ok'])
        self.assertEqual(first['data']['operation']['state'], 'completed')
        self.assertEqual(first['data']['operation']['result']['status'], 'completed')
        self.assertTrue(replay['ok'])
        self.assertTrue(replay['data']['replayed'])
        self.assertEqual(sync.call_count, 1)
        command = sync.call_args.args[0]
        self.assertEqual(command.account_ids, ('acct-one',))
        self.assertEqual(command.profile_refresh_budget, 0)

        unavailable = dispatcher.dispatch(
            'trove.sync',
            {
                'account_ids': ['acct-missing'],
                'idempotency_key': 'sync-fixture-key-0002',
            },
            request_id='req-sync-missing',
        )
        self.assertFalse(unavailable['ok'])
        self.assertEqual(unavailable['error']['code'], 'account_scope_unavailable')

    def test_files_export_requires_and_consumes_exact_human_approval(self):
        destination = self.vault.parent / 'approved-export'
        request = {
            'selection': ['asset-fixture-one'],
            'destination': str(destination),
        }
        pending = self.dispatcher.dispatch(
            'trove.files_export', request, request_id='req-export-pending',
        )
        self.assertFalse(pending['ok'])
        self.assertEqual(pending['error']['code'], 'approval_required')
        approval_id = pending['error']['details']['approval']['approval_id']
        ApprovalManager(self.vault).decide(approval_id, 'approved', note='fixture operator')

        with mock.patch(
            'trove_core.application.sensitive_commands.execute_files_archive',
            return_value={
                'ok': True, 'copied': 1, 'raw_paths_included': False,
            },
        ) as execute:
            completed = self.dispatcher.dispatch(
                'trove.files_export',
                {**request, 'approval_id': approval_id},
                request_id='req-export-complete',
            )
        self.assertTrue(completed['ok'])
        self.assertEqual(completed['data']['copied'], 1)
        self.assertEqual(execute.call_args.kwargs['selection']['asset_ids'], ['asset-fixture-one'])


if __name__ == '__main__':
    unittest.main()
