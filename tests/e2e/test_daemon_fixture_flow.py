from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.e2e.runtime_harness import RuntimeHarness, citations


class DaemonFixtureFlowTests(unittest.TestCase):
    def test_public_query_media_observation_and_operation_flow(self):
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(Path(directory) / 'vault') as runtime:
            accounts = runtime.call('trove.resolve', {'kind': 'account'}, 'accounts')
            self.assertEqual(
                {item['account_id'] for item in accounts['data']['accounts']},
                {'acct-personal', 'acct-work'},
            )

            recall = runtime.call(
                'trove.recall',
                {'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 20},
                'recall',
            )
            self.assertEqual(recall['coverage']['state'], 'complete')
            self.assertTrue(citations(recall['data']))

            summary = runtime.call(
                'trove.group_summary',
                {'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 20},
                'group',
            )
            self.assertTrue(citations(summary['data']))

            search = runtime.call(
                'trove.search',
                {'query': '客户卡在哪', 'account_id': 'acct-work', 'semantic': 'off', 'limit': 5},
                'search',
            )
            citation = citations(search['data'])[0]
            context = runtime.call(
                'trove.context', {'citation': citation, 'before': 2, 'after': 2}, 'context',
            )
            self.assertTrue(citations(context['data']))

            profile = runtime.call(
                'trove.profile', {'target': '示例教育', 'account_id': 'acct-work', 'limit': 5}, 'profile',
            )
            self.assertTrue(profile['ok'])

            files = runtime.call(
                'trove.files_list',
                {
                    'account_id': 'acct-work', 'conversation_id': 'conv-sales-review',
                    'media_types': ['image'], 'limit': 20,
                },
                'files',
            )
            self.assertGreaterEqual(files['data']['count'], 1)
            selected = next(
                item['citation'] for item in files['data']['files']
                if item['asset_id'] == 'asset-agent-flow-image'
            )
            fetched = runtime.call(
                'trove.media_fetch', {'citation': selected, 'allow_remote': False}, 'fetch',
            )
            self.assertTrue(fetched['data']['evidence_ok'])
            self.assertEqual(fetched['data']['status'], 'available')
            self.assertEqual(fetched['data']['mime'], 'image/png')

            enrich = runtime.call(
                'trove.media_enrich', {'citation': selected, 'kind': 'annotate'}, 'enrich',
            )
            operation_id = enrich['data']['operation']['operation_id']
            status = runtime.call(
                'trove.operation_status', {'operation_id': operation_id}, 'operation-status',
            )
            self.assertEqual(status['data']['operation']['state'], 'pending')

            observed = runtime.call(
                'trove.observe_add',
                {
                    'target': '示例教育', 'text': 'synthetic reviewed note',
                    'idempotency_key': ('fixture-' + 'observation-0001'),
                },
                'observe',
            )
            replay = runtime.call(
                'trove.observe_add',
                {
                    'target': '示例教育', 'text': 'synthetic reviewed note',
                    'idempotency_key': ('fixture-' + 'observation-0001'),
                },
                'observe-replay',
            )
            self.assertEqual(
                observed['data']['observation']['observation_id'],
                replay['data']['observation']['observation_id'],
            )
            listed = runtime.call(
                'trove.observe_list', {'target': '示例教育', 'limit': 10}, 'observe-list',
            )
            self.assertEqual(listed['data']['count'], 1)

    def test_agent_continuation_is_opaque_single_use(self):
        with tempfile.TemporaryDirectory() as directory, RuntimeHarness(Path(directory) / 'vault') as runtime:
            started = runtime.call(
                'trove.media_enrich',
                {'citation': 'trove://fixture/acct/item', 'kind': 'transcribe'},
                'continue-start',
            )
            operation_id = started['data']['operation']['operation_id']
            _waiting, token = runtime.dispatcher.context.operations.await_agent(
                operation_id, stage='awaiting_fixture_annotation',
            )
            runtime.dispatcher.context.continuations[operation_id] = (
                lambda _record, payload: {'accepted': payload.get('accepted') is True}
            )
            continued = runtime.call(
                'trove.operation_continue',
                {'operation_id': operation_id, 'token': token, 'payload': {'accepted': True}},
                'continue-once',
            )
            self.assertEqual(continued['data']['operation']['state'], 'completed')
            with self.assertRaises(Exception) as caught:
                runtime.call(
                    'trove.operation_continue',
                    {'operation_id': operation_id, 'token': token, 'payload': {'accepted': True}},
                    'continue-twice',
                )
            self.assertEqual(caught.exception.code, 'operation_conflict')


if __name__ == '__main__':
    unittest.main()
