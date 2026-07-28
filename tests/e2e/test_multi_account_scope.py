from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from trove_client import TroveClientError
from trove_core.wechat.fixture_factory import FixtureData, generate_fixture
from trove_core.wechat.models import Conversation, Message

from tests.e2e.runtime_harness import RuntimeHarness, citations


class MultiAccountScopeTests(unittest.TestCase):
    def test_default_aggregation_filter_account_inventory_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            fixture = generate_fixture()
            conversations = [
                *fixture.conversations,
                Conversation('conv-shared-work', 'acct-work', 'Shared Fixture Group', 'group', 1),
                Conversation('conv-shared-personal', 'acct-personal', 'Shared Fixture Group', 'group', 1),
            ]
            messages = [
                *fixture.messages,
                Message(
                    'acct-work', 'Work-WeChat', 'conv-shared-work', 'Shared Fixture Group',
                    'group', 'work-peer', 'Work peer',
                    datetime(2026, 6, 21, 9, 0, tzinfo=timezone.utc),
                    'crossaccountneedle work evidence', 'message_9', 1,
                ),
                Message(
                    'acct-personal', 'Personal-WeChat', 'conv-shared-personal', 'Shared Fixture Group',
                    'group', 'personal-peer', 'Personal peer',
                    datetime(2026, 6, 21, 9, 1, tzinfo=timezone.utc),
                    'crossaccountneedle personal evidence', 'message_9', 1,
                ),
            ]
            runtime = RuntimeHarness(
                vault, with_media=False,
                fixture_data=FixtureData(fixture.accounts, conversations, messages),
            )
            with runtime:
                inventory = runtime.call('trove.resolve', {'kind': 'account'}, 'account-inventory')
                by_account = {item['account_id']: item for item in inventory['data']['accounts']}
                self.assertEqual(set(by_account), {'acct-work', 'acct-personal'})
                self.assertGreater(by_account['acct-work']['message_count'], 0)
                self.assertGreater(by_account['acct-personal']['message_count'], 0)

                aggregated = runtime.call(
                    'trove.search',
                    {'query': 'crossaccountneedle', 'semantic': 'off', 'limit': 10},
                    'cross-account-search',
                )
                results = aggregated['data']['results']
                self.assertEqual({item['account_id'] for item in results}, {'acct-work', 'acct-personal'})
                self.assertTrue(all('/acct-' in citation for citation in citations(results)))

                scoped = runtime.call(
                    'trove.search',
                    {
                        'query': 'crossaccountneedle', 'semantic': 'off',
                        'account_id': 'acct-work', 'limit': 10,
                    },
                    'work-only-search',
                )
                self.assertEqual(
                    {item['account_id'] for item in scoped['data']['results']}, {'acct-work'},
                )

                with self.assertRaises(TroveClientError) as ambiguous:
                    runtime.call(
                        'trove.resolve', {'target': 'Shared Fixture Group'},
                        'ambiguous-group',
                    )
                self.assertEqual(ambiguous.exception.code, 'ambiguous_target')
                details = ambiguous.exception.response['error']['details']
                candidates = details.get('candidates') or details.get('matches') or []
                self.assertEqual(
                    {item['account_id'] for item in candidates},
                    {'acct-work', 'acct-personal'},
                )


if __name__ == '__main__':
    unittest.main()
