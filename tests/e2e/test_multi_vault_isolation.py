from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from trove_client import TroveClientError
from trove_core.wechat.fixture_factory import FixtureData, generate_fixture

from tests.e2e.runtime_harness import RuntimeHarness, citations


def _fixture_with_first_message(marker: str) -> FixtureData:
    fixture = generate_fixture()
    changed = replace(
        fixture.messages[0], content=f'{marker} isolated vault evidence',
    )
    return FixtureData(
        fixture.accounts, fixture.conversations, [changed, *fixture.messages[1:]],
    )


class MultiVaultIsolationTests(unittest.TestCase):
    def test_two_live_vaults_do_not_share_store_cache_operation_or_provider_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = RuntimeHarness(
                root / 'vault-a', with_media=False,
                fixture_data=_fixture_with_first_message('alphaonly'),
            )
            second = RuntimeHarness(
                root / 'vault-b', with_media=False,
                fixture_data=_fixture_with_first_message('betaonly'),
            )
            first.owner.attach_provider_registry(object(), {
                'ok': True, 'provider_id': 'fixture-a', 'generation': 'generation-a',
                'pure_vault_read_available': True,
            })
            second.owner.attach_provider_registry(object(), {
                'ok': True, 'provider_id': 'fixture-b', 'generation': 'generation-b',
                'pure_vault_read_available': True,
            })
            with first, second:
                alpha = first.call(
                    'trove.search', {'query': 'alphaonly', 'semantic': 'off', 'limit': 5}, 'alpha',
                )
                self.assertTrue(citations(alpha['data']))
                with self.assertRaises(TroveClientError) as beta_miss:
                    second.call(
                        'trove.search', {'query': 'alphaonly', 'semantic': 'off', 'limit': 5},
                        'beta-miss',
                    )
                self.assertEqual(beta_miss.exception.code, 'no_results')

                started = first.call(
                    'trove.media_enrich',
                    {'citation': 'trove://fixture/a/media', 'kind': 'annotate'},
                    'vault-a-operation',
                )
                operation_id = started['data']['operation']['operation_id']
                with self.assertRaises(TroveClientError) as missing:
                    second.call(
                        'trove.operation_status', {'operation_id': operation_id},
                        'vault-b-operation',
                    )
                self.assertEqual(missing.exception.code, 'operation_not_found')

                first_provider = first.call('trove.provider_status', {}, 'provider-a')
                second_provider = second.call('trove.provider_status', {}, 'provider-b')
                self.assertEqual(first_provider['data']['provider']['generation'], 'generation-a')
                self.assertEqual(second_provider['data']['provider']['generation'], 'generation-b')

                self.assertNotEqual(first.identity.socket_path, second.identity.socket_path)
                self.assertNotEqual(first.identity.cache_dir, second.identity.cache_dir)
                self.assertNotEqual(first.server.restart_id, second.server.restart_id)
                first_handle = first.server.cursors.issue(
                    capability='trove.search', filters={'query': 'alphaonly'},
                    keyset={'row': 1}, high_water='1', generation='g-a',
                )
                with self.assertRaises(Exception):
                    second.server.cursors.resolve(
                        first_handle, capability='trove.search',
                        filters={'query': 'alphaonly'}, generation='g-a',
                    )


if __name__ == '__main__':
    unittest.main()
