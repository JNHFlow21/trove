from __future__ import annotations

import unittest

from trove_protocol.cursors import CursorError, CursorStore


class CursorContractTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.store = CursorStore(ttl_seconds=30, max_entries=4, clock=lambda: self.now)
        self.handle = self.store.issue(
            capability='trove.recall', filters={'account_id': 'acct-fixture'},
            keyset={'timestamp': 'fixture'}, high_water='12',
            generation='generation-a', vault_identity='vault-a',
        )

    def test_cursor_is_opaque_and_bound_to_server_state(self):
        self.assertNotIn('acct-fixture', self.handle)
        state = self.store.resolve(
            self.handle, capability='trove.recall',
            filters={'account_id': 'acct-fixture'}, generation='generation-a',
            vault_identity='vault-a',
        )
        self.assertEqual(state.keyset, {'timestamp': 'fixture'})
        self.assertEqual(state.high_water, '12')

    def test_tamper_wrong_binding_expiry_and_generation_change_are_typed(self):
        cases = (
            (self.handle + 'x', {}, 'cursor_invalid'),
            (self.handle, {'capability': 'trove.search'}, 'cursor_mismatch'),
            (self.handle, {'filters': {'account_id': 'other'}}, 'cursor_mismatch'),
            (self.handle, {'generation': 'generation-b'}, 'cursor_stale'),
            (self.handle, {'vault_identity': 'vault-b'}, 'cursor_mismatch'),
        )
        base = {
            'capability': 'trove.recall', 'filters': {'account_id': 'acct-fixture'},
            'generation': 'generation-a', 'vault_identity': 'vault-a',
        }
        for handle, changes, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(CursorError) as raised:
                    self.store.resolve(handle, **(base | changes))
                self.assertEqual(raised.exception.code, code)
        self.now += 31
        with self.assertRaises(CursorError) as raised:
            self.store.resolve(self.handle, **base)
        self.assertEqual(raised.exception.code, 'cursor_expired')


if __name__ == '__main__':
    unittest.main()
