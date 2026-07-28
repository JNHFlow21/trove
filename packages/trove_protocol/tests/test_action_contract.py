from __future__ import annotations

import json
import unittest

from trove_protocol.capabilities import catalog_snapshot
from trove_protocol.experimental.actions import (
    ActionContractError,
    ActionExecuteRequest,
    ActionPreflightRequest,
    IdempotentActionExecutor,
)
from trove_protocol.experimental.entitlements import EntitlementDecision
from trove_protocol.target import TargetRef


class ActionContractTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetRef(
            provider_id='fixture-source', account_id='account-a', kind='contact',
            stable_id='contact-1', conversation_id='conversation-1', peer_id='peer-1',
        )

    def test_preflight_cannot_execute_and_execute_requires_idempotency(self):
        request = ActionPreflightRequest(
            action='send_fixture', target=self.target, arguments={'text': 'fixture'},
        )
        self.assertFalse(hasattr(request, 'idempotency_key'))
        with self.assertRaises(ActionContractError):
            ActionExecuteRequest(
                action='send_fixture', target=self.target, arguments={'text': 'fixture'},
                preflight_token='preflight-fixture', idempotency_key='',
            )

    def test_duplicate_execute_returns_one_operation(self):
        calls = []
        executor = IdempotentActionExecutor(lambda request: calls.append(request) or 'operation-fixture')
        request = ActionExecuteRequest(
            action='send_fixture', target=self.target, arguments={'text': 'fixture'},
            preflight_token='preflight-fixture', idempotency_key=('idempotency-' + 'fixture-0001'),
        )
        self.assertEqual(executor.execute(request), 'operation-fixture')
        self.assertEqual(executor.execute(request), 'operation-fixture')
        self.assertEqual(len(calls), 1)

    def test_experimental_types_are_not_in_the_public_catalog(self):
        snapshot = json.dumps(catalog_snapshot(), ensure_ascii=False)
        self.assertNotIn('trove_action_', snapshot)
        self.assertNotIn('entitlement', snapshot.lower())
        self.assertFalse(EntitlementDecision.deny('fixture').allowed)


if __name__ == '__main__':
    unittest.main()
