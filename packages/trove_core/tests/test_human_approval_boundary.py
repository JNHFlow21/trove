from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from trove_core.application.approval_control import ApprovalControl
from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.security.operator_confirm import OperatorConfirmationError, decide_from_controlling_terminal
from trove_core.wechat.indexer import index_fixture_vault
from trove_protocol.capabilities import CATALOG


class _NonTTY:
    def isatty(self):
        return False


class HumanApprovalBoundaryTests(unittest.TestCase):
    def test_catalog_dispatcher_and_agent_control_have_no_decision_capability(self):
        self.assertFalse(any('approve' in spec.capability_id or 'reject' in spec.capability_id for spec in CATALOG))
        self.assertFalse(hasattr(ApprovalControl, 'decide'))
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            payload = build_default_dispatcher(vault).dispatch(
                'trove.approval_decide', {'status': 'approved'}, request_id='req-decision',
            )
        self.assertEqual(payload['error']['code'], 'unknown_capability')

    def test_operator_path_has_no_yes_stdin_or_environment_bypass(self):
        parameters = inspect.signature(decide_from_controlling_terminal).parameters
        self.assertNotIn('yes', parameters)
        self.assertNotIn('stdin', parameters)
        self.assertNotIn('env', parameters)
        with self.assertRaises(OperatorConfirmationError):
            decide_from_controlling_terminal(
                object(), 'approval-fixture', 'approved', terminal=_NonTTY(),
            )


if __name__ == '__main__':
    unittest.main()
