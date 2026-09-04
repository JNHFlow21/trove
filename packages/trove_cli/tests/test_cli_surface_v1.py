from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from trove_client.control import control_reply_from_controlling_terminal
from trove_cli.parser import CLIInputError, build_parser, parse_args, public_routes
from trove_core.reply import ReplyServiceConfig
from trove_protocol.capabilities import CATALOG


EXPECTED_EXTRA = {
    ('start',), ('stop',), ('status',), ('version',),
    ('media', 'annotate'), ('media', 'status'),
    ('observe', 'propose'), ('provider', 'list'), ('approval', 'list'),
    ('operator', 'approve'), ('operator', 'reject'),
    ('operator', 'reply', 'arm'), ('operator', 'reply', 'disarm'),
    ('operator', 'reply', 'pair'),
    ('operator', 'reply', 'mode'),
    ('operator', 'reply', 'approve'), ('operator', 'reply', 'reject'),
}


class FakeTTY:
    def __init__(self, response: str):
        self.response = response
        self.output = ''

    def isatty(self):
        return True

    def write(self, value):
        self.output += value

    def flush(self):
        return None

    def readline(self):
        return self.response


class CLISurfaceV1Tests(unittest.TestCase):
    def test_every_catalog_leaf_exists_and_maps_to_exact_capability(self):
        mapped = {(route.path, route.spec.capability_id) for route in public_routes() if route.spec}
        for spec in CATALOG:
            self.assertIn((spec.cli_route, spec.capability_id), mapped)
        self.assertTrue(EXPECTED_EXTRA <= {route.path for route in public_routes()})

    def test_scoped_leaf_uses_account_flag_and_catalog_defaults(self):
        namespace, route, payload = parse_args([
            '--vault', '/tmp/vault', 'search', '--query', 'needle', '--account', 'account-a',
        ])
        self.assertEqual(route.spec.capability_id, 'trove.search')
        self.assertEqual(payload, {
            'query': 'needle', 'semantic': 'auto', 'limit': 10, 'account_id': 'account-a',
        })
        self.assertEqual(namespace.vault, '/tmp/vault')

    def test_media_leaf_fixes_kind_without_duplicate_flag(self):
        _, route, payload = parse_args(['media', 'annotate', '--citation', 'trove://source/a/c/s/1'])
        self.assertEqual(route.spec.capability_id, 'trove.media_enrich')
        self.assertEqual(payload['kind'], 'annotate')

    def test_legacy_hyphenated_routes_are_unknown_and_absent_from_help(self):
        parser = build_parser()
        for legacy in ('chat-recall', 'customer-profile', 'files-list', 'media-fetch', 'up', 'down', 'ps'):
            with self.subTest(legacy=legacy), self.assertRaises(CLIInputError):
                parser.parse_args([legacy])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exited:
            parser.parse_args(['--help'])
        self.assertEqual(exited.exception.code, 0)
        text = output.getvalue()
        for legacy in ('chat-recall', 'customer-profile', 'files-list', 'media-fetch', 'up', 'down', 'ps'):
            self.assertNotIn(f'    {legacy} ', text)

    def test_no_agent_facing_approval_decision_capability_exists(self):
        self.assertFalse(any('approve' in spec.capability_id or 'reject' in spec.capability_id for spec in CATALOG))
        operator = [route for route in public_routes() if route.operator_decision]
        self.assertEqual({route.path for route in operator}, {('operator', 'approve'), ('operator', 'reject')})
        self.assertTrue(all(route.spec is None for route in operator))
        reply_operator = [
            route for route in public_routes()
            if route.operator_reply_action
        ]
        self.assertEqual({
            route.path for route in reply_operator
        }, {
            ('operator', 'reply', 'arm'),
            ('operator', 'reply', 'disarm'),
            ('operator', 'reply', 'pair'),
            ('operator', 'reply', 'mode'),
            ('operator', 'reply', 'approve'),
            ('operator', 'reply', 'reject'),
        })
        self.assertTrue(all(route.spec is None for route in reply_operator))

    def test_reply_operator_review_route_requires_exact_review_id(self):
        namespace, route, payload = parse_args([
            'operator', 'reply', 'approve', 'review_fixture_0001',
        ])
        self.assertEqual(route.operator_reply_action, 'approve')
        self.assertEqual(namespace.review_id, 'review_fixture_0001')
        self.assertEqual(payload, {})

    def test_reply_operator_mode_route_is_bounded(self):
        namespace, route, payload = parse_args([
            'operator', 'reply', 'mode', 'review_queue',
        ])
        self.assertEqual(route.operator_reply_action, 'mode')
        self.assertEqual(namespace.mode, 'review_queue')
        self.assertEqual(payload, {})
        with self.assertRaises(CLIInputError):
            parse_args(['operator', 'reply', 'mode', 'invalid'])

    def test_reply_mode_change_requires_exact_tty_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            vault = Path(temporary) / 'vault'
            ReplyServiceConfig().save(vault)
            tty = FakeTTY('SET REPLY MODE shadow\n')

            result = control_reply_from_controlling_terminal(
                vault,
                'mode',
                mode='shadow',
                terminal=tty,
            )

            self.assertEqual(result['config']['mode'], 'shadow')
            self.assertTrue(result['daemon_restart_required'])
            self.assertIn('Type exactly: SET REPLY MODE shadow', tty.output)
            self.assertEqual(ReplyServiceConfig.load(vault).mode, 'shadow')

    def test_legacy_inventory_has_no_unreviewed_route_reuse(self):
        inventory = json.loads(
            (Path(__file__).resolve().parents[3] / 'tests/golden/trove_legacy_surface_inventory.json').read_text()
        )
        v1 = {' '.join(route.path) for route in public_routes()}
        legacy = {
            item['name'] for item in inventory['items']
            if item['surface'] == 'cli' and item['name'] not in v1
        }
        self.assertFalse(v1 & legacy)
        self.assertGreaterEqual(len(legacy), 70)


if __name__ == '__main__':
    unittest.main()
