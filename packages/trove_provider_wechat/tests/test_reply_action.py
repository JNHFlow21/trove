from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from trove_protocol.provider import ProviderManifest
from trove_provider_wechat import create_provider
from trove_provider_wechat.reply import (
    ContactIdentity,
    LiveMessage,
    SendOutcome,
    SenderReadiness,
    WeChatActionError,
    WeChatLiveConfig,
)


PACKAGE = Path(__file__).resolve().parents[1] / 'trove_provider_wechat'


class FakeLiveSource:
    def __init__(self) -> None:
        self.latest_position = 7
        self.echo: LiveMessage | None = None
        self.advance_during_echo_wait = False
        self.identity = ContactIdentity(
            target_id='raw-target',
            target_ref=hashlib.sha256(b'raw-target').hexdigest(),
            search_query='unique-search',
            header_candidates=('Expected',),
            unique_search=True,
        )

    def events(self, cursors, *, observed_at):
        del cursors
        return {
            'events': [{
                'event_id': 'event-fixture-7',
                'account_id': 'account-fixture',
                'conversation_id': self.identity.target_ref,
                'target_ref': self.identity.target_ref,
                'source_position': 7,
                'latest_fingerprint': 'f' * 64,
                'observed_at': observed_at,
                'messages': [{
                    'citation': f'provider://wechat/live/{self.identity.target_ref}/7',
                    'source_position': 7,
                    'observed_at': observed_at,
                    'kind': 'text',
                    'text': 'hello',
                    'trust': 'untrusted_evidence',
                }],
            }],
            'acknowledgements': [],
        }

    def resolve_identity(self, target_ref):
        if target_ref != self.identity.target_ref:
            raise RuntimeError('target_ref_not_found')
        return self.identity

    def current_position(self, target_id):
        self.assert_target(target_id)
        return self.latest_position

    def wait_for_outgoing_echo(
        self,
        target_id,
        *,
        after_source_position,
        expected_text,
        timeout_seconds,
    ):
        del after_source_position, expected_text, timeout_seconds
        self.assert_target(target_id)
        if self.advance_during_echo_wait:
            self.latest_position += 1
        return self.echo

    @staticmethod
    def assert_target(target_id):
        if target_id != 'raw-target':
            raise RuntimeError('target_mismatch')


class FakeSender:
    def __init__(self, outcome: SendOutcome) -> None:
        self.outcome = outcome
        self.calls = []

    def readiness(self):
        return SenderReadiness(True, True, app_pid=123)

    def send(self, identity, text, *, after_source_position, shortcut=None):
        self.calls.append((identity, text, after_source_position, shortcut))
        return self.outcome


def manifest() -> ProviderManifest:
    return ProviderManifest.from_dict(
        json.loads((PACKAGE / 'manifest.json').read_text(encoding='utf-8'))
    )


def payload(target_ref: str, **updates):
    text = updates.pop('text', 'reply')
    base = {
        'operation': 'send',
        'operation_id': 'send-fixture-0001',
        'idempotency_key': ('fixture-' + 'idempotency-0001'),
        'account_id': 'account-fixture',
        'target_ref': target_ref,
        'expected_source_position': 7,
        'draft_digest': hashlib.sha256(text.encode()).hexdigest(),
        'text': text,
    }
    return base | updates


class WeChatReplyActionTests(unittest.TestCase):
    def setUp(self):
        self.live = FakeLiveSource()
        self.sender = FakeSender(
            SendOutcome('completed', 'server_ack_verified', self.live.identity.target_ref, 123, 8)
        )
        self.config = WeChatLiveConfig(
            account_id='account-fixture',
            account_id_sha256=hashlib.sha256(
                b'account-fixture',
            ).hexdigest(),
            enabled=True,
            send_shortcut='return',
        )
        self.provider = create_provider(
            manifest(),
            live_config=self.config,
            live_source=self.live,
            sender=self.sender,
        )

    def test_manifest_privilege_and_secret_name_are_exact(self):
        item = manifest()
        self.assertEqual(item.capabilities, ('read', 'media', 'action'))
        self.assertEqual(item.secret_names, ('TROVE_WECHAT_KEY_STORE',))

    def test_live_events_are_normalized_untrusted_evidence(self):
        result = self.provider.invoke('action', {
            'operation': 'events',
            'account_id': 'account-fixture',
            'cursors': {},
            'observed_at': 1_000.0,
        })
        self.assertEqual(result['events'][0]['source_position'], 7)
        self.assertEqual(
            result['events'][0]['messages'][0]['trust'], 'untrusted_evidence',
        )

    def test_wrong_account_target_watermark_or_digest_never_reaches_sender(self):
        cases = (
            payload(self.live.identity.target_ref, account_id='other-account'),
            payload('b' * 64),
            payload(self.live.identity.target_ref, expected_source_position=6),
            payload(self.live.identity.target_ref, draft_digest='c' * 64),
        )
        for item in cases:
            with self.subTest(item=item):
                with self.assertRaises(WeChatActionError):
                    self.provider.invoke('action', item)
        self.assertEqual(self.sender.calls, [])

    def test_send_success_requires_exact_server_ack_outcome(self):
        result = self.provider.invoke(
            'action', payload(self.live.identity.target_ref),
        )
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['proof']['source_position'], 8)
        self.assertEqual(result['proof']['remote_ack'], True)
        self.assertEqual(len(self.sender.calls), 1)

    def test_unknown_outcome_stays_unknown(self):
        self.sender.outcome = SendOutcome(
            'unknown', 'send_event_without_server_ack',
            self.live.identity.target_ref, 123,
        )
        result = self.provider.invoke(
            'action', payload(self.live.identity.target_ref),
        )
        self.assertEqual(result['state'], 'unknown')
        self.assertNotIn('proof', result)

    def test_reconciliation_after_lost_response_never_sends_again(self):
        self.live.latest_position = 8
        self.live.echo = LiveMessage(
            target_id='raw-target',
            target_ref=self.live.identity.target_ref,
            source_name='message-fixture',
            source_position=8,
            server_id='server-8',
            local_type=1,
            create_time=1_001,
            is_outgoing=True,
            text='reply',
        )
        result = self.provider.invoke('action', {
            **payload(self.live.identity.target_ref),
            'operation': 'reconcile',
        })
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['proof']['source_position'], 8)
        self.assertEqual(self.sender.calls, [])

    def test_retry_preflight_requires_unchanged_source_and_no_prior_echo(self):
        result = self.provider.invoke('action', {
            **payload(self.live.identity.target_ref),
            'operation': 'retry_preflight',
        })
        self.assertEqual(result['state'], 'ready')
        self.assertEqual(result['stage'], 'retry_preflight_passed')
        self.assertEqual(result['operation_id'], 'send-fixture-0001')
        self.assertEqual(
            result['idempotency_key'], ('fixture-' + 'idempotency-0001'),
        )
        self.assertEqual(result['expected_source_position'], 7)
        self.assertEqual(
            result['draft_digest'],
            hashlib.sha256(b'reply').hexdigest(),
        )
        self.assertEqual(self.sender.calls, [])

        self.live.latest_position = 8
        with self.assertRaises(WeChatActionError):
            self.provider.invoke('action', {
                **payload(self.live.identity.target_ref),
                'operation': 'retry_preflight',
            })

    def test_retry_preflight_rechecks_source_after_echo_wait(self):
        self.live.advance_during_echo_wait = True

        with self.assertRaisesRegex(
            WeChatActionError,
            'source_position_advanced',
        ):
            self.provider.invoke('action', {
                **payload(self.live.identity.target_ref),
                'operation': 'retry_preflight',
            })
        self.assertEqual(self.sender.calls, [])


if __name__ == '__main__':
    unittest.main()
