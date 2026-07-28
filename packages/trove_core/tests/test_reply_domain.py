from __future__ import annotations

import unittest

from trove_core.reply.models import (
    EvidenceMessage,
    ReplyEvent,
    ReplyModelError,
    RoundTiming,
    SendIntent,
    sha256_text,
)


class ReplyDomainTests(unittest.TestCase):
    def test_default_timing_matches_product_contract(self):
        timing = RoundTiming()
        self.assertEqual(
            (
                timing.generation_prestart_ms,
                timing.quiet_min_ms,
                timing.quiet_default_ms,
                timing.quiet_max_ms,
                timing.max_collect_ms,
            ),
            (3_000, 6_000, 8_000, 15_000, 60_000),
        )

    def test_event_and_intent_are_bounded_and_exact(self):
        event = ReplyEvent(
            event_id='evt_fixture_0001',
            account_id='account-fixture',
            conversation_id='conversation-fixture',
            target_ref='a' * 64,
            source_position=7,
            latest_fingerprint=sha256_text('source-row-7'),
            messages=(
                EvidenceMessage(
                    citation='trove://message/fixture-7',
                    source_position=7,
                    observed_at=1_000.0,
                    kind='text',
                    text='hello',
                ),
            ),
            observed_at=1_000.0,
        )
        intent = SendIntent(
            operation_id='send_fixture_0001',
            idempotency_key=('fixture-' + 'idempotency-0001'),
            draft_id='draft_fixture_0001',
            account_id=event.account_id,
            conversation_id=event.conversation_id,
            target_ref=event.target_ref,
            expected_source_position=event.source_position,
            draft_digest=sha256_text('reply'),
            text='reply',
            grant_ref='review_fixture_0001',
        )
        self.assertEqual(intent.expected_source_position, event.source_position)

    def test_invalid_digest_and_unbounded_text_fail_closed(self):
        with self.assertRaises(ReplyModelError):
            SendIntent(
                operation_id='send_fixture_0002',
                idempotency_key=('fixture-' + 'idempotency-0002'),
                draft_id='draft_fixture_0002',
                account_id='account-fixture',
                conversation_id='conversation-fixture',
                target_ref='b' * 64,
                expected_source_position=8,
                draft_digest='not-a-digest',
                text='reply',
                grant_ref='review_fixture_0002',
            )
        with self.assertRaises(ReplyModelError):
            EvidenceMessage(
                citation='trove://message/fixture-8',
                source_position=8,
                observed_at=1_001.0,
                kind='text',
                text='x' * 8_001,
            )


if __name__ == '__main__':
    unittest.main()
