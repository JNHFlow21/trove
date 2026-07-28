from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from trove_core.reply.models import (
    EvidenceMessage,
    ReplyDraft,
    ReplyEvent,
    RoundTiming,
    SendIntent,
    sha256_text,
)
from trove_core.reply.rounds import RoundCoordinator, adaptive_quiet_ms
from trove_core.reply.store import ReplyStore


def event(
    conversation: str,
    positions: tuple[int, ...],
    *,
    kind: str = 'text',
    now: float = 10.0,
) -> ReplyEvent:
    messages = tuple(
        EvidenceMessage(
            citation=f'trove://message/{conversation}-{position}',
            source_position=position,
            observed_at=now,
            kind=kind if position == positions[-1] else 'text',
            text='fixture' if kind == 'text' else None,
        )
        for position in positions
    )
    return ReplyEvent(
        event_id=f'evt_{conversation}_{positions[-1]}',
        account_id='account-fixture',
        conversation_id=conversation,
        target_ref=sha256_text(conversation),
        source_position=positions[-1],
        latest_fingerprint=sha256_text(f'{conversation}:{positions[-1]}'),
        messages=messages,
        observed_at=now,
    )


class ReplyRoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ReplyStore.for_vault(Path(self.temp.name) / 'vault')
        self.timing = RoundTiming(
            generation_prestart_ms=100,
            quiet_min_ms=100,
            quiet_default_ms=400,
            quiet_max_ms=800,
            max_collect_ms=1_500,
            max_messages=20,
        )
        self.rounds = RoundCoordinator(self.store, timing=self.timing)

    def tearDown(self):
        self.temp.cleanup()

    def test_adaptive_quiet_window_grows_with_fragments_and_media(self):
        self.assertEqual(adaptive_quiet_ms(1, 'text', self.timing), 400)
        self.assertEqual(adaptive_quiet_ms(2, 'text', self.timing), 533)
        self.assertEqual(adaptive_quiet_ms(3, 'text', self.timing), 666)
        self.assertEqual(adaptive_quiet_ms(5, 'text', self.timing), 800)
        self.assertEqual(adaptive_quiet_ms(1, 'image', self.timing), 666)

    def test_repeated_event_does_not_extend_but_newer_event_does(self):
        first = self.rounds.observe(event('a', (1,)), now=10.0)
        repeated = self.rounds.observe(event('a', (1,)), now=10.3)
        self.assertEqual(first.revision, repeated.revision)
        self.assertEqual(repeated.ready_at, 10.4)
        extended = self.rounds.observe(event('a', (1, 2)), now=10.3)
        self.assertEqual(extended.revision, 2)
        self.assertEqual(extended.ready_at, 10.833)
        self.assertEqual(extended.deadline_at, 11.5)

    def test_multiple_contacts_are_independent_and_fair(self):
        first = self.rounds.observe(event('a', (1,)), now=10.0)
        second = self.rounds.observe(event('b', (1,)), now=10.1)
        ready = self.rounds.ready(now=10.5)
        self.assertEqual([item.round_id for item in ready], [first.round_id, second.round_id])

    def test_new_message_stales_draft_review_and_prepared_send_atomically(self):
        round_record = self.rounds.observe(event('a', (1,)), now=10.0)
        draft = ReplyDraft(
            draft_id='draft_fixture_0001',
            round_id=round_record.round_id,
            round_revision=round_record.revision,
            account_id=round_record.account_id,
            conversation_id=round_record.conversation_id,
            target_ref=round_record.target_ref,
            source_position=round_record.source_position,
            context_digest=sha256_text('context'),
            text='reply',
            backend='fixture',
            model='fixture-model',
            created_at=10.1,
        )
        self.store.save_draft(draft)
        review = self.store.enqueue_review(draft.draft_id, now=10.2)
        self.store.decide_review(review.review_id, decision='approved', now=10.3)
        intent = SendIntent(
            operation_id='send_fixture_0001',
            idempotency_key=('fixture-' + 'idempotency-0001'),
            draft_id=draft.draft_id,
            account_id=draft.account_id,
            conversation_id=draft.conversation_id,
            target_ref=draft.target_ref,
            expected_source_position=draft.source_position,
            draft_digest=sha256_text(draft.text),
            text=draft.text,
            grant_ref=review.review_id,
        )
        operation, _ = self.store.prepare_send(intent, now=10.4)

        updated = self.rounds.observe(event('a', (1, 2)), now=10.5)

        self.assertEqual(updated.revision, 2)
        self.assertEqual(self.store.get_draft(draft.draft_id).state, 'stale')
        self.assertEqual(self.store.get_review(review.review_id).state, 'stale')
        self.assertEqual(self.store.get_send(operation.operation_id).state, 'cancelled')

    def test_retry_backoff_does_not_starve_other_ready_contact(self):
        first = self.rounds.observe(event('a', (1,)), now=10.0)
        second = self.rounds.observe(event('b', (1,)), now=10.0)
        self.rounds.fail(
            first.round_id,
            reason='generation_backend_unavailable',
            retryable=True,
            now=10.4,
        )
        self.assertEqual(
            [item.round_id for item in self.rounds.ready(now=10.5)],
            [second.round_id],
        )
        self.assertEqual(
            [item.round_id for item in self.rounds.ready(now=10.9)],
            [second.round_id, first.round_id],
        )


if __name__ == '__main__':
    unittest.main()
