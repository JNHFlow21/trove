from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from trove_core.reply.models import ReplyDraft, RoundRecord, SendIntent, sha256_text
from trove_core.reply.store import ReplyStore, ReplyStoreConflict


class ReplyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vault_root = Path(self.temp.name) / 'vault'
        self.store = ReplyStore.for_vault(self.vault_root)

    def tearDown(self):
        self.temp.cleanup()

    def _draft(self, *, source_position: int = 7) -> ReplyDraft:
        return ReplyDraft(
            draft_id='draft_fixture_0001',
            round_id='round_fixture_0001',
            round_revision=1,
            account_id='account-fixture',
            conversation_id='conversation-fixture',
            target_ref='a' * 64,
            source_position=source_position,
            context_digest=sha256_text('context'),
            text='reply',
            backend='fixture',
            model='fixture-model',
            created_at=1_001.0,
        )

    def _seed_round(self) -> RoundRecord:
        return self.store.save_round(RoundRecord(
            round_id='round_fixture_0001',
            account_id='account-fixture',
            conversation_id='conversation-fixture',
            target_ref='a' * 64,
            first_seen_at=1_000.0,
            last_extended_at=1_000.0,
            preparation_at=1_001.0,
            earliest_ready_at=1_001.0,
            ready_at=1_001.0,
            deadline_at=1_060.0,
            quiet_target_ms=8_000,
            source_position=7,
            latest_fingerprint='f' * 64,
            inbound_message_count=1,
            latest_kind='text',
            revision=1,
        ))

    def _intent(self, grant_ref: str) -> SendIntent:
        return SendIntent(
            operation_id='send_fixture_0001',
            idempotency_key=('fixture-' + 'idempotency-0001'),
            draft_id='draft_fixture_0001',
            account_id='account-fixture',
            conversation_id='conversation-fixture',
            target_ref='a' * 64,
            expected_source_position=7,
            draft_digest=sha256_text('reply'),
            text='reply',
            grant_ref=grant_ref,
        )

    def test_schema_and_parent_are_owner_only(self):
        self.assertEqual(self.store.schema_version(), 1)
        self.assertEqual(os.stat(self.store.path.parent).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.store.path).st_mode & 0o777, 0o600)

    def test_restart_recovers_draft_review_and_prepared_send(self):
        self._seed_round()
        self.store.save_draft(self._draft())
        review = self.store.enqueue_review('draft_fixture_0001', now=1_002.0)
        approved = self.store.decide_review(
            review.review_id, decision='approved', now=1_003.0,
        )
        operation, replayed = self.store.prepare_send(
            self._intent(approved.review_id), now=1_004.0,
        )
        reopened = ReplyStore(self.store.path)
        self.assertEqual(reopened.get_draft('draft_fixture_0001').state, 'approved')
        self.assertEqual(reopened.get_review(approved.review_id).state, 'approved')
        self.assertEqual(reopened.get_send(operation.operation_id).state, 'prepared')
        self.assertFalse(replayed)

    def test_idempotency_binds_exact_intent(self):
        self._seed_round()
        self.store.save_draft(self._draft())
        review = self.store.enqueue_review('draft_fixture_0001', now=1_002.0)
        self.store.decide_review(review.review_id, decision='approved', now=1_003.0)
        intent = self._intent(review.review_id)
        first, replayed = self.store.prepare_send(intent, now=1_004.0)
        second, replayed_second = self.store.prepare_send(intent, now=1_005.0)
        self.assertFalse(replayed)
        self.assertTrue(replayed_second)
        self.assertEqual(first.operation_id, second.operation_id)
        changed = SendIntent(
            **{**intent.__dict__, 'grant_ref': 'review_fixture_changed'}
        )
        with self.assertRaises(ReplyStoreConflict):
            self.store.prepare_send(changed, now=1_006.0)

    def test_unknown_send_is_terminal_and_never_retryable(self):
        self._seed_round()
        self.store.save_draft(self._draft())
        review = self.store.enqueue_review('draft_fixture_0001', now=1_002.0)
        self.store.decide_review(review.review_id, decision='approved', now=1_003.0)
        operation, _ = self.store.prepare_send(
            self._intent(review.review_id), now=1_004.0,
        )
        self.store.mark_dispatched(
            operation.operation_id, external_ref='provider-ref-fixture', now=1_005.0,
        )
        terminal = self.store.finish_send(
            operation.operation_id,
            state='unknown',
            stage='ack_not_observed',
            now=1_006.0,
            error_code='remote_ack_unknown',
        )
        self.assertTrue(terminal.terminal)
        self.assertFalse(terminal.retryable)
        with self.assertRaises(ReplyStoreConflict):
            self.store.mark_dispatched(
                terminal.operation_id,
                external_ref='provider-ref-fixture',
                now=1_007.0,
            )

    def test_explicit_failed_send_retry_preserves_exact_intent(self):
        self._seed_round()
        self.store.save_draft(self._draft())
        review = self.store.enqueue_review('draft_fixture_0001', now=1_002.0)
        self.store.decide_review(
            review.review_id, decision='approved', now=1_003.0,
        )
        operation, _ = self.store.prepare_send(
            self._intent(review.review_id), now=1_004.0,
        )
        self.store.mark_dispatched(
            operation.operation_id,
            external_ref='provider-ref-fixture',
            now=1_005.0,
        )
        failed = self.store.finish_send(
            operation.operation_id,
            state='failed',
            stage='verify_draft_RuntimeError:copy_event_not_delivered',
            now=1_006.0,
            error_code='provider_send_failed',
        )

        reopened = self.store.reopen_failed_send(
            failed.operation_id,
            review_id=review.review_id,
            expected_stage=failed.stage,
            now=1_007.0,
        )

        self.assertEqual(reopened.state, 'prepared')
        self.assertEqual(reopened.stage, 'retry_authorized')
        self.assertEqual(reopened.retry_count, 1)
        self.assertEqual(
            self.store.get_send_intent(reopened.operation_id),
            self._intent(review.review_id),
        )

    def test_unknown_send_can_never_be_reopened(self):
        self._seed_round()
        self.store.save_draft(self._draft())
        review = self.store.enqueue_review('draft_fixture_0001', now=1_002.0)
        self.store.decide_review(
            review.review_id, decision='approved', now=1_003.0,
        )
        operation, _ = self.store.prepare_send(
            self._intent(review.review_id), now=1_004.0,
        )
        self.store.mark_dispatched(
            operation.operation_id,
            external_ref='provider-ref-fixture',
            now=1_005.0,
        )
        unknown = self.store.finish_send(
            operation.operation_id,
            state='unknown',
            stage='send_event_without_server_ack',
            now=1_006.0,
            error_code='provider_send_unknown',
        )

        with self.assertRaises(ReplyStoreConflict):
            self.store.reopen_failed_send(
                unknown.operation_id,
                review_id=review.review_id,
                expected_stage=unknown.stage,
                now=1_007.0,
            )


if __name__ == '__main__':
    unittest.main()
