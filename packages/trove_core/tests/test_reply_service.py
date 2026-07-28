from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
import unittest

from trove_core.reply.models import ReplyDraft
from trove_core.reply.service import ReplyService, ReplyServiceConfig
from trove_core.reply.store import ReplyStore


SOURCE_ACCOUNT = 'account-fixture'
ACCOUNT_HASH = hashlib.sha256(SOURCE_ACCOUNT.encode('utf-8')).hexdigest()
CONVERSATION_NAMESPACE = 'com.tencent.xinWeChat2__account-fixture'
ACCOUNT = (
    'acct-'
    + hashlib.sha256(CONVERSATION_NAMESPACE.encode('utf-8')).hexdigest()[:12]
)
TARGET = 'a' * 64


class Clock:
    def __init__(self, value: float = 1_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeGeneration:
    def __init__(self, store: ReplyStore):
        self.store = store
        self.calls = 0

    def generate(self, record, _event):
        self.calls += 1
        return self.store.save_draft(ReplyDraft(
            draft_id=f'draft_fixture_{record.revision}',
            round_id=record.round_id,
            round_revision=record.revision,
            account_id=record.account_id,
            conversation_id=record.conversation_id,
            target_ref=record.target_ref,
            source_position=record.source_position,
            context_digest='c' * 64,
            text='好的，可以',
            backend='fixture',
            model='fixture-model',
            created_at=1_001.0,
        ))


class FakeAction:
    def __init__(self):
        self.events: list[dict] = []
        self.send_results: list[dict] = []
        self.preflight_hook = None
        self.calls: list[str] = []
        self.send_calls = 0
        self.reconcile_calls = 0

    def __call__(self, payload):
        operation = payload['operation']
        self.calls.append(operation)
        if operation == 'status':
            return {'state': 'ready', 'ready': True}
        if operation == 'events':
            events, self.events = self.events, []
            return {
                'account_id': ACCOUNT,
                'events': events,
                'acknowledgements': [],
            }
        if operation == 'send':
            self.send_calls += 1
            if self.send_results:
                return self.send_results.pop(0)
            position = payload['expected_source_position'] + 1
            return {
                'state': 'completed',
                'stage': 'server_ack_verified',
                'proof': {
                    'source_position': position,
                    'remote_ack': True,
                    'text_sha256': payload['draft_digest'],
                },
            }
        if operation == 'reconcile':
            self.reconcile_calls += 1
            return {
                'state': 'unknown',
                'stage': 'server_ack_not_observed',
            }
        if operation == 'retry_preflight':
            if self.preflight_hook is not None:
                self.preflight_hook()
            return {
                'state': 'ready',
                'stage': 'retry_preflight_passed',
                'operation_id': payload['operation_id'],
                'idempotency_key': payload['idempotency_key'],
                'target_ref': payload['target_ref'],
                'expected_source_position':
                    payload['expected_source_position'],
                'draft_digest': payload['draft_digest'],
            }
        raise AssertionError(operation)


def event(position: int = 7) -> dict:
    return {
        'event_id': f'event-fixture-{position}',
        'account_id': ACCOUNT,
        'conversation_id': 'conversation-fixture',
        'target_ref': TARGET,
        'source_position': position,
        'latest_fingerprint': hashlib.sha256(
            f'row-{position}'.encode('utf-8')
        ).hexdigest(),
        'messages': [{
            'citation': f'provider://fixture/{TARGET}/{position}',
            'source_position': position,
            'observed_at': 1_000.0,
            'kind': 'text',
            'text': '需要合作吗？',
            'trust': 'untrusted_evidence',
        }],
        'observed_at': 1_000.0,
    }


class ReplyServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        self.store = ReplyStore.for_vault(self.vault)
        self.clock = Clock()
        self.action = FakeAction()
        self.generation = FakeGeneration(self.store)
        self.config = ReplyServiceConfig(
            armed=True,
            mode='review_queue',
            account_id=ACCOUNT,
            source_account_id=SOURCE_ACCOUNT,
            account_id_sha256=ACCOUNT_HASH,
            conversation_namespace=CONVERSATION_NAMESPACE,
            send_shortcut='return',
            target_scope='allowlist',
            allowed_target_refs=(TARGET,),
            poll_seconds=1.0,
            generation_prestart_ms=1_000,
            round_quiet_min_ms=1_000,
            round_quiet_default_ms=1_000,
            round_quiet_max_ms=1_000,
            round_max_collect_ms=1_000,
        )
        self.service = ReplyService(
            self.vault,
            self.config,
            action=self.action,
            generation=self.generation,
            store=self.store,
            now=self.clock,
        )

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def _finish_generation(self):
        deadline = time.time() + 2
        while time.time() < deadline:
            self.service.tick()
            if self.store.list_reviews(state='pending'):
                return
            time.sleep(0.01)
        self.fail('reply generation did not reach review')

    def test_review_flow_generates_then_sends_once_after_approval(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['draft']['text'], '好的，可以')
        self.assertEqual(self.action.send_calls, 0)

        self.service.decide_review(pending[0]['review_id'], 'approved')
        self.service.tick()
        self.service.tick()
        self.assertEqual(self.action.send_calls, 1)
        self.assertEqual(self.store.cursor_map()[TARGET], 7)
        self.assertEqual(
            self.store.list_sends()[0].state, 'completed',
        )
        cooldown = self.store.send_policy_status(
            TARGET,
            now=self.clock(),
            cooldown_seconds=15,
            daily_send_limit=300,
        )
        self.assertFalse(cooldown['allowed'])
        self.assertEqual(cooldown['reason'], 'target_cooldown')
        daily = self.store.send_policy_status(
            TARGET,
            now=self.clock() + 20,
            cooldown_seconds=15,
            daily_send_limit=1,
        )
        self.assertFalse(daily['allowed'])
        self.assertEqual(daily['reason'], 'daily_send_limit')

    def test_rejected_review_advances_cursor_without_send(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()
        self.service.decide_review(pending[0]['review_id'], 'rejected')
        self.service.tick()
        self.assertEqual(self.action.send_calls, 0)
        self.assertEqual(self.store.cursor_map()[TARGET], 7)

    def test_shadow_mode_generates_and_advances_without_review_or_send(self):
        self.service.close()
        self.service = ReplyService(
            self.vault,
            replace(self.config, mode='shadow'),
            action=self.action,
            generation=self.generation,
            store=self.store,
            now=self.clock,
        )
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        deadline = time.time() + 2
        while (
            self.store.cursor_map().get(TARGET) != 7
            and time.time() < deadline
        ):
            self.service.tick()
            time.sleep(0.01)

        self.assertEqual(self.store.cursor_map()[TARGET], 7)
        self.assertEqual(self.service.reviews(), [])
        self.assertEqual(self.action.send_calls, 0)
        self.assertEqual(
            self.service.activity(limit=1)[0]['event_type'],
            'shadow_draft',
        )

    def test_mode_change_requires_stopped_empty_runtime(self):
        with self.assertRaisesRegex(
            RuntimeError,
            'must be stopped before changing mode',
        ):
            self.service.set_mode('shadow')

        self.service.disarm()
        changed = self.service.set_mode('shadow')

        self.assertEqual(changed['config']['mode'], 'shadow')
        self.assertFalse(changed['armed'])
        self.assertEqual(
            ReplyServiceConfig.load(self.vault).mode,
            'shadow',
        )
        self.assertEqual(
            self.service.activity(limit=1)[0]['event_type'],
            'mode_changed',
        )

    def test_mode_change_refuses_to_orphan_pending_review(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        self.service.disarm()

        with self.assertRaisesRegex(
            RuntimeError,
            'queue must be resolved before changing mode',
        ):
            self.service.set_mode('shadow')

    def test_dispatched_restart_reconciles_and_never_resends(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()[0]
        self.service.decide_review(pending['review_id'], 'approved')
        draft = self.store.get_draft(pending['draft']['draft_id'])
        intent = self.service._intent(draft, pending['review_id'])
        operation, _ = self.store.prepare_send(intent, now=self.clock())
        self.store.mark_dispatched(
            operation.operation_id,
            external_ref='provider-fixture:send',
            now=self.clock(),
        )

        reopened = ReplyService(
            self.vault,
            replace(self.config, mode='live'),
            action=self.action,
            generation=self.generation,
            store=ReplyStore.for_vault(self.vault),
            now=self.clock,
        )
        try:
            reopened.tick()
            self.assertEqual(self.action.send_calls, 0)
            self.assertEqual(self.action.reconcile_calls, 1)
            self.assertEqual(
                reopened.store.get_send(operation.operation_id).state,
                'unknown',
            )
            reopened.tick()
            self.assertEqual(self.action.reconcile_calls, 1)
        finally:
            reopened.close()

    def test_explicit_retry_reopens_only_verified_pre_send_failure(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()[0]
        self.service.decide_review(pending['review_id'], 'approved')
        self.action.send_results.append({
            'state': 'failed',
            'stage': 'verify_draft_RuntimeError:copy_event_not_delivered',
        })
        self.service.tick()

        approved = self.service.reviews(state='approved')
        operation = self.store.list_sends()[0]
        self.assertEqual(self.action.send_calls, 1)
        self.assertEqual(operation.state, 'failed')
        self.assertEqual(self.store.cursor_map()[TARGET], 7)
        self.assertTrue(approved[0]['send']['retryable'])
        retried = self.service.retry_review(pending['review_id'])

        self.assertEqual(retried['send']['state'], 'prepared')
        self.assertEqual(retried['send']['retry_count'], 1)
        self.assertEqual(self.store.cursor_map()[TARGET], 7)
        tick_calls = len(self.action.calls)
        self.service.tick()
        self.assertEqual(
            self.action.calls[tick_calls:],
            ['events', 'send'],
        )
        self.assertEqual(self.action.send_calls, 2)
        self.assertEqual(
            self.store.get_send(operation.operation_id).state,
            'completed',
        )

    def test_partial_compose_failure_is_not_retryable(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()[0]
        self.service.decide_review(pending['review_id'], 'approved')
        self.action.send_results.append({
            'state': 'failed',
            'stage': 'compose_ready_RuntimeError:partial_input',
        })

        self.service.tick()

        approved = self.service.reviews(state='approved')
        self.assertEqual(self.action.send_calls, 1)
        self.assertFalse(approved[0]['send']['retryable'])
        with self.assertRaisesRegex(RuntimeError, 'not retryable'):
            self.service.retry_review(pending['review_id'])

    def test_retry_preflight_cannot_reopen_after_round_advances(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()[0]
        self.service.decide_review(pending['review_id'], 'approved')
        self.action.send_results.append({
            'state': 'failed',
            'stage': 'verify_draft_RuntimeError:copy_event_not_delivered',
        })
        self.service.tick()
        operation = self.store.list_sends()[0]
        self.action.preflight_hook = lambda: self.service.rounds.observe(
            self.service._event(event(8)),
            now=self.clock(),
        )

        with self.assertRaisesRegex(RuntimeError, 'eligible failed retry'):
            self.service.retry_review(pending['review_id'])

        self.assertEqual(
            self.store.get_send(operation.operation_id).state,
            'failed',
        )
        self.assertEqual(self.action.send_calls, 1)

    def test_round_advance_after_retry_cancels_before_dispatch(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()[0]
        self.service.decide_review(pending['review_id'], 'approved')
        self.action.send_results.append({
            'state': 'failed',
            'stage': 'verify_draft_RuntimeError:copy_event_not_delivered',
        })
        self.service.tick()
        operation = self.store.list_sends()[0]
        self.service.retry_review(pending['review_id'])

        self.service.rounds.observe(
            self.service._event(event(8)),
            now=self.clock(),
        )
        self.service.tick()

        self.assertEqual(self.action.send_calls, 1)
        self.assertEqual(
            self.store.get_send(operation.operation_id).state,
            'cancelled',
        )

    def test_unknown_send_is_never_exposed_as_retryable(self):
        self.action.events.append(event())
        self.service.tick()
        self.clock.value = 1_001.0
        self._finish_generation()
        pending = self.service.reviews()[0]
        self.service.decide_review(pending['review_id'], 'approved')
        draft = self.store.get_draft(pending['draft']['draft_id'])
        intent = self.service._intent(draft, pending['review_id'])
        operation, _ = self.store.prepare_send(intent, now=self.clock())
        self.store.mark_dispatched(
            operation.operation_id,
            external_ref='provider-fixture:send',
            now=self.clock(),
        )
        self.store.finish_send(
            operation.operation_id,
            state='unknown',
            stage='send_event_without_server_ack',
            now=self.clock(),
            error_code='provider_send_unknown',
        )

        approved = self.service.reviews(state='approved')
        self.assertFalse(approved[0]['send']['retryable'])
        with self.assertRaisesRegex(RuntimeError, 'not retryable'):
            self.service.retry_review(pending['review_id'])

    def test_config_round_trip_is_owner_only_and_redacted(self):
        self.config.save(self.vault)
        loaded = ReplyServiceConfig.load(self.vault)
        self.assertEqual(loaded, self.config)
        self.assertFalse(loaded.redacted()['secret_values_included'])
        self.assertNotIn(SOURCE_ACCOUNT, str(loaded.redacted()))
        path = ReplyServiceConfig.path_for_vault(self.vault)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
