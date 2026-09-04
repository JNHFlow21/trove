from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers import pending as pending_handlers
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.fixture_factory import FixtureData
from trove_core.wechat.indexer import index_fixture_data
from trove_core.wechat.models import Account, Conversation, Message


_SINCE = '2026-06-01T00:00:00Z'
_UNTIL = '2026-07-01T00:00:00Z'
_BASE = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def _message(
    conversation: Conversation,
    sender_id: str,
    sender_name: str,
    local_id: int,
    *,
    minutes: int = 0,
    direction: str = 'incoming',
    timestamp: datetime | None = None,
) -> Message:
    return Message(
        account_id=conversation.account_id,
        account_label='Work-WeChat',
        conversation_id=conversation.conversation_id,
        conversation_title=conversation.title,
        conversation_type=conversation.type,
        sender_id=sender_id,
        sender_name=sender_name,
        timestamp=timestamp or (_BASE + timedelta(minutes=minutes)),
        content=f'机密正文 {conversation.conversation_id}/{local_id}',
        shard_id='message_0',
        local_id=local_id,
        direction_hint=direction,
    )


def _fixture() -> FixtureData:
    accounts = [Account('acct-work', 'Work-WeChat', '工作微信')]
    alice = Conversation('conv-alice', 'acct-work', 'Alice', 'private', 2)
    bob = Conversation('conv-bob', 'acct-work', 'Bob', 'private', 2)
    carol = Conversation('conv-carol', 'acct-work', 'Carol', 'private', 2)
    dave = Conversation('conv-dave', 'acct-work', 'Dave', 'private', 2)
    eve = Conversation('conv-eve', 'acct-work', 'Eve', 'private', 2)
    fred = Conversation('conv-fred', 'acct-work', 'Fred', 'private', 2)
    team = Conversation('conv-team', 'acct-work', '产品群', 'group', 4)
    filehelper = Conversation('filehelper', 'acct-work', '文件传输助手', 'private', 1)
    official = Conversation('gh_fixture', 'acct-work', '示例服务号', 'private', 1)
    recent = Conversation('conv-recent', 'acct-work', 'Recent', 'private', 2)
    hourly = Conversation('conv-hourly', 'acct-work', 'Hourly', 'private', 2)
    daily = Conversation('conv-daily', 'acct-work', 'Daily', 'private', 2)
    weekly = Conversation('conv-weekly', 'acct-work', 'Weekly', 'private', 2)
    messages = [
        # conv-alice: incoming then a later outgoing reply — not pending.
        _message(alice, 'sender-alice', 'Alice', 1, minutes=1),
        _message(alice, 'me-work', '我', 2, minutes=2, direction='outgoing'),
        # conv-bob: two incoming, never replied — pending.
        _message(bob, 'sender-bob', 'Bob', 1, minutes=11),
        _message(bob, 'sender-bob', 'Bob', 2, minutes=12),
        # conv-carol: outgoing only — not pending.
        _message(carol, 'me-work', '我', 1, minutes=21, direction='outgoing'),
        # conv-dave: reply, then a newer incoming — pending again.
        _message(dave, 'sender-dave', 'Dave', 1, minutes=31),
        _message(dave, 'me-work', '我', 2, minutes=32, direction='outgoing'),
        _message(dave, 'sender-dave', 'Dave', 3, minutes=33),
        # conv-eve: last incoming before the window — not pending.
        _message(eve, 'sender-eve', 'Eve', 1, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        # conv-fred: an unknown-direction row after the incoming cannot prove
        # a reply — still pending.
        _message(fred, 'sender-fred', 'Fred', 1, minutes=41),
        _message(fred, 'me-work', '我', 2, minutes=42, direction='unknown'),
        # conv-team: group conversations are always excluded.
        _message(team, 'sender-gina', 'Gina', 1, minutes=51),
        # System id shapes are excluded even as private conversations.
        _message(filehelper, 'me-work', '我', 1, minutes=61),
        _message(official, 'gh_fixture', '示例服务号', 1, minutes=62),
        # Waiting-bucket conversations, all pending, keyed to _UNTIL.
        _message(recent, 'sender-recent', 'Recent', 1, timestamp=datetime(2026, 6, 30, 23, 30, tzinfo=timezone.utc)),
        _message(hourly, 'sender-hourly', 'Hourly', 1, timestamp=datetime(2026, 6, 30, 10, 0, tzinfo=timezone.utc)),
        _message(daily, 'sender-daily', 'Daily', 1, timestamp=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)),
        _message(weekly, 'sender-weekly', 'Weekly', 1, timestamp=datetime(2026, 6, 26, 12, 0, tzinfo=timezone.utc)),
    ]
    return FixtureData(
        accounts,
        [alice, bob, carol, dave, eve, fred, team, filehelper, official, recent, hourly, daily, weekly],
        messages,
    )


class _PendingVaultCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_data(self.vault, _fixture(), reset=True)
        self.config = VaultConfig.resolve(str(self.vault))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dispatcher(self):
        return build_default_dispatcher(self.vault)

    def _pending(self, payload, request_id='req-pending'):
        return self.dispatcher().dispatch(
            'trove.pending_replies', {'since': _SINCE, 'until': _UNTIL, **payload},
            request_id=request_id,
        )


class PendingRepliesTests(_PendingVaultCase):
    def test_lists_only_unanswered_private_conversations(self):
        response = self._pending({})
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['window'], {'since': _SINCE, 'until': _UNTIL})
        ids = {row['conversation_id'] for row in data['pending']}
        self.assertEqual(
            ids,
            {'conv-bob', 'conv-dave', 'conv-fred', 'conv-recent', 'conv-hourly', 'conv-daily', 'conv-weekly'},
        )
        self.assertEqual(data['matched_total'], 7)
        self.assertEqual(data['window_private_message_count'], 14)
        self.assertEqual(response['page'], {'has_more': False})
        self.assertEqual(response['coverage'], {'state': 'complete', 'returned': 7, 'remaining': 0})

    def test_order_is_most_recent_incoming_first_with_sender_and_counts(self):
        response = self._pending({})
        rows = response['data']['pending']
        self.assertEqual(
            [row['conversation_id'] for row in rows],
            ['conv-recent', 'conv-hourly', 'conv-daily', 'conv-weekly', 'conv-fred', 'conv-dave', 'conv-bob'],
        )
        bob = rows[-1]
        self.assertEqual(bob['citation'], 'trove://wechat/acct-work/conv-bob')
        self.assertEqual(bob['title'], 'Bob')
        self.assertEqual(bob['sender_id'], 'sender-bob')
        self.assertEqual(bob['sender_name'], 'Bob')
        self.assertEqual(bob['last_incoming'], '2026-06-10T12:12:00Z')
        self.assertEqual(bob['incoming_count'], 2)
        self.assertEqual(bob['outgoing_count'], 0)
        dave = rows[-2]
        self.assertEqual(dave['last_incoming'], '2026-06-10T12:33:00Z')
        self.assertEqual(dave['incoming_count'], 2)
        self.assertEqual(dave['outgoing_count'], 1)
        self.assertTrue(all(row['trust'] == 'untrusted_evidence' for row in rows))

    def test_waiting_buckets_track_window_end(self):
        response = self._pending({})
        buckets = {row['conversation_id']: row['waiting_bucket'] for row in response['data']['pending']}
        self.assertEqual(buckets['conv-recent'], 'lt_1h')
        self.assertEqual(buckets['conv-hourly'], '1h_1d')
        self.assertEqual(buckets['conv-daily'], '1d_3d')
        self.assertEqual(buckets['conv-weekly'], '3d_7d')
        self.assertEqual(buckets['conv-bob'], 'gt_7d')

    def test_limit_truncates_with_partial_coverage(self):
        response = self._pending({'limit': 2})
        self.assertTrue(response['ok'])
        self.assertEqual(
            [row['conversation_id'] for row in response['data']['pending']],
            ['conv-recent', 'conv-hourly'],
        )
        self.assertEqual(response['data']['matched_total'], 7)
        self.assertEqual(response['coverage'], {'state': 'partial', 'returned': 2, 'remaining': 5})

    def test_account_scope_narrows_results(self):
        response = self._pending({'account_id': 'acct-personal'})
        self.assertTrue(response['ok'])
        self.assertEqual(response['data']['pending'], [])
        self.assertEqual(response['data']['matched_total'], 0)
        self.assertEqual(response['data']['window_private_message_count'], 0)
        self.assertEqual(response['coverage']['state'], 'complete')

    def test_default_window_covers_last_seven_days(self):
        response = self.dispatcher().dispatch('trove.pending_replies', {}, request_id='req-default')
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        since = datetime.fromisoformat(data['window']['since'].replace('Z', '+00:00'))
        until = datetime.fromisoformat(data['window']['until'].replace('Z', '+00:00'))
        self.assertEqual(until - since, timedelta(days=7))
        self.assertEqual(data['pending'], [])

    def test_window_validation_fails_closed(self):
        dispatcher = self.dispatcher()
        inverted = dispatcher.dispatch(
            'trove.pending_replies', {'since': _UNTIL, 'until': _SINCE}, request_id='req-inverted',
        )
        self.assertEqual(inverted['error']['code'], 'invalid_request')
        oversized = dispatcher.dispatch(
            'trove.pending_replies',
            {'since': '2026-05-01T00:00:00Z', 'until': '2026-06-15T00:00:00Z'},
            request_id='req-oversized',
        )
        self.assertEqual(oversized['error']['code'], 'invalid_request')
        garbage = dispatcher.dispatch(
            'trove.pending_replies', {'since': '上周'}, request_id='req-garbage',
        )
        self.assertEqual(garbage['error']['code'], 'invalid_request')

    def test_epoch_bounds_are_accepted(self):
        since_epoch = str(int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()))
        until_epoch = str(int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()))
        response = self.dispatcher().dispatch(
            'trove.pending_replies', {'since': since_epoch, 'until': until_epoch},
            request_id='req-epoch',
        )
        self.assertTrue(response['ok'], response.get('error'))
        self.assertEqual(response['data']['window'], {'since': _SINCE, 'until': _UNTIL})
        self.assertEqual(response['data']['matched_total'], 7)

    def test_limit_bounds_fail_closed(self):
        dispatcher = self.dispatcher()
        for limit in (0, 51):
            response = dispatcher.dispatch(
                'trove.pending_replies', {'limit': limit}, request_id=f'req-limit-{limit}',
            )
            self.assertEqual(response['error']['code'], 'invalid_request')

    def test_results_are_metadata_only_and_never_carry_content(self):
        response = self._pending({'limit': 50})
        self.assertTrue(response['ok'])
        allowed = {
            'citation', 'account_id', 'conversation_id', 'title', 'member_count',
            'sender_id', 'sender_name', 'last_incoming', 'waiting_bucket',
            'incoming_count', 'outgoing_count', 'trust',
        }
        for row in response['data']['pending']:
            self.assertLessEqual(set(row), allowed)
        self.assertNotIn('机密正文', json.dumps(response, ensure_ascii=False))

    def test_pending_queries_use_bounded_indexes(self):
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    (
                        'SELECT account_id, conversation_id,'
                        " MAX(CASE WHEN direction='incoming' THEN timestamp END) AS last_incoming"
                        ' FROM messages INDEXED BY idx_messages_stats_time'
                        " WHERE timestamp>=? AND timestamp<? AND conversation_type='private'"
                        " AND conversation_id NOT IN ('filehelper')"
                        " AND conversation_id NOT LIKE 'gh\\_%' ESCAPE '\\'"
                        ' GROUP BY account_id, conversation_id'
                        ' HAVING last_incoming IS NOT NULL',
                        (_SINCE, _UNTIL),
                    ),
                    (
                        'SELECT COUNT(*) FROM messages INDEXED BY idx_messages_stats_time'
                        " WHERE timestamp>=? AND timestamp<? AND conversation_type='private'",
                        (_SINCE, _UNTIL),
                    ),
                    (
                        "SELECT sender_id, sender_name FROM messages"
                        " WHERE account_id=? AND conversation_id=? AND timestamp=? AND direction='incoming'"
                        ' ORDER BY shard_id DESC, local_id DESC LIMIT 1',
                        ('acct-work', 'conv-bob', '2026-06-10T12:12:00Z'),
                    ),
                ):
                    plans.extend(
                        str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params)
                    )
        finally:
            store.close()
        plan_text = ' | '.join(plans)
        self.assertNotIn('SCAN messages', plan_text)
        self.assertIn(
            'SEARCH messages USING COVERING INDEX idx_messages_stats_time (timestamp>? AND timestamp<?)',
            plan_text,
        )
        self.assertIn('idx_messages_context_window', plan_text)

    def test_missing_vault_returns_bounded_empty_result(self):
        missing = Path(self.temp.name) / 'missing-vault'
        config = VaultConfig.resolve(str(missing), env={})
        response = pending_handlers.pending_replies(config, {})
        self.assertTrue(response.ok)
        self.assertEqual(response.data['pending'], [])
        self.assertEqual(response.coverage['state'], 'complete')
        self.assertEqual(response.page, {'has_more': False})


if __name__ == '__main__':
    unittest.main()
