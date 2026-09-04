from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers import stats as stats_handlers
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
    account_label: str = 'Work-WeChat',
    direction: str = 'incoming',
    timestamp: datetime | None = None,
) -> Message:
    return Message(
        account_id=conversation.account_id,
        account_label=account_label,
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
    team = Conversation('conv-team', 'acct-work', '产品群', 'group', 4)
    messages = [
        # conv-alice: incoming 3 / outgoing 1 (plus one out-of-window row).
        *[_message(alice, 'sender-alice', 'Alice', index, minutes=index) for index in range(1, 4)],
        _message(alice, 'me-work', '我', 4, minutes=4, direction='outgoing'),
        _message(alice, 'sender-alice', 'Alice', 5, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        # conv-bob: incoming 2 / outgoing 2 / unknown 1.
        *[_message(bob, 'sender-bob', 'Bob', index, minutes=10 + index) for index in range(1, 3)],
        _message(bob, 'me-work', '我', 3, minutes=13, direction='outgoing'),
        _message(bob, 'me-work', '我', 4, minutes=14, direction='outgoing'),
        _message(bob, 'sender-bob', 'Bob', 5, minutes=15, direction='unknown'),
        # conv-team (group): sender-alice 5 / sender-carol 2 / me 1 / sender-dave 1 unknown.
        *[_message(team, 'sender-alice', 'Alice', index, minutes=20 + index) for index in range(1, 6)],
        *[_message(team, 'sender-carol', 'Carol', index, minutes=30 + index) for index in range(6, 8)],
        _message(team, 'me-work', '我', 8, minutes=38, direction='outgoing'),
        _message(team, 'sender-dave', 'Dave', 9, minutes=39, direction='unknown'),
    ]
    return FixtureData(accounts, [alice, bob, team], messages)


class _StatsVaultCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_data(self.vault, _fixture(), reset=True)
        self.config = VaultConfig.resolve(str(self.vault))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dispatcher(self):
        return build_default_dispatcher(self.vault)

    def _stats(self, payload, request_id='req-stats'):
        return self.dispatcher().dispatch(
            'trove.message_stats', {'since': _SINCE, 'until': _UNTIL, **payload},
            request_id=request_id,
        )


class MessageStatsTests(_StatsVaultCase):
    def test_by_conversation_ranks_and_splits_direction(self):
        response = self._stats({})
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['dimension'], 'by_conversation')
        self.assertEqual(data['window'], {'since': _SINCE, 'until': _UNTIL})
        self.assertEqual(data['window_message_count'], 18)
        self.assertEqual(data['matched_total'], 3)
        rows = data['rows']
        self.assertEqual(
            [row['conversation_id'] for row in rows],
            ['conv-team', 'conv-bob', 'conv-alice'],
        )
        team = rows[0]
        self.assertEqual(team['citation'], 'trove://wechat/acct-work/conv-team')
        self.assertEqual(team['title'], '产品群')
        self.assertEqual(team['conversation_type'], 'group')
        self.assertEqual(team['member_count'], 4)
        self.assertEqual(
            {key: team[key] for key in ('incoming', 'outgoing', 'unknown', 'total')},
            {'incoming': 7, 'outgoing': 1, 'unknown': 1, 'total': 9},
        )
        alice = rows[2]
        self.assertEqual(
            {key: alice[key] for key in ('incoming', 'outgoing', 'unknown', 'total')},
            {'incoming': 3, 'outgoing': 1, 'unknown': 0, 'total': 4},
        )
        self.assertTrue(all(row['trust'] == 'untrusted_evidence' for row in rows))
        self.assertEqual(response['page'], {'has_more': False})
        self.assertEqual(response['coverage'], {'state': 'complete', 'returned': 3, 'remaining': 0})

    def test_top_n_limit_truncates_with_partial_coverage(self):
        response = self._stats({'limit': 2})
        self.assertTrue(response['ok'])
        self.assertEqual([row['conversation_id'] for row in response['data']['rows']], ['conv-team', 'conv-bob'])
        self.assertEqual(response['data']['matched_total'], 3)
        self.assertEqual(response['coverage'], {'state': 'partial', 'returned': 2, 'remaining': 1})

    def test_by_sender_ranks_group_speakers_only(self):
        response = self._stats({'dimension': 'by_sender'})
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['window_message_count'], 9)
        self.assertEqual(data['matched_total'], 4)
        rows = data['rows']
        self.assertEqual(
            [row['sender_id'] for row in rows],
            ['sender-alice', 'sender-carol', 'me-work', 'sender-dave'],
        )
        alice = rows[0]
        self.assertEqual(alice['sender_name'], 'Alice')
        self.assertEqual(alice['conversation_count'], 1)
        self.assertEqual(
            {key: alice[key] for key in ('incoming', 'outgoing', 'unknown', 'total')},
            {'incoming': 5, 'outgoing': 0, 'unknown': 0, 'total': 5},
        )
        dave = rows[3]
        self.assertEqual(dave['unknown'], 1)
        self.assertEqual(dave['total'], 1)

    def test_by_sender_scoped_to_one_conversation(self):
        response = self._stats({'dimension': 'by_sender', 'conversation_id': 'conv-alice'})
        self.assertTrue(response['ok'], response.get('error'))
        rows = response['data']['rows']
        self.assertEqual([row['sender_id'] for row in rows], ['sender-alice', 'me-work'])
        self.assertEqual(rows[0]['total'], 3)
        self.assertEqual(rows[1]['outgoing'], 1)
        group = self._stats({'dimension': 'by_sender', 'conversation_id': 'conv-team'})
        self.assertEqual(group['data']['rows'][0]['sender_id'], 'sender-alice')
        self.assertEqual(group['data']['rows'][0]['total'], 5)

    def test_account_scope_narrows_aggregates(self):
        response = self._stats({'account_id': 'acct-personal'})
        self.assertTrue(response['ok'])
        self.assertEqual(response['data']['rows'], [])
        self.assertEqual(response['data']['window_message_count'], 0)
        self.assertEqual(response['coverage']['state'], 'complete')

    def test_default_window_covers_last_thirty_days(self):
        response = self.dispatcher().dispatch('trove.message_stats', {}, request_id='req-default')
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        since = datetime.fromisoformat(data['window']['since'].replace('Z', '+00:00'))
        until = datetime.fromisoformat(data['window']['until'].replace('Z', '+00:00'))
        self.assertEqual(until - since, timedelta(days=30))
        self.assertEqual(data['rows'], [])
        self.assertEqual(data['window_message_count'], 0)

    def test_window_validation_fails_closed(self):
        dispatcher = self.dispatcher()
        inverted = dispatcher.dispatch(
            'trove.message_stats', {'since': _UNTIL, 'until': _SINCE}, request_id='req-inverted',
        )
        self.assertEqual(inverted['error']['code'], 'invalid_request')
        oversized = dispatcher.dispatch(
            'trove.message_stats',
            {'since': '2025-01-01T00:00:00Z', 'until': '2026-06-01T00:00:00Z'},
            request_id='req-oversized',
        )
        self.assertEqual(oversized['error']['code'], 'invalid_request')
        garbage = dispatcher.dispatch(
            'trove.message_stats', {'since': '上个月'}, request_id='req-garbage',
        )
        self.assertEqual(garbage['error']['code'], 'invalid_request')

    def test_epoch_bounds_are_accepted(self):
        since_epoch = str(int(datetime(2026, 6, 1, tzinfo=timezone.utc).timestamp()))
        until_epoch = str(int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()))
        response = self.dispatcher().dispatch(
            'trove.message_stats', {'since': since_epoch, 'until': until_epoch},
            request_id='req-epoch',
        )
        self.assertTrue(response['ok'], response.get('error'))
        self.assertEqual(response['data']['window'], {'since': _SINCE, 'until': _UNTIL})
        self.assertEqual(response['data']['matched_total'], 3)

    def test_unknown_conversation_scope_is_typed_no_results(self):
        response = self._stats({'conversation_id': 'conv-missing'})
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'no_results')

    def test_invalid_dimension_and_limit_bounds_fail_closed(self):
        dispatcher = self.dispatcher()
        bad_dimension = dispatcher.dispatch(
            'trove.message_stats', {'dimension': 'by_content'}, request_id='req-dimension',
        )
        self.assertEqual(bad_dimension['error']['code'], 'invalid_request')
        for limit in (0, 51):
            response = dispatcher.dispatch(
                'trove.message_stats', {'limit': limit}, request_id=f'req-limit-{limit}',
            )
            self.assertEqual(response['error']['code'], 'invalid_request')

    def test_results_are_metadata_only_and_never_carry_content(self):
        response = self._stats({'limit': 50})
        self.assertTrue(response['ok'])
        allowed = {
            'citation', 'account_id', 'conversation_id', 'title', 'conversation_type',
            'member_count', 'sender_id', 'sender_name', 'conversation_count',
            'incoming', 'outgoing', 'unknown', 'total', 'trust',
        }
        for row in response['data']['rows']:
            self.assertLessEqual(set(row), allowed)
        encoded = json.dumps(response, ensure_ascii=False)
        self.assertNotIn('机密正文', encoded)

        senders = self._stats({'dimension': 'by_sender', 'limit': 50})
        for row in senders['data']['rows']:
            self.assertLessEqual(set(row), allowed)
        self.assertNotIn('机密正文', json.dumps(senders, ensure_ascii=False))

    def test_stats_queries_use_bounded_indexes(self):
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    (
                        "SELECT account_id, conversation_id, COUNT(*) AS total,"
                        " SUM(CASE WHEN direction='incoming' THEN 1 ELSE 0 END) AS incoming"
                        ' FROM messages INDEXED BY idx_messages_stats_time'
                        ' WHERE timestamp>=? AND timestamp<? GROUP BY account_id, conversation_id',
                        (_SINCE, _UNTIL),
                    ),
                    (
                        'SELECT account_id, sender_id, COUNT(*) AS total,'
                        ' COUNT(DISTINCT conversation_id) AS conversation_count'
                        ' FROM messages INDEXED BY idx_messages_stats_time'
                        " WHERE timestamp>=? AND timestamp<? AND conversation_type='group'"
                        ' GROUP BY account_id, sender_id',
                        (_SINCE, _UNTIL),
                    ),
                    (
                        'SELECT COUNT(*) FROM messages INDEXED BY idx_messages_stats_time'
                        ' WHERE timestamp>=? AND timestamp<? AND account_id=?',
                        (_SINCE, _UNTIL, 'acct-work'),
                    ),
                    (
                        'SELECT account_id, sender_id, COUNT(*) AS total FROM messages'
                        ' WHERE timestamp>=? AND timestamp<? AND conversation_id=?'
                        ' GROUP BY account_id, sender_id',
                        (_SINCE, _UNTIL, 'conv-team'),
                    ),
                    (
                        "SELECT sender_name FROM messages"
                        " WHERE account_id=? AND sender_id=? AND sender_name<>'' LIMIT 1",
                        ('acct-work', 'sender-alice'),
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
        self.assertIn(
            'SEARCH messages USING INDEX idx_messages_conversation_time (conversation_id=? AND timestamp>? AND timestamp<?)',
            plan_text,
        )
        self.assertIn('idx_messages_sender_time', plan_text)

    def test_missing_vault_returns_bounded_empty_result(self):
        missing = Path(self.temp.name) / 'missing-vault'
        config = VaultConfig.resolve(str(missing), env={})
        response = stats_handlers.message_stats(config, {})
        self.assertTrue(response.ok)
        self.assertEqual(response.data['rows'], [])
        self.assertEqual(response.coverage['state'], 'complete')
        self.assertEqual(response.page, {'has_more': False})


if __name__ == '__main__':
    unittest.main()
