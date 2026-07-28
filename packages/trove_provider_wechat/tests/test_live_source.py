from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from trove_provider_wechat.reply import LiveMessage, WeChatLiveConfig
from trove_provider_wechat.reply.live_source import WeChatLiveSource, WorkAccount


def message(
    target: str,
    position: int,
    *,
    outgoing: bool = False,
    created: int = 1_000,
) -> LiveMessage:
    target_ref = hashlib.sha256(target.encode()).hexdigest()
    return LiveMessage(
        target,
        target_ref,
        'message-fixture',
        position,
        f'server-{position}',
        1,
        created,
        outgoing,
        f'message-{position}',
    )


class FixtureLiveSource(WeChatLiveSource):
    def __init__(self):
        self.config = WeChatLiveConfig(
            account_id='account-fixture',
            account_id_sha256=hashlib.sha256(
                b'account-fixture',
            ).hexdigest(),
            enabled=True,
            send_shortcut='return',
        )
        self.rows = {}
        self.session_rows = ()
        self.names = {}

    def sessions(self):
        return self.session_rows

    def messages(
        self,
        target_id,
        *,
        after_source_position=None,
        through_source_position=None,
        limit=None,
    ):
        rows = list(self.rows.get(target_id, ()))
        if after_source_position is not None:
            rows = [
                row for row in rows
                if row.source_position > after_source_position
            ]
        if through_source_position is not None:
            rows = [
                row for row in rows
                if row.source_position <= through_source_position
            ]
        return tuple(rows[-limit:] if limit is not None else rows)

    def _display_name(self, target_id):
        return self.names.get(target_id, '')


class LiveSourceTests(unittest.TestCase):
    def test_query_snapshot_is_removed_after_use(self):
        class RecordingSQLCipher:
            snapshot = None

            def query(self, database, key_hex, sql):
                self.snapshot = Path(database)
                self.assertions = (
                    self.snapshot.is_file(),
                    key_hex == 'b' * 64,
                    sql == 'SELECT 1 AS n',
                )
                return [{'n': 1}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'message_0.db'
            source.write_bytes(b'0123456789abcdefpayload')
            account_id = 'account-fixture'
            config = WeChatLiveConfig(
                account_id=account_id,
                account_id_sha256=hashlib.sha256(
                    account_id.encode('utf-8'),
                ).hexdigest(),
            )
            sqlcipher = RecordingSQLCipher()
            account = WorkAccount(
                account_id, root, source, source, (source,),
            )
            live = WeChatLiveSource(
                config,
                {'30313233343536373839616263646566': {'dk': 'b' * 64}},
                runtime_root=root / 'runtime',
                sqlcipher=sqlcipher,
                account=account,
            )

            self.assertEqual(
                live._query(source, 'SELECT 1 AS n'),
                [{'n': 1}],
            )
            self.assertEqual(sqlcipher.assertions, (True, True, True))
            self.assertIsNotNone(sqlcipher.snapshot)
            self.assertFalse(sqlcipher.snapshot.exists())
            self.assertEqual(list(live.snapshot_root.iterdir()), [])

    def test_missing_historical_conversation_does_not_fail_the_poll(self):
        class MissingConversationSource(FixtureLiveSource):
            def messages(self, target_id, **kwargs):
                if target_id == 'missing':
                    raise RuntimeError('conversation_table_missing')
                return super().messages(target_id, **kwargs)

        source = MissingConversationSource()
        missing_ref = hashlib.sha256(b'missing').hexdigest()
        present_ref = hashlib.sha256(b'present').hexdigest()
        source.session_rows = (
            ('missing', missing_ref, 3),
            ('present', present_ref, 1),
        )
        source.rows['present'] = (message('present', 1),)
        source.names['present'] = 'display'

        result = source.events(
            {missing_ref: 0, present_ref: 0},
            observed_at=1_010.0,
        )

        self.assertEqual(len(result['events']), 1)
        self.assertEqual(
            result['acknowledgements'][0]['reason'],
            'conversation_unavailable',
        )

    def test_event_uses_canonical_vault_account_but_source_bound_conversation(self):
        source = FixtureLiveSource()
        source_account = 'private-source-account'
        namespace = 'com.tencent.xinWeChat2__private-source-account'
        canonical = (
            'acct-'
            + hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:12]
        )
        source.config = WeChatLiveConfig(
            account_id=canonical,
            source_account_id=source_account,
            conversation_namespace=namespace,
            account_id_sha256=hashlib.sha256(
                source_account.encode('utf-8'),
            ).hexdigest(),
            enabled=True,
            send_shortcut='return',
        )
        target_ref = hashlib.sha256(b'a').hexdigest()
        source.session_rows = (('a', target_ref, 1),)
        source.rows['a'] = (message('a', 1),)
        source.names['a'] = 'display'

        event = source.events(
            {target_ref: 0}, observed_at=1_010.0,
        )['events'][0]

        expected_conversation = (
            'conv-'
            + hashlib.sha256(
                f'{namespace}:a'.encode('utf-8'),
            ).hexdigest()[:12]
        )
        self.assertEqual(event['account_id'], canonical)
        self.assertEqual(event['conversation_id'], expected_conversation)
        self.assertNotEqual(event['conversation_id'], source_account)

    def test_inbound_delta_becomes_one_untrusted_event(self):
        source = FixtureLiveSource()
        target_ref = hashlib.sha256(b'a').hexdigest()
        source.session_rows = (('a', target_ref, 7),)
        source.rows['a'] = tuple(message('a', position) for position in range(1, 8))
        source.names['a'] = 'display'
        result = source.events({target_ref: 5}, observed_at=1_010.0)
        self.assertEqual(len(result['events']), 1)
        self.assertTrue(result['events'][0]['conversation_id'].startswith('conv-'))
        self.assertEqual(
            [item['source_position'] for item in result['events'][0]['messages']],
            [6, 7],
        )
        self.assertTrue(all(
            item['trust'] == 'untrusted_evidence'
            for item in result['events'][0]['messages']
        ))

    def test_outgoing_advance_is_acknowledged_not_replied_to(self):
        source = FixtureLiveSource()
        target_ref = hashlib.sha256(b'a').hexdigest()
        source.session_rows = (('a', target_ref, 7),)
        source.rows['a'] = (
            message('a', 6),
            message('a', 7, outgoing=True),
        )
        source.names['a'] = 'display'
        result = source.events({target_ref: 5}, observed_at=1_010.0)
        self.assertEqual(result['events'], [])
        self.assertEqual(result['acknowledgements'][0]['source_position'], 7)

    def test_unknown_stale_session_seeds_but_recent_message_is_observed(self):
        source = FixtureLiveSource()
        old_ref = hashlib.sha256(b'old').hexdigest()
        new_ref = hashlib.sha256(b'new').hexdigest()
        source.session_rows = (
            ('old', old_ref, 30),
            ('new', new_ref, 1),
        )
        source.rows['old'] = (message('old', 30, created=100),)
        source.rows['new'] = (message('new', 1, created=1_008),)
        source.names = {'old': 'old', 'new': 'new'}
        result = source.events({}, observed_at=1_010.0)
        self.assertEqual(
            [item['target_ref'] for item in result['events']], [new_ref],
        )
        self.assertEqual(
            result['acknowledgements'][0]['target_ref'], old_ref,
        )


if __name__ == '__main__':
    unittest.main()
