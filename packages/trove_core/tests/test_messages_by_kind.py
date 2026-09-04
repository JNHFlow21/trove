from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers import message_kinds as kind_handlers
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.fixture_factory import FixtureData
from trove_core.wechat.indexer import index_fixture_data
from trove_core.wechat.models import Account, Conversation, Message


_BASE = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
_SINCE = '2026-08-01T00:00:00Z'
_UNTIL = '2026-09-01T00:00:00Z'


def _payload(
    *,
    appmsg_type: int,
    normalized_type: str,
    fields: dict | None = None,
    display_text: str = '[appmsg/unsupported]',
    parse_status: str = 'parsed',
    unsupported_reason: str | None = None,
) -> dict:
    return {
        'source_hash': 'ab' * 32,
        'appmsg_type': appmsg_type,
        'normalized_type': normalized_type,
        'parse_status': parse_status,
        'fields': fields or {},
        'display_text': display_text,
        'unsupported_reason': unsupported_reason,
        'parser_version': 'appmsg-v2',
    }


def _message(
    conversation: Conversation,
    sender_id: str,
    sender_name: str,
    local_id: int,
    *,
    minutes: int = 0,
    direction: str = 'incoming',
    content_kind: str = 'text',
    content: str | None = None,
    payload: dict | None = None,
) -> Message:
    return Message(
        account_id=conversation.account_id,
        account_label='Work-WeChat',
        conversation_id=conversation.conversation_id,
        conversation_title=conversation.title,
        conversation_type=conversation.type,
        sender_id=sender_id,
        sender_name=sender_name,
        timestamp=_BASE + timedelta(minutes=minutes),
        content=content if content is not None else f'机密正文 {conversation.conversation_id}/{local_id}',
        shard_id='message_0',
        local_id=local_id,
        direction_hint=direction,
        content_kind=content_kind,
        normalized_payload=payload,
    )


def _fixture() -> FixtureData:
    accounts = [Account('acct-work', 'Work-WeChat', '工作微信')]
    alice = Conversation('conv-alice', 'acct-work', 'Alice', 'private', 2)
    team = Conversation('conv-team', 'acct-work', '产品群', 'group', 4)
    link_fields = {
        'title': '示例链接标题',
        'link_identity': {'scheme': 'https', 'host': 'example.com', 'path_hash': 'cd' * 32},
    }
    messages = [
        # Links: two outgoing then one incoming, spaced in time.
        _message(alice, 'me-work', '我', 1, minutes=10, direction='outgoing', content_kind='appmsg',
                 content='[appmsg/link] 示例链接标题',
                 payload=_payload(appmsg_type=5, normalized_type='link', fields=link_fields,
                                  display_text='[appmsg/link] 示例链接标题')),
        _message(alice, 'me-work', '我', 2, minutes=20, direction='outgoing', content_kind='appmsg',
                 content='[appmsg/link] 第二条链接',
                 payload=_payload(appmsg_type=5, normalized_type='link',
                                  fields={'title': '第二条链接'}, display_text='[appmsg/link] 第二条链接')),
        _message(alice, 'sender-alice', 'Alice', 3, minutes=30, content_kind='appmsg',
                 content='[appmsg/link] 对方发的链接',
                 payload=_payload(appmsg_type=5, normalized_type='link',
                                  fields={'title': '对方发的链接'}, display_text='[appmsg/link] 对方发的链接')),
        # Transfers: one outgoing, one incoming.
        _message(alice, 'me-work', '我', 4, minutes=40, direction='outgoing', content_kind='appmsg',
                 content='[appmsg/transfer_notice] 转账',
                 payload=_payload(appmsg_type=2000, normalized_type='transfer_notice',
                                  fields={'title': '转账给 Alice'}, display_text='[appmsg/transfer_notice] 转账给 Alice')),
        _message(alice, 'sender-alice', 'Alice', 5, minutes=50, content_kind='appmsg',
                 content='[appmsg/transfer_notice] 转账',
                 payload=_payload(appmsg_type=2000, normalized_type='transfer_notice',
                                  fields={'title': '来自 Alice 的转账'},
                                  display_text='[appmsg/transfer_notice] 来自 Alice 的转账')),
        # One red packet and one contact card: parser stores both as unsupported.
        _message(alice, 'sender-alice', 'Alice', 6, minutes=60, content_kind='appmsg',
                 content='[appmsg/unsupported] 红包',
                 payload=_payload(appmsg_type=2001, normalized_type='unsupported',
                                  fields={'title': '恭喜发财'}, display_text='[appmsg/unsupported] 恭喜发财',
                                  parse_status='unsupported', unsupported_reason='unsupported_appmsg_type')),
        _message(alice, 'sender-alice', 'Alice', 7, minutes=70, content_kind='appmsg',
                 content='[appmsg/unsupported] 名片',
                 payload=_payload(appmsg_type=42, normalized_type='unsupported',
                                  fields={'title': '名片'}, display_text='[appmsg/unsupported] 名片',
                                  parse_status='unsupported', unsupported_reason='unsupported_appmsg_type')),
        # One mini program and one file.
        _message(alice, 'me-work', '我', 8, minutes=80, direction='outgoing', content_kind='appmsg',
                 content='[appmsg/mini_program] 示例小程序',
                 payload=_payload(appmsg_type=33, normalized_type='mini_program',
                                  fields={'title': '示例小程序', 'mini_program_app_id': 'wxfixtureappid'},
                                  display_text='[appmsg/mini_program] 示例小程序')),
        _message(alice, 'sender-alice', 'Alice', 9, minutes=90, content_kind='appmsg',
                 content='[appmsg/file] 需求文档.pdf',
                 payload=_payload(appmsg_type=6, normalized_type='file',
                                  fields={'file_name': '需求文档.pdf', 'file_extension': 'pdf', 'file_size': 2048},
                                  display_text='[appmsg/file] 需求文档.pdf pdf 2048B')),
        # Coarse kinds.
        _message(alice, 'sender-alice', 'Alice', 10, minutes=100, content_kind='image', content='[图片二进制]'),
        _message(team, 'sender-gina', 'Gina', 11, minutes=110, content_kind='image', content='[群图二进制]'),
        _message(alice, 'sender-alice', 'Alice', 12, minutes=120, content_kind='voice', content='[语音二进制]'),
        _message(alice, 'me-work', '我', 13, minutes=130, direction='outgoing', content_kind='video', content='[视频二进制]'),
        _message(alice, 'sender-alice', 'Alice', 14, minutes=140, content_kind='sticker', content='[表情二进制]'),
        # Text rows, one intentionally long for excerpt truncation.
        _message(alice, 'sender-alice', 'Alice', 15, minutes=150),
        _message(alice, 'sender-alice', 'Alice', 16, minutes=160, content='长文本 ' + '密' * 300),
        _message(team, 'sender-gina', 'Gina', 17, minutes=170),
    ]
    return FixtureData(accounts, [alice, team], messages)


class _KindVaultCase(unittest.TestCase):
    def setUp(self) -> None:
        kind_handlers._reset_cursor_store_for_tests()
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_data(self.vault, _fixture(), reset=True)
        self.config = VaultConfig.resolve(str(self.vault))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dispatcher(self):
        return build_default_dispatcher(self.vault)

    def _by_kind(self, kind, payload=None, request_id='req-kind'):
        return self.dispatcher().dispatch(
            'trove.messages_by_kind', {'kind': kind, 'since': _SINCE, 'until': _UNTIL, **(payload or {})},
            request_id=request_id,
        )


class MessagesByKindTests(_KindVaultCase):
    def test_link_listing_is_newest_first_with_payload_metadata(self):
        response = self._by_kind('link')
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['kind'], 'link')
        self.assertEqual(data['matched_total'], 3)
        rows = data['messages']
        self.assertEqual([row['sender_name'] for row in rows], ['Alice', '我', '我'])
        first = rows[0]
        self.assertEqual(first['citation'], 'trove://wechat/acct-work/conv-alice/message_0/3')
        self.assertEqual(first['conversation_title'], 'Alice')
        self.assertEqual(first['direction'], 'incoming')
        self.assertEqual(first['kind'], 'link')
        self.assertEqual(first['content_kind'], 'appmsg')
        self.assertEqual(first['normalized_type'], 'link')
        self.assertEqual(first['appmsg_type'], 5)
        self.assertEqual(first['summary'], '[appmsg/link] 对方发的链接')
        self.assertFalse(first['summary_truncated'])
        self.assertEqual(rows[2]['metadata']['link_identity']['host'], 'example.com')
        self.assertEqual(rows[2]['metadata']['title'], '示例链接标题')
        self.assertTrue(all(row['trust'] == 'untrusted_evidence' for row in rows))
        self.assertEqual(response['page'], {'has_more': False})
        self.assertEqual(response['coverage'], {'state': 'complete', 'returned': 3, 'remaining': 0})

    def test_direction_filter_selects_outgoing_only(self):
        response = self._by_kind('link', {'direction': 'outgoing'})
        self.assertTrue(response['ok'])
        rows = response['data']['messages']
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['direction'] == 'outgoing' for row in rows))
        self.assertEqual(response['data']['matched_total'], 2)

    def test_transfer_redpacket_miniapp_file_and_contact_card_kinds(self):
        transfer = self._by_kind('transfer')
        self.assertEqual(transfer['data']['matched_total'], 2)
        self.assertTrue(all(row['normalized_type'] == 'transfer_notice' for row in transfer['data']['messages']))

        redpacket = self._by_kind('redpacket')
        self.assertEqual(redpacket['data']['matched_total'], 1)
        packet = redpacket['data']['messages'][0]
        self.assertEqual(packet['appmsg_type'], 2001)
        self.assertEqual(packet['normalized_type'], 'unsupported')
        self.assertEqual(packet['summary'], '[appmsg/unsupported] 恭喜发财')

        contact = self._by_kind('contact_card')
        self.assertEqual(contact['data']['matched_total'], 1)
        self.assertEqual(contact['data']['messages'][0]['appmsg_type'], 42)

        miniapp = self._by_kind('miniapp')
        self.assertEqual(miniapp['data']['matched_total'], 1)
        self.assertEqual(miniapp['data']['messages'][0]['metadata']['mini_program_app_id'], 'wxfixtureappid')

        files = self._by_kind('file')
        self.assertEqual(files['data']['matched_total'], 1)
        self.assertEqual(files['data']['messages'][0]['metadata']['file_name'], '需求文档.pdf')

    def test_coarse_kinds_read_from_content_kind(self):
        images = self._by_kind('image')
        self.assertEqual(images['data']['matched_total'], 2)
        self.assertEqual(
            {row['conversation_id'] for row in images['data']['messages']},
            {'conv-alice', 'conv-team'},
        )
        for row in images['data']['messages']:
            self.assertEqual(row['summary'], '[image]')
            self.assertEqual(row['metadata'], {})
            self.assertNotIn('normalized_type', row)
        self.assertEqual(self._by_kind('voice')['data']['matched_total'], 1)
        self.assertEqual(self._by_kind('video')['data']['matched_total'], 1)
        self.assertEqual(self._by_kind('sticker')['data']['matched_total'], 1)

    def test_text_kind_returns_bounded_excerpts(self):
        response = self._by_kind('text')
        self.assertEqual(response['data']['matched_total'], 3)
        rows = {row['citation']: row for row in response['data']['messages']}
        long_row = rows['trove://wechat/acct-work/conv-alice/message_0/16']
        self.assertTrue(long_row['summary_truncated'])
        self.assertEqual(len(long_row['summary']), 140)
        short_row = rows['trove://wechat/acct-work/conv-alice/message_0/15']
        self.assertFalse(short_row['summary_truncated'])
        self.assertEqual(short_row['summary'], '机密正文 conv-alice/15')

    def test_conversation_scope_is_exact_and_typed(self):
        scoped = self._by_kind('text', {'conversation_id': 'conv-team'})
        self.assertTrue(scoped['ok'])
        self.assertEqual(scoped['data']['matched_total'], 1)
        self.assertEqual(scoped['data']['scope']['account_id'], 'acct-work')
        missing = self._by_kind('text', {'conversation_id': 'conv-missing'})
        self.assertFalse(missing['ok'])
        self.assertEqual(missing['error']['code'], 'no_results')

    def test_cursor_paginates_without_duplicates(self):
        first = self._by_kind('link', {'limit': 2}, request_id='req-page-1')
        self.assertTrue(first['ok'])
        self.assertTrue(first['page']['has_more'])
        cursor = first['page']['next_cursor']
        self.assertEqual(first['coverage'], {'state': 'partial', 'returned': 2, 'remaining': 1})
        second = self._by_kind('link', {'limit': 2, 'cursor': cursor}, request_id='req-page-2')
        self.assertTrue(second['ok'])
        self.assertEqual(second['page'], {'has_more': False})
        self.assertEqual(second['coverage'], {'state': 'complete', 'returned': 1, 'remaining': 0})
        first_ids = {row['citation'] for row in first['data']['messages']}
        second_ids = {row['citation'] for row in second['data']['messages']}
        self.assertFalse(first_ids & second_ids)
        self.assertEqual(len(first_ids | second_ids), 3)

    def test_cursor_is_bound_to_filters(self):
        first = self._by_kind('link', {'limit': 2}, request_id='req-bind-1')
        cursor = first['page']['next_cursor']
        rebound = self._by_kind('file', {'limit': 2, 'cursor': cursor}, request_id='req-bind-2')
        self.assertFalse(rebound['ok'])
        self.assertEqual(rebound['error']['code'], 'cursor_mismatch')

    def test_time_window_narrows_listing(self):
        windowed = self._by_kind('link', {'since': '2026-08-20T12:25:00Z'})
        self.assertEqual(windowed['data']['matched_total'], 1)
        self.assertEqual(windowed['data']['messages'][0]['summary'], '[appmsg/link] 对方发的链接')
        epoch = self.dispatcher().dispatch(
            'trove.messages_by_kind',
            {
                'kind': 'link',
                'since': str(int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp())),
                'until': str(int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())),
            },
            request_id='req-epoch',
        )
        self.assertEqual(epoch['data']['matched_total'], 3)

    def test_invalid_inputs_fail_closed(self):
        dispatcher = self.dispatcher()
        bad_kind = dispatcher.dispatch(
            'trove.messages_by_kind', {'kind': 'appmsg'}, request_id='req-bad-kind',
        )
        self.assertEqual(bad_kind['error']['code'], 'invalid_request')
        bad_direction = dispatcher.dispatch(
            'trove.messages_by_kind', {'kind': 'link', 'direction': 'sideways'}, request_id='req-bad-direction',
        )
        self.assertEqual(bad_direction['error']['code'], 'invalid_request')
        inverted = dispatcher.dispatch(
            'trove.messages_by_kind',
            {'kind': 'link', 'since': _UNTIL, 'until': _SINCE},
            request_id='req-inverted',
        )
        self.assertEqual(inverted['error']['code'], 'invalid_request')
        for limit in (0, 51):
            response = dispatcher.dispatch(
                'trove.messages_by_kind', {'kind': 'link', 'limit': limit}, request_id=f'req-limit-{limit}',
            )
            self.assertEqual(response['error']['code'], 'invalid_request')

    def test_rows_are_bounded_and_never_carry_raw_content(self):
        allowed = {
            'citation', 'account_id', 'conversation_id', 'conversation_title', 'conversation_type',
            'sender_id', 'sender_name', 'timestamp', 'direction', 'kind', 'content_kind',
            'summary', 'summary_truncated', 'metadata', 'normalized_type', 'appmsg_type', 'trust',
        }
        for kind in ('link', 'redpacket', 'image', 'text'):
            response = self._by_kind(kind, {'limit': 50})
            for row in response['data']['messages']:
                self.assertLessEqual(set(row), allowed)
                self.assertNotIn('content', row)
        encoded = json.dumps(self._by_kind('text', {'limit': 50}), ensure_ascii=False)
        # The 300-char fixture text is only ever exposed as a 140-char excerpt.
        self.assertNotIn('密' * 141, encoded)
        self.assertNotIn('[图片二进制]', encoded)

    def test_kind_queries_use_bounded_indexes(self):
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    # Unscoped coarse kind: kind-leading covering index.
                    (
                        'SELECT citation, timestamp FROM messages INDEXED BY idx_messages_kind_time'
                        " WHERE content_kind=? AND timestamp>=? AND timestamp<?"
                        ' ORDER BY timestamp DESC, citation DESC LIMIT ?',
                        ('image', _SINCE, _UNTIL, 21),
                    ),
                    # Conversation-scoped coarse kind: conversation-time index.
                    (
                        'SELECT citation, timestamp FROM messages INDEXED BY idx_messages_conversation_time'
                        " WHERE content_kind=? AND account_id=? AND conversation_id=? AND timestamp>=? AND timestamp<?"
                        ' ORDER BY timestamp DESC, citation DESC LIMIT ?',
                        ('text', 'acct-work', 'conv-alice', _SINCE, _UNTIL, 21),
                    ),
                    # Parser-named payload subtype.
                    (
                        'SELECT m.citation, m.timestamp, p.display_text'
                        ' FROM message_payloads p INDEXED BY idx_message_payloads_type_citation'
                        ' CROSS JOIN messages m INDEXED BY idx_messages_citation ON m.citation = p.citation'
                        " WHERE p.normalized_type IN (?) AND m.timestamp>=? AND m.timestamp<?"
                        ' ORDER BY m.timestamp DESC, m.citation DESC LIMIT ?',
                        ('link', _SINCE, _UNTIL, 21),
                    ),
                    # Raw appmsg type subtype (red packet).
                    (
                        'SELECT m.citation, m.timestamp, p.display_text'
                        ' FROM message_payloads p INDEXED BY idx_message_payloads_appmsg_citation'
                        ' CROSS JOIN messages m INDEXED BY idx_messages_citation ON m.citation = p.citation'
                        ' WHERE p.appmsg_type IN (?)'
                        ' ORDER BY m.timestamp DESC, m.citation DESC LIMIT ?',
                        (2001, 21),
                    ),
                ):
                    plans.extend(
                        str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params)
                    )
        finally:
            store.close()
        plan_text = ' | '.join(plans)
        self.assertNotIn('SCAN messages', plan_text)
        self.assertNotIn('SCAN message_payloads', plan_text)
        self.assertIn(
            'SEARCH messages USING COVERING INDEX idx_messages_kind_time (content_kind=? AND timestamp>? AND timestamp<?)',
            plan_text,
        )
        self.assertIn('idx_messages_conversation_time', plan_text)
        self.assertIn(
            'SEARCH p USING INDEX idx_message_payloads_type_citation (normalized_type=?)',
            plan_text,
        )
        self.assertIn(
            'SEARCH p USING INDEX idx_message_payloads_appmsg_citation (appmsg_type=?)',
            plan_text,
        )
        self.assertIn('SEARCH m USING INDEX idx_messages_citation (citation=?)', plan_text)

    def test_missing_vault_returns_bounded_empty_result(self):
        missing = Path(self.temp.name) / 'missing-vault'
        config = VaultConfig.resolve(str(missing), env={})
        response = kind_handlers.messages_by_kind(config, {'kind': 'link'})
        self.assertTrue(response.ok)
        self.assertEqual(response.data['messages'], [])
        self.assertEqual(response.coverage['state'], 'complete')
        self.assertEqual(response.page, {'has_more': False})


if __name__ == '__main__':
    unittest.main()
