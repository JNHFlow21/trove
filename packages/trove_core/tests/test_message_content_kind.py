from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.search.context import ContextService
from trove_core.store.repositories import WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.content_kind import classify_content_kind, display_content_for_kind
from trove_core.wechat.models import Account, Conversation, Message


class MessageContentKindTests(unittest.TestCase):
    def test_backfill_classifies_appmsg_and_context_renders_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            raw_appmsg = '<msg><appmsg><title>不应直接展示的XML</title></appmsg></msg>'
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', '客户', 'private')],
                [Message('acct-a', 'A', 'conv-a', '客户', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), raw_appmsg, 's', 1)],
            )

            report = store.backfill_message_content_kind()

            self.assertEqual(report['kind_counts']['appmsg'], 1)
            with store.connect() as conn:
                row = conn.execute('SELECT content, content_kind FROM messages').fetchone()
            self.assertEqual(row['content_kind'], 'appmsg')
            self.assertIn('不应直接展示的XML', row['content'])
            self.assertNotIn('<appmsg', row['content'])
            self.assertTrue(store.exact_search('不应直接展示'))
            context = ContextService(store).fetch('trove://wechat/acct-a/conv-a/s/1')
            self.assertIn('不应直接展示的XML', context['messages'][0]['content'])

            second = store.backfill_message_content_kind()
            self.assertEqual(second['updated'], 0)
            with store.connect() as conn:
                row = conn.execute('SELECT content, content_kind FROM messages').fetchone()
            self.assertEqual(row['content_kind'], 'appmsg')
            self.assertIn('不应直接展示的XML', row['content'])

    def test_classifies_media_and_unknown_binary_placeholders(self):
        self.assertEqual(classify_content_kind('binary-looking payload', local_type=3), 'image')
        self.assertEqual(classify_content_kind('', local_type=34), 'voice')
        self.assertEqual(classify_content_kind('<msg>looks app</msg>', local_type=43), 'video')
        self.assertEqual(classify_content_kind('anything', local_type=244813135921), 'appmsg')
        self.assertEqual(classify_content_kind('<voicemsg length="1"/>'), 'voice')
        self.assertEqual(classify_content_kind('<msg><img /></msg>'), 'image')
        self.assertEqual(classify_content_kind('<videomsg length="1"/>'), 'video')
        self.assertEqual(classify_content_kind('<emoji md5="x"/>'), 'sticker')
        self.assertEqual(classify_content_kind('<quotemsg>hi</quotemsg>'), 'quote')
        self.assertEqual(display_content_for_kind('raw', 'quote'), '[引用消息]')
        self.assertEqual(display_content_for_kind('raw', 'video'), '[video]')

    def test_backfill_refreshes_chunks_vectors_and_dirty_without_raw_payload(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            raw = '<msg><appmsg><type>5</type><title>安全卡片</title><url>https://example.com/item?token=rawuniquealpha</url></appmsg></msg>'
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', '客户', 'private')],
                [Message('acct-a', 'A', 'conv-a', '客户', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), raw, 's', 1)],
            )
            store.rebuild_evidence_chunks()
            citation = 'trove://wechat/acct-a/conv-a/s/1'
            with store.connect() as conn:
                old_chunk = conn.execute('SELECT chunk_citation FROM evidence_chunks WHERE parent_citation=?', (citation,)).fetchone()['chunk_citation']
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS vector_entries (
                        citation TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        vector_json TEXT NOT NULL,
                        content_hash TEXT
                    )"""
                )
                conn.execute(
                    'INSERT INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)',
                    (old_chunk, 'test', 1, '[0.0]', 'old-hash'),
                )
                conn.commit()
            self.assertTrue(store.chunk_search('rawuniquealpha', {'source_type': 'message'}, limit=5))

            report = store.backfill_message_content_kind()

            self.assertEqual(report['updated'], 1)
            self.assertEqual(report['dirty_recorded'], 1)
            self.assertFalse(store.exact_search('rawuniquealpha', limit=5))
            self.assertFalse(store.fts_search_filtered('rawuniquealpha', filters={'source_type': 'message'}, limit=5))
            self.assertFalse(store.chunk_search('rawuniquealpha', {'source_type': 'message'}, limit=5))
            docs = list(store.iter_vector_documents(citations=[citation]))
            self.assertEqual(len(docs), 1)
            self.assertNotIn('rawuniquealpha', docs[0]['vector_text'])
            self.assertIn('[appmsg/link]', docs[0]['vector_text'])
            self.assertIn('安全卡片', docs[0]['vector_text'])
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM vector_entries WHERE citation=?', (old_chunk,)).fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM sync_dirty_citations WHERE citation=?', (citation,)).fetchone()[0], 1)

    def test_backfill_repairs_stale_chunks_when_message_row_is_already_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            citation = 'trove://wechat/acct-a/conv-a/s/1'
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', '客户', 'private')],
                [Message('acct-a', 'A', 'conv-a', '客户', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[unknown_binary]', 's', 1, content_kind='unknown_binary')],
            )
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f'{citation}#chunk-0', f'{citation}#chunk-0', citation, 'acct-a', 'A', 'message', 'conv-a', '客户', '客户', '2026-01-01T00:00:00Z', 'stalerawbeta', 0, '{}', 'active', '2026-01-01T00:00:00Z'),
                )
                conn.commit()

            report = store.backfill_message_content_kind()

            self.assertEqual(report['updated'], 0)
            self.assertEqual(report['stale_chunk_parents'], 1)
            self.assertFalse(store.chunk_search('stalerawbeta', {'source_type': 'message'}, limit=5))
            self.assertTrue(store.chunk_search('[unknown_binary]', {'source_type': 'message'}, limit=5))


if __name__ == '__main__':
    unittest.main()
