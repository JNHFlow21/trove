from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.store.sqlite_store import SQLiteStore, vector_document_text
from trove_core.vector.text_versions import vector_document_text_v4
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore
from trove_core.wechat.indexer import index_fixture_vault


class VectorContextualTextTests(unittest.TestCase):
    def test_iter_vector_documents_contains_contextual_text(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            row = next(store.iter_vector_documents(batch_size=2))
            self.assertIn('vector_text', row.keys())
            self.assertIn('来源类型:', row['vector_text'])
            self.assertIn('检索对象: 微信聊天证据', row['vector_text'])
            self.assertIn('证据正文:', row['vector_text'])
            self.assertIn(row['content'], row['vector_text'])

    def test_sqlite_vector_index_uses_context_contract(self):
        with tempfile.TemporaryDirectory() as d:
            index_fixture_vault(Path(d), reset=True)
            store = SQLiteStore(Path(d) / 'index' / 'trove.sqlite')
            indexed = SQLiteVectorStore(store).index_all_messages(FakeEmbeddingProvider(dimensions=8), max_messages=2)
            self.assertEqual(indexed, 2)
            self.assertIn('会话:', vector_document_text(next(store.iter_messages())))

    def test_semantic_tags_are_added_for_business_terms(self):
        text = vector_document_text({
            'source_type': 'message',
            'conversation_title': '客户群',
            'conversation_type': 'group',
            'sender_name': '客户',
            'direction': 'incoming',
            'timestamp': '2026-06-25T10:00:00',
            'content': '客户觉得报价太贵，需要老板审批，明天再跟进。',
        })
        self.assertIn('语义标签:', text)
        self.assertIn('商务条件/价格预算/付款异议', text)
        self.assertIn('决策进展/审批确认/上线试点', text)
        self.assertIn('下一步行动/跟进承诺', text)

    def test_v4_adds_bounded_neighbor_context_and_intent_tags(self):
        text = vector_document_text_v4(
            {
                'source_type': 'message',
                'conversation_title': '客户群',
                'conversation_type': 'group',
                'sender_name': '客户',
                'direction': 'incoming',
                'timestamp': '2026-06-25T10:00:00',
                'content': '客户需要负责人确认。',
            },
            previous_text='上文提到预算和合同付款卡点。',
            next_text='下文约明天继续跟进。',
            previous_actor='销售',
            next_actor='客户',
            max_neighbor_chars=6,
        )
        self.assertIn('结构化意图标签:', text)
        self.assertIn('customer_profile', text)
        self.assertIn('commercial_blocker', text)
        self.assertIn('follow_up', text)
        self.assertIn('相邻上文:', text)
        self.assertIn('相邻下文:', text)
        self.assertIn('上文提到预算…', text)


if __name__ == '__main__':
    unittest.main()
