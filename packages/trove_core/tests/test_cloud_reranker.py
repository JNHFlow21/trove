from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.approvals import ApprovalManager, ApprovalValidationError, claim_approval_grant
from trove_core.search.cloud_reranker import CloudRerankProvider, rerank_document_text, rerank_with_cloud_model
from trove_core.security.egress import cloud_rerank_payload


class FakeResponse:
    def __init__(self, status_code: int, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def ranked_row(citation: str, content: str):
    return ({
        'citation': citation,
        'source_type': 'message',
        'conversation_title': '客户群',
        'conversation_type': 'group',
        'sender_name': '客户',
        'direction': 'incoming',
        'timestamp': '2026-06-25T10:00:00',
        'content': content,
    }, ['exact'], 1.0)


class CloudRerankerTests(unittest.TestCase):
    def _grant(self, vault: Path, provider, rows, query: str):
        documents = [rerank_document_text(row) for row, _paths, _score in rows]
        payload = cloud_rerank_payload(
            query=query,
            documents=documents,
            top_n=len(documents),
            provider=provider.provider_name,
            model=provider.model,
            endpoint=provider.endpoint,
        )
        manager = ApprovalManager(vault)
        record = manager.request('cloud_rerank', 'cloud_rerank_upload', payload)
        manager.decide(record.approval_id, 'approved')
        grant = manager.require(
            'cloud_rerank',
            'cloud_rerank_upload',
            payload,
            approval_id=record.approval_id,
        )
        claim_approval_grant(
            grant,
            vault,
            action='cloud_rerank',
            danger_class='cloud_rerank_upload',
            payload=payload,
        )
        return grant, payload

    def test_provider_fails_closed_without_explicit_enable(self):
        with self.assertRaises(RuntimeError):
            CloudRerankProvider(endpoint='https://example.invalid/reranks', model='qwen3-rerank')

    def test_provider_reorders_candidates_and_redacts_status(self):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResponse(200, {'results': [
                {'index': 1, 'relevance_score': 0.93},
                {'index': 0, 'relevance_score': 0.12},
            ], 'usage': {'total_tokens': 123}})

        rows = [ranked_row('c1', '报价太贵'), ranked_row('c2', '客户预算审批通过')]
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            provider = CloudRerankProvider(
                enabled=True,
                endpoint='https://example.invalid/reranks',
                model='qwen3-rerank',
                api_key='test-key',
                provider_name='aliyun',
                post=post,
            )
            grant, payload = self._grant(vault, provider, rows, '预算审批')
            ordered, status = rerank_with_cloud_model(
                rows,
                '预算审批',
                provider=provider,
                limit=2,
                candidate_limit=2,
                approval_grant=grant,
                approval_payload=payload,
                vault_root=vault,
            )

        self.assertEqual(ordered[0][0]['citation'], 'c2')
        self.assertEqual(status['state'], 'available')
        self.assertEqual(status['candidate_count'], 2)
        self.assertEqual(status['returned_count'], 2)
        self.assertEqual(status['input_tokens'], 123)
        self.assertEqual(status['estimated_cost_usd'], 0.0000123)
        self.assertNotIn('test-key', str(status))
        self.assertEqual(calls[0][1]['json']['model'], 'qwen3-rerank')
        self.assertEqual(calls[0][1]['json']['top_n'], 2)
        self.assertIn('证据正文:', calls[0][1]['json']['documents'][0])

    def test_unconfigured_cloud_reranker_falls_back_to_features(self):
        rows = [ranked_row('c1', '报价太贵')]
        with patch.dict(os.environ, {}, clear=True):
            ordered, status = rerank_with_cloud_model(rows, '预算', provider=None, limit=1, candidate_limit=50)
        self.assertEqual(ordered, rows)
        self.assertEqual(status['state'], 'unavailable_fallback')
        self.assertEqual(status['reason_code'], 'cloud_reranker_requires_exact_approval')
        self.assertEqual(status['fallback_mode'], 'features')
        self.assertEqual(status['candidate_count'], 0)
        self.assertEqual(status['returned_count'], 0)

    def test_response_evidence_text_uses_bounded_snippet_contract(self):
        text = rerank_document_text({
            'citation': 'c1',
            'source_type': 'message',
            'conversation_title': 'fixture conversation',
            'conversation_type': 'private',
            'sender_name': 'fixture sender',
            'direction': 'incoming',
            'timestamp': '2026-01-01T00:00:00Z',
            'snippet': 'fixture bounded snippet',
        })
        self.assertIn('证据正文: fixture bounded snippet', text)
        self.assertIn('会话: fixture conversation', text)

    def test_cloud_environment_cannot_auto_upload_query_or_documents(self):
        rows = [ranked_row('c1', 'private synthetic evidence')]
        calls = []

        def post(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError('ordinary rerank must not perform network I/O')

        with patch.dict(os.environ, {
            'TROVE_ENABLE_CLOUD_RERANK': '1',
            'DASHSCOPE_API_KEY': 'synthetic-test-key',
        }, clear=True), patch('trove_core.search.cloud_reranker._httpx_post', side_effect=post):
            ordered, status = rerank_with_cloud_model(
                rows,
                'private synthetic query',
                provider=None,
                limit=1,
                candidate_limit=1,
            )

        self.assertEqual(ordered, rows)
        self.assertEqual(status['reason_code'], 'cloud_reranker_requires_exact_approval')
        self.assertEqual(calls, [])

    def test_malformed_response_degrades_without_reordering(self):
        def post(_url, **_kwargs):
            return FakeResponse(200, {'unexpected': []})

        rows = [ranked_row('c1', '报价太贵'), ranked_row('c2', '客户预算审批通过')]
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            provider = CloudRerankProvider(
                enabled=True,
                endpoint='https://example.invalid/reranks',
                model='qwen3-rerank',
                api_key='test-key',
                post=post,
            )
            grant, payload = self._grant(vault, provider, rows, '预算审批')
            ordered, status = rerank_with_cloud_model(
                rows,
                '预算审批',
                provider=provider,
                limit=2,
                candidate_limit=2,
                approval_grant=grant,
                approval_payload=payload,
                vault_root=vault,
            )
        self.assertEqual([r[0]['citation'] for r in ordered], ['c1', 'c2'])
        self.assertEqual(status['state'], 'degraded')
        self.assertEqual(status['reason_code'], 'cloud_rerank_missing_results')
        self.assertEqual(status['fallback_mode'], 'features')

    def test_cloud_reranker_import_is_local_safe_without_httpx(self):
        code = r'''
import builtins
import os

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "httpx" or name.startswith("httpx."):
        raise ModuleNotFoundError("No module named 'httpx'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
from trove_core.approvals import ApprovalValidationError
from trove_core.search.cloud_reranker import CloudRerankProvider, rerank_with_cloud_model

rows = [({"citation": "c1", "content": "local"}, ["exact"], 1.0)]
ordered, status = rerank_with_cloud_model(rows, "local", provider=None, limit=1, candidate_limit=1)
assert ordered == rows
assert status["state"] == "unavailable_fallback"

provider = CloudRerankProvider(
    enabled=True,
    endpoint="https://example.invalid/reranks",
    model="qwen3-rerank",
    api_key="test-key",
)
try:
    provider.rerank("local", ["doc"], top_n=1)
except ApprovalValidationError:
    raise SystemExit(0)
raise SystemExit(1)
'''
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == '__main__':
    unittest.main()
