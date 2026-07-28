#!/usr/bin/env python3
"""Redacted cloud-embedding probe for current vs experimental vector text.

This is not a full vector-index replacement benchmark. It answers a narrower
question cheaply: given a bounded evidence candidate pool, does a stronger
cloud embedding rank the expected evidence higher, and does richer vector text
help? It never writes raw queries, snippets, citations, paths, or vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

from trove_core.embedding.openai_compatible_provider import OpenAICompatibleEmbeddingProvider
from trove_core.approvals import (
    ApprovalManager,
    ApprovalRequired,
    approval_required_payload,
    claim_approval_grant,
)
from trove_core.security.egress import cloud_embedding_payload, content_set_digest
from trove_core.search.eval_schema import expected_citations, load_case_pack, stable_hash
from trove_core.store.sqlite_store import SQLiteStore, open_store, vector_document_text
from trove_core.vault.config import VaultConfig
from trove_core.vault.locks import VaultOperationLock


COMMERCIAL_TERMS = ('价格', '报价', '预算', '太贵', '费用', '成本', '付款', '合同')
DECISION_TERMS = ('决定', '决策', '确认', '审批', '上线', '试点', '交付', '推进')
CUSTOMER_TERMS = ('客户', '老板', '负责人', '联系人', '团队', '需求', '痛点')
RISK_TERMS = ('风险', '担心', '问题', '卡点', '阻碍', '异议', '不同意')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    if not denom:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def value(row: Any, key: str) -> str:
    try:
        if hasattr(row, 'keys') and key not in row.keys():
            return ''
        return str(row[key] or '')
    except Exception:
        return ''


def experimental_vector_text(row: Any) -> str:
    content = value(row, 'content')
    tags: list[str] = []
    if any(term in content for term in COMMERCIAL_TERMS):
        tags.append('商务条件/价格预算/付款异议')
    if any(term in content for term in DECISION_TERMS):
        tags.append('决策进展/审批确认/上线试点')
    if any(term in content for term in CUSTOMER_TERMS):
        tags.append('客户画像/联系人/需求痛点')
    if any(term in content for term in RISK_TERMS):
        tags.append('风险卡点/反对意见/推进阻碍')
    parts = [
        '检索对象: 微信聊天证据',
        f"来源类型: {value(row, 'source_type')}",
        f"会话标题: {value(row, 'conversation_title')}",
        f"说话人: {value(row, 'sender_name')}",
        f"方向: {value(row, 'direction')}",
        f"时间: {value(row, 'timestamp')}",
        f"语义标签: {'; '.join(tags)}" if tags else '',
        f"证据正文: {content}",
    ]
    return '\n'.join(part for part in parts if part.strip())


def dedupe_rows(rows: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for row in rows:
        citation = value(row, 'citation')
        if not citation or citation in seen:
            continue
        seen.add(citation)
        out.append(row)
    return out


def random_negative_rows(store: SQLiteStore, *, limit: int) -> list[Any]:
    if not store.path.exists() or limit <= 0:
        return []
    with store.connect() as conn:
        return list(conn.execute('SELECT * FROM messages ORDER BY RANDOM() LIMIT ?', (limit,)))


def candidate_rows(store: SQLiteStore, case: dict[str, Any], *, per_route: int, random_negatives: int) -> list[Any]:
    query = case['query']
    filters = dict(case.get('filters') or {})
    rows: list[Any] = []
    rows.extend(store.exact_search(query, filters=filters, limit=per_route))
    rows.extend(store.fts_search_filtered(query, filters=filters, limit=per_route, allow_like_fallback=False))
    rows.extend(store.chunk_search(query, filters=filters, limit=per_route))
    rows.extend(store.multisource_search(query, filters=filters, limit=per_route))
    if filters:
        rows.extend(store.metadata_search(query, filters=filters, limit=per_route))
    for citation in expected_citations(case):
        row = store.evidence_by_citation(citation)
        if row is not None:
            rows.append(row)
    rows.extend(random_negative_rows(store, limit=random_negatives))
    return dedupe_rows(rows)


def rank_case(provider: OpenAICompatibleEmbeddingProvider, store: SQLiteStore, case: dict[str, Any], *, per_route: int, random_negatives: int, k: int, text_mode: str) -> dict[str, Any]:
    rows = candidate_rows(store, case, per_route=per_route, random_negatives=random_negatives)
    expected = set(expected_citations(case))
    start = time.perf_counter()
    query_vector = provider.embed_query(case['query'])
    scored: list[tuple[float, str]] = []
    for row in rows:
        text = experimental_vector_text(row) if text_mode == 'experimental' else vector_document_text(row)
        scored.append((cosine(query_vector, provider.embed(text)), value(row, 'citation')))
    scored.sort(reverse=True, key=lambda item: item[0])
    top = scored[:k]
    relevance = [1 if citation in expected else 0 for _score, citation in top]
    hit = any(relevance)
    reciprocal = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            reciprocal = 1.0 / idx
            break
    return {
        'case_hash': stable_hash(case.get('case_id')),
        'candidate_count': len(rows),
        'has_expected_in_pool': bool(expected and any(value(row, 'citation') in expected for row in rows)),
        'hit': hit,
        'mrr': reciprocal,
        'precision_at_k': sum(relevance) / k,
        'elapsed_ms': round((time.perf_counter() - start) * 1000, 3),
    }


def summarize(rows: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    count = len(rows)
    if not count:
        return {'queries': 0}
    sorted_latency = sorted(float(row['elapsed_ms']) for row in rows)
    p95_idx = min(len(sorted_latency) - 1, max(0, math.ceil(len(sorted_latency) * 0.95) - 1))
    return {
        'queries': count,
        f'recall_at_{k}': sum(1 for row in rows if row['hit']) / count,
        'mrr': sum(float(row['mrr']) for row in rows) / count,
        f'precision_at_{k}': sum(float(row['precision_at_k']) for row in rows) / count,
        'avg_candidate_count': sum(int(row['candidate_count']) for row in rows) / count,
        'candidate_pool_expected_coverage': sum(1 for row in rows if row['has_expected_in_pool']) / count,
        'p95_latency_ms': sorted_latency[p95_idx],
    }


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sqlite_snapshot_digest(path: Path) -> str:
    values: list[str] = []
    for candidate in (path, Path(str(path) + '-wal'), Path(str(path) + '-shm')):
        if candidate.exists() and candidate.is_file():
            values.append(f'{candidate.name}:{candidate.stat().st_size}:{_file_sha256(candidate)}')
    return content_set_digest(values)


def _execute_probe(args: argparse.Namespace, cfg: VaultConfig, case_path: Path, store: SQLiteStore) -> int:
    cases = load_case_pack(case_path)
    rng = random.Random(args.sample_seed)
    if args.max_cases and len(cases) > args.max_cases:
        cases = rng.sample(cases, args.max_cases)
    snapshot_digest = _sqlite_snapshot_digest(cfg.paths.sqlite_path)
    controls_digest = content_set_digest([
        _file_sha256(case_path),
        snapshot_digest,
        str(args.max_cases),
        str(args.sample_seed),
        str(args.k),
        str(args.per_route),
        str(args.random_negatives),
        args.request_format,
    ])
    approval_payload = cloud_embedding_payload(
        operation='cloud_embedding_probe',
        provider='volcengine',
        model=args.model,
        dimensions=0,
        endpoint=args.endpoint,
        input_digest=controls_digest,
        item_count=len(cases),
    )
    manager = ApprovalManager(cfg.root)
    try:
        grant = manager.require(
            'cloud_embedding_probe',
            'cloud_embedding_upload',
            approval_payload,
            approval_id=args.approval_id,
            one_step_approval=args.yes,
        )
    except ApprovalRequired as exc:
        print(json.dumps(approval_required_payload(exc.record, code=exc.code), ensure_ascii=False, sort_keys=True))
        return 3
    claim_approval_grant(
        grant,
        cfg.root,
        action='cloud_embedding_probe',
        danger_class='cloud_embedding_upload',
        payload=approval_payload,
    )
    if _sqlite_snapshot_digest(cfg.paths.sqlite_path) != snapshot_digest:
        raise SystemExit('cloud embedding probe input changed before execution')

    # Construction is deliberately below the approval claim.  A missing,
    # mismatched, or replayed approval cannot initialize an outbound provider.
    provider = OpenAICompatibleEmbeddingProvider(
        enabled=True,
        endpoint=args.endpoint,
        model=args.model,
        api_key_env=args.api_key_env,
        request_format=args.request_format,
        provider_name='volcengine',
    )
    report: dict[str, Any] = {
        'schema_version': 1,
        'artifact_type': 'cloud_embedding_text_probe_redacted',
        'created_at': now_iso(),
        'case_count': len(cases),
        'k': args.k,
        'controls': {
            'max_cases': args.max_cases,
            'sample_seed': args.sample_seed,
            'per_route': args.per_route,
            'random_negatives': args.random_negatives,
            'model': args.model,
            'request_format': args.request_format,
        },
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
            'token_values_included': False,
            'vectors_included': False,
        },
        'approval': grant.to_dict(),
        'modes': {},
    }
    for text_mode in ['current', 'experimental']:
        rows = [rank_case(provider, store, case, per_route=args.per_route, random_negatives=args.random_negatives, k=args.k, text_mode=text_mode) for case in cases]
        report['modes'][text_mode] = {
            'metrics': summarize(rows, k=args.k),
            'cases': rows,
        }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'case_count': report['case_count'],
        'redacted_file': out.name,
        'modes': {mode: body['metrics'] for mode, body in report['modes'].items()},
        'raw_queries_printed': False,
        'raw_snippets_printed': False,
        'private_paths_printed': False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault')
    parser.add_argument('--cases', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-cases', type=int, default=10)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--per-route', type=int, default=20)
    parser.add_argument('--random-negatives', type=int, default=40)
    parser.add_argument('--endpoint', default='https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal')
    parser.add_argument('--model', default='doubao-embedding-vision-251215')
    parser.add_argument('--api-key-env', default='VOLCENGINE_ARK_API_KEY')
    parser.add_argument('--request-format', default='volcengine-multimodal')
    parser.add_argument('--approval', dest='approval_id')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)

    for field in ('max_cases', 'sample_seed', 'k', 'per_route', 'random_negatives'):
        value = getattr(args, field)
        minimum = 1 if field in {'k', 'per_route'} else 0
        if type(value) is not int or value < minimum or value > 1_000_000:
            parser.error(f'--{field.replace("_", "-")} is outside the supported bounds')

    cfg = VaultConfig.resolve(args.vault)
    cfg.require_configured_for_write(action='Cloud embedding probe')
    case_path = Path(args.cases).expanduser().resolve()
    store = open_store(cfg.paths.sqlite_path, readonly=True)
    try:
        # Keep the approved database/case snapshot stable for the entire
        # outbound benchmark.  Coordinated writers cannot change which text is
        # uploaded after the approval payload has been claimed.
        with VaultOperationLock(cfg, owner='vector-index'):
            return _execute_probe(args, cfg, case_path, store)
    finally:
        store.close()


if __name__ == '__main__':
    raise SystemExit(main())
