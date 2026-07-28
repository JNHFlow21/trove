#!/usr/bin/env python3
"""Run the Q5 vector_text_v4 throwaway sample-index experiment.

The script may read the selected private Vault and private case pack, but it
only writes a redacted aggregate report.  Scratch SQLite/ZVEC indexes are
created outside the product collection and removed by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

import argparse
import hashlib
import json
import math
import random
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from diagnose_retrieval_failures import diagnose
from trove_core.runtime import configured_embedding_provider
from trove_core.search.eval_schema import expected_citations, load_case_pack, stable_hash, validate_redacted_artifact
from trove_core.store.sqlite_store import EvidenceRow, SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vector.text_versions import (
    VECTOR_TEXT_V4_EXPERIMENT_VERSION,
    vector_document_text_v3,
    vector_document_text_v4,
)
from trove_core.vector.zvec_store import VECTOR_TEXT_VERSION as PRODUCTION_VECTOR_TEXT_VERSION
from trove_core.vector.zvec_store import ZVecStore


DEFAULT_DISTRACTORS = 50_000
DEFAULT_SEED = 20260709
DEFAULT_TOP_K = 50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sqlite_uri_readonly(path: Path) -> str:
    return 'file:' + quote(str(path.expanduser().resolve()), safe='/:') + '?mode=ro'


def batched(values: list[str], size: int = 500):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def row_value(row: Any, key: str) -> str:
    try:
        if hasattr(row, 'keys') and key not in row.keys():
            return ''
        return str(row[key] or '')
    except Exception:
        return ''


def source_family(case: dict[str, Any]) -> str:
    return str(case.get('source_family') or (case.get('filters') or {}).get('source_type') or 'message')


def case_is_positive(case: dict[str, Any]) -> bool:
    return bool(expected_citations(case))


def matches_expected_row(row: Any, expected: set[str]) -> bool:
    for key in ('citation', 'parent_citation', 'context_anchor'):
        value = row_value(row, key)
        if value and value in expected:
            return True
    return False


def first_rank(rows: list[Any], expected: set[str], *, limit: int) -> int | None:
    for idx, row in enumerate(rows[:limit], start=1):
        if matches_expected_row(row, expected):
            return idx
    return None


def rank_bucket(rank: int | None, *, top_k: int) -> str:
    if rank is None:
        return f'missing_top{top_k}'
    if rank <= 3:
        return '1_3'
    if rank <= 10:
        return '4_10'
    if rank <= 20:
        return '11_20'
    return f'21_{top_k}'


def coverage_summary(ranks: list[int | None], *, top_k: int) -> dict[str, Any]:
    total = len(ranks)
    hits = sum(1 for rank in ranks if rank is not None and rank <= top_k)
    finite = [rank for rank in ranks if rank is not None]
    return {
        'cases': total,
        f'top{top_k}_hits': hits,
        f'top{top_k}_coverage': (hits / total) if total else 0.0,
        'median_rank': sorted(finite)[len(finite) // 2] if finite else None,
        'best_rank': min(finite) if finite else None,
        'worst_rank': max(finite) if finite else None,
    }


def rank_distribution(ranks: list[int | None], *, top_k: int) -> dict[str, int]:
    buckets = Counter(rank_bucket(rank, top_k=top_k) for rank in ranks)
    ordered = ['1_3', '4_10', '11_20', f'21_{top_k}', f'missing_top{top_k}']
    return {bucket: int(buckets.get(bucket, 0)) for bucket in ordered}


def find_expected_chunk_map(conn: sqlite3.Connection, cases: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for case in cases:
        ids = expected_citations(case)
        case_hash = str(stable_hash(case.get('case_id')))
        if not ids:
            out[case_hash] = set()
            continue
        placeholders = ','.join('?' for _ in ids)
        rows = list(conn.execute(
            f"""SELECT chunk_citation
                FROM evidence_chunks
                WHERE status='active'
                  AND (chunk_citation IN ({placeholders}) OR parent_citation IN ({placeholders}))""",
            [*ids, *ids],
        ))
        out[case_hash] = {str(row['chunk_citation']) for row in rows if row['chunk_citation']}
    return out


def sample_distractor_chunks(
    conn: sqlite3.Connection,
    *,
    exclude_chunk_ids: set[str],
    target: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for row in conn.execute("SELECT chunk_citation FROM evidence_chunks WHERE status='active' ORDER BY rowid"):
        citation = str(row['chunk_citation'])
        if citation in exclude_chunk_ids:
            continue
        seen += 1
        if len(reservoir) < target:
            reservoir.append(citation)
            continue
        pos = rng.randrange(seen)
        if pos < target:
            reservoir[pos] = citation
    return reservoir


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row['name']) for row in conn.execute(f'PRAGMA table_info({table})')]


def fetch_chunk_rows(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    cols = table_columns(conn, 'evidence_chunks')
    col_sql = ','.join(cols)
    rows: list[dict[str, Any]] = []
    for batch in batched(list(dict.fromkeys(chunk_ids)), 500):
        placeholders = ','.join('?' for _ in batch)
        rows.extend(dict(row) for row in conn.execute(
            f"SELECT {col_sql} FROM evidence_chunks WHERE chunk_citation IN ({placeholders})",
            batch,
        ))
    return rows


def load_neighbor_context(conn: sqlite3.Connection, selected_chunk_ids: list[str]) -> dict[str, dict[str, str]]:
    if not selected_chunk_ids:
        return {}
    conn.execute('DROP TABLE IF EXISTS temp._q5_selected_chunks')
    conn.execute('CREATE TEMP TABLE _q5_selected_chunks(chunk_citation TEXT PRIMARY KEY)')
    conn.executemany(
        'INSERT OR IGNORE INTO _q5_selected_chunks(chunk_citation) VALUES(?)',
        [(chunk_id,) for chunk_id in selected_chunk_ids],
    )
    rows = conn.execute(
        """
        WITH selected_partitions AS (
            SELECT DISTINCT e.account_id, e.source_type, e.source_id
            FROM evidence_chunks e
            JOIN _q5_selected_chunks s ON s.chunk_citation=e.chunk_citation
            WHERE e.status='active'
        ),
        ordered AS (
            SELECT
                e.chunk_citation,
                LAG(e.content) OVER (
                    PARTITION BY e.account_id, e.source_type, e.source_id
                    ORDER BY e.timestamp, e.parent_citation, e.chunk_index, e.chunk_citation
                ) AS previous_text,
                LEAD(e.content) OVER (
                    PARTITION BY e.account_id, e.source_type, e.source_id
                    ORDER BY e.timestamp, e.parent_citation, e.chunk_index, e.chunk_citation
                ) AS next_text,
                LAG(e.actor) OVER (
                    PARTITION BY e.account_id, e.source_type, e.source_id
                    ORDER BY e.timestamp, e.parent_citation, e.chunk_index, e.chunk_citation
                ) AS previous_actor,
                LEAD(e.actor) OVER (
                    PARTITION BY e.account_id, e.source_type, e.source_id
                    ORDER BY e.timestamp, e.parent_citation, e.chunk_index, e.chunk_citation
                ) AS next_actor
            FROM evidence_chunks e
            JOIN selected_partitions p
              ON p.account_id=e.account_id
             AND p.source_type=e.source_type
             AND p.source_id=e.source_id
            WHERE e.status='active'
        )
        SELECT o.*
        FROM ordered o
        JOIN _q5_selected_chunks s ON s.chunk_citation=o.chunk_citation
        """
    )
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out[str(row['chunk_citation'])] = {
            'previous_text': str(row['previous_text'] or ''),
            'next_text': str(row['next_text'] or ''),
            'previous_actor': str(row['previous_actor'] or ''),
            'next_actor': str(row['next_actor'] or ''),
        }
    conn.execute('DROP TABLE IF EXISTS temp._q5_selected_chunks')
    return out


def create_sample_store(sample_db: Path, rows: list[dict[str, Any]]) -> SQLiteStore:
    store = SQLiteStore(sample_db)
    store.initialize()
    if not rows:
        return store
    with store.connect() as conn:
        conn.execute('DELETE FROM evidence_chunks')
        cols = table_columns(conn, 'evidence_chunks')
        placeholders = ','.join('?' for _ in cols)
        col_sql = ','.join(cols)
        values = [tuple(row.get(col) for col in cols) for row in rows]
        conn.executemany(f'INSERT OR REPLACE INTO evidence_chunks({col_sql}) VALUES({placeholders})', values)
        conn.commit()
    return store


class ExperimentTextStore(SQLiteStore):
    def __init__(self, path: Path, *, text_version: str, neighbor_context: dict[str, dict[str, str]]):
        super().__init__(path)
        self.text_version = text_version
        self.neighbor_context = neighbor_context

    def iter_vector_documents(self, batch_size: int = 500, citations=None):
        citation_filter = None if citations is None else list(dict.fromkeys(str(c) for c in citations if c))
        if citation_filter is not None and not citation_filter:
            return
        conn = self.connect_once()
        try:
            params: list[Any] = []
            where = "WHERE status='active'"
            if citation_filter is not None:
                placeholders = ','.join('?' for _ in citation_filter)
                where += f" AND (parent_citation IN ({placeholders}) OR chunk_citation IN ({placeholders}))"
                params.extend([*citation_filter, *citation_filter])
            cursor = conn.execute(
                f"""SELECT chunk_citation AS citation,
                           parent_citation,
                           account_id,
                           account_label,
                           source_id AS conversation_id,
                           title AS conversation_title,
                           'private' AS conversation_type,
                           actor AS sender_id,
                           actor AS sender_name,
                           timestamp,
                           content,
                           source_type,
                           'metadata' AS direction
                    FROM evidence_chunks
                    {where}
                    ORDER BY timestamp, chunk_citation""",
                params,
            )
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    data = dict(row)
                    if self.text_version == 'v4':
                        ctx = self.neighbor_context.get(str(data.get('citation'))) or {}
                        data['vector_text'] = vector_document_text_v4(
                            data,
                            previous_text=ctx.get('previous_text', ''),
                            next_text=ctx.get('next_text', ''),
                            previous_actor=ctx.get('previous_actor', ''),
                            next_actor=ctx.get('next_actor', ''),
                        )
                    else:
                        data['vector_text'] = vector_document_text_v3(data)
                    yield EvidenceRow(data)
        finally:
            conn.close()


def build_throwaway_index(
    *,
    sample_db: Path,
    collection_dir: Path,
    provider: Any,
    text_version: str,
    neighbor_context: dict[str, dict[str, str]],
    batch_size: int,
) -> dict[str, Any]:
    store = ExperimentTextStore(sample_db, text_version=text_version, neighbor_context=neighbor_context)
    vector = ZVecStore(collection_dir, store=store)
    start = time.perf_counter()
    indexed = vector.index_all_messages(provider, store=store, batch_size=batch_size)
    elapsed = round((time.perf_counter() - start) * 1000, 3)
    return {'store': store, 'vector': vector, 'indexed': indexed, 'elapsed_ms': elapsed}


def evaluate_cases(
    *,
    vector: ZVecStore,
    provider: Any,
    cases: list[dict[str, Any]],
    top_k: int,
) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for case in cases:
        case_hash = str(stable_hash(case.get('case_id')))
        expected = set(expected_citations(case))
        try:
            rows = vector.search(
                str(case.get('query') or ''),
                filters=dict(case.get('filters') or {}),
                limit=top_k,
                provider=provider,
            )
        except Exception:
            rows = []
        ranks[case_hash] = first_rank(rows, expected, limit=top_k)
    return ranks


def compare_segment(
    name: str,
    case_hashes: list[str],
    v3_ranks: dict[str, int | None],
    v4_ranks: dict[str, int | None],
    *,
    top_k: int,
) -> dict[str, Any]:
    v3 = [v3_ranks.get(case_hash) for case_hash in case_hashes]
    v4 = [v4_ranks.get(case_hash) for case_hash in case_hashes]
    v3_summary = coverage_summary(v3, top_k=top_k)
    v4_summary = coverage_summary(v4, top_k=top_k)
    v3_hits = int(v3_summary[f'top{top_k}_hits'])
    v4_hits = int(v4_summary[f'top{top_k}_hits'])
    return {
        'segment': name,
        'cases': len(case_hashes),
        'v3': v3_summary,
        'v4': v4_summary,
        'delta_hits': v4_hits - v3_hits,
        'delta_coverage_points': ((v4_hits - v3_hits) / len(case_hashes)) if case_hashes else 0.0,
    }


def run_experiment(
    *,
    vault_root: Path,
    cases_path: Path,
    out_path: Path,
    distractors: int,
    seed: int,
    top_k: int,
    batch_size: int,
    scratch_dir: Path | None,
    keep_scratch: bool,
    model_path: str | None,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root))
    scratch_created = scratch_dir is None
    scratch = scratch_dir or Path(tempfile.mkdtemp(prefix='trove-q5-vector-text-v4-sample-'))
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_removed = False
    cleanup_error = None
    report: dict[str, Any] | None = None
    try:
        all_cases = load_case_pack(cases_path)
        positive_cases = [case for case in all_cases if case_is_positive(case)]
        case_by_hash = {str(stable_hash(case.get('case_id'))): case for case in positive_cases}
        diagnosis = diagnose(cases_path, cfg.root, k=3, model_path=model_path, vector_top_n=top_k, vector_deep_n=200)
        true_l2_hashes = sorted(
            str(row.get('case_hash'))
            for row in diagnosis.get('cases', [])
            if row.get('vector_top50_missing_reason') == 'true_l2_recall'
        )
        provider = configured_embedding_provider(model_path)
        if provider is None:
            raise RuntimeError('embedding_provider_missing')

        ro_conn = sqlite3.connect(sqlite_uri_readonly(cfg.paths.sqlite_path), uri=True)
        ro_conn.row_factory = sqlite3.Row
        try:
            case_expected_chunks = find_expected_chunk_map(ro_conn, positive_cases)
            expected_chunk_ids = sorted({chunk_id for values in case_expected_chunks.values() for chunk_id in values})
            distractor_ids = sample_distractor_chunks(
                ro_conn,
                exclude_chunk_ids=set(expected_chunk_ids),
                target=distractors,
                seed=seed,
            )
            selected_chunk_ids = list(dict.fromkeys([*expected_chunk_ids, *distractor_ids]))
            chunk_rows = fetch_chunk_rows(ro_conn, selected_chunk_ids)
            neighbor_context = load_neighbor_context(ro_conn, selected_chunk_ids)
        finally:
            ro_conn.close()

        sample_db = scratch / 'sample.sqlite'
        create_sample_store(sample_db, chunk_rows)
        v3 = build_throwaway_index(
            sample_db=sample_db,
            collection_dir=scratch / 'zvec-v3',
            provider=provider,
            text_version='v3',
            neighbor_context={},
            batch_size=batch_size,
        )
        v4 = build_throwaway_index(
            sample_db=sample_db,
            collection_dir=scratch / 'zvec-v4',
            provider=provider,
            text_version='v4',
            neighbor_context=neighbor_context,
            batch_size=batch_size,
        )
        v3_ranks = evaluate_cases(vector=v3['vector'], provider=provider, cases=positive_cases, top_k=top_k)
        v4_ranks = evaluate_cases(vector=v4['vector'], provider=provider, cases=positive_cases, top_k=top_k)

        positive_hashes = sorted(case_by_hash)
        threshold_hits = math.ceil(len(true_l2_hashes) * 0.50)
        v4_true_l2_hits = sum(1 for case_hash in true_l2_hashes if v4_ranks.get(case_hash) is not None)
        threshold_met = bool(true_l2_hashes and v4_true_l2_hits >= threshold_hits)
        true_l2_case_rows = []
        for case_hash in true_l2_hashes:
            case = case_by_hash.get(case_hash) or {}
            v3_rank = v3_ranks.get(case_hash)
            v4_rank = v4_ranks.get(case_hash)
            true_l2_case_rows.append({
                'case_hash': case_hash,
                'category': case.get('category'),
                'source_family': source_family(case),
                'v3_rank': v3_rank,
                'v4_rank': v4_rank,
                'v3_bucket': rank_bucket(v3_rank, top_k=top_k),
                'v4_bucket': rank_bucket(v4_rank, top_k=top_k),
            })

        missing_expected_chunk_cases = sorted(
            case_hash for case_hash, chunks in case_expected_chunks.items()
            if case_hash in positive_hashes and not chunks
        )
        report = {
            'schema_version': 1,
            'artifact_type': 'q5_vector_text_v4_sample_index_experiment_redacted',
            'created_at': now_iso(),
            'privacy': {
                'raw_queries_included': False,
                'raw_snippets_included': False,
                'raw_citations_included': False,
                'private_paths_included': False,
                'token_values_included': False,
            },
            'case_pack_anchor': {
                'sha256_prefix': hashlib.sha256(cases_path.expanduser().read_bytes()).hexdigest()[:32],
                'loaded_cases': len(all_cases),
                'selected_positive_cases': len(positive_cases),
                'path_included': False,
            },
            'controls': {
                'experiment_name': 'vector_text_v4_contextual_embedding_sample',
                'sample_seed': seed,
                'distractor_target': distractors,
                'top_k': top_k,
                'diagnosis_vector_deep_n': 200,
                'batch_size': batch_size,
                'full_vector_rebuild': False,
                'holdout_run': False,
                'real_vault_access': 'read_only',
                'production_vector_text_version': PRODUCTION_VECTOR_TEXT_VERSION,
                'experiment_vector_text_version': VECTOR_TEXT_V4_EXPERIMENT_VERSION,
                'production_vector_version_bumped': False,
                'cloud_embedding_opt_in': False,
                'model_configured': provider is not None,
            },
            'baseline_diagnosis': {
                'positive_cases': diagnosis.get('summary', {}).get('positive_cases'),
                'hit_top3': diagnosis.get('summary', {}).get('hit_top3'),
                'miss_top3': diagnosis.get('summary', {}).get('miss_top3'),
                'leak_counts': diagnosis.get('summary', {}).get('leak_counts'),
                'vector_top50_missing_reasons': diagnosis.get('summary', {}).get('vector_top50_missing_reasons'),
            },
            'sample_index': {
                'expected_chunk_count': len(expected_chunk_ids),
                'distractor_chunk_count': len(distractor_ids),
                'total_chunk_count': len(chunk_rows),
                'missing_expected_chunk_case_count': len(missing_expected_chunk_cases),
                'missing_expected_chunk_case_hashes': missing_expected_chunk_cases[:50],
                'v3_indexed': v3['indexed'],
                'v4_indexed': v4['indexed'],
                'v3_build_ms': v3['elapsed_ms'],
                'v4_build_ms': v4['elapsed_ms'],
                'neighbor_context_rows': len(neighbor_context),
                'scratch_created': scratch_created,
            },
            'coverage_table': [
                compare_segment('true_l2_recall', true_l2_hashes, v3_ranks, v4_ranks, top_k=top_k),
                compare_segment('all_positive', positive_hashes, v3_ranks, v4_ranks, top_k=top_k),
            ],
            'rank_distribution': {
                'true_l2_recall': {
                    'v3': rank_distribution([v3_ranks.get(case_hash) for case_hash in true_l2_hashes], top_k=top_k),
                    'v4': rank_distribution([v4_ranks.get(case_hash) for case_hash in true_l2_hashes], top_k=top_k),
                },
                'all_positive': {
                    'v3': rank_distribution([v3_ranks.get(case_hash) for case_hash in positive_hashes], top_k=top_k),
                    'v4': rank_distribution([v4_ranks.get(case_hash) for case_hash in positive_hashes], top_k=top_k),
                },
            },
            'true_l2_case_ranks': true_l2_case_rows,
            'decision': {
                'threshold_true_l2_cases': len(true_l2_hashes),
                'threshold_top50_hits_required': threshold_hits,
                'v4_true_l2_top50_hits': v4_true_l2_hits,
                'threshold_met': threshold_met,
                'recommendation': (
                    'request_full_vector_rebuild_approval'
                    if threshold_met
                    else 'close_experiment_not_worth_full_rebuild'
                ),
            },
            'cleanup': {
                'scratch_removed': False,
                'remaining_entries_after_cleanup': None,
                'cleanup_error': None,
            },
        }
        return report
    finally:
        if not keep_scratch:
            try:
                if scratch.exists():
                    shutil.rmtree(scratch)
                scratch_removed = not scratch.exists()
            except Exception as exc:  # pragma: no cover - defensive cleanup report
                cleanup_error = exc.__class__.__name__
        if report is not None:
            remaining = 0
            if scratch.exists():
                try:
                    remaining = len(list(scratch.iterdir()))
                except Exception:
                    remaining = -1
            report['cleanup'] = {
                'scratch_removed': scratch_removed,
                'remaining_entries_after_cleanup': remaining if not scratch_removed else 0,
                'cleanup_error': cleanup_error,
            }
            validate_redacted_artifact(report)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Q5 vector_text_v4 sample-index experiment.')
    parser.add_argument('--vault', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--distractors', type=int, default=DEFAULT_DISTRACTORS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--scratch-dir')
    parser.add_argument('--keep-scratch', action='store_true')
    parser.add_argument('--model-path')
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser()
    report = run_experiment(
        vault_root=Path(args.vault).expanduser(),
        cases_path=Path(args.cases).expanduser(),
        out_path=out,
        distractors=max(0, int(args.distractors)),
        seed=int(args.seed),
        top_k=max(1, int(args.top_k)),
        batch_size=max(1, int(args.batch_size)),
        scratch_dir=Path(args.scratch_dir).expanduser() if args.scratch_dir else None,
        keep_scratch=bool(args.keep_scratch),
        model_path=args.model_path,
    )
    true_l2 = next(row for row in report['coverage_table'] if row['segment'] == 'true_l2_recall')
    print(json.dumps({
        'ok': True,
        'redacted_file': out.name,
        'true_l2_cases': true_l2['cases'],
        'v3_true_l2_top50_hits': true_l2['v3'][f'top{args.top_k}_hits'],
        'v4_true_l2_top50_hits': true_l2['v4'][f'top{args.top_k}_hits'],
        'threshold_met': report['decision']['threshold_met'],
        'recommendation': report['decision']['recommendation'],
        'scratch_removed': report['cleanup']['scratch_removed'],
        'raw_queries_printed': False,
        'raw_snippets_printed': False,
        'private_paths_printed': False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
