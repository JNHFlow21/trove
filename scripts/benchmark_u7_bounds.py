#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)

for relative in ('packages/trove_core',):
    path = str(_ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)

from trove_core.embedding.fake_provider import FakeEmbeddingProvider  # noqa: E402
from trove_core.store.schema import SCHEMA_VERSION  # noqa: E402
from trove_core.store.sqlite_store import SQLiteStore  # noqa: E402
from trove_core.vector.sqlite_vector_store import (  # noqa: E402
    SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT,
    SQLiteVectorStore,
)


def _elapsed_ms(call):
    start = time.perf_counter()
    value = call()
    return value, (time.perf_counter() - start) * 1000.0


def _build_fixture(store: SQLiteStore, rows: int) -> None:
    store.initialize()
    conversations = max(1, rows // 100)
    with store.connect() as conn:
        conn.execute('INSERT INTO accounts(account_id,label,display_name) VALUES(?,?,?)', ('acct-synthetic', 'fixture', 'Fixture'))
        conn.executemany(
            'INSERT INTO conversations(conversation_id,account_id,title,type,member_count) VALUES(?,?,?,?,?)',
            [(f'conv-{index:04d}', 'acct-synthetic', f'Conversation {index:04d}', 'private', 2) for index in range(conversations)],
        )
        messages = []
        vectors = []
        vector_json = json.dumps([0.125] * 8, separators=(',', ':'))
        for index in range(rows):
            conversation_id = f'conv-{index % conversations:04d}'
            content = f'synthetic bounded message {index:05d}'
            if index % 997 == 0:
                content += ' needle 针针'
            if index == rows - 1:
                content += ' normalized\nphrase'
            citation = f'trove://synthetic/{index:05d}'
            messages.append((
                citation, 'acct-synthetic', 'fixture', conversation_id, f'Conversation {index % conversations:04d}',
                'private', f'sender-{index % 10}', f'Sender {index % 10}', f'2026-01-01T00:{index % 60:02d}:00Z',
                content, 'text', 'synthetic-shard', index, 0, 'message', 'incoming',
            ))
            vectors.append((citation, 'synthetic', 8, vector_json, f'hash-{index:05d}'))
        conn.executemany(
            """INSERT INTO messages(
                citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
                sender_id,sender_name,timestamp,content,content_kind,shard_id,local_id,sent_by_me,source_type,direction
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            messages,
        )
        conn.executemany(
            'INSERT INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)',
            vectors,
        )
        conn.commit()


def run_benchmark(*, rows: int = 10_000, rounds: int = 3) -> dict:
    if type(rows) is not int or not 1 <= rows <= SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT:
        raise ValueError(f'rows must be 1..{SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT}')
    if type(rounds) is not int or not 1 <= rounds <= 20:
        raise ValueError('rounds must be 1..20')
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteStore(Path(directory) / 'trove.sqlite')
        _build_fixture(store, rows)
        vector_store = SQLiteVectorStore(store)
        provider = FakeEmbeddingProvider(dimensions=8)
        short_latencies: list[float] = []
        two_character_latencies: list[float] = []
        normalized_latencies: list[float] = []
        short_rows = []
        two_character_rows = []
        normalized_rows = []
        for _ in range(rounds):
            short_rows, elapsed = _elapsed_ms(lambda: store.exact_search('针', limit=10))
            short_latencies.append(elapsed)
            two_character_rows, elapsed = _elapsed_ms(lambda: store.exact_search('针针', limit=10))
            two_character_latencies.append(elapsed)
            normalized_rows, elapsed = _elapsed_ms(lambda: store.exact_search('normalized phrase', limit=10))
            normalized_latencies.append(elapsed)

        statements: list[str] = []
        store.connect().set_trace_callback(statements.append)
        vector_rows, vector_ms = _elapsed_ms(lambda: vector_store.search(
            'synthetic vector query',
            filters={'conversation_id': 'conv-0001'},
            limit=10,
            provider=provider,
        ))
        store.connect().set_trace_callback(None)
        vector_sql = [sql for sql in statements if sql.lstrip().upper().startswith(('SELECT', 'WITH'))]

        with store.connect() as conn:
            plans = {
                'conversation': ' '.join(str(tuple(row)) for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT * FROM conversations WHERE conversation_id=?', ('conv-0001',),
                )),
                'message_scope': ' '.join(str(tuple(row)) for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT * FROM messages WHERE conversation_id=? ORDER BY timestamp DESC LIMIT ?',
                    ('conv-0001', 10),
                )),
                'chunk_scope': ' '.join(str(tuple(row)) for row in conn.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM evidence_chunks WHERE source_id=? AND status='active' ORDER BY timestamp DESC LIMIT ?",
                    ('conv-0001', 10),
                )),
            }
            plans['vector_filter'] = (
                ' '.join(str(tuple(row)) for row in conn.execute('EXPLAIN QUERY PLAN ' + vector_sql[0]))
                if vector_sql else ''
            )

        expected_indexes = {
            'conversation': 'idx_conversations_id_account',
            'message_scope': 'idx_messages_conversation_time',
            'chunk_scope': 'idx_evidence_chunks_source_id_status_time',
            'vector_filter': 'idx_messages_conversation_time',
        }
        checks = {
            'short_results_bounded': len(short_rows) <= 10,
            'two_character_results_bounded': len(two_character_rows) <= 10,
            'normalized_results_bounded': len(normalized_rows) <= 10,
            'vector_results_bounded': len(vector_rows) <= 10,
            'vector_single_sql_statement': len(vector_sql) == 1,
            'explain_indexes_used': all(expected_indexes[name] in plans[name] for name in expected_indexes),
            'sqlite_vector_fixture_within_cap': rows <= SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT,
        }
        report = {
            'schema_version': 1,
            'artifact_type': 'u7_bounded_complexity_synthetic',
            'fixture': {'kind': 'synthetic', 'messages': rows, 'vectors': rows, 'schema_version': SCHEMA_VERSION},
            'rounds': rounds,
            'latency_ms': {
                'one_character_like_p50': round(statistics.median(short_latencies), 3),
                'two_character_like_p50': round(statistics.median(two_character_latencies), 3),
                'normalized_phrase_p50': round(statistics.median(normalized_latencies), 3),
                'scoped_sqlite_vector': round(vector_ms, 3),
            },
            'complexity': {
                'result_limit': 10,
                'one_character_returned': len(short_rows),
                'two_character_returned': len(two_character_rows),
                'normalized_phrase_returned': len(normalized_rows),
                'scoped_vector_returned': len(vector_rows),
                'scoped_vector_sql_statements': len(vector_sql),
                'sqlite_vector_matching_entry_cap': SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT,
            },
            'explain': {name: {'required_index': expected_indexes[name], 'used': expected_indexes[name] in plan} for name, plan in plans.items()},
            'checks': checks,
            'ok': all(checks.values()),
            'privacy': {
                'synthetic_only': True,
                'raw_queries_included': False,
                'raw_content_included': False,
                'raw_citations_included': False,
                'private_paths_included': False,
            },
        }
        store.close()
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the deterministic U7 10k bounded-query complexity benchmark.')
    parser.add_argument('--rows', type=int, default=10_000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    report = run_benchmark(rows=args.rows, rounds=args.rounds)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if args.out:
        Path(args.out).write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
