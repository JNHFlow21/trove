#!/usr/bin/env python3
"""Synthetic vector-metadata/dirty-ledger/watcher scaling measurement.

The default PR-sized run is 10k.  CI/nightly can pass ``--rows 100000``.
The manual release command is documented in ADR 0002 and accepts one million
ledger rows while independently capping filesystem fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.ledger import VectorIndexLedger
from trove_core.vector.zvec_store import ZVecStore
from trove_core.watch import (
    MAX_DIRECTORY_WATCHES,
    MAX_EVENTS_PER_TICK,
    MAX_SCAN_ENTRIES_PER_TICK,
    ManifestPollingBackend,
    load_watch_manifest,
)


DIRTY_BATCH_LIMIT = 256


def _bounded_watch_fields(entries: int) -> dict[str, int]:
    return {
        'bounded_fallback_max_entries_per_tick': MAX_SCAN_ENTRIES_PER_TICK,
        'bounded_fallback_ticks': (max(0, int(entries)) + MAX_SCAN_ENTRIES_PER_TICK - 1) // MAX_SCAN_ENTRIES_PER_TICK,
        'native_directory_watch_limit': MAX_DIRECTORY_WATCHES,
        'native_event_batch_limit': MAX_EVENTS_PER_TICK,
    }


def _legacy_snapshot_tree_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    latest = path.stat().st_mtime
    for child in path.rglob('*'):
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
    return latest


def _citation(index: int) -> str:
    return f'trove://synthetic/vector/{index:09d}'


def _legacy_metadata(rows: int) -> tuple[int, float]:
    started = time.perf_counter()
    hashes = {
        _citation(index): hashlib.sha256(f'synthetic-{index}'.encode('ascii')).hexdigest()
        for index in range(rows)
    }
    payload = json.dumps({'content_hashes': hashes}, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return len(payload), (time.perf_counter() - started) * 1000


def _production_vector_measurement(root: Path, rows: int) -> dict[str, object]:
    """Exercise the real authoritative ledger and constant ZVEC sidecar."""

    store = SQLiteStore(root / 'vector-ledger.sqlite')
    generation_id = 'synthetic-generation-00000001'
    ledger = VectorIndexLedger(store, backend='zvec')
    ledger.begin_generation(
        generation_id,
        vector_text_version=3,
        embedding_provider='synthetic',
        embedding_model='synthetic',
        dimensions=8,
        expected_count=rows,
    )
    with store.connect() as conn:
        conn.executemany(
            """INSERT INTO vector_index_ledger(
                   backend,generation_id,citation,content_hash,state,updated_at
               ) VALUES('zvec',?,?,?,'indexed','2026-01-01T00:00:00Z')""",
            (
                (generation_id, _citation(index), hashlib.sha256(f'synthetic-{index}'.encode('ascii')).hexdigest())
                for index in range(rows)
            ),
        )
        conn.execute(
            """UPDATE vector_index_generations SET indexed_count=?
               WHERE backend='zvec' AND generation_id=?""",
            (rows, generation_id),
        )
        conn.commit()
        authoritative_rows = int(conn.execute(
            """SELECT COUNT(*) FROM vector_index_ledger
               WHERE backend='zvec' AND generation_id=?""",
            (generation_id,),
        ).fetchone()[0])

    target = _citation(rows // 2)
    changed_hash = hashlib.sha256(b'synthetic-changed').hexdigest()
    started = time.perf_counter()
    delta = ledger.apply_delta(
        generation_id,
        upserts=[(target, changed_hash)],
        expected_count=rows,
    )
    delta_ms = (time.perf_counter() - started) * 1000
    generation = ledger.generation(generation_id)
    hash_verified = ledger.hashes(generation_id, [target]).get(target) == changed_hash
    ledger.mark_ready(generation_id, expected_count=rows)
    ledger.activate(generation_id)
    active = ledger.active_generation()

    zvec = ZVecStore(root / 'vectors' / 'messages', store)
    zvec._write_metadata({
        'schema_version': 4,
        'backend': 'zvec',
        'complete': True,
        'generation_id': generation_id,
        'generation_revision': generation.revision if generation is not None else 0,
        'indexed_count': rows,
        'expected_count': rows,
        'vector_text_version': 3,
        'embedding_provider': 'synthetic',
        'embedding_model': 'synthetic',
        'dimensions': 8,
    })
    metadata = zvec._read_metadata()
    authoritative_metadata = zvec._authoritative_score_metadata(metadata)
    sidecar_bytes = zvec.metadata_path.stat().st_size
    result = {
        'authoritative_rows': authoritative_rows,
        'generation_indexed_count': generation.indexed_count if generation is not None else -1,
        'generation_revision': generation.revision if generation is not None else 0,
        'active_generation_verified': bool(active is not None and active.generation_id == generation_id),
        'delta_candidate_rows': int(delta.get('candidate_rows', -1)),
        'delta_sql_statements': int(delta.get('sql_statements', -1)),
        'delta_commits': int(delta.get('commits', -1)),
        'delta_revision_incremented': int(delta.get('revision_incremented', -1)),
        'delta_count_delta': int(delta.get('count_delta', -1)),
        'delta_ms': round(delta_ms, 3),
        'hash_read_verified': hash_verified,
        'constant_sidecar_bytes': sidecar_bytes,
        'metadata_roundtrip_verified': bool(
            metadata.get('schema_version') == 4
            and metadata.get('backend') == 'zvec'
            and metadata.get('complete') is True
            and int(metadata.get('indexed_count', -1)) == rows
        ),
        'authoritative_revision_verified': bool(
            generation is not None
            and authoritative_metadata.get('generation_revision') == generation.revision
        ),
    }
    store.close()
    return result


def _dirty_measurement(path: Path, rows: int) -> dict[str, float | int]:
    with sqlite3.connect(path) as conn:
        conn.execute('PRAGMA journal_mode=OFF')
        conn.execute('CREATE TABLE dirty(citation TEXT PRIMARY KEY,updated_at INTEGER NOT NULL) WITHOUT ROWID')
        conn.executemany(
            'INSERT INTO dirty(citation,updated_at) VALUES(?,?)',
            ((_citation(index), index) for index in range(rows)),
        )
        started = time.perf_counter()
        full = list(conn.execute('SELECT citation FROM dirty ORDER BY updated_at,citation'))
        full_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        count = int(conn.execute('SELECT COUNT(*) FROM dirty').fetchone()[0])
        batch = list(conn.execute(
            'SELECT citation FROM dirty ORDER BY updated_at,citation LIMIT ?',
            (DIRTY_BATCH_LIMIT,),
        ))
        bounded_ms = (time.perf_counter() - started) * 1000
    return {
        'authoritative_count': count,
        'legacy_materialized_rows': len(full),
        'legacy_materialize_ms': round(full_ms, 3),
        'bounded_batch_rows': len(batch),
        'bounded_count_limit_ms': round(bounded_ms, 3),
        'bounded_sql_statements': 2,
    }


def _production_watch_probe(root: Path, files: int) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    probe_files = min(max(0, int(files)), MAX_SCAN_ENTRIES_PER_TICK + 1)
    for index in range(probe_files):
        (root / f'fixture-{index:05d}.db').touch()
    manifest_path = root.parent / 'watch-manifest.json'
    backend = ManifestPollingBackend(
        root,
        manifest_path,
        max_entries_per_tick=MAX_SCAN_ENTRIES_PER_TICK,
    )
    ticks = []
    try:
        maximum_ticks = max(3, (probe_files + 1 + MAX_SCAN_ENTRIES_PER_TICK - 1) // MAX_SCAN_ENTRIES_PER_TICK + 2)
        for _ in range(maximum_ticks):
            tick = backend.poll(timeout=0.0)
            ticks.append(tick)
            if tick.scan_complete and not tick.scan_discarded:
                break
        idle = backend.poll(timeout=0.0)
    finally:
        backend.close()
    manifest = load_watch_manifest(manifest_path)
    return {
        'probe_files': probe_files,
        'ticks': len(ticks),
        'scan_completed': bool(ticks and ticks[-1].scan_complete and not ticks[-1].scan_discarded),
        'max_entries_processed_per_tick': max((tick.entries_processed for tick in ticks), default=0),
        'idle_entries_processed': idle.entries_processed,
        'manifest_entry_count': manifest.entry_count if manifest is not None else -1,
        'manifest_bytes': manifest_path.stat().st_size if manifest_path.exists() else 0,
    }


def _watch_measurement(root: Path, files: int, *, logical: bool = False) -> dict[str, object]:
    production_probe = _production_watch_probe(root / 'production-probe', files)
    if files <= 0:
        return {
            'fixture_files': 0,
            'legacy_stat_calls': 0,
            'legacy_scan_ms': 0.0,
            'measurement': 'actual_filesystem',
            'production_probe': production_probe,
            **_bounded_watch_fields(0),
        }
    legacy_root = root / 'legacy'
    bucket_count = min(256, max(1, files // 1000))
    if logical:
        # The legacy implementation executes exactly one root stat and one
        # stat per descendant. A logical cardinality fixture measures that
        # exact operation count without consuming a million filesystem inodes.
        stat_calls = 1 + bucket_count + files
        return {
            'fixture_files': files,
            'fixture_directories': bucket_count,
            'legacy_stat_calls': stat_calls,
            'legacy_scan_ms': None,
            'measurement': 'logical_exact_operation_count',
            'production_probe': production_probe,
            **_bounded_watch_fields(stat_calls),
        }
    buckets = [legacy_root / f'bucket-{index:03d}' for index in range(bucket_count)]
    for bucket in buckets:
        bucket.mkdir(parents=True, exist_ok=True)
    for index in range(files):
        (buckets[index % bucket_count] / f'file-{index:09d}.db').touch()
    started = time.perf_counter()
    _legacy_snapshot_tree_mtime(legacy_root)
    elapsed_ms = (time.perf_counter() - started) * 1000
    # The audited legacy helper stats root once and every descendant once.
    stat_calls = 1 + bucket_count + files
    return {
        'fixture_files': files,
        'fixture_directories': bucket_count,
        'legacy_stat_calls': stat_calls,
        'legacy_scan_ms': round(elapsed_ms, 3),
        'measurement': 'actual_filesystem',
        'production_probe': production_probe,
        **_bounded_watch_fields(stat_calls),
    }


def measure(rows: int, watch_files: int, *, logical_watch: bool = False) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='trove-u10-synthetic-') as directory:
        root = Path(directory)
        legacy_bytes, legacy_ms = _legacy_metadata(rows)
        vector = _production_vector_measurement(root, rows)
        dirty = _dirty_measurement(root / 'dirty.sqlite', rows)
        watch = _watch_measurement(root / 'watch', watch_files, logical=logical_watch)
    thresholds = {
        'collection_metadata_max_bytes': 4096,
        'ledger_delta_max_candidate_rows': 1,
        'ledger_delta_max_sql_statements': 10,
        'ledger_delta_max_ms': 5000.0,
        'dirty_batch_max_rows': 512,
        'dirty_sql_statements_max': 2,
        'fallback_rescan_max_entries_per_tick': 4096,
        'native_directory_watch_max': 1024,
        'native_event_batch_max': 256,
        'idle_entries_processed_max': 0,
        'watch_manifest_max_bytes': 1024,
    }
    production_probe = watch['production_probe']
    checks = {
        'ledger_cardinality_verified': (
            vector['authoritative_rows'] == rows
            and vector['generation_indexed_count'] == rows
            and vector['active_generation_verified'] is True
        ),
        'ledger_delta_is_constant': (
            vector['delta_candidate_rows'] == thresholds['ledger_delta_max_candidate_rows']
            and 1 <= vector['delta_sql_statements'] <= thresholds['ledger_delta_max_sql_statements']
            and vector['delta_commits'] == 1
            and vector['delta_revision_incremented'] == 1
            and vector['delta_count_delta'] == 0
            and 0 <= vector['delta_ms'] <= thresholds['ledger_delta_max_ms']
            and vector['hash_read_verified'] is True
        ),
        'metadata_is_real_constant_sidecar': (
            0 < vector['constant_sidecar_bytes'] <= thresholds['collection_metadata_max_bytes']
            and vector['metadata_roundtrip_verified'] is True
            and vector['authoritative_revision_verified'] is True
        ),
        'dirty_backlog_is_bounded': (
            dirty['authoritative_count'] == rows
            and dirty['bounded_batch_rows'] <= thresholds['dirty_batch_max_rows']
            and dirty['bounded_sql_statements'] <= thresholds['dirty_sql_statements_max']
        ),
        'watch_plan_uses_production_bounds': (
            watch['bounded_fallback_max_entries_per_tick']
            <= thresholds['fallback_rescan_max_entries_per_tick']
            and watch['native_directory_watch_limit'] <= thresholds['native_directory_watch_max']
            and watch['native_event_batch_limit'] <= thresholds['native_event_batch_max']
            and production_probe['scan_completed'] is True
            and production_probe['max_entries_processed_per_tick']
            <= thresholds['fallback_rescan_max_entries_per_tick']
            and production_probe['idle_entries_processed'] <= thresholds['idle_entries_processed_max']
            and 0 < production_probe['manifest_bytes'] <= thresholds['watch_manifest_max_bytes']
            and production_probe['manifest_entry_count'] == production_probe['probe_files'] + 1
        ),
    }
    return {
        'ok': all(checks.values()),
        'fixture': {'ledger_rows': rows, 'watch_files': watch_files, 'synthetic_only': True},
        'vector_metadata': {
            'legacy_content_hash_bytes': legacy_bytes,
            'legacy_bytes_per_row': round(legacy_bytes / rows, 3),
            'legacy_serialize_ms': round(legacy_ms, 3),
            **vector,
        },
        'dirty_backlog': dirty,
        'watcher': watch,
        'thresholds': thresholds,
        'checks': checks,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=10_000)
    parser.add_argument('--watch-files', type=int)
    parser.add_argument('--logical-watch', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.rows <= 0:
        parser.error('--rows must be positive')
    watch_files = args.rows if args.watch_files is None else max(0, args.watch_files)
    report = measure(args.rows, watch_files, logical_watch=args.logical_watch)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
