"""Persistent import/sync change journal independent of either orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .sqlite_store import SQLiteStore


MAX_INLINE_DIRTY_CITATIONS = 512
# Compatibility key retained in existing Vaults.  It is now the shared
# publication generation for sync, full import, and standalone auxiliary import.
SYNC_COMMIT_GENERATION_KEY = 'sync_commit_generation'


def ensure_change_journal(store: SQLiteStore) -> None:
    store.initialize()


# Compatibility name retained for callers that historically imported it from
# sync.  New code should use the domain-specific name above.
ensure_sync_state = ensure_change_journal


def read_sync_commit_generation(store: SQLiteStore) -> int:
    """Return the stable sync publication generation.

    Even values are complete publications.  Odd values are an in-progress
    publication left visible so a concurrent scanner cannot prepare against a
    partially committed group of sync writes.
    """

    ensure_change_journal(store)
    with store.connect() as conn:
        row = conn.execute(
            'SELECT value FROM schema_meta WHERE key=?',
            (SYNC_COMMIT_GENERATION_KEY,),
        ).fetchone()
    return max(0, int(row['value'])) if row is not None else 0


def recover_sync_commit_generation(store: SQLiteStore) -> int:
    """Close an abandoned odd generation while holding the sync writer.

    A fresh sync will rescan the source after this recovery, repairing any
    partial prior publication without holding the writer during that scan.
    """

    ensure_change_journal(store)
    with store.connect() as conn:
        conn.execute(
            'INSERT OR IGNORE INTO schema_meta(key,value) VALUES(?,?)',
            (SYNC_COMMIT_GENERATION_KEY, '0'),
        )
        row = conn.execute(
            'SELECT value FROM schema_meta WHERE key=?',
            (SYNC_COMMIT_GENERATION_KEY,),
        ).fetchone()
        current = max(0, int(row['value']))
        if current % 2:
            current += 1
            conn.execute(
                'UPDATE schema_meta SET value=? WHERE key=?',
                (str(current), SYNC_COMMIT_GENERATION_KEY),
            )
        conn.commit()
    return current


def restore_sync_commit_generation_after_reset(store: SQLiteStore, generation: int) -> int:
    """Restore the monotonic publication generation after replacing SQLite."""

    generation = max(0, int(generation))
    if generation % 2:
        raise ValueError('restored sync generation must be even')
    ensure_change_journal(store)
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO schema_meta(key,value) VALUES(?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (SYNC_COMMIT_GENERATION_KEY, str(generation)),
        )
        conn.commit()
    return generation


def claim_sync_commit_generation(store: SQLiteStore, expected: int) -> int | None:
    """CAS a stable generation to its odd in-progress successor."""

    expected = max(0, int(expected))
    if expected % 2:
        return None
    ensure_change_journal(store)
    claimed = expected + 1
    with store.connect() as conn:
        conn.execute(
            'INSERT OR IGNORE INTO schema_meta(key,value) VALUES(?,?)',
            (SYNC_COMMIT_GENERATION_KEY, '0'),
        )
        cursor = conn.execute(
            'UPDATE schema_meta SET value=? WHERE key=? AND value=?',
            (str(claimed), SYNC_COMMIT_GENERATION_KEY, str(expected)),
        )
        conn.commit()
    return claimed if cursor.rowcount == 1 else None


def complete_sync_commit_generation(store: SQLiteStore, claimed: int) -> int:
    """Publish an odd claimed generation as complete."""

    claimed = max(0, int(claimed))
    if not claimed % 2:
        raise ValueError('claimed sync generation must be odd')
    completed = claimed + 1
    ensure_change_journal(store)
    with store.connect() as conn:
        cursor = conn.execute(
            'UPDATE schema_meta SET value=? WHERE key=? AND value=?',
            (str(completed), SYNC_COMMIT_GENERATION_KEY, str(claimed)),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise RuntimeError('sync commit generation claim was lost')
        conn.commit()
    return completed


def read_waterlines(store: SQLiteStore) -> dict[tuple[str, str, str], dict[str, Any]]:
    ensure_change_journal(store)
    with store.connect() as conn:
        rows = list(conn.execute('SELECT * FROM sync_state'))
    return {
        (row['account_id'], row['conversation_id'], row['shard_id']): {
            'max_local_id': int(row['max_local_id']),
            'max_create_time': int(row['max_create_time']),
            'max_timestamp': row['max_timestamp'] or '',
        }
        for row in rows
    }


def write_waterlines(store: SQLiteStore, updates: dict[tuple[str, str, str], dict[str, Any]]) -> int:
    if not updates:
        return 0
    ensure_change_journal(store)
    now = _now()
    rows = [
        (
            account_id,
            conversation_id,
            shard_id,
            int(state.get('max_local_id') or -1),
            int(state.get('max_create_time') or -1),
            str(state.get('max_timestamp') or ''),
            now,
        )
        for (account_id, conversation_id, shard_id), state in updates.items()
    ]
    with store.connect() as conn:
        conn.executemany(
            """INSERT INTO sync_state(account_id,conversation_id,shard_id,max_local_id,max_create_time,max_timestamp,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(account_id,conversation_id,shard_id) DO UPDATE SET
               max_local_id=max(sync_state.max_local_id, excluded.max_local_id),
               max_create_time=max(sync_state.max_create_time, excluded.max_create_time),
               max_timestamp=max(sync_state.max_timestamp, excluded.max_timestamp),
               updated_at=excluded.updated_at""",
            rows,
        )
        conn.commit()
    return len(updates)


def record_dirty_citations(store: SQLiteStore, refs: list[dict[str, str]]) -> int:
    if not refs:
        return 0
    ensure_change_journal(store)
    now = _now()
    rows = [
        (
            str(ref.get('citation') or ''),
            str(ref.get('account_id') or ''),
            str(ref.get('conversation_id') or ''),
            str(ref.get('source_type') or ''),
            now,
        )
        for ref in refs
        if ref.get('citation')
    ]
    if not rows:
        return 0
    with store.connect() as conn:
        conn.executemany(
            """INSERT INTO sync_dirty_citations(citation,account_id,conversation_id,source_type,updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(citation) DO UPDATE SET
               account_id=excluded.account_id,
               conversation_id=excluded.conversation_id,
               source_type=excluded.source_type,
               updated_at=excluded.updated_at""",
            rows,
        )
        conn.commit()
    return len({row[0] for row in rows})


def read_aux_fingerprints(store: SQLiteStore) -> dict[str, str]:
    ensure_change_journal(store)
    with store.connect() as conn:
        return {
            str(row['source_key']): str(row['fingerprint'])
            for row in conn.execute('SELECT source_key,fingerprint FROM sync_aux_state')
        }


def write_aux_fingerprints(store: SQLiteStore, updates: dict[str, str]) -> int:
    if not updates:
        return 0
    ensure_change_journal(store)
    now = _now()
    with store.connect() as conn:
        conn.executemany(
            """INSERT INTO sync_aux_state(source_key,fingerprint,updated_at)
               VALUES(?,?,?)
               ON CONFLICT(source_key) DO UPDATE SET
               fingerprint=excluded.fingerprint,
               updated_at=excluded.updated_at""",
            [(key, value, now) for key, value in sorted(updates.items())],
        )
        conn.commit()
    return len(updates)


def dirty_citation_count(store: SQLiteStore) -> int:
    ensure_change_journal(store)
    with store.connect() as conn:
        return int(conn.execute('SELECT COUNT(*) FROM sync_dirty_citations').fetchone()[0])


def read_dirty_citations(store: SQLiteStore, *, limit: int | None = None) -> list[str]:
    ensure_change_journal(store)
    with store.connect() as conn:
        # ``citation`` is the table primary key. Ordering by ``updated_at``
        # forced a full-table temp sort before every bounded 512-row batch on a
        # large backlog; deleting processed keys makes primary-key order fair and
        # naturally advances without a second queue index.
        sql = 'SELECT citation FROM sync_dirty_citations ORDER BY citation'
        params: list[Any] = []
        if limit is not None:
            sql += ' LIMIT ?'
            params.append(max(0, int(limit)))
        return [
            str(row['citation'])
            for row in conn.execute(sql, params)
            if row['citation']
        ]


def read_dirty_citation_batch(store: SQLiteStore, *, limit: int) -> list[tuple[str, str]]:
    """Return a deterministic dirty batch with CAS tokens, without evidence content."""

    ensure_change_journal(store)
    with store.connect() as conn:
        return [
            (str(row['citation']), str(row['updated_at']))
            for row in conn.execute(
                'SELECT citation,updated_at FROM sync_dirty_citations ORDER BY citation LIMIT ?',
                (max(0, int(limit)),),
            )
            if row['citation']
        ]


def clear_dirty_citations(store: SQLiteStore, citations: list[str]) -> int:
    if not citations:
        return 0
    ensure_change_journal(store)
    removed = 0
    unique = list(dict.fromkeys(str(citation) for citation in citations if citation))
    with store.connect() as conn:
        for start in range(0, len(unique), 500):
            batch = unique[start:start + 500]
            placeholders = ','.join('?' for _ in batch)
            cursor = conn.execute(
                f'DELETE FROM sync_dirty_citations WHERE citation IN ({placeholders})',
                batch,
            )
            removed += max(cursor.rowcount, 0)
        conn.commit()
    return removed


def clear_dirty_citation_batch(store: SQLiteStore, batch: list[tuple[str, str]]) -> int:
    """Clear only rows unchanged since the corresponding bounded read."""

    if not batch:
        return 0
    ensure_change_journal(store)
    unique = list(dict.fromkeys(
        (str(citation), str(updated_at))
        for citation, updated_at in batch
        if citation
    ))
    removed = 0
    with store.connect() as conn:
        for start in range(0, len(unique), 400):
            rows = unique[start:start + 400]
            clauses = ' OR '.join('(citation=? AND updated_at=?)' for _ in rows)
            params = [value for row in rows for value in row]
            cursor = conn.execute(f'DELETE FROM sync_dirty_citations WHERE {clauses}', params)
            removed += max(cursor.rowcount, 0)
        conn.commit()
    return removed


def clear_all_dirty_citations(store: SQLiteStore) -> int:
    ensure_change_journal(store)
    with store.connect() as conn:
        cursor = conn.execute('DELETE FROM sync_dirty_citations')
        removed = max(cursor.rowcount, 0)
        conn.commit()
    return removed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


__all__ = [
    'claim_sync_commit_generation',
    'clear_dirty_citations',
    'clear_dirty_citation_batch',
    'clear_all_dirty_citations',
    'complete_sync_commit_generation',
    'dirty_citation_count',
    'ensure_change_journal',
    'ensure_sync_state',
    'MAX_INLINE_DIRTY_CITATIONS',
    'read_aux_fingerprints',
    'read_dirty_citations',
    'read_dirty_citation_batch',
    'read_sync_commit_generation',
    'read_waterlines',
    'record_dirty_citations',
    'recover_sync_commit_generation',
    'SYNC_COMMIT_GENERATION_KEY',
    'write_aux_fingerprints',
    'write_waterlines',
]
