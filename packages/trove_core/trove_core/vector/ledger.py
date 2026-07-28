from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from trove_core.store.sqlite_store import SQLiteStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class VectorGeneration:
    backend: str
    generation_id: str
    status: str
    vector_text_version: int
    embedding_provider: str
    embedding_model: str
    dimensions: int
    expected_count: int | None
    indexed_count: int
    revision: int


class VectorIndexLedger:
    """Indexed, authoritative citation/hash ledger for one vector backend."""

    def __init__(self, store: SQLiteStore, *, backend: str = 'zvec'):
        self.store = store
        self.backend = str(backend)

    @staticmethod
    def _generation(row: Any | None) -> VectorGeneration | None:
        if row is None:
            return None
        return VectorGeneration(
            backend=str(row['backend']),
            generation_id=str(row['generation_id']),
            status=str(row['status']),
            vector_text_version=int(row['vector_text_version']),
            embedding_provider=str(row['embedding_provider'] or ''),
            embedding_model=str(row['embedding_model'] or ''),
            dimensions=int(row['dimensions'] or 0),
            expected_count=int(row['expected_count']) if row['expected_count'] is not None else None,
            indexed_count=int(row['indexed_count'] or 0),
            revision=max(1, int(row['revision'] or 1)),
        )

    def generation(self, generation_id: str) -> VectorGeneration | None:
        if not self.store.path.exists():
            return None
        with self.store.connect() as conn:
            if not self.store._table_exists(conn, 'vector_index_generations'):
                return None
            return self._generation(conn.execute(
                'SELECT * FROM vector_index_generations WHERE backend=? AND generation_id=?',
                (self.backend, str(generation_id)),
            ).fetchone())

    def active_generation(self) -> VectorGeneration | None:
        if not self.store.path.exists():
            return None
        with self.store.connect() as conn:
            if not self.store._table_exists(conn, 'vector_index_generations'):
                return None
            return self._generation(conn.execute(
                "SELECT * FROM vector_index_generations WHERE backend=? AND status='active' LIMIT 1",
                (self.backend,),
            ).fetchone())

    def begin_generation(
        self,
        generation_id: str,
        *,
        vector_text_version: int,
        embedding_provider: str,
        embedding_model: str,
        dimensions: int,
        expected_count: int | None,
    ) -> VectorGeneration:
        self.store.initialize()
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO vector_index_generations(
                       backend,generation_id,status,vector_text_version,embedding_provider,
                       embedding_model,dimensions,expected_count,indexed_count,created_at,activated_at)
                   VALUES(?,?,'building',?,?,?,?,?,0,?,NULL)
                   ON CONFLICT(backend,generation_id) DO UPDATE SET
                     vector_text_version=excluded.vector_text_version,
                     embedding_provider=excluded.embedding_provider,
                     embedding_model=excluded.embedding_model,
                     dimensions=excluded.dimensions,
                     expected_count=excluded.expected_count
                   WHERE vector_index_generations.status='building'""",
                (
                    self.backend, str(generation_id), int(vector_text_version),
                    str(embedding_provider), str(embedding_model), int(dimensions),
                    expected_count, _now(),
                ),
            )
            conn.commit()
            row = conn.execute(
                'SELECT * FROM vector_index_generations WHERE backend=? AND generation_id=?',
                (self.backend, str(generation_id)),
            ).fetchone()
        generation = self._generation(row)
        if generation is None:
            raise RuntimeError('vector generation could not be created')
        return generation

    def apply_delta(
        self,
        generation_id: str,
        *,
        upserts: Iterable[tuple[str, str]] = (),
        deletes: Iterable[str] = (),
        expected_count: int | None = None,
    ) -> dict[str, int]:
        """Apply exact citation deltas in one transaction without a full COUNT."""
        upsert_rows = list({str(citation): str(content_hash) for citation, content_hash in upserts if citation}.items())
        upsert_citations = {row[0] for row in upsert_rows}
        delete_rows = [
            citation
            for citation in dict.fromkeys(str(value) for value in deletes if value)
            if citation not in upsert_citations
        ]
        self.store.initialize()
        with self.store.connect() as conn:
            generation_row = conn.execute(
                "SELECT status FROM vector_index_generations WHERE backend=? AND generation_id=?",
                (self.backend, generation_id),
            ).fetchone()
            if generation_row is None or str(generation_row['status']) == 'retired':
                raise RuntimeError('vector generation does not accept deltas')
            conn.execute(
                """CREATE TEMP TABLE IF NOT EXISTS _trove_vector_ledger_delta(
                       citation TEXT PRIMARY KEY,
                       content_hash TEXT,
                       operation TEXT NOT NULL
                   ) WITHOUT ROWID"""
            )
            conn.execute('DELETE FROM _trove_vector_ledger_delta')
            if upsert_rows:
                conn.executemany(
                    "INSERT INTO _trove_vector_ledger_delta(citation,content_hash,operation) VALUES(?,?,'upsert')",
                    upsert_rows,
                )
            if delete_rows:
                conn.executemany(
                    "INSERT INTO _trove_vector_ledger_delta(citation,content_hash,operation) VALUES(?,NULL,'delete')",
                    [(citation,) for citation in delete_rows],
                )
            inserted = int(conn.execute(
                """SELECT COUNT(*) FROM _trove_vector_ledger_delta d
                   LEFT JOIN vector_index_ledger l
                     ON l.backend=? AND l.generation_id=? AND l.citation=d.citation
                   WHERE d.operation='upsert' AND l.citation IS NULL""",
                (self.backend, generation_id),
            ).fetchone()[0])
            removed = int(conn.execute(
                """SELECT COUNT(*) FROM _trove_vector_ledger_delta d
                   JOIN vector_index_ledger l
                     ON l.backend=? AND l.generation_id=? AND l.citation=d.citation
                   WHERE d.operation='delete'""",
                (self.backend, generation_id),
            ).fetchone()[0])
            cursor = conn.execute(
                """INSERT INTO vector_index_ledger(backend,generation_id,citation,content_hash,state,updated_at)
                   SELECT ?,?,citation,content_hash,'indexed',?
                   FROM _trove_vector_ledger_delta WHERE operation='upsert'
                   ON CONFLICT(backend,generation_id,citation) DO UPDATE SET
                     content_hash=excluded.content_hash,state='indexed',updated_at=excluded.updated_at
                   WHERE vector_index_ledger.content_hash IS NOT excluded.content_hash
                      OR vector_index_ledger.state<>'indexed'""",
                (self.backend, generation_id, _now()),
            )
            upserted = max(cursor.rowcount, 0)
            cursor = conn.execute(
                """DELETE FROM vector_index_ledger
                   WHERE backend=? AND generation_id=?
                     AND citation IN (
                       SELECT citation FROM _trove_vector_ledger_delta WHERE operation='delete'
                     )""",
                (self.backend, generation_id),
            )
            deleted = max(cursor.rowcount, 0)
            revision_incremented = int(bool(upserted or deleted))
            expected_sql = 'expected_count' if expected_count is None else '?'
            params: list[Any] = [inserted - removed, revision_incremented]
            if expected_count is not None:
                params.append(int(expected_count))
            params.extend([self.backend, generation_id])
            conn.execute(
                f"""UPDATE vector_index_generations
                    SET indexed_count=max(0,indexed_count+?),revision=revision+?,expected_count={expected_sql}
                    WHERE backend=? AND generation_id=?""",
                params,
            )
            conn.execute('DELETE FROM _trove_vector_ledger_delta')
            if upserted or deleted or expected_count is not None:
                conn.commit()
                commits = 1
            else:
                conn.rollback()
                commits = 0
        return {
            'candidate_rows': len(upsert_rows) + len(delete_rows),
            'upserted': upserted,
            'deleted': deleted,
            'count_delta': inserted - removed,
            'revision_incremented': revision_incremented,
            'commits': commits,
            'sql_statements': 10,
        }

    def hashes(self, generation_id: str, citations: Iterable[str]) -> dict[str, str]:
        unique = list(dict.fromkeys(str(value) for value in citations if value))
        if not unique or not self.store.path.exists():
            return {}
        out: dict[str, str] = {}
        with self.store.connect() as conn:
            for start in range(0, len(unique), 500):
                batch = unique[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                for row in conn.execute(
                    f"""SELECT citation,content_hash FROM vector_index_ledger
                        WHERE backend=? AND generation_id=? AND state='indexed'
                          AND citation IN ({placeholders})""",
                    [self.backend, generation_id, *batch],
                ):
                    out[str(row['citation'])] = str(row['content_hash'])
        return out

    def citations_for_dirty(self, generation_id: str, dirty_citations: Iterable[str]) -> list[str]:
        dirty = list(dict.fromkeys(str(value) for value in dirty_citations if value))
        if not dirty or not self.store.path.exists():
            return []
        out: set[str] = set()
        with self.store.connect() as conn:
            # Dirty batches are bounded, while an active ledger can contain
            # hundreds of thousands of rows.  Use two indexable probes per
            # parent (exact + child-chunk range) instead of parsing every
            # active citation in Python.
            for citation in dirty:
                exact = conn.execute(
                    """SELECT citation FROM vector_index_ledger
                       WHERE backend=? AND generation_id=? AND state='indexed'
                         AND citation=?""",
                    (self.backend, generation_id, citation),
                ).fetchone()
                if exact is not None:
                    out.add(str(exact['citation']))
                prefix = f'{citation}#chunk-'
                upper = prefix + '\U0010ffff'
                for row in conn.execute(
                    """SELECT citation FROM vector_index_ledger
                       WHERE backend=? AND generation_id=? AND state='indexed'
                         AND citation>=? AND citation<?
                       ORDER BY citation""",
                    (self.backend, generation_id, prefix, upper),
                ):
                    out.add(str(row['citation']))
        return sorted(out)

    def iter_entries(
        self,
        generation_id: str,
        *,
        batch_size: int = 1000,
    ) -> Iterator[tuple[str, str]]:
        """Stream indexed citation/hash pairs in deterministic key order."""

        after = ''
        while True:
            with self.store.connect() as conn:
                rows = list(conn.execute(
                    """SELECT citation,content_hash FROM vector_index_ledger
                       WHERE backend=? AND generation_id=? AND state='indexed' AND citation>?
                       ORDER BY citation LIMIT ?""",
                    (self.backend, generation_id, after, max(1, int(batch_size))),
                ))
            if not rows:
                return
            for row in rows:
                after = str(row['citation'])
                yield after, str(row['content_hash'])

    def iter_citations(self, generation_id: str, *, batch_size: int = 1000) -> Iterator[str]:
        for citation, _content_hash in self.iter_entries(generation_id, batch_size=batch_size):
            yield citation

    def begin_full_scan(self) -> None:
        self.store.initialize()
        with self.store.connect() as conn:
            conn.execute(
                'CREATE TEMP TABLE IF NOT EXISTS _trove_vector_seen(citation TEXT PRIMARY KEY) WITHOUT ROWID'
            )
            conn.execute('DELETE FROM _trove_vector_seen')

    def record_seen(self, citations: Iterable[str]) -> None:
        rows = [(str(value),) for value in citations if value]
        if not rows:
            return
        with self.store.connect() as conn:
            conn.executemany('INSERT OR IGNORE INTO _trove_vector_seen(citation) VALUES(?)', rows)

    def stale_after_full_scan(self, generation_id: str, *, limit: int = 1000) -> Iterator[list[str]]:
        after = ''
        while True:
            with self.store.connect() as conn:
                rows = list(conn.execute(
                    """SELECT l.citation FROM vector_index_ledger l
                       LEFT JOIN _trove_vector_seen s ON s.citation=l.citation
                       WHERE l.backend=? AND l.generation_id=? AND l.state='indexed'
                         AND s.citation IS NULL AND l.citation>?
                       ORDER BY l.citation LIMIT ?""",
                    (self.backend, generation_id, after, max(1, int(limit))),
                ))
            if not rows:
                return
            batch = [str(row['citation']) for row in rows]
            after = batch[-1]
            yield batch

    def end_full_scan(self) -> None:
        with self.store.connect() as conn:
            conn.execute('DELETE FROM _trove_vector_seen')

    def mark_ready(self, generation_id: str, *, expected_count: int | None) -> None:
        self.store.initialize()
        with self.store.connect() as conn:
            row = conn.execute(
                """SELECT status,indexed_count,expected_count FROM vector_index_generations
                   WHERE backend=? AND generation_id=?""",
                (self.backend, generation_id),
            ).fetchone()
            if row is None or str(row['status']) not in {'building', 'ready'}:
                raise RuntimeError('vector generation cannot be marked ready')
            target = expected_count if expected_count is not None else row['expected_count']
            if target is not None and int(row['indexed_count'] or 0) < int(target):
                raise RuntimeError('vector generation is incomplete')
            cursor = conn.execute(
                """UPDATE vector_index_generations SET status='ready',expected_count=?
                   WHERE backend=? AND generation_id=? AND status IN ('building','ready')""",
                (expected_count, self.backend, generation_id),
            )
            if max(cursor.rowcount, 0) == 0:
                raise RuntimeError('vector generation cannot be marked ready')
            conn.commit()

    def activate(self, generation_id: str) -> None:
        self.store.initialize()
        now = _now()
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM vector_index_generations WHERE backend=? AND generation_id=?",
                (self.backend, generation_id),
            ).fetchone()
            if row is None or str(row['status']) not in {'ready', 'active'}:
                raise RuntimeError('vector generation is not ready for activation')
            conn.execute(
                "UPDATE vector_index_generations SET status='retired' WHERE backend=? AND status='active' AND generation_id<>?",
                (self.backend, generation_id),
            )
            cursor = conn.execute(
                """UPDATE vector_index_generations SET status='active',activated_at=?
                   WHERE backend=? AND generation_id=? AND status IN ('ready','active')""",
                (now, self.backend, generation_id),
            )
            if max(cursor.rowcount, 0) == 0:
                raise RuntimeError('vector generation is not ready for activation')
            conn.commit()

    def discard(self, generation_id: str) -> int:
        if not generation_id or not self.store.path.exists():
            return 0
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM vector_index_generations WHERE backend=? AND generation_id=?",
                (self.backend, generation_id),
            ).fetchone()
            if row is None or str(row['status']) == 'active':
                conn.rollback()
                return 0
            # Trove intentionally does not enable PRAGMA foreign_keys on its
            # long-lived stores. Never rely on ON DELETE CASCADE for cleanup.
            conn.execute(
                'DELETE FROM vector_index_ledger WHERE backend=? AND generation_id=?',
                (self.backend, generation_id),
            )
            cursor = conn.execute(
                "DELETE FROM vector_index_generations WHERE backend=? AND generation_id=? AND status<>'active'",
                (self.backend, generation_id),
            )
            removed = max(cursor.rowcount, 0)
            if removed:
                conn.commit()
            else:
                conn.rollback()
            return removed

    def prune_retired(self, *, keep: int = 0) -> int:
        """Delete retired generations and their ledger rows with FK checks off."""
        if not self.store.path.exists():
            return 0
        keep = max(0, int(keep))
        with self.store.connect() as conn:
            rows = list(conn.execute(
                """SELECT generation_id FROM vector_index_generations
                   WHERE backend=? AND status='retired'
                   ORDER BY COALESCE(activated_at,created_at) DESC,generation_id DESC
                   LIMIT -1 OFFSET ?""",
                (self.backend, keep),
            ))
            generation_ids = [str(row['generation_id']) for row in rows]
            for generation_id in generation_ids:
                conn.execute(
                    'DELETE FROM vector_index_ledger WHERE backend=? AND generation_id=?',
                    (self.backend, generation_id),
                )
            if generation_ids:
                conn.executemany(
                    "DELETE FROM vector_index_generations WHERE backend=? AND generation_id=? AND status='retired'",
                    [(self.backend, generation_id) for generation_id in generation_ids],
                )
                conn.commit()
            else:
                conn.rollback()
        return len(generation_ids)
