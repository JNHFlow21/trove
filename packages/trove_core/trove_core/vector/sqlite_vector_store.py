from __future__ import annotations
import json
import math
import hashlib
from sqlite3 import Row

from trove_core.bounds import BoundedLimit, RERANK_CANDIDATES
from trove_core.store.sqlite_store import SQLiteStore, vector_document_text


VECTOR_TEXT_VERSION = 2
SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT = 10000


class SQLiteVectorUnavailable(RuntimeError):
    vector_state = 'unavailable_fallback'
    reason_code = 'sqlite_vector_diagnostic_limit_exceeded'



def _embedding_text(row) -> str:
    try:
        if hasattr(row, 'keys') and 'vector_text' in row.keys():
            return str(row['vector_text'] or '')
    except Exception:
        pass
    return vector_document_text(row)

class SQLiteVectorStore:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def initialize(self) -> None:
        self.store.initialize()

    def index_all_messages(self, provider, *, batch_size: int = 500, max_messages: int | None = None, citations=None) -> int:
        self.initialize()
        indexed = 0
        with self.store.connect() as conn:
            for row in self.store.iter_vector_documents(batch_size=batch_size, citations=citations):
                if max_messages is not None and indexed >= max_messages:
                    break
                text = _embedding_text(row)
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                existing = conn.execute('SELECT content_hash FROM vector_entries WHERE citation=?', (row['citation'],)).fetchone()
                if existing is not None and existing['content_hash'] == digest:
                    continue
                vector = provider.embed(text)
                conn.execute(
                    """INSERT OR REPLACE INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)""",
                    (row['citation'], provider.name, len(vector), json.dumps(vector), digest),
                )
                indexed += 1
            conn.commit()
        return indexed

    def search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10, provider=None) -> list[Row]:
        if provider is None:
            raise RuntimeError('SQLiteVectorStore.search requires an embedding provider')
        limit = BoundedLimit(limit, field='limit', spec=RERANK_CANDIDATES)
        self.initialize()
        scored: list[tuple[float, Row]] = []
        if not filters:
            # A bounded OFFSET probe rejects even orphan/corrupt vector rows
            # before query embedding. Scoped searches instead apply their SQL
            # filters first and enforce the same cap on matching evidence.
            with self.store.connect() as conn:
                oversized = conn.execute(
                    'SELECT 1 FROM vector_entries LIMIT 1 OFFSET ?',
                    (SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT,),
                ).fetchone()
            if oversized is not None:
                raise SQLiteVectorUnavailable(
                    f'SQLite vector search is a diagnostic fallback capped at '
                    f'{SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT} entries. '
                    'Use or rebuild ZVEC for production semantic search.'
                )
        entries = self.store.vector_entries_for_search(
            filters,
            limit=SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT + 1,
        )
        if len(entries) > SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT:
            raise SQLiteVectorUnavailable(
                f'SQLite vector search is a diagnostic fallback capped at '
                f'{SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT} matching entries. '
                'Use or rebuild ZVEC for production semantic search.'
            )
        if not entries:
            return []
        qv = provider.embed(query)
        for entry in entries:
            score = cosine(qv, json.loads(entry['vector_json']))
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1]['timestamp']))
        return [row for _, row in scored[:limit]]

class BoundSQLiteVectorSearch:
    def __init__(self, store: SQLiteVectorStore, provider):
        self.store = store
        self.provider = provider

    def search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10):
        return self.store.search(query, filters=filters, limit=limit, provider=self.provider)

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na * nb)
