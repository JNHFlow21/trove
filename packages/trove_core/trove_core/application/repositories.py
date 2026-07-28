"""Application-owned repository slices and unit-of-work boundary.

Protocol adapters (CLI, HTTP, MCP and agent tools) must depend on this module,
not on SQLite details.  The existing ``SQLiteStore`` remains the compatibility
facade while storage is split into narrow capabilities that can evolve
independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Protocol, Self, runtime_checkable

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig


Row = Mapping[str, Any]


@runtime_checkable
class MessageRepository(Protocol):
    def list_conversations(self, *, limit: int) -> list[dict[str, Any]]: ...

    def conversation_candidates(self, contact: str, *, limit: int) -> list[dict[str, Any]]: ...

    def context_window(self, citation: str, *, before: int, after: int) -> list[Row]: ...

    def conversation_messages(
        self,
        conversation_id: str,
        *,
        account_id: str | None,
        limit: int,
    ) -> list[Row]: ...


@runtime_checkable
class EvidenceRepository(Protocol):
    def by_citation(self, citation: str) -> Row | None: ...

    def citation_matches(self, citation: str, filters: Mapping[str, str]) -> bool: ...


@runtime_checkable
class MediaRepository(Protocol):
    def hints_for_citations(self, citations: list[str]) -> dict[str, dict[str, Any]]: ...

    def list_files(
        self,
        *,
        account_id: str | None,
        contact: str | None,
        conversation_id: str | None,
        file_name: str | None,
        media_types: list[str] | str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> dict[str, Any]: ...


@runtime_checkable
class SearchRepository(Protocol):
    def list_contacts(self, *, limit: int) -> list[dict[str, Any]]: ...

    def list_moments(self, *, limit: int) -> list[dict[str, Any]]: ...

    def list_favorites(self, *, limit: int) -> list[dict[str, Any]]: ...


@runtime_checkable
class VectorRepository(Protocol):
    def entries(self, filters: Mapping[str, str], *, limit: int) -> list[Row]: ...


@runtime_checkable
class UnitOfWork(Protocol):
    messages: MessageRepository
    evidence: EvidenceRepository
    media: MediaRepository
    search: SearchRepository
    vectors: VectorRepository

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...


@dataclass(frozen=True)
class SQLiteMessageRepository:
    store: SQLiteStore

    def list_conversations(self, *, limit: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.store.list_conversations(limit=limit)]

    def conversation_candidates(self, contact: str, *, limit: int) -> list[dict[str, Any]]:
        needle = str(contact or '').strip()
        if not needle or not self.store.path.exists():
            return []
        like = f'%{needle}%'
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        with self.store.connect() as conn:
            rows = conn.execute(
                """SELECT account_id, conversation_id, title, type,
                          'conversation' AS match_source
                   FROM conversations
                   WHERE title LIKE ? OR conversation_id=?
                   ORDER BY CASE WHEN title=? OR conversation_id=? THEN 0 ELSE 1 END,
                            title, conversation_id
                   LIMIT ?""",
                (like, needle, needle, needle, limit),
            )
            for row in rows:
                candidates[(row['account_id'], row['conversation_id'])] = dict(row)

            if len(candidates) < limit:
                rows = conn.execute(
                    """SELECT account_id, conversation_id,
                              conversation_title AS title,
                              conversation_type AS type,
                              'message_sender' AS match_source,
                              MAX(timestamp) AS last_timestamp
                       FROM messages
                       WHERE sender_name LIKE ? OR sender_id=?
                       GROUP BY account_id, conversation_id,
                                conversation_title, conversation_type
                       ORDER BY last_timestamp DESC
                       LIMIT ?""",
                    (like, needle, limit),
                )
                for row in rows:
                    key = (row['account_id'], row['conversation_id'])
                    candidates.setdefault(key, {
                        'account_id': row['account_id'],
                        'conversation_id': row['conversation_id'],
                        'title': row['title'],
                        'type': row['type'],
                        'match_source': row['match_source'],
                    })

        needle_lower = needle.lower()
        result = list(candidates.values())
        result.sort(key=lambda candidate: (
            0 if (
                str(candidate.get('conversation_id') or '').lower() == needle_lower
                or str(candidate.get('title') or '').lower() == needle_lower
            ) else 1,
            str(candidate.get('title') or ''),
            str(candidate.get('conversation_id') or ''),
        ))
        return result[:limit]

    def context_window(self, citation: str, *, before: int, after: int) -> list[Row]:
        return list(self.store.context_window(citation, before=before, after=after))

    def conversation_messages(
        self,
        conversation_id: str,
        *,
        account_id: str | None,
        limit: int,
    ) -> list[Row]:
        if not self.store.path.exists():
            return []
        where = ['conversation_id=?']
        params: list[Any] = [conversation_id]
        if account_id:
            where.append('account_id=?')
            params.append(account_id)
        with self.store.connect() as conn:
            return list(conn.execute(
                f"""SELECT * FROM (
                       SELECT * FROM messages
                       WHERE {' AND '.join(where)}
                       ORDER BY timestamp DESC, shard_id DESC, local_id DESC
                       LIMIT ?
                   ) ORDER BY timestamp ASC, shard_id ASC, local_id ASC""",
                (*params, limit),
            ))


@dataclass(frozen=True)
class SQLiteEvidenceRepository:
    store: SQLiteStore

    def by_citation(self, citation: str) -> Row | None:
        return self.store.evidence_by_citation(citation)

    def citation_matches(self, citation: str, filters: Mapping[str, str]) -> bool:
        row = self.by_citation(citation)
        if row is None:
            return False
        for key, value in filters.items():
            actual = _row_value(row, key)
            if key == 'sender':
                sender_name = str(_row_value(row, 'sender_name') or '')
                sender_id = str(_row_value(row, 'sender_id') or '')
                if value not in sender_name and value != sender_id:
                    return False
            elif key in {'source_family', 'scope_type'}:
                if value not in {'all', str(_row_value(row, 'source_type') or '')}:
                    return False
            elif key == 'since':
                if str(_row_value(row, 'timestamp') or '') < value:
                    return False
            elif key == 'until':
                if str(_row_value(row, 'timestamp') or '') > value:
                    return False
            elif str(actual or '') != value:
                return False
        return True


@dataclass(frozen=True)
class SQLiteMediaRepository:
    store: SQLiteStore

    def hints_for_citations(self, citations: list[str]) -> dict[str, dict[str, Any]]:
        return self.store.media_hints_for_citations(citations)

    def list_files(
        self,
        *,
        account_id: str | None,
        contact: str | None,
        conversation_id: str | None,
        file_name: str | None,
        media_types: list[str] | str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> dict[str, Any]:
        from trove_core.wechat.files import list_conversation_files

        scan_limit = 500 if account_id else limit
        payload = list_conversation_files(
            self.store,
            contact=contact,
            conversation_id=conversation_id,
            file_name=file_name,
            media_types=media_types,
            since=since,
            until=until,
            limit=scan_limit,
        )
        if not account_id:
            return payload
        files = [
            item for item in payload.get('files', [])
            if str((item.get('conversation') or {}).get('account_id') or '') == account_id
        ]
        return {
            **payload,
            'files': files[:limit],
            'count': min(len(files), limit),
            'total_candidates': len(files),
        }


@dataclass(frozen=True)
class SQLiteSearchRepository:
    store: SQLiteStore

    def list_contacts(self, *, limit: int) -> list[dict[str, Any]]:
        return self.store.list_contacts(limit=limit)

    def list_moments(self, *, limit: int) -> list[dict[str, Any]]:
        return self.store.list_moments(limit=limit)

    def list_favorites(self, *, limit: int) -> list[dict[str, Any]]:
        return self.store.list_favorites(limit=limit)


@dataclass(frozen=True)
class SQLiteVectorRepository:
    store: SQLiteStore

    def entries(self, filters: Mapping[str, str], *, limit: int) -> list[Row]:
        return list(self.store.vector_entries_for_search(dict(filters), limit=limit))


class SQLiteUnitOfWork:
    """Deterministic SQLite application lifetime with vertical repositories."""

    def __init__(
        self,
        config: VaultConfig | str | Path,
        *,
        readonly: bool = True,
        store: SQLiteStore | None = None,
        max_connections: int = 64,
        prepared_statement_cache_size: int = 128,
    ) -> None:
        if isinstance(config, VaultConfig):
            self.config = config
        else:
            self.config = VaultConfig.resolve(str(config))
        self.readonly = readonly
        self._owns_store = store is None
        self.store = store or SQLiteStore(
            self.config.paths.sqlite_path,
            readonly=readonly,
            max_connections=max_connections,
            prepared_statement_cache_size=prepared_statement_cache_size,
        )
        self.messages = SQLiteMessageRepository(self.store)
        self.evidence = SQLiteEvidenceRepository(self.store)
        self.media = SQLiteMediaRepository(self.store)
        self.search = SQLiteSearchRepository(self.store)
        self.vectors = SQLiteVectorRepository(self.store)
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError('unit of work cannot be entered twice')
        self._entered = True
        if self.store.path.exists() or not self.readonly:
            self.store.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._entered = False
        if self._owns_store:
            self.store.close_all()
        return False


@dataclass(frozen=True)
class RepositoryFacade:
    """Compatibility view for callers migrating away from ``SQLiteStore``.

    New code should request a narrow repository.  The facade intentionally
    exposes repositories rather than forwarding arbitrary store attributes.
    """

    messages: MessageRepository
    evidence: EvidenceRepository
    media: MediaRepository
    search: SearchRepository
    vectors: VectorRepository

    @classmethod
    def from_uow(cls, uow: UnitOfWork) -> 'RepositoryFacade':
        return cls(uow.messages, uow.evidence, uow.media, uow.search, uow.vectors)


def _row_value(row: Row, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


__all__ = [
    'EvidenceRepository',
    'MediaRepository',
    'MessageRepository',
    'RepositoryFacade',
    'SearchRepository',
    'SQLiteEvidenceRepository',
    'SQLiteMediaRepository',
    'SQLiteMessageRepository',
    'SQLiteSearchRepository',
    'SQLiteUnitOfWork',
    'SQLiteVectorRepository',
    'UnitOfWork',
    'VectorRepository',
]
