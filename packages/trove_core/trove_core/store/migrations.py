from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from typing import Callable, Any

from .schema import (
    EXPECTED_INDEX_COLUMNS,
    FTS_TOKENIZER_VERSION,
    FTS_TRIGGER_NAMES,
    PERSISTENT_SCHEMA,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    TABLES,
    TRIGRAM_FTS_SCHEMA,
    VECTOR_SOURCE_REVISION_KEY,
    VECTOR_SOURCE_TRIGGER_NAMES,
)


class SchemaMigrationRequired(RuntimeError):
    code = 'schema_migration_required'

    def __init__(
        self,
        current_version: int,
        required_version: int = SCHEMA_VERSION,
        *,
        missing_objects: list[str] | None = None,
        reason: str = 'schema version or manifest does not match this runtime',
    ):
        super().__init__(reason)
        self.current_version = int(current_version)
        self.required_version = int(required_version)
        self.missing_objects = sorted(set(missing_objects or []))

    def to_dict(self) -> dict[str, Any]:
        return {
            'error': {
                'code': self.code,
                'message': str(self),
                'current_version': self.current_version,
                'required_version': self.required_version,
                'missing_objects': self.missing_objects,
            }
        }


class SchemaVersionTooNew(SchemaMigrationRequired):
    code = 'schema_version_too_new'


class SchemaVersionMismatch(SchemaMigrationRequired):
    code = 'schema_version_mismatch'


class SchemaPreflightUnavailable(SchemaMigrationRequired):
    code = 'schema_preflight_unavailable'


def preflight_connection_versions(
    conn: sqlite3.Connection,
    *,
    required_version: int = SCHEMA_VERSION,
) -> tuple[int, int | None]:
    """Read both version authorities from one already-open coherent view."""

    current = current_schema_version(conn)
    metadata = metadata_schema_version(conn)
    newest = max(current, metadata or 0)
    if newest > required_version:
        raise SchemaVersionTooNew(newest, required_version, reason='database schema is newer than this runtime')
    if metadata is not None and metadata != current:
        raise SchemaVersionMismatch(
            current,
            required_version,
            reason='database user_version and schema_meta version disagree',
        )
    return current, metadata


def preflight_database_versions(
    path: str | Path,
    *,
    required_version: int = SCHEMA_VERSION,
) -> tuple[int, int | None]:
    """Coherently read main DB + live WAL without persistent PRAGMAs."""

    resolved = Path(path).expanduser().resolve()
    uri = resolved.as_uri() + '?mode=ro'
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            current, metadata = preflight_connection_versions(
                conn,
                required_version=required_version,
            )
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise SchemaPreflightUnavailable(
            0,
            required_version,
            reason='database schema version cannot be read safely',
        ) from exc
    return current, metadata


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection, Callable[[str], None] | None], None]


def _object_name(sql: str, kind: str) -> str | None:
    match = re.search(
        rf'CREATE\s+(?:UNIQUE\s+)?{kind}\s+(?:IF\s+NOT\s+EXISTS\s+)?["`\[]?([A-Za-z0-9_]+)',
        sql,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


EXPECTED_TABLES = frozenset(TABLES.values()) | {'schema_meta', 'message_fts', 'chunk_fts'}
EXPECTED_INDEXES = frozenset(
    name for sql in PERSISTENT_SCHEMA if (name := _object_name(sql, 'INDEX')) is not None
)
EXPECTED_TRIGGERS = frozenset([*FTS_TRIGGER_NAMES, *VECTOR_SOURCE_TRIGGER_NAMES])


def _normalize_sql(value: str | None) -> str:
    normalized = ' '.join(str(value or '').replace('"', '').replace('`', '').split()).lower()
    normalized = normalized.replace(' if not exists ', ' ')
    return re.sub(r'\s*([(),])\s*', r'\1', normalized)


def _check_clauses(sql: str | None) -> tuple[str, ...]:
    text = str(sql or '')
    lowered = text.lower()
    clauses: list[str] = []
    offset = 0
    while True:
        match = re.search(r'\bcheck\s*\(', lowered[offset:])
        if match is None:
            break
        open_at = offset + match.end() - 1
        depth = 0
        close_at = None
        for index in range(open_at, len(text)):
            if text[index] == '(':
                depth += 1
            elif text[index] == ')':
                depth -= 1
                if depth == 0:
                    close_at = index
                    break
        if close_at is None:
            clauses.append('malformed-check')
            break
        clauses.append(_normalize_sql(text[open_at + 1:close_at]))
        offset = close_at + 1
    return tuple(sorted(clauses))


def _manifest_shapes(conn: sqlite3.Connection) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    table_columns = {}
    table_keys = {}
    table_foreign_keys = {}
    for table in EXPECTED_TABLES:
        table_columns[table] = {
            str(row[1]): (str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
            for row in conn.execute(f'PRAGMA table_info("{table}")')
        }
        keys = []
        for row in conn.execute(f'PRAGMA index_list("{table}")'):
            origin = str(row[3])
            if origin == 'c':
                continue
            name = str(row[1])
            columns = tuple(str(item[2]) for item in conn.execute(f'PRAGMA index_info("{name}")'))
            keys.append((origin, int(row[2]), int(row[4]), columns))
        table_keys[table] = tuple(sorted(keys))
        table_foreign_keys[table] = tuple(sorted(
            (str(row[2]), str(row[3]), str(row[4]), str(row[5]), str(row[6]), str(row[7]))
            for row in conn.execute(f'PRAGMA foreign_key_list("{table}")')
        ))
    indexes = {}
    for name in EXPECTED_INDEXES:
        row = conn.execute(
            "SELECT tbl_name,sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()
        table = str(row[0])
        listing = next(item for item in conn.execute(f'PRAGMA index_list("{table}")') if str(item[1]) == name)
        indexes[name] = {
            'table': table,
            'columns': tuple(str(item[2]) for item in conn.execute(f'PRAGMA index_info("{name}")')),
            'unique': int(listing[2]),
            'partial': int(listing[4]),
            'sql': _normalize_sql(row[1]),
        }
    triggers = {
        name: _normalize_sql(conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
        ).fetchone()[0])
        for name in EXPECTED_TRIGGERS
    }
    virtual_tables = {
        name: _normalize_sql(conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()[0])
        for name in ('message_fts', 'chunk_fts')
    }
    table_sql = {}
    for table in EXPECTED_TABLES - {'message_fts', 'chunk_fts'}:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        normalized = _normalize_sql(row[0] if row else None)
        table_sql[table] = {
            'sql': normalized,
            'autoincrement': 'autoincrement' in normalized,
            'checks': _check_clauses(row[0] if row else None),
        }
    return table_columns, table_keys, table_foreign_keys, indexes, triggers, virtual_tables, table_sql


def _build_expected_shapes() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    conn = sqlite3.connect(':memory:')
    try:
        for sql in PERSISTENT_SCHEMA:
            conn.execute(sql)
        for sql in TRIGRAM_FTS_SCHEMA:
            conn.execute(sql)
        return _manifest_shapes(conn)
    finally:
        conn.close()


(
    EXPECTED_TABLE_SHAPES,
    EXPECTED_TABLE_KEY_SHAPES,
    EXPECTED_FOREIGN_KEY_SHAPES,
    EXPECTED_INDEX_SHAPES,
    EXPECTED_TRIGGER_SQL,
    EXPECTED_VIRTUAL_TABLE_SQL,
    EXPECTED_TABLE_SQL_SHAPES,
) = _build_expected_shapes()

ALTER_COMPATIBLE_TABLES = {
    'messages',
    'moment_interactions',
    'sync_dirty_citations',
    'vector_entries',
    'vector_index_generations',
    'approval_records',
    'profile_enrichment_runs',
    'profile_enrichment_tasks',
    'image_observations',
    'profile_snapshots',
}


def current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute('PRAGMA user_version').fetchone()
    return int(row[0]) if row else 0


def metadata_schema_version(conn: sqlite3.Connection) -> int | None:
    if not _table_exists(conn, 'schema_meta'):
        return None
    row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    try:
        return int(row[0]) if row is not None else None
    except (TypeError, ValueError):
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if _table_exists(conn, table) and column not in _column_names(conn, table):
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN {definition}')


_FORWARD_COMPATIBLE_COLUMNS = (
    ('messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'"),
    ('moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''"),
    ('sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''"),
    ('vector_entries', 'content_hash', 'content_hash TEXT'),
    ('approval_records', 'consumed_at', 'consumed_at TEXT'),
    ('approval_records', 'consumption_id', 'consumption_id TEXT'),
    ('vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1'),
    ('profile_enrichment_runs', 'execution_location', "execution_location TEXT NOT NULL DEFAULT 'local'"),
    ('profile_enrichment_runs', 'processor_identity', "processor_identity TEXT NOT NULL DEFAULT 'local-agent/default'"),
    ('profile_enrichment_runs', 'prompt_version', "prompt_version TEXT NOT NULL DEFAULT 'profile-enrichment/v1'"),
    ('profile_enrichment_runs', 'purpose', "purpose TEXT NOT NULL DEFAULT 'customer_profile_enrichment'"),
    ('profile_enrichment_runs', 'revoked_at', 'revoked_at TEXT'),
    ('profile_enrichment_tasks', 'approval_id', 'approval_id TEXT'),
    ('profile_enrichment_tasks', 'approval_scope_hash', 'approval_scope_hash TEXT'),
    ('profile_enrichment_tasks', 'delivery_token_hash', 'delivery_token_hash TEXT'),
    ('profile_enrichment_tasks', 'delivery_consumed_at', 'delivery_consumed_at TEXT'),
    ('profile_enrichment_tasks', 'prompt_version', "prompt_version TEXT NOT NULL DEFAULT 'profile-enrichment/v1'"),
    ('image_observations', 'content_sha256', "content_sha256 TEXT NOT NULL DEFAULT ''"),
    ('image_observations', 'model_id', "model_id TEXT NOT NULL DEFAULT ''"),
    ('image_observations', 'prompt_version', "prompt_version TEXT NOT NULL DEFAULT ''"),
    ('image_observations', 'updated_at', "updated_at TEXT NOT NULL DEFAULT ''"),
    ('profile_snapshots', 'content_hash', "content_hash TEXT NOT NULL DEFAULT ''"),
    ('profile_snapshots', 'source_revision', "source_revision TEXT NOT NULL DEFAULT 'legacy'"),
    ('profile_snapshots', 'run_id', 'run_id TEXT'),
    ('profile_snapshots', 'schema_version', "schema_version TEXT NOT NULL DEFAULT 'customer-profile/legacy'"),
    ('profile_snapshots', 'completeness_state', "completeness_state TEXT NOT NULL DEFAULT 'stale'"),
    ('profile_snapshots', 'evidence_citations_json', "evidence_citations_json TEXT NOT NULL DEFAULT '[]'"),
    ('profile_snapshots', 'enrichment_summary_json', "enrichment_summary_json TEXT NOT NULL DEFAULT '{}'"),
    ('profile_snapshots', 'gaps_json', "gaps_json TEXT NOT NULL DEFAULT '[]'"),
)


def _apply_current_persistent_schema(
    conn: sqlite3.Connection,
    *,
    skip_tokens: tuple[str, ...] = (),
) -> None:
    """Create current tables, add forward columns, then install safe indexes.

    Historical migrations run code from the current release. Existing legacy
    tables therefore need their later columns before current index DDL is
    evaluated. This two-phase helper keeps every old user_version retryable.
    """

    for sql in PERSISTENT_SCHEMA:
        if sql.lstrip().upper().startswith('CREATE TABLE'):
            conn.execute(sql)
    for table, column, definition in _FORWARD_COMPATIBLE_COLUMNS:
        _ensure_column(conn, table, column, definition)
    _repair_duplicate_profile_snapshot_versions(conn)
    for sql in PERSISTENT_SCHEMA:
        if not any(token in sql for token in skip_tokens):
            conn.execute(sql)


def _repair_duplicate_profile_snapshot_versions(conn: sqlite3.Connection) -> int:
    """Deterministically resequence only histories that violate version uniqueness."""

    if not _table_exists(conn, 'profile_snapshots'):
        return 0
    duplicate_entities = [str(row[0]) for row in conn.execute(
        """SELECT entity_id FROM profile_snapshots
             GROUP BY entity_id,version HAVING COUNT(*)>1"""
    )]
    if not duplicate_entities:
        return 0
    affected = sorted(set(duplicate_entities))
    conn.execute('DROP INDEX IF EXISTS idx_profile_snapshots_entity_version_unique')
    changed = 0
    for entity_id in affected:
        rows = list(conn.execute(
            """SELECT profile_id,version FROM profile_snapshots WHERE entity_id=?
                 ORDER BY created_at,version,profile_id""",
            (entity_id,),
        ))
        # Temporary negative values avoid collisions if a partially repaired
        # database already carries a uniqueness constraint under another name.
        for offset, row in enumerate(rows, start=1):
            conn.execute(
                'UPDATE profile_snapshots SET version=? WHERE profile_id=?',
                (-offset, row['profile_id']),
            )
        for offset, row in enumerate(rows, start=1):
            conn.execute(
                'UPDATE profile_snapshots SET version=? WHERE profile_id=?',
                (offset, row['profile_id']),
            )
            changed += int(row['version']) != offset
    return changed


_PROFILE_AUTOMATION_TABLES = (
    'profile_automation_subscriptions',
    'profile_refresh_queue',
)

_PROFILE_AUTOMATION_SUBSCRIPTION_COLUMNS = (
    'entity_id',
    'selector',
    'enabled',
    'debounce_seconds',
    'consent_scope',
    'last_profile_id',
    'last_refresh_at',
    'last_error_code',
    'created_at',
    'updated_at',
)

_PROFILE_REFRESH_QUEUE_COLUMNS = (
    'entity_id',
    'generation',
    'state',
    'reason',
    'available_at',
    'claimed_at',
    'attempt_count',
    'last_error_code',
    'created_at',
    'updated_at',
)


def _persistent_table_ddl(table: str) -> str:
    for sql in PERSISTENT_SCHEMA:
        if _object_name(sql, 'TABLE') == table:
            return sql
    raise RuntimeError(f'persistent table DDL is not registered: {table}')


def _profile_automation_tables_match_current(conn: sqlite3.Connection) -> bool:
    for table in _PROFILE_AUTOMATION_TABLES:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row is None:
            return False
        if _normalize_sql(row[0]) != EXPECTED_TABLE_SQL_SHAPES[table]['sql']:
            return False
    return True


def _repair_profile_automation_table_definitions(conn: sqlite3.Connection) -> bool:
    """Rebuild the v26 prototype tables without losing subscription state.

    Early v26 runtimes created these tables before their referential actions
    were finalized. SQLite cannot add or alter a foreign key in place, so v27
    preserves both tables' rows and replaces their DDL in the migration's
    existing transaction.
    """

    for table in _PROFILE_AUTOMATION_TABLES:
        conn.execute(_persistent_table_ddl(table))
    if _profile_automation_tables_match_current(conn):
        return False

    subscription_columns = ','.join(_PROFILE_AUTOMATION_SUBSCRIPTION_COLUMNS)
    queue_columns = ','.join(_PROFILE_REFRESH_QUEUE_COLUMNS)
    replacement = '__trove_profile_automation_subscriptions_v27'
    queue_backup = '__trove_profile_refresh_queue_v27_backup'

    conn.execute(f'DROP TABLE IF EXISTS "{replacement}"')
    conn.execute(f'DROP TABLE IF EXISTS temp."{queue_backup}"')
    conn.execute(
        f'CREATE TEMP TABLE "{queue_backup}" AS '
        f'SELECT {queue_columns} FROM profile_refresh_queue'
    )

    subscription_ddl = _persistent_table_ddl('profile_automation_subscriptions')
    create_prefix = 'CREATE TABLE IF NOT EXISTS profile_automation_subscriptions'
    if create_prefix not in subscription_ddl:
        raise RuntimeError('profile automation subscription DDL has an unexpected form')
    conn.execute(subscription_ddl.replace(
        create_prefix,
        f'CREATE TABLE "{replacement}"',
        1,
    ))
    conn.execute(
        f'INSERT INTO "{replacement}" ({subscription_columns}) '
        f'SELECT {subscription_columns} FROM profile_automation_subscriptions'
    )

    # Drop the dependent table first. The queue rows remain transactionally
    # protected in TEMP until the canonical parent table has been renamed.
    conn.execute('DROP TABLE profile_refresh_queue')
    conn.execute('DROP TABLE profile_automation_subscriptions')
    conn.execute(
        f'ALTER TABLE "{replacement}" RENAME TO profile_automation_subscriptions'
    )
    conn.execute(_persistent_table_ddl('profile_refresh_queue'))
    conn.execute(
        f'INSERT INTO profile_refresh_queue ({queue_columns}) '
        f'SELECT {queue_columns} FROM temp."{queue_backup}"'
    )
    conn.execute(f'DROP TABLE temp."{queue_backup}"')
    return True


def _fts_manifest_complete(conn: sqlite3.Connection) -> bool:
    """Check FTS DDL only; unlike reconciliation this never counts source rows."""

    tokenizer = None
    if _table_exists(conn, 'schema_meta'):
        row = conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()
        tokenizer = str(row[0]) if row else None
    return bool(
        tokenizer == FTS_TOKENIZER_VERSION
        and _table_exists(conn, 'message_fts')
        and _table_exists(conn, 'chunk_fts')
        and all(
            (row := conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
            ).fetchone())
            and _normalize_sql(row[0]) == EXPECTED_TRIGGER_SQL[name]
            for name in FTS_TRIGGER_NAMES
        )
        and all(
            (row := conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone())
            and _normalize_sql(row[0]) == EXPECTED_VIRTUAL_TABLE_SQL[name]
            for name in ('message_fts', 'chunk_fts')
        )
    )


def _reconcile_fts(
    conn: sqlite3.Connection,
    fault: Callable[[str], None] | None,
    *,
    force: bool = False,
) -> dict[str, int]:
    complete = _fts_manifest_complete(conn)
    if complete and not force:
        message_rows = int(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0])
        chunk_rows = int(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE status='active'").fetchone()[0])
        return {'message_rows': message_rows, 'chunk_rows': chunk_rows}
    check = conn.execute('PRAGMA quick_check').fetchone()
    if check is not None and str(check[0]).lower() != 'ok':
        raise sqlite3.DatabaseError(f'quick_check failed before FTS migration: {check[0]}')
    for name in FTS_TRIGGER_NAMES:
        conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')
    conn.execute('DROP TABLE IF EXISTS chunk_fts')
    conn.execute('DROP TABLE IF EXISTS message_fts')
    if fault:
        fault('after_fts_drop')
    for sql in TRIGRAM_FTS_SCHEMA:
        conn.execute(sql)
    message_rows = int(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0])
    chunk_rows = int(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE status='active'").fetchone()[0])
    if message_rows:
        conn.execute(
            """INSERT INTO message_fts(rowid,citation,content,sender_name,conversation_title)
               SELECT id,citation,content,sender_name,conversation_title FROM messages"""
        )
    if chunk_rows:
        conn.execute(
            """INSERT INTO chunk_fts(rowid,chunk_citation,content,title,actor)
               SELECT rowid,chunk_citation,content,title,actor FROM evidence_chunks WHERE status='active'"""
        )
    conn.execute(
        'INSERT OR REPLACE INTO schema_meta(key,value) VALUES(?,?)',
        ('fts_tokenizer', FTS_TOKENIZER_VERSION),
    )
    conn.execute('INSERT OR REPLACE INTO schema_meta(key,value) VALUES(?,?)', ('message_fts_rows', str(message_rows)))
    conn.execute('INSERT OR REPLACE INTO schema_meta(key,value) VALUES(?,?)', ('chunk_fts_rows', str(chunk_rows)))
    return {'message_rows': message_rows, 'chunk_rows': chunk_rows}


def _apply_v12_manifest(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    # Explicit application-owned indexes are cheap to rebuild and cannot be
    # repaired by CREATE IF NOT EXISTS when a same-named definition has drifted.
    # Never touch SQLite autoindexes: EXPECTED_INDEXES contains named DDL only.
    for name in EXPECTED_INDEXES:
        conn.execute(f'DROP INDEX IF EXISTS "{name}"')
    conn.execute('DROP INDEX IF EXISTS idx_media_assets_unique_ref')
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    if fault:
        fault('after_manifest_ddl')
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(
        conn,
        'vector_index_generations',
        'revision',
        'revision INTEGER NOT NULL DEFAULT 1',
    )
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')

    _reconcile_fts(conn, fault)


def _apply_v13_search_indexes(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    names = {
        'idx_conversations_id_account',
        'idx_messages_conversation_time',
        'idx_evidence_chunks_source_id_status_time',
    }
    for name in names:
        conn.execute(f'DROP INDEX IF EXISTS "{name}"')
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_bounded_search_indexes')


def _apply_v14_delta_state(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    """Install citation tombstones and media source watermarks.

    These tables belong to the same SQLite generation as the projections they
    describe.  Keeping their DDL in the manifest (rather than lazily creating
    tables in sync code) preserves atomic generation publication.
    """
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_delta_state')


def _apply_v15_vector_ledger(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_vector_ledger')


def _apply_v16_vector_generation_revision(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    """Add the transaction-authoritative vector score-domain revision."""

    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(
        conn,
        'vector_index_generations',
        'revision',
        'revision INTEGER NOT NULL DEFAULT 1',
    )
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_vector_generation_revision')


_ENTITY_USER_ID_KEYS = {'user_id', 'wechat_username', 'wechat_id', 'username', 'wxid', 'openim_id', 'primary_user_id'}
_ENTITY_ALIAS_KEYS = {'alias', 'aliases', 'remark', 'nickname', 'display_name', 'group_alias', 'group_display_name', 'group_name', 'name'}


def _normalized_identifier(value: Any) -> str:
    text = unicodedata.normalize('NFKC', str(value or '')).strip().casefold()
    return ' '.join(text.split())


def _identifier_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_identifier_values(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_identifier_values(item))
        return out
    normalized = _normalized_identifier(value)
    return [normalized] if normalized else []


def _rebuild_entity_identifier_index(conn: sqlite3.Connection) -> None:
    conn.execute('DELETE FROM entity_identifiers')
    for row in conn.execute('SELECT entity_id,display_name,identifiers_json,confidence,created_at,updated_at FROM entities'):
        entity_id, display_name, identifiers_json, confidence, created_at, updated_at = row
        records: set[tuple[str, str, str]] = set()
        display = _normalized_identifier(display_name)
        if display:
            records.add(('display_name', display, 'entity'))
        try:
            identifiers = json.loads(identifiers_json or '{}')
        except (TypeError, json.JSONDecodeError):
            identifiers = {}
        if isinstance(identifiers, dict):
            for key, value in identifiers.items():
                if key not in _ENTITY_USER_ID_KEYS and key not in _ENTITY_ALIAS_KEYS:
                    continue
                kind = 'user_id' if key in _ENTITY_USER_ID_KEYS else key
                for normalized in _identifier_values(value):
                    records.add((kind, normalized, 'entity'))
        for kind, normalized, source in records:
            conn.execute(
                """INSERT OR IGNORE INTO entity_identifiers(
                       entity_id,identifier_type,normalized_value,source,confidence,citation,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (entity_id, kind, normalized, source, min(max(float(confidence or 0), 0.0), 1.0), None, created_at, updated_at),
            )
    recognized = tuple(sorted(_ENTITY_USER_ID_KEYS | _ENTITY_ALIAS_KEYS))
    placeholders = ','.join('?' for _ in recognized)
    for row in conn.execute(
        f"""SELECT entity_id,observation_type,value_json,confidence,citation,created_at,updated_at
              FROM observations
             WHERE status IN ('active','needs_review','merge_candidate')
               AND lower(observation_type) IN ({placeholders})""",
        tuple(value.lower() for value in recognized),
    ):
        entity_id, observation_type, value_json, confidence, citation, created_at, updated_at = row
        try:
            value = json.loads(value_json or '{}')
        except (TypeError, json.JSONDecodeError):
            value = {}
        kind = 'user_id' if str(observation_type).lower() in _ENTITY_USER_ID_KEYS else str(observation_type).lower()
        for normalized in _identifier_values(value):
            conn.execute(
                """INSERT INTO entity_identifiers(
                       entity_id,identifier_type,normalized_value,source,confidence,citation,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(entity_id,identifier_type,normalized_value,source) DO UPDATE SET
                       confidence=MAX(entity_identifiers.confidence,excluded.confidence),
                       citation=COALESCE(excluded.citation,entity_identifiers.citation),
                       updated_at=excluded.updated_at""",
                (entity_id, kind, normalized, 'observation', min(max(float(confidence or 0), 0.0), 1.0), citation, created_at, updated_at),
            )


def _apply_v17_entity_identifier_index(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _rebuild_entity_identifier_index(conn)
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_entity_identifier_index')


def _apply_v18_message_payloads(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_message_payloads')


def _apply_v19_media_source_registry(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_media_source_registry')


def _apply_v20_profile_enrichment(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_transcripts_one_active_asset', 'idx_image_observations_projection_identity'),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _ensure_column(conn, 'profile_enrichment_runs', 'execution_location', "execution_location TEXT NOT NULL DEFAULT 'local'")
    _ensure_column(conn, 'profile_enrichment_runs', 'processor_identity', "processor_identity TEXT NOT NULL DEFAULT 'local-agent/default'")
    _ensure_column(conn, 'profile_enrichment_runs', 'purpose', "purpose TEXT NOT NULL DEFAULT 'customer_profile_enrichment'")
    _ensure_column(conn, 'profile_enrichment_runs', 'revoked_at', 'revoked_at TEXT')
    _ensure_column(conn, 'profile_enrichment_tasks', 'approval_id', 'approval_id TEXT')
    _ensure_column(conn, 'profile_enrichment_tasks', 'approval_scope_hash', 'approval_scope_hash TEXT')
    _ensure_column(conn, 'profile_enrichment_tasks', 'delivery_token_hash', 'delivery_token_hash TEXT')
    _ensure_column(conn, 'profile_enrichment_tasks', 'delivery_consumed_at', 'delivery_consumed_at TEXT')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_profile_enrichment')


def _apply_v21_voice_execution_policy(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    # Older provider-specific transcript ids could leave more than one active
    # row for an asset.  Keep the newest active projection before installing
    # the invariant used by the lazy execution policy.
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='transcripts'").fetchone()
    if exists is not None:
        conn.execute(
            """UPDATE transcripts SET status='superseded'
                 WHERE status='active' AND EXISTS (
                     SELECT 1 FROM transcripts newer
                      WHERE newer.asset_id=transcripts.asset_id AND newer.status='active'
                        AND (newer.created_at>transcripts.created_at
                             OR (newer.created_at=transcripts.created_at AND newer.transcript_id>transcripts.transcript_id))
                 )"""
        )
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_image_observations_projection_identity',),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_voice_execution_policy')


def _apply_v22_image_understanding_projection(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    # The projection identity index is installed after legacy rows receive
    # deterministic version fields.
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_image_observations_projection_identity',),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _ensure_column(conn, 'image_observations', 'content_sha256', "content_sha256 TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'image_observations', 'model_id', "model_id TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'image_observations', 'prompt_version', "prompt_version TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'image_observations', 'updated_at', "updated_at TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'profile_enrichment_runs', 'prompt_version', "prompt_version TEXT NOT NULL DEFAULT 'profile-enrichment/v1'")
    _ensure_column(conn, 'profile_enrichment_tasks', 'prompt_version', "prompt_version TEXT NOT NULL DEFAULT 'profile-enrichment/v1'")
    conn.execute(
        """UPDATE image_observations
              SET content_sha256=COALESCE((SELECT content_hash FROM media_assets ma WHERE ma.asset_id=image_observations.asset_id),'legacy'),
                  model_id=CASE WHEN model_id='' THEN 'legacy:' || observation_id ELSE model_id END,
                  prompt_version=CASE WHEN prompt_version='' THEN 'legacy' ELSE prompt_version END,
                  updated_at=CASE WHEN updated_at='' THEN created_at ELSE updated_at END
            WHERE content_sha256='' OR model_id='' OR prompt_version='' OR updated_at=''"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_image_observations_projection_identity
             ON image_observations(asset_id,citation,content_sha256,model_id,prompt_version)"""
    )
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_image_understanding_projection')


def _apply_v23_profile_snapshot_contract(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _apply_current_persistent_schema(
        conn,
        skip_tokens=('idx_profile_snapshots_entity_',),
    )
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _ensure_column(conn, 'profile_snapshots', 'content_hash', "content_hash TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'profile_snapshots', 'source_revision', "source_revision TEXT NOT NULL DEFAULT 'legacy'")
    _ensure_column(conn, 'profile_snapshots', 'run_id', 'run_id TEXT')
    _ensure_column(conn, 'profile_snapshots', 'schema_version', "schema_version TEXT NOT NULL DEFAULT 'customer-profile/legacy'")
    _ensure_column(conn, 'profile_snapshots', 'completeness_state', "completeness_state TEXT NOT NULL DEFAULT 'stale'")
    _ensure_column(conn, 'profile_snapshots', 'evidence_citations_json', "evidence_citations_json TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, 'profile_snapshots', 'enrichment_summary_json', "enrichment_summary_json TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, 'profile_snapshots', 'gaps_json', "gaps_json TEXT NOT NULL DEFAULT '[]'")
    for row in conn.execute("SELECT profile_id,entity_id,version,projection_json FROM profile_snapshots WHERE content_hash='' OR content_hash IS NULL"):
        digest = hashlib.sha256(json.dumps({
            'entity_id': row['entity_id'],
            'version': int(row['version']),
            'projection': row['projection_json'],
        }, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        conn.execute('UPDATE profile_snapshots SET content_hash=? WHERE profile_id=?', (digest, row['profile_id']))
    conn.execute('CREATE INDEX IF NOT EXISTS idx_profile_snapshots_entity_version ON profile_snapshots(entity_id,version)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_profile_snapshots_entity_hash ON profile_snapshots(entity_id,content_hash)')
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_profile_snapshot_contract')


def _apply_v24_derived_data_lifecycle(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    _ensure_column(conn, 'messages', 'content_kind', "content_kind TEXT NOT NULL DEFAULT 'text'")
    _ensure_column(conn, 'moment_interactions', 'actor_name', "actor_name TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'sync_dirty_citations', 'source_type', "source_type TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, 'vector_entries', 'content_hash', 'content_hash TEXT')
    _ensure_column(conn, 'approval_records', 'consumed_at', 'consumed_at TEXT')
    _ensure_column(conn, 'approval_records', 'consumption_id', 'consumption_id TEXT')
    _ensure_column(conn, 'vector_index_generations', 'revision', 'revision INTEGER NOT NULL DEFAULT 1')
    _apply_current_persistent_schema(conn)
    _reconcile_fts(conn, fault)
    if fault:
        fault('after_derived_data_lifecycle')


def _apply_v25_vector_source_revision(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    """Install the constant-time vector-source CAS without scanning source rows."""

    _apply_current_persistent_schema(conn)
    conn.execute(
        'INSERT OR IGNORE INTO schema_meta(key,value) VALUES(?,?)',
        (VECTOR_SOURCE_REVISION_KEY, '0'),
    )
    # A genuine v24 database already has the exact FTS manifest, so this is a
    # metadata-only check.  Repair only malformed historical fixtures/databases;
    # the normal v24 -> v25 migration never scans messages or evidence chunks.
    if not _fts_manifest_complete(conn):
        _reconcile_fts(conn, fault)
    if fault:
        fault('after_vector_source_revision')


def _apply_v26_profile_automation(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    """Install explicit per-person automatic profile maintenance state."""

    _apply_current_persistent_schema(conn)
    if not _fts_manifest_complete(conn):
        _reconcile_fts(conn, fault)
    if fault:
        fault('after_profile_automation')


def _apply_v27_profile_snapshot_version_uniqueness(
    conn: sqlite3.Connection,
    fault: Callable[[str], None] | None = None,
) -> None:
    """Repair merged histories and enforce one immutable row per version."""

    _repair_profile_automation_table_definitions(conn)
    _apply_current_persistent_schema(conn)
    if not _fts_manifest_complete(conn):
        _reconcile_fts(conn, fault)
    if fault:
        fault('after_profile_snapshot_version_uniqueness')


def _apply_v28_operation_journal(
    conn: sqlite3.Connection,
    fault: Callable[[str], None] | None = None,
) -> None:
    """Install the durable capability operation state machine."""

    _repair_profile_automation_table_definitions(conn)
    _apply_current_persistent_schema(conn)
    if not _fts_manifest_complete(conn):
        _reconcile_fts(conn, fault)
    if fault:
        fault('after_operation_journal')


def _repair_current_manifest(conn: sqlite3.Connection, fault: Callable[[str], None] | None = None) -> None:
    """Rebuild drifted DDL and reinstall deferred unique indexes safely."""

    _apply_v12_manifest(conn, fault)
    _apply_v21_voice_execution_policy(conn, fault)
    _apply_v22_image_understanding_projection(conn, fault)
    _apply_v23_profile_snapshot_contract(conn, fault)
    _apply_v24_derived_data_lifecycle(conn, fault)
    _apply_v25_vector_source_revision(conn, fault)
    _apply_v26_profile_automation(conn, fault)
    _apply_v27_profile_snapshot_version_uniqueness(conn, fault)
    _apply_v28_operation_journal(conn, fault)


MIGRATIONS = (
    Migration(12, 'complete-v12-manifest', _apply_v12_manifest),
    Migration(13, 'bounded-search-indexes', _apply_v13_search_indexes),
    Migration(14, 'delta-projection-state', _apply_v14_delta_state),
    Migration(15, 'authoritative-vector-ledger', _apply_v15_vector_ledger),
    Migration(16, 'vector-generation-revision', _apply_v16_vector_generation_revision),
    Migration(17, 'entity-identifier-index', _apply_v17_entity_identifier_index),
    Migration(18, 'normalized-message-payloads', _apply_v18_message_payloads),
    Migration(19, 'media-source-registry', _apply_v19_media_source_registry),
    Migration(20, 'profile-enrichment-runs', _apply_v20_profile_enrichment),
    Migration(21, 'voice-execution-policy', _apply_v21_voice_execution_policy),
    Migration(22, 'image-understanding-projection', _apply_v22_image_understanding_projection),
    Migration(23, 'profile-snapshot-contract', _apply_v23_profile_snapshot_contract),
    Migration(24, 'derived-data-lifecycle', _apply_v24_derived_data_lifecycle),
    Migration(25, 'vector-source-revision', _apply_v25_vector_source_revision),
    Migration(26, 'profile-automation', _apply_v26_profile_automation),
    Migration(27, 'profile-snapshot-version-uniqueness', _apply_v27_profile_snapshot_version_uniqueness),
    Migration(28, 'operation-journal', _apply_v28_operation_journal),
)
assert MIGRATIONS[-1].version == SCHEMA_VERSION
assert tuple(item.version for item in MIGRATIONS) == tuple(sorted({item.version for item in MIGRATIONS}))


def _migration_steps(current: int, target: int, *, repair_current: bool = False) -> list[Migration]:
    by_version = {item.version: item for item in MIGRATIONS}
    if repair_current:
        migration = by_version.get(target)
        if migration is None:
            raise SchemaMigrationRequired(current, target, reason='no repair migration is registered')
        return [migration]
    steps: list[Migration] = []
    first_numbered = MIGRATIONS[0].version
    for version in range(current + 1, target + 1):
        migration = by_version.get(version)
        if migration is not None:
            steps.append(migration)
        elif version >= first_numbered:
            raise SchemaMigrationRequired(current, target, reason=f'migration {version} is not registered')
    if current < target and not steps:
        raise SchemaMigrationRequired(current, target, reason='no migration path is registered')
    return steps


def migrate_schema(
    conn: sqlite3.Connection,
    *,
    target_version: int = SCHEMA_VERSION,
    fault_injector: Callable[[str], None] | None = None,
) -> int:
    """Upgrade legacy schemas in one transaction; version advances last."""

    current = current_schema_version(conn)
    metadata_version = metadata_schema_version(conn)
    newest = max(current, metadata_version or 0)
    if newest > target_version:
        raise SchemaVersionTooNew(
            newest,
            target_version,
            reason='database schema is newer than this runtime',
        )
    if metadata_version is not None and metadata_version != current:
        raise SchemaVersionMismatch(current, target_version, reason='database schema versions disagree')
    if current == target_version:
        try:
            validate_schema(conn, required_version=target_version)
            return current
        except SchemaMigrationRequired:
            # A previously scattered/partial schema may already claim the latest
            # version. Explicit writable initialization reconciles it transactionally;
            # read-only open still fails without changing a byte.
            pass
    if conn.in_transaction:
        raise RuntimeError('migrate_schema requires a connection with no active transaction')
    conn.execute('BEGIN IMMEDIATE')
    try:
        # BEGIN IMMEDIATE is the cross-process schema mutation coordinator. Re-read
        # after acquiring it so a waiter never repeats a migration completed by
        # the process that held the lock first.
        locked_current = current_schema_version(conn)
        locked_metadata = metadata_schema_version(conn)
        locked_newest = max(locked_current, locked_metadata or 0)
        if locked_newest > target_version:
            raise SchemaVersionTooNew(
                locked_newest,
                target_version,
                reason='database schema is newer than this runtime',
            )
        if locked_metadata is not None and locked_metadata != locked_current:
            raise SchemaVersionMismatch(
                locked_current,
                target_version,
                reason='database schema versions disagree',
            )
        repair_current = False
        if locked_current == target_version:
            try:
                validate_schema(conn, required_version=target_version)
            except SchemaMigrationRequired:
                repair_current = True
            else:
                conn.execute('COMMIT')
                return target_version
        if repair_current:
            # A latest-version database with any manifest drift needs the full
            # reconciler, not only the latest incremental migration.
            _repair_current_manifest(conn, fault_injector)
        else:
            steps = _migration_steps(locked_current, target_version)
            for migration in steps:
                migration.apply(conn, fault_injector)
        if fault_injector:
            fault_injector('before_version_write')
        conn.execute(f'PRAGMA user_version = {target_version}')
        conn.execute(
            'INSERT OR REPLACE INTO schema_meta(key,value) VALUES(?,?)',
            ('schema_version', str(target_version)),
        )
        validate_schema(conn, required_version=target_version)
        if fault_injector:
            fault_injector('before_commit')
        conn.execute('COMMIT')
    except BaseException:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise
    return target_version


def rebuild_fts_transaction(
    conn: sqlite3.Connection,
    *,
    fault_injector: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Force-rebuild both FTS generations atomically without changing versions."""

    if conn.in_transaction:
        raise RuntimeError('rebuild_fts_transaction requires no active transaction')
    conn.execute('BEGIN IMMEDIATE')
    try:
        current = current_schema_version(conn)
        metadata = metadata_schema_version(conn)
        if current != SCHEMA_VERSION or metadata != SCHEMA_VERSION:
            raise SchemaMigrationRequired(current, SCHEMA_VERSION)
        report = _reconcile_fts(conn, fault_injector, force=True)
        validate_schema(conn)
        if current_schema_version(conn) != current or metadata_schema_version(conn) != metadata:
            raise RuntimeError('FTS rebuild changed schema versions')
        if fault_injector:
            fault_injector('before_fts_commit')
        conn.execute('COMMIT')
        return report
    except BaseException:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise


def validate_schema(conn: sqlite3.Connection, *, required_version: int = SCHEMA_VERSION) -> None:
    current = current_schema_version(conn)
    metadata_version = metadata_schema_version(conn)
    newest = max(current, metadata_version or 0)
    if newest > required_version:
        raise SchemaVersionTooNew(newest, required_version, reason='database schema is newer than this runtime')
    if metadata_version is not None and metadata_version != current:
        raise SchemaVersionMismatch(current, required_version, reason='database schema versions disagree')
    rows = list(conn.execute(
        "SELECT type,name FROM sqlite_master WHERE type IN ('table','index','trigger')"
    ))
    present = {(str(row[0]), str(row[1])) for row in rows}
    missing = [f'table:{name}' for name in EXPECTED_TABLES if ('table', name) not in present]
    missing.extend(f'index:{name}' for name in EXPECTED_INDEXES if ('index', name) not in present)
    missing.extend(f'trigger:{name}' for name in EXPECTED_TRIGGERS if ('trigger', name) not in present)
    for table, columns in REQUIRED_COLUMNS.items():
        actual = _column_names(conn, table)
        missing.extend(f'column:{table}.{column}' for column in columns - actual)
    for name, expected in EXPECTED_INDEX_COLUMNS.items():
        if ('index', name) not in present:
            continue
        actual = tuple(str(row[2]) for row in conn.execute(f'PRAGMA index_info("{name}")'))
        if actual != expected:
            missing.append(f'index-definition:{name}')
    if not missing:
        (
            actual_tables,
            actual_keys,
            actual_foreign_keys,
            actual_indexes,
            actual_triggers,
            actual_virtual_tables,
            actual_table_sql,
        ) = _manifest_shapes(conn)
        for table, expected in EXPECTED_TABLE_SHAPES.items():
            if actual_tables.get(table) != expected:
                missing.append(f'table-definition:{table}')
            if actual_keys.get(table) != EXPECTED_TABLE_KEY_SHAPES[table]:
                missing.append(f'table-key-definition:{table}')
            if actual_foreign_keys.get(table) != EXPECTED_FOREIGN_KEY_SHAPES[table]:
                missing.append(f'foreign-key-definition:{table}')
        for name, expected in EXPECTED_INDEX_SHAPES.items():
            if actual_indexes.get(name) != expected:
                missing.append(f'index-definition:{name}')
        for name, expected in EXPECTED_TRIGGER_SQL.items():
            if actual_triggers.get(name) != expected:
                missing.append(f'trigger-definition:{name}')
        for name, expected in EXPECTED_VIRTUAL_TABLE_SQL.items():
            if actual_virtual_tables.get(name) != expected:
                missing.append(f'virtual-table-definition:{name}')
        for table, expected in EXPECTED_TABLE_SQL_SHAPES.items():
            actual = actual_table_sql.get(table) or {}
            if actual.get('autoincrement') != expected['autoincrement']:
                missing.append(f'table-autoincrement:{table}')
            if actual.get('checks') != expected['checks']:
                missing.append(f'table-check-definition:{table}')
            if table not in ALTER_COMPATIBLE_TABLES and actual.get('sql') != expected['sql']:
                missing.append(f'table-sql-definition:{table}')
    meta_version = None
    fts_tokenizer = None
    if ('table', 'schema_meta') in present:
        row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
        meta_version = int(row[0]) if row and str(row[0]).isdigit() else None
        row = conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()
        fts_tokenizer = str(row[0]) if row else None
    if fts_tokenizer != FTS_TOKENIZER_VERSION:
        missing.append('metadata:fts_tokenizer')
    if current != required_version or meta_version != required_version or missing:
        raise SchemaMigrationRequired(current, required_version, missing_objects=missing)


_SCHEMA_VALIDATION_CACHE_MAX = 128
_SCHEMA_VALIDATION_CACHE: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_SCHEMA_VALIDATION_CACHE_LOCK = threading.Lock()


def schema_file_identity(path: str | Path) -> tuple[Any, ...]:
    resolved = Path(path).expanduser().resolve()
    try:
        stat_result = resolved.stat()
    except OSError as exc:
        raise SchemaPreflightUnavailable(
            0,
            SCHEMA_VERSION,
            reason='database identity cannot be read safely',
        ) from exc
    return (
        str(resolved),
        int(stat_result.st_dev),
        int(stat_result.st_ino),
    )


def _schema_validation_token(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    required_version: int,
    expected_identity: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    identity_before = schema_file_identity(path)
    if expected_identity is not None and identity_before != expected_identity:
        raise SchemaPreflightUnavailable(
            current_schema_version(conn),
            required_version,
            reason='database changed after its read connection was opened',
        )
    current, metadata = preflight_connection_versions(conn, required_version=required_version)
    schema_cookie_row = conn.execute('PRAGMA schema_version').fetchone()
    schema_cookie = int(schema_cookie_row[0]) if schema_cookie_row else -1
    tokenizer = None
    if _table_exists(conn, 'schema_meta'):
        row = conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()
        tokenizer = str(row[0]) if row is not None else None
    identity_after = schema_file_identity(path)
    if identity_before != identity_after or (
        expected_identity is not None and identity_after != expected_identity
    ):
        raise SchemaPreflightUnavailable(
            current,
            required_version,
            reason='database changed during schema validation',
        )
    return (
        'schema-manifest-v1',
        *identity_after,
        int(current),
        int(metadata) if metadata is not None else None,
        schema_cookie,
        tokenizer,
        int(required_version),
    )


def validate_schema_cached(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    required_version: int = SCHEMA_VERSION,
    expected_identity: tuple[Any, ...] | None = None,
) -> None:
    """Validate the full manifest once per coherent SQLite schema generation.

    Every open still checks both version authorities.  The bounded process cache
    skips only the expensive sqlite_master/table-shape walk when the database
    identity, SQLite schema cookie, schema versions, and FTS contract are all
    unchanged.  DDL, file replacement, migration, or tokenizer drift therefore
    forces a full validation before the cache can be refreshed.
    """

    cache_key = _schema_validation_token(
        conn,
        path,
        required_version=required_version,
        expected_identity=expected_identity,
    )
    with _SCHEMA_VALIDATION_CACHE_LOCK:
        if cache_key in _SCHEMA_VALIDATION_CACHE:
            _SCHEMA_VALIDATION_CACHE.move_to_end(cache_key)
            return

    validate_schema(conn, required_version=required_version)

    # Do not cache a validation across a replacement that raced the manifest
    # walk.  Re-reading the token also catches DDL committed by another process.
    confirmed = _schema_validation_token(
        conn,
        path,
        required_version=required_version,
        expected_identity=expected_identity,
    )
    if confirmed != cache_key:
        raise SchemaPreflightUnavailable(
            current_schema_version(conn),
            required_version,
            reason='database changed during schema validation',
        )
    with _SCHEMA_VALIDATION_CACHE_LOCK:
        _SCHEMA_VALIDATION_CACHE[confirmed] = None
        _SCHEMA_VALIDATION_CACHE.move_to_end(confirmed)
        while len(_SCHEMA_VALIDATION_CACHE) > _SCHEMA_VALIDATION_CACHE_MAX:
            _SCHEMA_VALIDATION_CACHE.popitem(last=False)


def _clear_schema_validation_cache_for_tests() -> None:
    with _SCHEMA_VALIDATION_CACHE_LOCK:
        _SCHEMA_VALIDATION_CACHE.clear()


def foreign_key_compatibility_report(conn: sqlite3.Connection) -> dict[str, Any]:
    """Audit only. It intentionally does not enable foreign_keys."""

    enabled_row = conn.execute('PRAGMA foreign_keys').fetchone()
    violations = list(conn.execute('PRAGMA foreign_key_check'))
    orphan_counts = Counter(str(row[0]) for row in violations)
    actions: Counter[str] = Counter()
    declared = 0
    tables = [str(row[0]) for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]
    for table in tables:
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            declared += 1
            actions[f'on_update:{str(row[5]).lower()}'] += 1
            actions[f'on_delete:{str(row[6]).lower()}'] += 1
    return {
        'foreign_keys_enabled': bool(enabled_row and enabled_row[0]),
        'declared_foreign_keys': declared,
        'orphan_count': len(violations),
        'orphan_counts_by_table': dict(sorted(orphan_counts.items())),
        'action_counts': dict(sorted(actions.items())),
        'counts_only': True,
    }
