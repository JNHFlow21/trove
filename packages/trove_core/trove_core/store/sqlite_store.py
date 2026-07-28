from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import json
import re
import sqlite3
import threading
import os
import weakref
from typing import Any, Callable, Iterable
from datetime import datetime, timezone

from trove_core.domain.messages import Account, Conversation, Message
from trove_core.domain.content import classify_content_kind, display_content_for_kind
from trove_core.wechat.parsers.appmsg import parse_appmsg
from trove_core.search.chunking import chunk_text
from .schema import FTS_TOKENIZER_VERSION, SCHEMA_VERSION
from .migrations import (
    SchemaMigrationRequired,
    SchemaPreflightUnavailable,
    migrate_schema,
    preflight_connection_versions,
    preflight_database_versions,
    rebuild_fts_transaction,
    schema_file_identity,
    validate_schema,
    validate_schema_cached,
)


_MESSAGE_PAYLOAD_STATUSES = {'parsed', 'unsupported', 'malformed', 'rejected'}
_MESSAGE_PAYLOAD_FIELDS = {
    'title', 'description', 'link_identity', 'file_name', 'file_extension', 'file_size',
    'mini_program_app_id', 'mini_program_username', 'mini_program_page_hash',
    'quote_type', 'quote_sender', 'quote_title', 'location_label', 'latitude', 'longitude',
    'duration_seconds', 'room_type', 'message_type',
}
_MESSAGE_PAYLOAD_UPSERT_SQL = """INSERT INTO message_payloads(
       citation,appmsg_type,normalized_type,parse_status,normalized_json,display_text,
       source_hash,parser_version,unsupported_reason,created_at,updated_at
   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
   ON CONFLICT(citation) DO UPDATE SET
       appmsg_type=excluded.appmsg_type,
       normalized_type=excluded.normalized_type,
       parse_status=excluded.parse_status,
       normalized_json=excluded.normalized_json,
       display_text=excluded.display_text,
       source_hash=excluded.source_hash,
       parser_version=excluded.parser_version,
       unsupported_reason=excluded.unsupported_reason,
       updated_at=excluded.updated_at
   WHERE message_payloads.appmsg_type IS NOT excluded.appmsg_type
      OR message_payloads.normalized_type IS NOT excluded.normalized_type
      OR message_payloads.parse_status IS NOT excluded.parse_status
      OR message_payloads.normalized_json IS NOT excluded.normalized_json
      OR message_payloads.display_text IS NOT excluded.display_text
      OR message_payloads.source_hash IS NOT excluded.source_hash
      OR message_payloads.parser_version IS NOT excluded.parser_version
      OR message_payloads.unsupported_reason IS NOT excluded.unsupported_reason"""

_CLOUD_ASR_PROVIDER_NAME = 'volcengine-asr-flash'
_CLOUD_ASR_MODEL_ID = 'bigmodel:volc.bigasr.auc_turbo'


def _queryable_transcript_chunk_sql(alias: str = 'e') -> str:
    """Keep transcript projections pinned to the current cloud result."""

    return f"""(
        {alias}.source_type <> 'transcript'
        OR EXISTS(
            SELECT 1
              FROM transcripts t
              JOIN provider_jobs pj ON pj.job_id=t.job_id
              JOIN media_assets ma ON ma.asset_id=t.asset_id
             WHERE t.citation={alias}.parent_citation
               AND t.status='active'
               AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
               AND pj.model='{_CLOUD_ASR_MODEL_ID}'
               AND pj.status='completed'
               AND pj.request_hash=ma.content_hash
        )
    )"""


def _normalized_payload_values(citation: str, payload: dict[str, Any], *, timestamp: str) -> tuple[Any, ...]:
    if not isinstance(payload, dict):
        raise TypeError('normalized_payload must be an object')
    status = str(payload.get('parse_status') or '')
    normalized_type = str(payload.get('normalized_type') or '')
    source_hash = str(payload.get('source_hash') or '')
    parser_version = str(payload.get('parser_version') or '')
    display_text = str(payload.get('display_text') or '')
    unsupported_reason = payload.get('unsupported_reason')
    fields = payload.get('fields') or {}
    if status not in _MESSAGE_PAYLOAD_STATUSES:
        raise ValueError('invalid normalized payload status')
    if not re.fullmatch(r'[a-z0-9_]{1,64}', normalized_type):
        raise ValueError('invalid normalized payload type')
    if not re.fullmatch(r'[0-9a-f]{64}', source_hash):
        raise ValueError('invalid normalized payload source hash')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', parser_version):
        raise ValueError('invalid normalized payload parser version')
    if not display_text.startswith('[appmsg/') or len(display_text) > 1400:
        raise ValueError('invalid normalized payload display')
    if not isinstance(fields, dict) or set(fields) - _MESSAGE_PAYLOAD_FIELDS:
        raise ValueError('normalized payload contains unknown fields')
    normalized_json = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if len(normalized_json.encode('utf-8')) > 16 * 1024:
        raise ValueError('normalized payload fields exceed size limit')
    lowered = (normalized_json + display_text).lower()
    if any(marker in lowered for marker in ('<msg', '<appmsg', '<!doctype', 'http://', 'https://')):
        raise ValueError('normalized payload contains raw or fetchable content')
    appmsg_type = payload.get('appmsg_type')
    if appmsg_type is not None and type(appmsg_type) is not int:
        raise TypeError('appmsg_type must be an exact integer or null')
    if unsupported_reason is not None:
        unsupported_reason = str(unsupported_reason)
        if not re.fullmatch(r'[a-z0-9_]{1,80}', unsupported_reason):
            raise ValueError('invalid normalized payload unsupported reason')
    return (
        citation,
        appmsg_type,
        normalized_type,
        status,
        normalized_json,
        display_text,
        source_hash,
        parser_version,
        unsupported_reason,
        timestamp,
        timestamp,
    )


def _message_payload_values(message: Message, *, timestamp: str) -> tuple[Any, ...] | None:
    payload = message.normalized_payload
    if payload is None:
        return None
    return _normalized_payload_values(message.citation, payload, timestamp=timestamp)


class EvidenceRow(dict):
    """Small row-compatible mapping for non-message evidence records."""
    def __getitem__(self, key: str) -> Any:  # keep sqlite.Row-style access typing simple
        return dict.__getitem__(self, key)


class TrackedConnection(sqlite3.Connection):
    """SQLite handle registered with exactly one ``SQLiteStore`` generation."""

    _trove_closed = False
    _trove_generation = -1
    _trove_on_close: Callable[[sqlite3.Connection], None] | None = None

    def _trove_configure(
        self,
        *,
        generation: int,
        on_close: Callable[[sqlite3.Connection], None],
    ) -> None:
        self._trove_closed = False
        self._trove_generation = generation
        self._trove_on_close = on_close

    def close(self) -> None:
        if self._trove_closed:
            return
        callback = self._trove_on_close
        try:
            super().close()
        finally:
            self._trove_closed = True
            self._trove_on_close = None
            if callback is not None:
                callback(self)


class ClosingConnection(TrackedConnection):
    """sqlite3 connection whose context manager commits/rolls back and closes.

    Python's built-in sqlite3 context manager does not close the connection, which
    creates noisy ResourceWarnings under Python 3.14 during the full release suite.
    """

    def __exit__(self, exc_type, exc, tb) -> bool:
        super().__exit__(exc_type, exc, tb)
        self.close()
        return False


class ReusableConnection(TrackedConnection):
    """Thread-local connection whose context manager preserves the handle.

    sqlite3.Connection's default context manager already commits/rolls back
    without closing. This subclass makes that contract explicit for the
    thread-local pool and gives call sites a concrete type separate from
    ClosingConnection, which remains available for one-shot connections.
    """
class ReadOnlyConnection(ReusableConnection):
    """A read connection whose context manager never commits."""

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def commit(self) -> None:
        raise sqlite3.OperationalError('read-only store cannot commit')


class ReadOnlyClosingConnection(ReadOnlyConnection):
    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class ReadOnlyStoreError(RuntimeError):
    code = 'readonly_store_unavailable'

    def __init__(self, message: str = 'read-only SQLite snapshot is unavailable'):
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {'error': {'code': self.code, 'message': str(self)}}


SCHEMA_READ_ERRORS = (SchemaMigrationRequired, ReadOnlyStoreError)


class SQLiteConnectionLimit(RuntimeError):
    code = 'sqlite_connection_limit'

    def __init__(self, message: str = 'SQLite read pool reached its hard connection limit'):
        super().__init__(message)


class _ConnectionGate:
    """A small re-entrant reader/writer gate for connection generations.

    Normal SQLite operations hold a reader lease.  ``close_all`` takes the
    writer lease, so it can retire and close a whole generation only after
    every in-flight cursor/context has finished.  Readers remain concurrent.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            depth = self._reader_depths.get(thread_id, 0)
            if depth == 0 and self._writer_thread != thread_id:
                while self._writer_thread is not None or self._writers_waiting:
                    self._condition.wait()
            self._readers += 1
            self._reader_depths[thread_id] = depth + 1

    def release_read(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            depth = self._reader_depths.get(thread_id, 0)
            if depth <= 0:
                raise RuntimeError('SQLite connection reader lease is not held')
            if depth == 1:
                self._reader_depths.pop(thread_id, None)
            else:
                self._reader_depths[thread_id] = depth - 1
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if self._writer_thread == thread_id:
                self._writer_depth += 1
                return
            if self._reader_depths.get(thread_id, 0):
                raise RuntimeError('cannot close SQLiteStore from an active connection context')
            self._writers_waiting += 1
            try:
                while self._writer_thread is not None or self._readers:
                    self._condition.wait()
                self._writer_thread = thread_id
                self._writer_depth = 1
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            if self._writer_thread != thread_id:
                raise RuntimeError('SQLite connection writer lease is not held')
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer_thread = None
                self._condition.notify_all()


class _CursorLease:
    """Keep a store reader lease until a direct cursor is consumed/closed."""

    def __init__(self, cursor: sqlite3.Cursor, release: Callable[[], None]):
        self._cursor = cursor
        self._release = release
        self._released = False

    def _finish(self, *, close_cursor: bool = True) -> None:
        if self._released:
            return
        try:
            if close_cursor:
                self._cursor.close()
        finally:
            self._released = True
            self._release()

    def close(self) -> None:
        self._finish(close_cursor=True)

    def execute(self, *args, **kwargs):
        self._cursor.execute(*args, **kwargs)
        return self

    def executemany(self, *args, **kwargs):
        self._cursor.executemany(*args, **kwargs)
        return self

    def executescript(self, *args, **kwargs):
        self._cursor.executescript(*args, **kwargs)
        return self

    def fetchone(self):
        try:
            row = self._cursor.fetchone()
        except BaseException:
            self._finish()
            raise
        if row is None:
            self._finish()
        return row

    def fetchmany(self, size: int | None = None):
        try:
            rows = self._cursor.fetchmany() if size is None else self._cursor.fetchmany(size)
        except BaseException:
            self._finish()
            raise
        if not rows:
            self._finish()
        return rows

    def fetchall(self):
        try:
            return self._cursor.fetchall()
        finally:
            self._finish()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._cursor)
        except StopIteration:
            self._finish()
            raise
        except BaseException:
            self._finish()
            raise

    def __getattr__(self, name: str):
        return getattr(self._cursor, name)

    def __del__(self):  # pragma: no cover - deterministic on CPython, defensive elsewhere
        try:
            self._finish(close_cursor=True)
        except BaseException:
            pass


class _ConnectionProxy:
    """Per-call facade that resolves the current thread's live generation."""

    _INTERNAL_NAMES = {'_store', '_contexts', '_closed'}

    def __init__(self, store: 'SQLiteStore'):
        object.__setattr__(self, '_store', store)
        object.__setattr__(self, '_contexts', threading.local())
        object.__setattr__(self, '_closed', False)

    def _ensure_open(self) -> None:
        if self._closed:
            raise sqlite3.ProgrammingError('Cannot operate on a closed database.')

    def _call(self, name: str, *args, **kwargs):
        self._ensure_open()
        gate = self._store._connection_gate
        gate.acquire_read()
        try:
            conn = self._store._connection_for_current_thread()
            result = getattr(conn, name)(*args, **kwargs)
        except BaseException:
            gate.release_read()
            raise
        if isinstance(result, sqlite3.Cursor):
            return _CursorLease(result, gate.release_read)
        gate.release_read()
        return result

    def execute(self, *args, **kwargs):
        return self._call('execute', *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._call('executemany', *args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._call('executescript', *args, **kwargs)

    def cursor(self, *args, **kwargs):
        return self._call('cursor', *args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        contexts = getattr(self._contexts, 'stack', [])
        if contexts:
            raise RuntimeError('cannot close a SQLite connection from its active context')
        self._store._close_current_thread_connection()
        object.__setattr__(self, '_closed', True)

    def __enter__(self):
        self._ensure_open()
        gate = self._store._connection_gate
        gate.acquire_read()
        try:
            conn = self._store._connection_for_current_thread()
            conn.__enter__()
        except BaseException:
            gate.release_read()
            raise
        stack = getattr(self._contexts, 'stack', None)
        if stack is None:
            stack = []
            self._contexts.stack = stack
        stack.append(conn)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        stack = getattr(self._contexts, 'stack', None)
        if not stack:
            raise RuntimeError('SQLite connection context is not active')
        conn = stack.pop()
        try:
            return bool(conn.__exit__(exc_type, exc, tb))
        finally:
            self._store._connection_gate.release_read()

    def __getattr__(self, name: str):
        self._ensure_open()
        gate = self._store._connection_gate
        gate.acquire_read()
        try:
            conn = self._store._connection_for_current_thread()
            value = getattr(conn, name)
        finally:
            gate.release_read()
        if callable(value):
            return lambda *args, **kwargs: self._call(name, *args, **kwargs)
        return value

    def __setattr__(self, name: str, value) -> None:
        if name in self._INTERNAL_NAMES:
            object.__setattr__(self, name, value)
            return
        self._ensure_open()
        gate = self._store._connection_gate
        gate.acquire_read()
        try:
            setattr(self._store._connection_for_current_thread(), name, value)
        finally:
            gate.release_read()


class _OneShotConnectionProxy:
    """A one-shot handle whose reader lease lasts until explicit close."""

    _INTERNAL_NAMES = {'_store', '_connection', '_closed'}

    def __init__(self, store: 'SQLiteStore', connection: TrackedConnection):
        object.__setattr__(self, '_store', store)
        object.__setattr__(self, '_connection', connection)
        object.__setattr__(self, '_closed', False)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._connection.close()
        finally:
            object.__setattr__(self, '_closed', True)
            self._store._connection_gate.release_read()

    def __enter__(self):
        if self._closed:
            raise sqlite3.ProgrammingError('Cannot operate on a closed database.')
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            return bool(self._connection.__exit__(exc_type, exc, tb))
        finally:
            if not self._closed:
                object.__setattr__(self, '_closed', True)
                self._store._connection_gate.release_read()

    def __getattr__(self, name: str):
        if self._closed:
            raise sqlite3.ProgrammingError('Cannot operate on a closed database.')
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._INTERNAL_NAMES:
            object.__setattr__(self, name, value)
            return
        if self._closed:
            raise sqlite3.ProgrammingError('Cannot operate on a closed database.')
        setattr(self._connection, name, value)


def schema_migration_required_payload(exc: SchemaMigrationRequired | ReadOnlyStoreError) -> dict[str, Any]:
    """Normalize fail-closed search errors without exposing local paths."""

    detail = dict(exc.to_dict()['error'])
    reason_code = str(detail.pop('code'))
    detail['code'] = 'schema_migration_required'
    if reason_code != detail['code']:
        detail['reason_code'] = reason_code
    detail['action'] = {
        'schema_version_too_new': 'upgrade_runtime',
        'schema_version_mismatch': 'repair_schema_versions',
        'schema_preflight_unavailable': 'checkpoint_or_repair',
        'readonly_store_unavailable': 'checkpoint_or_repair',
    }.get(reason_code, 'run_schema_migration')
    return {'error': detail}


class SQLiteStore:
    def __init__(
        self,
        path: Path,
        *,
        readonly: bool = False,
        max_connections: int = 64,
        prepared_statement_cache_size: int = 128,
        page_cache_kib: int = 64_000,
        connection_wait_seconds: float = 1.0,
    ):
        if type(max_connections) is not int or max_connections < 1:
            raise ValueError('max_connections must be a positive integer')
        if type(prepared_statement_cache_size) is not int or not 0 <= prepared_statement_cache_size <= 4096:
            raise ValueError('prepared_statement_cache_size must be between 0 and 4096')
        if type(page_cache_kib) is not int or not 256 <= page_cache_kib <= 1024 * 1024:
            raise ValueError('page_cache_kib must be between 256 and 1048576')
        if not isinstance(connection_wait_seconds, (int, float)) or not 0 <= connection_wait_seconds <= 30:
            raise ValueError('connection_wait_seconds must be between 0 and 30 seconds')
        self.path = Path(path)
        self.readonly = bool(readonly)
        self.max_connections = max_connections
        self.prepared_statement_cache_size = prepared_statement_cache_size
        self.page_cache_kib = page_cache_kib
        self.connection_wait_seconds = float(connection_wait_seconds)
        self._local = threading.local()
        self._init_lock = threading.RLock()
        self._connections_lock = threading.RLock()
        self._connection_gate = _ConnectionGate()
        self._connections: set[TrackedConnection] = set()
        self._connection_slots = threading.BoundedSemaphore(max_connections)
        self._connection_open_count = 0
        self._connection_generation = 0
        self._process_id = os.getpid()
        self._initialized = False
        self._preflight_done = False
        self._count_cache: OrderedDict[str, tuple[tuple[Any, ...], int]] = OrderedDict()
        if hasattr(os, 'register_at_fork'):
            store_ref = weakref.ref(self)

            def reset_store_in_child() -> None:
                store = store_ref()
                if store is not None:
                    store._after_fork_child()

            os.register_at_fork(after_in_child=reset_store_in_child)

    def connect(self) -> sqlite3.Connection:
        """Return a reusable facade for this thread's current generation.

        The facade resolves the live handle for each operation.  This closes
        the small ``connect()``/``with`` race: a concurrent ``close_all`` may
        retire the old generation after this method returns, but the eventual
        operation will open and use only the new generation.
        """

        self._ensure_process_identity()
        self._connection_gate.acquire_read()
        try:
            self._connection_for_current_thread()
        finally:
            self._connection_gate.release_read()
        return _ConnectionProxy(self)  # type: ignore[return-value]

    def connect_once(self) -> sqlite3.Connection:
        """Return a one-shot connection whose lease lasts until ``close``."""

        self._ensure_process_identity()
        self._connection_gate.acquire_read()
        try:
            conn = self._open_tracked_connection(one_shot=True)
            return _OneShotConnectionProxy(self, conn)  # type: ignore[return-value]
        except BaseException:
            self._connection_gate.release_read()
            raise

    @property
    def active_connection_count(self) -> int:
        """Number of live SQLite handles owned by this store instance."""

        self._ensure_process_identity()
        with self._connections_lock:
            return sum(not conn._trove_closed for conn in self._connections)

    @property
    def connection_open_count(self) -> int:
        with self._connections_lock:
            return self._connection_open_count

    def _ensure_process_identity(self) -> None:
        if self._process_id != os.getpid():
            self._after_fork_child()

    def _after_fork_child(self) -> None:
        """Discard inherited handles and reset locks without acquiring them."""

        child_pid = os.getpid()
        if self._process_id == child_pid:
            return
        inherited = tuple(self._connections)
        self._process_id = child_pid
        self._local = threading.local()
        self._init_lock = threading.RLock()
        self._connections_lock = threading.RLock()
        self._connection_gate = _ConnectionGate()
        self._connections = set()
        self._connection_generation += 1
        self._initialized = False
        self._preflight_done = False
        self._count_cache = {}
        for conn in inherited:
            try:
                # The child owns an independent descriptor copy.  Closing it
                # cannot release the parent's handle, and prevents the child
                # from retaining WAL/read locks it must never reuse.
                conn._trove_on_close = None
                conn.close()
            except BaseException:
                pass
        self._connection_slots = threading.BoundedSemaphore(self.max_connections)
        self._connection_open_count = 0

    def _connection_for_current_thread(self) -> TrackedConnection:
        """Resolve/create a handle while the caller holds a reader lease."""

        self._ensure_process_identity()
        conn = getattr(self._local, 'connection', None)
        if (
            conn is not None
            and not conn._trove_closed
            and conn._trove_generation == self._connection_generation
        ):
            return conn
        if conn is not None and not conn._trove_closed:
            conn.close()
        self._local.connection = None
        conn = self._open_tracked_connection(one_shot=False)
        self._local.connection = conn
        return conn

    def _open_tracked_connection(self, *, one_shot: bool) -> TrackedConnection:
        if not self._connection_slots.acquire(timeout=self.connection_wait_seconds):
            raise SQLiteConnectionLimit()
        generation = self._connection_generation
        conn: TrackedConnection | None = None
        configured = False
        try:
            if self.readonly:
                factory = ReadOnlyClosingConnection if one_shot else ReadOnlyConnection
                conn = self._open_readonly_connection(factory)
            else:
                self._preflight_versions()
                factory = ClosingConnection if one_shot else ReusableConnection
                conn = sqlite3.connect(
                    self.path,
                    factory=factory,
                    check_same_thread=False,
                    cached_statements=self.prepared_statement_cache_size,
                )
            assert isinstance(conn, TrackedConnection)
            conn._trove_configure(generation=generation, on_close=self._unregister_connection)
            configured = True
            conn.row_factory = sqlite3.Row
            if self.readonly:
                self._preflight_readonly_connection(conn)
            self._apply_connection_pragmas(conn)
            with self._connections_lock:
                self._connections.add(conn)
                self._connection_open_count += 1
            return conn
        except BaseException as exc:
            if conn is not None and configured:
                conn.close()
            else:
                self._connection_slots.release()
            if self.readonly and isinstance(exc, sqlite3.Error):
                raise ReadOnlyStoreError() from exc
            raise

    def _unregister_connection(self, connection: sqlite3.Connection) -> None:
        with self._connections_lock:
            self._connections.discard(connection)  # type: ignore[arg-type]
        self._connection_slots.release()
        if getattr(self._local, 'connection', None) is connection:
            self._local.connection = None

    def _close_current_thread_connection(self) -> None:
        self._ensure_process_identity()
        self._connection_gate.acquire_read()
        try:
            conn = getattr(self._local, 'connection', None)
            if conn is not None:
                conn.close()
                self._local.connection = None
        finally:
            self._connection_gate.release_read()

    def _readonly_uri(self, *, immutable: bool | None = None) -> str:
        if immutable is None:
            immutable = not any(
                Path(f'{self.path}{suffix}').exists()
                for suffix in ('-wal', '-shm')
            )
        uri = self.path.expanduser().resolve().as_uri() + '?mode=ro'
        return uri + '&immutable=1' if immutable else uri

    def _open_readonly_connection(self, factory) -> sqlite3.Connection:
        """Bind a read connection to the inode validated by the schema cache."""

        identity_before = schema_file_identity(self.path)
        # A fully checkpointed published generation has no WAL/SHM sidecars.
        # Opening that snapshot with ``immutable=1`` keeps a read-only process
        # from creating empty write-side artifacts merely by searching.  Live
        # WAL databases still use ordinary ``mode=ro`` so committed WAL frames
        # remain visible and schema preflight stays coherent.
        sidecars = (Path(f'{self.path}-wal'), Path(f'{self.path}-shm'))
        immutable_snapshot = not any(sidecar.exists() for sidecar in sidecars)
        try:
            conn = sqlite3.connect(
                self._readonly_uri(immutable=immutable_snapshot),
                uri=True,
                factory=factory,
                check_same_thread=False,
                cached_statements=self.prepared_statement_cache_size,
            )
        except sqlite3.Error as exc:
            raise ReadOnlyStoreError() from exc
        try:
            identity_after = schema_file_identity(self.path)
            if identity_before != identity_after:
                raise SchemaPreflightUnavailable(
                    0,
                    SCHEMA_VERSION,
                    reason='database changed while its read connection was opened',
                )
            if immutable_snapshot and any(sidecar.exists() for sidecar in sidecars):
                raise SchemaPreflightUnavailable(
                    0,
                    SCHEMA_VERSION,
                    reason='database sidecars changed while its read connection was opened',
                )
            conn._trove_database_identity = identity_after  # type: ignore[attr-defined]
            return conn
        except BaseException:
            conn.close()
            raise

    def _preflight_readonly_connection(self, conn: sqlite3.Connection) -> None:
        """Version-check the same coherent read connection used by the query."""

        try:
            preflight_connection_versions(conn)
        except SchemaMigrationRequired:
            raise
        except sqlite3.DatabaseError as exc:
            raise SchemaPreflightUnavailable(
                0,
                SCHEMA_VERSION,
                reason='database schema version cannot be read safely',
            ) from exc
        self._preflight_done = True

    def _preflight_versions(self) -> None:
        with self._init_lock:
            if self._preflight_done or not self.path.exists():
                return
            preflight_database_versions(self.path)
            self._preflight_done = True

    def close(self) -> None:
        """Close every connection generation owned by this store.

        Historically this method closed only the caller thread's local
        connection.  Store invalidation is process-wide, so ``close`` now has
        the same explicit all-thread semantics as ``close_all``.
        """

        self.close_all()

    def close_all(self) -> None:
        """Retire the generation and close all handles after readers finish."""

        self._ensure_process_identity()
        self._connection_gate.acquire_write()
        try:
            self._connection_generation += 1
            with self._connections_lock:
                connections = tuple(self._connections)
                self._connections.clear()
            self._local.connection = None
            for conn in connections:
                conn.close()
            self._initialized = False
            self._preflight_done = False
            self._count_cache.clear()
        finally:
            self._connection_gate.release_write()

    def _apply_connection_pragmas(self, conn: sqlite3.Connection) -> None:
        if self.readonly:
            conn.execute('PRAGMA query_only=ON')
            conn.execute(f'PRAGMA cache_size=-{self.page_cache_kib}')
            conn.execute('PRAGMA mmap_size=268435456')
            conn.execute('PRAGMA temp_store=MEMORY')
            return
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute(f'PRAGMA cache_size=-{self.page_cache_kib}')
        conn.execute('PRAGMA mmap_size=268435456')
        conn.execute('PRAGMA temp_store=MEMORY')

    def initialize(self) -> None:
        if self._initialized and self.path.exists():
            return
        if self.readonly:
            if not self.path.is_file():
                raise SchemaMigrationRequired(0, SCHEMA_VERSION, reason='database does not exist')
            with self._init_lock:
                if self._initialized:
                    return
                try:
                    with self.connect() as conn:
                        validate_schema_cached(
                            conn,
                            self.path,
                            expected_identity=getattr(conn, '_trove_database_identity', None),
                        )
                        self._initialized = True
                except BaseException:
                    self.close()
                    raise
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._init_lock:
            if self._initialized and self.path.exists():
                return
            with self.connect() as conn:
                migrate_schema(conn)
                mode = conn.execute('PRAGMA journal_mode=WAL').fetchone()
                if mode is None or str(mode[0]).lower() != 'wal':
                    raise sqlite3.OperationalError('failed to preserve SQLite WAL mode')
                validate_schema(conn)
                self._initialized = True

    def rebuild_fts(self, *, fault_injector=None) -> dict[str, Any]:
        self.initialize()
        start = datetime.now(timezone.utc)
        with self.connect() as conn:
            report = rebuild_fts_transaction(conn, fault_injector=fault_injector)
        message_rows = report['message_rows']
        chunk_rows = report['chunk_rows']
        self._initialized = True
        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return {
            'ok': True,
            'fts_tokenizer': FTS_TOKENIZER_VERSION,
            'message_rows': message_rows,
            'chunk_rows': chunk_rows,
            'elapsed_ms': round(elapsed_ms, 3),
        }

    def backfill_message_content_kind(self, *, limit: int | None = None) -> dict[str, Any]:
        """Classify existing message rows and replace non-text noise with placeholders."""
        self.initialize()
        scanned = 0
        updated = 0
        dirty_refs: list[dict[str, str]] = []
        kind_counts: dict[str, int] = {}
        payload_counts: dict[str, int] = {}
        payloads_changed = 0
        query = 'SELECT id, citation, account_id, conversation_id, source_type, content, content_kind FROM messages ORDER BY id'
        if limit:
            query += f' LIMIT {int(limit)}'
        with self.connect() as conn:
            rows = list(conn.execute(query))
            for row in rows:
                scanned += 1
                existing_kind = str(row['content_kind'] or 'text')
                existing_display = display_content_for_kind(row['content'], existing_kind)
                if existing_kind != 'text' and str(row['content'] or '') == existing_display:
                    kind = existing_kind
                else:
                    kind = classify_content_kind(row['content'])
                parsed_payload = None
                if kind == 'appmsg' and not str(row['content'] or '').startswith('[appmsg/') and str(row['content'] or '') != '[appmsg]':
                    parsed_payload = parse_appmsg(str(row['content'] or ''))
                    display = parsed_payload.display_text
                    payload_counts[parsed_payload.parse_status] = payload_counts.get(parsed_payload.parse_status, 0) + 1
                else:
                    display = display_content_for_kind(row['content'], kind)
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                if row['content_kind'] != kind or row['content'] != display:
                    conn.execute('UPDATE messages SET content_kind=?, content=? WHERE id=?', (kind, display, row['id']))
                    dirty_refs.append({
                        'citation': str(row['citation']),
                        'account_id': str(row['account_id'] or ''),
                        'conversation_id': str(row['conversation_id'] or ''),
                        'source_type': str(row['source_type'] or 'message'),
                    })
                    updated += 1
                if parsed_payload is not None:
                    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                    cursor = conn.execute(
                        _MESSAGE_PAYLOAD_UPSERT_SQL,
                        _normalized_payload_values(str(row['citation']), parsed_payload.to_dict(), timestamp=timestamp),
                    )
                    payloads_changed += max(cursor.rowcount, 0)
            stale_refs = self._stale_non_text_message_chunk_refs_conn(conn)
            if stale_refs:
                seen_dirty = {ref['citation'] for ref in dirty_refs}
                for ref in stale_refs:
                    if ref['citation'] not in seen_dirty:
                        dirty_refs.append(ref)
                        seen_dirty.add(ref['citation'])
            chunk_report = self._rebuild_message_chunks_for_citations_conn(
                conn,
                [ref['citation'] for ref in dirty_refs],
            ) if dirty_refs else {'parents': 0, 'chunks': 0, 'citations': 0, 'deleted_chunks': 0, 'deleted_vectors': 0}
            dirty_recorded = self._record_dirty_refs_conn(conn, dirty_refs)
            conn.commit()
        return {
            'ok': True,
            'scanned': scanned,
            'updated': updated,
            'kind_counts': kind_counts,
            'payload_status_counts': payload_counts,
            'payloads_changed': payloads_changed,
            'dirty_recorded': dirty_recorded,
            'chunks': chunk_report,
            'stale_chunk_parents': len(stale_refs),
            'raw_content_included': False,
        }

    def appmsg_payload_reprocess_plan(self, messages: Iterable[Message]) -> dict[str, Any]:
        """Compare normalized source AppMsg payloads with indexed rows without writing."""
        self.initialize()
        candidates = list({message.citation: message for message in messages if message.normalized_payload is not None}.values())
        status_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        matched = 0
        would_change = 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.connect() as conn:
            for message in candidates:
                values = _message_payload_values(message, timestamp=now)
                if values is None:
                    continue
                status_counts[str(values[3])] = status_counts.get(str(values[3]), 0) + 1
                type_counts[str(values[2])] = type_counts.get(str(values[2]), 0) + 1
                current_message = conn.execute(
                    'SELECT content,content_kind FROM messages WHERE citation=?', (message.citation,),
                ).fetchone()
                if current_message is None:
                    continue
                matched += 1
                current_payload = conn.execute(
                    """SELECT appmsg_type,normalized_type,parse_status,normalized_json,display_text,
                              source_hash,parser_version,unsupported_reason
                         FROM message_payloads WHERE citation=?""",
                    (message.citation,),
                ).fetchone()
                desired = values[1:8] + (values[8],)
                if (
                    str(current_message['content'] or '') != str(values[5])
                    or str(current_message['content_kind'] or '') != 'appmsg'
                    or current_payload is None
                    or tuple(current_payload) != desired
                ):
                    would_change += 1
        return {
            'ok': True,
            'source_payloads': len(candidates),
            'matched_messages': matched,
            'missing_messages': len(candidates) - matched,
            'would_change': would_change,
            'parse_status_counts': dict(sorted(status_counts.items())),
            'normalized_type_counts': dict(sorted(type_counts.items())),
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def reprocess_appmsg_payloads(self, messages: Iterable[Message]) -> dict[str, Any]:
        """Persist only normalized AppMsg fields for citations already in the Vault."""
        self.initialize()
        candidates = list({message.citation: message for message in messages if message.normalized_payload is not None}.values())
        plan = self.appmsg_payload_reprocess_plan(candidates)
        changed_refs: list[dict[str, str]] = []
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.connect() as conn:
            for message in candidates:
                values = _message_payload_values(message, timestamp=now)
                if values is None:
                    continue
                current = conn.execute(
                    'SELECT account_id,conversation_id,source_type FROM messages WHERE citation=?',
                    (message.citation,),
                ).fetchone()
                if current is None:
                    continue
                before = conn.total_changes
                conn.execute(
                    """UPDATE messages SET content=?,content_kind='appmsg'
                       WHERE citation=? AND (content IS NOT ? OR content_kind IS NOT 'appmsg')""",
                    (values[5], message.citation, values[5]),
                )
                conn.execute(_MESSAGE_PAYLOAD_UPSERT_SQL, values)
                if conn.total_changes > before:
                    changed_refs.append({
                        'citation': message.citation,
                        'account_id': str(current['account_id'] or ''),
                        'conversation_id': str(current['conversation_id'] or ''),
                        'source_type': str(current['source_type'] or 'message'),
                    })
            chunks = self._rebuild_message_chunks_for_citations_conn(
                conn, [ref['citation'] for ref in changed_refs],
            ) if changed_refs else {'parents': 0, 'chunks': 0, 'citations': 0, 'deleted_chunks': 0, 'deleted_vectors': 0}
            dirty_recorded = self._record_dirty_refs_conn(conn, changed_refs)
            conn.commit()
        return {
            **plan,
            'changed': len(changed_refs),
            'would_change': 0,
            'dirty_recorded': dirty_recorded,
            'chunks': chunks,
        }

    def schema_version(self) -> int:
        if not self.path.exists():
            return 0
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone() if self._table_exists(conn, 'schema_meta') else None
            if row is not None:
                return int(row[0])
            return int(conn.execute('PRAGMA user_version').fetchone()[0])

    def _table_exists(self, conn, table: str) -> bool:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None

    def counts(self) -> dict[str, int]:
        if not self.path.exists():
            return {'accounts': 0, 'conversations': 0, 'messages': 0, 'chunks': 0}
        with self.connect() as conn:
            def count(table: str) -> int:
                if not self._table_exists(conn, table):
                    return 0
                return int(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
            chunk_count = count('evidence_chunks') or count('message_fts')
            return {
                'accounts': count('accounts'),
                'conversations': count('conversations'),
                'messages': count('messages'),
                'chunks': chunk_count,
            }

    def upsert_accounts(self, accounts: Iterable[Account]) -> None:
        self.initialize()
        with self.connect() as conn:
            conn.executemany(
                'INSERT OR REPLACE INTO accounts(account_id,label,display_name) VALUES(?,?,?)',
                [(a.account_id, a.label, a.display_name) for a in accounts],
            )
            conn.commit()

    def upsert_conversations(self, conversations: Iterable[Conversation]) -> None:
        self.initialize()
        with self.connect() as conn:
            conn.executemany(
                'INSERT OR REPLACE INTO conversations(conversation_id,account_id,title,type,member_count) VALUES(?,?,?,?,?)',
                [(c.conversation_id, c.account_id, c.title, c.type, c.member_count) for c in conversations],
            )
            conn.commit()

    def upsert_messages(self, messages: Iterable[Message]) -> int:
        self.initialize()
        changed = 0
        payload_changed = 0
        with self.connect() as conn:
            upsert_sql = """INSERT INTO messages(citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
               sender_id,sender_name,timestamp,content,content_kind,shard_id,local_id,sent_by_me,source_type,direction)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(account_id,conversation_id,shard_id,local_id) DO UPDATE SET
               citation=excluded.citation,account_label=excluded.account_label,conversation_title=excluded.conversation_title,
               conversation_type=excluded.conversation_type,sender_id=excluded.sender_id,content=excluded.content,
               content_kind=excluded.content_kind,
               timestamp=excluded.timestamp,sender_name=excluded.sender_name,sent_by_me=excluded.sent_by_me,
               direction=excluded.direction,source_type=excluded.source_type
               WHERE messages.citation IS NOT excluded.citation
                  OR messages.account_label IS NOT excluded.account_label
                  OR messages.conversation_title IS NOT excluded.conversation_title
                  OR messages.conversation_type IS NOT excluded.conversation_type
                  OR messages.sender_id IS NOT excluded.sender_id
                  OR messages.content IS NOT excluded.content
                  OR messages.content_kind IS NOT excluded.content_kind
                  OR messages.timestamp IS NOT excluded.timestamp
                  OR messages.sender_name IS NOT excluded.sender_name
                  OR messages.sent_by_me IS NOT excluded.sent_by_me
                  OR messages.direction IS NOT excluded.direction
                  OR messages.source_type IS NOT excluded.source_type"""
            for batch in _message_batches(messages, size=1000):
                values: list[tuple[Any, ...]] = []
                for m in batch:
                    data = m.safe_dict()
                    values.append((
                        m.citation,
                        m.account_id,
                        m.account_label,
                        m.conversation_id,
                        m.conversation_title,
                        m.conversation_type,
                        m.sender_id,
                        m.sender_name,
                        data['timestamp'],
                        m.content,
                        m.content_kind,
                        m.shard_id,
                        m.local_id,
                        int(m.sent_by_me),
                        m.source_type,
                        m.direction,
                    ))
                cursor = conn.executemany(upsert_sql, values)
                changed += max(cursor.rowcount, 0)
                now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                for message in batch:
                    payload = _message_payload_values(message, timestamp=now)
                    if payload is not None:
                        cursor = conn.execute(_MESSAGE_PAYLOAD_UPSERT_SQL, payload)
                        payload_changed += max(cursor.rowcount, 0)
                    elif message.content_kind != 'appmsg':
                        cursor = conn.execute('DELETE FROM message_payloads WHERE citation=?', (message.citation,))
                        payload_changed += max(cursor.rowcount, 0)
            conn.commit()
        return changed + payload_changed

    def apply_message_delta(
        self,
        accounts: Iterable[Account],
        conversations: Iterable[Conversation],
        messages: Iterable[Message],
        *,
        deleted_citations: Iterable[str] = (),
        source_key: str | None = None,
        source_snapshot_complete: bool = False,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        """Atomically apply the message projection delta and its dependants.

        The input is a *delta*, not a conversation snapshot.  Ordinary message
        edits therefore refresh only their citation.  A whole conversation is
        touched only when title/type metadata (which is denormalized into each
        child message and chunk) actually changes.  Deletions are explicit
        citation tombstones so downstream vector stores can remove stale rows.

        The returned counters intentionally describe bounded semantic work,
        rather than SQLite trigger internals.  They are stable enough for the
        10k/100k complexity gates and make an identical second application a
        zero-commit, zero-WAL operation.
        """
        self.initialize()
        account_rows = list({item.account_id: item for item in accounts}.values())
        conversation_rows = list({(item.account_id, item.conversation_id): item for item in conversations}.values())
        message_rows = list({
            (item.account_id, item.conversation_id, item.shard_id, int(item.local_id)): item
            for item in messages
        }.values())
        payload_rows = {
            message.citation: message
            for message in message_rows
            if message.normalized_payload is not None
        }
        tombstone_inputs = list(dict.fromkeys(str(value) for value in deleted_citations if value))
        wal_path = Path(f'{self.path}-wal')
        wal_before = wal_path.stat().st_size if wal_path.exists() else 0
        sql_statements = 0
        rows_scanned = len(message_rows) + len(tombstone_inputs)
        rows_written = 0
        direct_refs: dict[str, dict[str, str]] = {}
        old_citations: dict[str, dict[str, str]] = {}
        profile_identity_values: set[str] = set()
        profile_scope_changed = False

        def record_profile_identities(*values: Any) -> None:
            profile_identity_values.update(
                text for value in values if (text := str(value or '').strip())
            )

        metadata_account_ids: set[str] = set()
        metadata_conversation_keys: set[tuple[str, str]] = set()
        chunk_report: dict[str, Any] = {
            'parents': 0,
            'chunks': 0,
            'citations': 0,
            'deleted_chunks': 0,
            'deleted_vectors': 0,
            'max_chars': max_chars,
            'overlap_chars': overlap_chars,
        }
        dirty_recorded = 0
        source_rows_added = 0
        source_rows_deleted = 0
        derived_source_tombstones: list[str] = []
        durable_changed = False

        with self.connect() as conn:
            # Metadata comparisons are index probes over the supplied delta.
            for account in account_rows:
                sql_statements += 1
                existing = conn.execute(
                    'SELECT label,display_name FROM accounts WHERE account_id=?',
                    (account.account_id,),
                ).fetchone()
                if existing is not None and str(existing['label']) != account.label:
                    metadata_account_ids.add(account.account_id)
            for conversation in conversation_rows:
                sql_statements += 1
                existing = conn.execute(
                    'SELECT title,type,member_count FROM conversations WHERE account_id=? AND conversation_id=?',
                    (conversation.account_id, conversation.conversation_id),
                ).fetchone()
                if existing is not None and (
                    str(existing['title']) != conversation.title
                    or str(existing['type']) != conversation.type
                ):
                    metadata_conversation_keys.add((conversation.account_id, conversation.conversation_id))

            conn.execute(
                """CREATE TEMP TABLE IF NOT EXISTS _trove_message_delta_batch(
                    citation TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    account_label TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    conversation_title TEXT NOT NULL,
                    conversation_type TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_kind TEXT NOT NULL,
                    shard_id TEXT NOT NULL,
                    local_id INTEGER NOT NULL,
                    sent_by_me INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    PRIMARY KEY(account_id,conversation_id,shard_id,local_id)
                ) WITHOUT ROWID"""
            )
            conn.execute(
                """CREATE TEMP TABLE IF NOT EXISTS _trove_message_payload_keep(
                    citation TEXT PRIMARY KEY
                ) WITHOUT ROWID"""
            )
            conn.execute('DELETE FROM _trove_message_delta_batch')
            conn.execute('DELETE FROM _trove_message_payload_keep')
            if message_rows:
                conn.executemany(
                    """INSERT INTO _trove_message_delta_batch(
                        citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
                        sender_id,sender_name,timestamp,content,content_kind,shard_id,local_id,sent_by_me,source_type,direction
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [_message_value_tuple(item) for item in message_rows],
                )
            if payload_rows:
                conn.executemany(
                    'INSERT INTO _trove_message_payload_keep(citation) VALUES(?)',
                    [(citation,) for citation in payload_rows],
                )
            if message_rows:
                sql_statements += 1
                for row in conn.execute(
                    """SELECT b.citation,b.account_id,b.conversation_id,b.conversation_title,
                              b.sender_id,b.sender_name,m.citation AS old_citation,
                              m.conversation_id AS old_conversation_id,
                              m.conversation_title AS old_conversation_title,
                              m.sender_id AS old_sender_id,m.sender_name AS old_sender_name
                       FROM _trove_message_delta_batch b
                       LEFT JOIN messages m
                         ON m.account_id=b.account_id
                        AND m.conversation_id=b.conversation_id
                        AND m.shard_id=b.shard_id
                        AND m.local_id=b.local_id
                       WHERE m.id IS NULL
                          OR m.citation IS NOT b.citation
                          OR m.account_label IS NOT b.account_label
                          OR m.conversation_title IS NOT b.conversation_title
                          OR m.conversation_type IS NOT b.conversation_type
                          OR m.sender_id IS NOT b.sender_id
                          OR m.content IS NOT b.content
                          OR m.content_kind IS NOT b.content_kind
                          OR m.timestamp IS NOT b.timestamp
                          OR m.sender_name IS NOT b.sender_name
                          OR m.sent_by_me IS NOT b.sent_by_me
                          OR m.direction IS NOT b.direction
                          OR m.source_type IS NOT b.source_type"""
                ):
                    profile_scope_changed = True
                    record_profile_identities(
                        row['conversation_id'], row['conversation_title'],
                        row['sender_id'], row['sender_name'],
                        row['old_conversation_id'], row['old_conversation_title'],
                        row['old_sender_id'], row['old_sender_name'],
                    )
                    citation = str(row['citation'])
                    direct_refs[citation] = {
                        'citation': citation,
                        'account_id': str(row['account_id']),
                        'conversation_id': str(row['conversation_id']),
                        'source_type': 'message',
                    }
                    if row['old_citation'] and str(row['old_citation']) != citation:
                        old = str(row['old_citation'])
                        old_citations[old] = {
                            'citation': old,
                            'account_id': str(row['account_id']),
                            'conversation_id': str(row['conversation_id']),
                            'source_type': 'message',
                        }

            if source_key:
                source_key = str(source_key)
                stale_source_citations: list[str] = []
                if source_snapshot_complete:
                    sql_statements += 1
                    stale_source_citations = [
                        str(row['citation'])
                        for row in conn.execute(
                            """SELECT s.citation
                               FROM sync_message_source_rows s
                               LEFT JOIN _trove_message_delta_batch b ON b.citation=s.citation
                               WHERE s.source_key=? AND b.citation IS NULL
                               ORDER BY s.citation""",
                            (source_key,),
                        )
                    ]
                    rows_scanned += len(stale_source_citations)
                    sql_statements += 1
                    cursor = conn.execute(
                        """DELETE FROM sync_message_source_rows
                           WHERE source_key=?
                             AND citation NOT IN (SELECT citation FROM _trove_message_delta_batch)""",
                        (source_key,),
                    )
                    source_rows_deleted = max(cursor.rowcount, 0)
                    rows_written += source_rows_deleted
                    durable_changed = durable_changed or bool(source_rows_deleted)
                if message_rows:
                    sql_statements += 1
                    cursor = conn.execute(
                        """INSERT OR IGNORE INTO sync_message_source_rows(source_key,citation,updated_at)
                           SELECT ?,citation,? FROM _trove_message_delta_batch""",
                        (source_key, datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')),
                    )
                    source_rows_added = max(cursor.rowcount, 0)
                    rows_written += source_rows_added
                    durable_changed = durable_changed or bool(source_rows_added)
                # A citation is deleted only after its last authoritative
                # source releases ownership.
                for citation in stale_source_citations:
                    sql_statements += 1
                    if conn.execute(
                        'SELECT 1 FROM sync_message_source_rows WHERE citation=? LIMIT 1',
                        (citation,),
                    ).fetchone() is None:
                        derived_source_tombstones.append(citation)

            if account_rows:
                sql_statements += 1
                cursor = conn.executemany(
                    """INSERT INTO accounts(account_id,label,display_name) VALUES(?,?,?)
                       ON CONFLICT(account_id) DO UPDATE SET
                         label=excluded.label,display_name=excluded.display_name
                       WHERE accounts.label IS NOT excluded.label
                          OR accounts.display_name IS NOT excluded.display_name""",
                    [(item.account_id, item.label, item.display_name) for item in account_rows],
                )
                count = max(cursor.rowcount, 0)
                rows_written += count
                durable_changed = durable_changed or bool(count)

                now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                for citation, message in payload_rows.items():
                    values = _message_payload_values(message, timestamp=now)
                    if values is None:
                        continue
                    existing = conn.execute(
                        """SELECT appmsg_type,normalized_type,parse_status,normalized_json,display_text,
                                  source_hash,parser_version,unsupported_reason
                             FROM message_payloads WHERE citation=?""",
                        (citation,),
                    ).fetchone()
                    comparable = values[1:8] + (values[8],)
                    existing_values = tuple(existing) if existing is not None else None
                    cursor = conn.execute(_MESSAGE_PAYLOAD_UPSERT_SQL, values)
                    count = max(cursor.rowcount, 0)
                    if existing_values != comparable:
                        profile_scope_changed = True
                        record_profile_identities(
                            message.conversation_id,
                            message.conversation_title,
                            message.sender_id,
                            message.sender_name,
                        )
                        direct_refs[citation] = {
                            'citation': citation,
                            'account_id': message.account_id,
                            'conversation_id': message.conversation_id,
                            'source_type': 'message',
                        }
                    rows_written += count
                    durable_changed = durable_changed or bool(count)
                if message_rows:
                    sql_statements += 1
                    cursor = conn.execute(
                        """DELETE FROM message_payloads
                           WHERE citation IN (
                               SELECT batch.citation
                               FROM _trove_message_delta_batch AS batch
                               LEFT JOIN _trove_message_payload_keep AS keep
                                 ON keep.citation=batch.citation
                               WHERE batch.content_kind<>'appmsg'
                                 AND keep.citation IS NULL
                           )"""
                    )
                    count = max(cursor.rowcount, 0)
                    rows_written += count
                    durable_changed = durable_changed or bool(count)
            if conversation_rows:
                sql_statements += 1
                cursor = conn.executemany(
                    """INSERT INTO conversations(conversation_id,account_id,title,type,member_count) VALUES(?,?,?,?,?)
                       ON CONFLICT(account_id,conversation_id) DO UPDATE SET
                         title=excluded.title,type=excluded.type,member_count=excluded.member_count
                       WHERE conversations.title IS NOT excluded.title
                          OR conversations.type IS NOT excluded.type
                          OR conversations.member_count IS NOT excluded.member_count""",
                    [(item.conversation_id, item.account_id, item.title, item.type, item.member_count) for item in conversation_rows],
                )
                count = max(cursor.rowcount, 0)
                rows_written += count
                durable_changed = durable_changed or bool(count)

            if message_rows:
                sql_statements += 1
                cursor = conn.execute(
                    """INSERT INTO messages(citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
                           sender_id,sender_name,timestamp,content,content_kind,shard_id,local_id,sent_by_me,source_type,direction)
                       SELECT citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
                           sender_id,sender_name,timestamp,content,content_kind,shard_id,local_id,sent_by_me,source_type,direction
                       FROM _trove_message_delta_batch WHERE 1
                       ON CONFLICT(account_id,conversation_id,shard_id,local_id) DO UPDATE SET
                         citation=excluded.citation,account_label=excluded.account_label,conversation_title=excluded.conversation_title,
                         conversation_type=excluded.conversation_type,sender_id=excluded.sender_id,content=excluded.content,
                         content_kind=excluded.content_kind,timestamp=excluded.timestamp,sender_name=excluded.sender_name,
                         sent_by_me=excluded.sent_by_me,direction=excluded.direction,source_type=excluded.source_type
                       WHERE messages.citation IS NOT excluded.citation
                          OR messages.account_label IS NOT excluded.account_label
                          OR messages.conversation_title IS NOT excluded.conversation_title
                          OR messages.conversation_type IS NOT excluded.conversation_type
                          OR messages.sender_id IS NOT excluded.sender_id
                          OR messages.content IS NOT excluded.content
                          OR messages.content_kind IS NOT excluded.content_kind
                          OR messages.timestamp IS NOT excluded.timestamp
                          OR messages.sender_name IS NOT excluded.sender_name
                          OR messages.sent_by_me IS NOT excluded.sent_by_me
                          OR messages.direction IS NOT excluded.direction
                          OR messages.source_type IS NOT excluded.source_type"""
                )
                count = max(cursor.rowcount, 0)
                rows_written += count
                durable_changed = durable_changed or bool(count)

            # Only denormalized metadata changes fan out to all children.
            metadata_refs: dict[str, dict[str, str]] = {}
            for account_id in sorted(metadata_account_ids):
                account = next(item for item in account_rows if item.account_id == account_id)
                sql_statements += 2
                children = list(conn.execute(
                    'SELECT citation,conversation_id FROM messages WHERE account_id=?',
                    (account_id,),
                ))
                rows_scanned += len(children)
                for child in children:
                    citation = str(child['citation'])
                    metadata_refs[citation] = {
                        'citation': citation,
                        'account_id': account_id,
                        'conversation_id': str(child['conversation_id']),
                        'source_type': 'message',
                    }
                cursor = conn.execute(
                    'UPDATE messages SET account_label=? WHERE account_id=? AND account_label IS NOT ?',
                    (account.label, account_id, account.label),
                )
                count = max(cursor.rowcount, 0)
                rows_written += count
                durable_changed = durable_changed or bool(count)
            for account_id, conversation_id in sorted(metadata_conversation_keys):
                conversation = next(
                    item for item in conversation_rows
                    if item.account_id == account_id and item.conversation_id == conversation_id
                )
                sql_statements += 2
                children = list(conn.execute(
                    """SELECT citation,conversation_id,conversation_title,sender_id,sender_name
                         FROM messages WHERE account_id=? AND conversation_id=?""",
                    (account_id, conversation_id),
                ))
                rows_scanned += len(children)
                for child in children:
                    profile_scope_changed = True
                    record_profile_identities(
                        child['conversation_id'], child['conversation_title'],
                        child['sender_id'], child['sender_name'],
                    )
                    citation = str(child['citation'])
                    metadata_refs[citation] = {
                        'citation': citation,
                        'account_id': account_id,
                        'conversation_id': conversation_id,
                        'source_type': 'message',
                    }
                cursor = conn.execute(
                    """UPDATE messages SET conversation_title=?,conversation_type=?
                       WHERE account_id=? AND conversation_id=?
                         AND (conversation_title IS NOT ? OR conversation_type IS NOT ?)""",
                    (conversation.title, conversation.type, account_id, conversation_id, conversation.title, conversation.type),
                )
                count = max(cursor.rowcount, 0)
                rows_written += count
                durable_changed = durable_changed or bool(count)

            tombstone_refs = dict(old_citations)
            requested_tombstones = list(dict.fromkeys([*tombstone_inputs, *derived_source_tombstones, *old_citations]))
            for start in range(0, len(requested_tombstones), 500):
                batch = requested_tombstones[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                sql_statements += 2
                for row in conn.execute(
                    f"""SELECT citation,account_id,conversation_id,conversation_title,
                               sender_id,sender_name,source_type
                          FROM messages WHERE citation IN ({placeholders})""",
                    batch,
                ):
                    profile_scope_changed = True
                    record_profile_identities(
                        row['conversation_id'], row['conversation_title'],
                        row['sender_id'], row['sender_name'],
                    )
                    citation = str(row['citation'])
                    tombstone_refs[citation] = {
                        'citation': citation,
                        'account_id': str(row['account_id'] or ''),
                        'conversation_id': str(row['conversation_id'] or ''),
                        'source_type': str(row['source_type'] or 'message'),
                    }
                cursor = conn.execute(f'DELETE FROM messages WHERE citation IN ({placeholders})', batch)
                count = max(cursor.rowcount, 0)
                rows_written += count
                durable_changed = durable_changed or bool(count)

            now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            inserted_tombstones: dict[str, dict[str, str]] = {}
            if requested_tombstones:
                sql_statements += 1
                for citation in requested_tombstones:
                    ref = tombstone_refs.get(citation) or {
                        'citation': citation,
                        'account_id': '',
                        'conversation_id': '',
                        'source_type': 'message',
                    }
                    cursor = conn.execute(
                        """INSERT INTO sync_citation_tombstones(citation,account_id,conversation_id,source_type,deleted_at)
                           VALUES(?,?,?,?,?) ON CONFLICT(citation) DO NOTHING""",
                        (citation, ref['account_id'], ref['conversation_id'], ref['source_type'], now),
                    )
                    count = max(cursor.rowcount, 0)
                    if count:
                        inserted_tombstones[citation] = ref
                        rows_written += count
                        durable_changed = True

            live_refs = {**direct_refs, **metadata_refs}
            if live_refs:
                # Re-introducing a citation clears a prior deletion marker.
                live = list(live_refs)
                for start in range(0, len(live), 500):
                    batch = live[start:start + 500]
                    placeholders = ','.join('?' for _ in batch)
                    sql_statements += 1
                    cursor = conn.execute(
                        f'DELETE FROM sync_citation_tombstones WHERE citation IN ({placeholders})',
                        batch,
                    )
                    count = max(cursor.rowcount, 0)
                    rows_written += count
                    durable_changed = durable_changed or bool(count)

            refresh_refs = {**live_refs, **inserted_tombstones}
            if refresh_refs:
                sql_statements += 1
                chunk_report = self._rebuild_message_chunks_for_citations_conn(
                    conn,
                    refresh_refs,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
                chunk_writes = (
                    int(chunk_report.get('chunks') or 0)
                    + int(chunk_report.get('deleted_chunks') or 0)
                    + int(chunk_report.get('deleted_vectors') or 0)
                )
                rows_written += chunk_writes
                durable_changed = durable_changed or bool(chunk_writes)
                sql_statements += 1
                dirty_recorded = self._record_dirty_refs_conn(conn, refresh_refs.values())
                rows_written += dirty_recorded
                durable_changed = durable_changed or bool(dirty_recorded)

            conn.execute('DELETE FROM _trove_message_delta_batch')
            conn.execute('DELETE FROM _trove_message_payload_keep')
            if durable_changed:
                conn.commit()
                commits = 1
            else:
                # Discard TEMP-table work without publishing a generation.
                conn.rollback()
                commits = 0

        wal_after = wal_path.stat().st_size if wal_path.exists() else 0
        changed_conversations = {
            (ref['account_id'], ref['conversation_id'])
            for ref in {**direct_refs, **metadata_refs}.values()
            if ref.get('account_id') and ref.get('conversation_id')
        }
        return {
            'messages_changed': len(direct_refs),
            'citations_changed': len({**direct_refs, **metadata_refs}),
            'tombstones': len(inserted_tombstones),
            'tombstone_citations': sorted(inserted_tombstones),
            'source_rows_added': source_rows_added,
            'source_rows_deleted': source_rows_deleted,
            'changed_refs': list({**direct_refs, **metadata_refs, **inserted_tombstones}.values()),
            'profile_identity_values': sorted(profile_identity_values),
            'profile_scope_changed': profile_scope_changed or bool(inserted_tombstones),
            'conversations_changed': len(changed_conversations),
            'metadata_accounts': len(metadata_account_ids),
            'metadata_conversations': len(metadata_conversation_keys),
            'dirty_recorded': dirty_recorded,
            'chunks': chunk_report,
            'metrics': {
                'sql_statements': sql_statements,
                'commits': commits,
                'rows_scanned': rows_scanned,
                'candidate_rows': len(message_rows) + len(tombstone_inputs),
                'rows_written': rows_written,
                'wal_bytes': max(0, wal_after - wal_before),
            },
        }

    def changed_message_refs(self, messages: Iterable[Message]) -> list[dict[str, str]]:
        """Return message citations whose persisted row would change.

        This is intentionally separate from ``upsert_messages`` so the hot
        idempotent duplicate path can remain a true SQLite no-op: callers that
        need a dirty set for downstream indexes may pay this bounded batch
        comparison cost before writing.
        """
        self.initialize()
        refs: list[dict[str, str]] = []
        seen: set[str] = set()
        with self.connect() as conn:
            conn.execute(
                """CREATE TEMP TABLE IF NOT EXISTS _trove_message_change_batch(
                    citation TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    account_label TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    conversation_title TEXT NOT NULL,
                    conversation_type TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_kind TEXT NOT NULL DEFAULT 'text',
                    shard_id TEXT NOT NULL,
                    local_id INTEGER NOT NULL,
                    sent_by_me INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    direction TEXT NOT NULL
                )"""
            )
            for batch in _message_batches(messages, size=1000):
                conn.execute('DELETE FROM _trove_message_change_batch')
                conn.executemany(
                    """INSERT INTO _trove_message_change_batch(
                        citation,account_id,account_label,conversation_id,conversation_title,conversation_type,
                        sender_id,sender_name,timestamp,content,content_kind,shard_id,local_id,sent_by_me,source_type,direction
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    [_message_value_tuple(m) for m in batch],
                )
                rows = conn.execute(
                    """SELECT b.citation,b.account_id,b.conversation_id
                       FROM _trove_message_change_batch b
                       LEFT JOIN messages m
                         ON m.account_id=b.account_id
                        AND m.conversation_id=b.conversation_id
                        AND m.shard_id=b.shard_id
                        AND m.local_id=b.local_id
                       WHERE m.id IS NULL
                          OR m.citation IS NOT b.citation
                          OR m.account_label IS NOT b.account_label
                          OR m.conversation_title IS NOT b.conversation_title
                          OR m.conversation_type IS NOT b.conversation_type
                          OR m.sender_id IS NOT b.sender_id
                          OR m.content IS NOT b.content
                          OR m.content_kind IS NOT b.content_kind
                          OR m.timestamp IS NOT b.timestamp
                          OR m.sender_name IS NOT b.sender_name
                          OR m.sent_by_me IS NOT b.sent_by_me
                          OR m.direction IS NOT b.direction
                          OR m.source_type IS NOT b.source_type"""
                )
                for row in rows:
                    citation = str(row['citation'])
                    if citation in seen:
                        continue
                    seen.add(citation)
                    refs.append({
                        'citation': citation,
                        'account_id': str(row['account_id']),
                        'conversation_id': str(row['conversation_id']),
                    })
            conn.execute('DELETE FROM _trove_message_change_batch')
            conn.commit()
        return refs

    def _message_filter_sql(self, filters: dict[str, str], *, alias: str = 'm', strict_sender_match: bool = False) -> tuple[list[str], list[Any]]:
        prefix = f'{alias}.' if alias else ''
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in (filters or {}).items():
            if not value:
                continue
            if key == 'sender':
                if strict_sender_match:
                    clauses.append(f'({prefix}sender_name=? OR {prefix}sender_id=?)')
                    params.extend([value, value])
                else:
                    clauses.append(f'({prefix}sender_name LIKE ? OR {prefix}sender_id=?)')
                    params.extend([f'%{value}%', value])
            elif key in {'source_family', 'scope_type'}:
                if value != 'all':
                    clauses.append(f'{prefix}source_type=?')
                    params.append(value)
            elif key == 'since':
                clauses.append(f'{prefix}timestamp>=?')
                params.append(value)
            elif key == 'until':
                clauses.append(f'{prefix}timestamp<=?')
                params.append(value)
            elif key in {'account_id', 'conversation_id', 'conversation_type', 'source_type'}:
                clauses.append(f'{prefix}{key}=?')
                params.append(value)
        return clauses, params

    def _chunk_filter_sql(self, filters: dict[str, str], *, alias: str = 'e') -> tuple[list[str], list[Any]]:
        prefix = f'{alias}.' if alias else ''
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in (filters or {}).items():
            if not value:
                continue
            if key == 'sender':
                clauses.append(f'{prefix}actor LIKE ?')
                params.append(f'%{value}%')
            elif key == 'conversation_id':
                clauses.append(f'{prefix}source_id=?')
                params.append(value)
            elif key in {'source_family', 'scope_type'}:
                if value != 'all':
                    clauses.append(f'{prefix}source_type=?')
                    params.append(value)
            elif key == 'since':
                clauses.append(f'{prefix}timestamp>=?')
                params.append(value)
            elif key == 'until':
                clauses.append(f'{prefix}timestamp<=?')
                params.append(value)
            elif key in {'account_id', 'source_type'}:
                clauses.append(f'{prefix}{key}=?')
                params.append(value)
        return clauses, params

    def _sender_prefilter_limit(self) -> int:
        raw = os.environ.get('TROVE_REWRITE_SENDER_PREFILTER_LIMIT')
        if raw:
            try:
                return max(100, int(raw))
            except ValueError:
                pass
        return 100

    def _sender_prefilter_sql(self, filters: dict[str, str], *, strict_sender_match: bool = False) -> tuple[str | None, list[Any], dict[str, str]]:
        sender = (filters or {}).get('sender')
        if not sender:
            return None, [], dict(filters or {})
        remaining = {k: v for k, v in (filters or {}).items() if k != 'sender'}
        clauses, params = self._message_filter_sql(remaining, alias='', strict_sender_match=True)
        if strict_sender_match:
            where = ['(sender_name=? OR sender_id=?)'] + clauses
            sender_params: list[Any] = [sender, sender]
        else:
            where = ['(sender_name LIKE ? OR sender_id=?)'] + clauses
            sender_params = [f'%{sender}%', sender]
        return ' AND '.join(where), [*sender_params, *params], remaining

    def _chunk_actor_prefilter_sql(self, filters: dict[str, str], *, strict_actor_match: bool = False) -> tuple[str | None, list[Any], dict[str, str]]:
        sender = (filters or {}).get('sender')
        if not sender:
            return None, [], dict(filters or {})
        remaining = {k: v for k, v in (filters or {}).items() if k != 'sender'}
        clauses, params = self._chunk_filter_sql(remaining, alias='')
        if strict_actor_match:
            where = ['actor=?'] + clauses
            actor_params: list[Any] = [sender]
        else:
            where = ['actor LIKE ?'] + clauses
            actor_params = [f'%{sender}%']
        return ' AND '.join(where), [*actor_params, *params], remaining

    def _like_search_sql(self, query: str, filters: dict[str, str], *, limit: int, strict_sender_match: bool = False, sender_prefilter: bool = False) -> list[sqlite3.Row]:
        clauses, params = self._message_filter_sql(filters, alias='m', strict_sender_match=strict_sender_match)
        like = f'%{query}%'
        sql_params: list[Any] = [like, like, like, *params, limit]
        with self.connect() as conn:
            if not self._conversation_filter_exists(conn, filters):
                return []
            if not self._allow_message_like_fallback(conn, filters):
                return []
            if sender_prefilter and filters.get('sender'):
                sender_where, sender_params, _remaining = self._sender_prefilter_sql(filters, strict_sender_match=strict_sender_match)
                if sender_where:
                    return list(conn.execute(
                        f"""WITH sender_scope AS (
                                SELECT * FROM messages
                                WHERE {sender_where}
                                ORDER BY timestamp DESC LIMIT ?
                            )
                            SELECT m.* FROM sender_scope m
                            WHERE (m.content LIKE ? OR m.conversation_title LIKE ? OR m.sender_name LIKE ?)
                            ORDER BY m.timestamp DESC LIMIT ?""",
                        (*sender_params, self._sender_prefilter_limit(), like, like, like, limit),
                    ))
            where = ['(m.content LIKE ? OR m.conversation_title LIKE ? OR m.sender_name LIKE ?)'] + clauses
            sql = f"""SELECT m.* FROM messages m
                      WHERE {' AND '.join(where)}
                      ORDER BY m.timestamp DESC LIMIT ?"""
            return list(conn.execute(sql, sql_params))

    def _normalized_message_phrase_search(self, query: str, filters: dict[str, str], *, limit: int, strict_sender_match: bool = False, sender_prefilter: bool = False) -> list[sqlite3.Row]:
        pattern = _spanning_like_pattern(query)
        if not pattern:
            return []
        clauses, params = self._message_filter_sql(filters, alias='m', strict_sender_match=strict_sender_match)
        content_expr = "replace(replace(replace(m.content, char(13), ' '), char(10), ' '), char(9), ' ')"
        where = [f"({content_expr} LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\')"] + clauses
        with self.connect() as conn:
            if not self._conversation_filter_exists(conn, filters):
                return []
            if not self._allow_message_like_fallback(conn, filters):
                return []
            if sender_prefilter and filters.get('sender'):
                sender_where, sender_params, _remaining = self._sender_prefilter_sql(filters, strict_sender_match=strict_sender_match)
                if sender_where:
                    return list(conn.execute(
                        f"""WITH sender_scope AS (
                                SELECT * FROM messages
                                WHERE {sender_where}
                                ORDER BY timestamp DESC LIMIT ?
                            )
                            SELECT m.* FROM sender_scope m
                            WHERE ({content_expr} LIKE ? ESCAPE '\\' OR m.content LIKE ? ESCAPE '\\')
                            ORDER BY m.timestamp DESC LIMIT ?""",
                        (*sender_params, self._sender_prefilter_limit(), pattern, pattern, limit),
                    ))
            return list(conn.execute(
                f"""SELECT m.* FROM messages m
                    WHERE {' AND '.join(where)}
                    ORDER BY m.timestamp DESC LIMIT ?""",
                (pattern, pattern, *params, limit),
            ))

    def _conversation_filter_exists(self, conn: sqlite3.Connection, filters: dict[str, str]) -> bool:
        filters = filters or {}
        conversation_id = filters.get('conversation_id')
        account_id = filters.get('account_id')
        if not conversation_id or not self._table_exists(conn, 'conversations'):
            return True
        if account_id:
            return conn.execute('SELECT 1 FROM conversations WHERE account_id=? AND conversation_id=? LIMIT 1', (account_id, conversation_id)).fetchone() is not None
        return conn.execute('SELECT 1 FROM conversations WHERE conversation_id=? LIMIT 1', (conversation_id,)).fetchone() is not None

    def _allow_message_like_fallback(self, conn: sqlite3.Connection, filters: dict[str, str], *, threshold: int = 50000) -> bool:
        if not self._table_exists(conn, 'messages'):
            return False
        # Only indexed equality scopes may authorize a leading-wildcard scan.
        # A sender LIKE itself is not a scope proof: counting it would already
        # perform the unbounded scan that this guard exists to prevent.
        scope_filters = {
            key: value
            for key, value in filters.items()
            if key in {'account_id', 'conversation_id', 'source_type'} and value
        }
        if scope_filters:
            clauses, params = self._message_filter_sql(scope_filters, alias='m')
            return self._bounded_scope_count(
                conn,
                cache_key='messages:' + json.dumps(scope_filters, sort_keys=True, ensure_ascii=True),
                from_sql='messages m',
                clauses=clauses,
                params=params,
                threshold=threshold,
            )
        token = self._cardinality_cache_token(conn)
        cached = self._count_cache.get('messages')
        if cached is None or cached[0] != token:
            cached = (token, int(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]))
            self._remember_count('messages', cached)
        return cached[1] <= threshold

    def _allow_chunk_like_fallback(self, conn: sqlite3.Connection, filters: dict[str, str], *, threshold: int = 50000) -> bool:
        if not self._table_exists(conn, 'evidence_chunks'):
            return False
        scope_filters = {
            key: value
            for key, value in filters.items()
            if key in {'account_id', 'conversation_id', 'source_type'} and value
        }
        if scope_filters:
            clauses, params = self._chunk_filter_sql(scope_filters, alias='e')
            return self._bounded_scope_count(
                conn,
                cache_key='active_evidence_chunks:' + json.dumps(scope_filters, sort_keys=True, ensure_ascii=True),
                from_sql='evidence_chunks e',
                clauses=["e.status='active'", *clauses],
                params=params,
                threshold=threshold,
            )
        token = self._cardinality_cache_token(conn)
        cached = self._count_cache.get('active_evidence_chunks')
        if cached is None or cached[0] != token:
            cached = (token, int(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE status='active'").fetchone()[0]))
            self._remember_count('active_evidence_chunks', cached)
        return cached[1] <= threshold

    def _bounded_scope_count(
        self,
        conn: sqlite3.Connection,
        *,
        cache_key: str,
        from_sql: str,
        clauses: list[str],
        params: list[Any],
        threshold: int,
    ) -> bool:
        token = self._cardinality_cache_token(conn)
        cached = self._count_cache.get(cache_key)
        if cached is None or cached[0] != token:
            where = ' AND '.join(clauses) if clauses else '1=1'
            row = conn.execute(
                f'SELECT COUNT(*) FROM (SELECT 1 FROM {from_sql} WHERE {where} LIMIT ?)',
                (*params, threshold + 1),
            ).fetchone()
            cached = (token, int(row[0]) if row is not None else threshold + 1)
            self._remember_count(cache_key, cached)
        return cached[1] <= threshold

    def _remember_count(self, key: str, value: tuple[tuple[Any, ...], int]) -> None:
        self._count_cache[key] = value
        self._count_cache.move_to_end(key)
        while len(self._count_cache) > 256:
            self._count_cache.popitem(last=False)

    def _cardinality_cache_token(self, conn: sqlite3.Connection) -> tuple[Any, ...]:
        """Bind scan guards to both local and external SQLite generations."""

        try:
            data_version = int(conn.execute('PRAGMA data_version').fetchone()[0])
        except (sqlite3.Error, TypeError, ValueError):
            data_version = -1
        try:
            main = self.path.stat()
            main_token: tuple[Any, ...] = (main.st_dev, main.st_ino, main.st_mtime_ns, main.st_size)
        except OSError:
            main_token = ('missing',)
        wal_path = Path(str(self.path) + '-wal')
        try:
            wal = wal_path.stat()
            wal_token: tuple[Any, ...] = (wal.st_dev, wal.st_ino, wal.st_mtime_ns, wal.st_size)
        except OSError:
            wal_token = ('missing',)
        return (
            self._connection_generation,
            int(getattr(conn, 'total_changes', 0)),
            data_version,
            main_token,
            wal_token,
        )


    def fts_search(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.fts_search_filtered(query, filters={}, limit=limit)

    def messages_for_conversation(self, account_id: str, conversation_id: str) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        with self.connect() as conn:
            return list(conn.execute(
                """SELECT * FROM messages WHERE account_id=? AND conversation_id=? ORDER BY timestamp, shard_id, local_id""",
                (account_id, conversation_id),
            ))

    def get_message_by_citation(self, citation: str) -> sqlite3.Row | None:
        if not self.path.exists():
            return None
        with self.connect() as conn:
            return conn.execute('SELECT * FROM messages WHERE citation=?', (citation,)).fetchone()

    def evidence_by_citations(self, citations: list[str]) -> dict[str, EvidenceRow | sqlite3.Row]:
        if not self.path.exists() or not citations:
            return {}
        unique = list(dict.fromkeys(citations))
        out: dict[str, EvidenceRow | sqlite3.Row] = {}
        with self.connect() as conn:
            for start in range(0, len(unique), 500):
                batch = unique[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                for row in conn.execute(f'SELECT * FROM messages WHERE citation IN ({placeholders})', batch):
                    out[row['citation']] = row
            for start in range(0, len(unique), 500):
                batch = [citation for citation in unique[start:start + 500] if citation not in out]
                if not batch:
                    continue
                placeholders = ','.join('?' for _ in batch)
                chunk_predicate = _queryable_transcript_chunk_sql('e')
                for row in conn.execute(
                    f'''SELECT e.* FROM evidence_chunks e
                         WHERE e.chunk_citation IN ({placeholders})
                           AND {chunk_predicate}''',
                    batch,
                ):
                    out[row['chunk_citation']] = self._chunk_row_to_evidence(row)
            for start in range(0, len(unique), 150):
                batch = [citation for citation in unique[start:start + 150] if citation not in out]
                if not batch:
                    continue
                placeholders = ','.join('?' for _ in batch)
                sql = f"""
                    SELECT citation, account_id, moment_id AS __source_id, '' AS __title, author_id AS __actor,
                           text AS __content, timestamp, 'moment' AS __source_type
                    FROM moment_items WHERE citation IN ({placeholders})
                    UNION ALL
                    SELECT citation, account_id, interaction_id AS __source_id, interaction_type AS __title, COALESCE(NULLIF(actor_name,''), actor_id) AS __actor,
                           text AS __content, timestamp, 'moment' AS __source_type
                    FROM moment_interactions WHERE citation IN ({placeholders})
                    UNION ALL
                    SELECT citation, account_id, favorite_id AS __source_id, title AS __title, '' AS __actor,
                           text AS __content, timestamp, 'favorite' AS __source_type
                    FROM favorites WHERE citation IN ({placeholders})
                    UNION ALL
                    SELECT t.citation, '' AS account_id, t.transcript_id AS __source_id, '' AS __title, '' AS __actor,
                           t.text AS __content, t.created_at AS timestamp, 'transcript' AS __source_type
                    FROM transcripts t
                    JOIN provider_jobs pj ON pj.job_id=t.job_id
                    JOIN media_assets ma ON ma.asset_id=t.asset_id
                    WHERE t.citation IN ({placeholders}) AND t.status='active'
                      AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                      AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                      AND pj.status='completed' AND pj.request_hash=ma.content_hash
                    UNION ALL
                    SELECT citation, '' AS account_id, observation_id AS __source_id, '' AS __title, '' AS __actor,
                           caption AS __content, created_at AS timestamp, 'image_observation' AS __source_type
                    FROM image_observations WHERE citation IN ({placeholders})
                """
                params = [*batch, *batch, *batch, *batch, *batch]
                for row in conn.execute(sql, params):
                    out[row['citation']] = self._source_row_to_evidence(row, row['__source_type'])
            for start in range(0, len(unique), 500):
                batch = [citation for citation in unique[start:start + 500] if citation not in out]
                if not batch:
                    continue
                placeholders = ','.join('?' for _ in batch)
                for row in conn.execute(f'SELECT * FROM observations WHERE citation IN ({placeholders})', batch):
                    out[row['citation']] = self._observation_row_to_evidence(row)
        return out

    def vector_entries_for_search(
        self,
        filters: dict[str, str] | None = None,
        *,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Bulk-hydrate filtered vector evidence in one bounded SQL query."""

        if not self.path.exists():
            return []
        if type(limit) is not int or not 1 <= limit <= 50001:
            raise ValueError('vector entry scan limit must be an integer between 1 and 50001')
        filters = filters or {}
        clauses: list[str] = []
        params: list[Any] = []
        for key, raw in filters.items():
            value = str(raw)
            if key == 'account_id':
                clauses.append('e.account_id=?')
                params.append(value)
            elif key == 'conversation_id':
                clauses.append('e.conversation_id=?')
                params.append(value)
            elif key == 'conversation_type':
                clauses.append('e.conversation_type=?')
                params.append(value)
            elif key == 'sender':
                clauses.append('(instr(e.sender_name,?)>0 OR e.sender_id=?)')
                params.extend((value, value))
            elif key == 'source_type':
                clauses.append('e.source_type=?')
                params.append(value)
            elif key in {'source_family', 'scope_type'}:
                if value != 'all':
                    clauses.append('e.source_type=?')
                    params.append(value)
            elif key == 'since':
                clauses.append('e.timestamp>=?')
                params.append(value)
            elif key == 'until':
                clauses.append('e.timestamp<=?')
                params.append(value)
            else:
                clauses.append('0')

        # Priority mirrors evidence_by_citations: messages, active chunks,
        # multi-source evidence, then observations.  ROW_NUMBER prevents a
        # citation present in more than one projection from being scored twice.
        sql = f"""
            WITH evidence_meta AS (
                SELECT 0 AS priority,citation,citation AS parent_citation,account_id,account_label,
                       conversation_id,conversation_title,conversation_type,sender_id,sender_name,
                       timestamp,content,source_type,direction
                FROM messages
                UNION ALL
                SELECT 1,chunk_citation,parent_citation,account_id,COALESCE(NULLIF(account_label,''),account_id,'Vault'),
                       COALESCE(NULLIF(source_id,''),parent_citation),COALESCE(NULLIF(title,''),source_type),'private',
                       actor,COALESCE(NULLIF(actor,''),source_type),COALESCE(timestamp,''),content,source_type,
                       CASE WHEN source_type='message' THEN 'incoming' ELSE 'metadata' END
                FROM evidence_chunks e
                WHERE e.status='active' AND {_queryable_transcript_chunk_sql('e')}
                UNION ALL
                SELECT 2,citation,citation,account_id,COALESCE(NULLIF(account_id,''),'Vault'),source_id,
                       COALESCE(NULLIF(title,''),source_type),'private',actor,COALESCE(NULLIF(actor,''),source_type),
                       COALESCE(timestamp,''),content,source_type,'metadata'
                FROM evidence_items WHERE status='active'
                UNION ALL
                SELECT 2,citation,citation,account_id,COALESCE(NULLIF(account_id,''),'Vault'),moment_id,
                       'Moment','private',COALESCE(author_id,''),COALESCE(NULLIF(author_id,''),'Moment'),
                       COALESCE(timestamp,''),text,'moment','metadata'
                FROM moment_items WHERE status='active'
                UNION ALL
                SELECT 2,citation,citation,account_id,COALESCE(NULLIF(account_id,''),'Vault'),interaction_id,
                       interaction_type,'private',actor_id,COALESCE(NULLIF(actor_name,''),actor_id,'Moment'),
                       COALESCE(timestamp,''),text,'moment','metadata'
                FROM moment_interactions WHERE status='active'
                UNION ALL
                SELECT 2,citation,citation,account_id,COALESCE(NULLIF(account_id,''),'Vault'),favorite_id,
                       COALESCE(NULLIF(title,''),'Favorite'),'private','','Favorite',COALESCE(timestamp,''),
                       COALESCE(NULLIF(text,''),title,'Favorite'),'favorite','metadata'
                FROM favorites
                UNION ALL
                SELECT 2,t.citation,t.citation,'','Vault',t.transcript_id,'Voice transcript','private','','Voice transcript',
                       t.created_at,t.text,'transcript','metadata'
                FROM transcripts t
                JOIN provider_jobs pj ON pj.job_id=t.job_id
                JOIN media_assets ma ON ma.asset_id=t.asset_id
                WHERE t.status='active'
                  AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                  AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                  AND pj.status='completed' AND pj.request_hash=ma.content_hash
                UNION ALL
                SELECT 2,citation,citation,'','Vault',observation_id,'Image observation','private','','Image observation',
                       created_at,caption,'image_observation','metadata'
                FROM image_observations WHERE status IN ('active','proposed')
                UNION ALL
                SELECT 3,citation,citation,'','Vault',entity_id,'Contact observation','private','','',updated_at,
                       value_json,'contact','metadata'
                FROM observations WHERE status IN ('active','needs_review','merge_candidate')
            ), ranked AS (
                SELECT e.*,v.vector_json,
                       ROW_NUMBER() OVER (PARTITION BY v.citation ORDER BY e.priority) AS rn
                FROM vector_entries v JOIN evidence_meta e ON e.citation=v.citation
                {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            )
            SELECT citation,parent_citation,account_id,account_label,conversation_id,conversation_title,
                   conversation_type,sender_id,sender_name,timestamp,content,source_type,direction,vector_json
            FROM ranked r
            WHERE r.rn=1
            LIMIT ?
        """
        with self.connect() as conn:
            return list(conn.execute(sql, (*params, limit)))


    def _filter_row(self, row: sqlite3.Row, filters: dict[str, str]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if key == 'sender':
                if value not in row['sender_name'] and value != row['sender_id']:
                    return False
            elif key in {'source_family', 'scope_type'}:
                if value not in {'all', row['source_type']}:
                    return False
            elif key == 'since':
                if str(row['timestamp'] or '') < value:
                    return False
            elif key == 'until':
                if str(row['timestamp'] or '') > value:
                    return False
            elif row[key] != value:
                return False
        return True

    def exact_search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10, *, allow_like_fallback: bool = True, strict_sender_match: bool = False, sender_prefilter: bool = False) -> list[sqlite3.Row]:
        if not self.path.exists() or not query:
            return []
        filters = filters or {}
        if not _fts_can_match(query):
            return self._like_search_sql(query, filters, limit=limit, strict_sender_match=strict_sender_match, sender_prefilter=sender_prefilter) if allow_like_fallback else []
        clauses, params = self._message_filter_sql(filters, alias='m', strict_sender_match=strict_sender_match)
        where = ['message_fts MATCH ?'] + clauses
        with self.connect() as conn:
            if not self._conversation_filter_exists(conn, filters):
                return []
            try:
                rows = list(conn.execute(
                    f"""SELECT m.* FROM message_fts f JOIN messages m ON m.id=f.rowid
                        WHERE {' AND '.join(where)}
                        ORDER BY rank, m.timestamp DESC LIMIT ?""",
                    (_fts_phrase(query), *params, limit),
                ))
            except sqlite3.OperationalError:
                rows = []
            if rows:
                return rows[:limit]
            if allow_like_fallback and len(''.join(str(query or '').split())) >= 6:
                rows = self._normalized_message_phrase_search(query, filters, limit=limit, strict_sender_match=strict_sender_match, sender_prefilter=sender_prefilter)
                if rows:
                    return rows[:limit]
            terms = important_terms(query)
            fts_terms = [term for term in terms if _fts_can_match(term)]
            if fts_terms:
                try:
                    rows = list(conn.execute(
                        f"""SELECT m.* FROM message_fts f JOIN messages m ON m.id=f.rowid
                            WHERE {' AND '.join(where)}
                            ORDER BY rank, m.timestamp DESC LIMIT ?""",
                        (_fts_and_query(fts_terms), *params, limit),
                    ))
                except sqlite3.OperationalError:
                    rows = []
            elif terms and allow_like_fallback and self._allow_message_like_fallback(conn, filters):
                term_clauses = []
                like_params: list[Any] = []
                for term in terms:
                    term_clauses.append('(m.content LIKE ? OR m.conversation_title LIKE ? OR m.sender_name LIKE ?)')
                    like = f'%{term}%'
                    like_params.extend([like, like, like])
                filter_clauses, filter_params = self._message_filter_sql(filters, alias='m', strict_sender_match=strict_sender_match)
                like_where = term_clauses + filter_clauses
                rows = list(conn.execute(
                    f"SELECT m.* FROM messages m WHERE {' AND '.join(like_where)} ORDER BY m.timestamp DESC LIMIT ?",
                    (*like_params, *filter_params, limit),
                ))
        return rows[:limit]

    def fts_search_filtered(self, query: str, filters: dict[str, str] | None = None, limit: int = 10, *, allow_like_fallback: bool = True) -> list[sqlite3.Row]:
        if not self.path.exists() or not query:
            return []
        filters = filters or {}
        if not _fts_can_match(query):
            return self._like_search_sql(query, filters, limit=limit) if allow_like_fallback else []
        clauses, params = self._message_filter_sql(filters, alias='m')
        where = ['message_fts MATCH ?'] + clauses
        with self.connect() as conn:
            if not self._conversation_filter_exists(conn, filters):
                return []
            try:
                rows = list(conn.execute(
                    f"""SELECT m.* FROM message_fts f JOIN messages m ON m.id=f.rowid
                        WHERE {' AND '.join(where)}
                        ORDER BY rank, m.timestamp DESC LIMIT ?""",
                    (_fts_phrase(query), *params, limit),
                ))
            except sqlite3.OperationalError:
                rows = []
            return rows[:limit]

    def metadata_search(self, query: str, filters: dict[str, str], limit: int = 10) -> list[sqlite3.Row]:
        if not self.path.exists() or not query:
            return []
        filters = filters or {}
        # `metadata_search` is a low-weight route that complements exact/FTS when
        # the user has narrowed search to a specific conversation or sender. Broad
        # filters such as source_type/account_id/conversation_type already flow
        # through exact, FTS, chunk, evidence, and vector routes. Re-running a
        # leading-wildcard LIKE scan for those broad filters is redundant and can
        # dominate real-Vault feature reranking on hundreds of thousands of rows.
        if not any(filters.get(key) for key in ('conversation_id', 'sender')):
            return []
        return self.fts_search_filtered(query, filters=filters, limit=limit)

    def context_window(self, citation: str, before: int = 5, after: int = 5) -> list[sqlite3.Row]:
        from trove_core.bounds import BoundedLimit, CONTEXT_WINDOW

        before = BoundedLimit(before, field='before', spec=CONTEXT_WINDOW)
        after = BoundedLimit(after, field='after', spec=CONTEXT_WINDOW)
        anchor = self.get_message_by_citation(citation)
        if anchor is None:
            return []
        key = (anchor['timestamp'], anchor['shard_id'], anchor['local_id'])
        with self.connect() as conn:
            prev_rows = []
            if before:
                prev_rows = list(conn.execute(
                    """SELECT * FROM messages
                       WHERE account_id=? AND conversation_id=?
                         AND (timestamp, shard_id, local_id) < (?, ?, ?)
                       ORDER BY timestamp DESC, shard_id DESC, local_id DESC
                       LIMIT ?""",
                    (anchor['account_id'], anchor['conversation_id'], *key, before),
                ))
            next_rows = []
            if after:
                next_rows = list(conn.execute(
                    """SELECT * FROM messages
                       WHERE account_id=? AND conversation_id=?
                         AND (timestamp, shard_id, local_id) > (?, ?, ?)
                       ORDER BY timestamp ASC, shard_id ASC, local_id ASC
                       LIMIT ?""",
                    (anchor['account_id'], anchor['conversation_id'], *key, after),
                ))
        return list(reversed(prev_rows)) + [anchor] + next_rows

    def evidence_by_citation(self, citation: str) -> EvidenceRow | sqlite3.Row | None:
        row = self.get_message_by_citation(citation)
        if row is not None:
            return row
        if not self.path.exists():
            return None
        with self.connect() as conn:
            if self._table_exists(conn, 'evidence_chunks'):
                chunk = conn.execute(
                    f'''SELECT e.* FROM evidence_chunks e
                         WHERE e.chunk_citation=?
                           AND {_queryable_transcript_chunk_sql('e')}
                         LIMIT 1''',
                    (citation,),
                ).fetchone()
                if chunk is not None:
                    return self._chunk_row_to_evidence(chunk)
            for table, source_type, id_col, title_col, actor_col, text_col in [
                ('moment_items', 'moment', 'moment_id', "''", 'author_id', 'text'),
                ('moment_interactions', 'moment', 'interaction_id', "interaction_type", "COALESCE(NULLIF(actor_name,''), actor_id)", 'text'),
                ('favorites', 'favorite', 'favorite_id', 'title', "''", 'text'),
                ('image_observations', 'image_observation', 'observation_id', "''", "''", 'caption'),
            ]:
                if not self._table_exists(conn, table):
                    continue
                try:
                    sql = f'SELECT *, {id_col} AS __source_id, {title_col} AS __title, {actor_col} AS __actor, {text_col} AS __content FROM {table} WHERE citation=? LIMIT 1'
                    src = conn.execute(sql, (citation,)).fetchone()
                except sqlite3.DatabaseError:
                    src = None
                if src is not None:
                    return self._source_row_to_evidence(src, source_type)
            if self._table_exists(conn, 'transcripts'):
                src = conn.execute(
                    f'''SELECT t.*, t.transcript_id AS __source_id, '' AS __title,
                               '' AS __actor, t.text AS __content
                          FROM transcripts t
                          JOIN provider_jobs pj ON pj.job_id=t.job_id
                          JOIN media_assets ma ON ma.asset_id=t.asset_id
                         WHERE t.citation=? AND t.status='active'
                           AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                           AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                           AND pj.status='completed' AND pj.request_hash=ma.content_hash
                         LIMIT 1''',
                    (citation,),
                ).fetchone()
                if src is not None:
                    return self._source_row_to_evidence(src, 'transcript')
            if self._table_exists(conn, 'observations'):
                src = conn.execute('SELECT * FROM observations WHERE citation=? LIMIT 1', (citation,)).fetchone()
                if src is not None:
                    return self._observation_row_to_evidence(src)
        return None

    def media_hints_for_citations(self, citations: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Return redacted media availability hints keyed by input citation.

        A parent Moment citation can own multiple image/video citations
        (``#image-N``/``#video-N``). Hints never expose local paths; callers
        fetch bytes via the explicit media-fetch tool.
        """
        if not self.path.exists():
            return {}
        requested = list(dict.fromkeys(str(c) for c in citations if c))
        if not requested:
            return {}
        out: dict[str, dict[str, Any]] = {}

        def variants_for(citation: str) -> list[str]:
            variants = [citation]
            if '#chunk-' in citation:
                variants.append(citation.split('#chunk-', 1)[0])
            if '#image' in citation:
                variants.append(citation.split('#image', 1)[0])
            if '#video' in citation:
                variants.append(citation.split('#video', 1)[0])
            if '#' in citation:
                variants.append(citation.split('#', 1)[0])
            return list(dict.fromkeys(v for v in variants if v))

        with self.connect() as conn:
            if not self._table_exists(conn, 'media_assets'):
                return {}
            variants_by_input = {citation: variants_for(citation) for citation in requested}
            if self._table_exists(conn, 'evidence_chunks'):
                all_variants = list(dict.fromkeys(v for variants in variants_by_input.values() for v in variants if v))
                parent_by_chunk: dict[str, str] = {}
                for start in range(0, len(all_variants), 500):
                    batch = all_variants[start:start + 500]
                    if not batch:
                        continue
                    placeholders = ','.join('?' for _ in batch)
                    for row in conn.execute(
                        f'''SELECT e.chunk_citation, e.parent_citation
                              FROM evidence_chunks e
                             WHERE e.chunk_citation IN ({placeholders})
                               AND {_queryable_transcript_chunk_sql('e')}''',
                        batch,
                    ):
                        if row['parent_citation']:
                            parent_by_chunk[str(row['chunk_citation'])] = str(row['parent_citation'])
                for citation, variants in list(variants_by_input.items()):
                    for value in list(variants):
                        parent = parent_by_chunk.get(value)
                        if parent:
                            variants.extend(variants_for(parent))
                    variants_by_input[citation] = list(dict.fromkeys(v for v in variants if v))

            request_rows: list[tuple[str, str, str, str]] = []
            for citation, variants in variants_by_input.items():
                for variant in variants:
                    request_rows.append((citation, variant, f'{variant}#image-%', f'{variant}#video-%'))
            if request_rows:
                media_by_input: dict[str, dict[str, dict[str, Any]]] = {citation: {} for citation in requested}

                def prefix_upper_bound(prefix: str) -> str:
                    if not prefix:
                        return '\U0010ffff'
                    return prefix[:-1] + chr(ord(prefix[-1]) + 1)

                def citation_matches_variant(citation: str, variant: str) -> bool:
                    return (
                        citation == variant
                        or citation.startswith(f'{variant}#image-')
                        or citation.startswith(f'{variant}#video-')
                    )

                def collect_rows(sql_template: str, citation_expr: str, rows: list[tuple[str, str, str, str]]) -> None:
                    for start in range(0, len(rows), 100):
                        batch = rows[start:start + 100]
                        variants = list(dict.fromkeys(row[1] for row in batch if row[1]))
                        if not variants:
                            continue
                        clauses: list[str] = []
                        params: list[Any] = []
                        placeholders = ','.join('?' for _ in variants)
                        clauses.append(f'{citation_expr} IN ({placeholders})')
                        params.extend(variants)
                        for variant in variants:
                            image_prefix = f'{variant}#image-'
                            video_prefix = f'{variant}#video-'
                            clauses.append(f'({citation_expr} >= ? AND {citation_expr} < ?)')
                            params.extend([image_prefix, prefix_upper_bound(image_prefix)])
                            clauses.append(f'({citation_expr} >= ? AND {citation_expr} < ?)')
                            params.extend([video_prefix, prefix_upper_bound(video_prefix)])
                        sql = sql_template.format(citation_predicate=' OR '.join(clauses))
                        for row in conn.execute(sql, params):
                            media_citation = str(row['citation'])
                            for input_citation, variant, _image_like, _video_like in batch:
                                if not citation_matches_variant(media_citation, variant):
                                    continue
                                media_by_input.setdefault(input_citation, {}).setdefault(media_citation, {
                                    'citation': media_citation,
                                    'asset_id': row['asset_id'],
                                    'modality': row['modality'],
                                    'media_type': row['media_type'],
                                    'available': row['cache_state'] not in {'missing_local_cache', 'metadata_only', 'inventory_only'},
                                    'cache_state': row['cache_state'],
                                    'fetch_tool': 'trove_media_fetch',
                                    'raw_paths_included': False,
                                })

                def collect_voice_rows(sql_template: str, citation_expr: str) -> None:
                    for start in range(0, len(requested), 100):
                        batch_inputs = requested[start:start + 100]
                        variants = list(dict.fromkeys(
                            variant
                            for input_citation in batch_inputs
                            for variant in variants_by_input.get(input_citation, [])
                            if variant
                        ))
                        if not variants:
                            continue
                        placeholders = ','.join('?' for _ in variants)
                        sql = sql_template.format(citation_predicate=f'{citation_expr} IN ({placeholders})')
                        for row in conn.execute(sql, variants):
                            media_citation = str(row['citation'])
                            for input_citation in batch_inputs:
                                if media_citation not in variants_by_input.get(input_citation, []):
                                    continue
                                cache_state = row['cache_state']
                                media_by_input.setdefault(input_citation, {}).setdefault(media_citation, {
                                    'citation': media_citation,
                                    'asset_id': row['asset_id'],
                                    'modality': row['modality'],
                                    'media_type': row['media_type'],
                                    'available': cache_state not in {'missing_local_cache', 'metadata_only', 'inventory_only'},
                                    'cache_state': cache_state,
                                    'transcript_state': 'cached' if row['has_active_transcript'] else 'pending' if cache_state in {'normalized', 'copied', 'cached'} else 'unavailable',
                                    'transcribe_tool': 'trove_voice_transcribe_lazy',
                                    'raw_paths_included': False,
                                })

                if self._table_exists(conn, 'media_asset_links'):
                    collect_rows(
                        """SELECT l.source_citation AS citation, ma.asset_id, ma.cache_state, ma.media_type, ma.modality
                           FROM media_asset_links l
                           JOIN media_assets ma ON ma.asset_id=l.asset_id
                           WHERE l.accepted=1
                             AND ma.modality IN ('image','video','file','attachment','document')
                             AND ({citation_predicate})""",
                        'l.source_citation',
                        request_rows,
                    )
                collect_rows(
                    """SELECT ma.citation, ma.asset_id, ma.cache_state, ma.media_type, ma.modality
                       FROM media_assets ma
                       WHERE ma.modality IN ('image','video','file','attachment','document')
                         AND (
                           NOT EXISTS(SELECT 1 FROM messages mx WHERE mx.citation=ma.citation)
                           OR EXISTS(SELECT 1 FROM messages mx WHERE mx.citation=ma.citation AND mx.conversation_type='private')
                         )
                         AND ({citation_predicate})""",
                    'ma.citation',
                    request_rows,
                )
                transcript_exists_expr = (
                    f"""EXISTS(
                        SELECT 1 FROM transcripts t
                        JOIN provider_jobs pj ON pj.job_id=t.job_id
                        WHERE t.asset_id=ma.asset_id AND t.status='active'
                          AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                          AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                          AND pj.status='completed' AND pj.request_hash=ma.content_hash
                    )"""
                    if self._table_exists(conn, 'transcripts')
                    else '0'
                )
                if self._table_exists(conn, 'media_asset_links'):
                    collect_voice_rows(
                        f"""SELECT l.source_citation AS citation, ma.asset_id, ma.cache_state, ma.media_type, ma.modality,
                                  {transcript_exists_expr} AS has_active_transcript
                            FROM media_asset_links l
                            JOIN media_assets ma ON ma.asset_id=l.asset_id
                            JOIN messages m ON m.citation=l.source_citation AND m.conversation_type='private'
                            WHERE l.accepted=1
                              AND ma.modality='voice'
                              AND ({{citation_predicate}})""",
                        'l.source_citation',
                    )
                collect_voice_rows(
                    f"""SELECT ma.citation, ma.asset_id, ma.cache_state, ma.media_type, ma.modality,
                              {transcript_exists_expr} AS has_active_transcript
                        FROM media_assets ma
                        JOIN messages m ON m.citation=ma.citation AND m.conversation_type='private'
                        WHERE ma.modality='voice'
                          AND ({{citation_predicate}})""",
                    'ma.citation',
                )

                for citation, media_items in media_by_input.items():
                    if not media_items:
                        continue
                    items = [media_items[key] for key in sorted(media_items)]
                    image_count = sum(1 for item in items if item.get('modality') == 'image')
                    video_count = sum(1 for item in items if item.get('modality') == 'video')
                    voice_count = sum(1 for item in items if item.get('modality') == 'voice')
                    file_count = sum(1 for item in items if item.get('modality') in {'file', 'attachment', 'document'})
                    if len(items) == 1 and voice_count == 1:
                        out[citation] = items[0]
                    elif voice_count == 0:
                        payload = {
                            'type': 'image' if len(items) == 1 and image_count == 1 else 'video' if len(items) == 1 and video_count == 1 else 'file' if len(items) == 1 and file_count == 1 else 'images' if video_count == 0 and file_count == 0 else 'media',
                            'image_count': image_count,
                            'video_count': video_count,
                            'media_count': len(items),
                            'available_count': sum(1 for item in items if item.get('available')),
                            'items': items,
                            'fetch_tool': 'trove_media_fetch',
                            'raw_paths_included': False,
                        }
                        if file_count:
                            payload['file_count'] = file_count
                        out[citation] = payload
                    else:
                        out[citation] = {
                            'type': 'media',
                            'image_count': image_count,
                            'video_count': video_count,
                            'media_count': len(items),
                            'available_count': sum(1 for item in items if item.get('available')),
                            'items': items,
                            'fetch_tool': 'trove_media_fetch',
                            'raw_paths_included': False,
                        }
        return out

    def _source_row_to_evidence(self, row: sqlite3.Row, source_type: str) -> EvidenceRow:
        account_id = row['account_id'] if 'account_id' in row.keys() else ''
        timestamp = row['timestamp'] if 'timestamp' in row.keys() else (row['created_at'] if 'created_at' in row.keys() else '')
        source_id = row['__source_id'] if '__source_id' in row.keys() else row['citation']
        title = row['__title'] if '__title' in row.keys() else ''
        actor = row['__actor'] if '__actor' in row.keys() else ''
        content = row['__content'] if '__content' in row.keys() else ''
        label = {
            'moment': 'Moment',
            'favorite': 'Favorite',
            'transcript': 'Voice transcript',
            'image_observation': 'Image observation',
            'contact': 'Contact',
        }.get(source_type, source_type)
        return EvidenceRow({
            'citation': row['citation'],
            'account_id': account_id,
            'account_label': account_id or 'Vault',
            'conversation_id': str(source_id),
            'conversation_title': title or label,
            'conversation_type': 'private',
            'sender_name': actor or label,
            'timestamp': timestamp or '',
            'content': content or title or label,
            'source_type': source_type,
            'direction': 'metadata',
        })

    def _chunk_row_to_evidence(self, row: sqlite3.Row) -> EvidenceRow:
        try:
            metadata = json.loads(row['metadata_json'] or '{}')
        except Exception:
            metadata = {}
        direction = 'metadata'
        if row['source_type'] == 'message':
            direction = str(metadata.get('direction') or 'incoming')
        return EvidenceRow({
            'citation': row['chunk_citation'],
            'parent_citation': row['parent_citation'],
            'account_id': row['account_id'],
            'account_label': row['account_label'] or row['account_id'] or 'Vault',
            'conversation_id': row['source_id'] or row['parent_citation'],
            'conversation_title': row['title'] or row['source_type'],
            'conversation_type': 'private',
            'sender_id': row['actor'] or '',
            'sender_name': row['actor'] or row['source_type'],
            'timestamp': row['timestamp'] or '',
            'content': row['content'],
            'source_type': row['source_type'],
            'direction': direction,
        })

    def _observation_row_to_evidence(self, row: sqlite3.Row) -> EvidenceRow:
        try:
            value = json.loads(row['value_json'] or '{}')
        except json.JSONDecodeError:
            value = {}
        text = value.get('text') if isinstance(value, dict) else str(value)
        return EvidenceRow({
            'citation': row['citation'],
            'account_id': '',
            'account_label': '',
            'conversation_id': row['entity_id'],
            'conversation_title': 'Contact observation',
            'conversation_type': 'private',
            'sender_name': '',
            'timestamp': row['updated_at'],
            'content': str(text or ''),
            'source_type': 'contact',
            'direction': 'metadata',
        })

    def multisource_search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10) -> list[EvidenceRow]:
        if not self.path.exists() or not query:
            return []
        filters = filters or {}
        out: list[EvidenceRow] = []
        with self.connect() as conn:
            if not self._table_exists(conn, 'evidence_chunks'):
                return []
            if not self._conversation_filter_exists(conn, filters):
                return []
            filter_clauses, filter_params = self._chunk_filter_sql(filters, alias='e')
            base = [
                "e.status='active'",
                "e.source_type <> 'message'",
                _queryable_transcript_chunk_sql('e'),
            ] + filter_clauses
            if _fts_can_match(query) and self._table_exists(conn, 'chunk_fts'):
                where = ['chunk_fts MATCH ?'] + base
                phrase_like = f'%{_like_escape(query)}%'
                try:
                    rows = list(conn.execute(
                        f"""SELECT e.* FROM chunk_fts f JOIN evidence_chunks e ON e.rowid=f.rowid
                            WHERE {' AND '.join(where)}
                            ORDER BY bm25(chunk_fts, 0.0, 1.0, 3.5, 0.8),
                                     CASE
                                       WHEN e.title LIKE ? ESCAPE '\\' THEN 0
                                       WHEN e.content LIKE ? ESCAPE '\\' THEN 1
                                       ELSE 2
                                     END,
                                     e.timestamp DESC
                            LIMIT ?""",
                        (_fts_phrase(query), *filter_params, phrase_like, phrase_like, limit),
                    ))
                except sqlite3.OperationalError:
                    rows = []
            else:
                like = f'%{query}%'
                where = ['(e.content LIKE ? OR e.title LIKE ? OR e.actor LIKE ?)'] + base
                rows = list(conn.execute(
                    f"""SELECT e.* FROM evidence_chunks e
                        WHERE {' AND '.join(where)}
                        ORDER BY e.timestamp DESC LIMIT ?""",
                    (like, like, like, *filter_params, limit),
                ))
            for row in rows:
                ev = self._chunk_row_to_evidence(row)
                if self._filter_row(ev, filters):
                    out.append(ev)
                if len(out) >= limit:
                    break
        return out

    def scope_status(self) -> dict[str, Any]:
        counts = self.counts()
        status: dict[str, Any] = {'contract_version': 1, 'counts': counts, 'families': {}, 'excluded_queryable_count': 0}
        if not self.path.exists():
            return status
        with self.connect() as conn:
            def safe_count(table: str, where: str = '1=1') -> int:
                if not self._table_exists(conn, table):
                    return 0
                return int(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {where}').fetchone()[0])
            valid_cloud_transcripts = 0
            if all(self._table_exists(conn, table) for table in ('transcripts', 'provider_jobs', 'media_assets')):
                valid_cloud_transcripts = int(conn.execute(f'''
                    SELECT COUNT(*)
                      FROM transcripts t
                      JOIN provider_jobs pj ON pj.job_id=t.job_id
                      JOIN media_assets ma ON ma.asset_id=t.asset_id
                     WHERE t.status='active'
                       AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                       AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                       AND pj.status='completed' AND pj.request_hash=ma.content_hash
                ''').fetchone()[0])
            status['families'] = {
                'private_chat': safe_count('messages', "conversation_type='private'"),
                'group_chat': safe_count('messages', "conversation_type='group'"),
                'contact': safe_count('observations', "source_type='contact'"),
                'moment': safe_count('moment_items') + safe_count('moment_interactions'),
                'favorite': safe_count('favorites'),
                'transcript': valid_cloud_transcripts,
                'image_observation': safe_count('image_observations'),
            }
        return status


    def rebuild_evidence_chunks(self, *, max_chars: int = 900, overlap_chars: int = 120) -> dict[str, Any]:
        self.initialize()
        created = 0
        parents = 0
        deleted_vectors = 0
        dirty_recorded = 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.connect() as conn:
            dirty_recorded += self._record_dirty_chunk_parents_conn(conn)
            old_chunk_citations = [
                row['chunk_citation'] for row in conn.execute('SELECT chunk_citation FROM evidence_chunks')
            ] if self._table_exists(conn, 'evidence_chunks') else []
            conn.execute('DELETE FROM evidence_chunks')
            if old_chunk_citations and self._table_exists(conn, 'vector_entries'):
                deleted_vectors += self._delete_vector_entries_conn(conn, old_chunk_citations)

            def add(parent_citation: str, account_id: str, account_label: str, source_type: str, source_id: str, title: str, actor: str, timestamp: str, content: str, metadata: dict[str, Any] | None = None) -> None:
                nonlocal created, parents
                parts = chunk_text(content or title or source_type, max_chars=max_chars, overlap_chars=overlap_chars)
                if not parts:
                    return
                parents += 1
                for idx, part in enumerate(parts):
                    chunk_citation = f'{parent_citation}#chunk-{idx}'
                    conn.execute(
                        """INSERT OR REPLACE INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (chunk_citation, chunk_citation, parent_citation, account_id or '', account_label or account_id or '', source_type, source_id or '', title or '', actor or '', timestamp or '', part, idx, json.dumps(metadata or {}, ensure_ascii=False), 'active', now),
                    )
                    created += 1

            for row in conn.execute('SELECT citation,account_id,account_label,conversation_id,conversation_title,sender_name,timestamp,content,content_kind,source_type,direction FROM messages ORDER BY timestamp,citation'):
                content_kind = row['content_kind'] if 'content_kind' in row.keys() else 'text'
                add(row['citation'], row['account_id'], row['account_label'], row['source_type'], row['conversation_id'], row['conversation_title'], row['sender_name'], row['timestamp'], display_content_for_kind(row['content'], content_kind), {'family': 'message', 'content_kind': content_kind, 'direction': row['direction'] or 'unknown'})
            if self._table_exists(conn, 'favorites'):
                for row in conn.execute('SELECT citation,account_id,favorite_id,title,timestamp,(title || char(10) || text) AS content FROM favorites ORDER BY timestamp,citation'):
                    add(row['citation'], row['account_id'], row['account_id'], 'favorite', row['favorite_id'], row['title'], 'Favorite', row['timestamp'], row['content'], {'family': 'favorite'})
            if self._table_exists(conn, 'moment_items'):
                for row in conn.execute('SELECT citation,account_id,moment_id,author_id,timestamp,text FROM moment_items ORDER BY timestamp,citation'):
                    add(row['citation'], row['account_id'], row['account_id'], 'moment', row['moment_id'], 'Moment', row['author_id'], row['timestamp'], row['text'], {'family': 'moment'})
            if self._table_exists(conn, 'moment_interactions'):
                for row in conn.execute('SELECT citation,account_id,interaction_id,actor_id,actor_name,timestamp,text,interaction_type FROM moment_interactions ORDER BY timestamp,citation'):
                    add(row['citation'], row['account_id'], row['account_id'], 'moment', row['interaction_id'], row['interaction_type'], (row['actor_name'] or row['actor_id']), row['timestamp'], row['text'], {'family': 'moment_interaction'})
            if self._table_exists(conn, 'transcripts'):
                for row in conn.execute(f'''
                    SELECT t.citation,t.transcript_id,t.created_at,t.text
                      FROM transcripts t
                      JOIN provider_jobs pj ON pj.job_id=t.job_id
                      JOIN media_assets ma ON ma.asset_id=t.asset_id
                     WHERE t.status='active'
                       AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                       AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                       AND pj.status='completed' AND pj.request_hash=ma.content_hash
                  ORDER BY t.created_at,t.citation
                '''):
                    add(row['citation'], '', 'Vault', 'transcript', row['transcript_id'], 'Voice transcript', 'Transcript', row['created_at'], row['text'], {'family': 'transcript'})
            if self._table_exists(conn, 'image_observations'):
                for row in conn.execute("SELECT citation,observation_id,created_at,(caption || char(10) || visible_text) AS content FROM image_observations ORDER BY created_at,citation"):
                    add(row['citation'], '', 'Vault', 'image_observation', row['observation_id'], 'Image observation', 'Image', row['created_at'], row['content'], {'family': 'image_observation'})
            if self._table_exists(conn, 'observations'):
                for row in conn.execute('SELECT citation,entity_id,updated_at,value_json FROM observations WHERE source_type="contact" ORDER BY updated_at,citation'):
                    try:
                        value = json.loads(row['value_json'] or '{}')
                    except json.JSONDecodeError:
                        value = {}
                    text = value.get('text') if isinstance(value, dict) else str(value)
                    add(row['citation'], '', 'Vault', 'contact', row['entity_id'], 'Contact observation', 'Contact', row['updated_at'], str(text or ''), {'family': 'contact'})
            dirty_recorded += self._record_dirty_chunk_parents_conn(conn)
            conn.commit()
        return {'parents': parents, 'chunks': created, 'deleted_vectors': deleted_vectors, 'dirty_recorded': dirty_recorded, 'max_chars': max_chars, 'overlap_chars': overlap_chars}

    def rebuild_evidence_chunks_for_source_types(
        self,
        source_types: Iterable[str],
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        """Refresh non-message chunks for the selected evidence families only."""
        self.initialize()
        selected = sorted({str(source_type) for source_type in source_types if source_type in {'contact', 'moment', 'favorite', 'transcript', 'image_observation'}})
        if not selected:
            return {'parents': 0, 'chunks': 0, 'source_types': [], 'max_chars': max_chars, 'overlap_chars': overlap_chars}
        created = 0
        parents = 0
        dirty_recorded = 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.connect() as conn:
            source_placeholders = ','.join('?' for _ in selected)
            dirty_recorded += self._record_dirty_chunk_parents_conn(
                conn,
                where=f'source_type IN ({source_placeholders})',
                params=selected,
            )
            deleted_chunk_citations = [
                row['chunk_citation'] for row in conn.execute(
                    f"SELECT chunk_citation FROM evidence_chunks WHERE source_type IN ({source_placeholders})",
                    selected,
                )
            ]
            conn.execute(f"DELETE FROM evidence_chunks WHERE source_type IN ({source_placeholders})", selected)

            def add(parent_citation: str, account_id: str, account_label: str, source_type: str, source_id: str, title: str, actor: str, timestamp: str, content: str, metadata: dict[str, Any] | None = None) -> None:
                nonlocal created, parents
                parts = chunk_text(content or title or source_type, max_chars=max_chars, overlap_chars=overlap_chars)
                if not parts:
                    return
                parents += 1
                for idx, part in enumerate(parts):
                    chunk_citation = f'{parent_citation}#chunk-{idx}'
                    conn.execute(
                        """INSERT OR REPLACE INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (chunk_citation, chunk_citation, parent_citation, account_id or '', account_label or account_id or '', source_type, source_id or '', title or '', actor or '', timestamp or '', part, idx, json.dumps(metadata or {}, ensure_ascii=False), 'active', now),
                    )
                    created += 1

            if 'favorite' in selected and self._table_exists(conn, 'favorites'):
                for row in conn.execute('SELECT citation,account_id,favorite_id,title,timestamp,(title || char(10) || text) AS content FROM favorites ORDER BY timestamp,citation'):
                    add(row['citation'], row['account_id'], row['account_id'], 'favorite', row['favorite_id'], row['title'], 'Favorite', row['timestamp'], row['content'], {'family': 'favorite'})
            if 'moment' in selected and self._table_exists(conn, 'moment_items'):
                for row in conn.execute('SELECT citation,account_id,moment_id,author_id,timestamp,text FROM moment_items ORDER BY timestamp,citation'):
                    add(row['citation'], row['account_id'], row['account_id'], 'moment', row['moment_id'], 'Moment', row['author_id'], row['timestamp'], row['text'], {'family': 'moment'})
            if 'moment' in selected and self._table_exists(conn, 'moment_interactions'):
                for row in conn.execute('SELECT citation,account_id,interaction_id,actor_id,actor_name,timestamp,text,interaction_type FROM moment_interactions ORDER BY timestamp,citation'):
                    add(row['citation'], row['account_id'], row['account_id'], 'moment', row['interaction_id'], row['interaction_type'], (row['actor_name'] or row['actor_id']), row['timestamp'], row['text'], {'family': 'moment_interaction'})
            if 'transcript' in selected and self._table_exists(conn, 'transcripts'):
                for row in conn.execute(f'''
                    SELECT t.citation,t.transcript_id,t.created_at,t.text
                      FROM transcripts t
                      JOIN provider_jobs pj ON pj.job_id=t.job_id
                      JOIN media_assets ma ON ma.asset_id=t.asset_id
                     WHERE t.status='active'
                       AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                       AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                       AND pj.status='completed' AND pj.request_hash=ma.content_hash
                  ORDER BY t.created_at,t.citation
                '''):
                    add(row['citation'], '', 'Vault', 'transcript', row['transcript_id'], 'Voice transcript', 'Transcript', row['created_at'], row['text'], {'family': 'transcript'})
            if 'image_observation' in selected and self._table_exists(conn, 'image_observations'):
                for row in conn.execute("SELECT citation,observation_id,created_at,(caption || char(10) || visible_text) AS content FROM image_observations ORDER BY created_at,citation"):
                    add(row['citation'], '', 'Vault', 'image_observation', row['observation_id'], 'Image observation', 'Image', row['created_at'], row['content'], {'family': 'image_observation'})
            if 'contact' in selected and self._table_exists(conn, 'observations'):
                for row in conn.execute('SELECT citation,entity_id,updated_at,value_json FROM observations WHERE source_type="contact" ORDER BY updated_at,citation'):
                    try:
                        value = json.loads(row['value_json'] or '{}')
                    except json.JSONDecodeError:
                        value = {}
                    text = value.get('text') if isinstance(value, dict) else str(value)
                    add(row['citation'], '', 'Vault', 'contact', row['entity_id'], 'Contact observation', 'Contact', row['updated_at'], str(text or ''), {'family': 'contact'})
            if deleted_chunk_citations and self._table_exists(conn, 'vector_entries'):
                for start in range(0, len(deleted_chunk_citations), 500):
                    batch = deleted_chunk_citations[start:start + 500]
                    placeholders = ','.join('?' for _ in batch)
                    conn.execute(f'DELETE FROM vector_entries WHERE citation IN ({placeholders})', batch)
            dirty_recorded += self._record_dirty_chunk_parents_conn(
                conn,
                where=f'source_type IN ({source_placeholders})',
                params=selected,
            )
            conn.commit()
        return {'parents': parents, 'chunks': created, 'source_types': selected, 'dirty_recorded': dirty_recorded, 'max_chars': max_chars, 'overlap_chars': overlap_chars}

    def upsert_evidence_chunks_for_source_citations(
        self,
        source_type: str,
        citations: Iterable[str],
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        """Refresh chunks for exact non-message source citations only.

        Provider jobs append one transcript/image-observation row at a time.  A
        full family rebuild would be correct but turns each job into a table
        scan.  This path is bounded by caller-provided citations and uses the
        source citation indexes.
        """
        self.initialize()
        source_type = str(source_type)
        if source_type not in {'contact', 'moment', 'favorite', 'transcript', 'image_observation'}:
            return {'parents': 0, 'chunks': 0, 'source_type': source_type, 'citations': 0, 'max_chars': max_chars, 'overlap_chars': overlap_chars}
        unique = list(dict.fromkeys(str(citation) for citation in citations if citation))
        if not unique:
            return {'parents': 0, 'chunks': 0, 'source_type': source_type, 'citations': 0, 'max_chars': max_chars, 'overlap_chars': overlap_chars}
        created = 0
        parents = 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.connect() as conn:
            deleted_chunk_citations: list[str] = []
            source_id_column = 'transcript_id' if source_type == 'transcript' else 'observation_id'
            source_table = 'transcripts' if source_type == 'transcript' else 'image_observations'
            for start in range(0, len(unique), 500):
                batch = unique[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                source_ids: list[str] = []
                if source_type in {'transcript', 'image_observation'} and self._table_exists(conn, source_table):
                    source_ids = [
                        str(row['source_id']) for row in conn.execute(
                            f"SELECT {source_id_column} AS source_id FROM {source_table} WHERE citation IN ({placeholders})",
                            batch,
                        )
                        if row['source_id']
                    ]
                delete_clauses = [f"parent_citation IN ({placeholders})"]
                delete_params: list[Any] = [source_type, *batch]
                if source_ids:
                    source_id_placeholders = ','.join('?' for _ in source_ids)
                    delete_clauses.append(f"source_id IN ({source_id_placeholders})")
                    delete_params.extend(source_ids)
                delete_where = f"source_type=? AND ({' OR '.join(delete_clauses)})"
                deleted_chunk_citations.extend([
                    row['chunk_citation'] for row in conn.execute(
                        f"SELECT chunk_citation FROM evidence_chunks WHERE {delete_where}",
                        delete_params,
                    )
                ])
                conn.execute(
                    f"DELETE FROM evidence_chunks WHERE {delete_where}",
                    delete_params,
                )

                def add(parent_citation: str, source_id: str, title: str, actor: str, timestamp: str, content: str, metadata: dict[str, Any]) -> None:
                    nonlocal created, parents
                    parts = chunk_text(content or title or source_type, max_chars=max_chars, overlap_chars=overlap_chars)
                    if not parts:
                        return
                    parents += 1
                    for idx, part in enumerate(parts):
                        chunk_citation = f'{parent_citation}#chunk-{idx}'
                        conn.execute(
                            """INSERT OR REPLACE INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (chunk_citation, chunk_citation, parent_citation, '', 'Vault', source_type, source_id or '', title or '', actor or '', timestamp or '', part, idx, json.dumps(metadata, ensure_ascii=False), 'active', now),
                        )
                        created += 1

                if source_type == 'transcript' and self._table_exists(conn, 'transcripts'):
                    for row in conn.execute(
                        f"""SELECT t.citation,t.transcript_id,t.created_at,t.text
                              FROM transcripts t
                              JOIN provider_jobs pj ON pj.job_id=t.job_id
                              JOIN media_assets ma ON ma.asset_id=t.asset_id
                             WHERE t.citation IN ({placeholders}) AND t.status='active'
                               AND pj.provider='{_CLOUD_ASR_PROVIDER_NAME}'
                               AND pj.model='{_CLOUD_ASR_MODEL_ID}'
                               AND pj.status='completed' AND pj.request_hash=ma.content_hash
                          ORDER BY t.created_at,t.citation""",
                        batch,
                    ):
                        add(row['citation'], row['transcript_id'], 'Voice transcript', 'Transcript', row['created_at'], row['text'], {'family': 'transcript'})
                elif source_type == 'image_observation' and self._table_exists(conn, 'image_observations'):
                    for row in conn.execute(
                        f"SELECT citation,observation_id,created_at,(caption || char(10) || visible_text) AS content FROM image_observations WHERE citation IN ({placeholders}) ORDER BY created_at,citation",
                        batch,
                    ):
                        add(row['citation'], row['observation_id'], 'Image observation', 'Image', row['created_at'], row['content'], {'family': 'image_observation'})
                elif source_type == 'favorite' and self._table_exists(conn, 'favorites'):
                    for row in conn.execute(
                        f"SELECT citation,favorite_id,timestamp,title,(title || char(10) || text) AS content FROM favorites WHERE citation IN ({placeholders}) ORDER BY timestamp,citation",
                        batch,
                    ):
                        add(row['citation'], row['favorite_id'], row['title'], 'Favorite', row['timestamp'], row['content'], {'family': 'favorite'})
                elif source_type == 'moment':
                    if self._table_exists(conn, 'moment_items'):
                        for row in conn.execute(
                            f'SELECT citation,moment_id,author_id,timestamp,text FROM moment_items WHERE citation IN ({placeholders}) ORDER BY timestamp,citation',
                            batch,
                        ):
                            add(row['citation'], row['moment_id'], 'Moment', row['author_id'], row['timestamp'], row['text'], {'family': 'moment'})
                    if self._table_exists(conn, 'moment_interactions'):
                        for row in conn.execute(
                            f'SELECT citation,interaction_id,interaction_type,actor_id,actor_name,timestamp,text FROM moment_interactions WHERE citation IN ({placeholders}) ORDER BY timestamp,citation',
                            batch,
                        ):
                            add(row['citation'], row['interaction_id'], row['interaction_type'], row['actor_name'] or row['actor_id'], row['timestamp'], row['text'], {'family': 'moment_interaction'})
                elif source_type == 'contact' and self._table_exists(conn, 'observations'):
                    grouped: dict[str, list[str]] = {}
                    source_ids: dict[str, str] = {}
                    timestamps: dict[str, str] = {}
                    for row in conn.execute(
                        f'''SELECT citation,entity_id,updated_at,value_json
                            FROM observations
                            WHERE source_type='contact' AND citation IN ({placeholders})
                            ORDER BY citation,observation_id''',
                        batch,
                    ):
                        citation = str(row['citation'])
                        try:
                            value = json.loads(row['value_json'] or '{}')
                        except json.JSONDecodeError:
                            value = {}
                        text = value.get('text') if isinstance(value, dict) else str(value)
                        if text:
                            grouped.setdefault(citation, []).append(str(text))
                        source_ids[citation] = str(row['entity_id'] or '')
                        timestamps[citation] = str(row['updated_at'] or '')
                    for citation, values in grouped.items():
                        add(citation, source_ids.get(citation, ''), 'Contact observation', 'Contact', timestamps.get(citation, ''), '\n'.join(values), {'family': 'contact'})
            vector_citations = list(dict.fromkeys([*deleted_chunk_citations, *unique]))
            if vector_citations and self._table_exists(conn, 'vector_entries'):
                for start in range(0, len(vector_citations), 500):
                    batch = vector_citations[start:start + 500]
                    placeholders = ','.join('?' for _ in batch)
                    conn.execute(f'DELETE FROM vector_entries WHERE citation IN ({placeholders})', batch)
            dirty_recorded = self._record_dirty_refs_conn(conn, (
                {
                    'citation': citation,
                    'account_id': '',
                    'conversation_id': '',
                    'source_type': source_type,
                }
                for citation in unique
            ))
            conn.commit()
        return {
            'parents': parents,
            'chunks': created,
            'source_type': source_type,
            'citations': len(unique),
            'dirty_recorded': dirty_recorded,
            'max_chars': max_chars,
            'overlap_chars': overlap_chars,
        }

    def rebuild_evidence_chunks_for_source_citations(
        self,
        source_type: str,
        citations: Iterable[str],
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        """Named projection API for exact changed and deleted citations."""
        return self.upsert_evidence_chunks_for_source_citations(
            source_type,
            citations,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

    def rebuild_message_chunks_for_conversations(
        self,
        conversation_keys: Iterable[tuple[str, str]],
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        """Refresh message evidence chunks for a bounded conversation set.

        Full chunk rebuilds are safe but expensive on real Vaults.  Sync only
        appends/updates message parents, so the minimal local read-model repair is:
        delete message chunks for changed conversations, then recreate those
        chunks from current message rows.  FTS5 external-content triggers keep the
        chunk_fts table in step.
        """
        self.initialize()
        unique_keys = sorted({(str(a), str(c)) for a, c in conversation_keys if a and c})
        if not unique_keys:
            return {'parents': 0, 'chunks': 0, 'conversations': 0, 'max_chars': max_chars, 'overlap_chars': overlap_chars}
        created = 0
        parents = 0
        deleted_vectors = 0
        dirty_recorded = 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        with self.connect() as conn:
            deleted_chunk_citations: list[str] = []
            for account_id, conversation_id in unique_keys:
                dirty_recorded += self._record_dirty_chunk_parents_conn(
                    conn,
                    where="source_type='message' AND account_id=? AND source_id=?",
                    params=(account_id, conversation_id),
                )
                deleted_chunk_citations.extend([
                    row['chunk_citation'] for row in conn.execute(
                        "SELECT chunk_citation FROM evidence_chunks WHERE source_type='message' AND account_id=? AND source_id=?",
                        (account_id, conversation_id),
                    )
                ])
                conn.execute(
                    "DELETE FROM evidence_chunks WHERE source_type='message' AND account_id=? AND source_id=?",
                    (account_id, conversation_id),
                )
                for row in conn.execute(
                    """SELECT citation,account_id,account_label,conversation_id,conversation_title,sender_name,timestamp,content,content_kind,source_type,direction
                       FROM messages WHERE account_id=? AND conversation_id=? ORDER BY timestamp,citation""",
                    (account_id, conversation_id),
                ):
                    content_kind = row['content_kind'] if 'content_kind' in row.keys() else 'text'
                    content = display_content_for_kind(row['content'], content_kind)
                    parts = chunk_text(content or row['conversation_title'] or row['source_type'], max_chars=max_chars, overlap_chars=overlap_chars)
                    if not parts:
                        continue
                    parents += 1
                    for idx, part in enumerate(parts):
                        chunk_citation = f"{row['citation']}#chunk-{idx}"
                        conn.execute(
                            """INSERT OR REPLACE INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                chunk_citation,
                                chunk_citation,
                                row['citation'],
                                row['account_id'] or '',
                                row['account_label'] or row['account_id'] or '',
                                row['source_type'],
                                row['conversation_id'] or '',
                                row['conversation_title'] or '',
                                row['sender_name'] or '',
                                row['timestamp'] or '',
                                part,
                                idx,
                                json.dumps({'family': 'message', 'content_kind': content_kind, 'direction': row['direction'] or 'unknown'}, ensure_ascii=False),
                                'active',
                                now,
                            ),
                        )
                        created += 1
                dirty_recorded += self._record_dirty_chunk_parents_conn(
                    conn,
                    where="source_type='message' AND account_id=? AND source_id=?",
                    params=(account_id, conversation_id),
                )
            if deleted_chunk_citations and self._table_exists(conn, 'vector_entries'):
                deleted_vectors += self._delete_vector_entries_conn(conn, deleted_chunk_citations)
            conn.commit()
        return {'parents': parents, 'chunks': created, 'deleted_vectors': deleted_vectors, 'dirty_recorded': dirty_recorded, 'conversations': len(unique_keys), 'max_chars': max_chars, 'overlap_chars': overlap_chars}

    def rebuild_message_chunks_for_citations(
        self,
        citations: Iterable[str],
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        self.initialize()
        unique = list(dict.fromkeys(str(citation) for citation in citations if citation))
        with self.connect() as conn:
            report = self._rebuild_message_chunks_for_citations_conn(conn, unique, max_chars=max_chars, overlap_chars=overlap_chars)
            report['dirty_recorded'] = self._record_dirty_refs_conn(
                conn,
                ({'citation': citation, 'source_type': 'message'} for citation in unique),
            )
            conn.commit()
            return report

    def _delete_vector_entries_conn(self, conn: sqlite3.Connection, citations: Iterable[str]) -> int:
        unique = list(dict.fromkeys(str(citation) for citation in citations if citation))
        if not unique or not self._table_exists(conn, 'vector_entries'):
            return 0
        removed = 0
        for start in range(0, len(unique), 500):
            batch = unique[start:start + 500]
            placeholders = ','.join('?' for _ in batch)
            cursor = conn.execute(f'DELETE FROM vector_entries WHERE citation IN ({placeholders})', batch)
            removed += max(cursor.rowcount, 0)
        return removed

    def _record_dirty_refs_conn(self, conn: sqlite3.Connection, refs: Iterable[dict[str, str]]) -> int:
        rows = []
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        for ref in refs:
            citation = str(ref.get('citation') or '')
            if not citation:
                continue
            rows.append((
                citation,
                str(ref.get('account_id') or ''),
                str(ref.get('conversation_id') or ''),
                str(ref.get('source_type') or ''),
                now,
            ))
        if not rows:
            return 0
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
        return len({row[0] for row in rows})

    def _record_dirty_chunk_parents_conn(
        self,
        conn: sqlite3.Connection,
        *,
        where: str = '1=1',
        params: Iterable[Any] = (),
    ) -> int:
        """Journal existing chunk parents inside the same projection transaction."""

        if not self._table_exists(conn, 'evidence_chunks'):
            return 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        cursor = conn.execute(
            f"""INSERT OR REPLACE INTO sync_dirty_citations(
                       citation,account_id,conversation_id,source_type,updated_at
                   )
                   SELECT parent_citation,
                          MAX(account_id),
                          MAX(CASE WHEN source_type='message' THEN source_id ELSE '' END),
                          MAX(source_type),
                          ?
                     FROM evidence_chunks
                    WHERE parent_citation<>'' AND ({where})
                    GROUP BY parent_citation""",
            (now, *params),
        )
        return max(cursor.rowcount, 0)

    def record_citation_tombstones(self, refs: Iterable[dict[str, str]]) -> int:
        """Persist explicit deletion markers; identical repeats are true no-ops."""
        self.initialize()
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        rows: dict[str, tuple[str, str, str, str, str]] = {}
        for ref in refs:
            citation = str(ref.get('citation') or '')
            if citation:
                rows[citation] = (
                    citation,
                    str(ref.get('account_id') or ''),
                    str(ref.get('conversation_id') or ''),
                    str(ref.get('source_type') or ''),
                    now,
                )
        if not rows:
            return 0
        with self.connect() as conn:
            cursor = conn.executemany(
                """INSERT INTO sync_citation_tombstones(citation,account_id,conversation_id,source_type,deleted_at)
                   VALUES(?,?,?,?,?) ON CONFLICT(citation) DO NOTHING""",
                rows.values(),
            )
            changed = max(cursor.rowcount, 0)
            if changed:
                conn.commit()
            else:
                conn.rollback()
            return changed

    def clear_citation_tombstones(self, citations: Iterable[str]) -> int:
        unique = list(dict.fromkeys(str(value) for value in citations if value))
        if not unique:
            return 0
        self.initialize()
        removed = 0
        with self.connect() as conn:
            for start in range(0, len(unique), 500):
                batch = unique[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                cursor = conn.execute(
                    f'DELETE FROM sync_citation_tombstones WHERE citation IN ({placeholders})',
                    batch,
                )
                removed += max(cursor.rowcount, 0)
            if removed:
                conn.commit()
            else:
                conn.rollback()
        return removed

    def _stale_non_text_message_chunk_refs_conn(self, conn: sqlite3.Connection) -> list[dict[str, str]]:
        if not self._table_exists(conn, 'evidence_chunks') or not self._table_exists(conn, 'messages'):
            return []
        placeholder_case = (
            "CASE m.content_kind "
            "WHEN 'voice' THEN '[voice]' "
            "WHEN 'image' THEN '[image]' "
            "WHEN 'sticker' THEN '[sticker]' "
            "WHEN 'quote' THEN '[引用消息]' "
            "WHEN 'appmsg' THEN '[appmsg]' "
            "WHEN 'unknown_binary' THEN '[unknown_binary]' "
            "ELSE m.content END"
        )
        return [
            {
                'citation': str(row['citation']),
                'account_id': str(row['account_id'] or ''),
                'conversation_id': str(row['conversation_id'] or ''),
                'source_type': str(row['source_type'] or 'message'),
            }
            for row in conn.execute(
                f"""SELECT DISTINCT m.citation,m.account_id,m.conversation_id,m.source_type
                    FROM messages m
                    JOIN evidence_chunks e ON e.parent_citation=m.citation
                    WHERE e.source_type='message'
                      AND m.content_kind<>'text'
                      AND e.content <> {placeholder_case}
                    ORDER BY m.citation"""
            )
        ]

    def _rebuild_message_chunks_for_citations_conn(
        self,
        conn: sqlite3.Connection,
        citations: Iterable[str],
        *,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        unique = list(dict.fromkeys(str(citation) for citation in citations if citation))
        if not unique:
            return {'parents': 0, 'chunks': 0, 'citations': 0, 'deleted_chunks': 0, 'deleted_vectors': 0, 'max_chars': max_chars, 'overlap_chars': overlap_chars}
        created = 0
        parents = 0
        deleted_chunks = 0
        deleted_vectors = 0
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        for start in range(0, len(unique), 500):
            batch = unique[start:start + 500]
            placeholders = ','.join('?' for _ in batch)
            deleted_chunk_citations = [
                row['chunk_citation'] for row in conn.execute(
                    f"SELECT chunk_citation FROM evidence_chunks WHERE source_type='message' AND parent_citation IN ({placeholders})",
                    batch,
                )
            ] if self._table_exists(conn, 'evidence_chunks') else []
            if deleted_chunk_citations:
                conn.execute(
                    f"DELETE FROM evidence_chunks WHERE source_type='message' AND parent_citation IN ({placeholders})",
                    batch,
                )
                deleted_chunks += len(deleted_chunk_citations)
                deleted_vectors += self._delete_vector_entries_conn(conn, [*deleted_chunk_citations, *batch])
            elif self._table_exists(conn, 'vector_entries'):
                deleted_vectors += self._delete_vector_entries_conn(conn, batch)
            for row in conn.execute(
                f"""SELECT citation,account_id,account_label,conversation_id,conversation_title,sender_name,timestamp,content,content_kind,source_type,direction
                    FROM messages
                    WHERE citation IN ({placeholders})
                    ORDER BY timestamp,citation""",
                batch,
            ):
                content_kind = row['content_kind'] if 'content_kind' in row.keys() else 'text'
                content = display_content_for_kind(row['content'], content_kind)
                parts = chunk_text(content or row['conversation_title'] or row['source_type'], max_chars=max_chars, overlap_chars=overlap_chars)
                if not parts:
                    continue
                parents += 1
                for idx, part in enumerate(parts):
                    chunk_citation = f"{row['citation']}#chunk-{idx}"
                    conn.execute(
                        """INSERT OR REPLACE INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            chunk_citation,
                            chunk_citation,
                            row['citation'],
                            row['account_id'] or '',
                            row['account_label'] or row['account_id'] or '',
                            row['source_type'],
                            row['conversation_id'] or '',
                            row['conversation_title'] or '',
                            row['sender_name'] or '',
                            row['timestamp'] or '',
                            part,
                            idx,
                            json.dumps({'family': 'message', 'content_kind': content_kind, 'direction': row['direction'] or 'unknown'}, ensure_ascii=False),
                            'active',
                            now,
                        ),
                    )
                    created += 1
        return {'parents': parents, 'chunks': created, 'citations': len(unique), 'deleted_chunks': deleted_chunks, 'deleted_vectors': deleted_vectors, 'max_chars': max_chars, 'overlap_chars': overlap_chars}

    def chunk_search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10, *, allow_like_fallback: bool = True, actor_prefilter: bool = False) -> list[EvidenceRow]:
        if not self.path.exists() or not query:
            return []
        filters = filters or {}
        out: list[EvidenceRow] = []
        with self.connect() as conn:
            if not self._table_exists(conn, 'evidence_chunks'):
                return []
            if not self._conversation_filter_exists(conn, filters):
                return []
            filter_clauses, filter_params = self._chunk_filter_sql(filters, alias='e')
            queryable_chunk = _queryable_transcript_chunk_sql('e')
            rows: list[sqlite3.Row] = []
            if _fts_can_match(query) and self._table_exists(conn, 'chunk_fts'):
                where = ['chunk_fts MATCH ?', "e.status='active'", queryable_chunk] + filter_clauses
                phrase_like = f'%{_like_escape(query)}%'
                try:
                    rows = list(conn.execute(
                            f"""SELECT e.* FROM chunk_fts f JOIN evidence_chunks e ON e.rowid=f.rowid
                                WHERE {' AND '.join(where)}
                                ORDER BY bm25(chunk_fts, 0.0, 1.0, 3.5, 0.8),
                                         CASE
                                           WHEN e.title LIKE ? ESCAPE '\\' THEN 0
                                           WHEN e.content LIKE ? ESCAPE '\\' THEN 1
                                           ELSE 2
                                         END,
                                         e.timestamp DESC
                                LIMIT ?""",
                            (_fts_phrase(query), *filter_params, phrase_like, phrase_like, limit),
                        ))
                except sqlite3.OperationalError:
                    rows = []
            else:
                rows = rows
            if not rows and allow_like_fallback and len(''.join(str(query or '').split())) >= 6 and self._allow_chunk_like_fallback(conn, filters):
                pattern = _spanning_like_pattern(query)
                if pattern:
                    content_expr = "replace(replace(replace(e.content, char(13), ' '), char(10), ' '), char(9), ' ')"
                    where = ["e.status='active'", queryable_chunk, f"({content_expr} LIKE ? ESCAPE '\\' OR e.content LIKE ? ESCAPE '\\')"] + filter_clauses
                    actor_where, actor_params, _remaining = self._chunk_actor_prefilter_sql(filters) if actor_prefilter else (None, [], {})
                    if actor_where:
                        rows = list(conn.execute(
                            f"""WITH actor_scope AS (
                                    SELECT e.* FROM evidence_chunks e
                                    WHERE e.status='active' AND {queryable_chunk} AND {actor_where}
                                    ORDER BY timestamp DESC LIMIT ?
                                )
                                SELECT e.* FROM actor_scope e
                                WHERE ({content_expr} LIKE ? ESCAPE '\\' OR e.content LIKE ? ESCAPE '\\')
                                ORDER BY e.timestamp DESC LIMIT ?""",
                            (*actor_params, self._sender_prefilter_limit(), pattern, pattern, limit),
                        ))
                    else:
                        rows = list(conn.execute(
                            f"""SELECT e.* FROM evidence_chunks e
                                WHERE {' AND '.join(where)}
                                ORDER BY e.timestamp DESC LIMIT ?""",
                            (pattern, pattern, *filter_params, limit),
                        ))
            if not rows and allow_like_fallback and not _fts_can_match(query) and self._allow_chunk_like_fallback(conn, filters):
                like = f'%{query}%'
                where = ["e.status='active'", queryable_chunk, '(e.content LIKE ? OR e.title LIKE ? OR e.actor LIKE ?)'] + filter_clauses
                actor_where, actor_params, _remaining = self._chunk_actor_prefilter_sql(filters) if actor_prefilter else (None, [], {})
                if actor_where:
                    rows = list(conn.execute(
                        f"""WITH actor_scope AS (
                                SELECT e.* FROM evidence_chunks e
                                WHERE e.status='active' AND {queryable_chunk} AND {actor_where}
                                ORDER BY timestamp DESC LIMIT ?
                            )
                            SELECT e.* FROM actor_scope e
                            WHERE (e.content LIKE ? OR e.title LIKE ? OR e.actor LIKE ?)
                            ORDER BY e.timestamp DESC LIMIT ?""",
                        (*actor_params, self._sender_prefilter_limit(), like, like, like, limit),
                    ))
                else:
                    rows = list(conn.execute(
                        f"""SELECT e.* FROM evidence_chunks e
                            WHERE {' AND '.join(where)}
                            ORDER BY e.timestamp DESC LIMIT ?""",
                        (like, like, like, *filter_params, limit),
                    ))
            if not rows and allow_like_fallback and self._allow_chunk_like_fallback(conn, filters):
                terms = important_terms(query)
                if terms:
                    term_clauses = []
                    like_params: list[Any] = []
                    for term in terms:
                        term_clauses.append('(e.content LIKE ? OR e.title LIKE ? OR e.actor LIKE ?)')
                        term_like = f'%{term}%'
                        like_params.extend([term_like, term_like, term_like])
                    where = ["e.status='active'", queryable_chunk, *term_clauses, *filter_clauses]
                    actor_where, actor_params, _remaining = self._chunk_actor_prefilter_sql(filters) if actor_prefilter else (None, [], {})
                    if actor_where:
                        rows = list(conn.execute(
                            f"""WITH actor_scope AS (
                                    SELECT e.* FROM evidence_chunks e
                                    WHERE e.status='active' AND {queryable_chunk} AND {actor_where}
                                    ORDER BY timestamp DESC LIMIT ?
                                )
                                SELECT e.* FROM actor_scope e
                                WHERE {' AND '.join(term_clauses)}
                                ORDER BY e.timestamp DESC LIMIT ?""",
                            (*actor_params, self._sender_prefilter_limit(), *like_params, limit),
                        ))
                    else:
                        rows = list(conn.execute(
                            f"""SELECT e.* FROM evidence_chunks e
                                WHERE {' AND '.join(where)}
                                ORDER BY e.timestamp DESC LIMIT ?""",
                            (*like_params, *filter_params, limit),
                        ))
            for row in rows:
                ev = self._chunk_row_to_evidence(row)
                if self._filter_row(ev, filters):
                    out.append(ev)
                if len(out) >= limit:
                    break
        return out

    def iter_vector_documents(self, batch_size: int = 500, citations: Iterable[str] | None = None):
        if not self.path.exists():
            return
        citation_filter = None if citations is None else list(dict.fromkeys(str(c) for c in citations if c))
        if citation_filter is not None and not citation_filter:
            return
        conn = self.connect_once()
        try:
            if citation_filter is not None:
                conn.execute('CREATE TEMP TABLE IF NOT EXISTS _trove_vector_dirty_citations(citation TEXT PRIMARY KEY)')
                conn.execute('DELETE FROM _trove_vector_dirty_citations')
                conn.executemany(
                    'INSERT OR IGNORE INTO _trove_vector_dirty_citations(citation) VALUES(?)',
                    [(citation,) for citation in citation_filter],
                )
            if self._table_exists(conn, 'evidence_chunks') and int(conn.execute('SELECT COUNT(*) FROM evidence_chunks').fetchone()[0]) > 0:
                if citation_filter is None:
                    cursor = conn.execute(f"""SELECT e.chunk_citation AS citation,e.parent_citation,e.account_id,e.account_label,e.source_id AS conversation_id,e.title AS conversation_title,'private' AS conversation_type,e.actor AS sender_id,e.actor AS sender_name,e.timestamp,e.content,e.source_type,'metadata' AS direction
                                              FROM evidence_chunks e
                                             WHERE e.status='active' AND {_queryable_transcript_chunk_sql('e')}
                                          ORDER BY e.timestamp,e.chunk_citation""")
                else:
                    cursor = conn.execute(
                        f"""SELECT e.chunk_citation AS citation,e.parent_citation,e.account_id,e.account_label,e.source_id AS conversation_id,e.title AS conversation_title,'private' AS conversation_type,e.actor AS sender_id,e.actor AS sender_name,e.timestamp,e.content,e.source_type,'metadata' AS direction
                           FROM evidence_chunks e
                           JOIN _trove_vector_dirty_citations d
                             ON d.citation=e.parent_citation OR d.citation=e.chunk_citation
                           WHERE e.status='active' AND {_queryable_transcript_chunk_sql('e')}
                           ORDER BY e.timestamp,e.chunk_citation"""
                    )
            else:
                if citation_filter is None:
                    cursor = conn.execute('SELECT * FROM messages ORDER BY timestamp, citation')
                else:
                    cursor = conn.execute(
                        """SELECT m.* FROM messages m
                           JOIN _trove_vector_dirty_citations d ON d.citation=m.citation
                           ORDER BY m.timestamp,m.citation"""
                    )
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    data = dict(row)
                    data['vector_text'] = vector_document_text(data)
                    yield EvidenceRow(data)
        finally:
            conn.close()

    def purge_vectors(self, *, backend: str = 'all') -> dict[str, Any]:
        removed = 0
        if self.path.exists():
            self.initialize()
            with self.connect() as conn:
                if backend in {'all', 'sqlite'} and self._table_exists(conn, 'vector_entries'):
                    removed = int(conn.execute('SELECT COUNT(*) FROM vector_entries').fetchone()[0])
                    conn.execute('DELETE FROM vector_entries')
                    conn.commit()
        return {'backend': backend, 'removed_sqlite_entries': removed}


    def all_messages(self) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        with self.connect() as conn:
            return list(conn.execute('SELECT * FROM messages ORDER BY timestamp, citation'))

    def iter_messages(self, batch_size: int = 500):
        if not self.path.exists():
            return
        conn = self.connect_once()
        try:
            cursor = conn.execute('SELECT * FROM messages ORDER BY timestamp, citation')
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    yield row
        finally:
            conn.close()



    def list_conversations(self, limit: int = 100) -> list[sqlite3.Row]:
        from trove_core.bounds import BoundedLimit, PRIVATE_LIST

        limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
        if not self.path.exists():
            return []
        with self.connect() as conn:
            return list(conn.execute('SELECT * FROM conversations ORDER BY account_id, title LIMIT ?', (limit,)))

    def list_contacts(self, limit: int = 100) -> list[dict[str, Any]]:
        from trove_core.bounds import BoundedLimit, PRIVATE_LIST

        limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
        if not self.path.exists():
            return []
        with self.connect() as conn:
            if not self._table_exists(conn, 'entities'):
                return []
            return [dict(row) for row in conn.execute("SELECT entity_id, entity_type, display_name, identifiers_json, confidence FROM entities WHERE entity_type IN ('Customer','Person','Organization') ORDER BY display_name LIMIT ?", (limit,))]

    def list_moments(self, limit: int = 100) -> list[dict[str, Any]]:
        from trove_core.bounds import BoundedLimit, PRIVATE_LIST

        limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
        if not self.path.exists():
            return []
        with self.connect() as conn:
            if not self._table_exists(conn, 'moment_items'):
                return []
            return [dict(row) for row in conn.execute('SELECT moment_id, account_id, author_id, citation, timestamp, substr(text,1,240) AS text, status FROM moment_items ORDER BY timestamp DESC LIMIT ?', (limit,))]

    def list_favorites(self, limit: int = 100) -> list[dict[str, Any]]:
        from trove_core.bounds import BoundedLimit, PRIVATE_LIST

        limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
        if not self.path.exists():
            return []
        with self.connect() as conn:
            if not self._table_exists(conn, 'favorites'):
                return []
            return [dict(row) for row in conn.execute('SELECT favorite_id, account_id, citation, timestamp, title, substr(text,1,240) AS text FROM favorites ORDER BY timestamp DESC LIMIT ?', (limit,))]

    def purge_excluded_scope(self) -> dict[str, Any]:
        """Idempotently purge known excluded pre-scope rows from queryable surfaces."""
        from trove_core.wechat.scope import classify_wechat_identity

        def stored_identity(row: sqlite3.Row) -> str:
            # Be conservative for already-ingested Vaults: conversation titles are user-facing
            # labels and can contain words like "service" without being official accounts.
            # Only purge when the stored opaque conversation id itself matches a documented
            # excluded WeChat identity pattern.
            conversation_id = str(row['conversation_id'] or '').strip()
            if row['type'] == 'group' and not conversation_id.endswith('@chatroom'):
                return conversation_id + '@chatroom'
            return conversation_id

        if not self.path.exists():
            return {'purged_messages': 0, 'purged_conversations': 0, 'rebuilt_fts': 0}
        self.initialize()
        with self.connect() as conn:
            bad_conversations = []
            for row in conn.execute('SELECT account_id, conversation_id, title, type FROM conversations'):
                decision = classify_wechat_identity(stored_identity(row), has_chat_history=True)
                if not decision.allowed and decision.scope_type != 'excluded_unknown':
                    bad_conversations.append((row['account_id'], row['conversation_id']))
            purged_messages = 0
            for account_id, conversation_id in bad_conversations:
                rows = list(conn.execute('SELECT id, citation FROM messages WHERE account_id=? AND conversation_id=?', (account_id, conversation_id)))
                for row in rows:
                    if self._table_exists(conn, 'vector_entries'):
                        conn.execute('DELETE FROM vector_entries WHERE citation=?', (row['citation'],))
                purged_messages += len(rows)
                conn.execute('DELETE FROM messages WHERE account_id=? AND conversation_id=?', (account_id, conversation_id))
                conn.execute('DELETE FROM conversations WHERE account_id=? AND conversation_id=?', (account_id, conversation_id))
            conn.execute('DELETE FROM message_fts')
            conn.execute('INSERT INTO message_fts(rowid,citation,content,sender_name,conversation_title) SELECT id,citation,content,sender_name,conversation_title FROM messages ORDER BY id')
            remaining = int(conn.execute('SELECT COUNT(*) FROM message_fts').fetchone()[0])
            conn.commit()
        return {'purged_messages': purged_messages, 'purged_conversations': len(bad_conversations), 'rebuilt_fts': remaining}

    def purge_derived_data(
        self,
        *,
        scope_type: str,
        scope_id: str,
        purge_id: str,
        scope_hash: str,
        lifecycle_version: str,
        audit_retention_days: int = 365,
    ) -> dict[str, Any]:
        """Delete one person's/source's/run's derived graph in dependency order.

        Filesystem deletion is intentionally returned as opaque internal refs
        and performed by the Vault operation while it holds the same writer
        boundary. Public reports must strip all keys prefixed with ``_``.
        """

        from datetime import timedelta

        if scope_type not in {'entity', 'source', 'run', 'task'}:
            raise ValueError('scope_type must be entity, source, run, or task')
        if type(scope_id) is not str or not scope_id or len(scope_id) > 1000:
            raise ValueError('scope_id must be non-empty bounded text')
        if type(audit_retention_days) is not int or not 1 <= audit_retention_days <= 3650:
            raise ValueError('audit_retention_days must be from 1 to 3650')
        self.initialize()

        def chunks(values: set[str], size: int = 300):
            ordered = sorted(values)
            for start in range(0, len(ordered), size):
                yield ordered[start:start + size]

        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            counts: dict[str, int] = {}
            entity_ids: set[str] = set()
            source_revisions: set[str] = set()
            run_ids: set[str] = set()
            task_ids: set[str] = set()
            asset_ids: set[str] = set()
            citations: set[str] = set()
            conversation_ids: set[str] = set()
            moment_ids: set[str] = set()
            identity_values: set[str] = set()
            approval_ids: set[str] = set()
            file_refs: set[str] = set()
            source_root_refs: set[str] = set()
            content_hashes: set[str] = set()
            affected_profile_entities: set[str] = set()

            def add_count(table: str, amount: int) -> None:
                counts[table] = counts.get(table, 0) + max(int(amount), 0)

            def delete_in(table: str, column: str, values: set[str]) -> None:
                if not values or not self._table_exists(conn, table):
                    return
                for batch in chunks(values):
                    marks = ','.join('?' for _ in batch)
                    cursor = conn.execute(f'DELETE FROM {table} WHERE {column} IN ({marks})', batch)
                    add_count(table, cursor.rowcount)

            def select_rows(table: str, column: str, values: set[str], fields: str = '*') -> list[sqlite3.Row]:
                if not values or not self._table_exists(conn, table):
                    return []
                rows: list[sqlite3.Row] = []
                for batch in chunks(values):
                    marks = ','.join('?' for _ in batch)
                    rows.extend(conn.execute(f'SELECT {fields} FROM {table} WHERE {column} IN ({marks})', batch))
                return rows

            def citation_matches(table: str, column: str, values: set[str], fields: str = '*') -> list[sqlite3.Row]:
                if not values or not self._table_exists(conn, table):
                    return []
                bases = {value.split('#', 1)[0] for value in values if value}
                rows: list[sqlite3.Row] = []
                for batch in chunks(bases, 120):
                    clauses: list[str] = []
                    params: list[str] = []
                    for value in batch:
                        clauses.extend((f'{column}=?', f'{column} LIKE ?'))
                        params.extend((value, value + '#%'))
                    rows.extend(conn.execute(
                        f"SELECT {fields} FROM {table} WHERE {' OR '.join(clauses)}",
                        params,
                    ))
                return rows

            if scope_type == 'entity':
                entity_ids.add(scope_id)
                entity = conn.execute('SELECT identifiers_json FROM entities WHERE entity_id=?', (scope_id,)).fetchone()
                if entity is None:
                    conn.rollback()
                    raise ValueError('entity purge target does not exist')
                try:
                    identifiers = json.loads(entity['identifiers_json'] or '{}')
                except json.JSONDecodeError:
                    identifiers = {}
                for key in ('conversation_ids', 'sender_ids', 'aliases'):
                    value = identifiers.get(key)
                    if isinstance(value, list):
                        identity_values.update(str(item) for item in value if item)
                for key in ('primary_user_id', 'wechat_id', 'wechat_username', 'username', 'source_entity_ref'):
                    value = identifiers.get(key)
                    if value:
                        identity_values.add(str(value).removeprefix('unresolved:'))
                conversation_ids.update(identity_values)
                run_ids.update(str(row[0]) for row in conn.execute(
                    'SELECT run_id FROM profile_enrichment_runs WHERE entity_id=?', (scope_id,),
                ))
                for row in conn.execute('SELECT citation FROM observations WHERE entity_id=?', (scope_id,)):
                    citations.add(str(row[0]))
                for row in conn.execute('SELECT evidence_citations_json FROM profile_snapshots WHERE entity_id=?', (scope_id,)):
                    try:
                        citations.update(str(value) for value in json.loads(row[0] or '[]') if value)
                    except json.JSONDecodeError:
                        pass
                if identity_values:
                    for batch in chunks(identity_values):
                        marks = ','.join('?' for _ in batch)
                        for row in conn.execute(
                            f'''SELECT citation,conversation_id FROM messages
                                WHERE conversation_id IN ({marks}) OR sender_id IN ({marks})''',
                            (*batch, *batch),
                        ):
                            citations.add(str(row['citation']))
                            conversation_ids.add(str(row['conversation_id']))
                        for row in conn.execute(
                            f'SELECT citation,moment_id FROM moment_items WHERE author_id IN ({marks})', batch,
                        ):
                            citations.add(str(row['citation']))
                            moment_ids.add(str(row['moment_id']))
                        for row in conn.execute(
                            f'SELECT citation,moment_id FROM moment_interactions WHERE actor_id IN ({marks})', batch,
                        ):
                            citations.add(str(row['citation']))
                            moment_ids.add(str(row['moment_id']))
            elif scope_type == 'source':
                source_revisions.add(scope_id)
                snapshot = conn.execute('SELECT root_ref FROM source_snapshots WHERE snapshot_revision=?', (scope_id,)).fetchone()
                if snapshot is None:
                    conn.rollback()
                    raise ValueError('source purge target does not exist')
                if snapshot['root_ref']:
                    source_root_refs.add(str(snapshot['root_ref']))
                asset_ids.update(str(row[0]) for row in conn.execute(
                    'SELECT asset_id FROM media_source_bindings WHERE snapshot_revision=?', (scope_id,),
                ))
            elif scope_type == 'run':
                if conn.execute('SELECT 1 FROM profile_enrichment_runs WHERE run_id=?', (scope_id,)).fetchone() is None:
                    conn.rollback()
                    raise ValueError('run purge target does not exist')
                run_ids.add(scope_id)
            else:
                task = conn.execute('SELECT run_id FROM profile_enrichment_tasks WHERE task_id=?', (scope_id,)).fetchone()
                if task is None:
                    conn.rollback()
                    raise ValueError('task purge target does not exist')
                task_ids.add(scope_id)
                run_ids.add(str(task['run_id']))

            if run_ids and scope_type != 'task':
                task_ids.update(str(row['task_id']) for row in select_rows(
                    'profile_enrichment_tasks', 'run_id', run_ids, 'task_id',
                ))
            task_rows = select_rows('profile_enrichment_tasks', 'task_id', task_ids)
            for row in task_rows:
                if row['asset_id']:
                    asset_ids.add(str(row['asset_id']))
                if row['citation']:
                    citations.add(str(row['citation']))
                if row['approval_id']:
                    approval_ids.add(str(row['approval_id']))

            # Assets reachable from selected citations or links become part of
            # the derivative closure. This is deliberately conservative.
            if citations:
                bases = {value.split('#', 1)[0] for value in citations}
                for row in select_rows('media_assets', 'citation', bases, 'asset_id'):
                    asset_ids.add(str(row['asset_id']))
                for row in select_rows('media_asset_links', 'source_citation', bases, 'asset_id'):
                    asset_ids.add(str(row['asset_id']))
            if scope_type == 'source' and asset_ids:
                linked_tasks = select_rows('profile_enrichment_tasks', 'asset_id', asset_ids)
                for row in linked_tasks:
                    task_ids.add(str(row['task_id']))
                    run_ids.add(str(row['run_id']))
                    citations.add(str(row['citation']))
                    if row['approval_id']:
                        approval_ids.add(str(row['approval_id']))
            asset_rows = select_rows('media_assets', 'asset_id', asset_ids)
            for row in asset_rows:
                citations.add(str(row['citation']))
                if row['content_hash']:
                    content_hashes.add(str(row['content_hash']))
                if row['path_ref']:
                    file_refs.add(str(row['path_ref']))
            for row in select_rows('media_asset_links', 'asset_id', asset_ids, 'source_citation'):
                citations.add(str(row['source_citation']))
            for row in select_rows('media_decode_results', 'asset_id', asset_ids, 'derivative_ref'):
                if row['derivative_ref']:
                    file_refs.add(str(row['derivative_ref']))
            for row in select_rows('image_observations', 'asset_id', asset_ids, 'content_sha256'):
                if row['content_sha256']:
                    content_hashes.add(str(row['content_sha256']))

            # Evidence/vector derivatives must be captured before their rows
            # are removed so every chunk/vector citation is tombstoned too.
            chunk_rows = citation_matches('evidence_chunks', 'parent_citation', citations, 'chunk_citation')
            chunk_citations = {str(row['chunk_citation']) for row in chunk_rows}
            vector_citations = set(citations) | chunk_citations
            snapshot_ids: set[str] = set()
            citation_bases = {value.split('#', 1)[0] for value in citations}
            if citation_bases and self._table_exists(conn, 'profile_snapshots'):
                for row in conn.execute('SELECT profile_id,evidence_citations_json FROM profile_snapshots'):
                    try:
                        evidence = {str(value).split('#', 1)[0] for value in json.loads(row['evidence_citations_json'] or '[]') if value}
                    except json.JSONDecodeError:
                        evidence = set()
                    if evidence.intersection(citation_bases):
                        snapshot_ids.add(str(row['profile_id']))
            if scope_type != 'entity':
                for row in select_rows(
                    'profile_snapshots', 'run_id', run_ids, 'profile_id,entity_id',
                ):
                    snapshot_ids.add(str(row['profile_id']))
                    affected_profile_entities.add(str(row['entity_id']))
                for row in select_rows(
                    'profile_snapshots', 'profile_id', snapshot_ids, 'entity_id',
                ):
                    affected_profile_entities.add(str(row['entity_id']))

            # Dependency order: snapshot/task -> provider projection -> search
            # projection -> payload/ontology -> media -> source rows.
            if scope_type == 'entity':
                delete_in('profile_refresh_queue', 'entity_id', entity_ids)
                delete_in('profile_automation_subscriptions', 'entity_id', entity_ids)
                delete_in('profile_snapshots', 'entity_id', entity_ids)
            else:
                delete_in('profile_snapshots', 'run_id', run_ids)
            delete_in('profile_snapshots', 'profile_id', snapshot_ids)
            if affected_profile_entities:
                refresh_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                for affected_entity_id in sorted(affected_profile_entities):
                    latest = conn.execute(
                        """SELECT profile_id FROM profile_snapshots WHERE entity_id=?
                             ORDER BY version DESC,created_at DESC,profile_id DESC LIMIT 1""",
                        (affected_entity_id,),
                    ).fetchone()
                    subscription = conn.execute(
                        """SELECT enabled FROM profile_automation_subscriptions
                             WHERE entity_id=?""",
                        (affected_entity_id,),
                    ).fetchone()
                    if subscription is None:
                        continue
                    conn.execute(
                        """UPDATE profile_automation_subscriptions
                              SET last_profile_id=?,last_error_code=NULL,updated_at=?
                            WHERE entity_id=?""",
                        (
                            latest['profile_id'] if latest is not None else None,
                            refresh_at,
                            affected_entity_id,
                        ),
                    )
                    if bool(subscription['enabled']):
                        conn.execute(
                            """INSERT INTO profile_refresh_queue(
                                   entity_id,generation,state,reason,available_at,claimed_at,
                                   attempt_count,last_error_code,created_at,updated_at)
                               VALUES(?,1,'pending','derived_data_purged',?,NULL,0,NULL,?,?)
                               ON CONFLICT(entity_id) DO UPDATE SET
                                   generation=profile_refresh_queue.generation+1,
                                   state='pending',reason='derived_data_purged',available_at=excluded.available_at,
                                   claimed_at=NULL,attempt_count=0,last_error_code=NULL,
                                   updated_at=excluded.updated_at""",
                            (affected_entity_id, refresh_at, refresh_at, refresh_at),
                        )
            delete_in('profile_enrichment_tasks', 'task_id', task_ids)
            if scope_type in {'entity', 'run'}:
                delete_in('profile_enrichment_runs', 'run_id', run_ids)
            elif run_ids:
                for batch in chunks(run_ids):
                    marks = ','.join('?' for _ in batch)
                    conn.execute(
                        f"UPDATE profile_enrichment_runs SET state='cancelled',revoked_at=?,updated_at=? WHERE run_id IN ({marks})",
                        (datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), *batch),
                    )

            transcript_ids = {str(row['transcript_id']) for row in citation_matches('transcripts', 'citation', citations, 'transcript_id')}
            image_observation_ids = {str(row['observation_id']) for row in citation_matches('image_observations', 'citation', citations, 'observation_id')}
            provider_job_ids = {str(row['job_id']) for row in citation_matches('provider_jobs', 'citation', citations, 'job_id')}
            delete_in('transcripts', 'asset_id', asset_ids)
            delete_in('image_observations', 'asset_id', asset_ids)
            delete_in('provider_jobs', 'asset_id', asset_ids)
            delete_in('transcripts', 'transcript_id', transcript_ids)
            delete_in('image_observations', 'observation_id', image_observation_ids)
            delete_in('provider_jobs', 'job_id', provider_job_ids)
            delete_in('media_jobs', 'asset_id', asset_ids)
            delete_in('media_decode_results', 'asset_id', asset_ids)
            delete_in('media_understanding', 'content_sha256', content_hashes)

            delete_in('vector_entries', 'citation', vector_citations)
            delete_in('vector_index_ledger', 'citation', vector_citations)
            delete_in('evidence_chunks', 'chunk_citation', chunk_citations)
            evidence_ids = {str(row['evidence_id']) for row in citation_matches('evidence_items', 'citation', citations, 'evidence_id')}
            delete_in('evidence_items', 'evidence_id', evidence_ids)

            # AppMsg normalized payload is the task derivative even when the
            # raw message stays for run/task-only purges.
            base_citations = {value.split('#', 1)[0] for value in citations}
            delete_in('message_payloads', 'citation', base_citations)
            observation_ids = {str(row['observation_id']) for row in citation_matches('observations', 'citation', citations, 'observation_id')}
            delete_in('observations', 'observation_id', observation_ids)
            relationship_ids = {str(row['relationship_id']) for row in citation_matches('relationships', 'citation', citations, 'relationship_id')}
            delete_in('relationships', 'relationship_id', relationship_ids)

            delete_in('sns_cache_mappings', 'source_citation', base_citations)
            if scope_type in {'entity', 'source'}:
                delete_in('media_asset_links', 'asset_id', asset_ids)
                delete_in('media_source_bindings', 'asset_id', asset_ids)
                delete_in('media_source_rows', 'asset_id', asset_ids)
                delete_in('media_assets', 'asset_id', asset_ids)
            elif asset_ids:
                for batch in chunks(asset_ids):
                    marks = ','.join('?' for _ in batch)
                    conn.execute(
                        f'''UPDATE media_assets
                               SET path_ref=NULL,content_hash=NULL,processing_state='pending',
                                   cache_state=CASE WHEN EXISTS(
                                       SELECT 1 FROM media_source_bindings msb WHERE msb.asset_id=media_assets.asset_id
                                   ) THEN 'source_available' ELSE 'missing_local_cache' END,
                                   updated_at=?
                             WHERE asset_id IN ({marks})''',
                        (datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'), *batch),
                    )

            if scope_type == 'entity':
                delete_in('observations', 'entity_id', entity_ids)
                for table, column in (
                    ('relationships', 'subject_entity_id'),
                    ('relationships', 'object_entity_id'),
                    ('entity_identifiers', 'entity_id'),
                ):
                    delete_in(table, column, entity_ids)
                delete_in('messages', 'citation', base_citations)
                delete_in('moment_interactions', 'moment_id', moment_ids)
                delete_in('moment_items', 'moment_id', moment_ids)
                delete_in('favorites', 'citation', base_citations)
                if conversation_ids:
                    for batch in chunks(conversation_ids):
                        marks = ','.join('?' for _ in batch)
                        cursor = conn.execute(
                            f'''DELETE FROM conversations WHERE conversation_id IN ({marks})
                                AND NOT EXISTS(
                                    SELECT 1 FROM messages m
                                     WHERE m.account_id=conversations.account_id
                                       AND m.conversation_id=conversations.conversation_id
                                )''', batch,
                        )
                        add_count('conversations', cursor.rowcount)
                delete_in('entities', 'entity_id', entity_ids)
            elif scope_type == 'source':
                delete_in('messages', 'citation', base_citations)
                delete_in('source_snapshots', 'snapshot_revision', source_revisions)

            delete_in('sync_dirty_citations', 'citation', vector_citations)
            delete_in('sync_citation_tombstones', 'citation', vector_citations)
            delete_in('sync_message_source_rows', 'citation', base_citations)
            delete_in('approval_records', 'approval_id', approval_ids)

            now = datetime.now(timezone.utc)
            cursor = conn.execute(
                'DELETE FROM derived_data_purge_audit WHERE audit_retention_until<?',
                (now.isoformat().replace('+00:00', 'Z'),),
            )
            add_count('expired_purge_audit', cursor.rowcount)
            public_counts = dict(sorted(counts.items()))
            conn.execute(
                '''INSERT INTO derived_data_purge_audit(
                       purge_id,scope_type,scope_hash,lifecycle_version,status,counts_json,
                       backup_policy,audit_retention_until,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)''',
                (
                    purge_id, scope_type, scope_hash, lifecycle_version, 'completed',
                    json.dumps(public_counts, sort_keys=True, separators=(',', ':')),
                    'replace_all_pre_purge_backups_with_one_post_purge_backup',
                    (now + timedelta(days=audit_retention_days)).isoformat().replace('+00:00', 'Z'),
                    now.isoformat().replace('+00:00', 'Z'),
                ),
            )
            conn.commit()
        return {
            'ok': True,
            'purge_id': purge_id,
            'scope_type': scope_type,
            'scope_hash': scope_hash,
            'counts': public_counts,
            'audit_retention_days': audit_retention_days,
            '_approval_ids': sorted(approval_ids),
            '_file_refs': sorted(file_refs),
            '_source_root_refs': sorted(source_root_refs),
        }


def important_terms(query: str) -> list[str]:
    raw = str(query or '')
    q = raw.lower()
    dictionary = ['客户', '卡', '价格', '预算', '审批', '试点', '团队', '决定', '上线', 'token', 'evidence', 'citation', 'context', 'vault']
    terms = [term for term in dictionary if term in q]
    target_count = 2 if terms else 3
    if len(terms) >= target_count:
        return terms[:target_count]
    for token in re.findall(r'[A-Za-z][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}', raw):
        candidates: list[str]
        if re.fullmatch(r'[\u4e00-\u9fff]+', token) and len(token) > 4:
            mid = max(0, (len(token) - 4) // 2)
            candidates = [token[:4], token[mid:mid + 4], token[-4:]]
        else:
            candidates = [token]
        for candidate in candidates:
            if candidate and candidate not in terms:
                terms.append(candidate)
            if len(terms) >= target_count:
                return terms[:target_count]
    if not terms and ' ' in q:
        terms = [t for t in q.split() if len(t) >= 2]
    # Avoid too many broad terms; two good terms are enough for natural Chinese questions.
    return terms[:target_count]


def _like_escape(value: str) -> str:
    return str(value or '').replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def _spanning_like_pattern(query: str) -> str:
    parts = [part for part in re.split(r'\s+', str(query or '').strip()) if part]
    if not parts:
        return ''
    return '%' + '%'.join(_like_escape(part) for part in parts) + '%'


def _fts_can_match(query: str) -> bool:
    return len(''.join(str(query or '').split())) >= 3


def _fts_escape(value: str) -> str:
    return str(value or '').replace('"', '""')


def _fts_phrase(value: str) -> str:
    return f'"{_fts_escape(value)}"'


def _fts_and_query(terms: list[str]) -> str:
    return ' AND '.join(_fts_phrase(term) for term in terms if _fts_can_match(term))


def _message_batches(messages: Iterable[Message], *, size: int) -> Iterable[list[Message]]:
    batch: list[Message] = []
    for message in messages:
        batch.append(message)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _message_value_tuple(m: Message) -> tuple[Any, ...]:
    data = m.safe_dict()
    return (
        m.citation,
        m.account_id,
        m.account_label,
        m.conversation_id,
        m.conversation_title,
        m.conversation_type,
        m.sender_id,
        m.sender_name,
        data['timestamp'],
        m.content,
        m.content_kind,
        m.shard_id,
        m.local_id,
        int(m.sent_by_me),
        m.source_type,
        m.direction,
    )


def vector_document_text(row: Any) -> str:
    """Build local-only contextual text for embeddings.

    The vector text is never written to source artifacts; it gives local vector
    backends enough metadata to recover short/sparse queries while preserving the
    existing evidence citation as the durable lookup key.
    """
    def value(key: str) -> str:
        try:
            if hasattr(row, 'keys') and key not in row.keys():
                return ''
            return str(row[key] or '')
        except Exception:
            return ''

    content = display_content_for_kind(value('content'), value('content_kind') or 'text')
    semantic_tags: list[str] = []
    tag_rules = [
        ('商务条件/价格预算/付款异议', ('价格', '报价', '预算', '太贵', '费用', '成本', '付款', '合同')),
        ('决策进展/审批确认/上线试点', ('决定', '决策', '确认', '审批', '上线', '试点', '交付', '推进')),
        ('客户画像/联系人/需求痛点', ('客户', '老板', '负责人', '联系人', '团队', '需求', '痛点')),
        ('风险卡点/反对意见/推进阻碍', ('风险', '担心', '问题', '卡点', '阻碍', '异议', '不同意')),
        ('下一步行动/跟进承诺', ('下次', '明天', '后天', '周', '约', '跟进', '回访', '安排')),
    ]
    for tag, terms in tag_rules:
        if any(term in content for term in terms):
            semantic_tags.append(tag)

    parts = [
        '检索对象: 微信聊天证据',
        f"来源类型: {value('source_type')}",
        f"会话: {value('conversation_title')}",
        f"会话类型: {value('conversation_type')}",
        f"说话人: {value('sender_name')}",
        f"方向: {value('direction')}",
        f"时间: {value('timestamp')}",
        f"语义标签: {'; '.join(semantic_tags)}" if semantic_tags else '',
        f"证据正文: {content}",
    ]
    return '\n'.join(part for part in parts if part.strip())


def open_store(
    path: str | Path,
    *,
    readonly: bool = False,
    max_connections: int = 64,
    prepared_statement_cache_size: int = 128,
    page_cache_kib: int = 64_000,
    connection_wait_seconds: float = 1.0,
) -> SQLiteStore:
    """Open and validate a store; read-only mode never migrates or commits."""

    store = SQLiteStore(
        Path(path), readonly=readonly,
        max_connections=max_connections,
        prepared_statement_cache_size=prepared_statement_cache_size,
        page_cache_kib=page_cache_kib,
        connection_wait_seconds=connection_wait_seconds,
    )
    store.initialize()
    return store
