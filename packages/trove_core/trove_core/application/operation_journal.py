from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Iterator, Mapping

from trove_core.store.sqlite_store import SQLiteStore


TERMINAL_STATES = frozenset({'completed', 'failed', 'cancelled'})
OPERATION_STATES = frozenset({
    'pending', 'running', 'awaiting_agent', 'reconciling',
    'completed', 'failed', 'cancelled',
})
OWNERS = frozenset({'daemon', 'provider', 'agent', 'none'})
REPLAY_POLICIES = frozenset({'idempotent', 'journaled', 'never'})
STATE_OWNERS = {
    'pending': frozenset({'daemon'}),
    'running': frozenset({'daemon', 'provider'}),
    'awaiting_agent': frozenset({'agent'}),
    'reconciling': frozenset({'daemon', 'provider'}),
    'completed': frozenset({'none'}),
    'failed': frozenset({'none'}),
    'cancelled': frozenset({'none'}),
}


class OperationConflict(RuntimeError):
    code = 'operation_conflict'


class OperationNotFound(LookupError):
    code = 'operation_not_found'


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str
    capability_id: str
    request_digest: str
    idempotency_key: str
    replay_policy: str
    state: str
    stage: str
    owner: str
    result: Mapping[str, Any] | None
    error: Mapping[str, Any] | None
    has_continuation: bool
    external_ref: str | None
    version: int
    created_at: str
    updated_at: str

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'operation_id': self.operation_id,
            'capability_id': self.capability_id,
            'state': self.state,
            'stage': self.stage,
            'owner': self.owner,
            'replay_policy': self.replay_policy,
            'terminal': self.terminal,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
        if self.result is not None:
            payload['result'] = dict(self.result)
        if self.error is not None:
            payload['error'] = dict(self.error)
        if self.has_continuation:
            payload['continuation'] = {'required': True, 'owner': self.owner}
        if self.external_ref:
            payload['external_ref'] = self.external_ref
        return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _canonical_digest(capability_id: str, request: Mapping[str, Any]) -> str:
    try:
        body = json.dumps(
            {'capability': capability_id, 'input': request},
            ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise OperationConflict('operation input is not canonical JSON') from exc
    return hashlib.sha256(body).hexdigest()


def _json(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _decode(value: str | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise OperationConflict('journal JSON payload is corrupt')
    return payload


class OperationJournal:
    def __init__(self, database: str | Path | SQLiteStore):
        self.store = database if isinstance(database, SQLiteStore) else SQLiteStore(Path(database))
        self.store.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self.store.connect() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as conn:
            if conn.in_transaction:
                raise OperationConflict('operation transaction cannot nest')
            conn.execute('BEGIN IMMEDIATE')
            try:
                yield conn
                conn.execute('COMMIT')
            except BaseException:
                if conn.in_transaction:
                    conn.execute('ROLLBACK')
                raise

    def start(
        self,
        capability_id: str,
        request: Mapping[str, Any],
        *,
        idempotency_key: str,
        replay_policy: str,
        owner: str = 'daemon',
        connection: sqlite3.Connection | None = None,
    ) -> tuple[OperationRecord, bool]:
        if not isinstance(idempotency_key, str) or len(idempotency_key) < 16:
            raise OperationConflict('idempotency key must be bounded opaque text')
        if replay_policy not in REPLAY_POLICIES:
            raise OperationConflict('unsupported replay policy')
        if owner not in STATE_OWNERS['pending']:
            raise OperationConflict('unsupported operation owner')
        digest = _canonical_digest(capability_id, request)

        def apply(conn: sqlite3.Connection) -> tuple[OperationRecord, bool]:
            row = conn.execute(
                'SELECT * FROM operation_journal WHERE capability_id=? AND idempotency_key=?',
                (capability_id, idempotency_key),
            ).fetchone()
            if row is not None:
                record = self._record(row)
                if record.request_digest != digest or record.replay_policy != replay_policy:
                    raise OperationConflict('idempotency key is bound to a different request')
                if replay_policy == 'never':
                    raise OperationConflict('operation replay is forbidden')
                return record, True
            operation_id = 'op_' + secrets.token_urlsafe(24)
            timestamp = _now()
            conn.execute(
                """INSERT INTO operation_journal(
                       operation_id,capability_id,request_digest,idempotency_key,
                       replay_policy,state,stage,owner,version,created_at,updated_at
                   ) VALUES(?,?,?,?,?,'pending','created',?,1,?,?)""",
                (
                    operation_id, capability_id, digest, idempotency_key,
                    replay_policy, owner, timestamp, timestamp,
                ),
            )
            return self.get(operation_id, connection=conn), False

        if connection is not None:
            return apply(connection)
        with self.transaction() as conn:
            return apply(conn)

    def get(self, operation_id: str, *, connection: sqlite3.Connection | None = None) -> OperationRecord:
        def read(conn: sqlite3.Connection) -> OperationRecord:
            row = conn.execute(
                'SELECT * FROM operation_journal WHERE operation_id=?', (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationNotFound('operation does not exist')
            return self._record(row)

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def transition(
        self,
        operation_id: str,
        *,
        expected_states: set[str] | frozenset[str],
        state: str,
        stage: str,
        owner: str,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        continuation_token_hash: str | None = None,
        external_ref: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> OperationRecord:
        if state not in OPERATION_STATES or owner not in OWNERS:
            raise OperationConflict('invalid operation transition target')
        if owner not in STATE_OWNERS[state]:
            raise OperationConflict('operation owner is invalid for the target state')
        if not expected_states or not expected_states <= OPERATION_STATES:
            raise OperationConflict('invalid operation expected state set')
        if state == 'completed' and result is None:
            raise OperationConflict('completed operation requires a result')
        if state == 'failed' and error is None:
            raise OperationConflict('failed operation requires an error')
        if state in TERMINAL_STATES and owner != 'none':
            raise OperationConflict('terminal operation owner must be none')
        if state != 'awaiting_agent' and continuation_token_hash is not None:
            raise OperationConflict('only awaiting_agent can carry a continuation token')

        def apply(conn: sqlite3.Connection) -> OperationRecord:
            current = self.get(operation_id, connection=conn)
            if current.terminal:
                raise OperationConflict('terminal operation cannot transition')
            if current.state not in expected_states:
                raise OperationConflict('operation state changed before transition')
            timestamp = _now()
            cursor = conn.execute(
                """UPDATE operation_journal
                      SET state=?,stage=?,owner=?,result_json=?,error_json=?,
                          continuation_token_hash=?,external_ref=?,version=version+1,updated_at=?
                    WHERE operation_id=? AND version=?""",
                (
                    state, stage, owner, _json(result), _json(error),
                    continuation_token_hash, external_ref, timestamp,
                    operation_id, current.version,
                ),
            )
            if cursor.rowcount != 1:
                raise OperationConflict('operation version conflict')
            return self.get(operation_id, connection=conn)

        if connection is not None:
            return apply(connection)
        with self.transaction() as conn:
            return apply(conn)

    def consume_continuation(
        self,
        operation_id: str,
        *,
        token: str,
    ) -> OperationRecord:
        with self.transaction() as conn:
            current = self.get(operation_id, connection=conn)
            expected_hash = self.continuation_hash(current, token)
            row = conn.execute(
                'SELECT continuation_token_hash FROM operation_journal WHERE operation_id=?',
                (operation_id,),
            ).fetchone()
            stored = str(row['continuation_token_hash'] or '') if row is not None else ''
            if current.state != 'awaiting_agent' or not stored or not hmac.compare_digest(stored, expected_hash):
                raise OperationConflict('continuation token is invalid or already consumed')
            return self.transition(
                operation_id,
                expected_states={'awaiting_agent'},
                state='running',
                stage='continuation_received',
                owner='daemon',
                connection=conn,
            )

    @staticmethod
    def continuation_hash(record: OperationRecord, token: str) -> str:
        if not isinstance(token, str) or len(token) < 20:
            raise OperationConflict('continuation token is invalid')
        return hashlib.sha256(
            f'{record.operation_id}\x00{record.capability_id}\x00{token}'.encode('utf-8')
        ).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> OperationRecord:
        return OperationRecord(
            operation_id=str(row['operation_id']),
            capability_id=str(row['capability_id']),
            request_digest=str(row['request_digest']),
            idempotency_key=str(row['idempotency_key']),
            replay_policy=str(row['replay_policy']),
            state=str(row['state']),
            stage=str(row['stage']),
            owner=str(row['owner']),
            result=_decode(row['result_json']),
            error=_decode(row['error_json']),
            has_continuation=bool(row['continuation_token_hash']),
            external_ref=str(row['external_ref']) if row['external_ref'] is not None else None,
            version=int(row['version']),
            created_at=str(row['created_at']),
            updated_at=str(row['updated_at']),
        )


__all__ = [
    'OPERATION_STATES', 'OWNERS', 'OperationConflict', 'OperationJournal',
    'OperationNotFound', 'OperationRecord', 'REPLAY_POLICIES', 'STATE_OWNERS',
    'TERMINAL_STATES',
]
