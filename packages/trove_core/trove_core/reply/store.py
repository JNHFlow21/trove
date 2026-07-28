from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any, Iterator, Mapping

from .models import (
    EvidenceMessage,
    ReplyDraft,
    ReplyEvent,
    ReviewRecord,
    RoundRecord,
    SendIntent,
    SendOperationRecord,
)


SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reply_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO reply_metadata(key, value) VALUES('schema_version', '1');

CREATE TABLE IF NOT EXISTS reply_rounds (
    round_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    target_ref TEXT NOT NULL,
    first_seen_at REAL NOT NULL,
    last_extended_at REAL NOT NULL,
    preparation_at REAL NOT NULL,
    earliest_ready_at REAL NOT NULL,
    ready_at REAL NOT NULL,
    deadline_at REAL NOT NULL,
    quiet_target_ms INTEGER NOT NULL,
    source_position INTEGER NOT NULL,
    latest_fingerprint TEXT NOT NULL,
    inbound_message_count INTEGER NOT NULL,
    latest_kind TEXT NOT NULL,
    revision INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    not_before REAL NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    blocked INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    UNIQUE(account_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS reply_drafts (
    draft_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES reply_rounds(round_id),
    round_revision INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    target_ref TEXT NOT NULL,
    source_position INTEGER NOT NULL,
    context_digest TEXT NOT NULL,
    text TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reply_drafts_round_state
    ON reply_drafts(round_id, state, created_at);

CREATE TABLE IF NOT EXISTS reply_reviews (
    review_id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL UNIQUE REFERENCES reply_drafts(draft_id),
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    decided_at REAL
);
CREATE INDEX IF NOT EXISTS idx_reply_reviews_state_created
    ON reply_reviews(state, created_at);

CREATE TABLE IF NOT EXISTS reply_send_operations (
    operation_id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL REFERENCES reply_drafts(draft_id),
    idempotency_key TEXT NOT NULL UNIQUE,
    intent_digest TEXT NOT NULL,
    intent_json TEXT,
    state TEXT NOT NULL,
    stage TEXT NOT NULL,
    external_ref TEXT,
    result_json TEXT,
    error_code TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reply_send_state_updated
    ON reply_send_operations(state, updated_at);

CREATE TABLE IF NOT EXISTS reply_events (
    round_id TEXT NOT NULL REFERENCES reply_rounds(round_id),
    round_revision INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(round_id, round_revision)
);

CREATE TABLE IF NOT EXISTS reply_cursors (
    target_ref TEXT PRIMARY KEY,
    source_position INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reply_activity (
    activity_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    target_ref TEXT,
    conversation_id TEXT,
    display_label TEXT,
    text TEXT,
    state TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reply_activity_created
    ON reply_activity(created_at DESC);
"""


class ReplyStoreConflict(RuntimeError):
    code = 'reply_state_conflict'


class ReplyStoreNotFound(LookupError):
    code = 'reply_state_not_found'


class ReplyStore:
    """Owner-only durable state for one Vault ReplyService."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._initialize()

    @classmethod
    def for_vault(cls, vault_root: str | Path) -> 'ReplyStore':
        return cls(Path(vault_root) / 'jobs' / 'reply' / 'state.sqlite3')

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        with self.connection() as conn:
            conn.executescript(_SCHEMA)
            send_columns = {
                str(row['name'])
                for row in conn.execute(
                    'PRAGMA table_info(reply_send_operations)'
                ).fetchall()
            }
            if 'intent_json' not in send_columns:
                conn.execute(
                    'ALTER TABLE reply_send_operations ADD COLUMN intent_json TEXT'
                )
            if 'retry_count' not in send_columns:
                conn.execute(
                    """ALTER TABLE reply_send_operations
                       ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"""
                )
            version = conn.execute(
                "SELECT value FROM reply_metadata WHERE key='schema_version'"
            ).fetchone()
            if version is None or int(version['value']) != SCHEMA_VERSION:
                raise ReplyStoreConflict('unsupported reply store schema version')
        self._repair_sidecar_permissions()

    def _repair_sidecar_permissions(self) -> None:
        for suffix in ('', '-wal', '-shm'):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=5000')
        try:
            yield conn
            if conn.in_transaction:
                conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()
            self._repair_sidecar_permissions()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                yield conn
                conn.execute('COMMIT')
            except BaseException:
                if conn.in_transaction:
                    conn.execute('ROLLBACK')
                raise

    def schema_version(self) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM reply_metadata WHERE key='schema_version'"
            ).fetchone()
        if row is None:
            raise ReplyStoreConflict('reply store schema metadata is missing')
        return int(row['value'])

    def get_round(
        self,
        round_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> RoundRecord:
        def read(conn: sqlite3.Connection) -> RoundRecord:
            row = conn.execute(
                'SELECT * FROM reply_rounds WHERE round_id=?', (round_id,),
            ).fetchone()
            if row is None:
                raise ReplyStoreNotFound('reply round does not exist')
            return self._round(row)

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def find_round(
        self,
        account_id: str,
        conversation_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> RoundRecord | None:
        def read(conn: sqlite3.Connection) -> RoundRecord | None:
            row = conn.execute(
                """SELECT * FROM reply_rounds
                    WHERE account_id=? AND conversation_id=?""",
                (account_id, conversation_id),
            ).fetchone()
            return self._round(row) if row is not None else None

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def save_round(
        self,
        record: RoundRecord,
        *,
        now: float | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> RoundRecord:
        timestamp = record.last_extended_at if now is None else float(now)

        def apply(conn: sqlite3.Connection) -> RoundRecord:
            conn.execute(
                """INSERT INTO reply_rounds(
                       round_id,account_id,conversation_id,target_ref,
                       first_seen_at,last_extended_at,preparation_at,
                       earliest_ready_at,ready_at,deadline_at,quiet_target_ms,
                       source_position,latest_fingerprint,inbound_message_count,
                       latest_kind,revision,attempts,not_before,last_error,
                       blocked,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(round_id) DO UPDATE SET
                       target_ref=excluded.target_ref,
                       last_extended_at=excluded.last_extended_at,
                       preparation_at=excluded.preparation_at,
                       earliest_ready_at=excluded.earliest_ready_at,
                       ready_at=excluded.ready_at,
                       deadline_at=excluded.deadline_at,
                       quiet_target_ms=excluded.quiet_target_ms,
                       source_position=excluded.source_position,
                       latest_fingerprint=excluded.latest_fingerprint,
                       inbound_message_count=excluded.inbound_message_count,
                       latest_kind=excluded.latest_kind,
                       revision=excluded.revision,
                       attempts=excluded.attempts,
                       not_before=excluded.not_before,
                       last_error=excluded.last_error,
                       blocked=excluded.blocked,
                       updated_at=excluded.updated_at
                   WHERE reply_rounds.account_id=excluded.account_id
                     AND reply_rounds.conversation_id=excluded.conversation_id""",
                (
                    record.round_id,
                    record.account_id,
                    record.conversation_id,
                    record.target_ref,
                    record.first_seen_at,
                    record.last_extended_at,
                    record.preparation_at,
                    record.earliest_ready_at,
                    record.ready_at,
                    record.deadline_at,
                    record.quiet_target_ms,
                    record.source_position,
                    record.latest_fingerprint,
                    record.inbound_message_count,
                    record.latest_kind,
                    record.revision,
                    record.attempts,
                    record.not_before,
                    record.last_error,
                    int(record.blocked),
                    timestamp,
                ),
            )
            return self.get_round(record.round_id, connection=conn)

        if connection is not None:
            return apply(connection)
        with self.transaction() as conn:
            return apply(conn)

    def list_rounds(self) -> tuple[RoundRecord, ...]:
        with self.connection() as conn:
            rows = conn.execute('SELECT * FROM reply_rounds').fetchall()
        return tuple(self._round(row) for row in rows)

    def save_event(
        self,
        event: ReplyEvent,
        *,
        round_id: str,
        round_revision: int,
        now: float,
        connection: sqlite3.Connection | None = None,
    ) -> ReplyEvent:
        payload = {
            'event_id': event.event_id,
            'account_id': event.account_id,
            'conversation_id': event.conversation_id,
            'target_ref': event.target_ref,
            'source_position': event.source_position,
            'latest_fingerprint': event.latest_fingerprint,
            'observed_at': event.observed_at,
            'messages': [
                {
                    'citation': item.citation,
                    'source_position': item.source_position,
                    'observed_at': item.observed_at,
                    'kind': item.kind,
                    **({'text': item.text} if item.text is not None else {}),
                }
                for item in event.messages
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        )

        def apply(conn: sqlite3.Connection) -> ReplyEvent:
            round_record = self.get_round(round_id, connection=conn)
            if (
                round_record.revision != round_revision
                or round_record.account_id != event.account_id
                or round_record.conversation_id != event.conversation_id
                or round_record.target_ref != event.target_ref
                or round_record.source_position != event.source_position
                or round_record.latest_fingerprint != event.latest_fingerprint
            ):
                raise ReplyStoreConflict('event does not match the current reply round')
            conn.execute(
                """INSERT INTO reply_events(
                       round_id,round_revision,event_json,created_at
                   ) VALUES(?,?,?,?)
                   ON CONFLICT(round_id,round_revision) DO UPDATE SET
                       event_json=excluded.event_json,
                       created_at=excluded.created_at""",
                (round_id, round_revision, encoded, now),
            )
            return self.get_event(
                round_id, round_revision=round_revision, connection=conn,
            )

        if connection is not None:
            return apply(connection)
        with self.transaction() as conn:
            return apply(conn)

    def get_event(
        self,
        round_id: str,
        *,
        round_revision: int | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> ReplyEvent:
        def read(conn: sqlite3.Connection) -> ReplyEvent:
            if round_revision is None:
                row = conn.execute(
                    """SELECT event_json FROM reply_events
                        WHERE round_id=?
                        ORDER BY round_revision DESC LIMIT 1""",
                    (round_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT event_json FROM reply_events
                        WHERE round_id=? AND round_revision=?""",
                    (round_id, round_revision),
                ).fetchone()
            if row is None:
                raise ReplyStoreNotFound('reply event does not exist')
            try:
                payload = json.loads(str(row['event_json']))
                messages = tuple(
                    EvidenceMessage(
                        citation=str(item['citation']),
                        source_position=int(item['source_position']),
                        observed_at=float(item['observed_at']),
                        kind=str(item['kind']),
                        text=(
                            str(item['text'])
                            if item.get('text') is not None
                            else None
                        ),
                    )
                    for item in payload['messages']
                )
                return ReplyEvent(
                    event_id=str(payload['event_id']),
                    account_id=str(payload['account_id']),
                    conversation_id=str(payload['conversation_id']),
                    target_ref=str(payload['target_ref']),
                    source_position=int(payload['source_position']),
                    latest_fingerprint=str(payload['latest_fingerprint']),
                    messages=messages,
                    observed_at=float(payload['observed_at']),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ReplyStoreConflict('reply event payload is corrupt') from exc

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def cursor_map(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                'SELECT target_ref,source_position FROM reply_cursors'
            ).fetchall()
        return {
            str(row['target_ref']): int(row['source_position'])
            for row in rows
        }

    def advance_cursor(
        self,
        target_ref: str,
        source_position: int,
        *,
        now: float,
    ) -> int:
        if (
            not isinstance(target_ref, str)
            or not target_ref
            or type(source_position) is not int
            or source_position < 0
        ):
            raise ReplyStoreConflict('reply cursor is invalid')
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO reply_cursors(
                       target_ref,source_position,updated_at
                   ) VALUES(?,?,?)
                   ON CONFLICT(target_ref) DO UPDATE SET
                       source_position=MAX(
                           reply_cursors.source_position,
                           excluded.source_position
                       ),
                       updated_at=excluded.updated_at""",
                (target_ref, source_position, now),
            )
            row = conn.execute(
                'SELECT source_position FROM reply_cursors WHERE target_ref=?',
                (target_ref,),
            ).fetchone()
        assert row is not None
        return int(row['source_position'])

    def invalidate_superseded(
        self,
        round_id: str,
        *,
        current_revision: int,
        now: float,
        connection: sqlite3.Connection,
    ) -> None:
        draft_ids = [
            str(row['draft_id'])
            for row in connection.execute(
                """SELECT draft_id FROM reply_drafts
                    WHERE round_id=? AND round_revision<?
                      AND state IN ('generated','pending_review','approved')""",
                (round_id, current_revision),
            ).fetchall()
        ]
        if not draft_ids:
            return
        placeholders = ','.join('?' for _ in draft_ids)
        connection.execute(
            f"""UPDATE reply_drafts SET state='stale'
                  WHERE draft_id IN ({placeholders})""",
            draft_ids,
        )
        connection.execute(
            f"""UPDATE reply_reviews SET state='stale',decided_at=?
                  WHERE draft_id IN ({placeholders})
                    AND state IN ('pending','approved')""",
            (now, *draft_ids),
        )
        connection.execute(
            f"""UPDATE reply_send_operations
                  SET state='cancelled',stage='source_advanced',updated_at=?
                  WHERE draft_id IN ({placeholders}) AND state='prepared'""",
            (now, *draft_ids),
        )

    def save_draft(self, draft: ReplyDraft) -> ReplyDraft:
        with self.transaction() as conn:
            round_record = self.get_round(draft.round_id, connection=conn)
            if (
                round_record.revision != draft.round_revision
                or round_record.source_position != draft.source_position
                or round_record.account_id != draft.account_id
                or round_record.conversation_id != draft.conversation_id
                or round_record.target_ref != draft.target_ref
            ):
                raise ReplyStoreConflict('draft does not match the current reply round')
            try:
                conn.execute(
                    """INSERT INTO reply_drafts(
                           draft_id,round_id,round_revision,account_id,
                           conversation_id,target_ref,source_position,
                           context_digest,text,backend,model,state,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        draft.draft_id,
                        draft.round_id,
                        draft.round_revision,
                        draft.account_id,
                        draft.conversation_id,
                        draft.target_ref,
                        draft.source_position,
                        draft.context_digest,
                        draft.text,
                        draft.backend,
                        draft.model,
                        draft.state,
                        draft.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReplyStoreConflict('draft already exists') from exc
            return self.get_draft(draft.draft_id, connection=conn)

    def get_draft(
        self,
        draft_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ReplyDraft:
        def read(conn: sqlite3.Connection) -> ReplyDraft:
            row = conn.execute(
                'SELECT * FROM reply_drafts WHERE draft_id=?', (draft_id,),
            ).fetchone()
            if row is None:
                raise ReplyStoreNotFound('reply draft does not exist')
            return self._draft(row)

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def current_draft(
        self,
        round_id: str,
        *,
        round_revision: int,
    ) -> ReplyDraft | None:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT * FROM reply_drafts
                    WHERE round_id=? AND round_revision=?
                      AND state IN (
                          'generated','pending_review','approved','rejected'
                      )
                    ORDER BY created_at DESC LIMIT 1""",
                (round_id, round_revision),
            ).fetchone()
        return self._draft(row) if row is not None else None

    def enqueue_review(self, draft_id: str, *, now: float) -> ReviewRecord:
        with self.transaction() as conn:
            draft = self.get_draft(draft_id, connection=conn)
            if draft.state != 'generated':
                raise ReplyStoreConflict('only a current generated draft can enter review')
            review_id = 'review_' + secrets.token_urlsafe(18)
            conn.execute(
                """INSERT INTO reply_reviews(
                       review_id,draft_id,state,created_at
                   ) VALUES(?,?,'pending',?)""",
                (review_id, draft_id, now),
            )
            conn.execute(
                "UPDATE reply_drafts SET state='pending_review' WHERE draft_id=?",
                (draft_id,),
            )
            return self.get_review(review_id, connection=conn)

    def get_review(
        self,
        review_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ReviewRecord:
        def read(conn: sqlite3.Connection) -> ReviewRecord:
            row = conn.execute(
                'SELECT * FROM reply_reviews WHERE review_id=?', (review_id,),
            ).fetchone()
            if row is None:
                raise ReplyStoreNotFound('reply review does not exist')
            return self._review(row)

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def decide_review(
        self,
        review_id: str,
        *,
        decision: str,
        now: float,
    ) -> ReviewRecord:
        if decision not in {'approved', 'rejected'}:
            raise ReplyStoreConflict('review decision must be approved or rejected')
        with self.transaction() as conn:
            current = self.get_review(review_id, connection=conn)
            draft = self.get_draft(current.draft_id, connection=conn)
            round_record = self.get_round(draft.round_id, connection=conn)
            if (
                current.state != 'pending'
                or draft.state != 'pending_review'
                or draft.round_revision != round_record.revision
                or draft.source_position != round_record.source_position
            ):
                raise ReplyStoreConflict('review is no longer current')
            conn.execute(
                'UPDATE reply_reviews SET state=?,decided_at=? WHERE review_id=?',
                (decision, now, review_id),
            )
            conn.execute(
                'UPDATE reply_drafts SET state=? WHERE draft_id=?',
                (decision, draft.draft_id),
            )
            return self.get_review(review_id, connection=conn)

    def list_reviews(
        self,
        *,
        state: str | None = 'pending',
        limit: int = 100,
    ) -> tuple[tuple[ReviewRecord, ReplyDraft], ...]:
        bounded = max(1, min(int(limit), 1_000))
        where = '' if state is None else 'WHERE r.state=?'
        parameters: tuple[Any, ...] = () if state is None else (state,)
        with self.connection() as conn:
            rows = conn.execute(
                f"""SELECT r.review_id
                      FROM reply_reviews r
                      {where}
                     ORDER BY r.created_at DESC
                     LIMIT ?""",
                (*parameters, bounded),
            ).fetchall()
            result = tuple(
                (
                    self.get_review(str(row['review_id']), connection=conn),
                    self.get_draft(
                        self.get_review(
                            str(row['review_id']), connection=conn,
                        ).draft_id,
                        connection=conn,
                    ),
                )
                for row in rows
            )
        return result

    def prepare_send(
        self,
        intent: SendIntent,
        *,
        now: float,
    ) -> tuple[SendOperationRecord, bool]:
        intent_digest = intent.digest()
        intent_json = json.dumps(
            {
                'operation_id': intent.operation_id,
                'idempotency_key': intent.idempotency_key,
                'draft_id': intent.draft_id,
                'account_id': intent.account_id,
                'conversation_id': intent.conversation_id,
                'target_ref': intent.target_ref,
                'expected_source_position': intent.expected_source_position,
                'draft_digest': intent.draft_digest,
                'text': intent.text,
                'grant_ref': intent.grant_ref,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
        with self.transaction() as conn:
            existing = conn.execute(
                """SELECT * FROM reply_send_operations
                    WHERE idempotency_key=?""",
                (intent.idempotency_key,),
            ).fetchone()
            if existing is not None:
                record = self._send(existing)
                if (
                    record.operation_id != intent.operation_id
                    or record.intent_digest != intent_digest
                ):
                    raise ReplyStoreConflict(
                        'idempotency key is bound to a different send intent'
                    )
                return record, True

            draft = self.get_draft(intent.draft_id, connection=conn)
            round_record = self.get_round(draft.round_id, connection=conn)
            if (
                draft.state not in {'generated', 'approved'}
                or draft.account_id != intent.account_id
                or draft.conversation_id != intent.conversation_id
                or draft.target_ref != intent.target_ref
                or draft.source_position != intent.expected_source_position
                or draft.digest != intent.draft_digest
                or draft.text != intent.text
                or round_record.revision != draft.round_revision
                or round_record.source_position != draft.source_position
            ):
                raise ReplyStoreConflict('send intent does not match the current draft')
            if intent.grant_ref.startswith('review_'):
                review = self.get_review(intent.grant_ref, connection=conn)
                if review.draft_id != draft.draft_id or review.state != 'approved':
                    raise ReplyStoreConflict('send review grant is not current')
            try:
                conn.execute(
                    """INSERT INTO reply_send_operations(
                           operation_id,draft_id,idempotency_key,intent_digest,
                           intent_json,state,stage,created_at,updated_at
                       ) VALUES(?,?,?,?,?,'prepared','intent_bound',?,?)""",
                    (
                        intent.operation_id,
                        intent.draft_id,
                        intent.idempotency_key,
                        intent_digest,
                        intent_json,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ReplyStoreConflict('send operation already exists') from exc
            return self.get_send(intent.operation_id, connection=conn), False

    def get_send(
        self,
        operation_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> SendOperationRecord:
        def read(conn: sqlite3.Connection) -> SendOperationRecord:
            row = conn.execute(
                """SELECT * FROM reply_send_operations
                    WHERE operation_id=?""",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise ReplyStoreNotFound('send operation does not exist')
            return self._send(row)

        if connection is not None:
            return read(connection)
        with self.connection() as conn:
            return read(conn)

    def get_send_intent(self, operation_id: str) -> SendIntent:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT intent_json,intent_digest
                     FROM reply_send_operations
                    WHERE operation_id=?""",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise ReplyStoreNotFound('send operation does not exist')
        if row['intent_json'] is None:
            raise ReplyStoreConflict('send intent predates durable recovery')
        try:
            payload = json.loads(str(row['intent_json']))
            intent = SendIntent(
                operation_id=str(payload['operation_id']),
                idempotency_key=str(payload['idempotency_key']),
                draft_id=str(payload['draft_id']),
                account_id=str(payload['account_id']),
                conversation_id=str(payload['conversation_id']),
                target_ref=str(payload['target_ref']),
                expected_source_position=int(
                    payload['expected_source_position']
                ),
                draft_digest=str(payload['draft_digest']),
                text=str(payload['text']),
                grant_ref=str(payload['grant_ref']),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplyStoreConflict('durable send intent is corrupt') from exc
        if intent.digest() != str(row['intent_digest']):
            raise ReplyStoreConflict('durable send intent digest does not match')
        return intent

    def list_sends(
        self,
        *,
        states: tuple[str, ...] | None = None,
        limit: int = 100,
    ) -> tuple[SendOperationRecord, ...]:
        bounded = max(1, min(int(limit), 1_000))
        if states:
            placeholders = ','.join('?' for _ in states)
            where = f'WHERE state IN ({placeholders})'
            parameters: tuple[Any, ...] = tuple(states)
        else:
            where = ''
            parameters = ()
        with self.connection() as conn:
            rows = conn.execute(
                f"""SELECT * FROM reply_send_operations
                      {where}
                     ORDER BY updated_at,operation_id
                     LIMIT ?""",
                (*parameters, bounded),
            ).fetchall()
        return tuple(self._send(row) for row in rows)

    def send_policy_status(
        self,
        target_ref: str,
        *,
        now: float,
        cooldown_seconds: float,
        daily_send_limit: int,
    ) -> dict[str, Any]:
        """Return the daemon-side delivery limit for one exact target."""

        with self.connection() as conn:
            daily = int(conn.execute(
                """SELECT COUNT(*)
                     FROM reply_send_operations
                    WHERE state='completed' AND updated_at>=?""",
                (float(now) - 86_400.0,),
            ).fetchone()[0])
            row = conn.execute(
                """SELECT MAX(o.updated_at)
                     FROM reply_send_operations o
                     JOIN reply_drafts d ON d.draft_id=o.draft_id
                    WHERE o.state='completed' AND d.target_ref=?""",
                (target_ref,),
            ).fetchone()
        last_target = (
            float(row[0])
            if row is not None and row[0] is not None
            else None
        )
        if daily >= int(daily_send_limit):
            return {
                'allowed': False,
                'reason': 'daily_send_limit',
                'daily_completed': daily,
                'retry_at': float(now) + 86_400.0,
            }
        retry_at = (
            last_target + max(0.0, float(cooldown_seconds))
            if last_target is not None
            else 0.0
        )
        if retry_at > float(now):
            return {
                'allowed': False,
                'reason': 'target_cooldown',
                'daily_completed': daily,
                'retry_at': retry_at,
            }
        return {
            'allowed': True,
            'reason': 'allowed',
            'daily_completed': daily,
            'retry_at': 0.0,
        }

    def add_activity(
        self,
        event_type: str,
        *,
        state: str,
        now: float,
        target_ref: str | None = None,
        conversation_id: str | None = None,
        display_label: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        for name, value, maximum in (
            ('event_type', event_type, 128),
            ('state', state, 128),
            ('target_ref', target_ref, 256),
            ('conversation_id', conversation_id, 512),
            ('display_label', display_label, 512),
            ('text', text, 8_000),
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value
                or len(value.encode('utf-8')) > maximum
            ):
                raise ReplyStoreConflict(f'reply activity {name} is invalid')
        activity_id = 'activity_' + secrets.token_urlsafe(18)
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO reply_activity(
                       activity_id,event_type,target_ref,conversation_id,
                       display_label,text,state,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    activity_id, event_type, target_ref, conversation_id,
                    display_label, text, state, now,
                ),
            )
        return {
            'activity_id': activity_id,
            'event_type': event_type,
            'target_ref': target_ref,
            'conversation_id': conversation_id,
            'display_label': display_label,
            'text': text,
            'state': state,
            'created_at': now,
        }

    def list_activity(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        bounded = max(1, min(int(limit), 1_000))
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM reply_activity
                    ORDER BY created_at DESC,activity_id DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
        return tuple({
            'activity_id': str(row['activity_id']),
            'event_type': str(row['event_type']),
            'target_ref': (
                str(row['target_ref']) if row['target_ref'] is not None else None
            ),
            'conversation_id': (
                str(row['conversation_id'])
                if row['conversation_id'] is not None
                else None
            ),
            'display_label': (
                str(row['display_label'])
                if row['display_label'] is not None
                else None
            ),
            'text': str(row['text']) if row['text'] is not None else None,
            'state': str(row['state']),
            'created_at': float(row['created_at']),
        } for row in rows)

    def mark_dispatched(
        self,
        operation_id: str,
        *,
        external_ref: str,
        now: float,
    ) -> SendOperationRecord:
        with self.transaction() as conn:
            current = self.get_send(operation_id, connection=conn)
            if current.state != 'prepared':
                raise ReplyStoreConflict('only a prepared send may dispatch')
            conn.execute(
                """UPDATE reply_send_operations
                      SET state='dispatched',stage='provider_dispatched',
                          external_ref=?,updated_at=?
                    WHERE operation_id=?""",
                (external_ref, now, operation_id),
            )
            return self.get_send(operation_id, connection=conn)

    def mark_reconciling(
        self,
        operation_id: str,
        *,
        now: float,
    ) -> SendOperationRecord:
        with self.transaction() as conn:
            current = self.get_send(operation_id, connection=conn)
            if current.state not in {'dispatched', 'reconciling'}:
                raise ReplyStoreConflict('send is not eligible for reconciliation')
            conn.execute(
                """UPDATE reply_send_operations
                      SET state='reconciling',stage='awaiting_source_proof',
                          updated_at=?
                    WHERE operation_id=?""",
                (now, operation_id),
            )
            return self.get_send(operation_id, connection=conn)

    def finish_send(
        self,
        operation_id: str,
        *,
        state: str,
        stage: str,
        now: float,
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
    ) -> SendOperationRecord:
        if state not in {'completed', 'failed', 'unknown'}:
            raise ReplyStoreConflict('invalid terminal send state')
        if state == 'completed' and result is None:
            raise ReplyStoreConflict('completed send requires source proof')
        result_json = (
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            if result is not None
            else None
        )
        with self.transaction() as conn:
            current = self.get_send(operation_id, connection=conn)
            if current.state not in {'dispatched', 'reconciling'}:
                raise ReplyStoreConflict('send is not in a reconcilable state')
            conn.execute(
                """UPDATE reply_send_operations
                      SET state=?,stage=?,result_json=?,error_code=?,updated_at=?
                    WHERE operation_id=?""",
                (state, stage, result_json, error_code, now, operation_id),
            )
            return self.get_send(operation_id, connection=conn)

    def reopen_failed_send(
        self,
        operation_id: str,
        *,
        review_id: str,
        expected_stage: str,
        now: float,
        maximum_retries: int = 3,
    ) -> SendOperationRecord:
        """Reopen one provider-reconciled pre-send failure.

        The service owns the allowed-stage policy and first proves that no
        remote acknowledgement exists. The approved review, draft, and
        conversation round watermark are revalidated in the same transaction
        that reopens the operation. This closes the race where new source
        evidence arrives after Provider preflight. Requiring the exact
        terminal stage prevents a stale operator retry from reopening a
        changed send. Completed and unknown sends are never eligible.
        """

        bounded = max(1, min(int(maximum_retries), 10))
        with self.transaction() as conn:
            current = self.get_send(operation_id, connection=conn)
            review = self.get_review(review_id, connection=conn)
            draft = self.get_draft(current.draft_id, connection=conn)
            round_record = self.get_round(draft.round_id, connection=conn)
            if (
                current.state != 'failed'
                or current.stage != expected_stage
                or current.retry_count >= bounded
                or review.draft_id != draft.draft_id
                or review.state != 'approved'
                or draft.state != 'approved'
                or draft.round_revision != round_record.revision
                or draft.source_position != round_record.source_position
            ):
                raise ReplyStoreConflict(
                    'send operation is not an eligible failed retry'
                )
            conn.execute(
                """UPDATE reply_send_operations
                      SET state='prepared',
                          stage='retry_authorized',
                          external_ref=NULL,
                          result_json=NULL,
                          error_code=NULL,
                          retry_count=retry_count+1,
                          updated_at=?
                    WHERE operation_id=?""",
                (now, operation_id),
            )
            return self.get_send(operation_id, connection=conn)

    @staticmethod
    def _round(row: sqlite3.Row) -> RoundRecord:
        return RoundRecord(
            round_id=str(row['round_id']),
            account_id=str(row['account_id']),
            conversation_id=str(row['conversation_id']),
            target_ref=str(row['target_ref']),
            first_seen_at=float(row['first_seen_at']),
            last_extended_at=float(row['last_extended_at']),
            preparation_at=float(row['preparation_at']),
            earliest_ready_at=float(row['earliest_ready_at']),
            ready_at=float(row['ready_at']),
            deadline_at=float(row['deadline_at']),
            quiet_target_ms=int(row['quiet_target_ms']),
            source_position=int(row['source_position']),
            latest_fingerprint=str(row['latest_fingerprint']),
            inbound_message_count=int(row['inbound_message_count']),
            latest_kind=str(row['latest_kind']),
            revision=int(row['revision']),
            attempts=int(row['attempts']),
            not_before=float(row['not_before']),
            last_error=str(row['last_error']),
            blocked=bool(row['blocked']),
        )

    @staticmethod
    def _draft(row: sqlite3.Row) -> ReplyDraft:
        return ReplyDraft(
            draft_id=str(row['draft_id']),
            round_id=str(row['round_id']),
            round_revision=int(row['round_revision']),
            account_id=str(row['account_id']),
            conversation_id=str(row['conversation_id']),
            target_ref=str(row['target_ref']),
            source_position=int(row['source_position']),
            context_digest=str(row['context_digest']),
            text=str(row['text']),
            backend=str(row['backend']),
            model=str(row['model']),
            state=str(row['state']),
            created_at=float(row['created_at']),
        )

    @staticmethod
    def _review(row: sqlite3.Row) -> ReviewRecord:
        return ReviewRecord(
            review_id=str(row['review_id']),
            draft_id=str(row['draft_id']),
            state=str(row['state']),
            created_at=float(row['created_at']),
            decided_at=(
                float(row['decided_at']) if row['decided_at'] is not None else None
            ),
        )

    @staticmethod
    def _send(row: sqlite3.Row) -> SendOperationRecord:
        result = json.loads(row['result_json']) if row['result_json'] is not None else None
        if result is not None and not isinstance(result, dict):
            raise ReplyStoreConflict('send result payload is corrupt')
        return SendOperationRecord(
            operation_id=str(row['operation_id']),
            draft_id=str(row['draft_id']),
            idempotency_key=str(row['idempotency_key']),
            intent_digest=str(row['intent_digest']),
            state=str(row['state']),
            stage=str(row['stage']),
            external_ref=(
                str(row['external_ref']) if row['external_ref'] is not None else None
            ),
            result=result,
            error_code=(
                str(row['error_code']) if row['error_code'] is not None else None
            ),
            retry_count=int(row['retry_count']),
            created_at=float(row['created_at']),
            updated_at=float(row['updated_at']),
        )


__all__ = ['ReplyStore', 'ReplyStoreConflict', 'ReplyStoreNotFound', 'SCHEMA_VERSION']
