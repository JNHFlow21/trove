from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from trove_core.knowledge.entity_resolution import (
    ALIAS_KEYS,
    USER_ID_KEYS,
    normalize_identifier,
    resolve_customer,
)
from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError
from trove_core.knowledge.profile_snapshots import (
    AUTOMATIC_PROFILE_SNAPSHOT_SCHEMA,
    automatic_profile_snapshot_payload,
    persist_profile_snapshot_conn,
)
from trove_core.store.migrations import SchemaMigrationRequired
from trove_core.store.schema import VECTOR_SOURCE_REVISION_KEY
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.mutations import coordinated_vault_mutation


DEFAULT_DEBOUNCE_SECONDS = 180
MAX_DEBOUNCE_SECONDS = 3600
CLAIM_TIMEOUT_SECONDS = 300
MAX_REFRESH_ATTEMPTS = 5
AUTOMATIC_SNAPSHOT_HISTORY_LIMIT = 50
AUTOMATION_CONSENT_SCOPE = 'explicit-profile-auto-maintenance-v1'
PROFILE_RECONCILE_SOURCE_REVISION_KEY = 'profile_automation_reconcile_source_revision'
_PROFILE_IDENTITY_KEYS = frozenset({
    *USER_ID_KEYS,
    *ALIAS_KEYS,
    'conversation_id',
    'conversation_ids',
    'sender_id',
    'sender_ids',
})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _validate_debounce(value: int) -> int:
    if type(value) is not int or not 0 <= value <= MAX_DEBOUNCE_SECONDS:
        raise ProfileEnrichmentError(
            f'debounce_seconds must be from 0 to {MAX_DEBOUNCE_SECONDS}',
            code='invalid_profile_automation_debounce',
        )
    return value


def _identity_values(value: Any) -> set[str]:
    if isinstance(value, dict):
        values: set[str] = set()
        for key, item in value.items():
            if str(key) in _PROFILE_IDENTITY_KEYS:
                values.update(_identity_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        return {item for nested in value for item in _identity_values(nested)}
    text = _normalize_identity(value)
    return {text} if text else set()


def _normalize_identity(value: Any) -> str:
    return normalize_identifier(value)


class ProfileAutomationService:
    """Explicit per-person subscription and an idempotent local refresh queue."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self.store.initialize()

    def has_enabled_subscriptions(self) -> bool:
        with self.store.connect() as conn:
            return conn.execute(
                'SELECT 1 FROM profile_automation_subscriptions WHERE enabled=1 LIMIT 1'
            ).fetchone() is not None

    @staticmethod
    def _resolved(store: SQLiteStore, customer: str) -> dict[str, Any]:
        resolution = resolve_customer(store, customer)
        resolved = resolution.get('resolved')
        entity_id = str((resolved or {}).get('entity_id') or '').strip()
        if not entity_id or entity_id.startswith('unresolved:'):
            code = 'customer_ambiguous' if resolution.get('ambiguous') else 'canonical_entity_required'
            raise ProfileEnrichmentError(
                'profile automation requires one canonical entity', code=code,
            )
        return resolved

    def enable(
        self,
        customer: str,
        *,
        debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        debounce_seconds = _validate_debounce(debounce_seconds)
        resolved = self._resolved(self.store, customer)
        entity_id = str(resolved['entity_id'])
        selector = str(resolved.get('primary_user_id') or resolved.get('display_name') or customer)
        current = now or _now()
        timestamp = _iso(current)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                """INSERT INTO profile_automation_subscriptions(
                       entity_id,selector,enabled,debounce_seconds,consent_scope,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(entity_id) DO UPDATE SET
                       selector=excluded.selector,enabled=1,debounce_seconds=excluded.debounce_seconds,
                       consent_scope=excluded.consent_scope,last_error_code=NULL,updated_at=excluded.updated_at""",
                (
                    entity_id, selector, 1, debounce_seconds, AUTOMATION_CONSENT_SCOPE,
                    timestamp, timestamp,
                ),
            )
            self._enqueue_entity_conn(
                conn,
                entity_id,
                reason='subscription_enabled',
                current=current,
                debounce_seconds=0,
            )
            conn.commit()
        return {
            'ok': True,
            'type': 'profile_automation_subscription',
            'entity_id': entity_id,
            'enabled': True,
            'debounce_seconds': debounce_seconds,
            'queue_state': 'pending',
            'consent_scope': AUTOMATION_CONSENT_SCOPE,
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def disable(self, customer: str, *, now: datetime | None = None) -> dict[str, Any]:
        resolved = self._resolved(self.store, customer)
        entity_id = str(resolved['entity_id'])
        timestamp = _iso(now or _now())
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            cursor = conn.execute(
                "UPDATE profile_automation_subscriptions SET enabled=0,updated_at=? WHERE entity_id=?",
                (timestamp, entity_id),
            )
            conn.execute('DELETE FROM profile_refresh_queue WHERE entity_id=?', (entity_id,))
            conn.commit()
        return {
            'ok': True,
            'type': 'profile_automation_subscription',
            'entity_id': entity_id,
            'enabled': False,
            'existed': cursor.rowcount > 0,
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    @staticmethod
    def _enqueue_entity_conn(
        conn: Any,
        entity_id: str,
        *,
        reason: str,
        current: datetime,
        debounce_seconds: int,
    ) -> None:
        timestamp = _iso(current)
        available_at = _iso(current + timedelta(seconds=debounce_seconds))
        conn.execute(
            """INSERT INTO profile_refresh_queue(
                   entity_id,generation,state,reason,available_at,created_at,updated_at)
               VALUES(?,1,'pending',?,?,?,?)
               ON CONFLICT(entity_id) DO UPDATE SET
                   generation=profile_refresh_queue.generation+1,
                   state='pending',reason=excluded.reason,
                   available_at=CASE
                       WHEN profile_refresh_queue.state='pending'
                        AND profile_refresh_queue.available_at<excluded.available_at
                       THEN profile_refresh_queue.available_at
                       ELSE excluded.available_at
                   END,
                   claimed_at=NULL,attempt_count=0,last_error_code=NULL,updated_at=excluded.updated_at""",
            (entity_id, str(reason)[:100], available_at, timestamp, timestamp),
        )

    def enqueue_all(
        self,
        *,
        reason: str,
        debounce_override_seconds: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if debounce_override_seconds is not None:
            _validate_debounce(debounce_override_seconds)
        current = now or _now()
        with self.store.connect() as conn:
            rows = list(conn.execute(
                'SELECT entity_id,debounce_seconds FROM profile_automation_subscriptions WHERE enabled=1'
            ))
            conn.execute('BEGIN IMMEDIATE')
            for row in rows:
                self._enqueue_entity_conn(
                    conn,
                    str(row['entity_id']),
                    reason=reason,
                    current=current,
                    debounce_seconds=(
                        int(debounce_override_seconds)
                        if debounce_override_seconds is not None
                        else int(row['debounce_seconds'])
                    ),
                )
            conn.commit()
        return {
            'ok': True,
            'type': 'profile_refresh_queue',
            'queued': len(rows),
            'reason': str(reason)[:100],
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def enqueue_all_if_source_changed(
        self,
        *,
        reason: str,
        debounce_override_seconds: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Queue one reconciliation generation only when source revision advanced."""

        if debounce_override_seconds is not None:
            _validate_debounce(debounce_override_seconds)
        current = now or _now()
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            source = conn.execute(
                'SELECT value FROM schema_meta WHERE key=?',
                (VECTOR_SOURCE_REVISION_KEY,),
            ).fetchone()
            current_revision = str(source['value'] if source is not None else '0')
            marker = conn.execute(
                'SELECT value FROM schema_meta WHERE key=?',
                (PROFILE_RECONCILE_SOURCE_REVISION_KEY,),
            ).fetchone()
            if marker is not None and str(marker['value']) == current_revision:
                conn.rollback()
                return {
                    'ok': True,
                    'type': 'profile_refresh_queue',
                    'queued': 0,
                    'reason': 'source_revision_unchanged',
                    'source_changed': False,
                    'raw_content_included': False,
                    'raw_paths_included': False,
                }
            rows = list(conn.execute(
                'SELECT entity_id,debounce_seconds FROM profile_automation_subscriptions WHERE enabled=1'
            ))
            for row in rows:
                self._enqueue_entity_conn(
                    conn,
                    str(row['entity_id']),
                    reason=reason,
                    current=current,
                    debounce_seconds=(
                        int(debounce_override_seconds)
                        if debounce_override_seconds is not None
                        else int(row['debounce_seconds'])
                    ),
                )
            conn.execute(
                """INSERT INTO schema_meta(key,value) VALUES(?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (PROFILE_RECONCILE_SOURCE_REVISION_KEY, current_revision),
            )
            conn.commit()
        return {
            'ok': True,
            'type': 'profile_refresh_queue',
            'queued': len(rows),
            'reason': str(reason)[:100],
            'source_changed': True,
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def enqueue_impacted(
        self,
        identity_values: set[str] | list[str] | tuple[str, ...],
        *,
        reason: str,
        debounce_override_seconds: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Queue only subscriptions whose canonical identity scope changed."""

        if debounce_override_seconds is not None:
            _validate_debounce(debounce_override_seconds)
        changed = {
            normalized for value in identity_values
            if (normalized := _normalize_identity(value))
        }
        current = now or _now()
        with self.store.connect() as conn:
            indexed_entity_ids: set[str] = set()
            changed_values = sorted(changed)
            for start in range(0, len(changed_values), 500):
                batch = changed_values[start:start + 500]
                marks = ','.join('?' for _ in batch)
                indexed_entity_ids.update(str(row['entity_id']) for row in conn.execute(
                    f"""SELECT DISTINCT s.entity_id
                           FROM profile_automation_subscriptions s
                           JOIN entity_identifiers ei ON ei.entity_id=s.entity_id
                          WHERE s.enabled=1 AND ei.normalized_value IN ({marks})""",
                    batch,
                ))
            rows = list(conn.execute(
                """SELECT s.entity_id,s.selector,s.debounce_seconds,e.display_name,e.identifiers_json
                       FROM profile_automation_subscriptions s
                       JOIN entities e ON e.entity_id=s.entity_id
                      WHERE s.enabled=1"""
            ))
            impacted = []
            for row in rows:
                if str(row['entity_id']) in indexed_entity_ids:
                    impacted.append(row)
                    continue
                try:
                    identifiers = json.loads(row['identifiers_json'] or '{}')
                except json.JSONDecodeError:
                    identifiers = {}
                scope = _identity_values(identifiers) | {
                    _normalize_identity(row['entity_id']),
                    _normalize_identity(row['selector']),
                    _normalize_identity(row['display_name']),
                }
                if changed.intersection(scope):
                    impacted.append(row)
            conn.execute('BEGIN IMMEDIATE')
            for row in impacted:
                self._enqueue_entity_conn(
                    conn,
                    str(row['entity_id']),
                    reason=reason,
                    current=current,
                    debounce_seconds=(
                        int(debounce_override_seconds)
                        if debounce_override_seconds is not None
                        else int(row['debounce_seconds'])
                    ),
                )
            conn.commit()
        return {
            'ok': True,
            'type': 'profile_refresh_queue',
            'queued': len(impacted),
            'subscriptions_checked': len(rows),
            'targeted': True,
            'reason': str(reason)[:100],
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def identity_values_for_changes(
        self,
        *,
        citations: set[str] | list[str] | tuple[str, ...] = (),
        asset_ids: set[str] | list[str] | tuple[str, ...] = (),
    ) -> set[str]:
        """Resolve bounded changed evidence handles to profile identity keys."""

        source_citations = {str(value).split('#', 1)[0] for value in citations if str(value)}
        assets = {str(value) for value in asset_ids if str(value)}
        identities: set[str] = set()
        with self.store.connect() as conn:
            ordered_assets = sorted(assets)
            for start in range(0, len(ordered_assets), 500):
                batch = ordered_assets[start:start + 500]
                marks = ','.join('?' for _ in batch)
                source_citations.update(str(row[0]).split('#', 1)[0] for row in conn.execute(
                    f"""SELECT source_citation FROM media_asset_links
                          WHERE asset_id IN ({marks}) AND accepted=1
                        UNION
                        SELECT citation FROM media_assets
                          WHERE asset_id IN ({marks}) AND citation IS NOT NULL""",
                    (*batch, *batch),
                ))
            ordered = sorted(source_citations)
            for start in range(0, len(ordered), 300):
                batch = ordered[start:start + 300]
                marks = ','.join('?' for _ in batch)
                for row in conn.execute(
                    f"""SELECT conversation_id,sender_id FROM messages
                          WHERE citation IN ({marks})""",
                    batch,
                ):
                    identities.update((str(row['conversation_id']), str(row['sender_id'])))
                identities.update(str(row[0]) for row in conn.execute(
                    f"SELECT entity_id FROM observations WHERE citation IN ({marks})",
                    batch,
                ))
                for row in conn.execute(
                    f"SELECT author_id FROM moment_items WHERE citation IN ({marks})",
                    batch,
                ):
                    identities.add(str(row['author_id']))
                for row in conn.execute(
                    f"""SELECT mi.actor_id,m.author_id
                           FROM moment_interactions mi
                      LEFT JOIN moment_items m ON m.moment_id=mi.moment_id
                          WHERE mi.citation IN ({marks})""",
                    batch,
                ):
                    identities.update((str(row['actor_id']), str(row['author_id'] or '')))
        return {value for value in identities if _normalize_identity(value)}

    def enqueue_customer(
        self,
        customer: str,
        *,
        reason: str = 'manual_refresh',
        now: datetime | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolved(self.store, customer)
        entity_id = str(resolved['entity_id'])
        current = now or _now()
        with self.store.connect() as conn:
            row = conn.execute(
                'SELECT enabled FROM profile_automation_subscriptions WHERE entity_id=?',
                (entity_id,),
            ).fetchone()
            if row is None or not bool(row['enabled']):
                raise ProfileEnrichmentError(
                    'profile automation is not enabled for this entity',
                    code='profile_automation_not_enabled',
                )
            conn.execute('BEGIN IMMEDIATE')
            self._enqueue_entity_conn(
                conn, entity_id, reason=reason, current=current, debounce_seconds=0,
            )
            conn.commit()
        return {
            'ok': True,
            'entity_id': entity_id,
            'queue_state': 'pending',
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def status(self, customer: str | None = None, *, limit: int = 100) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ProfileEnrichmentError('limit must be from 1 to 100', code='invalid_snapshot_limit')
        entity_id = None
        if customer:
            entity_id = str(self._resolved(self.store, customer)['entity_id'])
        where = 'WHERE s.entity_id=?' if entity_id else ''
        params: tuple[Any, ...] = (entity_id,) if entity_id else ()
        with self.store.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                f"""SELECT s.entity_id,e.display_name,s.enabled,s.debounce_seconds,s.consent_scope,
                           s.last_profile_id,s.last_refresh_at,s.last_error_code,s.updated_at,
                           q.state AS queue_state,q.reason AS queue_reason,q.available_at,q.attempt_count
                      FROM profile_automation_subscriptions s
                      JOIN entities e ON e.entity_id=s.entity_id
                 LEFT JOIN profile_refresh_queue q ON q.entity_id=s.entity_id
                    {where}
                  ORDER BY s.enabled DESC,s.updated_at DESC LIMIT ?""",
                (*params, limit),
            )]
        return {
            'ok': True,
            'type': 'profile_automation_status',
            'count': len(rows),
            'items': rows,
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def has_due(
        self,
        *,
        entity_id: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        current = now or _now()
        timestamp = _iso(current)
        expired = _iso(current - timedelta(seconds=CLAIM_TIMEOUT_SECONDS))
        target_clause = ' AND q.entity_id=?' if entity_id else ''
        params: tuple[Any, ...] = (timestamp, expired, entity_id) if entity_id else (timestamp, expired)
        with self.store.connect() as conn:
            return conn.execute(
                """SELECT 1 FROM profile_refresh_queue q
                     JOIN profile_automation_subscriptions s ON s.entity_id=q.entity_id AND s.enabled=1
                    WHERE ((q.state='pending' AND q.available_at<=?)
                       OR (q.state='processing' AND q.claimed_at<=?))
                """ + target_clause + ' LIMIT 1',
                params,
            ).fetchone() is not None

    def due_count(
        self,
        *,
        entity_id: str | None = None,
        now: datetime | None = None,
    ) -> int:
        current = now or _now()
        timestamp = _iso(current)
        expired = _iso(current - timedelta(seconds=CLAIM_TIMEOUT_SECONDS))
        target_clause = ' AND q.entity_id=?' if entity_id else ''
        params: tuple[Any, ...] = (
            (timestamp, expired, entity_id) if entity_id else (timestamp, expired)
        )
        with self.store.connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM profile_refresh_queue q
                     JOIN profile_automation_subscriptions s
                       ON s.entity_id=q.entity_id AND s.enabled=1
                    WHERE ((q.state='pending' AND q.available_at<=?)
                       OR (q.state='processing' AND q.claimed_at<=?))"""
                + target_clause,
                params,
            ).fetchone()
        return int(row[0] or 0)

    def claim_due(
        self,
        *,
        entity_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = now or _now()
        timestamp = _iso(current)
        expired = _iso(current - timedelta(seconds=CLAIM_TIMEOUT_SECONDS))
        reset_target = ' AND entity_id=?' if entity_id else ''
        select_target = ' AND q.entity_id=?' if entity_id else ''
        reset_params: tuple[Any, ...] = (
            (timestamp, timestamp, expired, entity_id)
            if entity_id else (timestamp, timestamp, expired)
        )
        select_params: tuple[Any, ...] = (timestamp, entity_id) if entity_id else (timestamp,)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                """UPDATE profile_refresh_queue
                      SET generation=generation+1,state='pending',claimed_at=NULL,
                          available_at=?,updated_at=?
                    WHERE state='processing' AND claimed_at<=?""" + reset_target,
                reset_params,
            )
            row = conn.execute(
                """SELECT q.*,s.selector FROM profile_refresh_queue q
                     JOIN profile_automation_subscriptions s ON s.entity_id=q.entity_id AND s.enabled=1
                    WHERE q.state='pending' AND q.available_at<=?
                """ + select_target + ' ORDER BY q.available_at,q.updated_at,q.entity_id LIMIT 1',
                select_params,
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            cursor = conn.execute(
                """UPDATE profile_refresh_queue
                      SET state='processing',claimed_at=?,attempt_count=attempt_count+1,updated_at=?
                    WHERE entity_id=? AND generation=? AND state='pending'""",
                (timestamp, timestamp, row['entity_id'], row['generation']),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return None
            claimed = dict(conn.execute(
                """SELECT q.*,s.selector FROM profile_refresh_queue q
                     JOIN profile_automation_subscriptions s ON s.entity_id=q.entity_id
                    WHERE q.entity_id=?""",
                (row['entity_id'],),
            ).fetchone())
            conn.commit()
        return claimed

    def prepare(self, claim: dict[str, Any]) -> dict[str, Any]:
        return automatic_profile_snapshot_payload(
            self.store,
            entity_id=str(claim['entity_id']),
            selector=str(claim['selector']),
        )

    def publish(
        self,
        claim: dict[str, Any],
        prepared: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        timestamp = _iso(current)
        entity_id = str(claim['entity_id'])
        generation = int(claim['generation'])
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            queue = conn.execute(
                'SELECT generation,state FROM profile_refresh_queue WHERE entity_id=?',
                (entity_id,),
            ).fetchone()
            if queue is None or int(queue['generation']) != generation or str(queue['state']) != 'processing':
                conn.rollback()
                return {
                    'ok': False,
                    'status': 'retry_required',
                    'reason': 'profile_refresh_generation_changed',
                    'entity_id': entity_id,
                    'raw_content_included': False,
                }
            snapshot, created = persist_profile_snapshot_conn(
                conn,
                entity_id=entity_id,
                content=prepared['content'],
                content_hash=prepared['content_hash'],
                source_revision=prepared['source_revision'],
                run_id=None,
                schema_version=prepared['schema_version'],
                completeness_state=prepared['completeness_state'],
                created_at=timestamp,
            )
            profile_id = str(snapshot['profile_id'])
            version = int(snapshot['version'])
            pruned_snapshots = 0
            if created:
                cursor = conn.execute(
                    """DELETE FROM profile_snapshots
                         WHERE profile_id IN (
                             SELECT profile_id FROM profile_snapshots
                              WHERE entity_id=? AND schema_version=?
                              ORDER BY version DESC,created_at DESC
                              LIMIT -1 OFFSET ?
                         )""",
                    (
                        entity_id,
                        AUTOMATIC_PROFILE_SNAPSHOT_SCHEMA,
                        AUTOMATIC_SNAPSHOT_HISTORY_LIMIT,
                    ),
                )
                pruned_snapshots = max(0, cursor.rowcount)
            conn.execute(
                """UPDATE profile_automation_subscriptions
                      SET last_profile_id=?,last_refresh_at=?,last_error_code=NULL,updated_at=?
                    WHERE entity_id=?""",
                (profile_id, timestamp, timestamp, entity_id),
            )
            conn.execute(
                'DELETE FROM profile_refresh_queue WHERE entity_id=? AND generation=?',
                (entity_id, generation),
            )
            conn.commit()
        return {
            'ok': True,
            'status': 'created' if created else 'unchanged',
            'created': created,
            'cache_hit': not created,
            'entity_id': entity_id,
            'profile_id': profile_id,
            'version': version,
            'automatic_history_limit': AUTOMATIC_SNAPSHOT_HISTORY_LIMIT,
            'pruned_snapshots': pruned_snapshots,
            'completeness_state': prepared['completeness_state'],
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def fail(
        self,
        claim: dict[str, Any],
        error_code: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        timestamp = _iso(current)
        attempts = max(1, int(claim.get('attempt_count') or 1))
        available = _iso(current + timedelta(seconds=min(300, attempts * 30)))
        bounded_error = str(error_code or 'profile_refresh_failed')[:100]
        terminal = attempts >= MAX_REFRESH_ATTEMPTS
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            cursor = conn.execute(
                """UPDATE profile_refresh_queue
                      SET state=?,claimed_at=NULL,available_at=?,last_error_code=?,updated_at=?
                    WHERE entity_id=? AND generation=? AND state='processing'""",
                (
                    'failed' if terminal else 'pending',
                    available,
                    bounded_error,
                    timestamp,
                    claim['entity_id'],
                    claim['generation'],
                ),
            )
            applied = cursor.rowcount == 1
            if applied:
                conn.execute(
                    """UPDATE profile_automation_subscriptions
                          SET last_error_code=?,updated_at=? WHERE entity_id=?""",
                    (bounded_error, timestamp, claim['entity_id']),
                )
            conn.commit()
        return {'applied': applied, 'terminal': terminal and applied}


def process_profile_refresh_queue(
    config: VaultConfig | str | Path,
    *,
    limit: int = 10,
    entity_id: str | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ProfileEnrichmentError('limit must be from 1 to 100', code='invalid_snapshot_limit')
    cfg = config if isinstance(config, VaultConfig) else VaultConfig.resolve(str(config), env={})
    if not cfg.paths.sqlite_path.is_file():
        return {
            'ok': True,
            'status': 'no_index',
            'processed': 0,
            'created_snapshots': 0,
            'cache_hits': 0,
            'retry_required': 0,
            'failures': 0,
            'terminal_failures': 0,
            'remaining_due': 0,
            'drained': True,
            'errors': [],
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    # Normal worker polls stay strictly read-only. A historical schema is
    # upgraded only under the same coordinated writer used by publication.
    read_store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    try:
        service = ProfileAutomationService(read_store)
    except SchemaMigrationRequired:
        read_store.close()
        with coordinated_vault_mutation(
            cfg, operation='profile_automation', write_session=write_session,
        ):
            migration_store = SQLiteStore(cfg.paths.sqlite_path)
            try:
                migration_store.initialize()
            finally:
                migration_store.close()
        read_store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
        try:
            service = ProfileAutomationService(read_store)
        except Exception:
            read_store.close()
            raise
    except Exception:
        read_store.close()
        raise

    write_store = SQLiteStore(cfg.paths.sqlite_path)
    write_service: ProfileAutomationService | None = None

    def write_call(method: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal write_service
        if write_service is None:
            write_service = ProfileAutomationService(write_store)
        return getattr(write_service, method)(*args, **kwargs)

    processed = created = cache_hits = retry_required = failures = terminal_failures = 0
    errors: list[str] = []
    try:
        for _ in range(limit):
            due = (
                service.has_due(entity_id=entity_id)
                if write_service is None
                else write_call('has_due', entity_id=entity_id)
            )
            if not due:
                break
            try:
                with coordinated_vault_mutation(
                    cfg, operation='profile_automation', write_session=write_session,
                ):
                    claim = write_call('claim_due', entity_id=entity_id)
            except VaultOperationLocked:
                return {
                    'ok': False,
                    'status': 'locked',
                    'processed': processed,
                    'created_snapshots': created,
                    'cache_hits': cache_hits,
                    'retry_required': retry_required,
                    'failures': failures,
                    'terminal_failures': terminal_failures,
                    'remaining_due': service.due_count(entity_id=entity_id),
                    'drained': False,
                    'errors': errors,
                    'raw_content_included': False,
                    'raw_paths_included': False,
                }
            if claim is None:
                break
            try:
                prepared = service.prepare(claim)
                with coordinated_vault_mutation(
                    cfg, operation='profile_automation', write_session=write_session,
                ):
                    result = write_call('publish', claim, prepared)
                processed += 1
                if result.get('status') == 'retry_required':
                    retry_required += 1
                elif result.get('created'):
                    created += 1
                else:
                    cache_hits += 1
            except Exception as exc:
                failures += 1
                errors.append(getattr(exc, 'code', exc.__class__.__name__))
                try:
                    with coordinated_vault_mutation(
                        cfg, operation='profile_automation', write_session=write_session,
                    ):
                        failed = write_call(
                            'fail', claim, getattr(exc, 'code', exc.__class__.__name__),
                        )
                    terminal_failures += int(bool(failed.get('terminal')))
                    retry_required += int(not bool(failed.get('applied')))
                except VaultOperationLocked:
                    errors.append('VaultOperationLocked')
                    break
                except Exception as fail_exc:
                    errors.append(getattr(fail_exc, 'code', fail_exc.__class__.__name__))
                    break
        remaining_due = write_call('due_count', entity_id=entity_id)
    finally:
        read_store.close()
        write_store.close()
    drained = remaining_due == 0
    status = 'partial' if failures else ('completed' if drained else 'backlog_remaining')
    return {
        'ok': failures == 0 and drained,
        'status': status,
        'processed': processed,
        'created_snapshots': created,
        'cache_hits': cache_hits,
        'retry_required': retry_required,
        'failures': failures,
        'terminal_failures': terminal_failures,
        'remaining_due': remaining_due,
        'drained': drained,
        'errors': errors[:20],
        'raw_content_included': False,
        'raw_paths_included': False,
    }
