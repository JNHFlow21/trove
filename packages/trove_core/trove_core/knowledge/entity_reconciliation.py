from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.store.sqlite_store import SQLiteStore


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}'


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _resequence_snapshot_history(conn: Any, entity_id: str) -> None:
    rows = list(conn.execute(
        """SELECT profile_id FROM profile_snapshots WHERE entity_id=?
             ORDER BY created_at,version,profile_id""",
        (entity_id,),
    ))
    for version, row in enumerate(rows, start=1):
        conn.execute(
            'UPDATE profile_snapshots SET version=? WHERE profile_id=?',
            (-version, row['profile_id']),
        )
    for version, row in enumerate(rows, start=1):
        conn.execute(
            'UPDATE profile_snapshots SET version=? WHERE profile_id=?',
            (version, row['profile_id']),
        )


def _append_snapshot_history(conn: Any, canonical_id: str, duplicate_id: str) -> None:
    """Move one history without ever violating the live unique version index."""

    rows = list(conn.execute(
        """SELECT profile_id FROM profile_snapshots WHERE entity_id=?
             ORDER BY created_at,version,profile_id""",
        (duplicate_id,),
    ))
    if not rows:
        return
    maximum = int(conn.execute(
        'SELECT COALESCE(MAX(version),0) FROM profile_snapshots WHERE entity_id=?',
        (canonical_id,),
    ).fetchone()[0])
    for offset, row in enumerate(rows, start=1):
        conn.execute(
            'UPDATE profile_snapshots SET version=? WHERE profile_id=?',
            (-offset, row['profile_id']),
        )
    for offset, row in enumerate(rows, start=1):
        conn.execute(
            'UPDATE profile_snapshots SET version=? WHERE profile_id=?',
            (maximum + offset, row['profile_id']),
        )
    conn.execute(
        'UPDATE profile_snapshots SET entity_id=? WHERE entity_id=?',
        (canonical_id, duplicate_id),
    )


def reconciliation_plan(store: SQLiteStore, customer: str) -> dict[str, Any]:
    resolution = resolve_customer(store, customer)
    resolved = resolution.get('resolved') or {}
    canonical_id = str(resolved.get('entity_id') or '')
    if not canonical_id or canonical_id.startswith('unresolved:'):
        return {
            'ok': False,
            'status': 'canonical_entity_unresolved',
            'canonical_entity_id': None,
            'duplicate_entity_ids': [],
            'raw_content_included': False,
        }
    conversation_ids = set(str(value) for value in (resolved.get('conversation_ids') or []) if value)
    candidates: list[str] = []
    with store.connect() as conn:
        for candidate in resolution.get('candidates') or []:
            for entity_id in candidate.get('entity_ids') or [candidate.get('entity_id')]:
                entity_id = str(entity_id or '')
                if not entity_id or entity_id == canonical_id or entity_id.startswith('unresolved:'):
                    continue
                row = conn.execute('SELECT display_name,identifiers_json,status FROM entities WHERE entity_id=?', (entity_id,)).fetchone()
                if row is None or row['status'] == 'merged':
                    continue
                try:
                    identifiers = json.loads(row['identifiers_json'] or '{}')
                except json.JSONDecodeError:
                    identifiers = {}
                materialized = identifiers.get('resolution_source') == 'observe_materialized_unresolved_customer'
                source_ref = str(identifiers.get('source_entity_ref') or '')
                primary = str(identifiers.get('primary_user_id') or '')
                bound = primary in conversation_ids or source_ref in {f'unresolved:{value}' for value in conversation_ids}
                if materialized and bound:
                    candidates.append(entity_id)
        duplicate_ids = sorted(set(candidates))
        counts = {
            'observations': 0,
            'relationships': 0,
            'profile_snapshots': 0,
            'profile_automation_subscriptions': 0,
        }
        if duplicate_ids:
            placeholders = ','.join('?' for _ in duplicate_ids)
            counts['observations'] = int(conn.execute(
                f'SELECT COUNT(*) FROM observations WHERE entity_id IN ({placeholders})', duplicate_ids,
            ).fetchone()[0])
            counts['relationships'] = int(conn.execute(
                f'''SELECT COUNT(*) FROM relationships
                    WHERE subject_entity_id IN ({placeholders}) OR object_entity_id IN ({placeholders})''',
                (*duplicate_ids, *duplicate_ids),
            ).fetchone()[0])
            counts['profile_snapshots'] = int(conn.execute(
                f'SELECT COUNT(*) FROM profile_snapshots WHERE entity_id IN ({placeholders})', duplicate_ids,
            ).fetchone()[0])
            counts['profile_automation_subscriptions'] = int(conn.execute(
                f'SELECT COUNT(*) FROM profile_automation_subscriptions WHERE entity_id IN ({placeholders})', duplicate_ids,
            ).fetchone()[0])
    return {
        'ok': True,
        'status': 'ready' if duplicate_ids else 'no_duplicates',
        'canonical_entity_id': canonical_id,
        'duplicate_entity_ids': duplicate_ids,
        'counts': counts,
        'raw_content_included': False,
    }


def reconcile_customer_entities(store: SQLiteStore, customer: str, *, apply: bool = False) -> dict[str, Any]:
    plan = reconciliation_plan(store, customer)
    if not plan.get('ok') or not plan.get('duplicate_entity_ids'):
        return plan | {'applied': False}
    if not apply:
        return plan | {'applied': False, 'dry_run': True}
    canonical_id = str(plan['canonical_entity_id'])
    duplicate_ids = [str(value) for value in plan['duplicate_entity_ids']]
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            canonical = conn.execute(
                'SELECT display_name,identifiers_json FROM entities WHERE entity_id=?',
                (canonical_id,),
            ).fetchone()
            identifiers = json.loads(canonical['identifiers_json'] or '{}') if canonical else {}
            canonical_selector = str(
                identifiers.get('primary_user_id')
                or identifiers.get('wechat_id')
                or (canonical['display_name'] if canonical else customer)
            )
            aliases = list(identifiers.get('aliases') or []) if isinstance(identifiers.get('aliases'), list) else []
            conversation_ids = list(identifiers.get('conversation_ids') or []) if isinstance(identifiers.get('conversation_ids'), list) else []
            sender_ids = list(identifiers.get('sender_ids') or []) if isinstance(identifiers.get('sender_ids'), list) else []
            for duplicate_id in duplicate_ids:
                duplicate = conn.execute('SELECT display_name,identifiers_json FROM entities WHERE entity_id=?', (duplicate_id,)).fetchone()
                if duplicate is None:
                    continue
                duplicate_identifiers = json.loads(duplicate['identifiers_json'] or '{}')
                aliases.extend([duplicate['display_name'], *(duplicate_identifiers.get('aliases') or [])])
                primary = str(duplicate_identifiers.get('primary_user_id') or '')
                if primary:
                    conversation_ids.append(primary)
                conn.execute('UPDATE observations SET entity_id=? WHERE entity_id=?', (canonical_id, duplicate_id))
                conn.execute('UPDATE relationships SET subject_entity_id=? WHERE subject_entity_id=?', (canonical_id, duplicate_id))
                conn.execute('UPDATE relationships SET object_entity_id=? WHERE object_entity_id=?', (canonical_id, duplicate_id))
                _append_snapshot_history(conn, canonical_id, duplicate_id)
                duplicate_subscription = conn.execute(
                    'SELECT * FROM profile_automation_subscriptions WHERE entity_id=?',
                    (duplicate_id,),
                ).fetchone()
                if duplicate_subscription is not None:
                    canonical_subscription = conn.execute(
                        'SELECT * FROM profile_automation_subscriptions WHERE entity_id=?',
                        (canonical_id,),
                    ).fetchone()
                    if canonical_subscription is None:
                        conn.execute(
                            """INSERT INTO profile_automation_subscriptions(
                                   entity_id,selector,enabled,debounce_seconds,consent_scope,
                                   last_profile_id,last_refresh_at,last_error_code,created_at,updated_at)
                               VALUES(?,?,?,?,?,?,?,?,?,datetime('now'))""",
                            (
                                canonical_id,
                                canonical_selector,
                                duplicate_subscription['enabled'],
                                duplicate_subscription['debounce_seconds'],
                                duplicate_subscription['consent_scope'],
                                duplicate_subscription['last_profile_id'],
                                duplicate_subscription['last_refresh_at'],
                                duplicate_subscription['last_error_code'],
                                duplicate_subscription['created_at'],
                            ),
                        )
                    else:
                        conn.execute(
                            """UPDATE profile_automation_subscriptions
                                  SET enabled=MAX(enabled,?),debounce_seconds=MIN(debounce_seconds,?),
                                      selector=?,last_profile_id=COALESCE(last_profile_id,?),
                                      last_refresh_at=CASE
                                          WHEN last_refresh_at IS NULL THEN ?
                                          WHEN ? IS NULL THEN last_refresh_at
                                          ELSE MAX(last_refresh_at,?)
                                      END,
                                      last_error_code=NULL,updated_at=datetime('now')
                                WHERE entity_id=?""",
                            (
                                duplicate_subscription['enabled'],
                                duplicate_subscription['debounce_seconds'],
                                canonical_selector,
                                duplicate_subscription['last_profile_id'],
                                duplicate_subscription['last_refresh_at'],
                                duplicate_subscription['last_refresh_at'],
                                duplicate_subscription['last_refresh_at'],
                                canonical_id,
                            ),
                        )
                    duplicate_queue = conn.execute(
                        'SELECT * FROM profile_refresh_queue WHERE entity_id=?',
                        (duplicate_id,),
                    ).fetchone()
                    if duplicate_queue is not None:
                        conn.execute(
                            """INSERT INTO profile_refresh_queue(
                                   entity_id,generation,state,reason,available_at,claimed_at,
                                   attempt_count,last_error_code,created_at,updated_at)
                               VALUES(?,?,'pending','entity_reconciled',?,NULL,?,NULL,?,datetime('now'))
                               ON CONFLICT(entity_id) DO UPDATE SET
                                   generation=MAX(profile_refresh_queue.generation,excluded.generation)+1,
                                   state='pending',reason='entity_reconciled',
                                   available_at=MIN(profile_refresh_queue.available_at,excluded.available_at),
                                   claimed_at=NULL,
                                   attempt_count=MAX(profile_refresh_queue.attempt_count,excluded.attempt_count),
                                   last_error_code=NULL,updated_at=datetime('now')""",
                            (
                                canonical_id,
                                duplicate_queue['generation'],
                                duplicate_queue['available_at'],
                                duplicate_queue['attempt_count'],
                                duplicate_queue['created_at'],
                            ),
                        )
                    conn.execute('DELETE FROM profile_refresh_queue WHERE entity_id=?', (duplicate_id,))
                    conn.execute('DELETE FROM profile_automation_subscriptions WHERE entity_id=?', (duplicate_id,))
                conn.execute('DELETE FROM entity_identifiers WHERE entity_id=?', (duplicate_id,))
                duplicate_identifiers['merged_into'] = canonical_id
                conn.execute(
                    "UPDATE entities SET status='merged',confidence=MIN(confidence,1.0),identifiers_json=?,updated_at=datetime('now') WHERE entity_id=?",
                    (_json(duplicate_identifiers), duplicate_id),
                )
                relationship_id = _stable('rel', f'{duplicate_id}:same_as:{canonical_id}')
                conn.execute(
                    """INSERT OR REPLACE INTO relationships(
                           relationship_id,subject_entity_id,predicate,object_entity_id,object_ref,citation,confidence,status,metadata_json,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,datetime('now'))""",
                    (relationship_id, duplicate_id, 'same_as', canonical_id, None, None, 1.0, 'merged', _json({'reconciled': True})),
                )
            _resequence_snapshot_history(conn, canonical_id)
            identifiers['aliases'] = list(dict.fromkeys(str(value) for value in aliases if value))[:100]
            identifiers['conversation_ids'] = list(dict.fromkeys(conversation_ids))[:100]
            identifiers['sender_ids'] = list(dict.fromkeys(sender_ids))[:100]
            conn.execute(
                "UPDATE entities SET identifiers_json=?,confidence=MIN(confidence,1.0),updated_at=datetime('now') WHERE entity_id=?",
                (_json(identifiers), canonical_id),
            )
            conn.execute(
                """UPDATE profile_automation_subscriptions
                      SET selector=?,last_profile_id=(
                          SELECT profile_id FROM profile_snapshots
                           WHERE entity_id=? ORDER BY version DESC,created_at DESC LIMIT 1
                      ),updated_at=datetime('now')
                    WHERE entity_id=?""",
                (canonical_selector, canonical_id, canonical_id),
            )
            conn.execute(
                """DELETE FROM profile_refresh_queue
                     WHERE entity_id=? AND EXISTS(
                         SELECT 1 FROM profile_automation_subscriptions
                          WHERE entity_id=? AND enabled=0
                     )""",
                (canonical_id, canonical_id),
            )
            subscription = conn.execute(
                'SELECT enabled FROM profile_automation_subscriptions WHERE entity_id=?',
                (canonical_id,),
            ).fetchone()
            if subscription is not None and bool(subscription['enabled']):
                conn.execute(
                    """INSERT INTO profile_refresh_queue(
                           entity_id,generation,state,reason,available_at,claimed_at,
                           attempt_count,last_error_code,created_at,updated_at)
                       VALUES(?,1,'pending','entity_reconciled',datetime('now'),NULL,0,NULL,
                              datetime('now'),datetime('now'))
                       ON CONFLICT(entity_id) DO UPDATE SET
                           generation=profile_refresh_queue.generation+1,
                           state='pending',reason='entity_reconciled',available_at=datetime('now'),
                           claimed_at=NULL,attempt_count=0,last_error_code=NULL,
                           updated_at=datetime('now')""",
                    (canonical_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return plan | {'status': 'reconciled', 'applied': True}
