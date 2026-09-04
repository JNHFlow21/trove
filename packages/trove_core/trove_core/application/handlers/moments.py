"""Read-only bounded moment (timeline) evidence queries.

Both capabilities resolve one exact person before reading: an exact stored id
wins, otherwise entity identifier aliases (including contact remarks) must
converge on one stored id, with the shared contact resolver as the bounded
fuzzy fallback for partial names.  Ambiguity never picks a candidate
silently.  Pagination uses the protocol opaque cursor store bound to
capability, filters, generation, Vault identity and TTL.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Mapping

from trove_core.bounds import (
    MOMENT_INTERACTIONS,
    MOMENT_TIMELINE,
    BoundedInputError,
    bounded_limit,
)
from trove_core.knowledge.entity_resolution import normalize_identifier, resolve_customer
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.generation import vault_generation_read
from trove_protocol.cursors import CursorError, CursorStore

from .base import HandlerOutcome


_TIMELINE_CAPABILITY = 'trove.moment_timeline'
_INTERACTIONS_CAPABILITY = 'trove.moment_interactions'
_TEXT_EXCERPT_CHARS = 240
_RESOLUTION_CANDIDATE_LIMIT = 10

_CURSOR_LOCK = threading.Lock()
_CURSOR_STORE = CursorStore()


def _reset_cursor_store_for_tests() -> None:
    global _CURSOR_STORE
    with _CURSOR_LOCK:
        _CURSOR_STORE = CursorStore()


def _issue_cursor(**kwargs: Any) -> str:
    with _CURSOR_LOCK:
        return _CURSOR_STORE.issue(**kwargs)


def _resolve_cursor(handle: str, **kwargs: Any):
    with _CURSOR_LOCK:
        return _CURSOR_STORE.resolve(handle, **kwargs)


def _owner(config: Any) -> Any | None:
    return config if hasattr(config, 'read_store') and hasattr(config, 'config') else None


def _vault_identity(cfg: Any) -> str:
    return hashlib.sha256(str(cfg.root).encode('utf-8')).hexdigest()


def _generation_digest(token: Any) -> str:
    return hashlib.sha256(repr(token.cache_key()).encode('utf-8')).hexdigest()


def _bounded(field: str, value: Any, spec: Any) -> int | HandlerOutcome:
    try:
        return bounded_limit(
            spec.default if value is None else value, field=field, spec=spec,
        )
    except BoundedInputError as exc:
        return HandlerOutcome.failure(exc.code, str(exc), details=exc.to_dict())


def _excerpt(text: Any) -> tuple[str, bool]:
    value = str(text or '')
    return value[:_TEXT_EXCERPT_CHARS], len(value) > _TEXT_EXCERPT_CHARS


def _json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or ''))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or ''))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _link_summary(link_json: Any) -> dict[str, Any] | None:
    link = _json_dict(link_json)
    if not link:
        return None
    summary: dict[str, Any] = {}
    for key in ('title', 'url', 'description'):
        text = str(link.get(key) or '').strip()
        if text:
            summary[key] = text[:120]
    return summary or {'keys': sorted(str(key) for key in link)[:8]}


def _placeholders(count: int) -> str:
    return ','.join('?' for _ in range(count))


def _entity_user_ids(conn: Any, target: str) -> dict[str, list[str]]:
    """Map candidate user ids to entity display names via the identifier index."""

    normalized = normalize_identifier(target)
    if not normalized:
        return {}
    entity_rows = conn.execute(
        'SELECT DISTINCT entity_id FROM entity_identifiers WHERE normalized_value=? LIMIT 25',
        (normalized,),
    ).fetchall()
    entity_ids = [str(row[0]) for row in entity_rows]
    if not entity_ids:
        return {}
    user_rows = conn.execute(
        f"""SELECT DISTINCT normalized_value FROM entity_identifiers
             WHERE identifier_type='user_id' AND entity_id IN ({_placeholders(len(entity_ids))})
             LIMIT 25""",
        entity_ids,
    ).fetchall()
    user_ids = [str(row[0]) for row in user_rows if str(row[0] or '')]
    if not user_ids:
        return {}
    name_rows = conn.execute(
        f"""SELECT ei.normalized_value, e.display_name
              FROM entity_identifiers ei
              JOIN entities e ON e.entity_id=ei.entity_id
             WHERE ei.identifier_type='user_id'
               AND ei.normalized_value IN ({_placeholders(len(user_ids))})
             ORDER BY e.display_name LIMIT 50""",
        user_ids,
    ).fetchall()
    names: dict[str, list[str]] = {}
    for user_id, display_name in name_rows:
        bucket = names.setdefault(str(user_id), [])
        name = str(display_name or '')
        if name and name not in bucket:
            bucket.append(name)
    return {user_id: names.get(user_id, []) for user_id in user_ids}


def _id_counts(
    conn: Any,
    *,
    table: str,
    id_column: str,
    ids: list[str],
    account_id: str | None,
) -> list[dict[str, Any]]:
    if not ids:
        return []
    clauses = [f'{id_column} IN ({_placeholders(len(ids))})', "status='active'"]
    params: list[Any] = list(ids)
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    rows = conn.execute(
        f"""SELECT {id_column}, account_id, COUNT(*) AS n
              FROM {table}
             WHERE {' AND '.join(clauses)}
             GROUP BY {id_column}, account_id
             ORDER BY {id_column}, account_id
             LIMIT 64""",
        params,
    ).fetchall()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = grouped.setdefault(str(row[0]), {'id': str(row[0]), 'count': 0, 'accounts': []})
        entry['count'] += int(row[2])
        entry['accounts'].append(str(row[1]))
    return sorted(grouped.values(), key=lambda item: (-int(item['count']), str(item['id'])))


def _candidate_user_ids(conn: Any, candidate: Mapping[str, Any]) -> list[str]:
    """Collect stored user-id forms of one resolver candidate, bounded."""

    ids: list[str] = []
    primary = str(candidate.get('primary_user_id') or '').strip()
    if primary:
        ids.append(primary)
    entity_ids = [
        str(value)
        for value in (candidate.get('entity_ids') or [candidate.get('entity_id')])
        if value
    ][:5]
    if entity_ids:
        rows = conn.execute(
            f"""SELECT DISTINCT normalized_value FROM entity_identifiers
                 WHERE identifier_type='user_id' AND entity_id IN ({_placeholders(len(entity_ids))})
                 LIMIT 10""",
            entity_ids,
        ).fetchall()
        ids.extend(str(row[0]) for row in rows if str(row[0] or ''))
    return list(dict.fromkeys(ids))


def _fuzzy_person_candidates(
    store: Any,
    conn: Any,
    target: str,
    *,
    table: str,
    id_column: str,
    account_id: str | None,
) -> list[dict[str, Any]]:
    """Fall back to the shared contact resolver for partial/composite names.

    ``resolve_customer`` owns remark/substring/multi-token matching; this only
    keeps candidates that have stored evidence in the requested table.
    """

    resolved = resolve_customer(store, target)
    raw_candidates = list(resolved.get('candidates') or [])[:_RESOLUTION_CANDIDATE_LIMIT]
    if not raw_candidates:
        return []
    all_ids: list[str] = []
    per_candidate: list[tuple[Mapping[str, Any], list[str]]] = []
    for candidate in raw_candidates:
        user_ids = _candidate_user_ids(conn, candidate)
        per_candidate.append((candidate, user_ids))
        all_ids.extend(user_ids)
    counts = {
        item['id']: item
        for item in _id_counts(
            conn, table=table, id_column=id_column, ids=all_ids, account_id=account_id,
        )
    }
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate, user_ids in per_candidate:
        for user_id in user_ids:
            if user_id in seen or user_id not in counts:
                continue
            seen.add(user_id)
            entry = dict(counts[user_id])
            names = [str(candidate.get('display_name') or '')]
            names.extend(str(value) for value in (candidate.get('aliases') or []))
            entry['names'] = [name for name in dict.fromkeys(names) if name][:5]
            entry['match'] = 'entity_fuzzy'
            entry['reasons'] = [str(reason) for reason in (candidate.get('match_reasons') or [])][:3]
            ordered.append(entry)
    return ordered[:_RESOLUTION_CANDIDATE_LIMIT]


def _resolve_person(
    conn: Any,
    target: str,
    *,
    table: str,
    id_column: str,
    account_id: str | None,
    name_column: str | None = None,
    store: Any | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Resolve one person to exactly one stored id or typed candidates."""

    exact_clauses = [f'{id_column}=?', "status='active'"]
    exact_params: list[Any] = [target]
    if account_id:
        exact_clauses.append('account_id=?')
        exact_params.append(account_id)
    exact = conn.execute(
        f'SELECT 1 FROM {table} WHERE {" AND ".join(exact_clauses)} LIMIT 1',
        exact_params,
    ).fetchone()
    if exact is not None:
        return target, [{'id': target, 'count': -1, 'accounts': [], 'names': [], 'match': 'exact_id'}]

    names_by_id = _entity_user_ids(conn, target)
    candidates = _id_counts(
        conn, table=table, id_column=id_column,
        ids=list(names_by_id), account_id=account_id,
    )
    for candidate in candidates:
        candidate['names'] = names_by_id.get(candidate['id'], [])[:5]
        candidate['match'] = 'entity_alias'

    if name_column is not None:
        name_clauses = [f'lower({name_column})=lower(?)', "status='active'", f"{id_column}<>''"]
        name_params: list[Any] = [target]
        if account_id:
            name_clauses.append('account_id=?')
            name_params.append(account_id)
        name_rows = conn.execute(
            f"""SELECT {id_column}, {name_column}, COUNT(*) AS n
                  FROM {table}
                 WHERE {' AND '.join(name_clauses)}
                 GROUP BY {id_column}, {name_column}
                 ORDER BY n DESC, {id_column}
                 LIMIT 32""",
            name_params,
        ).fetchall()
        by_id: dict[str, dict[str, Any]] = {str(item['id']): item for item in candidates}
        for row in name_rows:
            actor_id = str(row[0])
            entry = by_id.get(actor_id)
            if entry is None:
                entry = {'id': actor_id, 'count': 0, 'accounts': [], 'names': [], 'match': 'actor_name'}
                by_id[actor_id] = entry
                candidates.append(entry)
            entry['count'] += int(row[2])
            stored_name = str(row[1] or '')
            if stored_name and stored_name not in entry['names']:
                entry['names'].append(stored_name)
        candidates.sort(key=lambda item: (-int(item['count']), str(item['id'])))

    if not candidates and store is not None:
        candidates = _fuzzy_person_candidates(
            store, conn, target,
            table=table, id_column=id_column, account_id=account_id,
        )

    candidates = candidates[:_RESOLUTION_CANDIDATE_LIMIT]
    if len(candidates) == 1:
        return str(candidates[0]['id']), candidates
    return None, candidates


def _page(
    conn: Any,
    *,
    capability: str,
    filters: Mapping[str, Any],
    select_sql: str,
    count_sql: str,
    params: list[Any],
    count_params: list[Any],
    id_column: str,
    keyset: Mapping[str, Any] | None,
    limit: int,
    generation: str,
    vault_identity: str,
    time_column: str = 'timestamp',
) -> tuple[list[dict[str, Any]], int, str | None]:
    clauses: list[str] = []
    keyset_params: list[Any] = []
    if keyset:
        key_ts = keyset.get('timestamp')
        key_id = str(keyset.get('id') or '')
        if key_ts is None:
            clauses.append(f'({time_column} IS NULL AND {id_column}<?)')
            keyset_params.append(key_id)
        else:
            clauses.append(
                f'({time_column}<? OR ({time_column}=? AND {id_column}<?) OR {time_column} IS NULL)'
            )
            keyset_params.extend([str(key_ts), str(key_ts), key_id])
    sql = select_sql
    if clauses:
        sql += ' AND ' + ' AND '.join(clauses)
    sql += f' ORDER BY {time_column} DESC, {id_column} DESC LIMIT ?'
    rows = [dict(row) for row in conn.execute(sql, (*params, *keyset_params, limit + 1)).fetchall()]
    has_more = len(rows) > limit
    items = rows[:limit]
    total = int(conn.execute(count_sql, count_params).fetchone()[0])
    seen = int((keyset or {}).get('seen') or 0)
    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = _issue_cursor(
            capability=capability,
            filters=filters,
            keyset={'timestamp': last.get('timestamp'), 'id': str(last.get(id_column) or ''), 'seen': seen + len(items)},
            high_water=str(total),
            generation=generation,
            vault_identity=vault_identity,
        )
    return items, total, next_cursor


def _finish(
    data: Mapping[str, Any],
    *,
    returned: int,
    total: int,
    seen: int,
    next_cursor: str | None,
) -> HandlerOutcome:
    has_more = next_cursor is not None
    page: dict[str, Any] = {'has_more': has_more}
    if next_cursor is not None:
        page['next_cursor'] = next_cursor
    remaining = max(total - seen - returned, 0)
    return HandlerOutcome.success(
        data,
        page=page,
        coverage={'state': 'partial' if has_more else 'complete', 'returned': returned, 'remaining': remaining},
    )


def _open(config: Any):
    owner = _owner(config)
    cfg = owner.config if owner is not None else config
    if not cfg.paths.sqlite_path.is_file():
        return None, None, cfg
    store = owner.read_store if owner is not None else SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    store.initialize()
    return owner, store, cfg


def _close(owner: Any, store: Any) -> None:
    if owner is None and store is not None:
        store.close()


def _cursor_state(
    payload: Mapping[str, Any],
    *,
    capability: str,
    filters: Mapping[str, Any],
    generation: str,
    vault_identity: str,
) -> tuple[Mapping[str, Any] | None, HandlerOutcome | None]:
    handle = payload.get('cursor')
    if not handle:
        return None, None
    try:
        state = _resolve_cursor(
            str(handle),
            capability=capability,
            filters=filters,
            generation=generation,
            vault_identity=vault_identity,
        )
    except CursorError as exc:
        return None, HandlerOutcome.failure(exc.code, str(exc))
    return state.keyset, None


def moment_timeline(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    target = str(payload.get('target') or '').strip()
    if not target:
        return HandlerOutcome.failure('invalid_request', 'A target author is required.')
    limit = _bounded('limit', payload.get('limit'), MOMENT_TIMELINE)
    if isinstance(limit, HandlerOutcome):
        return limit
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None
    since = str(payload['since']) if payload.get('since') else None
    until = str(payload['until']) if payload.get('until') else None

    owner, store, cfg = _open(config)
    if store is None:
        return _finish(
            {'moments': [], 'scope': {'target': target, 'account_id': account_id}, 'matched_total': 0},
            returned=0, total=0, seen=0, next_cursor=None,
        )
    try:
        with vault_generation_read(cfg) as token:
            generation = _generation_digest(token)
            vault_identity = _vault_identity(cfg)
            with store.connect() as conn:
                author_id, candidates = _resolve_person(
                    conn, target,
                    table='moment_items', id_column='author_id', account_id=account_id,
                    store=store,
                )
                if author_id is None:
                    return HandlerOutcome.failure(
                        'ambiguous_target' if candidates else 'no_results',
                        'Target does not resolve to exactly one moment author.',
                        details={'candidates': candidates},
                    )
                filters = {
                    'author_id': author_id,
                    'account_id': account_id or '',
                    'since': since or '',
                    'until': until or '',
                }
                keyset, failure = _cursor_state(
                    payload,
                    capability=_TIMELINE_CAPABILITY,
                    filters=filters,
                    generation=generation,
                    vault_identity=vault_identity,
                )
                if failure is not None:
                    return failure
                clauses = ['author_id=?', "status='active'"]
                params: list[Any] = [author_id]
                if account_id:
                    clauses.append('account_id=?')
                    params.append(account_id)
                if since:
                    clauses.append('timestamp>=?')
                    params.append(since)
                if until:
                    clauses.append('timestamp<?')
                    params.append(until)
                where = ' WHERE ' + ' AND '.join(clauses)
                items, total, next_cursor = _page(
                    conn,
                    capability=_TIMELINE_CAPABILITY,
                    filters=filters,
                    select_sql=(
                        'SELECT moment_id, account_id, author_id, citation, timestamp,'
                        ' text, media_refs_json, link_json FROM moment_items' + where
                    ),
                    count_sql='SELECT COUNT(*) FROM moment_items' + where,
                    params=params,
                    count_params=params,
                    id_column='moment_id',
                    keyset=keyset,
                    limit=limit,
                    generation=generation,
                    vault_identity=vault_identity,
                )
                counts = _interaction_counts(conn, [str(item['moment_id']) for item in items])
            moments = [_timeline_item(item, counts.get(str(item['moment_id']), {})) for item in items]
            seen = int((keyset or {}).get('seen') or 0)
            return _finish(
                {
                    'moments': moments,
                    'scope': {'author_id': author_id, 'account_id': account_id},
                    'matched_total': total,
                },
                returned=len(moments), total=total, seen=seen, next_cursor=next_cursor,
            )
    finally:
        _close(owner, store)


def _interaction_counts(conn: Any, moment_ids: list[str]) -> dict[str, dict[str, int]]:
    if not moment_ids:
        return {}
    rows = conn.execute(
        f"""SELECT moment_id, interaction_type, COUNT(*) AS n
              FROM moment_interactions
             WHERE status='active' AND moment_id IN ({_placeholders(len(moment_ids))})
             GROUP BY moment_id, interaction_type""",
        moment_ids,
    ).fetchall()
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = counts.setdefault(str(row[0]), {'likes': 0, 'comments': 0, 'total': 0})
        kind = 'likes' if str(row[1]) == 'like' else 'comments' if str(row[1]) == 'comment' else None
        if kind:
            entry[kind] += int(row[2])
        entry['total'] += int(row[2])
    return counts


def _timeline_item(row: Mapping[str, Any], counts: Mapping[str, int]) -> dict[str, Any]:
    excerpt, truncated = _excerpt(row.get('text'))
    return {
        'citation': str(row.get('citation') or ''),
        'account_id': str(row.get('account_id') or ''),
        'author_id': str(row.get('author_id') or ''),
        'timestamp': row.get('timestamp'),
        'text': excerpt,
        'text_truncated': truncated,
        'media_count': len(_json_list(row.get('media_refs_json'))),
        'link': _link_summary(row.get('link_json')),
        'interactions': {
            'likes': int(counts.get('likes') or 0),
            'comments': int(counts.get('comments') or 0),
            'total': int(counts.get('total') or 0),
        },
        'trust': 'untrusted_evidence',
    }


def moment_interactions(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    citation = str(payload.get('citation') or '').strip()
    target = str(payload.get('target') or '').strip()
    if not citation and not target:
        return HandlerOutcome.failure('invalid_request', 'A moment citation or an actor target is required.')
    limit = _bounded('limit', payload.get('limit'), MOMENT_INTERACTIONS)
    if isinstance(limit, HandlerOutcome):
        return limit
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None
    since = str(payload['since']) if payload.get('since') else None
    until = str(payload['until']) if payload.get('until') else None

    owner, store, cfg = _open(config)
    if store is None:
        return _finish(
            {'interactions': [], 'matched_total': 0},
            returned=0, total=0, seen=0, next_cursor=None,
        )
    try:
        with vault_generation_read(cfg) as token:
            generation = _generation_digest(token)
            vault_identity = _vault_identity(cfg)
            with store.connect() as conn:
                moment_summary = None
                if citation:
                    moment_row = conn.execute(
                        'SELECT moment_id, account_id, author_id, citation, timestamp, text, status'
                        ' FROM moment_items WHERE citation=? LIMIT 1',
                        (citation,),
                    ).fetchone()
                    if moment_row is None or str(moment_row['status']) != 'active':
                        return HandlerOutcome.failure('no_results', 'No moment matched the citation.')
                    if account_id and str(moment_row['account_id']) != account_id:
                        return HandlerOutcome.failure('no_results', 'No moment matched the citation in this account scope.')
                    excerpt, truncated = _excerpt(moment_row['text'])
                    moment_summary = {
                        'citation': str(moment_row['citation']),
                        'account_id': str(moment_row['account_id']),
                        'author_id': str(moment_row['author_id'] or ''),
                        'timestamp': moment_row['timestamp'],
                        'text': excerpt,
                        'text_truncated': truncated,
                    }
                    scope_clauses = ['moment_id=?']
                    scope_params: list[Any] = [str(moment_row['moment_id'])]
                    filters = {
                        'mode': 'moment',
                        'moment_id': str(moment_row['moment_id']),
                        'since': since or '',
                        'until': until or '',
                    }
                    scope = {'moment_citation': citation, 'account_id': str(moment_row['account_id'])}
                else:
                    actor_id, candidates = _resolve_person(
                        conn, target,
                        table='moment_interactions', id_column='actor_id',
                        account_id=account_id, name_column='actor_name',
                        store=store,
                    )
                    if actor_id is None:
                        return HandlerOutcome.failure(
                            'ambiguous_target' if candidates else 'no_results',
                            'Target does not resolve to exactly one moment actor.',
                            details={'candidates': candidates},
                        )
                    scope_clauses = ['actor_id=?']
                    scope_params = [actor_id]
                    if account_id:
                        scope_clauses.append('account_id=?')
                        scope_params.append(account_id)
                    filters = {
                        'mode': 'actor',
                        'actor_id': actor_id,
                        'account_id': account_id or '',
                        'since': since or '',
                        'until': until or '',
                    }
                    scope = {'actor_id': actor_id, 'account_id': account_id}
                keyset, failure = _cursor_state(
                    payload,
                    capability=_INTERACTIONS_CAPABILITY,
                    filters=filters,
                    generation=generation,
                    vault_identity=vault_identity,
                )
                if failure is not None:
                    return failure
                clauses = [*scope_clauses, "status='active'"]
                params = list(scope_params)
                if since:
                    clauses.append('timestamp>=?')
                    params.append(since)
                if until:
                    clauses.append('timestamp<?')
                    params.append(until)
                where = ' WHERE ' + ' AND '.join(clauses)
                items, total, next_cursor = _page(
                    conn,
                    capability=_INTERACTIONS_CAPABILITY,
                    filters=filters,
                    select_sql=(
                        'SELECT interaction_id, moment_id, account_id, citation, interaction_type,'
                        ' actor_id, actor_name, text, timestamp FROM moment_interactions' + where
                    ),
                    count_sql='SELECT COUNT(*) FROM moment_interactions' + where,
                    params=params,
                    count_params=params,
                    id_column='interaction_id',
                    keyset=keyset,
                    limit=limit,
                    generation=generation,
                    vault_identity=vault_identity,
                )
            interactions = [_interaction_item(item) for item in items]
            data: dict[str, Any] = {
                'interactions': interactions,
                'scope': scope,
                'matched_total': total,
            }
            if moment_summary is not None:
                data['moment'] = moment_summary
            seen = int((keyset or {}).get('seen') or 0)
            return _finish(
                data,
                returned=len(interactions), total=total, seen=seen, next_cursor=next_cursor,
            )
    finally:
        _close(owner, store)


def _interaction_item(row: Mapping[str, Any]) -> dict[str, Any]:
    citation = str(row.get('citation') or '')
    moment_citation = citation.split('/interaction/', 1)[0] if '/interaction/' in citation else ''
    excerpt, truncated = _excerpt(row.get('text'))
    return {
        'citation': citation,
        'account_id': str(row.get('account_id') or ''),
        'moment_citation': moment_citation,
        'interaction_type': str(row.get('interaction_type') or ''),
        'actor_id': str(row.get('actor_id') or ''),
        'actor_name': str(row.get('actor_name') or ''),
        'timestamp': row.get('timestamp'),
        'text': excerpt,
        'text_truncated': truncated,
        'trust': 'untrusted_evidence',
    }


__all__ = ['moment_interactions', 'moment_timeline']
