"""Read-only bounded favorites list evidence queries.

Favorites are user-curated Vault rows, not bulk traffic.  Pages are
keyset-ordered by (timestamp, favorite_id) with opaque cursors bound to
capability, filters, Vault generation, identity and TTL.  Stored timestamps
are source epoch seconds, so time filters accept epoch seconds or ISO 8601
and normalize to the stored form.  Kind is a deterministic derived class:
'media' when the row carries media references, otherwise 'note'.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import threading
from typing import Any, Mapping

from trove_core.bounds import FAVORITES_LIST, BoundedInputError, bounded_limit
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.generation import vault_generation_read
from trove_protocol.cursors import CursorError, CursorStore

from .base import HandlerOutcome


_FAVORITES_CAPABILITY = 'trove.favorites_list'
_TEXT_EXCERPT_CHARS = 240
_TITLE_EXCERPT_CHARS = 120

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


def _clip(text: Any, chars: int) -> tuple[str, bool]:
    value = str(text or '')
    return value[:chars], len(value) > chars


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


def _time_bound(field: str, value: Any) -> str | None | HandlerOutcome:
    """Normalize one time filter to the stored epoch-second text form."""

    text = str(value or '').strip()
    if not text:
        return None
    if text.isdigit():
        return text
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return HandlerOutcome.failure(
            'invalid_request',
            f'{field} must be epoch seconds or an ISO 8601 timestamp.',
            details={'field': field},
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return str(int(parsed.timestamp()))


def _like_pattern(keyword: str) -> str:
    escaped = keyword.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return f'%{escaped}%'


def _page(
    conn: Any,
    *,
    capability: str,
    filters: Mapping[str, Any],
    select_sql: str,
    count_sql: str,
    params: list[Any],
    keyset: Mapping[str, Any] | None,
    limit: int,
    generation: str,
    vault_identity: str,
) -> tuple[list[dict[str, Any]], int, str | None]:
    clauses: list[str] = []
    keyset_params: list[Any] = []
    if keyset:
        key_ts = keyset.get('timestamp')
        key_id = str(keyset.get('id') or '')
        if key_ts is None:
            clauses.append('(timestamp IS NULL AND favorite_id<?)')
            keyset_params.append(key_id)
        else:
            clauses.append(
                '(timestamp<? OR (timestamp=? AND favorite_id<?) OR timestamp IS NULL)'
            )
            keyset_params.extend([str(key_ts), str(key_ts), key_id])
    sql = select_sql
    if clauses:
        sql += (' WHERE ' if ' WHERE ' not in select_sql else ' AND ') + ' AND '.join(clauses)
    sql += ' ORDER BY timestamp DESC, favorite_id DESC LIMIT ?'
    rows = [dict(row) for row in conn.execute(sql, (*params, *keyset_params, limit + 1)).fetchall()]
    has_more = len(rows) > limit
    items = rows[:limit]
    total = int(conn.execute(count_sql, params).fetchone()[0])
    seen = int((keyset or {}).get('seen') or 0)
    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = _issue_cursor(
            capability=capability,
            filters=filters,
            keyset={'timestamp': last.get('timestamp'), 'id': str(last.get('favorite_id') or ''), 'seen': seen + len(items)},
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


def _favorite_item(row: Mapping[str, Any]) -> dict[str, Any]:
    text, text_truncated = _clip(row.get('text'), _TEXT_EXCERPT_CHARS)
    title, title_truncated = _clip(row.get('title'), _TITLE_EXCERPT_CHARS)
    media_refs = _json_list(row.get('media_refs_json'))
    metadata = _json_dict(row.get('metadata_json'))
    source = None
    table = str(metadata.get('table') or '')
    if table:
        source = {'table': table}
        rowid = metadata.get('rowid')
        if type(rowid) is int:
            source['rowid'] = rowid
    return {
        'citation': str(row.get('citation') or ''),
        'account_id': str(row.get('account_id') or ''),
        'timestamp': row.get('timestamp'),
        'kind': 'media' if media_refs else 'note',
        'title': title,
        'title_truncated': title_truncated,
        'text': text,
        'text_truncated': text_truncated,
        'media_count': len(media_refs),
        'source': source,
        'trust': 'untrusted_evidence',
    }


def favorites_list(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    keyword = str(payload.get('keyword') or '').strip()
    kind = payload.get('kind')
    kind = str(kind) if kind else None
    limit = _bounded('limit', payload.get('limit'), FAVORITES_LIST)
    if isinstance(limit, HandlerOutcome):
        return limit
    since = _time_bound('since', payload.get('since'))
    if isinstance(since, HandlerOutcome):
        return since
    until = _time_bound('until', payload.get('until'))
    if isinstance(until, HandlerOutcome):
        return until
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None

    owner, store, cfg = _open(config)
    if store is None:
        return _finish(
            {'favorites': [], 'scope': {'account_id': account_id}, 'matched_total': 0},
            returned=0, total=0, seen=0, next_cursor=None,
        )
    try:
        with vault_generation_read(cfg) as token:
            generation = _generation_digest(token)
            vault_identity = _vault_identity(cfg)
            with store.connect() as conn:
                filters = {
                    'keyword': keyword,
                    'kind': kind or '',
                    'account_id': account_id or '',
                    'since': since or '',
                    'until': until or '',
                }
                keyset, failure = _cursor_state(
                    payload,
                    capability=_FAVORITES_CAPABILITY,
                    filters=filters,
                    generation=generation,
                    vault_identity=vault_identity,
                )
                if failure is not None:
                    return failure
                clauses: list[str] = []
                params: list[Any] = []
                if account_id:
                    clauses.append('account_id=?')
                    params.append(account_id)
                if since:
                    clauses.append('timestamp>=?')
                    params.append(since)
                if until:
                    clauses.append('timestamp<?')
                    params.append(until)
                if kind == 'media':
                    clauses.append("media_refs_json<>'[]'")
                elif kind == 'note':
                    clauses.append("media_refs_json='[]'")
                if keyword:
                    clauses.append("(title LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\')")
                    pattern = _like_pattern(keyword)
                    params.extend([pattern, pattern])
                where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
                items, total, next_cursor = _page(
                    conn,
                    capability=_FAVORITES_CAPABILITY,
                    filters=filters,
                    select_sql=(
                        'SELECT favorite_id, account_id, citation, timestamp, title,'
                        ' text, media_refs_json, metadata_json FROM favorites' + where
                    ),
                    count_sql='SELECT COUNT(*) FROM favorites' + where,
                    params=params,
                    keyset=keyset,
                    limit=limit,
                    generation=generation,
                    vault_identity=vault_identity,
                )
            favorites = [_favorite_item(item) for item in items]
            seen = int((keyset or {}).get('seen') or 0)
            return _finish(
                {
                    'favorites': favorites,
                    'scope': {
                        'account_id': account_id,
                        'keyword': keyword or None,
                        'kind': kind,
                        'since': since,
                        'until': until,
                    },
                    'matched_total': total,
                },
                returned=len(favorites), total=total, seen=seen, next_cursor=next_cursor,
            )
    finally:
        _close(owner, store)


__all__ = ['favorites_list']
