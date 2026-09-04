"""Read-only bounded metadata-only message statistics aggregation.

Aggregates message COUNTS over one bounded time window.  The result carries
only counts and conversation/sender identifiers with citations — never
message content.  The window is mandatory in effect: it defaults to the
last 30 days and is hard-capped so aggregation over the bulk messages table
stays a bounded index range scan instead of a full-table sweep.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from trove_core.bounds import MESSAGE_STATS, BoundedInputError, bounded_limit
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.generation import vault_generation_read

from .base import HandlerOutcome


_STATS_CAPABILITY = 'trove.message_stats'
_DEFAULT_WINDOW = timedelta(days=30)
_MAX_WINDOW = timedelta(days=370)
_DIRECTIONS = ('incoming', 'outgoing', 'unknown')
_DIMENSIONS = ('by_conversation', 'by_sender')


def _owner(config: Any) -> Any | None:
    return config if hasattr(config, 'read_store') and hasattr(config, 'config') else None


def _bounded(field: str, value: Any, spec: Any) -> int | HandlerOutcome:
    try:
        return bounded_limit(
            spec.default if value is None else value, field=field, spec=spec,
        )
    except BoundedInputError as exc:
        return HandlerOutcome.failure(exc.code, str(exc), details=exc.to_dict())


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _parse_time(field: str, value: Any) -> datetime | None | HandlerOutcome:
    """Parse one time filter given as ISO 8601 or epoch seconds."""

    text = str(value or '').strip()
    if not text:
        return None
    try:
        if text.isdigit():
            parsed = datetime.fromtimestamp(int(text), tz=timezone.utc)
        else:
            parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return HandlerOutcome.failure(
            'invalid_request',
            f'{field} must be an ISO 8601 timestamp or epoch seconds.',
            details={'field': field},
        )
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _window(payload: Mapping[str, Any]) -> tuple[str, str] | HandlerOutcome:
    until = _parse_time('until', payload.get('until'))
    if isinstance(until, HandlerOutcome):
        return until
    since = _parse_time('since', payload.get('since'))
    if isinstance(since, HandlerOutcome):
        return since
    until = until or _parse_time('until', _iso(datetime.now(timezone.utc)))
    since = since or (until - _DEFAULT_WINDOW)
    if since >= until:
        return HandlerOutcome.failure(
            'invalid_request',
            'since must be earlier than until.',
            details={'since': _iso(since), 'until': _iso(until)},
        )
    if until - since > _MAX_WINDOW:
        return HandlerOutcome.failure(
            'invalid_request',
            f'time window must not exceed {int(_MAX_WINDOW.days)} days.',
            details={'max_days': int(_MAX_WINDOW.days)},
        )
    return _iso(since), _iso(until)


def _direction_pivots() -> str:
    return ','.join(
        f"SUM(CASE WHEN direction='{direction}' THEN 1 ELSE 0 END) AS {direction}"
        for direction in _DIRECTIONS
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


def _conversation_exists(conn: Any, conversation_id: str, account_id: str | None) -> bool:
    clauses = ['conversation_id=?']
    params: list[Any] = [conversation_id]
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    where = ' AND '.join(clauses)
    if conn.execute(f'SELECT 1 FROM conversations WHERE {where} LIMIT 1', params).fetchone():
        return True
    return conn.execute(f'SELECT 1 FROM messages WHERE {where} LIMIT 1', params).fetchone() is not None


def _window_filters(
    *,
    since: str,
    until: str,
    account_id: str | None,
    conversation_id: str | None,
    group_only: bool,
) -> tuple[str, list[Any]]:
    clauses = ['timestamp>=?', 'timestamp<?']
    params: list[Any] = [since, until]
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    if conversation_id:
        clauses.append('conversation_id=?')
        params.append(conversation_id)
    elif group_only:
        clauses.append("conversation_type='group'")
    return ' AND '.join(clauses), params


def _from_messages(conversation_id: str | None) -> str:
    # A conversation scope seeks idx_messages_conversation_time directly into
    # one conversation's window.  Every other shape must stay a bounded
    # timestamp range scan: without the pin the planner prefers an
    # account-leading index and scans the account's full history instead of
    # the window.
    if conversation_id:
        return 'messages'
    return 'messages INDEXED BY idx_messages_stats_time'


def _group_rows(
    conn: Any,
    *,
    dimension: str,
    from_clause: str,
    where: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    pivots = _direction_pivots()
    if dimension == 'by_conversation':
        sql = (
            f'SELECT account_id, conversation_id, COUNT(*) AS total, {pivots}'
            f' FROM {from_clause} WHERE {where} GROUP BY account_id, conversation_id'
        )
        key = 'conversation_id'
    else:
        sql = (
            f'SELECT account_id, sender_id, COUNT(*) AS total, {pivots},'
            ' COUNT(DISTINCT conversation_id) AS conversation_count'
            f' FROM {from_clause} WHERE {where} GROUP BY account_id, sender_id'
        )
        key = 'sender_id'
    rows = [
        {
            'account_id': str(row['account_id'] or ''),
            key: str(row[key] or ''),
            'total': int(row['total']),
            **{direction: int(row[direction]) for direction in _DIRECTIONS},
            **({'conversation_count': int(row['conversation_count'])} if dimension == 'by_sender' else {}),
        }
        for row in conn.execute(sql, params)
    ]
    return sorted(rows, key=lambda item: (-item['total'], item[key], item['account_id']))


def _enrich_conversations(conn: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        lookup = conn.execute(
            'SELECT title, type, member_count FROM conversations'
            ' WHERE conversation_id=? AND account_id=? LIMIT 1',
            (row['conversation_id'], row['account_id']),
        ).fetchone()
        account_id = row['account_id']
        conversation_id = row['conversation_id']
        enriched.append({
            'citation': f'trove://wechat/{account_id}/{conversation_id}',
            'account_id': account_id,
            'conversation_id': conversation_id,
            'title': str(lookup['title'])[:120] if lookup else '',
            'conversation_type': str(lookup['type']) if lookup else '',
            'member_count': int(lookup['member_count']) if lookup else None,
            'incoming': row['incoming'],
            'outgoing': row['outgoing'],
            'unknown': row['unknown'],
            'total': row['total'],
            'trust': 'untrusted_evidence',
        })
    return enriched


def _enrich_senders(conn: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        lookup = conn.execute(
            "SELECT sender_name FROM messages"
            " WHERE account_id=? AND sender_id=? AND sender_name<>'' LIMIT 1",
            (row['account_id'], row['sender_id']),
        ).fetchone()
        enriched.append({
            'account_id': row['account_id'],
            'sender_id': row['sender_id'],
            'sender_name': str(lookup['sender_name'])[:120] if lookup else '',
            'conversation_count': row['conversation_count'],
            'incoming': row['incoming'],
            'outgoing': row['outgoing'],
            'unknown': row['unknown'],
            'total': row['total'],
            'trust': 'untrusted_evidence',
        })
    return enriched


def message_stats(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    dimension = str(payload.get('dimension') or 'by_conversation')
    if dimension not in _DIMENSIONS:
        return HandlerOutcome.failure(
            'invalid_request',
            'dimension must be one of by_conversation, by_sender.',
            details={'dimension': dimension},
        )
    window = _window(payload)
    if not isinstance(window, tuple):
        return window
    since, until = window
    limit = _bounded('limit', payload.get('limit'), MESSAGE_STATS)
    if isinstance(limit, HandlerOutcome):
        return limit
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None
    conversation_id = payload.get('conversation_id')
    conversation_id = str(conversation_id) if conversation_id else None

    owner, store, cfg = _open(config)
    if store is None:
        return HandlerOutcome.success(
            {
                'dimension': dimension,
                'window': {'since': since, 'until': until},
                'scope': {'account_id': account_id, 'conversation_id': conversation_id},
                'window_message_count': 0,
                'matched_total': 0,
                'rows': [],
            },
            page={'has_more': False},
            coverage={'state': 'complete', 'returned': 0, 'remaining': 0},
        )
    try:
        with vault_generation_read(cfg):
            with store.connect() as conn:
                if conversation_id and not _conversation_exists(conn, conversation_id, account_id):
                    return HandlerOutcome.failure(
                        'no_results',
                        'No stored conversation matches the conversation scope.',
                        details={'conversation_id': conversation_id},
                    )
                where, params = _window_filters(
                    since=since, until=until,
                    account_id=account_id, conversation_id=conversation_id,
                    group_only=dimension == 'by_sender',
                )
                from_clause = _from_messages(conversation_id)
                window_total = int(
                    conn.execute(f'SELECT COUNT(*) FROM {from_clause} WHERE {where}', params).fetchone()[0]
                )
                grouped = _group_rows(conn, dimension=dimension, from_clause=from_clause, where=where, params=params)
                top = grouped[:limit]
                if dimension == 'by_conversation':
                    rows = _enrich_conversations(conn, top)
                else:
                    rows = _enrich_senders(conn, top)
            matched_total = len(grouped)
            return HandlerOutcome.success(
                {
                    'dimension': dimension,
                    'window': {'since': since, 'until': until},
                    'scope': {'account_id': account_id, 'conversation_id': conversation_id},
                    'window_message_count': window_total,
                    'matched_total': matched_total,
                    'rows': rows,
                },
                # Top-N has no continuation: the page is always final and the
                # coverage block carries the truncation state.
                page={'has_more': False},
                coverage={
                    'state': 'complete' if matched_total <= len(rows) else 'partial',
                    'returned': len(rows),
                    'remaining': max(matched_total - len(rows), 0),
                },
            )
    finally:
        _close(owner, store)


__all__ = ['message_stats']
