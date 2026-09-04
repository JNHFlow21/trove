"""Read-only bounded metadata-only pending reply triage.

Lists PRIVATE conversations whose latest incoming message inside one bounded
time window has no later outgoing reply as of the window end.  The result
carries conversation identifiers, the counterpart sender, timestamps, counts
and a waiting-duration bucket — never message content.  The window is
mandatory in effect: it defaults to the last 7 days and is hard-capped at 30
days so the aggregate stays one bounded covering range scan over
idx_messages_stats_time instead of a full-table sweep.

Known v1 limits, all deliberate: group conversations are excluded because
@-mention detection is unreliable; rows whose direction is 'unknown' cannot
prove a reply and are ignored; well-known system conversations (file
transfer, official accounts) are excluded by id shape because the catalog
carries no synthetic-conversation flag.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from trove_core.bounds import PENDING_REPLIES, BoundedInputError, bounded_limit
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.generation import vault_generation_read

from .base import HandlerOutcome


_DEFAULT_WINDOW = timedelta(days=7)
_MAX_WINDOW = timedelta(days=30)
# Well-known WeChat system conversations that never need a reply.  The
# conversations catalog has no synthetic-conversation marker, so the filter
# is an explicit id-shape denylist on top of conversation_type='private'.
_SYSTEM_CONVERSATION_IDS = ('filehelper', 'fmessage', 'medianote', 'floatbottle')
_SYSTEM_CONVERSATION_GLOB = 'gh_%'

_WAIT_BUCKETS = (
    (timedelta(hours=1), 'lt_1h'),
    (timedelta(days=1), '1h_1d'),
    (timedelta(days=3), '1d_3d'),
    (timedelta(days=7), '3d_7d'),
    (None, 'gt_7d'),
)


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


def _window(payload: Mapping[str, Any]) -> tuple[str, str, datetime] | HandlerOutcome:
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
    return _iso(since), _iso(until), until


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


def _waiting_bucket(last_incoming: str, until: datetime) -> str:
    try:
        moment = datetime.fromisoformat(str(last_incoming).replace('Z', '+00:00'))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
    except ValueError:
        return 'unknown'
    wait = until - moment.astimezone(timezone.utc)
    for ceiling, label in _WAIT_BUCKETS:
        if ceiling is None or wait < ceiling:
            return label
    return 'gt_7d'


def _window_clauses(*, since: str, until: str, account_id: str | None) -> tuple[str, list[Any]]:
    clauses = [
        'timestamp>=?', 'timestamp<?', "conversation_type='private'",
        f"conversation_id NOT IN ({','.join('?' for _ in _SYSTEM_CONVERSATION_IDS)})",
        "conversation_id NOT LIKE 'gh\\_%' ESCAPE '\\'",
    ]
    params: list[Any] = [since, until, *_SYSTEM_CONVERSATION_IDS]
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    return ' AND '.join(clauses), params


def _pending_rows(
    conn: Any,
    *,
    since: str,
    until: str,
    account_id: str | None,
) -> list[dict[str, Any]]:
    where, params = _window_clauses(since=since, until=until, account_id=account_id)
    # The pin keeps this a bounded covering range scan over the window;
    # without it the planner can trade the window for a full account-history
    # scan on conversation-leading indexes.
    sql = (
        'SELECT account_id, conversation_id,'
        " MAX(CASE WHEN direction='incoming' THEN timestamp END) AS last_incoming,"
        " MAX(CASE WHEN direction='outgoing' THEN timestamp END) AS last_outgoing,"
        " SUM(CASE WHEN direction='incoming' THEN 1 ELSE 0 END) AS incoming_count,"
        " SUM(CASE WHEN direction='outgoing' THEN 1 ELSE 0 END) AS outgoing_count"
        ' FROM messages INDEXED BY idx_messages_stats_time'
        f' WHERE {where}'
        ' GROUP BY account_id, conversation_id'
        ' HAVING last_incoming IS NOT NULL'
        ' AND (last_outgoing IS NULL OR last_outgoing < last_incoming)'
    )
    rows = [
        {
            'account_id': str(row['account_id'] or ''),
            'conversation_id': str(row['conversation_id'] or ''),
            'last_incoming': str(row['last_incoming']),
            'incoming_count': int(row['incoming_count']),
            'outgoing_count': int(row['outgoing_count']),
        }
        for row in conn.execute(sql, params)
    ]
    return sorted(
        rows,
        key=lambda item: (item['last_incoming'], item['conversation_id'], item['account_id']),
        reverse=True,
    )


def _window_private_count(conn: Any, *, since: str, until: str, account_id: str | None) -> int:
    where, params = _window_clauses(since=since, until=until, account_id=account_id)
    row = conn.execute(
        'SELECT COUNT(*) FROM messages INDEXED BY idx_messages_stats_time'
        f' WHERE {where}',
        params,
    ).fetchone()
    return int(row[0])


def _enrich(conn: Any, rows: list[dict[str, Any]], *, until: datetime) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        account_id = row['account_id']
        conversation_id = row['conversation_id']
        lookup = conn.execute(
            'SELECT title, member_count FROM conversations'
            ' WHERE conversation_id=? AND account_id=? LIMIT 1',
            (conversation_id, account_id),
        ).fetchone()
        # The counterpart is the sender of the latest incoming message; the
        # conversation-time index keeps this a bounded seek per emitted row.
        sender = conn.execute(
            "SELECT sender_id, sender_name FROM messages"
            " WHERE account_id=? AND conversation_id=? AND timestamp=? AND direction='incoming'"
            ' ORDER BY shard_id DESC, local_id DESC LIMIT 1',
            (account_id, conversation_id, row['last_incoming']),
        ).fetchone()
        enriched.append({
            'citation': f'trove://wechat/{account_id}/{conversation_id}',
            'account_id': account_id,
            'conversation_id': conversation_id,
            'title': str(lookup['title'])[:120] if lookup else '',
            'member_count': int(lookup['member_count']) if lookup else None,
            'sender_id': str(sender['sender_id']) if sender else '',
            'sender_name': str(sender['sender_name'])[:120] if sender else '',
            'last_incoming': row['last_incoming'],
            'waiting_bucket': _waiting_bucket(row['last_incoming'], until),
            'incoming_count': row['incoming_count'],
            'outgoing_count': row['outgoing_count'],
            'trust': 'untrusted_evidence',
        })
    return enriched


def pending_replies(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    window = _window(payload)
    if not isinstance(window, tuple):
        return window
    since, until, until_dt = window
    limit = _bounded('limit', payload.get('limit'), PENDING_REPLIES)
    if isinstance(limit, HandlerOutcome):
        return limit
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None

    owner, store, cfg = _open(config)
    if store is None:
        return HandlerOutcome.success(
            {
                'window': {'since': since, 'until': until},
                'scope': {'account_id': account_id},
                'window_private_message_count': 0,
                'matched_total': 0,
                'pending': [],
            },
            page={'has_more': False},
            coverage={'state': 'complete', 'returned': 0, 'remaining': 0},
        )
    try:
        with vault_generation_read(cfg):
            with store.connect() as conn:
                pending = _pending_rows(conn, since=since, until=until, account_id=account_id)
                window_count = _window_private_count(conn, since=since, until=until, account_id=account_id)
                rows = _enrich(conn, pending[:limit], until=until_dt)
            matched_total = len(pending)
            return HandlerOutcome.success(
                {
                    'window': {'since': since, 'until': until},
                    'scope': {'account_id': account_id},
                    'window_private_message_count': window_count,
                    'matched_total': matched_total,
                    'pending': rows,
                },
                # A ranked triage digest has no continuation: the page is
                # always final and the coverage block carries the truncation.
                page={'has_more': False},
                coverage={
                    'state': 'complete' if matched_total <= len(rows) else 'partial',
                    'returned': len(rows),
                    'remaining': max(matched_total - len(rows), 0),
                },
            )
    finally:
        _close(owner, store)


__all__ = ['pending_replies']
