"""Read-only bounded kind-filtered message listings.

Lists stored messages of exactly one declared content kind, newest first,
with opaque cursor pagination.  Coarse kinds (text/image/video/voice/
sticker) filter messages.content_kind through idx_messages_kind_time;
application-message kinds (link/file/miniapp/transfer/redpacket/
contact_card) filter the parsed payload catalog and join back to messages
by citation through covering payload indexes.  Red packets and contact
cards select on the raw appmsg type (2001/42) because the parser does not
name those subtypes yet.  Every row carries a citation, scope metadata, a
bounded summary and parser-shaped metadata — message content itself is
only exposed as the same bounded excerpt other listings already return.

Name resolution is deliberately absent: callers pass a conversation_id
obtained from trove.resolve, and an unknown or ambiguous id fails typed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import threading
from typing import Any, Mapping

from trove_core.bounds import MESSAGES_BY_KIND, BoundedInputError, bounded_limit
from trove_core.domain.content import display_content_for_kind
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.generation import vault_generation_read
from trove_protocol.cursors import CursorError, CursorStore

from .base import HandlerOutcome


_KIND_CAPABILITY = 'trove.messages_by_kind'
_SUMMARY_CHARS = 140
_METADATA_VALUE_CHARS = 200

_CONTENT_KINDS = frozenset({'text', 'image', 'video', 'voice', 'sticker'})
# Parser-named application subtypes, matched on message_payloads.normalized_type.
_PAYLOAD_TYPE_KINDS = {
    'link': ('link',),
    'file': ('file',),
    'miniapp': ('mini_program',),
    'transfer': ('transfer_notice',),
}
# Subtypes the parser stores as 'unsupported'; matched on the raw appmsg type.
_PAYLOAD_APPMSG_KINDS = {
    'redpacket': (2001,),
    'contact_card': (42,),
}
_KINDS = frozenset(_CONTENT_KINDS | _PAYLOAD_TYPE_KINDS.keys() | _PAYLOAD_APPMSG_KINDS.keys())
_DIRECTIONS = frozenset({'incoming', 'outgoing', 'unknown'})

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


def _time_filter(field: str, value: Any) -> str | None | HandlerOutcome:
    """Normalize one optional time filter given as ISO 8601 or epoch seconds."""

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
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


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
            capability=_KIND_CAPABILITY,
            filters=filters,
            generation=generation,
            vault_identity=vault_identity,
        )
    except CursorError as exc:
        return None, HandlerOutcome.failure(exc.code, str(exc))
    return state.keyset, None


def _resolve_conversation(
    conn: Any,
    conversation_id: str,
    account_id: str | None,
) -> tuple[str | None, HandlerOutcome | None]:
    """Scope to exactly one stored conversation account or fail typed."""

    clauses = ['conversation_id=?']
    params: list[Any] = [conversation_id]
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    rows = conn.execute(
        f'''SELECT account_id FROM conversations WHERE {' AND '.join(clauses)}
             ORDER BY account_id LIMIT 11''',
        params,
    ).fetchall()
    candidates = [str(row[0]) for row in rows]
    if len(candidates) == 1:
        return candidates[0], None
    if candidates:
        return None, HandlerOutcome.failure(
            'ambiguous_target',
            'Conversation id matches multiple accounts; pass account_id explicitly.',
            details={'candidates': candidates},
        )
    return None, HandlerOutcome.failure(
        'no_results',
        'No stored conversation matches the conversation scope.',
        details={'conversation_id': conversation_id},
    )


def _keyset_clause(keyset: Mapping[str, Any] | None, prefix: str) -> tuple[str, list[Any]]:
    if not keyset:
        return '', []
    key_ts = str(keyset.get('timestamp') or '')
    key_id = str(keyset.get('id') or '')
    return (
        f' AND ({prefix}timestamp<? OR ({prefix}timestamp=? AND {prefix}citation<?))',
        [key_ts, key_ts, key_id],
    )


def _bounded_metadata(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or ''))
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}

    def shrink(item: Any) -> Any:
        if isinstance(item, str):
            return item[:_METADATA_VALUE_CHARS]
        if isinstance(item, dict):
            return {str(key)[:64]: shrink(val) for key, val in list(item.items())[:16]}
        if isinstance(item, list):
            return [shrink(entry) for entry in item[:16]]
        return item

    return {str(key)[:64]: shrink(val) for key, val in list(parsed.items())[:16]}


def _summary(text: Any) -> tuple[str, bool]:
    value = str(text or '')
    return value[:_SUMMARY_CHARS], len(value) > _SUMMARY_CHARS


def _message_columns(prefix: str) -> str:
    return (
        f'{prefix}citation, {prefix}account_id, {prefix}conversation_id,'
        f' {prefix}conversation_title, {prefix}conversation_type, {prefix}sender_id,'
        f' {prefix}sender_name, {prefix}timestamp, {prefix}direction, {prefix}content_kind'
    )


def _payload_path(kind: str) -> tuple[str, str, list[Any]]:
    """SQL pieces for kinds stored in the parsed app message payload catalog."""

    if kind in _PAYLOAD_TYPE_KINDS:
        names = _PAYLOAD_TYPE_KINDS[kind]
        placeholders = ','.join('?' for _ in names)
        return (
            f'p.normalized_type IN ({placeholders})',
            'idx_message_payloads_type_citation',
            list(names),
        )
    types = _PAYLOAD_APPMSG_KINDS[kind]
    placeholders = ','.join('?' for _ in types)
    return (
        f'p.appmsg_type IN ({placeholders})',
        'idx_message_payloads_appmsg_citation',
        list(types),
    )


def messages_by_kind(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    kind = str(payload.get('kind') or '').strip()
    if kind not in _KINDS:
        return HandlerOutcome.failure(
            'invalid_request',
            f'kind must be one of {", ".join(sorted(_KINDS))}.',
            details={'kind': kind},
        )
    limit = _bounded('limit', payload.get('limit'), MESSAGES_BY_KIND)
    if isinstance(limit, HandlerOutcome):
        return limit
    direction = payload.get('direction')
    direction = str(direction) if direction else None
    if direction is not None and direction not in _DIRECTIONS:
        return HandlerOutcome.failure(
            'invalid_request',
            'direction must be one of incoming, outgoing, unknown.',
            details={'direction': direction},
        )
    since = _time_filter('since', payload.get('since'))
    if isinstance(since, HandlerOutcome):
        return since
    until = _time_filter('until', payload.get('until'))
    if isinstance(until, HandlerOutcome):
        return until
    if since and until and since >= until:
        return HandlerOutcome.failure(
            'invalid_request',
            'since must be earlier than until.',
            details={'since': since, 'until': until},
        )
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None
    conversation_id = payload.get('conversation_id')
    conversation_id = str(conversation_id) if conversation_id else None

    owner, store, cfg = _open(config)
    if store is None:
        return HandlerOutcome.success(
            {
                'kind': kind,
                'scope': {'account_id': account_id, 'conversation_id': conversation_id},
                'matched_total': 0,
                'messages': [],
            },
            page={'has_more': False},
            coverage={'state': 'complete', 'returned': 0, 'remaining': 0},
        )
    try:
        with vault_generation_read(cfg) as token:
            generation = _generation_digest(token)
            vault_identity = _vault_identity(cfg)
            with store.connect() as conn:
                if conversation_id:
                    resolved_account, failure = _resolve_conversation(conn, conversation_id, account_id)
                    if failure is not None:
                        return failure
                    account_id = resolved_account
                filters = {
                    'kind': kind,
                    'account_id': account_id or '',
                    'conversation_id': conversation_id or '',
                    'direction': direction or '',
                    'since': since or '',
                    'until': until or '',
                }
                keyset, failure = _cursor_state(
                    payload, filters=filters, generation=generation, vault_identity=vault_identity,
                )
                if failure is not None:
                    return failure
                message_clauses: list[str] = []
                params: list[Any] = []
                if account_id:
                    message_clauses.append('account_id=?')
                    params.append(account_id)
                if conversation_id:
                    message_clauses.append('conversation_id=?')
                    params.append(conversation_id)
                if direction:
                    message_clauses.append('direction=?')
                    params.append(direction)
                if since:
                    message_clauses.append('timestamp>=?')
                    params.append(since)
                if until:
                    message_clauses.append('timestamp<?')
                    params.append(until)
                if kind in _CONTENT_KINDS:
                    prefix = ''
                    # A conversation scope seeks the conversation-time index and
                    # filters the kind per row; an unscoped listing walks the
                    # kind-leading covering index backwards from the keyset.
                    message_index = (
                        'idx_messages_conversation_time' if conversation_id else 'idx_messages_kind_time'
                    )
                    where = ' AND '.join(['content_kind=?', *message_clauses])
                    keyset_sql, keyset_params = _keyset_clause(keyset, '')
                    select_sql = (
                        f'SELECT {_message_columns(prefix)}, content'
                        f' FROM messages INDEXED BY {message_index}'
                        f' WHERE {where}{keyset_sql}'
                    )
                    base_params = [kind, *params]
                    count_sql = f'SELECT COUNT(*) FROM messages INDEXED BY {message_index} WHERE {where}'
                else:
                    prefix = 'm.'
                    payload_filter, payload_index, payload_params = _payload_path(kind)
                    where = ' AND '.join(
                        [payload_filter, *[f'm.{clause}' for clause in message_clauses]],
                    )
                    keyset_sql, keyset_params = _keyset_clause(keyset, prefix)
                    # CROSS JOIN pins the bounded payload subtype seek ahead of
                    # the per-row message citation lookup.
                    select_sql = (
                        f'SELECT {_message_columns(prefix)},'
                        ' p.display_text, p.normalized_json, p.normalized_type, p.appmsg_type'
                        f' FROM message_payloads p INDEXED BY {payload_index}'
                        ' CROSS JOIN messages m INDEXED BY idx_messages_citation ON m.citation = p.citation'
                        f' WHERE {where}{keyset_sql}'
                    )
                    base_params = [*payload_params, *params]
                    count_sql = (
                        'SELECT COUNT(*)'
                        f' FROM message_payloads p INDEXED BY {payload_index}'
                        ' CROSS JOIN messages m INDEXED BY idx_messages_citation ON m.citation = p.citation'
                        f' WHERE {where}'
                    )
                rows = [
                    dict(row)
                    for row in conn.execute(
                        f'{select_sql} ORDER BY {prefix}timestamp DESC, {prefix}citation DESC LIMIT ?',
                        (*base_params, *keyset_params, limit + 1),
                    ).fetchall()
                ]
                has_more = len(rows) > limit
                items = rows[:limit]
                total = int(conn.execute(count_sql, base_params).fetchone()[0])
                seen = int((keyset or {}).get('seen') or 0)
                next_cursor = None
                if has_more and items:
                    last = items[-1]
                    next_cursor = _issue_cursor(
                        capability=_KIND_CAPABILITY,
                        filters=filters,
                        keyset={
                            'timestamp': str(last.get('timestamp') or ''),
                            'id': str(last.get('citation') or ''),
                            'seen': seen + len(items),
                        },
                        high_water=str(total),
                        generation=generation,
                        vault_identity=vault_identity,
                    )
            messages = [_item(item, kind) for item in items]
            has_more = next_cursor is not None
            page: dict[str, Any] = {'has_more': has_more}
            if next_cursor is not None:
                page['next_cursor'] = next_cursor
            return HandlerOutcome.success(
                {
                    'kind': kind,
                    'scope': {'account_id': account_id, 'conversation_id': conversation_id},
                    'matched_total': total,
                    'messages': messages,
                },
                page=page,
                coverage={
                    'state': 'partial' if has_more else 'complete',
                    'returned': len(messages),
                    'remaining': max(total - seen - len(messages), 0),
                },
            )
    finally:
        _close(owner, store)


def _item(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    content_kind = str(row.get('content_kind') or '')
    if kind in _CONTENT_KINDS:
        summary, truncated = _summary(display_content_for_kind(str(row.get('content') or ''), content_kind))
        metadata: dict[str, Any] = {}
    else:
        summary, truncated = _summary(row.get('display_text'))
        metadata = _bounded_metadata(row.get('normalized_json'))
    item: dict[str, Any] = {
        'citation': str(row.get('citation') or ''),
        'account_id': str(row.get('account_id') or ''),
        'conversation_id': str(row.get('conversation_id') or ''),
        'conversation_title': str(row.get('conversation_title') or '')[:120],
        'conversation_type': str(row.get('conversation_type') or ''),
        'sender_id': str(row.get('sender_id') or ''),
        'sender_name': str(row.get('sender_name') or '')[:120],
        'timestamp': str(row.get('timestamp') or ''),
        'direction': str(row.get('direction') or ''),
        'kind': kind,
        'content_kind': content_kind,
        'summary': summary,
        'summary_truncated': truncated,
        'metadata': metadata,
        'trust': 'untrusted_evidence',
    }
    if kind not in _CONTENT_KINDS:
        item['normalized_type'] = str(row.get('normalized_type') or '')
        appmsg_type = row.get('appmsg_type')
        item['appmsg_type'] = int(appmsg_type) if appmsg_type is not None else None
    return item


__all__ = ['messages_by_kind']
