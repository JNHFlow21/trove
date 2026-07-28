from __future__ import annotations

from typing import Any, Mapping

from trove_core.application.queries import (
    ContextQuery,
    FilesQuery,
    SearchQuery,
    TroveQueries,
)
from trove_core.store.sqlite_store import SQLiteStore

from .base import HandlerOutcome, from_query_result


def _queries(runtime: Any) -> TroveQueries:
    owned = getattr(runtime, 'queries', None)
    return owned if isinstance(owned, TroveQueries) else TroveQueries(runtime)


def _config(runtime: Any):
    return getattr(runtime, 'config', runtime)


def _recall(config: Any, payload: Mapping[str, Any], *, group_only: bool) -> HandlerOutcome:
    owner = config if hasattr(config, 'read_store') and hasattr(config, 'config') else None
    cfg = owner.config if owner is not None else config
    store = owner.read_store if owner is not None else SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    target = str(payload.get('target') or '').strip()
    account_id = payload.get('account_id')
    conversation_id = payload.get('conversation_id')
    try:
        store.initialize()
        with store.connect() as connection:
            candidates: list[dict[str, Any]] = []
            if target and not conversation_id:
                clauses = ['lower(title)=lower(?)']
                parameters: list[Any] = [target]
                if account_id:
                    clauses.append('account_id=?')
                    parameters.append(account_id)
                rows = connection.execute(
                    f'''SELECT account_id,conversation_id,title,type,member_count
                          FROM conversations WHERE {' AND '.join(clauses)}
                         ORDER BY account_id,conversation_id LIMIT 11''',
                    parameters,
                ).fetchall()
                candidates = [dict(row) for row in rows]
                if len(candidates) != 1:
                    return HandlerOutcome.failure(
                        'ambiguous_target' if candidates else 'no_results',
                        'Target does not resolve to exactly one conversation.',
                        details={'candidates': candidates},
                    )
                account_id = candidates[0]['account_id']
                conversation_id = candidates[0]['conversation_id']
            if conversation_id:
                clauses = ['conversation_id=?']
                parameters = [conversation_id]
                if account_id:
                    clauses.append('account_id=?')
                    parameters.append(account_id)
                rows = connection.execute(
                    f'''SELECT account_id,conversation_id,title,type,member_count
                          FROM conversations WHERE {' AND '.join(clauses)}
                         ORDER BY account_id LIMIT 11''',
                    parameters,
                ).fetchall()
                candidates = [dict(row) for row in rows]
                if len(candidates) != 1:
                    return HandlerOutcome.failure(
                        'ambiguous_target' if candidates else 'no_results',
                        'Conversation does not resolve to exactly one account.',
                        details={'candidates': candidates},
                    )
                account_id = candidates[0]['account_id']
                if group_only and candidates[0]['type'] != 'group':
                    return HandlerOutcome.failure('group_scope_required', 'Group summary requires one group conversation.')
            where: list[str] = []
            params: list[Any] = []
            for field in ('account_id', 'conversation_id'):
                value = account_id if field == 'account_id' else conversation_id
                if value:
                    where.append(f'{field}=?')
                    params.append(value)
            direction = str(payload.get('direction') or 'both')
            if direction != 'both':
                where.append('direction=?')
                params.append(direction)
            if payload.get('since'):
                where.append('timestamp>=?')
                params.append(payload['since'])
            if payload.get('until'):
                where.append('timestamp<?')
                params.append(payload['until'])
            query = str(payload.get('query') or '').strip()
            if query:
                where.append('content LIKE ?')
                params.append(f'%{query}%')
            limit = int(payload.get('limit') or 100)
            predicate = ' WHERE ' + ' AND '.join(where) if where else ''
            total = int(connection.execute(
                f'SELECT COUNT(*) FROM messages{predicate}', params,
            ).fetchone()[0])
            rows = connection.execute(
                f'''SELECT account_id,account_label,conversation_id,conversation_title,
                           conversation_type,sender_id,sender_name,timestamp,content,
                           shard_id,local_id,direction,source_type,content_kind,citation
                      FROM messages{predicate}
                     ORDER BY timestamp,shard_id,local_id LIMIT ?''',
                (*params, limit),
            ).fetchall()
            messages = [dict(row) | {'trust': 'untrusted_evidence'} for row in rows]
    finally:
        if owner is None:
            store.close()
    has_more = total > len(messages)
    return HandlerOutcome.success(
        {
            'messages': messages,
            'scope': {'account_id': account_id, 'conversation_id': conversation_id},
            'matched_total': total,
        },
        page={'has_more': has_more},
        coverage={'state': 'partial' if has_more else 'complete', 'returned': len(messages), 'remaining': max(total - len(messages), 0)},
    )


def recall(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    return _recall(config, payload, group_only=False)


def group_summary(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    return _recall(config, payload, group_only=True)


def search(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    result = _queries(config).search(SearchQuery(
        str(payload['query']),
        contact=payload.get('target'),
        account_id=payload.get('account_id'),
        semantic=str(payload.get('semantic') or 'auto'),
        limit=int(payload.get('limit') or 100),
    ))
    return from_query_result(result, paginated=True)


def context(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    result = _queries(config).context(ContextQuery(
        str(payload['citation']),
        before=int(payload.get('before', 5)),
        after=int(payload.get('after', 5)),
    ))
    return from_query_result(result)


def resolve(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    target = str(payload.get('target') or '').strip()
    if not target:
        return HandlerOutcome.failure('no_results', 'A target is required to resolve one conversation.')
    result = _queries(config).resolve_contact(
        contact=target,
        conversation_id=None,
        account_id=payload.get('account_id'),
    )
    return from_query_result(result)


def files_list(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    result = _queries(config).list_files(FilesQuery(
        account_id=payload.get('account_id'),
        contact=payload.get('target'),
        conversation_id=payload.get('conversation_id'),
        media_types=payload.get('media_types'),
        limit=int(payload.get('limit') or 100),
    ))
    return from_query_result(result, paginated=True)


def profile(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    from trove_core.agent_tools.tools import person_profile

    result = person_profile(
        _config(config).root,
        str(payload['target']),
        evidence_limit=int(payload.get('limit') or 12),
    )
    if result.get('ok') is False:
        error = result.get('error') or {}
        return HandlerOutcome.failure(
            str(error.get('code') or result.get('code') or 'profile_failed'),
            str(error.get('message') or 'Profile query failed.'),
            details={key: value for key, value in error.items() if key not in {'code', 'message'}},
        )
    return HandlerOutcome.success({key: value for key, value in result.items() if key not in {'ok', 'code'}})


def media_fetch(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    from trove_core.agent_tools.tools import media_fetch as fetch

    result = fetch(
        _config(config).root,
        str(payload['citation']),
        allow_remote=bool(payload.get('allow_remote', False)),
    )
    if result.get('ok') is False:
        error = result.get('error') or {}
        return HandlerOutcome.failure(
            str(error.get('code') or result.get('code') or 'media_fetch_failed'),
            str(error.get('message') or 'Media fetch failed.'),
            details={key: value for key, value in error.items() if key not in {'code', 'message'}},
        )
    return HandlerOutcome.success(result)


__all__ = [
    'context', 'files_list', 'group_summary', 'media_fetch', 'profile',
    'recall', 'resolve', 'search',
]
