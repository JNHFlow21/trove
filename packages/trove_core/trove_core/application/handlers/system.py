from __future__ import annotations

from typing import Any, Mapping

from trove_core.application.queries import TroveQueries
from trove_core.store.sqlite_store import SQLiteStore
from trove_protocol.capabilities import CATALOG, PROTOCOL_VERSION

from .base import HandlerOutcome, from_query_result


_ACCOUNT_SUMMARY_SQL = """
    WITH message_summary AS (
        SELECT account_id,
               COUNT(*) AS message_count,
               MAX(timestamp) AS latest_message
          FROM messages
         GROUP BY account_id
    ),
    sync_summary AS (
        SELECT account_id,
               MAX(updated_at) AS latest_sync
          FROM sync_state
         GROUP BY account_id
    )
    SELECT a.account_id,
           a.label,
           COALESCE(m.message_count, 0) AS message_count,
           m.latest_message,
           s.latest_sync
      FROM accounts a
      LEFT JOIN message_summary m ON m.account_id = a.account_id
      LEFT JOIN sync_summary s ON s.account_id = a.account_id
     ORDER BY a.account_id
"""


def _accounts(config: Any) -> list[dict[str, Any]]:
    owner = config if hasattr(config, 'read_store') and hasattr(config, 'config') else None
    config = owner.config if owner is not None else config
    if not config.paths.sqlite_path.is_file():
        return []
    store = owner.read_store if owner is not None else SQLiteStore(config.paths.sqlite_path, readonly=True)
    store.initialize()
    try:
        with store.connect() as conn:
            rows = conn.execute(_ACCOUNT_SUMMARY_SQL)
            return [
                {
                    'account_id': str(row['account_id']),
                    'label': str(row['label']),
                    'message_count': int(row['message_count'] or 0),
                    'latest_sync': row['latest_sync'] or row['latest_message'],
                }
                for row in rows
            ]
    finally:
        if owner is None:
            store.close()


def _provider_methods(config: Any) -> set[str]:
    owner = config if hasattr(config, 'provider_registry') else None
    if owner is None or not bool((owner.status().get('provider_load') or {}).get('ok')):
        return set()
    registry = owner.provider_registry
    if registry is None:
        return set()
    methods: set[str] = set()
    for item in registry.status():
        methods.update(str(value) for value in (item.get('capabilities') or ()))
    return methods


def capabilities(
    config: Any,
    _payload: Mapping[str, Any],
    *,
    installed_capabilities: frozenset[str] | set[str] | None = None,
) -> HandlerOutcome:
    installed = set(installed_capabilities or ())
    provider_methods = _provider_methods(config)

    def available(spec: Any) -> bool:
        if spec.capability_id not in installed:
            return False
        requirement = spec.provider_requirement
        if requirement == 'read_provider':
            return 'read' in provider_methods
        if requirement == 'media_provider':
            return 'media' in provider_methods
        return True

    return HandlerOutcome.success({
        'protocol': PROTOCOL_VERSION,
        'accounts': _accounts(config),
        'capabilities': [
            {
                'id': spec.capability_id,
                'pack': spec.pack,
                'risk': spec.risk,
                'available': available(spec),
                **({'provider_requirement': spec.provider_requirement} if spec.provider_requirement else {}),
            }
            for spec in CATALOG
        ],
    })


def resolve(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    target = str(payload.get('target') or '').strip()
    if not target or payload.get('kind') == 'account':
        accounts = _accounts(config)
        if payload.get('account_id'):
            accounts = [item for item in accounts if item['account_id'] == payload['account_id']]
        return HandlerOutcome.success({'accounts': accounts})
    queries = getattr(config, 'queries', None)
    queries = queries if isinstance(queries, TroveQueries) else TroveQueries(config)
    return from_query_result(queries.resolve_contact(
        contact=target,
        conversation_id=None,
        account_id=payload.get('account_id'),
    ))


def diagnostics(config: Any, _payload: Mapping[str, Any]) -> HandlerOutcome:
    owner = config if hasattr(config, 'status') and hasattr(config, 'config') else None
    cfg = owner.config if owner is not None else config
    runtime = owner.status() if owner is not None else {}
    provider = runtime.get('provider_load') or {
        'ok': False,
        'error': {'code': 'provider_not_loaded', 'retryable': False},
        'next_action': {
            'capability': 'trove.provider_status',
            'action': 'install_or_repair_provider',
        },
        'pure_vault_read_available': True,
    }
    return HandlerOutcome.success({
        'protocol': PROTOCOL_VERSION,
        'vault_ready': cfg.paths.sqlite_path.is_file(),
        'transport': 'unix',
        'provider': provider,
        'runtime': {
            'workers': runtime.get('workers') or {},
            'cache_entries': int(runtime.get('result_cache_entries') or 0),
            'cache_bytes': int(runtime.get('result_cache_bytes') or 0),
        },
    })


def provider_status(config: Any, _payload: Mapping[str, Any]) -> HandlerOutcome:
    owner = config if hasattr(config, 'status') and hasattr(config, 'config') else None
    runtime = owner.status() if owner is not None else {}
    return HandlerOutcome.success({
        'provider': runtime.get('provider_load') or {
            'ok': False,
            'error': {'code': 'provider_not_loaded', 'retryable': False},
            'next_action': {
                'capability': 'trove.provider_status',
                'action': 'install_or_repair_provider',
            },
            'pure_vault_read_available': True,
        },
    })


__all__ = ['capabilities', 'diagnostics', 'provider_status', 'resolve']
