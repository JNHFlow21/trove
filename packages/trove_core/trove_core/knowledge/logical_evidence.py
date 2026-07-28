from __future__ import annotations

from collections.abc import Iterable
import unicodedata
from typing import Any, Callable, Hashable


def _value(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def _text(value: Any) -> str:
    return ' '.join(unicodedata.normalize('NFKC', str(value or '')).casefold().split())


def logical_message_key(row: Any) -> tuple[Hashable, ...]:
    """Identify one WeChat event across duplicate imports of the same archive.

    Content and inferred direction are intentionally excluded: parser and
    account-identity improvements may change those projections while the
    source event remains the same. The table-local shard/local id, timestamp,
    context, sender label, and modality together retain distinct messages.
    """

    return (
        'wechat-message/v1',
        _text(_value(row, 'conversation_type')),
        _text(_value(row, 'conversation_title')),
        _text(_value(row, 'sender_name')),
        str(_value(row, 'timestamp') or ''),
        str(_value(row, 'shard_id') or ''),
        int(_value(row, 'local_id') or 0),
        _text(_value(row, 'content_kind') or 'text'),
    )


def logical_moment_key(row: Any) -> tuple[Hashable, ...]:
    return (
        'wechat-moment/v1',
        str(_value(row, 'author_id') or ''),
        str(_value(row, 'timestamp') or _value(row, 'evidence_timestamp') or ''),
        _text(_value(row, 'text')),
    )


def logical_moment_interaction_key(row: Any) -> tuple[Hashable, ...]:
    return (
        'wechat-moment-interaction/v1',
        str(_value(row, 'author_id') or ''),
        str(_value(row, 'actor_id') or ''),
        _text(_value(row, 'interaction_type')),
        str(_value(row, 'timestamp') or ''),
    )


def logical_moment_media_key(row: Any) -> tuple[Hashable, ...]:
    citation = str(_value(row, 'citation') or '')
    source_id = str(_value(row, 'source_id') or '')
    position = citation.rpartition('#')[2] if '#' in citation else source_id.rpartition('#')[2]
    return (
        *logical_moment_key(row),
        'media',
        _text(_value(row, 'modality')),
        position,
    )


def deduplicate_logical_rows(
    rows: Iterable[Any],
    *,
    key: Callable[[Any], Hashable],
) -> list[Any]:
    kept: list[Any] = []
    seen: set[Hashable] = set()
    for row in rows:
        identity = key(row)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(row)
    return kept


__all__ = [
    'deduplicate_logical_rows',
    'logical_message_key',
    'logical_moment_interaction_key',
    'logical_moment_key',
    'logical_moment_media_key',
]
