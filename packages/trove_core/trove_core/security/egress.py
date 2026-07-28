from __future__ import annotations

import hashlib
import re
from typing import Any


_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_MAX_CONTROL_TEXT_BYTES = 16 * 1024


def _exact_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    """Reject coercions at the egress-control boundary.

    Approval payloads are capabilities, not display data.  Accepting values via
    ``str(value)`` (or accepting ``str`` subclasses) lets an object bind one
    payload while presenting a different value to a downstream provider.
    """

    if type(value) is not str:
        raise TypeError(f'{field} must be an exact string')
    if not allow_empty and not value:
        raise ValueError(f'{field} must not be empty')
    if len(value.encode('utf-8')) > _MAX_CONTROL_TEXT_BYTES:
        raise ValueError(f'{field} exceeds the egress control size limit')
    return value


def _exact_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f'{field} must be an exact integer')
    if value < 0:
        raise ValueError(f'{field} must be non-negative')
    return value


def _digest(value: object, *, field: str) -> str:
    value = _exact_text(value, field=field)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def cloud_asr_payload(
    *,
    citation: str,
    provider: str,
    model: str,
    resource_id: str,
    endpoint: str,
) -> dict[str, Any]:
    return {
        'citation_hash': _digest(citation, field='citation'),
        'provider': _exact_text(provider, field='provider'),
        'model': _exact_text(model, field='model'),
        'resource_id': _exact_text(resource_id, field='resource_id'),
        'endpoint_hash': _digest(endpoint, field='endpoint'),
    }


def cloud_vision_payload(
    *,
    citation: str,
    provider: str,
    model: str,
    endpoint: str,
) -> dict[str, Any]:
    return {
        'citation_hash': _digest(citation, field='citation'),
        'provider': _exact_text(provider, field='provider'),
        'model': _exact_text(model, field='model'),
        'endpoint_hash': _digest(endpoint, field='endpoint'),
    }


def cloud_embedding_payload(
    *,
    operation: str,
    provider: str,
    model: str,
    dimensions: int,
    endpoint: str,
    input_digest: str,
    item_count: int,
) -> dict[str, Any]:
    digest = _exact_text(input_digest, field='input_digest')
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError('input_digest must be a lowercase SHA-256 digest')
    return {
        'operation': _exact_text(operation, field='operation'),
        'provider': _exact_text(provider, field='provider'),
        'model': _exact_text(model, field='model'),
        'dimensions': _exact_nonnegative_int(dimensions, field='dimensions'),
        'endpoint_hash': _digest(endpoint, field='endpoint'),
        'input_digest': digest,
        'item_count': _exact_nonnegative_int(item_count, field='item_count'),
    }


def cloud_rerank_payload(
    *,
    query: str,
    documents: list[str],
    top_n: int,
    provider: str,
    model: str,
    endpoint: str,
) -> dict[str, Any]:
    if type(documents) is not list:
        raise TypeError('documents must be an exact list')
    if type(top_n) is not int or top_n < 1 or top_n > len(documents):
        raise ValueError('top_n must be a positive exact integer within the document count')
    return {
        'query_digest': _digest(query, field='query'),
        'input_digest': content_set_digest(documents),
        'item_count': len(documents),
        'top_n': top_n,
        'provider': _exact_text(provider, field='provider'),
        'model': _exact_text(model, field='model'),
        'endpoint_hash': _digest(endpoint, field='endpoint'),
    }


def content_set_digest(values: list[str]) -> str:
    if type(values) is not list:
        raise TypeError('values must be an exact list')
    hasher = hashlib.sha256()
    for index, value in enumerate(values):
        value = _exact_text(value, field=f'values[{index}]', allow_empty=True)
        encoded = value.encode('utf-8')
        hasher.update(len(encoded).to_bytes(8, 'big'))
        hasher.update(encoded)
    return hasher.hexdigest()
