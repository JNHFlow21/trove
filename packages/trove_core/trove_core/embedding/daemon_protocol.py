from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .model_registry import MODEL_REGISTRY, read_local_model_manifest, resolve_model_spec

PROTOCOL_VERSION = 2
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TEXTS_PER_REQUEST = 64
MAX_TEXT_CHARACTERS = 256 * 1024
MAX_DIMENSIONS = 65_536
DAEMON_ERROR_CODES = {
    'daemon_dimensions_invalid',
    'daemon_dimensions_mismatch',
    'daemon_identity_invalid',
    'daemon_identity_mismatch',
    'daemon_identity_missing',
    'daemon_internal_error',
    'daemon_operation_invalid',
    'daemon_protocol_error',
    'daemon_protocol_invalid',
    'daemon_protocol_mismatch',
    'daemon_provider_failed',
    'daemon_queue_saturated',
    'daemon_request_timeout',
    'daemon_request_too_large',
    'daemon_response_too_large',
    'daemon_stopped',
    'daemon_text_batch_invalid',
    'daemon_text_batch_too_large',
    'daemon_transport_error',
    'daemon_unavailable',
    'daemon_vector_count_mismatch',
    'daemon_vector_invalid',
    'daemon_vector_missing',
}
_IDENTITY_LABEL_RE = re.compile(r'^[A-Za-z0-9._:/-]{1,256}$')
_HASH_RE = re.compile(r'^[a-f0-9]{64}$')


class DaemonProtocolError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DaemonIdentityMismatch(DaemonProtocolError):
    def __init__(self) -> None:
        super().__init__('daemon_identity_mismatch')


class DaemonQueueSaturated(DaemonProtocolError):
    def __init__(self) -> None:
        super().__init__('daemon_queue_saturated')


class DaemonRequestTimeout(DaemonProtocolError):
    def __init__(self) -> None:
        super().__init__('daemon_request_timeout')


@dataclass(frozen=True)
class DaemonIdentity:
    provider: str
    model_id: str
    model_hash: str
    dimensions: int
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Any) -> 'DaemonIdentity':
        if type(payload) is not dict:
            raise DaemonProtocolError('daemon_identity_missing')
        try:
            provider = payload['provider']
            model_id = payload['model_id']
            model_hash = payload['model_hash']
            dimensions = payload['dimensions']
            protocol_version = payload['protocol_version']
        except KeyError as exc:
            raise DaemonProtocolError('daemon_identity_missing') from exc
        if (
            any(type(value) is not str for value in (provider, model_id, model_hash))
            or not _IDENTITY_LABEL_RE.fullmatch(provider)
            or not _IDENTITY_LABEL_RE.fullmatch(model_id)
            or not _HASH_RE.fullmatch(model_hash)
            or provider.startswith(('/', '~'))
            or model_id.startswith(('/', '~'))
            or '..' in provider
            or '..' in model_id
        ):
            raise DaemonProtocolError('daemon_identity_invalid')
        if type(dimensions) is not int or dimensions < 0 or dimensions > MAX_DIMENSIONS:
            raise DaemonProtocolError('daemon_dimensions_invalid')
        if type(protocol_version) is not int:
            raise DaemonProtocolError('daemon_protocol_invalid')
        return cls(provider, model_id, model_hash, dimensions, protocol_version)

    def matches(self, other: 'DaemonIdentity') -> bool:
        if self.protocol_version != other.protocol_version:
            return False
        if self.provider != other.provider or self.model_id != other.model_id or self.model_hash != other.model_hash:
            return False
        # A zero dimension means a custom model whose dimension is not known
        # until its first local load. Known dimensions must match strictly.
        return not (self.dimensions and other.dimensions and self.dimensions != other.dimensions)

    def require_match(self, other: 'DaemonIdentity') -> None:
        if not self.matches(other):
            raise DaemonIdentityMismatch()


def _safe_identity_label(value: object, *, prefix: str) -> str:
    text = value if type(value) is str else ''
    if (
        _IDENTITY_LABEL_RE.fullmatch(text)
        and not text.startswith(('/', '~'))
        and '..' not in text
    ):
        return text
    digest = hashlib.sha256(str(value).encode('utf-8', errors='replace')).hexdigest()[:16]
    return f'{prefix}-{digest}'


def _model_id_for_path(path: Path, requested: str | None, manifest: dict[str, Any]) -> str:
    if requested:
        return _safe_identity_label(requested, prefix='custom-local')
    for key in ('model_id', 'repo_id', 'name'):
        value = manifest.get(key)
        if type(value) is str and value:
            return _safe_identity_label(value, prefix='custom-local')
    for spec in MODEL_REGISTRY.values():
        if path.name == spec.safe_name:
            return spec.model_id
    return 'custom-local-' + hashlib.sha256(path.name.encode('utf-8', errors='replace')).hexdigest()[:16]


def _model_hash(path: Path, *, provider: str, model_id: str, manifest: dict[str, Any]) -> str:
    """Hash bounded model identity metadata without returning a private path."""

    digest = hashlib.sha256()
    digest.update(json.dumps(
        {'provider': provider, 'model_id': model_id, 'manifest': manifest},
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    ).encode('utf-8'))
    entries = 0
    if path.exists():
        for root, directories, files in os.walk(path, followlinks=False):
            directories[:] = sorted(directories)
            for filename in sorted(files):
                if entries >= 4096:
                    directories[:] = []
                    break
                child = Path(root) / filename
                try:
                    stat = child.stat()
                    relative = child.relative_to(path).as_posix()
                except OSError:
                    continue
                digest.update(relative.encode('utf-8', errors='replace'))
                digest.update(str(stat.st_size).encode('ascii'))
                digest.update(str(stat.st_mtime_ns).encode('ascii'))
                if relative in {'config.json', 'modules.json', 'trove_model_manifest.json'} and stat.st_size <= 1024 * 1024:
                    try:
                        digest.update(hashlib.sha256(child.read_bytes()).digest())
                    except OSError:
                        pass
                entries += 1
            if entries >= 4096:
                break
    return digest.hexdigest()


def identity_for_model(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    dimensions: int | None = None,
) -> DaemonIdentity:
    path = Path(model_path).expanduser()
    manifest = read_local_model_manifest(path)
    resolved_id = _model_id_for_path(path, model_id, manifest)
    spec = resolve_model_spec(resolved_id)
    provider = _safe_identity_label(
        manifest.get('provider') or spec.provider or 'sentence-transformers',
        prefix='provider',
    )
    manifest_dimensions = (
        manifest.get('dimensions')
        or manifest.get('embedding_dimensions')
        or manifest.get('expected_dimensions')
    )
    try:
        resolved_dimensions = int(dimensions if dimensions is not None else manifest_dimensions or spec.dimensions or 0)
    except (TypeError, ValueError):
        resolved_dimensions = 0
    return DaemonIdentity(
        provider=provider,
        model_id=resolved_id,
        model_hash=_model_hash(path, provider=provider, model_id=resolved_id, manifest=manifest),
        dimensions=max(0, min(resolved_dimensions, MAX_DIMENSIONS)),
    )


def validate_texts(texts: Any) -> list[str]:
    if type(texts) is not list or not texts or len(texts) > MAX_TEXTS_PER_REQUEST:
        raise DaemonProtocolError('daemon_text_batch_invalid')
    if any(type(text) is not str for text in texts):
        raise DaemonProtocolError('daemon_text_batch_invalid')
    if sum(len(text) for text in texts) > MAX_TEXT_CHARACTERS:
        raise DaemonProtocolError('daemon_text_batch_too_large')
    return texts


def safe_error_code(code: object) -> str:
    return code if type(code) is str and code in DAEMON_ERROR_CODES else 'daemon_internal_error'


def error_payload(code: str) -> dict[str, Any]:
    return {
        'ok': False,
        'error': {'type': 'daemon_error', 'code': safe_error_code(code)},
        'raw_content_included': False,
        'raw_paths_included': False,
        'secret_values_included': False,
    }
