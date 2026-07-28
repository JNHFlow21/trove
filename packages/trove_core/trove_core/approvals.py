from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
import threading
import uuid
import weakref

from trove_core.vault.tracing import TraceStorageError, TraceTimeline


DANGEROUS_ACTIONS = {
    'full_import',
    'destructive_rebuild',
    'vector_purge_rebuild',
    'delete_or_purge',
    'cloud_asr_upload',
    'cloud_vision_upload',
    'cloud_embedding_upload',
    'cloud_rerank_upload',
    'real_media_processing',
    'local-file-export',
    'agent_sensitive_tool',
    'remote_media_fetch',
}

# This is the application-layer inventory. Adapters must call a command boundary
# that consumes one of these approval classes instead of calling destructive
# repositories or runtime helpers directly.
SENSITIVE_CAPABILITY_INVENTORY = {
    'full_import': 'full_import',
    'reset_index_cache': 'destructive_rebuild',
    'scope_rebuild': 'destructive_rebuild',
    'vector_purge_rebuild': 'vector_purge_rebuild',
    'vector_rebuild': 'vector_purge_rebuild',
    'content_kind_backfill': 'delete_or_purge',
    'appmsg_backfill': 'delete_or_purge',
    'message_media_backfill': 'delete_or_purge',
    'wechat_cdn_fetch': 'remote_media_fetch',
    'media_understanding_invalidate': 'delete_or_purge',
    'recover_writer_marker': 'delete_or_purge',
    'voice_cloud_asr': 'cloud_asr_upload',
    'image_cloud_vision': 'cloud_vision_upload',
    'cloud_embedding_probe': 'cloud_embedding_upload',
    'cloud_vector_index': 'cloud_embedding_upload',
    'cloud_rerank': 'cloud_rerank_upload',
    'real_media_processing': 'real_media_processing',
    'files_archive': 'local-file-export',
    'observe_approve': 'agent_sensitive_tool',
    'observe_retire': 'agent_sensitive_tool',
    'entity_reconcile': 'destructive_rebuild',
    'derived_data_purge': 'delete_or_purge',
}

_APPROVAL_ID_RE = re.compile(r'^appr-[0-9a-f]{16}$')
_GRANT_SEAL = object()
_MAX_APPROVAL_PAYLOAD_DEPTH = 16
_MAX_APPROVAL_PAYLOAD_NODES = 2000
_MAX_APPROVAL_PAYLOAD_BYTES = 64 * 1024
_PATH_KEY_PARTS = {
    'path', 'paths', 'root', 'dir', 'directory', 'dest', 'destination',
    'source', 'sources', 'file', 'files', 'key_store', 'live_root',
}
_SECRET_KEY_PARTS = {
    'token', 'authorization', 'secret', 'password', 'credential', 'raw_text', 'content',
    'contact', 'conversation', 'asset', 'citation', 'observation', 'entity', 'account',
}
_SAFE_APPROVAL_FIELD_NAMES = {
    'action', 'backend', 'batch_size', 'case_count', 'command', 'count',
    'dimensions', 'endpoint_hash', 'generation', 'input_digest', 'k',
    'limit_per_sqlite', 'max_cases', 'max_messages', 'mode', 'model',
    'per_route', 'process_config_hash', 'provider', 'purge',
    'random_negatives', 'request_format', 'reset_index_cache', 'route',
    'sample_seed', 'scope', 'source_set_hash', 'sources_count',
}
_SAFE_APPROVAL_COUNT_FIELDS = {
    'batch_size', 'case_count', 'count', 'dimensions', 'generation', 'k',
    'limit_per_sqlite', 'max_cases', 'max_messages', 'per_route',
    'random_negatives', 'sample_seed', 'sources_count',
}
_SAFE_APPROVAL_BOOL_FIELDS = {'purge', 'reset_index_cache'}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _parse_iso(value: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError('timestamp is missing')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timestamp must include a timezone')
    return parsed.astimezone(timezone.utc)


def _canonical_value(value: Any, *, _depth: int = 0, _budget: list[int] | None = None) -> Any:
    budget = _budget if _budget is not None else [_MAX_APPROVAL_PAYLOAD_NODES]
    budget[0] -= 1
    if budget[0] < 0 or _depth > _MAX_APPROVAL_PAYLOAD_DEPTH:
        raise ValueError('approval payload exceeds structural bounds')
    if isinstance(value, Path):
        return str(value)
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError('approval payload keys must be strings')
        return {key: _canonical_value(item, _depth=_depth + 1, _budget=budget) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_canonical_value(item, _depth=_depth + 1, _budget=budget) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f'approval payload contains unsupported value type: {type(value).__name__}')


def canonical_payload_digest(payload: dict[str, Any] | None) -> str:
    if payload is not None and type(payload) is not dict:
        raise TypeError('approval payload must be an object')
    canonical = _canonical_value({} if payload is None else payload)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    if len(encoded) > _MAX_APPROVAL_PAYLOAD_BYTES:
        raise ValueError('approval payload exceeds encoded size bound')
    return hashlib.sha256(encoded).hexdigest()


def _redacted_marker(value: Any) -> str:
    digest = canonical_payload_digest({'value': _canonical_value(value)})[:12]
    return f'redacted:{digest}'


def _key_matches(key: str, parts: set[str]) -> bool:
    normalized = key.lower().replace('-', '_')
    tokens = set(normalized.split('_'))
    return normalized in parts or bool(tokens & parts)


def redact_approval_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return a useful summary without persisting command paths or raw data."""

    def visit(value: Any, *, key: str = '') -> Any:
        if _key_matches(key, _SECRET_KEY_PARTS):
            return _redacted_marker(value)
        if _key_matches(key, _PATH_KEY_PARTS):
            if isinstance(value, (list, tuple)):
                return [_redacted_marker(item) for item in value[:50]]
            return _redacted_marker(value)
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for raw_key, item in value.items():
                field = str(raw_key)
                persisted_key = field if field in _SAFE_APPROVAL_FIELD_NAMES else f'field_{hashlib.sha256(field.encode("utf-8")).hexdigest()[:12]}'
                redacted[persisted_key] = visit(item, key=field)
            return redacted
        if isinstance(value, (list, tuple)):
            return [visit(item, key=key) for item in value[:50]]
        if isinstance(value, (str, Path)):
            return _redacted_marker(value)
        if type(value) is bool and key in _SAFE_APPROVAL_BOOL_FIELDS:
            return value
        if type(value) is int and key in _SAFE_APPROVAL_COUNT_FIELDS and 0 <= value <= 1_000_000_000:
            return value
        if value is None:
            return None
        return _redacted_marker(_canonical_value(value))

    return visit(payload or {})


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    action: str
    danger_class: str
    status: str
    requested_at: str
    expires_at: str
    decided_at: str | None
    request_hash: str
    payload: dict[str, Any]
    decision_note: str | None = None
    consumed_at: str | None = None
    consumption_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ApprovalClaimState:
    """Authority retained by the issuing module, never stored on the grant."""

    __slots__ = ('lock', 'claimed', 'fingerprint')

    def __init__(self, fingerprint: str) -> None:
        self.lock = threading.Lock()
        self.claimed = False
        self.fingerprint = fingerprint


_GRANT_REGISTRY: dict[int, tuple[weakref.ReferenceType[Any], _ApprovalClaimState]] = {}
_GRANT_REGISTRY_LOCK = threading.RLock()


def _grant_registry_after_fork() -> None:
    """Invalidate inherited grants without touching possibly locked mutexes."""

    global _GRANT_REGISTRY, _GRANT_REGISTRY_LOCK
    _GRANT_REGISTRY = {}
    _GRANT_REGISTRY_LOCK = threading.RLock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_grant_registry_after_fork)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ApprovalGrant:
    approval_id: str
    action: str
    danger_class: str
    request_hash: str
    expires_at: str
    consumed_at: str
    consumption_id: str
    vault_hash: str
    _seal: object = field(repr=False, compare=False)
    _issued_pid: int = field(repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError('ApprovalGrant cannot be subclassed')

    def __post_init__(self) -> None:
        values = (
            self.approval_id,
            self.action,
            self.danger_class,
            self.request_hash,
            self.expires_at,
            self.consumed_at,
            self.consumption_id,
            self.vault_hash,
        )
        if (
            self._seal is not _GRANT_SEAL
            or any(type(value) is not str for value in values)
            or type(self._issued_pid) is not int
        ):
            raise ValueError('ApprovalGrant can only be issued by ApprovalManager')

    def to_dict(self) -> dict[str, Any]:
        return {
            'approval_id': self.approval_id,
            'approval_status': 'consumed',
            'action': self.action,
            'danger_class': self.danger_class,
            'request_hash': self.request_hash,
            'expires_at': self.expires_at,
            'consumed_at': self.consumed_at,
            'consumption_id': self.consumption_id,
        }

    def __copy__(self) -> 'ApprovalGrant':
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> 'ApprovalGrant':
        memo[id(self)] = self
        return self

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def validate_for(
        self,
        vault_root: str | Path,
        *,
        action: str,
        danger_class: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if type(self) is not ApprovalGrant:
            raise ApprovalValidationError('approval grant type is invalid', code='invalid_grant')
        state = _issued_grant_state(self)
        with state.lock:
            self._validate_for_locked(
                state,
                vault_root,
                action=action,
                danger_class=danger_class,
                payload=payload,
            )

    def claim_for(
        self,
        vault_root: str | Path,
        *,
        action: str,
        danger_class: str,
        payload: dict[str, Any] | None = None,
    ) -> 'ApprovalGrant':
        """Validate and atomically consume this capability at a core boundary."""

        if type(self) is not ApprovalGrant:
            raise ApprovalValidationError('approval grant type is invalid', code='invalid_grant')
        state = _issued_grant_state(self)
        with state.lock:
            if state.claimed:
                raise ApprovalValidationError(
                    'approval grant was already claimed by a command',
                    code='approval_grant_replayed',
                )
            self._validate_for_locked(
                state,
                vault_root,
                action=action,
                danger_class=danger_class,
                payload=payload,
            )
            state.claimed = True
        return self

    def _validate_for_locked(
        self,
        state: _ApprovalClaimState,
        vault_root: str | Path,
        *,
        action: str,
        danger_class: str,
        payload: dict[str, Any] | None,
    ) -> None:
        if self._seal is not _GRANT_SEAL or type(self._issued_pid) is not int or self._issued_pid != os.getpid():
            raise ApprovalValidationError('approval grant belongs to another process', code='cross_process_grant')
        if not _constant_text_equal(_grant_fingerprint(self), state.fingerprint):
            raise ApprovalValidationError('approval grant fields changed after issue', code='invalid_grant')
        if type(action) is not str or type(danger_class) is not str:
            raise ApprovalValidationError('approval command fields are malformed', code='grant_mismatch')
        try:
            expires_at = _parse_iso(self.expires_at)
        except (TypeError, ValueError) as exc:
            raise ApprovalValidationError('approval grant expiry is malformed', code='invalid_grant') from exc
        if datetime.now(timezone.utc) >= expires_at:
            raise ApprovalValidationError('approval grant expired before command execution', code='approval_grant_expired')
        expected_vault = hashlib.sha256(str(Path(vault_root).expanduser().resolve()).encode('utf-8')).hexdigest()
        if not _constant_text_equal(self.vault_hash, expected_vault):
            raise ApprovalValidationError('approval grant belongs to another vault', code='cross_vault_grant')
        if not _constant_text_equal(self.action, action) or not _constant_text_equal(self.danger_class, danger_class):
            raise ApprovalValidationError('approval grant does not match command', code='grant_mismatch')
        if payload is not None and type(payload) is not dict:
            raise ApprovalValidationError('approval command payload is malformed', code='grant_payload_mismatch')
        if not _constant_text_equal(
            self.request_hash,
            _validated_payload_digest(payload, code='grant_payload_mismatch'),
        ):
            raise ApprovalValidationError('approval grant does not match payload', code='grant_payload_mismatch')


class ApprovalValidationError(RuntimeError):
    def __init__(self, message: str, *, code: str = 'invalid_approval'):
        super().__init__(message)
        self.code = code


def _constant_text_equal(left: str, right: str) -> bool:
    return type(left) is str and type(right) is str and hmac.compare_digest(left, right)


def _grant_fingerprint(grant: ApprovalGrant) -> str:
    if type(grant) is not ApprovalGrant or type(object.__getattribute__(grant, '_issued_pid')) is not int:
        raise ApprovalValidationError('approval grant type is invalid', code='invalid_grant')
    values = (
        object.__getattribute__(grant, 'approval_id'),
        object.__getattribute__(grant, 'action'),
        object.__getattribute__(grant, 'danger_class'),
        object.__getattribute__(grant, 'request_hash'),
        object.__getattribute__(grant, 'expires_at'),
        object.__getattribute__(grant, 'consumed_at'),
        object.__getattribute__(grant, 'consumption_id'),
        object.__getattribute__(grant, 'vault_hash'),
        str(object.__getattribute__(grant, '_issued_pid')),
    )
    if any(type(value) is not str for value in values):
        raise ApprovalValidationError('approval grant fields are malformed', code='invalid_grant')
    encoded = json.dumps(values, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def _validated_payload_digest(payload: dict[str, Any] | None, *, code: str) -> str:
    try:
        return canonical_payload_digest(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ApprovalValidationError('approval payload is malformed', code=code) from exc


def _issued_grant_state(grant: ApprovalGrant) -> _ApprovalClaimState:
    if type(grant) is not ApprovalGrant or grant._seal is not _GRANT_SEAL:
        raise ApprovalValidationError('approval grant is not authentic', code='invalid_grant')
    if type(grant._issued_pid) is not int or grant._issued_pid != os.getpid():
        raise ApprovalValidationError('approval grant belongs to another process', code='cross_process_grant')
    with _GRANT_REGISTRY_LOCK:
        entry = _GRANT_REGISTRY.get(id(grant))
        if entry is None or entry[0]() is not grant:
            raise ApprovalValidationError('approval grant was not issued in this process', code='invalid_grant')
        return entry[1]


def _register_issued_grant(grant: ApprovalGrant) -> ApprovalGrant:
    if type(grant._issued_pid) is not int or grant._issued_pid != os.getpid():
        raise ApprovalValidationError('approval grant process identity is invalid', code='invalid_grant')
    key = id(grant)
    state = _ApprovalClaimState(_grant_fingerprint(grant))

    def cleanup(reference: weakref.ReferenceType[Any]) -> None:
        with _GRANT_REGISTRY_LOCK:
            current = _GRANT_REGISTRY.get(key)
            if current is not None and current[0] is reference:
                _GRANT_REGISTRY.pop(key, None)

    reference = weakref.ref(grant, cleanup)
    with _GRANT_REGISTRY_LOCK:
        current = _GRANT_REGISTRY.get(key)
        if current is not None and current[0]() is not None:
            raise ApprovalValidationError('approval grant registry collision', code='invalid_grant')
        _GRANT_REGISTRY[key] = (reference, state)
    return grant


def claim_approval_grant(
    grant: ApprovalGrant,
    vault_root: str | Path,
    *,
    action: str,
    danger_class: str,
    payload: dict[str, Any] | None,
) -> ApprovalGrant:
    """Non-overridable application-boundary entry point for a grant claim."""

    if type(grant) is not ApprovalGrant:
        raise ApprovalValidationError('approval grant type is invalid', code='invalid_grant')
    return ApprovalGrant.claim_for(
        grant,
        vault_root,
        action=action,
        danger_class=danger_class,
        payload=payload,
    )


def require_claimed_approval_grant(
    grant: ApprovalGrant,
    vault_root: str | Path,
    *,
    action: str,
    danger_class: str,
    payload: dict[str, Any] | None,
) -> ApprovalGrant:
    """Verify a downstream sink received the exact capability already claimed upstream."""

    if type(grant) is not ApprovalGrant:
        raise ApprovalValidationError('approval grant type is invalid', code='invalid_grant')
    state = _issued_grant_state(grant)
    with state.lock:
        if not state.claimed:
            raise ApprovalValidationError('approval grant was not claimed by an application command', code='approval_grant_unclaimed')
        ApprovalGrant._validate_for_locked(
            grant,
            state,
            vault_root,
            action=action,
            danger_class=danger_class,
            payload=payload,
        )
    return grant


class ApprovalRequired(ApprovalValidationError):
    def __init__(self, record: ApprovalRecord, *, code: str = 'approval_required'):
        super().__init__('approval required', code=code)
        self.record = record


def _file_identity(result: os.stat_result) -> tuple[int, int]:
    return int(result.st_dev), int(result.st_ino)


@dataclass(frozen=True)
class _ApprovalStorageLease:
    manager: 'ApprovalManager'
    dir_fd: int
    lock_fd: int
    directory_identity: tuple[int, int]
    lock_identity: tuple[int, int]

    def validate(self) -> None:
        try:
            bound_dir = os.fstat(self.dir_fd)
            current_dir = os.lstat(self.manager.dir)
            bound_lock = os.fstat(self.lock_fd)
            current_lock = os.stat(
                self.manager._AUTHORITY_FILE,
                dir_fd=self.dir_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ApprovalValidationError(
                'approval storage identity changed during operation',
                code='approval_storage_identity_changed',
            ) from exc
        if (
            not stat.S_ISDIR(bound_dir.st_mode)
            or _file_identity(bound_dir) != self.directory_identity
            or _file_identity(current_dir) != self.directory_identity
            or not stat.S_ISREG(bound_lock.st_mode)
            or bound_lock.st_nlink != 1
            or _file_identity(bound_lock) != self.lock_identity
            or _file_identity(current_lock) != self.lock_identity
        ):
            raise ApprovalValidationError(
                'approval storage identity changed during operation',
                code='approval_storage_identity_changed',
            )


class ApprovalManager:
    _AUTHORITY_FILE = '.approval-authority.lock'

    def __init__(self, vault_root: str | Path):
        self.vault_root = Path(vault_root).expanduser().resolve()
        self.dir = self.vault_root / 'approvals'
        self.trace = TraceTimeline(self.vault_root)

    def _append_trace(self, status: str, payload: dict[str, Any]) -> bool:
        """Best-effort projection of the durable approval record/claim audit.

        Approval records and no-replace consumption claims are authoritative.
        A trace storage failure must not consume an approval without returning
        the already minted capability to the caller.
        """

        try:
            self.trace.append('approval', status, payload)
            return True
        except TraceStorageError:
            return False

    @property
    def _vault_hash(self) -> str:
        return hashlib.sha256(str(self.vault_root.resolve()).encode('utf-8')).hexdigest()

    def _open_dir(self) -> int:
        dir_fd: int | None = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            listed = os.lstat(self.dir)
            if not stat.S_ISDIR(listed.st_mode) or stat.S_ISLNK(listed.st_mode):
                raise OSError('approval directory is not a real directory')
            flags = os.O_RDONLY
            if hasattr(os, 'O_DIRECTORY'):
                flags |= os.O_DIRECTORY
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            dir_fd = os.open(self.dir, flags)
            bound = os.fstat(dir_fd)
            if not stat.S_ISDIR(bound.st_mode) or _file_identity(bound) != _file_identity(listed):
                raise OSError('approval directory identity changed')
            os.fchmod(dir_fd, 0o700)
            return dir_fd
        except OSError as exc:
            if dir_fd is not None:
                try:
                    os.close(dir_fd)
                except OSError:
                    pass
            raise ApprovalValidationError(
                'approval storage permissions could not be secured',
                code='approval_storage_permissions',
            ) from exc

    def _ensure_dir(self) -> None:
        dir_fd = self._open_dir()
        os.close(dir_fd)

    def _path(self, approval_id: str) -> Path:
        if type(approval_id) is not str or not _APPROVAL_ID_RE.fullmatch(approval_id):
            raise ApprovalValidationError('malformed approval id', code='malformed_approval_id')
        return self.dir / f'{approval_id}.redacted.json'

    def _lock_path(self, approval_id: str) -> Path:
        self._path(approval_id)
        return self.dir / self._AUTHORITY_FILE

    def _consumption_claim_name(self, approval_id: str) -> str:
        self._path(approval_id)
        return f'.{approval_id}.consumed.json'

    def _normalize_claim_link_count(
        self,
        lease: _ApprovalStorageLease,
        approval_id: str,
        claim_identity: tuple[int, int],
        link_count: int,
    ) -> None:
        if link_count == 1:
            return
        if link_count != 2:
            raise ApprovalValidationError('approval consumption claim is unsafe', code='approval_consumption_state_invalid')
        prefix = f'.{approval_id}.consumption.'
        matches: list[str] = []
        try:
            for name in os.listdir(lease.dir_fd):
                if not name.startswith(prefix) or not name.endswith('.tmp'):
                    continue
                listed = os.stat(name, dir_fd=lease.dir_fd, follow_symlinks=False)
                if _file_identity(listed) == claim_identity:
                    matches.append(name)
        except OSError as exc:
            raise ApprovalValidationError('approval consumption claim is unsafe', code='approval_consumption_state_invalid') from exc
        if len(matches) != 1:
            raise ApprovalValidationError('approval consumption claim is unsafe', code='approval_consumption_state_invalid')
        os.unlink(matches[0], dir_fd=lease.dir_fd)
        try:
            os.fsync(lease.dir_fd)
        except OSError:
            pass

    def _read_consumption_claim(
        self,
        lease: _ApprovalStorageLease,
        approval_id: str,
    ) -> dict[str, Any] | None:
        lease.validate()
        name = self._consumption_claim_name(approval_id)
        flags = os.O_RDONLY
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        fd: int | None = None
        try:
            try:
                fd = os.open(name, flags, dir_fd=lease.dir_fd)
            except FileNotFoundError:
                return None
            opened = os.fstat(fd)
            listed = os.stat(name, dir_fd=lease.dir_fd, follow_symlinks=False)
            identity = _file_identity(opened)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_mode & 0o077
                or opened.st_size > 16 * 1024
                or identity != _file_identity(listed)
            ):
                raise ApprovalValidationError('approval consumption claim is unsafe', code='approval_consumption_state_invalid')
            self._normalize_claim_link_count(lease, approval_id, identity, int(opened.st_nlink))
            opened = os.fstat(fd)
            listed = os.stat(name, dir_fd=lease.dir_fd, follow_symlinks=False)
            if opened.st_nlink != 1 or identity != _file_identity(listed):
                raise ApprovalValidationError('approval consumption claim is unsafe', code='approval_consumption_state_invalid')
            raw = os.read(fd, 16 * 1024 + 1)
            data = json.loads(raw.decode('utf-8'))
            lease.validate()
        except ApprovalValidationError:
            raise
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ApprovalValidationError('approval consumption claim is malformed', code='approval_consumption_state_invalid') from exc
        finally:
            if fd is not None:
                os.close(fd)
        required = {
            'schema', 'approval_id', 'action', 'danger_class', 'request_hash',
            'consumed_at', 'consumption_id',
        }
        if type(data) is not dict or set(data) != required:
            raise ApprovalValidationError('approval consumption claim is malformed', code='approval_consumption_state_invalid')
        if (
            type(data['schema']) is not int
            or data['schema'] != 1
            or type(data['approval_id']) is not str
            or data['approval_id'] != approval_id
            or type(data['action']) is not str
            or type(data['danger_class']) is not str
            or type(data['request_hash']) is not str
            or not re.fullmatch(r'[0-9a-f]{64}', data['request_hash'])
            or type(data['consumed_at']) is not str
            or type(data['consumption_id']) is not str
            or not re.fullmatch(r'consume-[0-9a-f]{16}', data['consumption_id'])
        ):
            raise ApprovalValidationError('approval consumption claim is malformed', code='approval_consumption_state_invalid')
        try:
            _parse_iso(data['consumed_at'])
        except (TypeError, ValueError) as exc:
            raise ApprovalValidationError('approval consumption claim is malformed', code='approval_consumption_state_invalid') from exc
        return data

    def _publish_consumption_claim(
        self,
        lease: _ApprovalStorageLease,
        claim: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        approval_id = str(claim['approval_id'])
        name = self._consumption_claim_name(approval_id)
        tmp_name = f'.{approval_id}.consumption.{claim["consumption_id"]}.{uuid.uuid4().hex}.tmp'
        encoded = (json.dumps(claim, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
        fd: int | None = None
        linked = False
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            fd = os.open(tmp_name, flags, 0o600, dir_fd=lease.dir_fd)
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ApprovalValidationError('approval claim temp file is unsafe', code='approval_storage_identity_changed')
            os.fchmod(fd, 0o600)
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError('short approval claim write')
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = None
            lease.validate()
            try:
                os.link(
                    tmp_name,
                    name,
                    src_dir_fd=lease.dir_fd,
                    dst_dir_fd=lease.dir_fd,
                    follow_symlinks=False,
                )
                linked = True
                try:
                    os.fsync(lease.dir_fd)
                except OSError:
                    pass
            except FileExistsError:
                linked = False
        except ApprovalValidationError:
            raise
        except OSError as exc:
            raise ApprovalValidationError('approval consumption claim could not be published', code='approval_storage_identity_changed') from exc
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(tmp_name, dir_fd=lease.dir_fd)
            except FileNotFoundError:
                pass
        persisted = self._read_consumption_claim(lease, approval_id)
        if persisted is None:
            raise ApprovalValidationError('approval consumption claim disappeared', code='approval_consumption_state_invalid')
        return linked, persisted

    @contextmanager
    def _locked(self, approval_id: str) -> Iterator[_ApprovalStorageLease]:
        self._path(approval_id)
        dir_fd = self._open_dir()
        lock_fd: int | None = None
        try:
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            lock_fd = os.open(self._AUTHORITY_FILE, flags, 0o600, dir_fd=dir_fd)
            opened = os.fstat(lock_fd)
            listed = os.stat(self._AUTHORITY_FILE, dir_fd=dir_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _file_identity(opened) != _file_identity(listed)
            ):
                raise ApprovalValidationError(
                    'approval authority is not a private regular file',
                    code='approval_storage_identity_changed',
                )
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            lease = _ApprovalStorageLease(
                manager=self,
                dir_fd=dir_fd,
                lock_fd=lock_fd,
                directory_identity=_file_identity(os.fstat(dir_fd)),
                lock_identity=_file_identity(os.fstat(lock_fd)),
            )
            lease.validate()
            yield lease
            lease.validate()
        except ApprovalValidationError:
            raise
        except OSError as exc:
            raise ApprovalValidationError(
                'approval authority could not be secured',
                code='approval_storage_identity_changed',
            ) from exc
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(dir_fd)

    def _write(self, record: ApprovalRecord, lease: _ApprovalStorageLease | None = None) -> None:
        if lease is None:
            with self._locked(record.approval_id) as acquired:
                self._write(record, acquired)
            return
        self._path(record.approval_id)
        lease.validate()
        tmp_name = f'.{record.approval_id}.{uuid.uuid4().hex}.tmp'
        fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            fd = os.open(tmp_name, flags, 0o600, dir_fd=lease.dir_fd)
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise ApprovalValidationError('approval record temp file is unsafe', code='approval_storage_identity_changed')
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                fd = None
                json.dump(record.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            lease.validate()
            os.replace(
                tmp_name,
                f'{record.approval_id}.redacted.json',
                src_dir_fd=lease.dir_fd,
                dst_dir_fd=lease.dir_fd,
            )
            try:
                os.fsync(lease.dir_fd)
            except OSError:
                pass
            lease.validate()
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(tmp_name, dir_fd=lease.dir_fd)
            except FileNotFoundError:
                pass

    def _load_unlocked(self, approval_id: str, lease: _ApprovalStorageLease) -> ApprovalRecord:
        self._path(approval_id)
        lease.validate()
        fd: int | None = None
        try:
            flags = os.O_RDONLY
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            fd = os.open(f'{approval_id}.redacted.json', flags, dir_fd=lease.dir_fd)
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size > 1024 * 1024
                or opened.st_mode & 0o077
            ):
                raise OSError('approval record is not a bounded regular file')
            chunks: list[bytes] = []
            remaining = 1024 * 1024 + 1
            while remaining > 0:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = json.loads(b''.join(chunks).decode('utf-8'))
            lease.validate()
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise ApprovalValidationError('approval record is missing or malformed', code='malformed_approval') from exc
        finally:
            if fd is not None:
                os.close(fd)
        if type(data) is not dict:
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval')
        data.setdefault('decision_note', None)
        data.setdefault('consumed_at', None)
        data.setdefault('consumption_id', None)
        try:
            record = ApprovalRecord(**data)
        except (TypeError, ValueError) as exc:
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval') from exc
        if (
            type(record.approval_id) is not str
            or record.approval_id != approval_id
            or type(record.action) is not str
            or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', record.action)
            or type(record.status) is not str
            or record.status not in {'pending', 'approved', 'rejected', 'consumed'}
        ):
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval')
        if type(record.danger_class) is not str or record.danger_class not in DANGEROUS_ACTIONS:
            raise ApprovalValidationError('approval record has unknown danger class', code='malformed_approval')
        if (
            type(record.payload) is not dict
            or type(record.request_hash) is not str
            or not re.fullmatch(r'[0-9a-f]{64}', record.request_hash)
            or (record.decision_note is not None and type(record.decision_note) is not str)
            or (record.consumption_id is not None and type(record.consumption_id) is not str)
        ):
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval')
        if record.consumption_id is not None and not re.fullmatch(r'consume-[0-9a-f]{16}', record.consumption_id):
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval')
        if record.status == 'consumed' and (record.consumed_at is None or record.consumption_id is None):
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval')
        if record.status != 'consumed' and (record.consumed_at is not None or record.consumption_id is not None):
            raise ApprovalValidationError('approval record is malformed', code='malformed_approval')
        try:
            _parse_iso(record.requested_at)
            _parse_iso(record.expires_at)
            if record.decided_at is not None:
                _parse_iso(record.decided_at)
            if record.consumed_at is not None:
                _parse_iso(record.consumed_at)
        except (TypeError, ValueError) as exc:
            raise ApprovalValidationError('approval timestamps are malformed', code='malformed_approval') from exc
        return record

    def request(
        self,
        action: str,
        danger_class: str,
        payload: dict[str, Any] | None = None,
        *,
        ttl_minutes: int = 60,
    ) -> ApprovalRecord:
        if type(action) is not str or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', action):
            raise ApprovalValidationError('approval action is required', code='invalid_action')
        if type(danger_class) is not str or danger_class not in DANGEROUS_ACTIONS:
            raise ApprovalValidationError('unknown approval danger class', code='invalid_danger_class')
        if payload is not None and type(payload) is not dict:
            raise ApprovalValidationError('approval payload must be an object', code='invalid_payload')
        if type(ttl_minutes) is not int:
            raise ApprovalValidationError('approval ttl is out of range', code='invalid_ttl')
        ttl = ttl_minutes
        if ttl <= 0 or ttl > 24 * 60:
            raise ApprovalValidationError('approval ttl is out of range', code='invalid_ttl')
        self._ensure_dir()
        aid = 'appr-' + uuid.uuid4().hex[:16]
        requested = datetime.now(timezone.utc)
        request_hash = _validated_payload_digest(payload, code='invalid_payload')
        rec = ApprovalRecord(
            approval_id=aid,
            action=action,
            danger_class=danger_class,
            status='pending',
            requested_at=requested.isoformat().replace('+00:00', 'Z'),
            expires_at=(requested + timedelta(minutes=ttl)).isoformat().replace('+00:00', 'Z'),
            decided_at=None,
            request_hash=request_hash,
            payload=redact_approval_payload(payload),
        )
        with self._locked(aid) as lease:
            self._write(rec, lease)
        self._append_trace(
            'pending',
            {
                'approval_id': aid,
                'action': action,
                'danger_class': danger_class,
                'request_hash': rec.request_hash,
                'payload': rec.payload,
            },
        )
        return rec

    def load(self, approval_id: str) -> ApprovalRecord:
        with self._locked(approval_id) as lease:
            return self._load_unlocked(approval_id, lease)

    def decide(self, approval_id: str, status: str, note: str | None = None) -> ApprovalRecord:
        if type(status) is not str or status not in {'approved', 'rejected'}:
            raise ApprovalValidationError('approval decision must be approved or rejected', code='invalid_decision')
        if note is not None and type(note) is not str:
            raise ApprovalValidationError('approval decision note must be text', code='invalid_decision')
        with self._locked(approval_id) as lease:
            rec = self._load_unlocked(approval_id, lease)
            if rec.status != 'pending':
                raise ApprovalRequired(rec, code='approval_already_decided')
            if datetime.now(timezone.utc) >= _parse_iso(rec.expires_at):
                raise ApprovalRequired(rec, code='approval_expired')
            data = rec.to_dict()
            data['status'] = status
            data['decided_at'] = now_iso()
            if note:
                data['decision_note'] = _redacted_marker(str(note))
            new = ApprovalRecord(**data)
            self._write(new, lease)
        self._append_trace(
            status,
            {'approval_id': approval_id, 'action': rec.action, 'request_hash': rec.request_hash},
        )
        return new

    def purge_records(self, approval_ids: list[str] | tuple[str, ...] | set[str]) -> dict[str, int]:
        """Remove exact task-bound approval records and consumption claims.

        The caller must already hold the separate derived-data purge approval.
        The shared authority file and redacted trace timeline are retained as
        audit controls; only records explicitly linked to the deleted scope are
        removed.
        """

        unique = sorted(set(approval_ids))
        for approval_id in unique:
            self._path(approval_id)
        removed_records = 0
        removed_claims = 0
        for approval_id in unique:
            with self._locked(approval_id) as lease:
                for name, counter in (
                    (f'{approval_id}.redacted.json', 'record'),
                    (self._consumption_claim_name(approval_id), 'claim'),
                ):
                    try:
                        os.unlink(name, dir_fd=lease.dir_fd)
                    except FileNotFoundError:
                        continue
                    if counter == 'record':
                        removed_records += 1
                    else:
                        removed_claims += 1
                try:
                    os.fsync(lease.dir_fd)
                except OSError:
                    pass
        return {'approval_records': removed_records, 'approval_claims': removed_claims}

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        from trove_core.bounds import BoundedLimit, TRACE_EVENTS_APPROVALS

        limit = BoundedLimit(limit, field='limit', spec=TRACE_EVENTS_APPROVALS)
        try:
            os.lstat(self.dir)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ApprovalValidationError('approval storage is unavailable', code='approval_storage_permissions') from exc
        rows: list[dict[str, Any]] = []
        sentinel = 'appr-' + '0' * 16
        with self._locked(sentinel) as lease:
            candidates: list[tuple[int, str]] = []
            try:
                names = os.listdir(lease.dir_fd)
            except OSError as exc:
                raise ApprovalValidationError('approval storage is unavailable', code='approval_storage_permissions') from exc
            for name in names:
                match = re.fullmatch(r'(appr-[0-9a-f]{16})\.redacted\.json', name)
                if match is None:
                    continue
                try:
                    listed = os.stat(name, dir_fd=lease.dir_fd, follow_symlinks=False)
                except OSError:
                    continue
                candidates.append((int(listed.st_mtime_ns), match.group(1)))
            for _mtime, approval_id in sorted(candidates, reverse=True):
                try:
                    rows.append(self._load_unlocked(approval_id, lease).to_dict())
                except ApprovalValidationError:
                    continue
                if len(rows) >= limit:
                    break
        return rows

    @staticmethod
    def _claim_matches(
        claim: dict[str, Any],
        record: ApprovalRecord,
        *,
        action: str,
        danger_class: str,
        request_hash: str,
    ) -> bool:
        return (
            _constant_text_equal(claim['approval_id'], record.approval_id)
            and _constant_text_equal(claim['action'], action)
            and _constant_text_equal(claim['danger_class'], danger_class)
            and _constant_text_equal(claim['request_hash'], request_hash)
            and _constant_text_equal(record.action, action)
            and _constant_text_equal(record.danger_class, danger_class)
            and _constant_text_equal(record.request_hash, request_hash)
        )

    @staticmethod
    def _record_from_claim(record: ApprovalRecord, claim: dict[str, Any]) -> ApprovalRecord:
        data = record.to_dict()
        data.update(
            status='consumed',
            consumed_at=claim['consumed_at'],
            consumption_id=claim['consumption_id'],
        )
        return ApprovalRecord(**data)

    def _replay_from_existing_claim(
        self,
        lease: _ApprovalStorageLease,
        record: ApprovalRecord,
        claim: dict[str, Any],
        *,
        action: str,
        danger_class: str,
        request_hash: str,
    ) -> None:
        if not self._claim_matches(
            claim,
            record,
            action=action,
            danger_class=danger_class,
            request_hash=request_hash,
        ):
            raise ApprovalValidationError('approval consumption claim conflicts with record', code='approval_consumption_state_invalid')
        if record.status not in {'approved', 'consumed'}:
            raise ApprovalValidationError('approval consumption claim conflicts with status', code='approval_consumption_state_invalid')
        consumed = self._record_from_claim(record, claim)
        if record.status == 'approved':
            self._write(consumed, lease)
        elif (
            not _constant_text_equal(record.consumed_at or '', claim['consumed_at'])
            or not _constant_text_equal(record.consumption_id or '', claim['consumption_id'])
        ):
            raise ApprovalValidationError('approval consumption claim conflicts with record', code='approval_consumption_state_invalid')
        raise ApprovalRequired(consumed, code='approval_replayed')

    def consume(
        self,
        action: str,
        danger_class: str,
        payload: dict[str, Any] | None,
        *,
        approval_id: str,
    ) -> ApprovalGrant:
        if type(action) is not str or type(danger_class) is not str:
            raise ApprovalValidationError('approval command fields are malformed', code='invalid_action')
        if payload is not None and type(payload) is not dict:
            raise ApprovalValidationError('approval payload must be an object', code='invalid_payload')
        if type(approval_id) is not str:
            raise ApprovalValidationError('malformed approval id', code='malformed_approval_id')
        expected_hash = _validated_payload_digest(payload, code='invalid_payload')
        consumed_at = now_iso()
        consumption_id = 'consume-' + uuid.uuid4().hex[:16]
        with self._locked(approval_id) as lease:
            rec = self._load_unlocked(approval_id, lease)
            if not _constant_text_equal(rec.action, action):
                raise ApprovalRequired(rec, code='approval_action_mismatch')
            if not _constant_text_equal(rec.danger_class, danger_class):
                raise ApprovalRequired(rec, code='approval_danger_class_mismatch')
            if not _constant_text_equal(rec.request_hash, expected_hash):
                raise ApprovalRequired(rec, code='approval_payload_mismatch')
            existing_claim = self._read_consumption_claim(lease, approval_id)
            if existing_claim is not None:
                self._replay_from_existing_claim(
                    lease,
                    rec,
                    existing_claim,
                    action=action,
                    danger_class=danger_class,
                    request_hash=expected_hash,
                )
            if datetime.now(timezone.utc) >= _parse_iso(rec.expires_at):
                raise ApprovalRequired(rec, code='approval_expired')
            if rec.status == 'consumed':
                raise ApprovalRequired(rec, code='approval_replayed')
            if rec.status != 'approved':
                raise ApprovalRequired(rec, code=f'approval_{rec.status}')
            proposed_claim = {
                'schema': 1,
                'approval_id': approval_id,
                'action': action,
                'danger_class': danger_class,
                'request_hash': expected_hash,
                'consumed_at': consumed_at,
                'consumption_id': consumption_id,
            }
            created, persisted_claim = self._publish_consumption_claim(lease, proposed_claim)
            if not created:
                self._replay_from_existing_claim(
                    lease,
                    rec,
                    persisted_claim,
                    action=action,
                    danger_class=danger_class,
                    request_hash=expected_hash,
                )
            if persisted_claim != proposed_claim:
                raise ApprovalValidationError('approval consumption claim changed during publish', code='approval_consumption_state_invalid')
            consumed = self._record_from_claim(rec, persisted_claim)
            self._write(consumed, lease)
        self._append_trace(
            'consumed',
            {
                'approval_id': approval_id,
                'action': action,
                'danger_class': danger_class,
                'request_hash': expected_hash,
                'consumption_id': consumption_id,
            },
        )
        grant = ApprovalGrant(
            approval_id=approval_id,
            action=action,
            danger_class=danger_class,
            request_hash=expected_hash,
            expires_at=consumed.expires_at,
            consumed_at=consumed_at,
            consumption_id=consumption_id,
            vault_hash=self._vault_hash,
            _seal=_GRANT_SEAL,
            _issued_pid=os.getpid(),
        )
        return _register_issued_grant(grant)

    def require(
        self,
        action: str,
        danger_class: str,
        payload: dict[str, Any] | None = None,
        *,
        approval_id: str | None = None,
        one_step_approval: bool = False,
    ) -> ApprovalGrant:
        if type(one_step_approval) is not bool:
            raise ApprovalValidationError('one-step approval flag must be boolean', code='invalid_one_step_approval')
        if approval_id is not None and type(approval_id) is not str:
            raise ApprovalValidationError('malformed approval id', code='malformed_approval_id')
        if approval_id:
            return self.consume(action, danger_class, payload, approval_id=approval_id)
        rec = self.request(action, danger_class, payload)
        if one_step_approval is not True:
            raise ApprovalRequired(rec)
        approved = self.decide(rec.approval_id, 'approved', note='local explicit confirmation')
        return self.consume(action, danger_class, payload, approval_id=approved.approval_id)


def approval_required_payload(record: ApprovalRecord, *, code: str = 'approval_required') -> dict[str, Any]:
    return {
        'error': {
            'code': code,
            'message': 'Dangerous local action requires an exact, unexpired approval before execution.',
        },
        'approval': record.to_dict(),
    }
