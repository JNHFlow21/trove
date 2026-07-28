from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
import re
import threading
from typing import Iterator
from uuid import uuid4
import weakref

from trove_core.vault.config import VaultConfig
from trove_core.vault.locks import (
    LOCK_STALE_SECONDS,
    _StableVaultWriterLock,
    _process_start_time,
    VaultOperationLocked,
)


_SESSION_SEAL = object()
_OWNER_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$')
_PROCESS_IDENTITY_PID: int | None = None
_PROCESS_IDENTITY_BIRTH: str | None = None
_PROCESS_IDENTITY_LOCK = threading.Lock()
_PROCESS_COORDINATORS: dict[str, 'VaultOperationCoordinator'] = {}
_PROCESS_COORDINATORS_LOCK = threading.RLock()
_PROCESS_COORDINATORS_PID = os.getpid()
_ALL_COORDINATORS: weakref.WeakSet['VaultOperationCoordinator'] = weakref.WeakSet()


class MutationOutsideCoordinator(RuntimeError):
    """Raised when mutation code lacks an authentic, active writer session."""

    def __init__(
        self,
        message: str = 'Vault mutation requires an active writer session',
        *,
        code: str = 'mutation_outside_coordinator',
    ):
        super().__init__(message)
        self.code = code


def _coerce_config(vault: VaultConfig | str | Path) -> VaultConfig:
    cfg = (
        vault
        if isinstance(vault, VaultConfig)
        else VaultConfig.resolve(str(Path(vault).expanduser()), env={})
    )
    # Freeze a lexical absolute path.  Do not follow a final symlink here: the
    # lock layer rejects it and binds the real directory inode at acquisition.
    root = Path(os.path.abspath(os.path.expanduser(str(cfg.root))))
    return replace(cfg, root=root)


def _vault_hash(vault: VaultConfig | str | Path) -> str:
    cfg = _coerce_config(vault)
    canonical = str(cfg.root.expanduser().resolve())
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _valid_owner_nonce(value: str | None) -> bool:
    return value is None or bool(re.fullmatch(r'[0-9a-f]{32}', value))


def _current_process_identity() -> tuple[int, str]:
    global _PROCESS_IDENTITY_PID, _PROCESS_IDENTITY_BIRTH
    pid = os.getpid()
    with _PROCESS_IDENTITY_LOCK:
        if _PROCESS_IDENTITY_PID != pid or _PROCESS_IDENTITY_BIRTH is None:
            # A process birth marker is immutable, so cache it instead of
            # spawning `ps` for every nested validation.  The PID check makes
            # inherited cache state fork-safe.
            process_birth = _process_start_time(pid) or f'fallback:{uuid4().hex}'
            _PROCESS_IDENTITY_PID = pid
            _PROCESS_IDENTITY_BIRTH = process_birth
        return pid, _PROCESS_IDENTITY_BIRTH


def _process_registry() -> dict[str, 'VaultOperationCoordinator']:
    """Return the current process registry, discarding state inherited by fork."""

    global _PROCESS_COORDINATORS_PID
    pid = os.getpid()
    if _PROCESS_COORDINATORS_PID != pid:
        _PROCESS_COORDINATORS.clear()
        _PROCESS_COORDINATORS_PID = pid
    return _PROCESS_COORDINATORS


@dataclass(frozen=True, slots=True)
class VaultWriteSession:
    """An unforgeable, process-bound capability for one coordinated mutation."""

    owner: str
    vault_hash: str
    pid: int
    process_birth: str = field(repr=False)
    coordinator_nonce: str = field(repr=False)
    session_nonce: str = field(repr=False)
    parent_nonce: str | None = field(default=None, repr=False)
    _coordinator: 'VaultOperationCoordinator' = field(  # type: ignore[assignment]
        repr=False,
        compare=False,
        default=None,
    )
    _seal: object = field(repr=False, compare=False, default=None)
    _released: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self) -> None:
        if self._seal is not _SESSION_SEAL or self._coordinator is None:
            raise MutationOutsideCoordinator('Vault writer session is forged', code='forged_write_session')

    @property
    def active(self) -> bool:
        if self._released:
            return False
        try:
            self._coordinator.validate(self)
        except MutationOutsideCoordinator:
            return False
        return True

    @property
    def released(self) -> bool:
        return self._released

    def validate_for(self, vault: VaultConfig | str | Path) -> 'VaultWriteSession':
        return self._coordinator.validate_mutation(self, vault=vault)

    def nested(self, owner: str) -> Iterator['VaultWriteSession']:
        """Open an explicitly parented nested session on the same coordinator."""

        return self._coordinator.write(owner=owner, parent=self)


class VaultOperationCoordinator:
    """Serialize Vault writers and issue scoped capabilities to mutation code.

    The outermost session owns a kernel `flock` on a stable inode.  Nested work
    never reacquires the OS lock: it must present the current active session as
    its explicit parent, making re-entrancy visible in call signatures.
    """

    def __init__(self, vault: VaultConfig | str | Path, *, stale_seconds: int = LOCK_STALE_SECONDS):
        self.cfg = _coerce_config(vault)
        self.vault_hash = _vault_hash(self.cfg)
        self.stale_seconds = stale_seconds
        self.coordinator_nonce = uuid4().hex
        self._mutex = threading.RLock()
        self._sessions: dict[str, VaultWriteSession] = {}
        self._stack: list[VaultWriteSession] = []
        self._outer_lock: _StableVaultWriterLock | None = None
        self._pid: int | None = None
        self._process_birth: str | None = None
        _ALL_COORDINATORS.add(self)

    @contextmanager
    def write(
        self,
        *,
        owner: str,
        parent: VaultWriteSession | None = None,
    ) -> Iterator[VaultWriteSession]:
        session = self.acquire(owner=owner, parent=parent)
        try:
            yield session
        finally:
            self.release(session)

    def acquire(
        self,
        *,
        owner: str,
        parent: VaultWriteSession | None = None,
        owner_nonce: str | None = None,
    ) -> VaultWriteSession:
        if not isinstance(owner, str) or not _OWNER_RE.fullmatch(owner):
            raise MutationOutsideCoordinator(
                'Vault writer owner must be a short identifier',
                code='invalid_writer_owner',
            )
        if not _valid_owner_nonce(owner_nonce):
            raise MutationOutsideCoordinator(
                'Vault writer nonce must be 32 lowercase hexadecimal characters',
                code='invalid_writer_nonce',
            )
        if parent is not None and owner_nonce is not None:
            raise MutationOutsideCoordinator(
                'Nested Vault writers cannot override the session nonce',
                code='invalid_writer_nonce',
            )

        with self._mutex:
            if parent is not None:
                return self._acquire_nested(owner=owner, parent=parent)
            if self._stack:
                raise MutationOutsideCoordinator(
                    'Nested Vault mutation requires an explicit parent session',
                    code='parent_session_required',
                )
            return self._acquire_outer(owner=owner, owner_nonce=owner_nonce)

    def _acquire_outer(self, *, owner: str, owner_nonce: str | None) -> VaultWriteSession:
        self.cfg.require_configured_for_write(action='Vault mutation')
        pid, process_birth = _current_process_identity()
        session_nonce = owner_nonce or uuid4().hex

        with _PROCESS_COORDINATORS_LOCK:
            registry = _process_registry()
            active = registry.get(self.vault_hash)
            if active is not None and active is not self:
                raise VaultOperationLocked()

            os_lock = _StableVaultWriterLock(
                self.cfg,
                owner=owner,
                owner_nonce=session_nonce,
                process_birth=process_birth,
                vault_hash=self.vault_hash,
                stale_seconds=self.stale_seconds,
            )
            os_lock.acquire()
            try:
                session = VaultWriteSession(
                    owner=owner,
                    vault_hash=self.vault_hash,
                    pid=pid,
                    process_birth=process_birth,
                    coordinator_nonce=self.coordinator_nonce,
                    session_nonce=session_nonce,
                    parent_nonce=None,
                    _coordinator=self,
                    _seal=_SESSION_SEAL,
                )
                self._outer_lock = os_lock
                self._pid = pid
                self._process_birth = process_birth
                self._sessions[session_nonce] = session
                self._stack.append(session)
                registry[self.vault_hash] = self
            except Exception:
                self._sessions.pop(session_nonce, None)
                self._stack.clear()
                self._outer_lock = None
                self._pid = None
                self._process_birth = None
                os_lock.release()
                raise
        return session

    def _acquire_nested(self, *, owner: str, parent: VaultWriteSession) -> VaultWriteSession:
        self.validate(parent)
        if not self._stack or self._stack[-1] is not parent:
            raise MutationOutsideCoordinator(
                'Nested Vault mutation parent is not the active leaf session',
                code='invalid_parent_session',
            )
        child = VaultWriteSession(
            owner=owner,
            vault_hash=self.vault_hash,
            pid=parent.pid,
            process_birth=parent.process_birth,
            coordinator_nonce=self.coordinator_nonce,
            session_nonce=uuid4().hex,
            parent_nonce=parent.session_nonce,
            _coordinator=self,
            _seal=_SESSION_SEAL,
        )
        self._sessions[child.session_nonce] = child
        self._stack.append(child)
        return child

    def validate(
        self,
        session: VaultWriteSession | None,
        *,
        vault: VaultConfig | str | Path | None = None,
    ) -> VaultWriteSession:
        with self._mutex:
            return self._validate_session(session, vault=vault, check_path=True)

    def _validate_session(
        self,
        session: VaultWriteSession | None,
        *,
        vault: VaultConfig | str | Path | None = None,
        check_path: bool,
    ) -> VaultWriteSession:
        if not isinstance(session, VaultWriteSession):
            raise MutationOutsideCoordinator(code='missing_write_session')
        if session._seal is not _SESSION_SEAL or session._coordinator is not self:
            raise MutationOutsideCoordinator('Vault writer session is forged', code='forged_write_session')
        if session._released:
            raise MutationOutsideCoordinator('Vault writer session was released', code='released_write_session')
        if session.vault_hash != self.vault_hash:
            raise MutationOutsideCoordinator(
                'Vault writer session belongs to another Vault',
                code='cross_vault_write_session',
            )
        if session.coordinator_nonce != self.coordinator_nonce:
            raise MutationOutsideCoordinator(
                'Vault writer coordinator nonce is invalid',
                code='forged_write_session',
            )
        pid, process_birth = _current_process_identity()
        if session.pid != pid:
            raise MutationOutsideCoordinator(
                'Vault writer session belongs to another process',
                code='cross_process_write_session',
            )
        if session.process_birth != process_birth:
            raise MutationOutsideCoordinator(
                'Vault writer process identity changed',
                code='stale_process_write_session',
            )
        registered = self._sessions.get(session.session_nonce)
        if registered is not session:
            raise MutationOutsideCoordinator(
                'Vault writer session is inactive or forged',
                code='inactive_write_session',
            )
        if self._pid != pid or self._process_birth != process_birth or self._outer_lock is None:
            raise MutationOutsideCoordinator('Vault writer coordinator is inactive', code='inactive_write_session')
        if check_path:
            try:
                self._outer_lock.validate_bound_identity()
            except VaultOperationLocked as exc:
                raise MutationOutsideCoordinator(
                    'Vault writer path identity changed',
                    code='vault_writer_path_changed',
                ) from exc
        if vault is not None and session.vault_hash != _vault_hash(vault):
            raise MutationOutsideCoordinator(
                'Vault writer session belongs to another Vault',
                code='cross_vault_write_session',
            )
        return session

    def validate_mutation(
        self,
        session: VaultWriteSession | None,
        *,
        vault: VaultConfig | str | Path | None = None,
    ) -> VaultWriteSession:
        """Validate the leaf capability currently allowed to mutate state."""

        with self._mutex:
            validated = self.validate(session, vault=vault)
            if not self._stack or self._stack[-1] is not validated:
                raise MutationOutsideCoordinator(
                    'Only the active leaf writer session may mutate the Vault',
                    code='non_leaf_write_session',
                )
            return validated

    def release(self, session: VaultWriteSession) -> None:
        with self._mutex:
            # Pathname replacement must invalidate mutations, but it must never
            # prevent releasing the descriptors bound to the original inode.
            self._validate_session(session, check_path=False)
            if not self._stack or self._stack[-1] is not session:
                raise MutationOutsideCoordinator(
                    'Vault writer sessions must be released in nesting order',
                    code='active_child_write_session',
                )
            self._stack.pop()
            self._sessions.pop(session.session_nonce, None)
            object.__setattr__(session, '_released', True)

            if session.parent_nonce is not None:
                return

            os_lock = self._outer_lock
            self._outer_lock = None
            self._pid = None
            self._process_birth = None
            with _PROCESS_COORDINATORS_LOCK:
                registry = _process_registry()
                if registry.get(self.vault_hash) is self:
                    registry.pop(self.vault_hash, None)
            if os_lock is not None:
                os_lock.release()

    def _after_fork_in_child(self) -> None:
        """Invalidate inherited capabilities without taking inherited locks."""

        os_lock = self._outer_lock
        self._outer_lock = None
        if os_lock is not None:
            os_lock._after_fork_in_child()
        for session in tuple(self._sessions.values()):
            object.__setattr__(session, '_released', True)
        self._sessions = {}
        self._stack = []
        self._pid = None
        self._process_birth = None
        self._mutex = threading.RLock()


def require_vault_write_session(
    vault: VaultConfig | str | Path,
    session: VaultWriteSession | None,
) -> VaultWriteSession:
    """Validate and return an authentic active session for `vault`."""

    if not isinstance(session, VaultWriteSession):
        raise MutationOutsideCoordinator(code='missing_write_session')
    return session.validate_for(vault)


def _after_fork_in_child() -> None:
    """Reset process globals and close every inherited writer descriptor."""

    global _PROCESS_IDENTITY_PID, _PROCESS_IDENTITY_BIRTH, _PROCESS_IDENTITY_LOCK
    global _PROCESS_COORDINATORS, _PROCESS_COORDINATORS_LOCK, _PROCESS_COORDINATORS_PID
    global _ALL_COORDINATORS

    inherited = tuple(_ALL_COORDINATORS)
    _PROCESS_IDENTITY_PID = None
    _PROCESS_IDENTITY_BIRTH = None
    _PROCESS_IDENTITY_LOCK = threading.Lock()
    _PROCESS_COORDINATORS = {}
    _PROCESS_COORDINATORS_LOCK = threading.RLock()
    _PROCESS_COORDINATORS_PID = os.getpid()
    _ALL_COORDINATORS = weakref.WeakSet()
    for coordinator in inherited:
        coordinator._after_fork_in_child()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork_in_child)
