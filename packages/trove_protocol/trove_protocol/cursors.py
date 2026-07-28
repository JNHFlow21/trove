from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import secrets
import time
from typing import Any, Callable, Mapping


class CursorError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise CursorError('cursor_invalid', 'cursor filters are not canonical JSON') from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CursorState:
    capability: str
    filters_digest: str
    keyset: Mapping[str, Any]
    high_water: str
    generation: str
    vault_identity: str
    issued_at: float
    expires_at: float


class CursorStore:
    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.time,
    ):
        if ttl_seconds <= 0 or max_entries < 1:
            raise ValueError('cursor bounds must be positive')
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.clock = clock
        self._entries: OrderedDict[str, CursorState] = OrderedDict()

    def issue(
        self,
        *,
        capability: str,
        filters: Mapping[str, Any],
        keyset: Mapping[str, Any],
        high_water: str,
        generation: str,
        vault_identity: str,
    ) -> str:
        if not all(isinstance(value, str) and value for value in (
            capability, high_water, generation, vault_identity,
        )):
            raise CursorError('cursor_invalid', 'cursor binding fields are required')
        now = self.clock()
        self._prune(now)
        handle = secrets.token_urlsafe(32)
        while handle in self._entries:
            handle = secrets.token_urlsafe(32)
        self._entries[handle] = CursorState(
            capability=capability,
            filters_digest=_canonical_digest(filters),
            keyset=dict(keyset),
            high_water=high_water,
            generation=generation,
            vault_identity=vault_identity,
            issued_at=now,
            expires_at=now + self.ttl_seconds,
        )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return handle

    def resolve(
        self,
        handle: str,
        *,
        capability: str,
        filters: Mapping[str, Any],
        generation: str,
        vault_identity: str,
    ) -> CursorState:
        if not isinstance(handle, str) or len(handle) < 20:
            raise CursorError('cursor_invalid', 'cursor handle is invalid')
        state = self._entries.get(handle)
        if state is None:
            raise CursorError('cursor_invalid', 'cursor is unknown')
        now = self.clock()
        if state.expires_at <= now:
            del self._entries[handle]
            raise CursorError('cursor_expired', 'cursor has expired')
        if state.generation != generation:
            raise CursorError('cursor_stale', 'Vault generation changed after cursor issuance')
        if (
            state.capability != capability
            or state.filters_digest != _canonical_digest(filters)
            or state.vault_identity != vault_identity
        ):
            raise CursorError('cursor_mismatch', 'cursor is bound to a different request or Vault')
        self._entries.move_to_end(handle)
        return state

    def _prune(self, now: float) -> None:
        for handle in [
            handle for handle, state in self._entries.items()
            if state.expires_at <= now
        ]:
            del self._entries[handle]


__all__ = ['CursorError', 'CursorState', 'CursorStore']
