from __future__ import annotations

import re
import secrets
import time
from typing import Any, Callable, Mapping

from trove_protocol.cursors import CursorError, CursorState, CursorStore


class DaemonCursorStore:
    """Restart-, Vault-, generation-, filter-, and TTL-bound random cursors."""

    def __init__(
        self,
        *,
        vault_identity: str,
        restart_id: str | None = None,
        ttl_seconds: float = 300.0,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.time,
    ):
        self.vault_identity = vault_identity
        self.restart_id = restart_id or secrets.token_hex(16)
        if not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', self.restart_id):
            raise ValueError('restart_id is invalid')
        self._store = CursorStore(ttl_seconds=ttl_seconds, max_entries=max_entries, clock=clock)

    def issue(
        self,
        *,
        capability: str,
        filters: Mapping[str, Any],
        keyset: Mapping[str, Any],
        high_water: str,
        generation: str,
    ) -> str:
        inner = self._store.issue(
            capability=capability, filters=filters, keyset=keyset,
            high_water=high_water, generation=generation,
            vault_identity=self.vault_identity,
        )
        return f'{self.restart_id}.{inner}'

    def resolve(
        self,
        handle: str,
        *,
        capability: str,
        filters: Mapping[str, Any],
        generation: str,
    ) -> CursorState:
        if not isinstance(handle, str) or '.' not in handle:
            raise CursorError('cursor_invalid', 'cursor handle is invalid')
        restart_id, inner = handle.split('.', 1)
        if restart_id != self.restart_id:
            raise CursorError('cursor_stale', 'cursor belongs to a prior daemon restart')
        return self._store.resolve(
            inner, capability=capability, filters=filters,
            generation=generation, vault_identity=self.vault_identity,
        )


__all__ = ['DaemonCursorStore']
