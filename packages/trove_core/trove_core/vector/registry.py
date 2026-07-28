from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore, SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT
from trove_core.vector.zvec_store import ZVecStore

VECTOR_STATES = {'available', 'unavailable_fallback', 'degraded'}


@dataclass(frozen=True)
class VectorBackendStatus:
    state: str
    selected_backend: str
    preferred_backend: str
    reason_code: str | None
    message: str | None
    baseline_search_available: bool
    zvec: dict[str, Any]
    sqlite: dict[str, Any]
    message_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['states'] = sorted(VECTOR_STATES)
        return data


class VectorBackendRegistry:
    """Single product contract for local vector availability and fallback.

    The registry is intentionally local-only. It never downloads models, never calls cloud
    providers, and treats exact/FTS as the shippable fallback when semantic backends are absent.
    """

    def __init__(self, *, store: SQLiteStore, zvec_path: Path, provider: object | None = None):
        self.store = store
        self.zvec_path = Path(zvec_path)
        self.provider = provider
        ledger_backend = 'zvec-cloud' if getattr(provider, 'egress_kind', None) == 'cloud_embedding_upload' else 'zvec'
        self._zvec = ZVecStore(self.zvec_path, store=store, ledger_backend=ledger_backend)

    def sqlite_entries(self) -> int:
        if not self.store.path.exists():
            return 0
        try:
            with self.store.connect() as conn:
                if not self.store._table_exists(conn, 'vector_entries'):
                    return 0
                return int(conn.execute('SELECT COUNT(*) FROM vector_entries').fetchone()[0])
        except Exception:
            return 0

    def message_count(self) -> int:
        try:
            if not self.store.path.exists():
                return 0
            with self.store.connect() as conn:
                if not self.store._table_exists(conn, 'messages'):
                    return 0
                # Status needs one number, not the four table-wide counts in
                # ``SQLiteStore.counts``.  On a real Vault the unused chunk
                # count alone can add seconds to every maintain/status call.
                return int(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0])
        except Exception:
            return 0

    def _sqlite_status(self) -> dict[str, Any]:
        entries = self.sqlite_entries()
        return {
            'backend': 'sqlite',
            'available': bool(self.provider and entries > 0 and entries <= SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT),
            'collection_exists': entries > 0,
            'entries': entries,
            'diagnostic_only': True,
            'max_search_entries': SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT,
            'reason_code': None if 0 < entries <= SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT else ('sqlite_vector_diagnostic_limit_exceeded' if entries > SQLITE_VECTOR_DIAGNOSTIC_SEARCH_LIMIT else 'sqlite_vectors_missing'),
        }

    def status(self, preferred_backend: str = 'zvec') -> VectorBackendStatus:
        zvec_status = self._zvec.status(provider=self.provider)
        sqlite_status = self._sqlite_status()
        baseline = self.store.path.exists()
        provider_ready = self.provider is not None

        if preferred_backend == 'sqlite':
            if sqlite_status['available']:
                return VectorBackendStatus('available', 'sqlite', preferred_backend, None, None, baseline, zvec_status, sqlite_status, self.message_count())
            reason = 'local_embedding_model_missing' if not provider_ready else sqlite_status.get('reason_code') or 'sqlite_vectors_missing'
            return VectorBackendStatus('unavailable_fallback', 'none', preferred_backend, reason, 'SQLite vector diagnostic fallback is not ready; exact/FTS remains active.', baseline, zvec_status, sqlite_status, self.message_count())

        if not provider_ready:
            return VectorBackendStatus('unavailable_fallback', 'none', preferred_backend, 'local_embedding_model_missing', 'No local embedding model is configured; exact/FTS remains active.', baseline, zvec_status, sqlite_status, self.message_count())
        if zvec_status.get('available') and (zvec_status.get('stale') or zvec_status.get('rebuild_required')):
            if sqlite_status['available']:
                return VectorBackendStatus('available', 'sqlite', preferred_backend, 'zvec_rebuild_required_sqlite_active', 'ZVEC metadata is stale; SQLite local vectors are active until rebuild.', baseline, zvec_status, sqlite_status, self.message_count())
            return VectorBackendStatus('unavailable_fallback', 'none', preferred_backend, 'zvec_rebuild_required', 'ZVEC collection needs rebuild for the current vector text contract; exact/FTS remains active.', baseline, zvec_status, sqlite_status, self.message_count())
        if zvec_status.get('available') and zvec_status.get('collection_exists'):
            if zvec_status.get('health') == 'degraded':
                return VectorBackendStatus('degraded', 'none', preferred_backend, zvec_status.get('reason_code') or 'zvec_degraded', zvec_status.get('unavailable_reason') or 'ZVEC collection could not be opened; exact/FTS remains active.', baseline, zvec_status, sqlite_status, self.message_count())
            return VectorBackendStatus('available', 'zvec', preferred_backend, None, None, baseline, zvec_status, sqlite_status, self.message_count())
        if sqlite_status['available']:
            return VectorBackendStatus('available', 'sqlite', preferred_backend, 'zvec_unavailable_sqlite_active', 'ZVEC is unavailable; SQLite local vectors are active.', baseline, zvec_status, sqlite_status, self.message_count())
        if zvec_status.get('available') and not zvec_status.get('collection_exists'):
            return VectorBackendStatus('unavailable_fallback', 'none', preferred_backend, 'zvec_collection_missing', 'ZVEC is installed but no local collection has been built; exact/FTS remains active.', baseline, zvec_status, sqlite_status, self.message_count())
        return VectorBackendStatus('unavailable_fallback', 'none', preferred_backend, zvec_status.get('reason_code') or 'zvec_import_unavailable', zvec_status.get('unavailable_reason') or 'ZVEC unavailable; exact/FTS remains active.', baseline, zvec_status, sqlite_status, self.message_count())

    def select(self, preferred_backend: str = 'zvec') -> tuple[object | None, VectorBackendStatus]:
        status = self.status(preferred_backend)
        if status.state != 'available':
            return None, status
        if status.selected_backend == 'zvec':
            return self._zvec, status
        if status.selected_backend == 'sqlite':
            return SQLiteVectorStore(self.store), status
        return None, status

    def test(self, preferred_backend: str = 'zvec') -> dict[str, Any]:
        status = self.status(preferred_backend).to_dict()
        probe = {'ok': status['state'] == 'available', 'selected_backend': status['selected_backend'], 'state': status['state']}
        if status['state'] != 'available':
            probe['fallback_ok'] = True
            probe['reason_code'] = status['reason_code']
            return probe
        try:
            backend, _ = self.select(preferred_backend)
            if backend is None:
                raise RuntimeError('no vector backend selected')
            probe['search_probe'] = 'ready'
            return probe
        except Exception as exc:
            return {'ok': False, 'fallback_ok': True, 'state': 'degraded', 'selected_backend': 'none', 'reason_code': 'vector_probe_failed', 'message': str(exc)}
