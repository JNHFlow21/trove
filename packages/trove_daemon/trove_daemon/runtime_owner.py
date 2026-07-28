from __future__ import annotations

import gc
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig


class RuntimeOwner:
    """Daemon-scoped owner for reusable repositories, search, caches, and workers."""

    def __init__(
        self,
        config: VaultConfig | str | Path,
        *,
        provider_factory: Callable[[], object | None] | None = None,
        read_pool_size: int = 8,
        read_page_cache_kib: int = 8 * 1024,
        search_workers: int = 8,
        search_queue: int = 32,
        heavy_idle_seconds: float = 120.0,
        result_cache_max_entries: int = 64,
        result_cache_max_bytes: int = 4 * 1024 * 1024,
    ):
        if read_pool_size < 1 or search_workers < 1 or search_queue < 0:
            raise ValueError('runtime owner resource bounds are invalid')
        if type(read_page_cache_kib) is not int or not 256 <= read_page_cache_kib <= 64 * 1024:
            raise ValueError('runtime owner read cache budget is invalid')
        if heavy_idle_seconds <= 0:
            raise ValueError('heavy_idle_seconds must be positive')
        self.config = config if isinstance(config, VaultConfig) else VaultConfig.resolve(str(config))
        self.read_pool_size = int(read_pool_size)
        self.read_page_cache_kib = read_page_cache_kib
        self.search_workers = int(search_workers)
        self.search_queue = int(search_queue)
        self.heavy_idle_seconds = float(heavy_idle_seconds)
        self.result_cache_max_entries = int(result_cache_max_entries)
        self.result_cache_max_bytes = int(result_cache_max_bytes)
        self._provider_factory = provider_factory
        self._lock = threading.RLock()
        self._read_store: SQLiteStore | None = None
        self._queries = None
        self._search_runtime = None
        self._search_runtime_builds = 0
        self._provider_builds = 0
        self._generation_dirty = False
        self._last_heavy_use = 0.0
        self._idle_timer: threading.Timer | None = None
        self._closed = False
        self.provider_registry = None
        self.provider_load_status: dict[str, Any] | None = None
        self._reply_service = None

    def attach_provider_registry(self, registry: object, load_status: Mapping[str, Any]) -> None:
        with self._lock:
            self._ensure_open()
            self.provider_registry = registry
            self.provider_load_status = dict(load_status)

    def attach_reply_service(self, service: object) -> None:
        with self._lock:
            self._ensure_open()
            if self._reply_service is not None:
                raise RuntimeError('reply service is already attached')
            self._reply_service = service

    @property
    def reply_service(self):
        with self._lock:
            return self._reply_service

    @property
    def keepalive_required(self) -> bool:
        with self._lock:
            service = self._reply_service
        if service is not None:
            return bool(getattr(service, 'keepalive_required', False))
        from trove_core.reply import ReplyServiceConfig

        try:
            return ReplyServiceConfig.load(self.config.root).armed
        except Exception:
            return False

    def reply_status(self) -> dict[str, Any]:
        with self._lock:
            service = self._reply_service
        if service is not None:
            return dict(service.status())
        from trove_core.reply import ReplyServiceConfig

        config = ReplyServiceConfig.load(self.config.root)
        return {
            'state': 'stopped',
            'running': False,
            'armed': config.armed,
            'config': config.redacted(),
            'pending_reviews': 0,
            'unresolved_sends': 0,
            'last_poll_at': None,
            'last_error': (
                'reply_provider_unavailable' if config.configured else None
            ),
            'provider': {},
        }

    def reply_reviews(
        self,
        *,
        state: str = 'pending',
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            service = self._reply_service
        return (
            list(service.reviews(state=state, limit=limit))
            if service is not None
            else []
        )

    def reply_activity(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            service = self._reply_service
        return (
            list(service.activity(limit=limit))
            if service is not None
            else []
        )

    @property
    def read_store(self) -> SQLiteStore:
        with self._lock:
            self._ensure_open()
            if self._read_store is None:
                self._read_store = SQLiteStore(
                    self.config.paths.sqlite_path,
                    readonly=True,
                    max_connections=self.read_pool_size,
                    prepared_statement_cache_size=128,
                    page_cache_kib=self.read_page_cache_kib,
                )
            return self._read_store

    def uow_factory(self, config, *, readonly: bool = True):
        from trove_core.application.repositories import SQLiteUnitOfWork

        if not readonly:
            return SQLiteUnitOfWork(config, readonly=False)
        return SQLiteUnitOfWork(config, readonly=True, store=self.read_store)

    @property
    def queries(self):
        with self._lock:
            self._ensure_open()
            if self._queries is None:
                from trove_core.application.queries import TroveQueries

                self._queries = TroveQueries(
                    self.config, runtime=self, uow_factory=self.uow_factory,
                )
            return self._queries

    def _counted_provider_factory(self):
        with self._lock:
            self._provider_builds += 1
        if self._provider_factory is not None:
            return self._provider_factory()
        from trove_core.runtime import configured_embedding_provider

        return configured_embedding_provider(vault_root=self.config.root)

    def _runtime(self):
        with self._lock:
            self._ensure_open()
            if self._search_runtime is None:
                from trove_core.runtime import SearchRuntimeCache

                self._search_runtime = SearchRuntimeCache(
                    self.config,
                    provider_factory=self._counted_provider_factory,
                    max_workers=self.search_workers,
                    max_queue=self.search_queue,
                    result_cache_max_entries=self.result_cache_max_entries,
                    result_cache_max_bytes=self.result_cache_max_bytes,
                )
                self._search_runtime_builds += 1
            if self._generation_dirty:
                self._search_runtime.invalidate('daemon_generation_dirty')
                self._generation_dirty = False
            return self._search_runtime

    def search(self, request):
        response, _metrics = self.search_with_metrics(request)
        return response

    def search_with_metrics(self, request):
        runtime = self._runtime()
        try:
            return runtime.search_with_metrics(request)
        finally:
            self._touch_heavy()

    def memoize_generation(self, namespace, key, loader, *, cache_if=None):
        runtime = self._runtime()
        try:
            return runtime.memoize_generation(namespace, key, loader, cache_if=cache_if)
        finally:
            self._touch_heavy()

    def mark_generation_dirty(self) -> None:
        with self._lock:
            self._ensure_open()
            self._generation_dirty = True

    def _touch_heavy(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._last_heavy_use = time.monotonic()
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(self.heavy_idle_seconds, self._release_if_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _release_if_idle(self) -> None:
        with self._lock:
            if self._closed or self._search_runtime is None:
                return
            elapsed = time.monotonic() - self._last_heavy_use
            if elapsed < self.heavy_idle_seconds:
                remaining = self.heavy_idle_seconds - elapsed
                self._idle_timer = threading.Timer(remaining, self._release_if_idle)
                self._idle_timer.daemon = True
                self._idle_timer.start()
                return
            self._search_runtime.release_resources()
            self._idle_timer = None
        gc.collect()

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtime_status = self._search_runtime.status() if self._search_runtime is not None else {}
            read_connections = self._read_store.active_connection_count if self._read_store is not None else 0
            search_connections = int((runtime_status.get('resource_counts') or {}).get('engine_connections') or 0)
            return {
                'search_runtime_builds': self._search_runtime_builds,
                'search_engine_builds': int(runtime_status.get('engine_builds') or 0),
                'provider_builds': self._provider_builds,
                'heavy_loaded': bool(runtime_status.get('loaded')),
                'generation_dirty': self._generation_dirty,
                'read_connections': read_connections + search_connections,
                'max_read_connections': self.read_pool_size + self.search_workers + 1,
                'read_page_cache_kib': self.read_page_cache_kib,
                'max_read_page_cache_bytes': self.read_pool_size * self.read_page_cache_kib * 1024,
                'result_cache_entries': int(runtime_status.get('result_cache_entries') or 0),
                'result_cache_bytes': int(runtime_status.get('result_cache_bytes') or 0),
                'result_cache_max_entries': self.result_cache_max_entries,
                'result_cache_max_bytes': self.result_cache_max_bytes,
                'workers': runtime_status.get('workers') or {
                    'active_workers': 0, 'queued_workers': 0,
                    'max_workers': self.search_workers, 'max_queue': self.search_queue,
                    'closed': True,
                },
                'provider_load': dict(self.provider_load_status or {
                    'ok': False, 'error': {'code': 'provider_not_loaded'},
                    'next_action': {
                        'capability': 'trove.provider_status',
                        'action': 'install_or_repair_provider',
                    },
                    'pure_vault_read_available': True,
                }),
                'reply': self.reply_status(),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            timer, self._idle_timer = self._idle_timer, None
            runtime, self._search_runtime = self._search_runtime, None
            store, self._read_store = self._read_store, None
            reply, self._reply_service = self._reply_service, None
        if timer is not None:
            timer.cancel()
        if runtime is not None:
            runtime.close()
        if reply is not None:
            reply.close()
        if store is not None:
            store.close_all()
        gc.collect()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('daemon runtime owner is closed')


__all__ = ['RuntimeOwner']
