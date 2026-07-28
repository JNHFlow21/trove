from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import gc
import json
import os
import sqlite3
import threading
import time
import sys
from typing import Callable, Any, Iterable, Iterator
import hashlib
import uuid

from trove_core.approvals import ApprovalGrant, ApprovalValidationError, require_claimed_approval_grant
from trove_core.providers.config import ProviderConfig
from trove_core.providers.factory import ProviderFactory
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest, SearchResponse
from trove_core.store.sqlite_store import SQLiteStore, open_store, vector_document_text
from trove_core.store.change_journal import clear_all_dirty_citations
from trove_core.store.schema import VECTOR_SOURCE_REVISION_KEY
from trove_core.security.egress import cloud_embedding_payload
from trove_core.vault.config import VaultConfig
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore
from trove_core.vector.zvec_store import ZVecStore
from trove_core.vector.ledger import VectorIndexLedger
from trove_core.vector.registry import VectorBackendRegistry
from trove_core.vault.tracing import TraceTimeline
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.vault.generation import (
    VaultGenerationLease,
    VaultGenerationToken,
    coordinated_vault_generation_publish,
    vault_generation_read,
)


def _bounded_value_size(value: Any) -> int:
    candidate = value.to_dict() if callable(getattr(value, 'to_dict', None)) else value
    try:
        return len(json.dumps(
            candidate, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8'))
    except (TypeError, ValueError):
        return max(1, sys.getsizeof(candidate))


class ByteBoundedLRU:
    """Small LRU with independent entry and retained-byte hard limits."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        sizeof: Callable[[Any], int] = _bounded_value_size,
    ):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('max_bytes must be a positive integer')
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._sizeof = sizeof
        self._items: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self.current_bytes = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __getitem__(self, key: Any) -> Any:
        return self._items[key][0]

    def __setitem__(self, key: Any, value: Any) -> None:
        weight = int(self._sizeof(value)) + _bounded_value_size(key)
        if weight < 0:
            raise ValueError('cache value size cannot be negative')
        existing = self._items.pop(key, None)
        if existing is not None:
            self.current_bytes -= existing[1]
        if weight > self.max_bytes:
            self.evictions += 1
            return
        self._items[key] = (value, weight)
        self.current_bytes += weight
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries or self.current_bytes > self.max_bytes:
            _old_key, (_old_value, old_weight) = self._items.popitem(last=False)
            self.current_bytes -= old_weight
            self.evictions += 1

    def get(self, key: Any, default: Any = None) -> Any:
        item = self._items.get(key)
        return default if item is None else item[0]

    def move_to_end(self, key: Any) -> None:
        self._items.move_to_end(key)

    def popitem(self, *, last: bool = True) -> tuple[Any, Any]:
        key, (value, weight) = self._items.popitem(last=last)
        self.current_bytes -= weight
        return key, value

    def clear(self) -> None:
        self._items.clear()
        self.current_bytes = 0


def configured_local_provider(model_path: str | None = None):
    """Resolve local embedding through the shared provider factory."""

    return ProviderFactory.resolve().select_local_embedding(model_path).provider


def configured_embedding_provider(
    model_path: str | None = None,
    *,
    strict: bool = False,
    vault_root: str | Path | None = None,
    prefer_cloud: bool = False,
):
    """Resolve the explicitly selected Vault provider, then local fallback."""

    if prefer_cloud or vault_root is not None:
        from trove_core.providers.cloud_policy import cloud_retrieval_environment, cloud_retrieval_policy

        policy_enabled = bool(vault_root is not None and cloud_retrieval_policy(vault_root)['enabled'])
        if prefer_cloud or policy_enabled:
            env = dict(os.environ)
            env['TROVE_ENABLE_CLOUD_EMBEDDING'] = '1'
            if vault_root is not None:
                env = cloud_retrieval_environment(vault_root, env)
            factory = ProviderFactory.resolve(env)
            readiness = factory.readiness('embedding')
            if readiness.ready:
                return factory.create_cloud_embedding()
            # A persisted cloud-retrieval policy selects one score/vector
            # domain for the Vault. Background jobs must fail closed when its
            # credential transport is unavailable; silently updating the
            # local collection would strand dirty work and can leave recovery
            # markers for the wrong generation.
            if policy_enabled or (prefer_cloud and strict):
                raise RuntimeError(readiness.reason_code or 'cloud_embedding_unavailable')

    selection = ProviderFactory.resolve().select_local_embedding(model_path)
    if selection.provider is None and strict and ProviderConfig.resolve().cloud_embedding_enabled:
        raise RuntimeError('cloud_embedding_requires_exact_approval')
    return selection.provider


def zvec_collection_path(cfg: VaultConfig) -> Path:
    return cfg.paths.vector_dir / 'zvec' / 'messages'


def _cloud_embedding_provider(provider: Any | None) -> bool:
    return getattr(provider, 'egress_kind', None) == 'cloud_embedding_upload'


def zvec_collection_path_for_provider(cfg: VaultConfig, provider: Any | None) -> Path:
    if _cloud_embedding_provider(provider):
        return cfg.paths.vector_dir / 'zvec' / 'messages-cloud'
    return zvec_collection_path(cfg)


def zvec_ledger_backend(provider: Any | None) -> str:
    return 'zvec-cloud' if _cloud_embedding_provider(provider) else 'zvec'


def vector_entries_count(store: SQLiteStore) -> int:
    if not store.path.exists():
        return 0
    try:
        with store.connect() as conn:
            return int(conn.execute('SELECT COUNT(*) FROM vector_entries').fetchone()[0])
    except Exception:
        return 0


def vector_registry(cfg: VaultConfig, provider=None) -> VectorBackendRegistry:
    store = open_store(cfg.paths.sqlite_path, readonly=True) if cfg.paths.sqlite_path.exists() else SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    return VectorBackendRegistry(store=store, zvec_path=zvec_collection_path_for_provider(cfg, provider), provider=provider)


def _build_search_engine(cfg: VaultConfig) -> HyperSearch:
    store = open_store(cfg.paths.sqlite_path, readonly=True)
    provider = configured_embedding_provider(vault_root=cfg.root)
    registry = VectorBackendRegistry(store=store, zvec_path=zvec_collection_path_for_provider(cfg, provider), provider=provider)
    vector, status = registry.select('zvec')
    if _cloud_embedding_provider(provider) and status.state != 'available':
        local = configured_embedding_provider()
        local_registry = VectorBackendRegistry(store=store, zvec_path=zvec_collection_path(cfg), provider=local)
        local_vector, local_status = local_registry.select('zvec')
        if local_status.state == 'available':
            provider, vector, status = local, local_vector, local_status
    episode_store = None
    selector = None
    status_payload = status.to_dict()
    if _cloud_embedding_provider(provider):
        from trove_core.search.episodes import BoundedEvidenceSelector, EpisodeZVecStore, episode_collection_path

        candidate = EpisodeZVecStore(episode_collection_path(cfg.paths.vector_dir), store=store)
        status_payload['episodes'] = candidate.status(provider)
        if status_payload['episodes']['state'] == 'available':
            episode_store = candidate
            selector = BoundedEvidenceSelector(cfg.root)
    return HyperSearch(
        store,
        vector_store=vector,
        embedding_provider=provider,
        vector_status=status_payload,
        episode_store=episode_store,
        evidence_selector=selector,
    )


def build_search_engine(cfg: VaultConfig) -> 'GenerationSafeSearchEngine':
    """Return a search facade that leases one complete Vault generation/query."""

    return GenerationSafeSearchEngine(SearchRuntimeCache(cfg))


def warm_search_engine(engine: HyperSearch) -> dict[str, Any]:
    if os.environ.get('TROVE_DISABLE_SEARCH_WARMUP') == '1':
        return {'ok': False, 'skipped': True, 'reason_code': 'disabled_by_env'}
    return engine.warm_query_path()


class RuntimeBoundedError(RuntimeError):
    """Typed base class for bounded local-runtime failures."""

    code = 'runtime_bounded_error'

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'message': str(self),
            'retryable': True,
            'raw_content_included': False,
        }


class RuntimeOverloaded(RuntimeBoundedError):
    code = 'runtime_overloaded'

    def __init__(self, message: str = 'The bounded local runtime is at capacity.'):
        super().__init__(message)


class RuntimeTimedOut(RuntimeBoundedError):
    code = 'runtime_timeout'

    def __init__(self, message: str = 'The bounded local runtime exceeded its execution timeout.'):
        super().__init__(message)


class BoundedExecutor:
    """A ThreadPoolExecutor with a hard bound on workers plus queued work.

    ``ThreadPoolExecutor`` deliberately uses an unbounded queue.  Product
    processes must instead reject overload predictably, so admission is
    guarded by a semaphore before a future can be created.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        max_queue: int,
        thread_name_prefix: str,
        submit_timeout_seconds: float = 0.05,
    ):
        if type(max_workers) is not int or max_workers <= 0:
            raise ValueError('max_workers must be a positive integer')
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError('max_queue must be a non-negative integer')
        if not isinstance(submit_timeout_seconds, (int, float)) or submit_timeout_seconds < 0:
            raise ValueError('submit_timeout_seconds must be non-negative')
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.submit_timeout_seconds = float(submit_timeout_seconds)
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._slots = threading.BoundedSemaphore(max_workers + max_queue)
        self._state_lock = threading.Lock()
        self._active = 0
        self._queued = 0
        self._closed = False

    def submit(self, fn: Callable[..., Any], /, *args, **kwargs) -> Future:
        with self._state_lock:
            if self._closed:
                raise RuntimeOverloaded('The bounded local runtime is closed.')
        if not self._slots.acquire(timeout=self.submit_timeout_seconds):
            raise RuntimeOverloaded()
        with self._state_lock:
            if self._closed:
                self._slots.release()
                raise RuntimeOverloaded('The bounded local runtime is closed.')
            self._queued += 1

        admission_lock = threading.Lock()
        admission_state = {'value': 'queued'}

        def run():
            with admission_lock:
                admission_state['value'] = 'running'
                with self._state_lock:
                    self._queued -= 1
                    self._active += 1
            try:
                return fn(*args, **kwargs)
            finally:
                with self._state_lock:
                    self._active -= 1
                self._slots.release()

        try:
            executor = self._executor
            if executor is None:
                raise RuntimeOverloaded('The bounded local runtime is closed.')
            future = executor.submit(run)

            def release_cancelled(item: Future) -> None:
                if not item.cancelled():
                    return
                with admission_lock:
                    if admission_state['value'] != 'queued':
                        return
                    admission_state['value'] = 'cancelled'
                    with self._state_lock:
                        self._queued -= 1
                    self._slots.release()

            future.add_done_callback(release_cancelled)
            return future
        except BaseException:
            with self._state_lock:
                self._queued -= 1
            self._slots.release()
            raise

    def status(self) -> dict[str, int | bool]:
        with self._state_lock:
            return {
                'active_workers': self._active,
                'queued_workers': self._queued,
                'max_workers': self.max_workers,
                'max_queue': self.max_queue,
                'closed': self._closed,
            }

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)


class SearchRuntimeCache:
    """Thread-safe lazy cache for search dependencies in the local API process."""

    def __init__(
        self,
        cfg: VaultConfig,
        *,
        provider_factory: Callable[[], object | None] | None = None,
        max_workers: int = 8,
        max_queue: int = 32,
        timeout_seconds: float = 15.0,
        submit_timeout_seconds: float = 0.05,
        result_cache_max_entries: int = 64,
        result_cache_max_bytes: int = 4 * 1024 * 1024,
        memo_cache_max_entries: int = 64,
        memo_cache_max_bytes: int = 2 * 1024 * 1024,
    ):
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError('timeout_seconds must be greater than zero and at most 300')
        self.cfg = cfg
        self.provider_factory = provider_factory or (lambda: configured_embedding_provider(vault_root=cfg.root))
        self.max_workers = max_workers
        self.max_queue = max_queue
        self.timeout_seconds = float(timeout_seconds)
        self.submit_timeout_seconds = float(submit_timeout_seconds)
        self._lock = threading.RLock()
        self._generation = 0
        self._engine: HyperSearch | None = None
        self._provider = None
        self._vector = None
        self._status: dict | None = None
        self._last_invalidation_reason: str | None = None
        self._result_cache_max = result_cache_max_entries
        self._result_cache = ByteBoundedLRU(
            max_entries=result_cache_max_entries, max_bytes=result_cache_max_bytes,
        )
        self._result_inflight: dict[tuple[Any, ...], Future] = {}
        self._memo_cache_max = memo_cache_max_entries
        self._memo_cache = ByteBoundedLRU(
            max_entries=memo_cache_max_entries, max_bytes=memo_cache_max_bytes,
        )
        self._memo_inflight: dict[tuple[Any, ...], Future] = {}
        self._freshness_conn: sqlite3.Connection | None = None
        self._freshness_sqlite_identity: tuple[int, int] | None = None
        self._observed_generation_token: tuple[Any, ...] | None = None
        self._executor: BoundedExecutor | None = None
        self._cache_hits = 0
        self._cache_misses = 0
        self._singleflight_followers = 0
        self._memo_cache_hits = 0
        self._memo_cache_misses = 0
        self._memo_singleflight_followers = 0
        self._engine_builds = 0

    def _executor_locked(self) -> BoundedExecutor:
        if self._executor is None or bool(self._executor.status()['closed']):
            self._executor = BoundedExecutor(
                max_workers=self.max_workers,
                max_queue=self.max_queue,
                thread_name_prefix='trove-search',
                submit_timeout_seconds=self.submit_timeout_seconds,
            )
        return self._executor

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def invalidate(self, reason: str = 'mutation') -> dict:
        # Capture the post-mutation token under a shared lease.  Explicit API
        # invalidation therefore advances the process cache exactly once; the
        # next read observes the same token and cannot invalidate it again.
        with VaultGenerationLease(self.cfg, mode='read') as generation_lease:
            with self._lock:
                # Close old-generation read handles before capturing the
                # stable token.  Their final close may checkpoint/remove WAL
                # sidecars; capturing first would make that cleanup look like
                # a second generation on the next query.
                self._drop_engine_locked(reason, close_freshness=True)
                observed = self._freshness_token(generation_lease.refresh_token())
                self._observed_generation_token = observed
                return self.status()

    def _drop_engine_locked(self, reason: str, *, close_freshness: bool) -> None:
        self._generation += 1
        engine = self._engine
        self._engine = None
        if engine is not None:
            engine.store.close()
        if close_freshness and self._freshness_conn is not None:
            self._freshness_conn.close()
        self._provider = None
        self._vector = None
        self._status = None
        if close_freshness:
            self._freshness_conn = None
            self._freshness_sqlite_identity = None
        self._result_cache.clear()
        self._memo_cache.clear()
        self._last_invalidation_reason = reason
        del engine
        # sqlite3 may keep a closed handle alive until unreachable cursor
        # cycles are finalized.  Generation invalidation is infrequent and is
        # the correct deterministic boundary for reclaiming those descriptors.
        gc.collect()

    def _observe_generation_locked(self, generation_token: VaultGenerationToken) -> tuple[Any, ...]:
        observed = self._freshness_token(generation_token)
        if self._observed_generation_token is None:
            self._observed_generation_token = observed
        elif observed != self._observed_generation_token:
            # Keep the freshness connection that produced ``observed``.  A new
            # SQLite connection has a connection-local data_version baseline;
            # closing it here would make the same generation look new twice.
            self._drop_engine_locked('vault_generation_changed', close_freshness=False)
            self._observed_generation_token = observed
        return observed

    def close(self) -> None:
        executor: BoundedExecutor | None
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        self.release_resources()
        gc.collect()

    def release_resources(self) -> None:
        """Close cached handles without waiting for or destroying the worker pool.

        Mutation admission uses this before an atomic fixture claim.  The
        generation coordinator remains the source of truth for whether an
        active read permits publication; resource release must not wait behind
        that reader and accidentally turn a fast conflict into a long stall.
        """

        with self._lock:
            if self._engine is not None:
                self._engine.store.close()
            if self._freshness_conn is not None:
                self._freshness_conn.close()
            self._engine = None
            self._freshness_conn = None
            self._freshness_sqlite_identity = None
            self._provider = None
            self._vector = None
            self._status = None
            self._result_cache.clear()
            self._memo_cache.clear()

    def get(self) -> HyperSearch:
        """Return the current raw engine for diagnostics.

        Product reads should call :meth:`search`, which retains the generation
        lease for the whole logical query.  The raw object is kept for existing
        local diagnostics that inspect its readonly store.
        """

        with vault_generation_read(self.cfg) as generation_token:
            with self._lock:
                self._observe_generation_locked(generation_token)
                return self._get_locked()

    def _get_locked(self) -> HyperSearch:
        if self._engine is not None:
            return self._engine
        store = open_store(
            self.cfg.paths.sqlite_path,
            readonly=True,
            max_connections=self.max_workers + 1,
            prepared_statement_cache_size=128,
        )
        provider = self.provider_factory()
        registry = VectorBackendRegistry(store=store, zvec_path=zvec_collection_path_for_provider(self.cfg, provider), provider=provider)
        vector, status = registry.select('zvec')
        if _cloud_embedding_provider(provider) and status.state != 'available':
            local = configured_embedding_provider()
            local_registry = VectorBackendRegistry(store=store, zvec_path=zvec_collection_path(self.cfg), provider=local)
            local_vector, local_status = local_registry.select('zvec')
            if local_status.state == 'available':
                provider, vector, status = local, local_vector, local_status
        self._provider = provider
        self._vector = vector
        self._status = status.to_dict()
        episode_store = None
        selector = None
        if _cloud_embedding_provider(provider):
            from trove_core.search.episodes import BoundedEvidenceSelector, EpisodeZVecStore, episode_collection_path

            candidate = EpisodeZVecStore(episode_collection_path(self.cfg.paths.vector_dir), store=store)
            self._status['episodes'] = candidate.status(provider)
            if self._status['episodes']['state'] == 'available':
                episode_store = candidate
                selector = BoundedEvidenceSelector(self.cfg.root)
        self._engine = HyperSearch(
            store,
            vector_store=vector,
            embedding_provider=provider,
            vector_status=self._status,
            episode_store=episode_store,
            evidence_selector=selector,
        )
        self._engine_builds += 1
        # Daemon startup already has an explicit warmup.  Doing it again while
        # constructing every search runtime makes the first lexical-only query
        # load (or even fall back to loading) an embedding model it may never use.
        # Keep eager warmup as an explicit operator opt-in; semantic requests
        # otherwise initialize the vector path lazily on first use.
        if os.environ.get('TROVE_SEARCH_EAGER_WARMUP') == '1':
            self._status['warmup'] = warm_search_engine(self._engine)
        else:
            self._status['warmup'] = {
                'ok': True,
                'skipped': True,
                'reason_code': 'lazy_until_semantic_query',
            }
        return self._engine

    def search(self, request: SearchRequest) -> SearchResponse:
        response, _metrics = self.search_with_metrics(request)
        return response

    def search_with_metrics(self, request: SearchRequest) -> tuple[SearchResponse, dict[str, Any]]:
        """Run one bounded query and return content-free runtime metrics."""

        with self._lock:
            executor = self._executor_locked()
        future = executor.submit(self._search_once, request)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            # Python threads cannot be killed safely.  The worker retains its
            # generation lease and bounded slot until it reaches a safe return;
            # the caller gets a typed timeout without spawning replacement work.
            raise RuntimeTimedOut() from exc

    def _search_once(self, request: SearchRequest) -> tuple[SearchResponse, dict[str, Any]]:
        started = time.perf_counter()
        with vault_generation_read(self.cfg) as generation_token:
            leader = False
            with self._lock:
                freshness = self._observe_generation_locked(generation_token)
                generation = self._generation
                key = self._result_cache_key(request, generation, freshness)
                cached = self._result_cache.get(key)
                if cached is not None:
                    self._result_cache.move_to_end(key)
                    self._cache_hits += 1
                    return cached, {
                        'cache_hit': True,
                        'candidate_count': len(cached.results),
                        'duration_ms': round((time.perf_counter() - started) * 1000, 3),
                        'resource_count': self._resource_count_locked(),
                        'generation': generation,
                        'singleflight_shared': False,
                    }
                in_flight = self._result_inflight.get(key)
                if in_flight is None:
                    in_flight = Future()
                    self._result_inflight[key] = in_flight
                    self._cache_misses += 1
                    leader = True
                    try:
                        # Preserve the documented test/adapter seam where a
                        # runtime subclass supplies a synthetic engine through
                        # ``get``.
                        engine = self._get_locked() if type(self).get is SearchRuntimeCache.get else self.get()
                    except BaseException as exc:
                        self._result_inflight.pop(key, None)
                        in_flight.set_exception(exc)
                        raise
                else:
                    self._singleflight_followers += 1
            if not leader:
                response = in_flight.result()
                with self._lock:
                    resource_count = self._resource_count_locked()
                return response, {
                    'cache_hit': False,
                    'candidate_count': len(response.results),
                    'duration_ms': round((time.perf_counter() - started) * 1000, 3),
                    'resource_count': resource_count,
                    'generation': generation,
                    'singleflight_shared': True,
                }
            try:
                response = engine.search(request)
            except BaseException as exc:
                with self._lock:
                    current = self._result_inflight.get(key)
                    if current is in_flight:
                        in_flight.set_exception(exc)
                        self._result_inflight.pop(key, None)
                raise
            with self._lock:
                # Publication cannot change the token while this read lease is
                # active.  A same-process explicit invalidation can still bump
                # the runtime generation, so retain that guard.
                if generation == self._generation:
                    self._result_cache[key] = response
                    while len(self._result_cache) > self._result_cache_max:
                        self._result_cache.popitem(last=False)
                resource_count = self._resource_count_locked()
                current = self._result_inflight.get(key)
                if current is in_flight:
                    in_flight.set_result(response)
                    self._result_inflight.pop(key, None)
            return response, {
                'cache_hit': False,
                'candidate_count': len(response.results),
                'duration_ms': round((time.perf_counter() - started) * 1000, 3),
                'resource_count': resource_count,
                'generation': generation,
                'singleflight_shared': False,
            }

    def memoize_generation(
        self,
        namespace: str,
        key: tuple[Any, ...],
        loader: Callable[[], Any],
        *,
        cache_if: Callable[[Any], bool] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Memoize one derived read per immutable Vault generation.

        This is used for bounded cloud post-processing such as reranking.  It
        shares concurrent identical work and never publishes a value after a
        same-process runtime invalidation.  Callers decide which results are
        safe to retain, so transient cloud failures are not cached.
        """

        started = time.perf_counter()
        with vault_generation_read(self.cfg) as generation_token:
            leader = False
            with self._lock:
                freshness = self._observe_generation_locked(generation_token)
                generation = self._generation
                memo_key = (generation, freshness, str(namespace), key)
                if memo_key in self._memo_cache:
                    value = self._memo_cache[memo_key]
                    self._memo_cache.move_to_end(memo_key)
                    self._memo_cache_hits += 1
                    return value, {
                        'cache_hit': True,
                        'singleflight_shared': False,
                        'generation': generation,
                        'duration_ms': round((time.perf_counter() - started) * 1000, 3),
                    }
                in_flight = self._memo_inflight.get(memo_key)
                if in_flight is None:
                    in_flight = Future()
                    self._memo_inflight[memo_key] = in_flight
                    self._memo_cache_misses += 1
                    leader = True
                else:
                    self._memo_singleflight_followers += 1
            if not leader:
                value = in_flight.result()
                return value, {
                    'cache_hit': False,
                    'singleflight_shared': True,
                    'generation': generation,
                    'duration_ms': round((time.perf_counter() - started) * 1000, 3),
                }
            try:
                value = loader()
                retain = True if cache_if is None else bool(cache_if(value))
            except BaseException as exc:
                with self._lock:
                    current = self._memo_inflight.get(memo_key)
                    if current is in_flight:
                        in_flight.set_exception(exc)
                        self._memo_inflight.pop(memo_key, None)
                raise
            with self._lock:
                if retain and generation == self._generation:
                    self._memo_cache[memo_key] = value
                    while len(self._memo_cache) > self._memo_cache_max:
                        self._memo_cache.popitem(last=False)
                current = self._memo_inflight.get(memo_key)
                if current is in_flight:
                    in_flight.set_result(value)
                    self._memo_inflight.pop(memo_key, None)
            return value, {
                'cache_hit': False,
                'singleflight_shared': False,
                'generation': generation,
                'duration_ms': round((time.perf_counter() - started) * 1000, 3),
            }

    def _resource_count_locked(self) -> int:
        return sum(self._resource_counts_locked().values())

    def _resource_counts_locked(self) -> dict[str, int]:
        return {
            'engine_connections': int(getattr(self._engine.store, 'active_connection_count', 0) or 0) if self._engine is not None else 0,
            'freshness_connections': int(self._freshness_conn is not None),
            'providers': int(self._provider is not None),
            'vectors': int(self._vector is not None),
        }

    @staticmethod
    def _file_fingerprint(path: Path) -> tuple[Any, ...]:
        try:
            st = path.stat()
        except FileNotFoundError:
            return ('missing',)
        except OSError as exc:
            return ('error', exc.errno)
        return ('file', st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)

    def _sqlite_data_version(self, sqlite_fingerprint: tuple[Any, ...]) -> int | None:
        if not sqlite_fingerprint or sqlite_fingerprint[0] != 'file':
            if self._freshness_conn is not None:
                try:
                    self._freshness_conn.close()
                except Exception:
                    pass
            self._freshness_conn = None
            self._freshness_sqlite_identity = None
            return None
        identity = (int(sqlite_fingerprint[1]), int(sqlite_fingerprint[2]))
        if self._freshness_sqlite_identity != identity:
            if self._freshness_conn is not None:
                try:
                    self._freshness_conn.close()
                except Exception:
                    pass
            self._freshness_conn = None
            self._freshness_sqlite_identity = None
        try:
            if self._freshness_conn is None:
                readonly_uri = SQLiteStore(self.cfg.paths.sqlite_path, readonly=True)._readonly_uri()
                self._freshness_conn = sqlite3.connect(
                    readonly_uri,
                    uri=True,
                    timeout=1.0,
                    isolation_level=None,
                    check_same_thread=False,
                )
                self._freshness_sqlite_identity = identity
            row = self._freshness_conn.execute('PRAGMA data_version').fetchone()
            return int(row[0]) if row is not None else None
        except sqlite3.Error:
            if self._freshness_conn is not None:
                try:
                    self._freshness_conn.close()
                except Exception:
                    pass
            self._freshness_conn = None
            self._freshness_sqlite_identity = None
            return None

    def _freshness_token(self, generation_token: VaultGenerationToken | None = None) -> tuple[Any, ...]:
        sqlite_fingerprint = generation_token.sqlite if generation_token is not None else self._file_fingerprint(self.cfg.paths.sqlite_path)
        vector_metadata = Path(str(zvec_collection_path(self.cfg)) + '.trove-meta.json')
        cloud_vector_metadata = Path(str(self.cfg.paths.vector_dir / 'zvec' / 'messages-cloud') + '.trove-meta.json')
        return (
            'freshness-v2',
            generation_token.cache_key() if generation_token is not None else None,
            sqlite_fingerprint,
            self._sqlite_data_version(sqlite_fingerprint),
            generation_token.vector_metadata if generation_token is not None else self._file_fingerprint(vector_metadata),
            self._file_fingerprint(cloud_vector_metadata),
        )

    def _result_cache_key(self, request: SearchRequest, generation: int, freshness: tuple[Any, ...]) -> tuple[Any, ...]:
        return (
            generation,
            freshness,
            request.query,
            tuple(sorted(request.filters.items())),
            request.limit,
            request.include_vector,
            request.semantic,
            request.ranking_mode,
            request.reranker_mode,
            request.reranker_model_path,
            request.reranker_timeout_ms,
            request.retrieval_candidate_limit,
            request.fusion_candidate_limit,
            request.reranker_candidate_limit,
            request.expand_query,
            request.include_media_hints,
        )

    def status(self) -> dict:
        with self._lock:
            executor_status = self._executor.status() if self._executor is not None else {
                'active_workers': 0,
                'queued_workers': 0,
                'max_workers': self.max_workers,
                'max_queue': self.max_queue,
                'closed': True,
            }
            return {
                'generation': self._generation,
                'loaded': self._engine is not None,
                'last_invalidation_reason': self._last_invalidation_reason,
                'vector_state': (self._status or {}).get('state'),
                'selected_backend': (self._status or {}).get('selected_backend'),
                'result_cache_entries': len(self._result_cache),
                'result_cache_max': self._result_cache_max,
                'result_cache_bytes': self._result_cache.current_bytes,
                'result_cache_max_bytes': self._result_cache.max_bytes,
                'result_cache_evictions': self._result_cache.evictions,
                'result_inflight': len(self._result_inflight),
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'singleflight_followers': self._singleflight_followers,
                'memo_cache_entries': len(self._memo_cache),
                'memo_cache_max': self._memo_cache_max,
                'memo_cache_bytes': self._memo_cache.current_bytes,
                'memo_cache_max_bytes': self._memo_cache.max_bytes,
                'memo_cache_evictions': self._memo_cache.evictions,
                'memo_inflight': len(self._memo_inflight),
                'memo_cache_hits': self._memo_cache_hits,
                'memo_cache_misses': self._memo_cache_misses,
                'memo_singleflight_followers': self._memo_singleflight_followers,
                'engine_builds': self._engine_builds,
                'resource_count': self._resource_count_locked(),
                'resource_counts': self._resource_counts_locked(),
                'workers': executor_status,
                'timeout_seconds': self.timeout_seconds,
            }


class GenerationSafeSearchEngine:
    """Compatibility facade over a generation-leased process runtime."""

    def __init__(self, runtime: SearchRuntimeCache):
        self._runtime = runtime

    def search(self, request: SearchRequest) -> SearchResponse:
        return self._runtime.search(request)

    def close(self) -> None:
        self._runtime.close()

    def __getattr__(self, name: str):
        # Preserve readonly inspection used by benchmarks and migration tests.
        # Stateful product search always goes through the leased method above.
        return getattr(self._runtime.get(), name)


def vector_status_payload(cfg: VaultConfig, backend: str = 'zvec', provider=None) -> dict:
    provider = provider if provider is not None else configured_embedding_provider(vault_root=cfg.root)
    status = vector_registry(cfg, provider=provider).status(backend).to_dict()
    # Backward-compatible keys for existing clients.
    status['backend'] = backend
    status['sqlite_vector_entries'] = status['sqlite']['entries']
    status['embedding_provider_configured'] = provider is not None
    status['embedding_auto_discovered'] = bool(getattr(provider, 'auto_discovered', False)) if provider else False
    telemetry = getattr(provider, 'daemon_telemetry', None) if provider is not None else None
    status['embedding_daemon'] = telemetry() if callable(telemetry) else {
        'requests': 0,
        'hits': 0,
        'fallback_count': 0,
        'last_reason_code': None,
        'fallback_mode': 'exact_fts' if provider is None else None,
        'raw_content_included': False,
        'raw_paths_included': False,
        'secret_values_included': False,
    }
    zvec = status.get('zvec') or {}
    if _cloud_embedding_provider(provider):
        from trove_core.search.episodes import EpisodeZVecStore, episode_collection_path

        status['episodes'] = EpisodeZVecStore(
            episode_collection_path(cfg.paths.vector_dir)
        ).status(provider)
    if zvec.get('rebuild_required') or status.get('reason_code') == 'zvec_rebuild_required':
        status['conclusion'] = 'needs rebuild'
    elif zvec.get('catchup_pending') or status.get('reason_code') == 'zvec_catchup_pending':
        status['conclusion'] = 'catchup_pending'
    elif status.get('state') == 'available':
        status['conclusion'] = 'ready'
    else:
        status['conclusion'] = 'stale' if zvec.get('stale') else status.get('state', 'unavailable')
    return status


def vector_cloud_approval_payload(
    cfg: VaultConfig,
    provider,
    *,
    backend: str,
    batch_size: int,
    max_messages: int | None,
    purge: bool,
    citations=None,
) -> dict[str, Any]:
    payload, _source_snapshot = _vector_cloud_approval_payload_and_snapshot(
        cfg,
        provider,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
        purge=purge,
        citations=citations,
    )
    return payload


def _vector_source_snapshot(cfg: VaultConfig) -> str:
    """Return a constant-time token for the rows that feed vector documents.

    Source-table triggers advance ``vector_source_revision`` in the same
    transaction as message/evidence mutations.  Consequently, profile, media,
    trace, and vector-ledger commits do not discard an already-paid embedding
    batch.  File identity remains part of the token so an atomic database
    replacement still fails the CAS.

    The file-stat fallback is only for a pre-v25 database before its normal
    writable initialization installs the revision triggers.  It preserves the
    old fail-closed behavior without scanning source rows during migration.
    """

    path = cfg.paths.sqlite_path

    def identity() -> tuple[Any, ...]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return ('missing',)
        return (int(stat.st_dev), int(stat.st_ino))

    def legacy_token() -> str:
        parts: list[tuple[Any, ...]] = []
        for source_path in (path, Path(str(path) + '-wal')):
            try:
                stat = source_path.stat()
            except FileNotFoundError:
                parts.append(('missing',))
            else:
                parts.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256(repr(('legacy-file-state', tuple(parts))).encode('utf-8')).hexdigest()

    identity_before = identity()
    if identity_before == ('missing',):
        return legacy_token()
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            row = conn.execute(
                'SELECT value FROM schema_meta WHERE key=?',
                (VECTOR_SOURCE_REVISION_KEY,),
            ).fetchone()
            version_row = conn.execute('PRAGMA user_version').fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return legacy_token()
    if row is None:
        return legacy_token()
    try:
        revision = int(row[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError('invalid_vector_source_revision') from exc
    identity_after = identity()
    token = (
        'vector-source-revision-v1',
        identity_before,
        identity_after,
        int(version_row[0]) if version_row else 0,
        revision,
    )
    return hashlib.sha256(repr(token).encode('utf-8')).hexdigest()


def _read_vector_documents_readonly(
    path: Path,
    *,
    batch_size: int,
    citations: list[str] | None,
    max_messages: int | None,
) -> list[dict[str, Any]]:
    """Read the exact vector source without creating WAL sidecars."""

    if not path.is_file() or max_messages == 0:
        return []
    store = open_store(path, readonly=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_cursor(cursor) -> bool:
        while True:
            batch = cursor.fetchmany(max(1, int(batch_size)))
            if not batch:
                return False
            for row in batch:
                source = dict(row)
                citation = str(source['citation'])
                if citation in seen:
                    continue
                seen.add(citation)
                source['vector_text'] = vector_document_text(source)
                rows.append(source)
                if max_messages is not None and len(rows) >= max_messages:
                    cursor.close()
                    return True

    try:
        with store.connect() as conn:
            has_chunks = bool(
                store._table_exists(conn, 'evidence_chunks')
                and conn.execute(
                    "SELECT 1 FROM evidence_chunks WHERE status='active' LIMIT 1"
                ).fetchone() is not None
            )
            if citations is None:
                if has_chunks:
                    cursor = conn.execute(
                        """SELECT chunk_citation AS citation,parent_citation,account_id,account_label,
                                  source_id AS conversation_id,title AS conversation_title,
                                  'private' AS conversation_type,actor AS sender_id,actor AS sender_name,
                                  timestamp,content,source_type,'metadata' AS direction
                             FROM evidence_chunks WHERE status='active'
                            ORDER BY timestamp,chunk_citation"""
                    )
                else:
                    cursor = conn.execute('SELECT * FROM messages ORDER BY timestamp,citation')
                append_cursor(cursor)
            else:
                for start in range(0, len(citations), 400):
                    citation_batch = citations[start:start + 400]
                    if not citation_batch:
                        continue
                    placeholders = ','.join('?' for _ in citation_batch)
                    if has_chunks:
                        # Keep both predicates independently indexable.  The
                        # former OR shape made SQLite full-scan the complete
                        # evidence table once per 400 dirty citations, turning
                        # a 24k catch-up into tens of minutes of CPU before the
                        # first embedding request. ``seen`` deduplicates rows
                        # that match both the parent and chunk citation query.
                        for citation_column in ('parent_citation', 'chunk_citation'):
                            cursor = conn.execute(
                                f"""SELECT chunk_citation AS citation,parent_citation,account_id,account_label,
                                           source_id AS conversation_id,title AS conversation_title,
                                           'private' AS conversation_type,actor AS sender_id,actor AS sender_name,
                                           timestamp,content,source_type,'metadata' AS direction
                                      FROM evidence_chunks
                                     WHERE status='active'
                                       AND {citation_column} IN ({placeholders})
                                     ORDER BY timestamp,chunk_citation""",
                                citation_batch,
                            )
                            if append_cursor(cursor):
                                break
                        if max_messages is not None and len(rows) >= max_messages:
                            break
                    else:
                        cursor = conn.execute(
                            f'SELECT * FROM messages WHERE citation IN ({placeholders}) ORDER BY timestamp,citation',
                            citation_batch,
                        )
                    if not has_chunks and append_cursor(cursor):
                        break
    finally:
        store.close()
    return rows


def _iter_vector_documents_by_citation_readonly(
    path: Path,
    *,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    """Stream the full vector source in ledger-compatible citation order."""

    if not path.is_file():
        return
    store = open_store(path, readonly=True)
    conn = None
    try:
        conn = store.connect_once()
        has_chunks = bool(
            store._table_exists(conn, 'evidence_chunks')
            and conn.execute(
                "SELECT 1 FROM evidence_chunks WHERE status='active' LIMIT 1"
            ).fetchone() is not None
        )
        if has_chunks:
            cursor = conn.execute(
                """SELECT chunk_citation AS citation,parent_citation,account_id,account_label,
                          source_id AS conversation_id,title AS conversation_title,
                          'private' AS conversation_type,actor AS sender_id,actor AS sender_name,
                          timestamp,content,source_type,'metadata' AS direction
                     FROM evidence_chunks WHERE status='active'
                    ORDER BY chunk_citation"""
            )
        else:
            cursor = conn.execute('SELECT * FROM messages ORDER BY citation')
        while True:
            rows = cursor.fetchmany(max(1, int(batch_size)))
            if not rows:
                return
            for row in rows:
                source = dict(row)
                source['vector_text'] = vector_document_text(source)
                yield source
    finally:
        if conn is not None:
            conn.close()
        store.close()


def _reconcile_sorted_vector_documents(
    source_rows: Iterable[dict[str, Any]],
    ledger_entries: Iterable[tuple[str, str]],
    *,
    force_all: bool = False,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Merge two sorted streams without corpus-sized hash maps or sets."""

    changed: list[dict[str, Any]] = []
    deletes: list[str] = []
    source_count = 0
    previous_source = ''
    ledger_iterator = iter(ledger_entries)
    current_ledger = next(ledger_iterator, None)
    for source in source_rows:
        citation = str(source.get('citation') or '')
        if not citation or (previous_source and citation <= previous_source):
            raise RuntimeError('vector_source_citation_order_invalid')
        previous_source = citation
        source_count += 1
        text_value = str(source['vector_text'])
        content_hash = hashlib.sha256(text_value.encode('utf-8')).hexdigest()
        source['content_hash'] = content_hash
        while current_ledger is not None and current_ledger[0] < citation:
            deletes.append(str(current_ledger[0]))
            current_ledger = next(ledger_iterator, None)
        previous_hash = None
        if current_ledger is not None and current_ledger[0] == citation:
            previous_hash = str(current_ledger[1])
            current_ledger = next(ledger_iterator, None)
        if force_all or previous_hash != content_hash:
            changed.append(source)
    while current_ledger is not None:
        deletes.append(str(current_ledger[0]))
        current_ledger = next(ledger_iterator, None)
    return changed, deletes, source_count


def _vector_cloud_approval_payload_and_snapshot(
    cfg: VaultConfig,
    provider,
    *,
    backend: str,
    batch_size: int,
    max_messages: int | None,
    purge: bool,
    citations=None,
) -> tuple[dict[str, Any], str]:
    if type(backend) is not str or backend not in {'sqlite', 'zvec'}:
        raise ValueError('cloud vector backend must be exactly sqlite or zvec')
    if type(batch_size) is not int or batch_size <= 0 or batch_size > 100_000:
        raise ValueError('cloud vector batch_size must be an exact integer from 1 to 100000')
    if max_messages is not None and (
        type(max_messages) is not int or max_messages < 0 or max_messages > 1_000_000_000
    ):
        raise ValueError('cloud vector max_messages must be an exact non-negative integer or None')
    if type(purge) is not bool:
        raise TypeError('cloud vector purge must be an exact boolean')
    if citations is None:
        citation_filter = None
    else:
        if type(citations) not in {list, tuple}:
            raise TypeError('cloud vector citations must be an exact list or tuple')
        if len(citations) > 100_000:
            raise ValueError('cloud vector citations exceeds the control bound')
        if any(type(item) is not str or not item for item in citations):
            raise TypeError('cloud vector citations must contain exact non-empty strings')
        citation_filter = list(dict.fromkeys(citations))
    if purge and citation_filter is not None:
        raise ValueError('cloud vector purge cannot be combined with a citation subset')

    source_before = _vector_source_snapshot(cfg)
    hasher = hashlib.sha256()
    rows = _read_vector_documents_readonly(
        cfg.paths.sqlite_path,
        batch_size=batch_size,
        citations=citation_filter,
        max_messages=max_messages,
    )
    for row in rows:
        citation = row['citation']
        document = row['vector_text']
        if type(citation) is not str or not citation:
            raise RuntimeError('cloud vector input contains an invalid citation')
        if type(document) is not str:
            raise RuntimeError('cloud vector input contains invalid document text')
        for value in (citation, document):
            encoded = value.encode('utf-8')
            hasher.update(len(encoded).to_bytes(8, 'big'))
            hasher.update(encoded)

    source_after = _vector_source_snapshot(cfg)
    if source_before != source_after:
        raise RuntimeError('vector_source_changed_during_approval_digest')

    provider_name = getattr(provider, 'provider_name', getattr(provider, 'name', None))
    model = getattr(provider, 'model', None)
    dimensions = getattr(provider, 'dimensions', None)
    endpoint = getattr(provider, 'endpoint', None)
    payload = cloud_embedding_payload(
        operation='cloud_vector_rebuild' if purge else 'cloud_vector_index',
        provider=provider_name,
        model=model,
        dimensions=dimensions,
        endpoint=endpoint,
        input_digest=hasher.hexdigest(),
        item_count=len(rows),
    ) | {
        'backend': backend,
        'batch_size': batch_size,
        'max_messages': max_messages,
        'purge': purge,
    }
    return payload, source_after


def _require_cloud_vector_approval(
    cfg: VaultConfig,
    provider,
    *,
    backend: str,
    batch_size: int,
    max_messages: int | None,
    purge: bool,
    citations,
    approval_grant: ApprovalGrant | None,
    approval_payload: dict[str, Any] | None,
) -> str | None:
    egress_kind = getattr(provider, 'egress_kind', None)
    if egress_kind is None:
        return None
    if type(egress_kind) is not str or egress_kind != 'cloud_embedding_upload':
        raise RuntimeError('unsupported_embedding_egress_kind')
    from trove_core.providers.cloud_policy import cloud_retrieval_policy

    if cloud_retrieval_policy(cfg.root)['enabled']:
        # The operator has persisted continuous retrieval consent for this
        # private Vault. Destructive rebuild publication still has its own
        # vector-rebuild approval; this branch only authorizes bounded egress.
        return _vector_source_snapshot(cfg)
    expected, source_snapshot = _vector_cloud_approval_payload_and_snapshot(
        cfg,
        provider,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
        purge=purge,
        citations=citations,
    )
    if type(approval_payload) is not dict or approval_payload != expected:
        raise ApprovalValidationError(
            'cloud embedding approval payload does not match the outbound request',
            code='grant_payload_mismatch',
        )
    require_claimed_approval_grant(
        approval_grant,  # type: ignore[arg-type]
        cfg.root,
        action='cloud_vector_index',
        danger_class='cloud_embedding_upload',
        payload=expected,
    )
    return source_snapshot


def _require_vector_source_snapshot(cfg: VaultConfig, source_snapshot: str | None) -> None:
    """Cheap CAS under the writer lease; never repeat the full message digest."""

    if source_snapshot is None:
        return
    if _vector_source_snapshot(cfg) != source_snapshot:
        raise ApprovalValidationError(
            'cloud embedding source changed before the writer lease was acquired',
            code='vector_source_snapshot_changed',
        )


class VectorIndexSourceChanged(RuntimeError):
    code = 'vector_source_snapshot_changed'
    retryable = True


class VectorFullRebuildRequired(RuntimeError):
    code = 'vector_full_rebuild_required'
    retryable = False

    def __init__(self, reason_code: str):
        self.reason_code = str(reason_code or 'zvec_rebuild_required')
        super().__init__(self.reason_code)


def _raise_if_vector_source_changed(cfg: VaultConfig, source_snapshot: str) -> None:
    if _vector_source_snapshot(cfg) != source_snapshot:
        raise VectorIndexSourceChanged('vector source changed during lock-free preparation')


def _embed_many(provider, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embed_many = getattr(provider, 'embed_many', None)
    if callable(embed_many):
        vectors = embed_many(texts)
    else:
        vectors = [provider.embed(text) for text in texts]
    vectors = [[float(value) for value in vector] for vector in vectors]
    if len(vectors) != len(texts) or any(not vector for vector in vectors):
        raise RuntimeError('embedding provider returned an invalid batch')
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise RuntimeError('embedding provider returned inconsistent dimensions')
    if int(getattr(provider, 'dimensions', 0) or 0) <= 0:
        provider.dimensions = dimensions
    return vectors


class _BatchEmbeddingAdapter:
    def __init__(self, provider):
        self._provider = provider

    @property
    def dimensions(self) -> int:
        return int(getattr(self._provider, 'dimensions', 0) or 0)

    @dimensions.setter
    def dimensions(self, value: int) -> None:
        self._provider.dimensions = int(value)

    def __getattr__(self, name: str):
        return getattr(self._provider, name)

    def embed(self, text: str):
        return self._provider.embed(text)

    def embed_many(self, texts: list[str]):
        return _embed_many(self._provider, texts)


def _prepare_incremental_vectors(
    cfg: VaultConfig,
    store: SQLiteStore,
    provider,
    *,
    backend: str,
    citations: list[str] | None,
    batch_size: int,
    max_messages: int | None,
    force_all: bool = False,
) -> tuple[str, list[dict[str, Any]], list[str], int | None]:
    source_snapshot = _vector_source_snapshot(cfg)
    if not store.path.exists():
        return source_snapshot, [], [], 0
    source_rows: list[dict[str, Any]] = []
    existing_hashes: dict[str, str] = {}
    deletes: list[str] = []
    expected_count: int | None = None
    changed: list[dict[str, Any]] | None = None
    read_store = open_store(store.path, readonly=True)
    try:
        if backend == 'zvec':
            ledger_backend = zvec_ledger_backend(provider)
            vector = ZVecStore(
                zvec_collection_path_for_provider(cfg, provider),
                store=read_store,
                ledger_backend=ledger_backend,
            )
            metadata = vector._read_metadata()
            generation_id = str(metadata.get('generation_id') or '')
            ledger = VectorIndexLedger(read_store, backend=ledger_backend)
            generation = ledger.generation(generation_id) if generation_id else None
            if generation is None or generation.status != 'active':
                raise RuntimeError('ZVEC incremental indexing requires an active generation; run rebuild')
            if citations is None and max_messages is None:
                # Healthy full reconciliation is a sorted merge.  The old path
                # retained every source row, every ledger hash, and a full
                # citation set at once even when nothing changed.
                changed, deletes, expected_count = _reconcile_sorted_vector_documents(
                    _iter_vector_documents_by_citation_readonly(
                        store.path,
                        batch_size=batch_size,
                    ),
                    ledger.iter_entries(generation_id, batch_size=batch_size),
                    force_all=force_all,
                )
            else:
                source_rows = _read_vector_documents_readonly(
                    store.path,
                    batch_size=batch_size,
                    citations=citations,
                    max_messages=max_messages,
                )
                for source in source_rows:
                    text_value = str(source['vector_text'])
                    source['content_hash'] = hashlib.sha256(text_value.encode('utf-8')).hexdigest()
                existing_hashes = ledger.hashes(
                    generation_id,
                    [str(row['citation']) for row in source_rows],
                )
                if citations is not None:
                    previous = set(ledger.citations_for_dirty(generation_id, citations))
                    current = {str(row['citation']) for row in source_rows}
                    deletes = sorted(previous - current)
                expected_count = vector._expected_document_count(read_store)
        else:
            source_rows = _read_vector_documents_readonly(
                store.path,
                batch_size=batch_size,
                citations=citations,
                max_messages=max_messages,
            )
            for source in source_rows:
                text_value = str(source['vector_text'])
                source['content_hash'] = hashlib.sha256(text_value.encode('utf-8')).hexdigest()
        if backend == 'sqlite' and source_rows:
            with read_store.connect() as conn:
                for start in range(0, len(source_rows), 500):
                    batch = source_rows[start:start + 500]
                    if not batch:
                        continue
                    placeholders = ','.join('?' for _ in batch)
                    for row in conn.execute(
                        f'SELECT citation,content_hash FROM vector_entries WHERE citation IN ({placeholders})',
                        [str(item['citation']) for item in batch],
                    ):
                        existing_hashes[str(row['citation'])] = str(row['content_hash'])
    finally:
        read_store.close()

    if changed is None:
        changed = [
            row for row in source_rows
            if force_all or existing_hashes.get(str(row['citation'])) != str(row['content_hash'])
        ]
    for start in range(0, len(changed), max(1, int(batch_size))):
        batch = changed[start:start + max(1, int(batch_size))]
        if bool(getattr(provider, 'supports_sparse', False)):
            embeddings = provider.embed_hybrid_many(
                [str(row['vector_text']) for row in batch], text_type='document'
            )
            for row, embedding in zip(batch, embeddings):
                row['vector'] = [float(value) for value in embedding.dense]
                row['sparse_vector'] = {
                    int(index): float(value) for index, value in embedding.sparse.items()
                }
        else:
            vectors = _embed_many(provider, [str(row['vector_text']) for row in batch])
            for row, vector in zip(batch, vectors):
                row['vector'] = vector
    _raise_if_vector_source_changed(cfg, source_snapshot)
    return source_snapshot, changed, deletes, expected_count


def _import_staged_zvec_ledger(
    live_store: SQLiteStore,
    stage_store: SQLiteStore,
    *,
    ledger_backend: str = 'zvec',
    validate: Callable[[], None] | None = None,
) -> str | None:
    with stage_store.connect() as stage_conn:
        row = stage_conn.execute(
            "SELECT generation_id FROM vector_index_generations WHERE backend=? AND status='ready' LIMIT 1",
            (ledger_backend,),
        ).fetchone()
    if row is None:
        return None
    if validate is not None:
        validate()
    generation_id = str(row['generation_id'])
    with live_store.connect() as conn:
        conn.execute('ATTACH DATABASE ? AS vector_stage', (str(stage_store.path),))
        try:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                "DELETE FROM vector_index_ledger WHERE backend=? AND generation_id=?",
                (ledger_backend, generation_id),
            )
            conn.execute(
                "DELETE FROM vector_index_generations WHERE backend=? AND generation_id=? AND status<>'active'",
                (ledger_backend, generation_id),
            )
            conn.execute(
                """INSERT INTO vector_index_generations
                   SELECT * FROM vector_stage.vector_index_generations
                    WHERE backend=? AND generation_id=?""",
                (ledger_backend, generation_id),
            )
            conn.execute(
                """INSERT INTO vector_index_ledger
                   SELECT * FROM vector_stage.vector_index_ledger
                    WHERE backend=? AND generation_id=?""",
                (ledger_backend, generation_id),
            )
            conn.commit()
        finally:
            conn.execute('DETACH DATABASE vector_stage')
    return generation_id


def _zvec_rebuild_two_phase(
    cfg: VaultConfig,
    provider,
    *,
    operation: str,
    batch_size: int,
    max_messages: int | None,
    source_snapshot: str | None,
) -> tuple[int, ZVecStore, int]:
    store = SQLiteStore(cfg.paths.sqlite_path)
    ledger_backend = zvec_ledger_backend(provider)
    vector = ZVecStore(
        zvec_collection_path_for_provider(cfg, provider),
        store=store,
        ledger_backend=ledger_backend,
    )
    snapshot = source_snapshot or _vector_source_snapshot(cfg)
    _raise_if_vector_source_changed(cfg, snapshot)
    source_store = open_store(cfg.paths.sqlite_path, readonly=True)
    stage_path = cfg.paths.vector_dir / f'.zvec-ledger-stage-{uuid.uuid4().hex}.sqlite'
    stage_store = SQLiteStore(stage_path)
    stage_store.initialize()
    publication_started = False
    publication_attempted = False
    published_generation_id: str | None = None
    dirty_cleared = 0

    from contextlib import contextmanager

    @contextmanager
    def publish():
        nonlocal publication_attempted, publication_started, published_generation_id, dirty_cleared
        with coordinated_vault_mutation(cfg, operation=operation):
            if not publication_started:
                with stage_store.connect() as stage_conn:
                    ready = stage_conn.execute(
                        "SELECT 1 FROM vector_index_generations WHERE backend=? AND status='ready' LIMIT 1",
                        (ledger_backend,),
                    ).fetchone()
                if ready is not None and not publication_attempted:
                    publication_attempted = True
                    generation_id = _import_staged_zvec_ledger(
                        store,
                        stage_store,
                        ledger_backend=ledger_backend,
                        validate=lambda: _raise_if_vector_source_changed(cfg, snapshot),
                    )
                    if generation_id is not None:
                        # The ATTACH import above is the final SQLite publication
                        # preparation; the existing atomic swap activates it.
                        publication_started = True
                        published_generation_id = generation_id
            with coordinated_vault_generation_publish(cfg, operation=operation):
                yield
            active_generation = VectorIndexLedger(store, backend=ledger_backend).active_generation()
            final_metadata = vector._read_metadata()
            published_current = bool(
                published_generation_id
                and active_generation is not None
                and active_generation.generation_id == published_generation_id
                and final_metadata.get('generation_id') == published_generation_id
                and final_metadata.get('complete') is True
            )
            if published_current:
                # The source CAS was checked while this writer was held and the
                # new generation now covers the complete SQLite corpus. Clear
                # the old dirty backlog once, so the next sync starts from true
                # post-rebuild changes instead of replaying every historical row.
                dirty_cleared = clear_all_dirty_citations(store)

    try:
        build_provider = provider if callable(getattr(provider, 'embed_many', None)) else _BatchEmbeddingAdapter(provider)
        indexed = vector.atomic_rebuild(
            build_provider,
            store=source_store,
            ledger_store=stage_store,
            batch_size=batch_size,
            max_messages=max_messages,
            generation_publish=publish,
        )
        return indexed, vector, dirty_cleared
    finally:
        source_store.close_all()
        stage_store.close_all()
        for path in (stage_path, Path(str(stage_path) + '-wal'), Path(str(stage_path) + '-shm')):
            path.unlink(missing_ok=True)


def _rebuild_cloud_episode_vectors(cfg: VaultConfig, provider, *, source_snapshot: str | None = None) -> dict[str, Any]:
    """Build/recover the cloud episode side collection without rebuilding messages."""

    from trove_core.providers.cloud_policy import cloud_retrieval_policy
    from trove_core.search.episodes import EpisodeZVecStore, episode_collection_path

    if not _cloud_embedding_provider(provider):
        raise RuntimeError('episode_cloud_embedding_provider_required')
    if not cloud_retrieval_policy(cfg.root)['enabled']:
        raise RuntimeError('cloud_retrieval_policy_disabled')
    snapshot = source_snapshot or _vector_source_snapshot(cfg)
    if hasattr(provider, 'max_workers'):
        provider.max_workers = min(int(provider.max_workers), 12)
    return EpisodeZVecStore(
        episode_collection_path(cfg.paths.vector_dir),
        store=SQLiteStore(cfg.paths.sqlite_path, readonly=True),
    ).rebuild(provider, cfg=cfg, source_snapshot=snapshot)


@mutation_entrypoint('vector_rebuild')
def rebuild_vectors_atomic(
    cfg: VaultConfig,
    provider,
    *,
    backend: str = 'zvec',
    batch_size: int = 256,
    max_messages: int | None = None,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict[str, Any] | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict:
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('vector rebuild preparation cannot run inside an outer writer session')
    if backend != 'zvec':
        # Record the public rebuild boundary without wrapping embedding work.
        with coordinated_vault_mutation(cfg, operation='vector_rebuild'):
            pass
        return index_vectors(
            cfg,
            provider,
            backend=backend,
            batch_size=batch_size,
            max_messages=max_messages,
            purge=True,
            approval_grant=approval_grant,
            approval_payload=approval_payload,
        )
    # Validate an outbound cloud request before creating any lock artifacts.
    source_snapshot = _require_cloud_vector_approval(
        cfg,
        provider,
        backend=backend,
        batch_size=batch_size,
        max_messages=max_messages,
        purge=True,
        citations=None,
        approval_grant=approval_grant,
        approval_payload=approval_payload,
    )
    indexed, vector, dirty_cleared = _zvec_rebuild_two_phase(
        cfg,
        provider,
        operation='vector_rebuild',
        batch_size=batch_size,
        max_messages=max_messages,
        source_snapshot=source_snapshot,
    )
    episode_status = None
    if _cloud_embedding_provider(provider):
        episode_status = _rebuild_cloud_episode_vectors(
            cfg,
            provider,
            source_snapshot=source_snapshot,
        )
    return {
        'ok': True,
        'backend': 'zvec',
        'indexed': indexed,
        'atomic': True,
        'dirty_cleared': dirty_cleared,
        'collection': 'messages',
        'vector': vector.status(provider=provider),
        'episodes': episode_status,
    }


def _commit_sqlite_vector_delta(
    store: SQLiteStore,
    provider,
    *,
    rows: list[dict[str, Any]],
    purge: bool,
) -> int:
    """Publish precomputed vectors in one short SQLite transaction."""

    store.initialize()
    provider_name = str(getattr(provider, 'name', provider.__class__.__name__))
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if purge:
            conn.execute('DELETE FROM vector_entries')
        if rows:
            conn.executemany(
                """INSERT OR REPLACE INTO vector_entries(
                       citation,provider,dimensions,vector_json,content_hash
                   ) VALUES(?,?,?,?,?)""",
                [
                    (
                        str(row['citation']),
                        provider_name,
                        len(row['vector']),
                        json.dumps(row['vector']),
                        str(row['content_hash']),
                    )
                    for row in rows
                ],
            )
        conn.commit()
    return len(rows)


@mutation_entrypoint('vector_index')
def index_vectors(
    cfg: VaultConfig,
    provider,
    *,
    backend: str = 'zvec',
    batch_size: int = 256,
    max_messages: int | None = None,
    purge: bool = False,
    use_lock: bool = True,
    citations=None,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict[str, Any] | None = None,
    write_session: VaultWriteSession | None = None,
    _approval_already_validated: bool = False,
    _approved_source_snapshot: str | None = None,
) -> dict:
    # ``use_lock`` is retained for source compatibility only. False no longer
    # disables coordination.
    _ = use_lock
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('vector preparation cannot run inside an outer writer session')
    if _approval_already_validated:
        # Private compatibility arguments are intentionally not an authority
        # bypass. All public calls validate their own exact outbound payload.
        raise TypeError('prevalidated vector approval is not accepted by index_vectors')
    else:
        source_snapshot = _require_cloud_vector_approval(
            cfg,
            provider,
            backend=backend,
            batch_size=batch_size,
            max_messages=max_messages,
            purge=purge,
            citations=citations,
            approval_grant=approval_grant,
            approval_payload=approval_payload,
        )
    if source_snapshot is not None:
        # One constant-time preflight prevents an approved cloud upload from
        # starting after another writer changed its exact source. Embedding and
        # every source scan still run after this lease is released.
        with coordinated_vault_mutation(cfg, operation='vector_index'):
            _require_vector_source_snapshot(cfg, source_snapshot)

    store = SQLiteStore(cfg.paths.sqlite_path)
    trace = TraceTimeline(cfg.root)
    citation_filter = None if citations is None else list(
        dict.fromkeys(str(c) for c in citations if c)
    )
    if purge and citation_filter is not None:
        raise ValueError('purge cannot be combined with a citation subset')
    span_id = trace.start(
        'vector_index',
        {
            'backend': backend,
            'purge': bool(purge),
            'max_messages': max_messages,
            'dirty_count': len(citation_filter) if citation_filter is not None else None,
        },
    )
    try:
        if backend == 'zvec':
            episode_delta = None
            current = ZVecStore(
                zvec_collection_path_for_provider(cfg, provider),
                store=store,
                ledger_backend=zvec_ledger_backend(provider),
            )
            current_status = current.status(provider=provider)
            incremental_rebuild_required = bool(
                citation_filter is not None
                and (
                    current_status.get('collection_exists') is not True
                    or current_status.get('rebuild_required')
                    or current_status.get('provider_mismatch')
                    or current_status.get('incomplete')
                )
            )
            if incremental_rebuild_required:
                # Do not scan or embed a dirty batch that cannot be committed
                # into the current generation. Keep the journal intact and ask
                # for the one explicit full rebuild instead.
                raise VectorFullRebuildRequired(
                    str(current_status.get('reason_code') or 'zvec_rebuild_required')
                )
            rebuild_required = bool(
                purge
                or citation_filter is None and (
                    current_status.get('collection_exists') is not True
                    or current_status.get('rebuild_required')
                    or current_status.get('provider_mismatch')
                    or current_status.get('incomplete')
                )
            )
            if rebuild_required:
                indexed, vector, dirty_cleared = _zvec_rebuild_two_phase(
                    cfg,
                    provider,
                    operation='vector_index',
                    batch_size=batch_size,
                    max_messages=max_messages,
                    source_snapshot=source_snapshot,
                )
            else:
                dirty_cleared = 0
                prepared_snapshot, rows, deletes, expected_count = _prepare_incremental_vectors(
                    cfg,
                    store,
                    provider,
                    backend='zvec',
                    citations=citation_filter,
                    batch_size=batch_size,
                    max_messages=max_messages,
                )
                with coordinated_vault_mutation(cfg, operation='vector_index'):
                    _raise_if_vector_source_changed(cfg, prepared_snapshot)
                    with coordinated_vault_generation_publish(cfg, operation='vector-index'):
                        delta = current.apply_precomputed_delta(
                            provider,
                            rows=rows,
                            deletes=deletes,
                            expected_count=expected_count,
                        )
                indexed = int(delta['indexed'])
                vector = current
                if _cloud_embedding_provider(provider) and citation_filter:
                    from trove_core.search.episodes import EpisodeZVecStore, episode_collection_path

                    episode_delta = EpisodeZVecStore(
                        episode_collection_path(cfg.paths.vector_dir),
                        store=store,
                    ).sync_dirty_conversations(
                        provider,
                        cfg=cfg,
                        citations=citation_filter,
                    )
            report = {
                'ok': True,
                'backend': 'zvec',
                'indexed': indexed,
                'dirty_count': len(citation_filter) if citation_filter is not None else None,
                'dirty_cleared': dirty_cleared,
                'dimensions': int(getattr(provider, 'dimensions', 0) or 0),
                'vector': vector.status(provider=provider),
                'episodes': episode_delta,
                'trace_id': span_id,
            }
        elif backend == 'sqlite':
            prepared_snapshot, rows, _deletes, _expected_count = _prepare_incremental_vectors(
                cfg,
                store,
                provider,
                backend='sqlite',
                citations=citation_filter,
                batch_size=batch_size,
                max_messages=max_messages,
                force_all=purge,
            )
            with coordinated_vault_mutation(cfg, operation='vector_index'):
                _raise_if_vector_source_changed(cfg, prepared_snapshot)
                with coordinated_vault_generation_publish(cfg, operation='vector-index'):
                    indexed = _commit_sqlite_vector_delta(
                        store,
                        provider,
                        rows=rows,
                        purge=purge,
                    )
            report = {
                'ok': True,
                'backend': 'sqlite',
                'indexed': indexed,
                'dirty_count': len(citation_filter) if citation_filter is not None else None,
                'dimensions': int(getattr(provider, 'dimensions', 0) or 0),
                'sqlite_vector_entries': vector_entries_count(store),
                'trace_id': span_id,
            }
        else:
            raise ValueError(f'unsupported vector backend: {backend}')
        trace.complete(span_id, {'indexed': report.get('indexed'), 'backend': backend})
        return report
    except Exception as exc:
        trace.fail(span_id, {'error_code': exc.__class__.__name__})
        raise
