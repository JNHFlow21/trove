from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import OrderedDict
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable

from trove_core.bounds import BoundedLimit, RERANK_CANDIDATES
from trove_core.embedding.daemon_protocol import DaemonIdentity, identity_for_model
from trove_core.store.sqlite_store import vector_document_text

from .fusion import RankedRow

ModelFactory = Callable[[Path], Any]

_EXECUTOR_WORKERS = 2
_RUNTIME_CACHE_MAX = 2
_IDENTITY_CACHE_MAX = 4
_IDENTITY_CACHE_TTL_SECONDS = 30.0
_MAX_QUERY_CHARACTERS = 4096
_MAX_DOCUMENT_CHARACTERS = 4096
_EXECUTOR = ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS, thread_name_prefix='trove-local-rerank')
_EXECUTOR_SLOTS = threading.BoundedSemaphore(_EXECUTOR_WORKERS)
_RUNTIME_CACHE: OrderedDict[str, '_CrossEncoderRuntime'] = OrderedDict()
_RUNTIME_CACHE_LOCK = threading.Lock()
_IDENTITY_CACHE: OrderedDict[str, '_IdentityCacheEntry'] = OrderedDict()
_IDENTITY_CACHE_LOCK = threading.Lock()

_MODEL_FINGERPRINT_FILES = (
    'trove_model_manifest.json',
    'config.json',
    'modules.json',
    'model.safetensors',
    'model.safetensors.index.json',
    'pytorch_model.bin',
    'pytorch_model.bin.index.json',
    'tokenizer.json',
    'tokenizer_config.json',
)


class _LocalModelLoadFailed(RuntimeError):
    pass


class _LocalModelInferenceFailed(RuntimeError):
    pass


class _LocalModelIdentityUnstable(RuntimeError):
    pass


@dataclass(frozen=True)
class _IdentityCacheEntry:
    identity: DaemonIdentity
    fingerprint: tuple[Any, ...]
    expires_at: float


def _stat_fingerprint(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))


def _model_generation_fingerprint(path: Path) -> tuple[Any, ...]:
    """Cheap bounded invalidation signal for an otherwise immutable model."""

    return (
        _stat_fingerprint(path),
        *(
            (name, _stat_fingerprint(path / name))
            for name in _MODEL_FINGERPRINT_FILES
        ),
    )


def _cached_identity_for_model(path: Path) -> DaemonIdentity:
    """Singleflight the expensive bounded model walk and invalidate safely.

    Critical root/model metadata changes invalidate immediately.  The short TTL
    revalidates nested shard metadata while a loaded runtime continues to
    truthfully report the immutable identity it was constructed with.
    """

    key = str(path.resolve())
    with _IDENTITY_CACHE_LOCK:
        now = time.monotonic()
        fingerprint = _model_generation_fingerprint(path)
        cached = _IDENTITY_CACHE.get(key)
        if cached is not None and cached.fingerprint == fingerprint and now < cached.expires_at:
            _IDENTITY_CACHE.move_to_end(key)
            return cached.identity
        for _attempt in range(2):
            before = _model_generation_fingerprint(path)
            identity = identity_for_model(path)
            after = _model_generation_fingerprint(path)
            if before == after:
                _IDENTITY_CACHE[key] = _IdentityCacheEntry(
                    identity=identity,
                    fingerprint=after,
                    expires_at=time.monotonic() + _IDENTITY_CACHE_TTL_SECONDS,
                )
                _IDENTITY_CACHE.move_to_end(key)
                while len(_IDENTITY_CACHE) > _IDENTITY_CACHE_MAX:
                    _IDENTITY_CACHE.popitem(last=False)
                return identity
        raise _LocalModelIdentityUnstable()


def _default_model_factory(path: Path) -> Any:
    from sentence_transformers import CrossEncoder  # type: ignore

    device = 'cpu'
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            device = 'mps'
    except Exception:
        pass

    return CrossEncoder(
        str(path),
        device=device,
        trust_remote_code=False,
        local_files_only=True,
    )


def _score_value(value: Any) -> float:
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            raise _LocalModelInferenceFailed()
        # Binary classifiers commonly expose [negative, positive] logits.
        value = value[-1]
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise _LocalModelInferenceFailed() from exc
    if not math.isfinite(score):
        raise _LocalModelInferenceFailed()
    return score


class _CrossEncoderRuntime:
    def __init__(self, path: Path, factory: ModelFactory) -> None:
        self._path = path
        self._factory = factory
        self._model: Any | None = None
        self._lock = threading.Lock()
        self.load_count = 0
        self.invocation_count = 0

    def predict(self, pairs: list[tuple[str, str]]) -> tuple[list[float], int, int]:
        with self._lock:
            if self._model is None:
                try:
                    self._model = self._factory(self._path)
                except Exception as exc:
                    raise _LocalModelLoadFailed() from exc
                self.load_count += 1
            self.invocation_count += 1
            invocation_count = self.invocation_count
            try:
                raw_scores = self._model.predict(
                    pairs,
                    batch_size=min(32, max(1, len(pairs))),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                scores = [_score_value(value) for value in raw_scores]
            except _LocalModelInferenceFailed:
                raise
            except Exception as exc:
                raise _LocalModelInferenceFailed() from exc
            if len(scores) != len(pairs):
                raise _LocalModelInferenceFailed()
            return scores, invocation_count, self.load_count


def _prepare_and_predict(
    runtime: _CrossEncoderRuntime,
    candidates: list[RankedRow],
    query: str,
) -> tuple[list[float], int, int, bool]:
    bounded_query = query[:_MAX_QUERY_CHARACTERS]
    input_truncated = len(bounded_query) != len(query)
    documents: list[str] = []
    for row, _paths, _score in candidates:
        raw_document = vector_document_text(row)
        if len(raw_document) > _MAX_DOCUMENT_CHARACTERS:
            input_truncated = True
        documents.append(raw_document[:_MAX_DOCUMENT_CHARACTERS])
    pairs = [(bounded_query, document) for document in documents]
    scores, invocation_count, load_count = runtime.predict(pairs)
    return scores, invocation_count, load_count, input_truncated


def _runtime_for_model(path: Path, identity: DaemonIdentity, factory: ModelFactory | None) -> _CrossEncoderRuntime:
    if factory is not None:
        return _CrossEncoderRuntime(path, factory)
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(identity.model_hash)
        if runtime is None:
            runtime = _CrossEncoderRuntime(path, _default_model_factory)
            _RUNTIME_CACHE[identity.model_hash] = runtime
            while len(_RUNTIME_CACHE) > _RUNTIME_CACHE_MAX:
                _RUNTIME_CACHE.popitem(last=False)
        else:
            _RUNTIME_CACHE.move_to_end(identity.model_hash)
        return runtime


def _identity_status(identity: DaemonIdentity) -> dict[str, Any]:
    return {
        'provider': identity.provider,
        'model_id': identity.model_id,
        'model_hash': identity.model_hash,
    }


def _fallback(
    reason_code: str,
    *,
    elapsed_ms: float = 0.0,
    invoked: bool = False,
    candidate_count: int = 0,
    identity: DaemonIdentity | None = None,
    phase: str | None = None,
    task_submitted: bool = False,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        'state': 'degraded' if invoked else 'unavailable_fallback',
        'mode': 'local-bge',
        'reason_code': reason_code,
        'fallback_mode': 'features',
        'invoked': invoked,
        'task_submitted': task_submitted,
        'candidate_count': candidate_count,
        'returned_count': 0,
        'elapsed_ms': round(elapsed_ms, 3),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
    if identity is not None:
        status['identity'] = _identity_status(identity)
    if phase:
        status['phase'] = phase
    return status


def _release_slot(_future: Future[Any]) -> None:
    _EXECUTOR_SLOTS.release()


def _remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.perf_counter())


def _submit_bounded(fn: Callable[..., Any], *args: Any) -> Future[Any] | None:
    if not _EXECUTOR_SLOTS.acquire(blocking=False):
        return None
    try:
        return _EXECUTOR.submit(fn, *args)
    except Exception:
        _EXECUTOR_SLOTS.release()
        raise


def rerank_with_local_model(
    ranked: list[RankedRow],
    query: str,
    *,
    model_path: str | None,
    timeout_ms: int,
    limit: int,
    model_factory: ModelFactory | None = None,
) -> tuple[list[RankedRow], dict[str, Any]]:
    """Rerank a bounded semantic candidate window with a real local model.

    The path must already exist locally; ``local_files_only`` and
    ``trust_remote_code=False`` prevent implicit downloads or remote code.  A
    small bounded executor makes the timeout an actual request latency bound.
    """

    start = time.perf_counter()
    limit = BoundedLimit(limit, field='reranker_candidate_limit', spec=RERANK_CANDIDATES)
    if type(timeout_ms) is not int or timeout_ms < 1:
        raise ValueError('reranker_timeout_ms must be >= 1')
    deadline = start + (timeout_ms / 1000.0)
    if not ranked:
        return [], _fallback('local_reranker_no_candidates', elapsed_ms=(time.perf_counter() - start) * 1000)
    if not model_path:
        return ranked[:limit], _fallback('local_reranker_model_missing', elapsed_ms=(time.perf_counter() - start) * 1000)
    if model_path == '__mock__':
        return ranked[:limit], _fallback('local_reranker_mock_forbidden', elapsed_ms=(time.perf_counter() - start) * 1000)
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return ranked[:limit], _fallback('local_reranker_model_missing', elapsed_ms=(time.perf_counter() - start) * 1000)
    if model_factory is None and importlib.util.find_spec('sentence_transformers') is None:
        return ranked[:limit], _fallback('local_reranker_dependency_missing', elapsed_ms=(time.perf_counter() - start) * 1000)

    candidates = ranked[:limit]
    try:
        identity_future = _submit_bounded(_cached_identity_for_model, path)
    except Exception:
        return ranked[:limit], _fallback(
            'local_reranker_executor_failed',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            phase='identity',
        )
    if identity_future is None:
        return ranked[:limit], _fallback(
            'local_reranker_saturated',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            phase='identity',
        )
    release_identity_now = True
    try:
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise FutureTimeoutError()
        identity = identity_future.result(timeout=remaining)
    except FutureTimeoutError:
        release_identity_now = False
        identity_future.add_done_callback(_release_slot)
        return ranked[:limit], _fallback(
            'local_reranker_timeout',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            phase='identity',
            task_submitted=True,
        )
    except _LocalModelIdentityUnstable:
        return ranked[:limit], _fallback(
            'local_reranker_identity_unstable',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            phase='identity',
            task_submitted=True,
        )
    except Exception:
        return ranked[:limit], _fallback(
            'local_reranker_identity_failed',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            phase='identity',
            task_submitted=True,
        )
    finally:
        if release_identity_now:
            _EXECUTOR_SLOTS.release()

    if _remaining_seconds(deadline) <= 0:
        return ranked[:limit], _fallback(
            'local_reranker_timeout',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            identity=identity,
            phase='identity',
        )

    runtime = _runtime_for_model(path, identity, model_factory)
    invocation_count_before = runtime.invocation_count
    try:
        future = _submit_bounded(_prepare_and_predict, runtime, candidates, query)
    except Exception:
        return ranked[:limit], _fallback(
            'local_reranker_executor_failed',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            identity=identity,
            phase='inference',
        )
    if future is None:
        return ranked[:limit], _fallback(
            'local_reranker_saturated',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            candidate_count=len(candidates),
            identity=identity,
            phase='inference',
        )
    release_inference_now = True
    try:
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise FutureTimeoutError()
        scores, invocation_count, load_count, input_truncated = future.result(timeout=remaining)
    except FutureTimeoutError:
        release_inference_now = False
        future.add_done_callback(_release_slot)
        invoked = runtime.invocation_count > invocation_count_before
        return ranked[:limit], _fallback(
            'local_reranker_timeout',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            invoked=invoked,
            candidate_count=len(candidates),
            identity=identity,
            phase='inference',
            task_submitted=True,
        )
    except _LocalModelLoadFailed:
        return ranked[:limit], _fallback(
            'local_reranker_load_failed',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            invoked=runtime.invocation_count > invocation_count_before,
            candidate_count=len(candidates),
            identity=identity,
            phase='inference',
            task_submitted=True,
        )
    except _LocalModelInferenceFailed:
        return ranked[:limit], _fallback(
            'local_reranker_inference_failed',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            invoked=True,
            candidate_count=len(candidates),
            identity=identity,
            phase='inference',
            task_submitted=True,
        )
    except Exception:
        return ranked[:limit], _fallback(
            'local_reranker_failed',
            elapsed_ms=(time.perf_counter() - start) * 1000,
            invoked=runtime.invocation_count > invocation_count_before,
            candidate_count=len(candidates),
            identity=identity,
            phase='inference',
            task_submitted=True,
        )
    finally:
        if release_inference_now:
            _EXECUTOR_SLOTS.release()

    ordered = [
        (item[0], item[1], float(score))
        for score, _index, item in sorted(
            ((scores[index], index, item) for index, item in enumerate(candidates)),
            key=lambda value: (-value[0], value[1]),
        )
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ordered, {
        'state': 'available',
        'mode': 'local-bge',
        'identity': _identity_status(identity),
        'model_configured': True,
        'invoked': True,
        'task_submitted': True,
        'completed': True,
        'invocation_count': invocation_count,
        'load_count': load_count,
        'candidate_count': len(candidates),
        'returned_count': len(ordered),
        'input_truncated': input_truncated,
        'phase': 'complete',
        'elapsed_ms': round(elapsed_ms, 3),
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def warm_local_reranker(model_path: str, *, timeout_ms: int = 120_000) -> dict[str, Any]:
    """Load and invoke a local reranker with static non-private evidence."""

    synthetic_row = {
        'citation': 'synthetic-reranker-warmup',
        'content': 'synthetic local reranker warmup evidence',
        'content_kind': 'text',
        'source_type': 'synthetic',
        'conversation_title': 'Synthetic',
        'conversation_type': 'private',
        'sender_name': 'Synthetic',
        'direction': 'metadata',
        'timestamp': '1970-01-01T00:00:00Z',
    }
    _rows, status = rerank_with_local_model(
        [(synthetic_row, ['vector'], 1.0)],  # type: ignore[list-item]
        'synthetic local reranker warmup query',
        model_path=model_path,
        timeout_ms=timeout_ms,
        limit=1,
    )
    return {
        **status,
        'ok': status.get('state') == 'available' and bool(status.get('invoked')),
        'private_text_used': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }
