from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import threading
from typing import Any, Callable

from .base import EmbeddingProvider
from .daemon_client import EmbeddingDaemonClient
from .daemon_protocol import DaemonProtocolError, identity_for_model

DEFAULT_EMBED_SOCKET = '/tmp/trove-embed.sock'


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local-only sentence-transformers provider with strict daemon identity."""

    name = 'local-sentence-transformers'

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        normalize_embeddings: bool = True,
        model_factory: Callable[..., Any] | None = None,
        dimensions: int = 0,
        use_daemon: bool = True,
        daemon_timeout_ms: int | None = None,
    ):
        if not model_path:
            raise RuntimeError('Local embeddings require an explicit local model path; no default model is bundled.')
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise RuntimeError('Local embedding model path does not exist.')
        self.normalize_embeddings = normalize_embeddings
        self.use_daemon = bool(use_daemon and model_factory is None)
        self.socket_path = os.environ.get('TROVE_EMBEDDING_SOCKET', DEFAULT_EMBED_SOCKET)
        raw_timeout = os.environ.get('TROVE_EMBEDDING_DAEMON_TIMEOUT_MS', '500')
        try:
            configured_timeout = int(raw_timeout)
        except ValueError:
            configured_timeout = 500
        self.daemon_timeout_ms = max(1, min(int(daemon_timeout_ms or configured_timeout), 120_000))
        self._model_factory = model_factory
        self._device = device
        self._model = None
        self._model_lock = threading.Lock()
        self.dimensions = int(dimensions or 0)
        initial_identity = identity_for_model(
            self.model_path,
            dimensions=self.dimensions or None,
        )
        self.dimensions = initial_identity.dimensions
        self.model = initial_identity.model_id
        self.model_id = initial_identity.model_id
        self._daemon_identity = initial_identity
        self._daemon_identity_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self.auto_discovered = False
        self._daemon_fallback_count = 0
        self._daemon_last_reason_code: str | None = None
        self._daemon_requests = 0
        self._daemon_hits = 0
        self._daemon_last_telemetry: dict[str, Any] = {}
        if model_factory is not None:
            self._model = self._load_model(model_factory, device=device)
            self._update_dimensions(self._model)

    def _default_model_factory(self):
        if self._model_factory is not None:
            return self._model_factory
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError('sentence-transformers is not installed in this Python environment.') from exc
        return SentenceTransformer

    def _update_dimensions(self, model: Any) -> None:
        dim_fn = getattr(model, 'get_embedding_dimension', None) or getattr(model, 'get_sentence_embedding_dimension', lambda: None)
        dim = dim_fn()
        self.dimensions = int(dim or self.dimensions or 0)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        # One LocalEmbeddingProvider can serve many API threads. Loading a
        # multi-GB model must be singleflight just like daemon loading.
        with self._model_lock:
            if self._model is None:
                self._model = self._load_model(self._default_model_factory(), device=self._device)
                self._update_dimensions(self._model)
        return self._model

    def _load_model(self, factory: Callable[..., Any], *, device: str | None):
        kwargs = {'device': device} if device else {}
        try:
            return factory(str(self.model_path), local_files_only=True, **kwargs)
        except TypeError:
            return factory(str(self.model_path), **kwargs)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self.use_daemon and 0 < len(texts) <= 16 and os.environ.get('TROVE_DISABLE_EMBED_DAEMON_CLIENT') != '1':
            vectors = self._try_daemon(texts)
            if vectors is not None:
                if vectors and not self.dimensions:
                    self.dimensions = len(vectors[0])
                return vectors
        model = self._ensure_model()
        try:
            vectors = model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except TypeError:
            vectors = model.encode(texts)
        result: list[list[float]] = []
        for vector in vectors:
            result.append([float(value) for value in list(vector)])
        if result and not self.dimensions:
            self.dimensions = len(result[0])
        return result

    def _try_daemon(self, texts: list[str]) -> list[list[float]] | None:
        with self._telemetry_lock:
            self._daemon_requests += 1
        with self._daemon_identity_lock:
            identity = self._daemon_identity
            if identity.model_id != self.model_id:
                identity = identity_for_model(
                    self.model_path,
                    model_id=self.model_id,
                    dimensions=self.dimensions,
                )
            elif identity.dimensions != self.dimensions and self.dimensions:
                identity = replace(identity, dimensions=self.dimensions)
            self._daemon_identity = identity
        client = EmbeddingDaemonClient(
            self.socket_path,
            identity=identity,
            timeout_ms=self.daemon_timeout_ms,
        )
        try:
            vectors, telemetry = client.embed(texts)
        except DaemonProtocolError as exc:
            with self._telemetry_lock:
                self._daemon_fallback_count += 1
                self._daemon_last_reason_code = exc.code
                self._daemon_last_telemetry = {}
            return None
        with self._telemetry_lock:
            self._daemon_hits += 1
            self._daemon_last_reason_code = None
            self._daemon_last_telemetry = {
                key: value
                for key, value in telemetry.items()
                if key in {'queue_depth', 'load_count', 'batched_requests', 'completed_requests'}
                and type(value) in {int, float}
            }
        return vectors


    def daemon_telemetry(self) -> dict[str, Any]:
        with self._telemetry_lock:
            return {
                'requests': self._daemon_requests,
                'hits': self._daemon_hits,
                'fallback_count': self._daemon_fallback_count,
                'last_reason_code': self._daemon_last_reason_code,
                'fallback_mode': 'in_process_local' if self._daemon_last_reason_code else None,
                'daemon': dict(self._daemon_last_telemetry),
                'raw_content_included': False,
                'raw_paths_included': False,
                'secret_values_included': False,
            }
