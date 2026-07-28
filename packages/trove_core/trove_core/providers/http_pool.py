from __future__ import annotations

import atexit
import os
import threading
from typing import Any


_lock = threading.RLock()
_client: Any | None = None
_client_pid: int | None = None


def _shared_client() -> Any:
    """Return one process-local, thread-safe HTTP connection pool."""

    global _client, _client_pid
    pid = os.getpid()
    with _lock:
        if _client is not None and _client_pid == pid:
            return _client
        try:
            import httpx  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'Cloud providers require optional dependency httpx; '
                'install the cloud runtime extras to use this feature.'
            ) from exc
        _client = httpx.Client(
            limits=httpx.Limits(
                max_connections=32,
                max_keepalive_connections=16,
                keepalive_expiry=30.0,
            ),
        )
        _client_pid = pid
        return _client


def post(*args: Any, **kwargs: Any) -> Any:
    return _shared_client().post(*args, **kwargs)


def close_shared_http_client() -> None:
    global _client, _client_pid
    with _lock:
        client = _client
        _client = None
        _client_pid = None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


atexit.register(close_shared_http_client)
