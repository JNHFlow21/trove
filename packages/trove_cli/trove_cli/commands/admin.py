from __future__ import annotations

from pathlib import Path
import time
from typing import Any
from importlib.metadata import PackageNotFoundError, version

from trove_client.autostart import AutostartCoordinator, probe_daemon
from trove_client.control import stop_daemon
from trove_client.product_config import resolve_vault_root
from trove_daemon.lifecycle import RuntimeIdentity
from trove_protocol.envelope import Envelope


def _product_version() -> str:
    for distribution in ('trove-runtime',):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return '1.0.0-dev'


def resolve_vault(value: str | None, *, create: bool = False) -> Path:
    return resolve_vault_root(value, create=create)


def lifecycle(
    action: str,
    identity: RuntimeIdentity | None,
    *,
    request_id: str,
) -> dict[str, Any]:
    if action == 'version':
        return Envelope.success({'version': _product_version(), 'protocol': 'trove/1'}, request_id=request_id).to_dict()
    if identity is None:
        raise ValueError('runtime identity is required')
    if action == 'start':
        AutostartCoordinator.system(identity).ensure_running()
        state = probe_daemon(identity)
        return Envelope.success({'state': state, 'running': state == 'compatible'}, request_id=request_id).to_dict()
    if action == 'status':
        state = probe_daemon(identity)
        return Envelope.success({'state': state, 'running': state == 'compatible'}, request_id=request_id).to_dict()
    if action == 'stop':
        result = stop_daemon(identity, timeout=5.0)
        deadline = time.monotonic() + 5.0
        while probe_daemon(identity, timeout=0.1) != 'unavailable' and time.monotonic() < deadline:
            time.sleep(0.05)
        return Envelope.success({
            'running': probe_daemon(identity, timeout=0.1) == 'compatible',
            'drained': bool(result.get('ok', False)),
        }, request_id=request_id).to_dict()
    raise ValueError('unknown lifecycle action')


__all__ = ['lifecycle', 'resolve_vault']
