from __future__ import annotations

import secrets
import sys
from typing import Any, Mapping, Sequence, TextIO

from trove_client import TroveClient, TroveClientError
from trove_client.control import (
    control_reply_from_controlling_terminal,
    stop_daemon,
)
from trove_daemon.lifecycle import LifecycleError, RuntimeIdentity
from trove_protocol.capabilities import CatalogValidationError, validate_input
from trove_protocol.envelope import Envelope
from trove_protocol.errors import ErrorDetail

from .commands.admin import lifecycle, resolve_vault
from .commands.operations import execute_operation
from .commands.query import execute_query
from .operator_approval import decide as operator_decide
from .parser import CLIInputError, Route, parse_args
from .render import render_envelope


EXIT_BY_CODE = {
    'invalid_input': 2,
    'invalid_request': 2,
    'unknown_capability': 2,
    'ambiguous_target': 3,
    'approval_required': 4,
    'busy': 5,
    'deadline_exceeded': 5,
    'daemon_unavailable': 5,
    'capability_unavailable': 5,
    'platform_unsupported': 6,
    'operator_confirmation_required': 7,
}


def _request_id(value: str | None) -> str:
    return value or f'cli-{secrets.token_urlsafe(18)}'


def _error(request_id: str, code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return Envelope.failure(
        ErrorDetail(code, retryable=retryable, message=message),
        request_id=request_id,
    ).to_dict()


def _execute_capability(
    route: Route,
    payload: Mapping[str, Any],
    *,
    identity: RuntimeIdentity,
    request_id: str,
    timeout: float,
    client_factory=TroveClient,
) -> dict[str, Any]:
    assert route.spec is not None
    normalized = validate_input(route.spec, payload)
    with client_factory(identity, role='cli') as client:
        if route.spec.pack == 'standard' and route.spec.risk == 'read':
            return execute_query(
                client, route.spec, normalized,
                request_id=request_id, timeout=timeout,
            )
        return execute_operation(
            client, route.spec, normalized,
            request_id=request_id, timeout=timeout,
        )


def run(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    client_factory=TroveClient,
) -> int:
    stdout = stdout or sys.stdout
    request_id = _request_id(None)
    pretty = False
    try:
        namespace, route, payload = parse_args(argv)
        pretty = bool(namespace.pretty)
        request_id = _request_id(namespace.request_id)
        if route.lifecycle == 'version':
            result = lifecycle('version', None, request_id=request_id)
        else:
            vault = resolve_vault(namespace.vault, create=route.lifecycle == 'start')
            if route.operator_decision is not None:
                decision = operator_decide(
                    vault, namespace.approval_id, route.operator_decision,
                    note=namespace.note,
                )
                result = Envelope.success(
                    {'approval': decision}, request_id=request_id,
                ).to_dict()
            elif route.operator_reply_action is not None:
                reply_control = control_reply_from_controlling_terminal(
                    vault,
                    route.operator_reply_action,
                    review_id=getattr(namespace, 'review_id', None),
                    app_path=getattr(namespace, 'app_path', None),
                    mode=getattr(namespace, 'mode', None),
                )
                if reply_control.get('daemon_restart_required'):
                    stop_daemon(
                        RuntimeIdentity.for_vault(vault), timeout=5.0,
                    )
                result = Envelope.success(
                    {'reply_control': reply_control},
                    request_id=request_id,
                ).to_dict()
            else:
                identity = RuntimeIdentity.for_vault(vault)
                if route.lifecycle is not None:
                    result = lifecycle(route.lifecycle, identity, request_id=request_id)
                else:
                    result = _execute_capability(
                        route, payload, identity=identity,
                        request_id=request_id, timeout=namespace.timeout,
                        client_factory=client_factory,
                    )
    except CLIInputError as exc:
        result = _error(request_id, 'invalid_input', str(exc))
    except CatalogValidationError as exc:
        result = _error(request_id, 'invalid_request', str(exc))
    except TroveClientError as exc:
        result = dict(exc.response) if exc.response else _error(
            request_id, exc.code, str(exc), retryable=exc.retryable,
        )
    except (LifecycleError, FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        code = getattr(exc, 'code', 'vault_unavailable')
        result = _error(request_id, str(code), 'The selected local runtime is unavailable.')
    except Exception as exc:
        code = str(getattr(exc, 'code', 'cli_failed'))
        result = _error(request_id, code, 'The command failed without producing a result.')
    render_envelope(result, pretty=pretty, stream=stdout)
    if result.get('ok') is True:
        return 0
    error = result.get('error') if isinstance(result.get('error'), Mapping) else {}
    return EXIT_BY_CODE.get(str(error.get('code') or ''), 1)


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
