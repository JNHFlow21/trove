from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from typing import Any, Mapping, Sequence

from trove_protocol.capabilities import CATALOG, CapabilitySpec


class CLIInputError(ValueError):
    code = 'invalid_input'


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIInputError(message)


@dataclass(frozen=True)
class Route:
    path: tuple[str, ...]
    spec: CapabilitySpec | None = None
    lifecycle: str | None = None
    fixed: Mapping[str, Any] | None = None
    operator_decision: str | None = None
    operator_reply_action: str | None = None


LIFECYCLE_ROUTES = (
    Route(('start',), lifecycle='start'),
    Route(('stop',), lifecycle='stop'),
    Route(('status',), lifecycle='status'),
    Route(('version',), lifecycle='version'),
)


def _spec(capability_id: str) -> CapabilitySpec:
    return next(item for item in CATALOG if item.capability_id == capability_id)


ALIASES = (
    Route(('media', 'annotate'), _spec('trove.media_enrich'), fixed={'kind': 'annotate'}),
    Route(('media', 'status'), _spec('trove.operation_status')),
    Route(('observe', 'propose'), _spec('trove.approval_request'), fixed={'action': 'observe_propose'}),
    Route(('provider', 'list'), _spec('trove.provider_status')),
    Route(('approval', 'list'), _spec('trove.approval_status')),
    Route(('operator', 'approve'), operator_decision='approved'),
    Route(('operator', 'reject'), operator_decision='rejected'),
    Route(('operator', 'reply', 'arm'), operator_reply_action='arm'),
    Route(('operator', 'reply', 'disarm'), operator_reply_action='disarm'),
    Route(('operator', 'reply', 'pair'), operator_reply_action='pair'),
    Route(('operator', 'reply', 'mode'), operator_reply_action='mode'),
    Route(('operator', 'reply', 'approve'), operator_reply_action='approve'),
    Route(('operator', 'reply', 'reject'), operator_reply_action='reject'),
)


def public_routes() -> tuple[Route, ...]:
    routes = [Route(spec.cli_route, spec) for spec in CATALOG]
    # CLI grammar fixes media kind at the leaf rather than asking the caller to
    # repeat it in JSON-shaped input.
    routes = [
        Route(route.path, route.spec, fixed={'kind': 'transcribe'})
        if route.spec and route.spec.capability_id == 'trove.media_enrich'
        else route
        for route in routes
    ]
    return (*LIFECYCLE_ROUTES, *routes, *ALIASES)


def _json_object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError('expected one JSON object') from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError('expected one JSON object')
    return result


def _field_type(schema: Mapping[str, Any]):
    kind = schema.get('type')
    kinds = kind if isinstance(kind, list) else [kind]
    if 'integer' in kinds:
        return int
    if 'object' in kinds:
        return _json_object
    return str


def _add_schema_arguments(parser: argparse.ArgumentParser, route: Route) -> None:
    if route.operator_decision:
        parser.add_argument('approval_id')
        parser.add_argument('--note')
        return
    if route.operator_reply_action:
        if route.operator_reply_action in {'approve', 'reject'}:
            parser.add_argument('review_id')
        elif route.operator_reply_action == 'pair':
            parser.add_argument('--app', dest='app_path', required=True)
        elif route.operator_reply_action == 'mode':
            parser.add_argument(
                'mode',
                choices=('shadow', 'review_queue', 'live'),
            )
        return
    spec = route.spec
    if spec is None:
        return
    fixed = dict(route.fixed or {})
    properties = spec.input_schema.get('properties') or {}
    required = set(spec.input_schema.get('required') or ())
    for field, schema in properties.items():
        if field in fixed:
            continue
        option = '--account' if field == 'account_id' else '--' + field.replace('_', '-')
        kind = schema.get('type')
        kinds = kind if isinstance(kind, list) else [kind]
        kwargs: dict[str, Any] = {'dest': field, 'help': argparse.SUPPRESS}
        if 'boolean' in kinds:
            kwargs['action'] = argparse.BooleanOptionalAction
            kwargs['default'] = schema.get('default', None)
        elif 'array' in kinds:
            kwargs['action'] = 'append'
            kwargs['default'] = None
        else:
            kwargs['type'] = _field_type(schema)
            kwargs['default'] = schema.get('default', None)
        if field in required and field not in fixed:
            kwargs['required'] = True
        parser.add_argument(option, **kwargs)


def _add_common_leaf_arguments(parser: argparse.ArgumentParser) -> None:
    # Accept transport flags after the leaf as well as before it. SUPPRESS
    # preserves a value already parsed by the root parser.
    parser.add_argument('--vault', default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--pretty', action='store_true', default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--timeout', type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--request-id', default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog='trove', add_help=True)
    parser.add_argument('--vault', help=argparse.SUPPRESS)
    parser.add_argument('--pretty', action='store_true', help='indent the same JSON envelope')
    parser.add_argument('--timeout', type=float, default=30.0, help=argparse.SUPPRESS)
    parser.add_argument('--request-id', help=argparse.SUPPRESS)
    root = parser.add_subparsers(dest='_root', required=True)
    nodes: dict[tuple[str, ...], argparse.ArgumentParser] = {}
    children: dict[tuple[str, ...], argparse._SubParsersAction] = {(): root}
    routes = sorted(public_routes(), key=lambda item: (len(item.path), item.path))
    for route in routes:
        prefix: tuple[str, ...] = ()
        for index, component in enumerate(route.path):
            current = (*prefix, component)
            is_leaf = index == len(route.path) - 1
            node = nodes.get(current)
            if node is None:
                parser_options: dict[str, Any] = {}
                if is_leaf and route.spec:
                    parser_options.update(help=route.spec.description, description=route.spec.description)
                node = children[prefix].add_parser(component, **parser_options)
                nodes[current] = node
            if is_leaf:
                node.set_defaults(_route=route)
                _add_common_leaf_arguments(node)
                _add_schema_arguments(node, route)
            else:
                if current not in children:
                    children[current] = node.add_subparsers(dest=f'_route_{index}', required=True)
            prefix = current
    return parser


def payload_from_namespace(namespace: argparse.Namespace, route: Route) -> dict[str, Any]:
    if route.spec is None:
        return {}
    properties = route.spec.input_schema.get('properties') or {}
    payload = dict(route.fixed or {})
    for field in properties:
        if field in payload:
            continue
        value = getattr(namespace, field, None)
        if value is not None:
            payload[field] = value
    return payload


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, Route, dict[str, Any]]:
    namespace = build_parser().parse_args(argv)
    route = getattr(namespace, '_route', None)
    if not isinstance(route, Route):
        raise CLIInputError('a command leaf is required')
    if namespace.timeout <= 0 or namespace.timeout > 300:
        raise CLIInputError('--timeout must be in (0, 300]')
    return namespace, route, payload_from_namespace(namespace, route)


__all__ = [
    'ALIASES', 'CLIInputError', 'LIFECYCLE_ROUTES', 'Route', 'build_parser',
    'parse_args', 'payload_from_namespace', 'public_routes',
]
