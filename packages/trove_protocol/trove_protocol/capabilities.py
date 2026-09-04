from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


PROTOCOL_VERSION = 'trove/1'
PACK_ORDER = {'standard': 0, 'operations': 1, 'admin': 2}
REPLAY_POLICIES = frozenset({'read', 'idempotent', 'journaled', 'never'})
TRUST_CLASSES = frozenset({'control', 'untrusted_evidence'})
_ID_RE = re.compile(r'^trove\.[a-z][a-z0-9_]*$')
_MCP_RE = re.compile(r'^trove_[a-z][a-z0-9_]*$')
_FORBIDDEN_GENERIC_BRANDING = 'wechat'


class CatalogValidationError(ValueError):
    def __init__(self, issues: Iterable[str] | str):
        self.issues = tuple(sorted(set([issues] if isinstance(issues, str) else issues)))
        super().__init__('; '.join(self.issues))


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    mcp_name: str
    cli_route: tuple[str, ...]
    description: str
    pack: str
    risk: str
    replay_policy: str
    trust_class: str
    provider_requirement: str | None
    response_budget: int
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    scoped: bool = False
    paginated: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            'id': self.capability_id,
            'mcp': self.mcp_name,
            'cli': list(self.cli_route),
            'description': self.description,
            'pack': self.pack,
            'risk': self.risk,
            'replay_policy': self.replay_policy,
            'trust_class': self.trust_class,
            'response_budget': self.response_budget,
            'scoped': self.scoped,
            'paginated': self.paginated,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
        }
        if self.provider_requirement:
            payload['provider_requirement'] = self.provider_requirement
        return payload


def _citation_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'uri': {'type': 'string', 'minLength': 1},
            'account_id': {'type': 'string', 'minLength': 1},
            'source_type': {'type': 'string', 'minLength': 1},
        },
        'required': ['uri', 'account_id', 'source_type'],
    }


def _input(
    properties: Mapping[str, Any] | None = None,
    *,
    required: tuple[str, ...] = (),
    scoped: bool = False,
) -> dict[str, Any]:
    fields = dict(properties or {})
    if scoped:
        fields['account_id'] = {'type': ['string', 'null'], 'minLength': 1}
    result: dict[str, Any] = {
        'type': 'object', 'additionalProperties': False, 'properties': fields,
    }
    if required:
        result['required'] = list(required)
    return result


def _output(*, scoped: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {'type': 'object', 'additionalProperties': True}
    if scoped:
        result['$defs'] = {'citation': _citation_schema()}
    return result


_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 1000, 'default': 100}
# trove.search is bounded by SEARCH_RESULTS (1..50, default 10) on the daemon
# side; keep the wire schema identical so a defaulted CLI call validates
# instead of failing daemon-side with invalid_limit.
_SEARCH_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 10}
_MOMENT_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 50}
_FAVORITE_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 50}
_STATS_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 10}
_PENDING_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 20}
_KIND_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 20}
_PLAN_LIMIT = {'type': 'integer', 'minimum': 1, 'maximum': 1000, 'default': 200}
_MESSAGE_KINDS = [
    'text', 'image', 'video', 'voice', 'sticker',
    'link', 'file', 'miniapp', 'transfer', 'redpacket', 'contact_card',
]
_CURSOR = {'type': ['string', 'null'], 'minLength': 20}
_TARGET = {'type': ['string', 'null'], 'minLength': 1}


def _spec(
    capability_id: str,
    mcp_name: str,
    cli_route: tuple[str, ...],
    description: str,
    *,
    pack: str = 'standard',
    risk: str = 'read',
    replay_policy: str = 'read',
    trust_class: str = 'untrusted_evidence',
    provider_requirement: str | None = None,
    response_budget: int = 65_536,
    properties: Mapping[str, Any] | None = None,
    required: tuple[str, ...] = (),
    scoped: bool = False,
    paginated: bool = False,
) -> CapabilitySpec:
    return CapabilitySpec(
        capability_id=capability_id,
        mcp_name=mcp_name,
        cli_route=cli_route,
        description=description,
        pack=pack,
        risk=risk,
        replay_policy=replay_policy,
        trust_class=trust_class,
        provider_requirement=provider_requirement,
        response_budget=response_budget,
        input_schema=_input(properties, required=required, scoped=scoped),
        output_schema=_output(scoped=scoped),
        scoped=scoped,
        paginated=paginated,
    )


CATALOG: tuple[CapabilitySpec, ...] = (
    _spec(
        'trove.capabilities', 'trove_capabilities', ('capabilities',),
        'List installed capabilities, availability, packs, and bounded account metadata.',
        trust_class='control', properties={},
    ),
    _spec(
        'trove.resolve', 'trove_resolve', ('accounts',),
        'List accounts or resolve a contact or conversation without silent ambiguity.',
        properties={'target': _TARGET, 'kind': {'type': ['string', 'null'], 'enum': ['contact', 'conversation', 'account', None]}},
        scoped=True,
    ),
    _spec(
        'trove.recall', 'trove_recall', ('recall',),
        'Read one bounded deterministic message timeline with citations and coverage.',
        properties={
            'target': _TARGET, 'conversation_id': _TARGET, 'direction': {'type': 'string', 'enum': ['incoming', 'outgoing', 'both'], 'default': 'both'},
            'since': _TARGET, 'until': _TARGET, 'query': {'type': 'string'}, 'limit': _LIMIT, 'cursor': _CURSOR,
        }, scoped=True, paginated=True,
    ),
    _spec(
        'trove.group_summary', 'trove_group_summary', ('group', 'summary'),
        'Read bounded group evidence and aggregates; the caller produces the final summary.',
        properties={'target': _TARGET, 'conversation_id': _TARGET, 'since': _TARGET, 'until': _TARGET, 'limit': _LIMIT, 'cursor': _CURSOR},
        scoped=True, paginated=True,
    ),
    _spec(
        'trove.search', 'trove_search', ('search',),
        'Search bounded local evidence with explicit retrieval status and citations.',
        properties={'query': {'type': 'string', 'minLength': 1}, 'target': _TARGET, 'semantic': {'type': 'string', 'enum': ['auto', 'on', 'off'], 'default': 'auto'}, 'limit': _SEARCH_LIMIT, 'cursor': _CURSOR},
        required=('query',), scoped=True, paginated=True,
    ),
    _spec(
        'trove.context', 'trove_context', ('context',),
        'Read a bounded context window around one citation.',
        properties={'citation': {'type': 'string', 'minLength': 1}, 'before': {'type': 'integer', 'minimum': 0, 'maximum': 200, 'default': 5}, 'after': {'type': 'integer', 'minimum': 0, 'maximum': 200, 'default': 5}},
        required=('citation',), scoped=True,
    ),
    _spec(
        'trove.profile', 'trove_profile', ('profile', 'show'),
        'Read a bounded cited person or relationship profile.',
        properties={'target': {'type': 'string', 'minLength': 1}, 'limit': _LIMIT},
        required=('target',), scoped=True,
    ),
    _spec(
        'trove.files_list', 'trove_files_list', ('files', 'list'),
        'List bounded file evidence without materializing content.',
        properties={'target': _TARGET, 'conversation_id': _TARGET, 'media_types': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 16}, 'limit': _LIMIT, 'cursor': _CURSOR},
        scoped=True, paginated=True,
    ),
    _spec(
        'trove.favorites_list', 'trove_favorites_list', ('favorites', 'list'),
        'Read one bounded cited favorites list with keyword, derived-kind, and time filters.',
        response_budget=131_072,
        properties={
            'keyword': _TARGET,
            'kind': {'type': ['string', 'null'], 'enum': ['note', 'media', None]},
            'since': _TARGET, 'until': _TARGET, 'limit': _FAVORITE_LIMIT, 'cursor': _CURSOR,
        },
        scoped=True, paginated=True,
    ),
    _spec(
        'trove.moment_timeline', 'trove_moment_timeline', ('moments', 'timeline'),
        'Read one bounded cited moment timeline for exactly one resolved author.',
        response_budget=131_072,
        properties={'target': {'type': 'string', 'minLength': 1}, 'since': _TARGET, 'until': _TARGET, 'limit': _MOMENT_LIMIT, 'cursor': _CURSOR},
        required=('target',), scoped=True, paginated=True,
    ),
    _spec(
        'trove.moment_interactions', 'trove_moment_interactions', ('moments', 'interactions'),
        'Read bounded cited moment interactions for one moment citation or one resolved actor.',
        response_budget=131_072,
        properties={'citation': _TARGET, 'target': _TARGET, 'since': _TARGET, 'until': _TARGET, 'limit': _MOMENT_LIMIT, 'cursor': _CURSOR},
        scoped=True, paginated=True,
    ),
    _spec(
        'trove.message_stats', 'trove_message_stats', ('messages', 'stats'),
        'Read bounded metadata-only message count aggregates over one bounded time window.',
        properties={
            'dimension': {'type': 'string', 'enum': ['by_conversation', 'by_sender'], 'default': 'by_conversation'},
            'conversation_id': _TARGET,
            'since': _TARGET, 'until': _TARGET, 'limit': _STATS_LIMIT,
        },
        scoped=True,
    ),
    _spec(
        'trove.pending_replies', 'trove_pending_replies', ('messages', 'pending'),
        'Read bounded metadata-only private conversations whose latest incoming message awaits a reply.',
        properties={'since': _TARGET, 'until': _TARGET, 'limit': _PENDING_LIMIT},
        scoped=True,
    ),
    _spec(
        'trove.messages_by_kind', 'trove_messages_by_kind', ('messages', 'by-kind'),
        'Read one bounded cited message listing filtered by exact content kind, newest first.',
        response_budget=131_072,
        properties={
            'kind': {'type': 'string', 'enum': _MESSAGE_KINDS},
            'conversation_id': _TARGET,
            'direction': {'type': ['string', 'null'], 'enum': ['incoming', 'outgoing', 'unknown', None]},
            'since': _TARGET, 'until': _TARGET, 'limit': _KIND_LIMIT, 'cursor': _CURSOR,
        },
        required=('kind',), scoped=True, paginated=True,
    ),
    _spec(
        'trove.reply_status', 'trove_reply_status', ('reply', 'status'),
        'Read redacted local reply runtime state and readiness.',
        pack='admin', trust_class='control', properties={},
    ),
    _spec(
        'trove.reply_reviews', 'trove_reply_reviews', ('reply', 'reviews'),
        'List bounded pending or decided reply drafts as untrusted evidence.',
        pack='admin',
        properties={
            'state': {
                'type': 'string',
                'enum': ['pending', 'approved', 'rejected', 'stale'],
                'default': 'pending',
            },
            'limit': _LIMIT,
        },
    ),
    _spec(
        'trove.reply_activity', 'trove_reply_activity', ('reply', 'activity'),
        'List bounded reply runtime activity as untrusted evidence.',
        pack='admin', properties={'limit': _LIMIT},
    ),
    _spec(
        'trove.media_fetch', 'trove_media_fetch', ('files', 'fetch'),
        'Materialize one cited media object through a bounded verified path.',
        properties={'citation': {'type': 'string', 'minLength': 1}, 'allow_remote': {'type': 'boolean', 'default': False}},
        required=('citation',), scoped=True, provider_requirement='media_provider',
    ),
    _spec(
        'trove.media_enrich', 'trove_media_enrich', ('media', 'transcribe'),
        'Read cached media understanding or create one bounded durable enrichment operation.',
        properties={'citation': {'type': 'string', 'minLength': 1}, 'kind': {'type': 'string', 'enum': ['transcribe', 'annotate']}},
        required=('citation', 'kind'), scoped=True, provider_requirement='media_provider', replay_policy='journaled',
    ),
    _spec(
        'trove.media_enrich_plan', 'trove_media_enrich_plan', ('media', 'plan'),
        'Read a bounded read-only preview of media understanding work for one scope: candidate counts by state, local/cloud split, cost and duration estimates.',
        properties={
            'conversation_id': _TARGET,
            'author_id': _TARGET,
            'media_types': {'type': 'array', 'items': {'type': 'string', 'enum': ['image', 'voice', 'file']}, 'maxItems': 3},
            'kinds': {'type': 'array', 'items': {'type': 'string', 'enum': ['ocr', 'caption', 'transcribe']}, 'maxItems': 3},
            'execution': {'type': 'string', 'enum': ['auto', 'local_only'], 'default': 'auto'},
            'since': _TARGET, 'until': _TARGET, 'limit': _PLAN_LIMIT,
        },
        scoped=True,
    ),
    _spec(
        'trove.operation_status', 'trove_operation_status', ('operation', 'status'),
        'Read one durable operation state and its typed continuation owner.',
        trust_class='control', properties={'operation_id': {'type': 'string', 'minLength': 20}}, required=('operation_id',),
    ),
    _spec(
        'trove.operation_continue', 'trove_operation_continue', ('operation', 'continue'),
        'Continue an awaiting caller operation with an opaque single-operation token.',
        risk='controlled_write', replay_policy='journaled', trust_class='control',
        properties={'operation_id': {'type': 'string', 'minLength': 20}, 'token': {'type': 'string', 'minLength': 20}, 'payload': {'type': 'object'}},
        required=('operation_id', 'token', 'payload'),
    ),
    _spec(
        'trove.profile_build', 'trove_profile_build', ('profile', 'build'),
        'Create a durable cited profile operation.', pack='operations', risk='controlled_write',
        replay_policy='journaled', trust_class='control', properties={'target': {'type': 'string', 'minLength': 1}}, required=('target',), scoped=True,
    ),
    _spec(
        'trove.files_export', 'trove_files_export', ('files', 'export'),
        'Request an approval-gated export of an exact file selection.', pack='operations',
        risk='sensitive_write', replay_policy='journaled', trust_class='control',
        properties={'selection': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 100}, 'destination': {'type': 'string', 'minLength': 1}, 'approval_id': _TARGET},
        required=('selection', 'destination'), scoped=True,
    ),
    _spec(
        'trove.observe_add', 'trove_observe_add', ('observe', 'add'),
        'Add one explicit bounded observation with provenance.', pack='operations',
        risk='controlled_write', replay_policy='journaled', trust_class='control',
        properties={'target': {'type': 'string', 'minLength': 1}, 'text': {'type': 'string', 'minLength': 1, 'maxLength': 8000}, 'idempotency_key': {'type': 'string', 'minLength': 16}},
        required=('target', 'text', 'idempotency_key'), scoped=True,
    ),
    _spec(
        'trove.observe_list', 'trove_observe_list', ('observe', 'list'),
        'List bounded reviewed observations and their citations.', pack='operations',
        properties={'target': _TARGET, 'limit': _LIMIT, 'cursor': _CURSOR}, scoped=True, paginated=True,
    ),
    _spec(
        'trove.operation_cancel', 'trove_operation_cancel', ('operation', 'cancel'),
        'Cancel one durable operation only while its current stage is cancellable.',
        pack='operations', risk='controlled_write', replay_policy='idempotent', trust_class='control',
        properties={'operation_id': {'type': 'string', 'minLength': 20}}, required=('operation_id',),
    ),
    _spec(
        'trove.approval_request', 'trove_approval_request', ('approval', 'request'),
        'Create a pending request for a precise sensitive payload; never decide it.',
        pack='operations', risk='approval_request', replay_policy='idempotent', trust_class='control',
        properties={
            'action': {'type': 'string', 'minLength': 1},
            'danger_class': {'type': 'string', 'minLength': 1},
            'payload': {'type': 'object'},
            'idempotency_key': {'type': 'string', 'minLength': 16},
        },
        required=('action', 'danger_class', 'payload', 'idempotency_key'),
    ),
    _spec(
        'trove.approval_status', 'trove_approval_status', ('approval', 'status'),
        'Read pending or terminal approval state without decision authority.',
        pack='operations', trust_class='control', properties={'approval_id': {'type': 'string', 'minLength': 16}}, required=('approval_id',),
    ),
    _spec(
        'trove.sync', 'trove_sync', ('sync',),
        'Incrementally ingest selected source accounts through an installed read provider.',
        pack='admin', risk='admin_write', replay_policy='journaled', trust_class='control', provider_requirement='read_provider',
        properties={'account_ids': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 32}, 'full': {'type': 'boolean', 'default': False}, 'idempotency_key': {'type': 'string', 'minLength': 16}},
        required=('idempotency_key',),
    ),
    _spec(
        'trove.provider_status', 'trove_provider_status', ('provider', 'status'),
        'List verified installed providers and bounded health metadata.', pack='admin', trust_class='control', properties={},
    ),
    _spec(
        'trove.provider_reload', 'trove_provider_reload', ('provider', 'reload'),
        'Validate an installed provider then request bounded daemon drain and restart.',
        pack='admin', risk='admin_write', replay_policy='journaled', trust_class='control',
        properties={'provider_id': {'type': 'string', 'minLength': 1}, 'idempotency_key': {'type': 'string', 'minLength': 16}}, required=('provider_id', 'idempotency_key'),
    ),
    _spec(
        'trove.repair', 'trove_repair', ('repair', 'run'),
        'Run one catalog-defined bounded repair with explicit approval when required.',
        pack='admin', risk='admin_write', replay_policy='journaled', trust_class='control',
        properties={'repair_id': {'type': 'string', 'minLength': 1}, 'input': {'type': 'object'}, 'idempotency_key': {'type': 'string', 'minLength': 16}}, required=('repair_id', 'input', 'idempotency_key'),
    ),
    _spec(
        'trove.diagnostics', 'trove_diagnostics', ('doctor',),
        'Read redacted runtime, protocol, provider, and Vault health metadata.', pack='admin', trust_class='control', properties={},
    ),
)


STANDARD_MCP_TOOLS = frozenset(spec.mcp_name for spec in CATALOG if spec.pack == 'standard')
CATALOG_BY_ID = {spec.capability_id: spec for spec in CATALOG}
CATALOG_BY_MCP = {spec.mcp_name: spec for spec in CATALOG}


def capabilities_for_pack(pack: str) -> tuple[CapabilitySpec, ...]:
    if pack not in PACK_ORDER:
        raise CatalogValidationError(f'unknown pack:{pack}')
    ceiling = PACK_ORDER[pack]
    return tuple(spec for spec in CATALOG if PACK_ORDER[spec.pack] <= ceiling)


def _contains_forbidden_branding(spec: CapabilitySpec) -> bool:
    generic = {
        'id': spec.capability_id,
        'mcp': spec.mcp_name,
        'cli': spec.cli_route,
        'description': spec.description,
    }
    return _FORBIDDEN_GENERIC_BRANDING in json.dumps(generic, ensure_ascii=False).lower()


def validate_catalog(catalog: Iterable[CapabilitySpec]) -> None:
    specs = tuple(catalog)
    issues: list[str] = []
    if not specs:
        issues.append('catalog is empty')
    for field, values in (
        ('capability id', [spec.capability_id for spec in specs]),
        ('MCP name', [spec.mcp_name for spec in specs]),
        ('CLI route', [spec.cli_route for spec in specs]),
    ):
        if len(values) != len(set(values)):
            issues.append(f'duplicate {field}')
    for spec in specs:
        if not _ID_RE.fullmatch(spec.capability_id):
            issues.append(f'invalid capability id:{spec.capability_id}')
        if not _MCP_RE.fullmatch(spec.mcp_name):
            issues.append(f'invalid MCP name:{spec.mcp_name}')
        if not spec.cli_route or any(not part or '_' in part for part in spec.cli_route):
            issues.append(f'invalid CLI route:{spec.capability_id}')
        if spec.pack not in PACK_ORDER:
            issues.append(f'invalid pack:{spec.capability_id}')
        if spec.replay_policy not in REPLAY_POLICIES:
            issues.append(f'invalid replay policy:{spec.capability_id}')
        if spec.trust_class not in TRUST_CLASSES:
            issues.append(f'invalid trust class:{spec.capability_id}')
        if not 1 <= spec.response_budget <= 262_144:
            issues.append(f'invalid response budget:{spec.capability_id}')
        if _contains_forbidden_branding(spec):
            issues.append(f'source-specific name in generic surface:{spec.capability_id}')
        properties = spec.input_schema.get('properties') or {}
        if spec.scoped and 'account_id' not in properties:
            issues.append(f'scoped input missing account_id:{spec.capability_id}')
        citation = ((spec.output_schema.get('$defs') or {}).get('citation') or {})
        if spec.scoped and 'account_id' not in (citation.get('required') or ()):
            issues.append(f'scoped citation missing account_id:{spec.capability_id}')
    if issues:
        raise CatalogValidationError(issues)


def _matches_type(value: Any, expected: Any) -> bool:
    values = expected if isinstance(expected, list) else [expected]
    for item in values:
        if item == 'null' and value is None:
            return True
        if item == 'string' and isinstance(value, str):
            return True
        if item == 'integer' and type(value) is int:
            return True
        if item == 'boolean' and type(value) is bool:
            return True
        if item == 'object' and isinstance(value, Mapping):
            return True
        if item == 'array' and isinstance(value, list):
            return True
    return False


def validate_input(spec: CapabilitySpec, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CatalogValidationError(f'{spec.capability_id}: input must be an object')
    schema = spec.input_schema
    properties = schema.get('properties') or {}
    unknown = set(payload) - set(properties)
    if unknown:
        raise CatalogValidationError(f'{spec.capability_id}: unknown field:{sorted(unknown)[0]}')
    missing = set(schema.get('required') or ()) - set(payload)
    if missing:
        raise CatalogValidationError(f'{spec.capability_id}: missing field:{sorted(missing)[0]}')
    for field, value in payload.items():
        field_schema = properties[field]
        if 'type' in field_schema and not _matches_type(value, field_schema['type']):
            raise CatalogValidationError(f'{spec.capability_id}: invalid type:{field}')
        if value is not None and 'enum' in field_schema and value not in field_schema['enum']:
            raise CatalogValidationError(f'{spec.capability_id}: invalid enum:{field}')
        if isinstance(value, str) and len(value) < int(field_schema.get('minLength', 0)):
            raise CatalogValidationError(f'{spec.capability_id}: value too short:{field}')
        if type(value) is int:
            if value < int(field_schema.get('minimum', value)) or value > int(field_schema.get('maximum', value)):
                raise CatalogValidationError(f'{spec.capability_id}: value out of bounds:{field}')
        if isinstance(value, list):
            if len(value) < int(field_schema.get('minItems', 0)) or len(value) > int(field_schema.get('maxItems', len(value))):
                raise CatalogValidationError(f'{spec.capability_id}: list out of bounds:{field}')
    return dict(payload)


def catalog_snapshot() -> dict[str, Any]:
    specs = [spec.to_dict() for spec in CATALOG]
    canonical = json.dumps(specs, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {
        'protocol': PROTOCOL_VERSION,
        'catalog_sha256': hashlib.sha256(canonical).hexdigest(),
        'packs': {
            pack: [spec.capability_id for spec in capabilities_for_pack(pack)]
            for pack in PACK_ORDER
        },
        'capabilities': specs,
    }


validate_catalog(CATALOG)


__all__ = [
    'CATALOG', 'CATALOG_BY_ID', 'CATALOG_BY_MCP', 'CapabilitySpec',
    'CatalogValidationError', 'PACK_ORDER', 'PROTOCOL_VERSION',
    'STANDARD_MCP_TOOLS', 'capabilities_for_pack', 'catalog_snapshot',
    'validate_catalog', 'validate_input',
]
