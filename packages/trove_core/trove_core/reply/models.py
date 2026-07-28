from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


CONTENT_KINDS = frozenset({
    'text', 'image', 'sticker', 'voice', 'video', 'file', 'link', 'other',
})
DRAFT_STATES = frozenset({
    'generated', 'pending_review', 'approved', 'rejected', 'stale',
})
REVIEW_STATES = frozenset({'pending', 'approved', 'rejected', 'stale'})
SEND_STATES = frozenset({
    'prepared', 'dispatched', 'reconciling',
    'completed', 'failed', 'unknown', 'cancelled',
})
TERMINAL_SEND_STATES = frozenset({'completed', 'failed', 'unknown', 'cancelled'})
_DIGEST = re.compile(r'[0-9a-f]{64}')


class ReplyModelError(ValueError):
    code = 'invalid_reply_contract'


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical_digest(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ReplyModelError('reply payload must be canonical JSON') from exc
    return hashlib.sha256(encoded).hexdigest()


def _bounded_text(
    name: str,
    value: str,
    *,
    minimum: int = 1,
    maximum_bytes: int = 512,
) -> str:
    if not isinstance(value, str):
        raise ReplyModelError(f'{name} must be text')
    encoded = value.encode('utf-8')
    if len(encoded) < minimum or len(encoded) > maximum_bytes:
        raise ReplyModelError(f'{name} is outside its byte bound')
    if any(ord(char) < 0x20 for char in value):
        raise ReplyModelError(f'{name} contains control characters')
    return value


def _digest(name: str, value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ReplyModelError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _timestamp(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplyModelError(f'{name} must be a timestamp')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ReplyModelError(f'{name} must be a non-negative finite timestamp')
    return result


@dataclass(frozen=True)
class EvidenceMessage:
    citation: str
    source_position: int
    observed_at: float
    kind: str
    text: str | None = None

    def __post_init__(self) -> None:
        _bounded_text('citation', self.citation, maximum_bytes=1_024)
        if type(self.source_position) is not int or self.source_position <= 0:
            raise ReplyModelError('source_position must be a positive integer')
        _timestamp('observed_at', self.observed_at)
        if self.kind not in CONTENT_KINDS:
            raise ReplyModelError('unsupported evidence content kind')
        if self.text is not None:
            if not isinstance(self.text, str) or len(self.text.encode('utf-8')) > 8_000:
                raise ReplyModelError('message text exceeds its byte bound')


@dataclass(frozen=True)
class ReplyEvent:
    event_id: str
    account_id: str
    conversation_id: str
    target_ref: str
    source_position: int
    latest_fingerprint: str
    messages: tuple[EvidenceMessage, ...]
    observed_at: float

    def __post_init__(self) -> None:
        _bounded_text('event_id', self.event_id)
        _bounded_text('account_id', self.account_id)
        _bounded_text('conversation_id', self.conversation_id)
        _bounded_text('target_ref', self.target_ref, minimum=16, maximum_bytes=256)
        if type(self.source_position) is not int or self.source_position <= 0:
            raise ReplyModelError('source_position must be a positive integer')
        _digest('latest_fingerprint', self.latest_fingerprint)
        _timestamp('observed_at', self.observed_at)
        if not isinstance(self.messages, tuple) or not 1 <= len(self.messages) <= 200:
            raise ReplyModelError('event messages must be a bounded non-empty tuple')
        positions = [item.source_position for item in self.messages]
        if positions != sorted(positions) or positions[-1] != self.source_position:
            raise ReplyModelError('event messages must end at the source position')

    @property
    def latest_kind(self) -> str:
        return self.messages[-1].kind


@dataclass(frozen=True)
class RoundTiming:
    generation_prestart_ms: int = 3_000
    quiet_min_ms: int = 6_000
    quiet_default_ms: int = 8_000
    quiet_max_ms: int = 15_000
    max_collect_ms: int = 60_000
    max_messages: int = 50

    def __post_init__(self) -> None:
        values = (
            self.generation_prestart_ms,
            self.quiet_min_ms,
            self.quiet_default_ms,
            self.quiet_max_ms,
            self.max_collect_ms,
            self.max_messages,
        )
        if any(type(value) is not int for value in values):
            raise ReplyModelError('round timing values must be exact integers')
        if not 1 <= self.generation_prestart_ms <= 10_000:
            raise ReplyModelError('generation prestart is outside its bound')
        if not (
            1 <= self.quiet_min_ms
            <= self.quiet_default_ms
            <= self.quiet_max_ms
            <= self.max_collect_ms
            <= 300_000
        ):
            raise ReplyModelError('quiet timing range is invalid')
        if not 1 <= self.max_messages <= 200:
            raise ReplyModelError('max messages is outside its bound')


@dataclass(frozen=True)
class RoundRecord:
    round_id: str
    account_id: str
    conversation_id: str
    target_ref: str
    first_seen_at: float
    last_extended_at: float
    preparation_at: float
    earliest_ready_at: float
    ready_at: float
    deadline_at: float
    quiet_target_ms: int
    source_position: int
    latest_fingerprint: str
    inbound_message_count: int
    latest_kind: str
    revision: int
    attempts: int = 0
    not_before: float = 0.0
    last_error: str = ''
    blocked: bool = False

    def __post_init__(self) -> None:
        _bounded_text('round_id', self.round_id)
        _bounded_text('account_id', self.account_id)
        _bounded_text('conversation_id', self.conversation_id)
        _bounded_text('target_ref', self.target_ref, minimum=16, maximum_bytes=256)
        for name in (
            'first_seen_at', 'last_extended_at', 'preparation_at',
            'earliest_ready_at', 'ready_at', 'deadline_at', 'not_before',
        ):
            _timestamp(name, getattr(self, name))
        if type(self.source_position) is not int or self.source_position <= 0:
            raise ReplyModelError('source_position must be a positive integer')
        if type(self.revision) is not int or self.revision <= 0:
            raise ReplyModelError('round revision must be a positive integer')
        if type(self.inbound_message_count) is not int or self.inbound_message_count <= 0:
            raise ReplyModelError('round message count must be positive')
        if type(self.quiet_target_ms) is not int or self.quiet_target_ms <= 0:
            raise ReplyModelError('round quiet target must be positive')
        if type(self.attempts) is not int or self.attempts < 0:
            raise ReplyModelError('round attempts cannot be negative')
        _digest('latest_fingerprint', self.latest_fingerprint)
        if self.latest_kind not in CONTENT_KINDS:
            raise ReplyModelError('unsupported round content kind')
        if len(self.last_error.encode('utf-8')) > 160:
            raise ReplyModelError('round error exceeds its bound')

    def ready(self, now: float) -> bool:
        return not self.blocked and now >= max(
            self.earliest_ready_at, self.ready_at, self.not_before,
        )

    def preparable(self, now: float) -> bool:
        return not self.blocked and now >= max(self.preparation_at, self.not_before)


@dataclass(frozen=True)
class ReplyDraft:
    draft_id: str
    round_id: str
    round_revision: int
    account_id: str
    conversation_id: str
    target_ref: str
    source_position: int
    context_digest: str
    text: str
    backend: str
    model: str
    created_at: float
    state: str = 'generated'

    def __post_init__(self) -> None:
        for name in (
            'draft_id', 'round_id', 'account_id', 'conversation_id',
            'backend', 'model',
        ):
            _bounded_text(name, getattr(self, name))
        _bounded_text('target_ref', self.target_ref, minimum=16, maximum_bytes=256)
        if type(self.round_revision) is not int or self.round_revision <= 0:
            raise ReplyModelError('draft round revision must be positive')
        if type(self.source_position) is not int or self.source_position <= 0:
            raise ReplyModelError('draft source position must be positive')
        _digest('context_digest', self.context_digest)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ReplyModelError('draft text cannot be empty')
        if len(self.text.encode('utf-8')) > 8_000:
            raise ReplyModelError('draft text exceeds its byte bound')
        _timestamp('created_at', self.created_at)
        if self.state not in DRAFT_STATES:
            raise ReplyModelError('unsupported draft state')

    @property
    def digest(self) -> str:
        return sha256_text(self.text)


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    draft_id: str
    state: str
    created_at: float
    decided_at: float | None = None

    def __post_init__(self) -> None:
        _bounded_text('review_id', self.review_id)
        _bounded_text('draft_id', self.draft_id)
        if self.state not in REVIEW_STATES:
            raise ReplyModelError('unsupported review state')
        _timestamp('created_at', self.created_at)
        if self.decided_at is not None:
            _timestamp('decided_at', self.decided_at)


@dataclass(frozen=True)
class SendIntent:
    operation_id: str
    idempotency_key: str
    draft_id: str
    account_id: str
    conversation_id: str
    target_ref: str
    expected_source_position: int
    draft_digest: str
    text: str
    grant_ref: str

    def __post_init__(self) -> None:
        for name in (
            'operation_id', 'draft_id', 'account_id', 'conversation_id', 'grant_ref',
        ):
            _bounded_text(name, getattr(self, name))
        _bounded_text(
            'idempotency_key', self.idempotency_key, minimum=16, maximum_bytes=256,
        )
        _bounded_text('target_ref', self.target_ref, minimum=16, maximum_bytes=256)
        if type(self.expected_source_position) is not int or self.expected_source_position <= 0:
            raise ReplyModelError('expected source position must be positive')
        _digest('draft_digest', self.draft_digest)
        if not isinstance(self.text, str) or not self.text.strip():
            raise ReplyModelError('send text cannot be empty')
        if len(self.text.encode('utf-8')) > 8_000:
            raise ReplyModelError('send text exceeds its byte bound')
        if sha256_text(self.text) != self.draft_digest:
            raise ReplyModelError('send text does not match the draft digest')

    def digest(self) -> str:
        return canonical_digest({
            'operation_id': self.operation_id,
            'idempotency_key': self.idempotency_key,
            'draft_id': self.draft_id,
            'account_id': self.account_id,
            'conversation_id': self.conversation_id,
            'target_ref': self.target_ref,
            'expected_source_position': self.expected_source_position,
            'draft_digest': self.draft_digest,
            'text': self.text,
            'grant_ref': self.grant_ref,
        })


@dataclass(frozen=True)
class SendOperationRecord:
    operation_id: str
    draft_id: str
    idempotency_key: str
    intent_digest: str
    state: str
    stage: str
    external_ref: str | None
    result: Mapping[str, Any] | None
    error_code: str | None
    retry_count: int
    created_at: float
    updated_at: float

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_SEND_STATES

    @property
    def retryable(self) -> bool:
        return not self.terminal

    def __post_init__(self) -> None:
        for name in ('operation_id', 'draft_id', 'idempotency_key', 'stage'):
            _bounded_text(name, getattr(self, name))
        _digest('intent_digest', self.intent_digest)
        if self.state not in SEND_STATES:
            raise ReplyModelError('unsupported send state')
        if self.external_ref is not None:
            _bounded_text('external_ref', self.external_ref, maximum_bytes=1_024)
        if self.error_code is not None:
            _bounded_text('error_code', self.error_code)
        if (
            type(self.retry_count) is not int
            or not 0 <= self.retry_count <= 10
        ):
            raise ReplyModelError('send retry count is invalid')
        _timestamp('created_at', self.created_at)
        _timestamp('updated_at', self.updated_at)
