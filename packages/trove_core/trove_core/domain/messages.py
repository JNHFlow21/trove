from __future__ import annotations
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Literal
import re

ConversationType = Literal['private', 'group']
SourceType = Literal['message', 'favorite', 'moment', 'contact', 'transcript', 'image_observation']
Direction = Literal['incoming', 'outgoing', 'unknown']


def isoformat(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class Account:
    account_id: str
    label: str
    display_name: str

    def safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    account_id: str
    title: str
    type: ConversationType
    member_count: int = 1

    def safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Message:
    account_id: str
    account_label: str
    conversation_id: str
    conversation_title: str
    conversation_type: ConversationType
    sender_id: str
    sender_name: str
    timestamp: datetime
    content: str
    shard_id: str
    local_id: int
    sent_by_me: bool = False
    source_type: SourceType = 'message'
    content_kind: str = 'text'
    direction_hint: Direction | None = None
    normalized_payload: dict[str, Any] | None = None
    citation_source: str = field(default='wechat', repr=False, compare=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r'[a-z][a-z0-9._-]{0,63}', self.citation_source):
            raise ValueError('citation_source is invalid')

    @property
    def direction(self) -> str:
        if self.direction_hint in {'incoming', 'outgoing', 'unknown'}:
            return self.direction_hint
        return 'outgoing' if self.sent_by_me else 'incoming'

    @property
    def citation(self) -> str:
        return f'trove://{self.citation_source}/{self.account_id}/{self.conversation_id}/{self.shard_id}/{self.local_id}'

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop('direction_hint', None)
        data.pop('normalized_payload', None)
        data.pop('citation_source', None)
        data['timestamp'] = isoformat(self.timestamp)
        data['direction'] = self.direction
        data['citation'] = self.citation
        return data


@dataclass(frozen=True)
class Evidence:
    citation: str
    account_id: str
    account_label: str
    conversation_id: str
    conversation_title: str
    conversation_type: ConversationType
    sender_name: str
    timestamp: str
    snippet: str
    source_type: SourceType
    direction: str
    score: float
    retrieval_paths: list[str]
    context_anchor: str
    media_hint: dict[str, Any] | None = None
    # A multi-hop Episode is one bounded evidence object, not a single message
    # chosen out of a flattened window.  The representative citation remains
    # backward compatible while these citations expose the complete chain to
    # callers and evaluators.
    supporting_citations: tuple[str, ...] = ()
    evidence_kind: str = 'message'
    # Provider-only bounded text for ranking a complete Episode.  It is never
    # serialized to API/CLI callers; the public snippet stays compact.
    _rerank_text: str | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop('_rerank_text', None)
        return payload


@dataclass(frozen=True)
class ContextMessage:
    citation: str
    sender_name: str
    timestamp: str
    content: str
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
