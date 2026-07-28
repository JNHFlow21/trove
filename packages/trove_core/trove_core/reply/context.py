from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

from trove_core.application.repositories import SQLiteUnitOfWork
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import vault_generation_read

from .media import ReplyMediaResolver
from .models import ReplyEvent, canonical_digest


class ContextBridgeError(RuntimeError):
    code = 'reply_context_unavailable'


@dataclass(frozen=True)
class ReplyContextMessage:
    citation: str
    source_position: int
    timestamp: str
    direction: str
    kind: str
    text: str
    live_delta: bool
    trust: str = 'untrusted_evidence'

    def to_dict(self) -> dict[str, Any]:
        return {
            'citation': self.citation,
            'source_position': self.source_position,
            'timestamp': self.timestamp,
            'direction': self.direction,
            'kind': self.kind,
            'text': self.text,
            'live_delta': self.live_delta,
            'trust': self.trust,
        }


@dataclass(frozen=True)
class ReplyContextEnvelope:
    event_id: str
    round_id: str
    round_revision: int
    account_id: str
    conversation_id: str
    target_ref: str
    source_position: int
    vault_generation: str
    messages: tuple[ReplyContextMessage, ...]
    new_message_citations: tuple[str, ...]
    profile: Mapping[str, Any]
    knowledge_refs: tuple[Mapping[str, Any], ...]
    media: tuple[Mapping[str, Any], ...]
    coverage: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'event_id': self.event_id,
            'round_id': self.round_id,
            'round_revision': self.round_revision,
            'account_id': self.account_id,
            'conversation_id': self.conversation_id,
            'target_ref': self.target_ref,
            'source_position': self.source_position,
            'vault_generation': self.vault_generation,
            'messages': [item.to_dict() for item in self.messages],
            'new_message_citations': list(self.new_message_citations),
            'profile': dict(self.profile),
            'knowledge_refs': [dict(item) for item in self.knowledge_refs],
            'media': [dict(item) for item in self.media],
            'coverage': dict(self.coverage),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


def _iso_timestamp(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat()
        .replace('+00:00', 'Z')
    )


def _generation_digest(token: Any) -> str:
    encoded = json.dumps(
        token.cache_key(),
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _bounded_profile(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        return {}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
    except (TypeError, ValueError):
        return {}
    if len(encoded) <= 16_000:
        return payload
    compact: dict[str, Any] = {}
    for key in (
        'summary', 'identity', 'needs', 'objections', 'next_actions',
        'commitments', 'evidence_citations', 'completeness_state',
    ):
        if key in payload:
            compact[key] = payload[key]
    encoded = json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return compact if len(encoded) <= 16_000 else {}


class ContextBridge:
    """Assemble one cited, account-exact, generation-consistent reply context."""

    def __init__(
        self,
        config: VaultConfig | str,
        *,
        history_limit: int = 50,
        workspace: Any | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, VaultConfig)
            else VaultConfig.resolve(str(config), env={})
        )
        self.history_limit = max(1, min(int(history_limit), 200))
        self.workspace = workspace

    def build(
        self,
        event: ReplyEvent,
        *,
        round_id: str,
        round_revision: int,
    ) -> ReplyContextEnvelope:
        with vault_generation_read(self.config) as generation:
            with SQLiteUnitOfWork(self.config) as uow:
                with uow.store.connect() as conn:
                    conversations = conn.execute(
                        """SELECT account_id,conversation_id,title,type
                             FROM conversations
                            WHERE account_id=? AND conversation_id=?
                            LIMIT 2""",
                        (event.account_id, event.conversation_id),
                    ).fetchall()
                if len(conversations) != 1:
                    raise ContextBridgeError(
                        'reply event does not resolve to one exact conversation'
                    )
                fetch_limit = min(
                    400,
                    self.history_limit + len(event.messages) + 20,
                )
                rows = uow.messages.conversation_messages(
                    event.conversation_id,
                    account_id=event.account_id,
                    limit=fetch_limit,
                )
                history = [
                    ReplyContextMessage(
                        citation=str(row['citation']),
                        source_position=int(row['local_id']),
                        timestamp=str(row['timestamp']),
                        direction=str(row['direction']),
                        kind=str(row['content_kind'] or 'text'),
                        text=str(row['content'] or '')[:8_000],
                        live_delta=False,
                    )
                    for row in rows
                ]
                merged, new_citations = self._merge_live(history, event)
                truncated = len(merged) > self.history_limit
                bounded = tuple(merged[-self.history_limit:])
                visible_citations = [item.citation for item in bounded]
                media_hints = uow.media.hints_for_citations(visible_citations)
                profile = self._profile_context(uow.store, rows)

        knowledge_refs: tuple[Mapping[str, Any], ...] = ()
        if self.workspace is not None:
            try:
                knowledge_refs = tuple(self.workspace.visible_references())
            except Exception as exc:
                raise ContextBridgeError('reply workspace references are invalid') from exc
        new_citations = tuple(
            citation for citation in new_citations
            if any(item.citation == citation for item in bounded)
        )
        if not new_citations and bounded:
            latest_positions = {item.source_position for item in event.messages}
            new_citations = tuple(
                item.citation for item in bounded
                if item.source_position in latest_positions
            )
        media = ReplyMediaResolver(
            self.config,
            workspace=self.workspace,
        ).resolve(
            bounded,
            new_message_citations=new_citations,
            hints=media_hints,
        )
        return ReplyContextEnvelope(
            event_id=event.event_id,
            round_id=round_id,
            round_revision=round_revision,
            account_id=event.account_id,
            conversation_id=event.conversation_id,
            target_ref=event.target_ref,
            source_position=event.source_position,
            vault_generation=_generation_digest(generation),
            messages=bounded,
            new_message_citations=new_citations,
            profile=profile,
            knowledge_refs=knowledge_refs,
            media=media,
            coverage={
                'state': 'partial' if truncated else 'complete',
                'truncated': truncated,
                'returned': len(bounded),
                'history_limit': self.history_limit,
            },
        )

    @staticmethod
    def _merge_live(
        history: list[ReplyContextMessage],
        event: ReplyEvent,
    ) -> tuple[list[ReplyContextMessage], tuple[str, ...]]:
        merged = list(history)
        citations: list[str] = []
        for item in event.messages:
            match = next((
                existing for existing in merged
                if existing.source_position == item.source_position
                and existing.direction == 'incoming'
                and (
                    (
                        item.kind != 'text'
                        and existing.kind == item.kind
                    )
                    or (
                        item.kind == 'text'
                        and existing.text.strip()
                        == str(item.text or '').strip()
                    )
                )
            ), None)
            if match is not None:
                citations.append(match.citation)
                continue
            live = ReplyContextMessage(
                citation=item.citation,
                source_position=item.source_position,
                timestamp=_iso_timestamp(item.observed_at),
                direction='incoming',
                kind=item.kind,
                text=str(item.text or '')[:8_000],
                live_delta=True,
            )
            merged.append(live)
            citations.append(live.citation)
        merged.sort(key=lambda item: (item.source_position, item.timestamp, item.citation))
        return merged, tuple(citations)

    @staticmethod
    def _profile_context(store: Any, rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        peer_ids = sorted({
            str(row['sender_id'])
            for row in rows
            if str(row['direction']) == 'incoming' and str(row['sender_id'] or '')
        })
        if not peer_ids:
            return {}
        placeholders = ','.join('?' for _ in peer_ids)
        with store.connect() as conn:
            entities = conn.execute(
                f"""SELECT DISTINCT entity_id
                      FROM entity_identifiers
                     WHERE normalized_value IN ({placeholders})
                       AND confidence>=0.5
                     ORDER BY entity_id LIMIT 2""",
                peer_ids,
            ).fetchall()
            if len(entities) != 1:
                return {}
            row = conn.execute(
                """SELECT projection_json,completeness_state
                     FROM profile_snapshots
                    WHERE entity_id=?
                    ORDER BY version DESC,created_at DESC LIMIT 1""",
                (str(entities[0]['entity_id']),),
            ).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(str(row['projection_json']))
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            payload.setdefault('completeness_state', str(row['completeness_state']))
        return _bounded_profile(payload)


__all__ = [
    'ContextBridge',
    'ContextBridgeError',
    'ReplyContextEnvelope',
    'ReplyContextMessage',
]
