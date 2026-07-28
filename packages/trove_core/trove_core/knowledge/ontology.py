from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ENTITY_TYPES = {'Customer', 'Person', 'Organization', 'Conversation', 'Message', 'Moment', 'Favorite', 'MediaAsset', 'Opportunity', 'PainPoint', 'Objection', 'Need', 'NextAction', 'Commitment'}
RELATIONSHIP_TYPES = {'mentioned_in', 'participant_of', 'posted_by', 'speaker_in', 'same_as', 'stakeholder_of', 'blocks', 'owns_next_action', 'references_media', 'supports_claim'}
OBSERVATION_STATUSES = {'proposed', 'active', 'superseded', 'rejected', 'merge_candidate', 'merged', 'needs_review'}


@dataclass(frozen=True)
class ProfileClaim:
    value: str
    citations: list[str]
    confidence: float = 0.0
    status: str = 'active'
    source_type: str = 'observation'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_entity_type(entity_type: str) -> None:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f'unsupported entity type: {entity_type}')


def validate_relationship_type(predicate: str) -> None:
    if predicate not in RELATIONSHIP_TYPES:
        raise ValueError(f'unsupported relationship type: {predicate}')
