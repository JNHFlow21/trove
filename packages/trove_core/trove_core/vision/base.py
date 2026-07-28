from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VisionUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_rmb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VisionRequest:
    asset_id: str
    image_path: Path | None = None
    image_url: str | None = None
    prompt: str = 'Return compact structured observations for customer-intelligence evidence.'
    citation: str | None = None


@dataclass(frozen=True)
class ImageObservationResult:
    caption: str
    visible_text: str | None
    objects: list[str]
    business_signals: list[str]
    entity_mentions: list[str]
    confidence: float
    usage: VisionUsage
    citations: list[str] = field(default_factory=list)
    raw_provider_payload_stored: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['usage'] = self.usage.to_dict()
        return data


class VisionProvider(ABC):
    name: str = 'vision-base'
    model: str = ''
    egress_kind: str | None = None

    @abstractmethod
    def observe(self, request: VisionRequest) -> ImageObservationResult:
        raise NotImplementedError
