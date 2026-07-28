from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ASRUsage:
    duration_seconds: float
    estimated_cost_rmb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ASRRequest:
    asset_id: str
    audio_path: Path | None = None
    audio_url: str | None = None
    language: str | None = None
    citation: str | None = None


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str | None
    confidence: float | None
    usage: ASRUsage
    citations: list[str] = field(default_factory=list)
    provider_status: str = 'completed'

    def to_dict(self) -> dict:
        data = asdict(self)
        data['usage'] = self.usage.to_dict()
        return data


class ASRProvider(ABC):
    name: str = 'asr-base'
    model_name: str = ''
    resource_id: str = ''
    egress_kind: str | None = None

    @abstractmethod
    def transcribe(self, request: ASRRequest) -> ASRResult:
        raise NotImplementedError
