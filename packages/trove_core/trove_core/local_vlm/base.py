from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageCaptionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0
    estimated_cost_rmb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageCaptionRequest:
    asset_id: str
    image_path: Path
    prompt: str = '请用中文一句话描述这张图片，不超过60字。可另起一行以“标签：”列出最多5个标签词。'
    citation: str | None = None


@dataclass(frozen=True)
class ImageCaptionResult:
    caption: str
    labels: list[str] = field(default_factory=list)
    confidence: float = 0.8
    usage: ImageCaptionUsage = field(default_factory=ImageCaptionUsage)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['usage'] = self.usage.to_dict()
        return data


class LocalVLMCaptionProvider(ABC):
    name: str = 'local-vlm-base'
    model: str = ''
    resource_id: str = 'local'

    @abstractmethod
    def caption(self, request: ImageCaptionRequest) -> ImageCaptionResult:
        raise NotImplementedError
