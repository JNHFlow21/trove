from __future__ import annotations

from .base import ImageCaptionRequest, ImageCaptionResult, ImageCaptionUsage, LocalVLMCaptionProvider


class FakeLocalVLMCaptionProvider(LocalVLMCaptionProvider):
    name = 'fake-local-vlm'
    model = 'fixture'
    resource_id = 'local'

    def __init__(self, caption: str = '这是一张测试图片', labels: list[str] | None = None, confidence: float = 1.0):
        self.fixture_caption = caption
        self.fixture_labels = labels or []
        self.fixture_confidence = confidence
        self.calls = 0

    def caption(self, request: ImageCaptionRequest) -> ImageCaptionResult:
        self.calls += 1
        return ImageCaptionResult(
            caption=self.fixture_caption,
            labels=list(self.fixture_labels),
            confidence=self.fixture_confidence,
            usage=ImageCaptionUsage(estimated_cost_rmb=0.0),
            citations=[request.citation] if request.citation else [],
            metadata={'local_only': True, 'fixture': True},
        )
