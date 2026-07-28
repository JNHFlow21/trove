from __future__ import annotations

from .base import ImageObservationResult, VisionProvider, VisionRequest, VisionUsage


class FakeVisionProvider(VisionProvider):
    name = 'fake-vision'
    model = 'fixture'

    def __init__(self, caption: str = 'fixture image observation', confidence: float = 1.0):
        self.caption = caption
        self.confidence = confidence

    def observe(self, request: VisionRequest) -> ImageObservationResult:
        return ImageObservationResult(self.caption, '', [], [], [], self.confidence, VisionUsage(estimated_cost_rmb=0.0), citations=[request.citation] if request.citation else [])
