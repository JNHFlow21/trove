from __future__ import annotations

from .base import ASRProvider, ASRRequest, ASRResult, ASRUsage


class FakeASRProvider(ASRProvider):
    name = 'fake-asr'
    model_name = 'fixture'
    resource_id = 'fixture'

    def __init__(self, text: str = 'fixture transcript', duration_seconds: float = 1.0):
        self.text = text
        self.duration_seconds = duration_seconds

    def transcribe(self, request: ASRRequest) -> ASRResult:
        return ASRResult(self.text, 'zh', 1.0, ASRUsage(duration_seconds=self.duration_seconds, estimated_cost_rmb=0.0), citations=[request.citation] if request.citation else [])
