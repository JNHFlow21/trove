from __future__ import annotations

from pathlib import Path
import math
from typing import Any

from .base import ASRProvider, ASRRequest, ASRResult, ASRUsage


class FasterWhisperASRProvider(ASRProvider):
    """Local faster-whisper ASR provider.

    The provider loads model weights from a caller-supplied local cache and only
    reads the local audio path.  No media bytes or transcript snippets are sent
    to a cloud API.
    """

    name = 'local-faster-whisper'
    resource_id = 'local'

    def __init__(
        self,
        *,
        model_size: str = 'small',
        cache_dir: str | Path | None = None,
        device: str = 'auto',
        compute_type: str = 'auto',
        language: str = 'zh',
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.model_name = model_size
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model: Any | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised when optional dep missing.
            raise RuntimeError('faster_whisper_not_installed') from exc
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.cache_dir) if self.cache_dir is not None else None,
        )
        return self._model

    def prepare(self) -> None:
        """Load/download model weights before a caller enters its DB commit."""

        self._load_model()

    def transcribe(self, request: ASRRequest) -> ASRResult:
        if request.audio_path is None:
            raise ValueError('audio_path_required')
        audio_path = Path(request.audio_path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError('audio_path_missing')
        model = self._load_model()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        segments = list(segments_iter)
        parts = [str(getattr(seg, 'text', '') or '').strip() for seg in segments]
        text = ''.join(part for part in parts if part)
        avg_logprobs = [float(getattr(seg, 'avg_logprob')) for seg in segments if getattr(seg, 'avg_logprob', None) is not None]
        confidence = 0.0
        if avg_logprobs:
            confidence = max(0.0, min(1.0, math.exp(sum(avg_logprobs) / len(avg_logprobs))))
        duration = float(getattr(info, 'duration', 0.0) or 0.0)
        language = str(getattr(info, 'language', '') or self.language or 'zh')
        return ASRResult(
            text=text,
            language=language,
            confidence=confidence,
            usage=ASRUsage(duration_seconds=duration, estimated_cost_rmb=0.0),
            citations=[request.citation] if request.citation else [],
        )
