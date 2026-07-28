from __future__ import annotations

from pathlib import Path

from .base import ImageObservationResult, VisionProvider, VisionRequest, VisionUsage


class MacOSVisionOCRProvider(VisionProvider):
    """Local macOS Vision OCR provider.

    The provider reads only a local image file and runs Apple's on-device Vision
    text recognizer through PyObjC.  It performs no network or cloud calls.
    """

    name = 'local-macos-vision'
    model = 'VNRecognizeTextRequest'

    def __init__(self, *, languages: list[str] | None = None, accurate: bool = True):
        self.languages = languages or ['zh-Hans', 'zh-Hant', 'en-US']
        self.accurate = accurate

    def observe(self, request: VisionRequest) -> ImageObservationResult:
        if request.image_path is None:
            raise ValueError('image_path_required')
        image_path = Path(request.image_path).expanduser()
        if not image_path.exists():
            raise FileNotFoundError('image_path_missing')
        try:
            import Foundation
            import Vision
        except ImportError as exc:  # pragma: no cover - exercised when optional dep missing.
            raise RuntimeError('pyobjc_vision_not_installed') from exc

        url = Foundation.NSURL.fileURLWithPath_(str(image_path))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        request_obj = Vision.VNRecognizeTextRequest.alloc().init()
        if hasattr(request_obj, 'setRecognitionLanguages_'):
            request_obj.setRecognitionLanguages_(self.languages)
        if hasattr(request_obj, 'setRecognitionLevel_'):
            level = Vision.VNRequestTextRecognitionLevelAccurate if self.accurate else Vision.VNRequestTextRecognitionLevelFast
            request_obj.setRecognitionLevel_(level)
        if hasattr(request_obj, 'setUsesLanguageCorrection_'):
            request_obj.setUsesLanguageCorrection_(True)

        ok, error = handler.performRequests_error_([request_obj], None)
        if not ok:
            raise RuntimeError(str(error) if error else 'vision_ocr_failed')

        lines: list[str] = []
        confidences: list[float] = []
        for observation in request_obj.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string() or '').strip()
            if text:
                lines.append(text)
                confidences.append(float(candidate.confidence()))
        visible_text = '\n'.join(lines)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return ImageObservationResult(
            caption='',
            visible_text=visible_text,
            objects=[],
            business_signals=[],
            entity_mentions=[],
            confidence=max(0.0, min(1.0, confidence)),
            usage=VisionUsage(estimated_cost_rmb=0.0),
            citations=[request.citation] if request.citation else [],
            metadata={'local_only': True, 'ocr_line_count': len(lines)},
        )
