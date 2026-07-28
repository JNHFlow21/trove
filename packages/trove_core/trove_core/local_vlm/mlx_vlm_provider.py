from __future__ import annotations

import json
from pathlib import Path
import os
import re
import time
from typing import Any

from .base import ImageCaptionRequest, ImageCaptionResult, ImageCaptionUsage, LocalVLMCaptionProvider

DEFAULT_MLX_VLM_MODEL_ID = 'mlx-community/Qwen2.5-VL-3B-Instruct-4bit'
DEFAULT_MODEL_NAME = 'Qwen2.5-VL-3B-Instruct'
DEFAULT_CAPTION_PROMPT = '请用中文一句话描述这张图片，不超过60字。可另起一行以“标签：”列出最多5个标签词。不要输出JSON。'

_LABEL_PREFIX_RE = re.compile(r'^(标签|关键词|labels?)\s*[:：]\s*', re.I)
_SEP_RE = re.compile(r'[，,、；;\s]+')
_BULLET_PREFIX_RE = re.compile(r'^(?:[\-–—*•]\s*)+')
_LIST_ORDINAL_PREFIX_RE = re.compile(r'^\d+\s*(?:[、)）]|[.．](?!\d))\s*')


def _is_json_object(value: str) -> bool:
    value = value.strip()
    if not (value.startswith('{') and value.endswith('}')):
        return False
    try:
        return isinstance(json.loads(value), dict)
    except json.JSONDecodeError:
        return False


def _clean_line(value: str) -> str:
    value = value.strip().strip('`').strip()
    value = _BULLET_PREFIX_RE.sub('', value).strip()
    value = _LIST_ORDINAL_PREFIX_RE.sub('', value).strip()
    return value.strip('"“”')


def parse_caption_text(text: str, *, max_chars: int = 60) -> tuple[str, list[str]]:
    """Normalize a local VLM answer into one compact Chinese caption plus labels."""
    raw = (text or '').strip()
    if not raw:
        raise ValueError('empty_caption_output')
    if _is_json_object(raw):
        raise ValueError('empty_caption_output')
    lines = [_clean_line(line) for line in raw.replace('\r', '\n').split('\n')]
    lines = [line for line in lines if line]
    labels: list[str] = []
    caption_parts: list[str] = []
    for line in lines:
        if _LABEL_PREFIX_RE.match(line):
            label_text = _LABEL_PREFIX_RE.sub('', line).strip()
            for item in _SEP_RE.split(label_text):
                item = _clean_line(item)
                if item and item not in labels:
                    labels.append(item)
            continue
        if _is_json_object(line):
            continue
        caption_parts.append(line)
    caption = ' '.join(caption_parts).strip()
    caption = _LABEL_PREFIX_RE.sub('', caption).strip()
    caption = re.sub(r'\s+', ' ', caption)
    if len(caption) > max_chars:
        caption = caption[:max_chars].rstrip('，。,.；;：: ')
    if not caption:
        raise ValueError('empty_caption_output')
    return caption, labels[:5]


class MlxVLMCaptionProvider(LocalVLMCaptionProvider):
    """Apple Silicon local caption provider backed by mlx-vlm.

    Model and MLX imports are loaded lazily.  The provider reads only local image
    files and does not send media bytes to any cloud API.
    """

    name = 'local-vlm-qwen25-vl'
    model = DEFAULT_MODEL_NAME
    resource_id = 'local'

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MLX_VLM_MODEL_ID,
        cache_dir: str | Path | None = None,
        max_tokens: int = 80,
        temperature: float = 0.0,
        local_files_only: bool | None = None,
    ):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        if local_files_only is None:
            local_files_only = os.environ.get('TROVE_LOCAL_VLM_LOCAL_FILES_ONLY', '').lower() in {'1', 'true', 'yes'}
        self.local_files_only = bool(local_files_only)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._config: Any | None = None
        self._generate: Any | None = None
        self._apply_chat_template: Any | None = None

    def _model_path(self) -> str:
        candidate = Path(self.model_id).expanduser()
        if candidate.exists():
            return str(candidate)
        if self.cache_dir is None:
            return self.model_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency branch.
            raise RuntimeError('huggingface_hub_not_installed') from exc
        try:
            return snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )
        except Exception as exc:  # pragma: no cover - depends on local model cache/network.
            raise RuntimeError(f'local_vlm_model_unavailable:{exc.__class__.__name__}') from exc

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._processor is not None and self._generate is not None:
            return self._model, self._processor, self._generate
        try:
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:  # pragma: no cover - exercised when optional dep missing.
            raise RuntimeError('mlx_vlm_not_installed') from exc
        try:
            from mlx_vlm.generate import generate
        except ImportError:  # pragma: no cover - older package export shape.
            from mlx_vlm import generate
        model_path = self._model_path()
        self._model, self._processor = load(model_path)
        try:
            self._config = load_config(model_path)
        except Exception:
            self._config = getattr(self._model, 'config', None)
        self._generate = generate
        self._apply_chat_template = apply_chat_template
        return self._model, self._processor, self._generate

    def _format_prompt(self, prompt: str, *, image_count: int = 1) -> str:
        if self._apply_chat_template is None:
            return prompt
        try:
            return self._apply_chat_template(self._processor, self._config or getattr(self._model, 'config', None), prompt, num_images=image_count)
        except Exception:
            return prompt

    def caption(self, request: ImageCaptionRequest) -> ImageCaptionResult:
        image_path = Path(request.image_path).expanduser()
        if not image_path.exists():
            raise FileNotFoundError('image_path_missing')
        started = time.perf_counter()
        _model, _processor, generate = self._load_model()
        prompt = request.prompt or DEFAULT_CAPTION_PROMPT
        formatted_prompt = self._format_prompt(prompt, image_count=1)
        images = [str(image_path)]
        try:
            output = generate(
                self._model,
                self._processor,
                formatted_prompt,
                images,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                verbose=False,
            )
        except TypeError:
            output = generate(
                self._model,
                self._processor,
                formatted_prompt,
                image=images,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                verbose=False,
            )
        text = getattr(output, 'text', output)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ''
        caption, labels = parse_caption_text(str(text), max_chars=60)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ImageCaptionResult(
            caption=caption,
            labels=labels,
            confidence=0.8,
            usage=ImageCaptionUsage(elapsed_ms=round(elapsed_ms, 3), estimated_cost_rmb=0.0),
            citations=[request.citation] if request.citation else [],
            metadata={
                'local_only': True,
                'provider_payload_included': False,
                'model_id': self.model_id,
            },
        )
