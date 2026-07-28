from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import re


DEFAULT_MODEL_ID = 'BAAI/bge-small-zh-v1.5'


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    dimensions: int
    max_tokens: int
    language: str
    recommendation: str

    @property
    def safe_name(self) -> str:
        return safe_model_name(self.model_id)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['safe_name'] = self.safe_name
        return data


MODEL_REGISTRY: dict[str, ModelSpec] = {
    'bge-small-zh-v1.5': ModelSpec(
        model_id='BAAI/bge-small-zh-v1.5',
        provider='sentence-transformers',
        dimensions=512,
        max_tokens=512,
        language='zh',
        recommendation='default-lightweight-zh-local',
    ),
    'bge-base-zh-v1.5': ModelSpec(
        model_id='BAAI/bge-base-zh-v1.5',
        provider='sentence-transformers',
        dimensions=768,
        max_tokens=512,
        language='zh',
        recommendation='higher-quality-larger-zh-local',
    ),
    'multilingual-e5-small': ModelSpec(
        model_id='intfloat/multilingual-e5-small',
        provider='sentence-transformers',
        dimensions=384,
        max_tokens=512,
        language='multilingual',
        recommendation='small-multilingual-fallback',
    ),
}


def safe_model_name(model_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '__', model_id).strip('_')


def registry_snapshot() -> dict:
    return {
        'default_model_id': DEFAULT_MODEL_ID,
        'models': {key: spec.to_dict() for key, spec in MODEL_REGISTRY.items()},
    }


def default_model_cache_root() -> Path:
    configured = os.environ.get('TROVE_MODEL_CACHE') or os.environ.get('TROVE_EMBEDDING_CACHE')
    if configured:
        return Path(configured).expanduser()
    return Path.home() / '.cache' / 'trove' / 'models'


def resolve_model_spec(model: str | None = None) -> ModelSpec:
    if not model:
        model = DEFAULT_MODEL_ID
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    for spec in MODEL_REGISTRY.values():
        if model == spec.model_id:
            return spec
    # Unknown model IDs are allowed, but must be explicit and locally present.
    return ModelSpec(
        model_id=model,
        provider='sentence-transformers',
        dimensions=0,
        max_tokens=0,
        language='unknown',
        recommendation='custom-local-explicit',
    )


def model_dir_for(model: str | None = None, cache_root: Path | None = None) -> Path:
    spec = resolve_model_spec(model)
    return (cache_root or default_model_cache_root()) / spec.safe_name


def default_local_model_path() -> Path | None:
    path = model_dir_for(DEFAULT_MODEL_ID)
    if not path.exists():
        return None
    if read_local_model_manifest(path) or (path / 'config.json').exists() or (path / 'modules.json').exists():
        return path
    return None


def read_local_model_manifest(model_path: Path) -> dict:
    manifest_path = Path(model_path) / 'trove_model_manifest.json'
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def model_status(model_path: str | None = None, model: str | None = None, cache_root: str | None = None) -> dict:
    spec = resolve_model_spec(model)
    path = Path(model_path).expanduser() if model_path else model_dir_for(spec.model_id, Path(cache_root).expanduser() if cache_root else None)
    manifest = read_local_model_manifest(path)
    return {
        'model_id': spec.model_id,
        'provider': spec.provider,
        'expected_dimensions': spec.dimensions,
        'max_tokens': spec.max_tokens,
        'language': spec.language,
        'local_path_configured': bool(model_path),
        'local_path_exists': path.exists(),
        'manifest_present': bool(manifest),
        'manifest': manifest,
    }
