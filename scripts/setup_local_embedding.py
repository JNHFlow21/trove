#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from trove_core.embedding.model_registry import DEFAULT_MODEL_ID, default_model_cache_root, model_dir_for, resolve_model_spec


def main() -> int:
    parser = argparse.ArgumentParser(description='Download a local-only embedding model into a TROVE model cache.')
    parser.add_argument('--model', default=DEFAULT_MODEL_ID, help='Hugging Face model id or registry alias.')
    parser.add_argument('--cache-root', help='Model cache root. Defaults to TROVE_MODEL_CACHE or ~/.cache/trove/models.')
    parser.add_argument('--local-dir', help='Exact local model directory to create/use.')
    parser.add_argument('--revision')
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()

    spec = resolve_model_spec(args.model)
    cache_root = Path(args.cache_root).expanduser() if args.cache_root else default_model_cache_root()
    local_dir = Path(args.local_dir).expanduser() if args.local_dir else model_dir_for(spec.model_id, cache_root)
    payload = {
        'model_id': spec.model_id,
        'provider': spec.provider,
        'expected_dimensions': spec.dimensions,
        'local_path_exists': local_dir.exists(),
        'safe_name': spec.safe_name,
    }
    if args.status_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise SystemExit('huggingface_hub is required: install sentence-transformers or huggingface-hub first.') from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.model_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=False,
    )
    manifest = {
        'model_id': spec.model_id,
        'provider': spec.provider,
        'expected_dimensions': spec.dimensions,
        'max_tokens': spec.max_tokens,
        'language': spec.language,
        'downloaded_at': datetime.now(timezone.utc).isoformat(),
    }
    (local_dir / 'trove_model_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    payload.update({'local_path_exists': True, 'manifest_present': True})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
