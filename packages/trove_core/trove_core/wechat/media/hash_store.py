from __future__ import annotations

from pathlib import Path
import hashlib


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_derivative_path(vault_root: Path, *parts: str) -> Path:
    root = Path(vault_root).expanduser().resolve()
    out = root.joinpath(*parts).resolve()
    if root not in out.parents and out != root:
        raise ValueError('derived media output must stay under the runtime Vault root')
    return out
