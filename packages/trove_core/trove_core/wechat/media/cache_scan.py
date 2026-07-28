from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import os

MEDIA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.dat', '.mp3', '.m4a', '.wav', '.amr', '.silk', '.mp4', '.mov'}


@dataclass(frozen=True)
class CachedMediaFile:
    redacted_name: str
    suffix: str
    size_bytes: int
    sha256: str
    modality: str

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _modality(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in {'.mp3', '.m4a', '.wav', '.amr', '.silk'}:
        return 'voice'
    if suffix in {'.mp4', '.mov'}:
        return 'video'
    return 'image'


def scan_media_cache(root: Path, *, max_files: int | None = None) -> list[CachedMediaFile]:
    root = Path(root).expanduser()
    out: list[CachedMediaFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {'.git', 'node_modules', '.venv', '__pycache__'}]
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            if suffix not in MEDIA_SUFFIXES:
                continue
            try:
                st = path.stat()
                out.append(CachedMediaFile(redacted_name=path.name, suffix=suffix, size_bytes=st.st_size, sha256=_hash_file(path), modality=_modality(suffix)))
            except OSError:
                continue
            if max_files and len(out) >= max_files:
                return out
    return out
