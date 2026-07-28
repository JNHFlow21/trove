"""Pure source discovery shared by full import and incremental sync."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


IMPORTABLE_SUFFIXES = frozenset({'.jsonl', '.db', '.sqlite', '.sqlite3'})


def is_wechat_decrypted_account_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob('message_*.db')) and (path / 'contact.db').exists()


def iter_importable_files(source: Path) -> Iterator[Path]:
    """Yield canonical source units without importing either orchestrator."""

    if source.is_file():
        yield source
        return
    if is_wechat_decrypted_account_dir(source):
        yield source
        return
    account_dirs = [
        path for path in source.iterdir()
        if path.is_dir() and is_wechat_decrypted_account_dir(path)
    ] if source.exists() else []
    if account_dirs:
        yield from sorted(account_dirs)
        return
    if not source.exists():
        return
    for path in source.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMPORTABLE_SUFFIXES:
            if path.name.endswith(('-wal', '-shm')):
                continue
            yield path


__all__ = ['IMPORTABLE_SUFFIXES', 'is_wechat_decrypted_account_dir', 'iter_importable_files']
