from __future__ import annotations

from pathlib import Path


def resolved_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def require_existing_under(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError('path_escape') from exc
    return resolved


def require_output_under(path: Path, root: Path) -> Path:
    root_resolved = root.resolve(strict=False)
    parent = path.parent.resolve(strict=False)
    try:
        parent.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError('path_escape') from exc
    return path
