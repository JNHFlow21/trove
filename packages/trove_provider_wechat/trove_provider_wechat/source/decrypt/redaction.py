from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_SECRETISH_RE = re.compile(r'(?i)(key|token|secret|password|cipher|pragma\s+key)\s*[:=]\s*([^\s,;]+)')
_HEXISH_RE = re.compile(r'\b[0-9a-fA-F]{32,}\b')
_HOME_RE = re.compile(r'/(Users|home)/[^\s,;\"]+')


def stable_hash(value: str | Path | None, *, length: int = 16) -> str:
    text = '' if value is None else str(value)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]


def path_ref(path: Path, *, root: Path | None = None) -> str:
    """Return a non-sensitive path reference.

    If `path` is inside `root`, expose only a relative ref. Otherwise expose a
    hash. This prevents private absolute paths from entering reports.
    """
    try:
        resolved = path.resolve()
        if root is not None:
            root_resolved = root.resolve()
            try:
                return str(resolved.relative_to(root_resolved))
            except ValueError:
                pass
    except OSError:
        pass
    return f'path-sha256:{stable_hash(path)}'


def redact_text(text: str | bytes | None) -> str:
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    redacted = _SECRETISH_RE.sub(lambda m: f'{m.group(1)}=<redacted>', text)
    redacted = _HEXISH_RE.sub('<hex-redacted>', redacted)
    redacted = _HOME_RE.sub('/<private-path-redacted>', redacted)
    return redacted


def redact_obj(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ('key', 'secret', 'token', 'password')):
                out[str(key)] = '<redacted>' if item else item
            else:
                out[str(key)] = redact_obj(item)
        return out
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return [redact_obj(item) for item in value]
    if isinstance(value, (str, bytes)):
        return redact_text(value)
    return value
