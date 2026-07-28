from __future__ import annotations

import json
from typing import Any, Mapping, TextIO


def render_envelope(payload: Mapping[str, Any], *, pretty: bool, stream: TextIO) -> None:
    if pretty:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    else:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    stream.write(encoded + '\n')
    stream.flush()


__all__ = ['render_envelope']
