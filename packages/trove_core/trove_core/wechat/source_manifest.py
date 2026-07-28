from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

from .source_inventory import SourceCandidate


@dataclass(frozen=True)
class RedactedSourceManifest:
    generated_at: float
    sources: list[dict]
    canonical_source_ids: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')


def build_manifest(candidates: list[SourceCandidate], canonical_limit: int = 1) -> RedactedSourceManifest:
    importable = [c for c in candidates if c.importable and not c.sensitive]
    canonical = importable[:canonical_limit]
    notes = []
    if canonical:
        notes.append('Canonical source ids are redacted stable ids; full paths stay local to the runtime Vault.')
    else:
        notes.append('No importable non-sensitive source candidate found.')
    return RedactedSourceManifest(
        generated_at=time.time(),
        sources=[c.to_dict() for c in candidates],
        canonical_source_ids=[c.source_id for c in canonical],
        notes=notes,
    )
