from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class VaultPaths:
    root: Path

    @property
    def index_dir(self) -> Path:
        return self.root / 'index'

    @property
    def sqlite_path(self) -> Path:
        return self.index_dir / 'trove.sqlite'

    @property
    def token_path(self) -> Path:
        return self.root / 'api' / 'local_token'

    @property
    def logs_dir(self) -> Path:
        return self.root / 'logs'

    @property
    def vector_dir(self) -> Path:
        return self.root / 'vectors'


    @property
    def sources_dir(self) -> Path:
        return self.root / 'sources'

    @property
    def manifests_dir(self) -> Path:
        return self.root / 'manifests'

    @property
    def jobs_dir(self) -> Path:
        return self.root / 'jobs'

    @property
    def proof_dir(self) -> Path:
        return self.root / 'proof'
