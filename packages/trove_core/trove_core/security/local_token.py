from __future__ import annotations

import secrets
from pathlib import Path


class LocalTokenManager:
    def __init__(self, token_path: Path):
        self.token_path = Path(token_path)
        self._ephemeral: str | None = None

    def get_or_create(self) -> str:
        if self.token_path.exists():
            return self.token_path.read_text(encoding='utf-8').strip()
        value = self._ephemeral or ('trove-local-' + secrets.token_urlsafe(24))
        if self.token_path.parent.parent.exists():
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(value + '\n', encoding='utf-8')
            self.token_path.chmod(0o600)
        else:
            self._ephemeral = value
        return value

    def verify(self, header_value: str | None) -> bool:
        if not header_value or not header_value.startswith('Bearer '):
            return False
        return secrets.compare_digest(header_value.removeprefix('Bearer ').strip(), self.get_or_create())
