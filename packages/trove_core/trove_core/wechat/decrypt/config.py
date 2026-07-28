from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import fnmatch
from typing import Any

from .redaction import path_ref, stable_hash

ALLOWED_FILE_FAMILIES: tuple[str, ...] = (
    'message',
    'contact',
    'session',
    'sns',
    'favorite',
    'message_resource',
    'media_db',
    'media_kvdb',
    'hardlink',
    'head_image',
)

OUT_OF_SCOPE_FILE_FAMILIES: tuple[str, ...] = (
    'message_fts',
    'contact_fts',
    'favorite_fts',
)

_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ('message', 'message_*.db'),
    ('contact', 'contact.db'),
    ('session', 'session.db'),
    ('sns', 'sns.db'),
    ('favorite', 'favorite.db'),
    ('message_resource', 'message_resource.db'),
    ('media_db', 'media_*.db'),
    ('media_kvdb', 'media_*.kvdb'),
    ('hardlink', 'hardlink.db'),
    ('head_image', 'head_image.db'),
)

_OUT_OF_SCOPE_PATTERNS: tuple[tuple[str, str], ...] = (
    ('message_fts', 'message_fts.db'),
    ('contact_fts', 'contact_fts.db'),
    ('favorite_fts', 'favorite_fts.db'),
)


def classify_file_family(path: Path | str) -> str | None:
    name = Path(path).name
    lowered = name.lower()
    for family, pattern in _OUT_OF_SCOPE_PATTERNS:
        if fnmatch.fnmatch(lowered, pattern):
            return family
    for family, pattern in _FAMILY_PATTERNS:
        if fnmatch.fnmatch(lowered, pattern):
            return family
    return None


@dataclass(frozen=True)
class SelectedAccount:
    """A selected WeChat account/container without secret values."""

    account_id: str
    container_id: str | None = None
    root_name: str | None = None
    root_hash: str | None = None
    output_name: str | None = None
    secret_name: str | None = None

    def matches(self, root: Path) -> bool:
        name = root.name
        if self.container_id and self.container_id not in set(root.parts):
            return False
        if self.root_name and self.root_name == name:
            return True
        if self.root_hash and self.root_hash == stable_hash(name):
            return True
        tokens = [self.account_id, self.container_id]
        return any(token and token in name for token in tokens)

    def secret_ref(self, default_secret_name: str | None = None) -> str | None:
        return self.secret_name or default_secret_name

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'account_id_hash': stable_hash(self.account_id),
            'container_id_hash': stable_hash(self.container_id) if self.container_id else None,
            'root_name_hash': stable_hash(self.root_name) if self.root_name else None,
            'root_hash': self.root_hash,
            'output_name_hash': stable_hash(self.output_name) if self.output_name else None,
            'secret_name_configured': bool(self.secret_name),
        }


@dataclass(frozen=True)
class DecryptConfig:
    live_root: Path
    vault_root: Path
    selected_accounts: tuple[SelectedAccount, ...]
    secret_name: str | None = None
    key_store_path: Path | None = None
    output_source_name: str = 'wechat-integrated-decrypted'
    allowed_file_families: tuple[str, ...] = ALLOWED_FILE_FAMILIES
    fail_on_unselected_snapshot_account: bool = True
    allow_partial_accounts: bool = False
    persist_account_identity: bool = True

    def selected_for_root(self, root: Path) -> SelectedAccount | None:
        for account in self.selected_accounts:
            if account.matches(root):
                return account
        return None

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'live_root': path_ref(self.live_root),
            'vault_root': path_ref(self.vault_root),
            'selected_accounts': [account.to_redacted_dict() for account in self.selected_accounts],
            'selected_account_count': len(self.selected_accounts),
            'secret_name_configured': bool(self.secret_name),
            'key_store_configured': self.key_store_path is not None,
            'output_source_name': self.output_source_name,
            'allowed_file_families': list(self.allowed_file_families),
            'fail_on_unselected_snapshot_account': self.fail_on_unselected_snapshot_account,
            'allow_partial_accounts': self.allow_partial_accounts,
            'persist_account_identity': self.persist_account_identity,
            'raw_paths_included': False,
        }


@dataclass(frozen=True)
class DecryptFilePlan:
    account_ref_hash: str
    account_root: Path
    source_path: Path
    file_family: str
    secret_name: str | None
    output_relative: Path
    status: str = 'planned'
    reason: str | None = None

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'account_ref_hash': self.account_ref_hash,
            'source_path_hash': stable_hash(self.source_path),
            'file_name': self.source_path.name,
            'file_family': self.file_family,
            'secret_name_configured': bool(self.secret_name),
            'output_file_name': self.output_relative.name,
            'output_account_ref_hash': self.account_ref_hash,
            'status': self.status,
            'reason': self.reason,
            'raw_paths_included': False,
        }


@dataclass(frozen=True)
class SkippedAccount:
    root: Path
    reason: str

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'account_ref_hash': stable_hash(self.root.name),
            'reason': self.reason,
            'raw_paths_included': False,
        }


@dataclass(frozen=True)
class DecryptPlan:
    config: DecryptConfig
    files: tuple[DecryptFilePlan, ...]
    skipped_accounts: tuple[SkippedAccount, ...] = ()
    skipped_files: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    generated_at: str = ''

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.config.selected_accounts)

    def to_redacted_dict(self) -> dict[str, Any]:
        by_family: dict[str, int] = {}
        for item in self.files:
            by_family[item.file_family] = by_family.get(item.file_family, 0) + 1
        return {
            'ok': self.ok,
            'generated_at': self.generated_at,
            'config': self.config.to_redacted_dict(),
            'files_planned': len(self.files),
            'by_family': dict(sorted(by_family.items())),
            'skipped_accounts': [a.to_redacted_dict() for a in self.skipped_accounts],
            'skipped_account_count': len(self.skipped_accounts),
            'skipped_files': list(self.skipped_files),
            'errors': list(self.errors),
            'raw_content_included': False,
            'raw_paths_included': False,
        }


def selected_accounts_from_strings(values: list[str], *, secret_name: str | None = None) -> tuple[SelectedAccount, ...]:
    accounts: list[SelectedAccount] = []
    for value in values:
        text = str(value or '').strip()
        if not text:
            continue
        if ':' in text:
            account_token, root_name = text.split(':', 1)
        else:
            account_token, root_name = text, text
        if '@' in account_token:
            account_id, container_id = account_token.split('@', 1)
        else:
            account_id, container_id = account_token, None
        accounts.append(SelectedAccount(
            account_id=account_id.strip(),
            container_id=container_id.strip() if container_id else None,
            root_name=root_name.strip(),
            secret_name=secret_name,
        ))
    return tuple(accounts)
