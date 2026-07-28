from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import platform

from .paths import VaultPaths

LEGACY_ENV = 'WECHAT_KOS_VAULT_ROOT'
TROVE_ENV = 'TROVE_VAULT_ROOT'
AUTO_VAULT_RELATIVE = ('Trove', 'trove-vault')


def platform_data_root() -> Path:
    home = Path.home()
    if platform.system() == 'Darwin':
        return home / 'Library' / 'Application Support' / 'TROVE'
    if platform.system() == 'Windows':
        base = os.environ.get('LOCALAPPDATA') or str(home / 'AppData' / 'Local')
        return Path(base) / 'TROVE'
    return Path(os.environ.get('XDG_DATA_HOME', str(home / '.local' / 'share'))) / 'trove'


def default_vault_root() -> Path:
    return platform_data_root() / 'vault'


def product_vault_root(env: dict[str, str] | None = None, *, allow_default_home: bool = True) -> Path | None:
    home_text = env.get('HOME') if env is not None else None
    if home_text:
        home = Path(home_text).expanduser()
    elif allow_default_home:
        home = Path.home()
    else:
        return None
    return home.joinpath(*AUTO_VAULT_RELATIVE)


def has_vault_structure(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / 'index').is_dir()


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_unsafe_runtime_path(path: Path) -> bool:
    text = str(path.resolve())
    if '/Library/Mobile Documents/' in text or '/Knowledge_OS/' in text or text.endswith('/Knowledge_OS'):
        return True
    repo = Path(__file__).resolve().parents[4]
    return path_is_under(path, repo)


@dataclass(frozen=True)
class VaultStatus:
    root: str
    source: str
    available: bool
    problem: str | None
    index_path: str
    counts: dict[str, int]
    legacy_config_detected: bool = False
    migration_guidance: str | None = None

    def to_dict(self) -> dict:
        return {
            'root': self.root,
            'source': self.source,
            'available': self.available,
            'problem': self.problem,
            'index_path': self.index_path,
            'counts': self.counts,
            'legacy_config_detected': self.legacy_config_detected,
            'migration_guidance': self.migration_guidance,
        }


@dataclass(frozen=True)
class VaultConfig:
    root: Path
    source: str = 'default'
    legacy_config_detected: bool = False
    migration_guidance: str | None = None

    @classmethod
    def resolve(cls, vault_arg: str | None = None, env: dict[str, str] | None = None) -> 'VaultConfig':
        explicit_env = env is not None
        env = env if env is not None else os.environ
        legacy = env.get(LEGACY_ENV)
        if vault_arg:
            root = Path(vault_arg).expanduser()
            source = 'arg'
            legacy_detected = bool(legacy)
        elif env.get(TROVE_ENV):
            root = Path(env[TROVE_ENV]).expanduser()
            source = 'env'
            legacy_detected = bool(legacy)
        elif (auto_root := product_vault_root(env, allow_default_home=not explicit_env)) is not None and has_vault_structure(auto_root):
            root = auto_root
            source = 'auto-discovered'
            legacy_detected = bool(legacy)
        elif legacy:
            root = Path(legacy).expanduser()
            source = 'legacy-env-readonly'
            legacy_detected = True
        else:
            root = product_vault_root(env, allow_default_home=True) or default_vault_root()
            source = 'unconfigured'
            legacy_detected = False
        guidance = None
        if legacy_detected:
            guidance = 'Legacy WeChat KOS Vault configuration detected; TROVE reads it as compatibility input but will not write secrets.'
        if source == 'unconfigured':
            guidance = 'No TROVE Vault found. Pass --vault, set TROVE_VAULT_ROOT, or create ~/Trove/trove-vault with an index/ directory.'
        return cls(root=root, source=source, legacy_config_detected=legacy_detected, migration_guidance=guidance)

    @property
    def paths(self) -> VaultPaths:
        return VaultPaths(self.root)

    def validate_runtime_path(self) -> None:
        if is_unsafe_runtime_path(self.root):
            raise ValueError(f'Unsafe Vault root: {self.root}')

    def ensure(self) -> None:
        self.validate_runtime_path()
        if self.source == 'unconfigured':
            raise ValueError(self.migration_guidance or 'No TROVE Vault configured.')
        self.paths.index_dir.mkdir(parents=True, exist_ok=True)
        self.paths.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.paths.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.sources_dir.mkdir(parents=True, exist_ok=True)
        self.paths.proof_dir.mkdir(parents=True, exist_ok=True)

    def require_configured_for_write(self, action: str = 'write') -> None:
        """Reject mutations when Vault discovery fell back to an unconfigured path.

        Read-only status commands may report the product default candidate, but a
        mutating command must not create that fallback tree implicitly.  The
        caller should ask the operator to pass --vault, set TROVE_VAULT_ROOT, or
        initialize a real Vault first.
        """
        self.validate_runtime_path()
        if self.source == 'unconfigured':
            guidance = self.migration_guidance or 'No TROVE Vault configured.'
            raise ValueError(f'{action} requires a configured TROVE Vault. {guidance}')

    def status(self, counts: dict[str, int] | None = None) -> VaultStatus:
        unsafe = is_unsafe_runtime_path(self.root)
        available = self.root.exists() and not unsafe and self.source != 'unconfigured'
        problem = None
        if unsafe:
            problem = 'Vault root is inside iCloud, Knowledge_OS, or the product repo.'
        elif self.source in {'arg', 'env', 'legacy-env-readonly'} and not self.root.exists():
            problem = 'Configured Vault root is unavailable or not mounted; TROVE did not create a fallback Vault.'
        elif self.source == 'unconfigured':
            problem = self.migration_guidance
        return VaultStatus(
            root=str(self.root),
            source=self.source,
            available=available,
            problem=problem,
            index_path=str(self.paths.sqlite_path),
            counts=counts or {'accounts': 0, 'conversations': 0, 'messages': 0, 'chunks': 0},
            legacy_config_detected=self.legacy_config_detected,
            migration_guidance=self.migration_guidance,
        )
