from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    ALLOWED_FILE_FAMILIES,
    OUT_OF_SCOPE_FILE_FAMILIES,
    DecryptConfig,
    DecryptFilePlan,
    DecryptPlan,
    SkippedAccount,
    classify_file_family,
)
from .path_safety import resolved_under
from .redaction import stable_hash


WECHAT_DB_STORAGE_DIRS: tuple[str, ...] = (
    'contact',
    'favorite',
    'hardlink',
    'head_image',
    'message',
    'session',
    'sns',
)


def _db_storage_candidate_files(storage: Path) -> list[Path]:
    """Enumerate only reviewed WeChat DB locations, never the whole tree."""

    candidates: list[Path] = []
    family_dirs: list[Path] = []
    for name in WECHAT_DB_STORAGE_DIRS:
        directory = storage / name
        try:
            if not directory.is_dir() or directory.is_symlink():
                continue
            family_dirs.append(directory)
        except OSError:
            continue
    if family_dirs:
        for directory in family_dirs:
            try:
                candidates.extend(path for path in directory.iterdir() if path.is_file())
            except OSError:
                continue
    else:
        try:
            # Flat layouts are fixtures/manual snapshots, not desktop WeChat roots.
            candidates.extend(path for path in storage.iterdir() if path.is_file())
        except OSError:
            return []
    return sorted(dict.fromkeys(candidates), key=lambda path: (path.name, str(path.parent)))


def _is_wechat_decrypted_account_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob('message_*.db')) and (path / 'contact.db').exists()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _looks_like_account_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / 'db_storage').is_dir():
        return any(
            classify_file_family(child) is not None
            for child in _db_storage_candidate_files(path / 'db_storage')
        )
    names = {p.name.lower() for p in path.iterdir() if p.is_file()}
    if any(name.startswith('message_') and name.endswith('.db') for name in names):
        return True
    if {'contact.db', 'sns.db', 'favorite.db', 'message_resource.db'} & names:
        return True
    if {'media_0.db', 'media_0.kvdb', 'hardlink.db', 'head_image.db'} & names:
        return True
    return _is_wechat_decrypted_account_dir(path)


def discover_account_roots(live_root: Path, *, max_depth: int = 8) -> list[Path]:
    root = live_root.expanduser()
    if not root.exists():
        return []
    results: list[Path] = []
    seen: set[Path] = set()
    def walk(path: Path, depth: int) -> None:
        if depth < 0:
            return
        try:
            if path.is_symlink():
                return
            if _looks_like_account_root(path):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    results.append(path)
                return
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                if child.is_dir() and not child.is_symlink():
                    walk(child, depth - 1)
        except OSError:
            return
    walk(root, max_depth)
    return sorted(results, key=lambda p: p.name)


def _direct_selected_account_roots(config: DecryptConfig) -> list[Path] | None:
    """Resolve the known desktop-WeChat layout without scanning all containers.

    Return ``None`` when any selector is incomplete or the expected layout is
    absent so generic/manual configurations retain discovery compatibility.
    """

    roots: list[Path] = []
    seen: set[Path] = set()
    live_root = config.live_root.expanduser()
    for selected in config.selected_accounts:
        container = str(selected.container_id or '').strip()
        root_name = str(selected.root_name or '').strip()
        if (
            not container
            or not root_name
            or Path(container).name != container
            or Path(root_name).name != root_name
            or container in {'.', '..'}
            or root_name in {'.', '..'}
        ):
            return None
        candidate = live_root / container / 'Data' / 'Documents' / 'xwechat_files' / root_name
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        if (
            not resolved_under(resolved, live_root)
            or not selected.matches(resolved)
            or not _looks_like_account_root(resolved)
        ):
            return None
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return sorted(roots, key=lambda path: path.name)


def _output_account_dir_name(selected_output_name: str | None, selected_root_name: str | None, account_root: Path) -> str:
    value = str(selected_output_name or selected_root_name or account_root.name).strip()
    # Keep output within one account directory even if a caller passes a path.
    return Path(value).name or account_root.name


def _candidate_files(account_root: Path) -> list[Path]:
    if (account_root / 'db_storage').is_dir():
        candidates = _db_storage_candidate_files(account_root / 'db_storage')
    else:
        try:
            candidates = [path for path in account_root.iterdir() if path.is_file()]
        except OSError:
            candidates = []
    out: list[Path] = []
    for child in candidates:
        lowered = child.name.lower()
        if lowered.endswith(('-wal', '-shm', '.db-wal', '.db-shm')):
            continue
        if classify_file_family(child) is not None:
            out.append(child)
    return sorted(out, key=lambda p: (p.name, str(p.parent)))


def build_decrypt_plan(config: DecryptConfig) -> DecryptPlan:
    errors: list[str] = []
    files: list[DecryptFilePlan] = []
    skipped_accounts: list[SkippedAccount] = []
    skipped_files: list[dict[str, Any]] = []
    live_root = config.live_root.expanduser()
    if not config.selected_accounts:
        errors.append('no_selected_accounts')
    if not live_root.exists():
        errors.append('live_root_missing')
    allowed = set(config.allowed_file_families or ALLOWED_FILE_FAMILIES)
    out_of_scope = set(OUT_OF_SCOPE_FILE_FAMILIES)
    direct_roots = _direct_selected_account_roots(config)
    account_roots = direct_roots if direct_roots is not None else discover_account_roots(live_root)
    for account_root in account_roots:
        if not resolved_under(account_root, live_root):
            skipped_accounts.append(SkippedAccount(account_root, 'path_escape'))
            continue
        selected = config.selected_for_root(account_root)
        if selected is None:
            skipped_accounts.append(SkippedAccount(account_root, 'not_selected_account'))
            continue
        account_hash = stable_hash(str(account_root.resolve()))
        output_dir_name = _output_account_dir_name(selected.output_name, selected.root_name, account_root)
        for child in _candidate_files(account_root):
            family = classify_file_family(child)
            if family is None:
                continue
            if family in out_of_scope:
                skipped_files.append({
                    'account_ref_hash': account_hash,
                    'file_name': child.name,
                    'file_family': family,
                    'status': 'skipped',
                    'reason': 'out_of_scope',
                    'raw_paths_included': False,
                })
                continue
            if family not in allowed:
                skipped_files.append({
                    'account_ref_hash': account_hash,
                    'file_name': child.name,
                    'file_family': family,
                    'status': 'skipped',
                    'reason': 'not_allowlisted',
                    'raw_paths_included': False,
                })
                continue
            files.append(DecryptFilePlan(
                account_ref_hash=account_hash,
                account_root=account_root,
                source_path=child,
                file_family=family,
                secret_name=selected.secret_ref(config.secret_name),
                output_relative=Path(output_dir_name) / child.name,
            ))
    return DecryptPlan(
        config=config,
        files=tuple(files),
        skipped_accounts=tuple(skipped_accounts),
        skipped_files=tuple(skipped_files),
        errors=tuple(errors),
        generated_at=_now(),
    )
