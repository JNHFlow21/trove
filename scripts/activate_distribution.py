#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
from verify_distribution import verify_distribution

for relative in ('packages/trove_protocol', 'packages/trove_core'):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)
from trove_core.product_config import (
    ProductConfig, load_product_config, write_product_config,
)


class ActivationError(RuntimeError):
    pass


class UpgradeCapacityError(ActivationError):
    pass


MAX_RELEASES = 3
MAX_UPGRADE_BACKUPS = 2
MIN_FREE_FRACTION = 0.15
MIN_FREE_FLOOR_BYTES = 16 * 1024**3
UPGRADE_BACKUP_ENTRIES = ('index', 'manifests', 'vault.db')


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ActivationError('activation directory is unsafe')
    os.chmod(path, 0o700)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise ActivationError('activation destination has no existing parent')
        current = current.parent
    return current


def _backup_sources(vault: Path) -> tuple[Path, ...]:
    if vault.is_symlink() or not vault.is_dir() or vault.stat().st_uid != os.getuid():
        raise ActivationError('vault backup source is unsafe')
    return tuple(path for name in UPGRADE_BACKUP_ENTRIES if (path := vault / name).exists())


def _logical_size(path: Path) -> int:
    stat = path.lstat()
    if path.is_symlink() or stat.st_uid != os.getuid():
        raise ActivationError('vault backup entry is unsafe')
    if path.is_file():
        return stat.st_size
    if not path.is_dir():
        raise ActivationError('vault backup entry has unsupported type')
    return sum(_logical_size(child) for child in path.iterdir())


def check_upgrade_capacity(
    vault: Path,
    install_root: Path,
    *,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, int]:
    """Fail before an upgrade backup can consume the host's swap reserve."""
    sources = _backup_sources(vault)
    backup_bytes = sum(_logical_size(path) for path in sources)
    usage = disk_usage(_nearest_existing_parent(install_root))
    reserve_bytes = max(MIN_FREE_FLOOR_BYTES, int(usage.total * MIN_FREE_FRACTION))
    required_bytes = backup_bytes + reserve_bytes
    if usage.free < required_bytes:
        raise UpgradeCapacityError('insufficient free space for upgrade backup')
    return {
        'backup_bytes': backup_bytes,
        'free_before_bytes': int(usage.free),
        'reserve_bytes': reserve_bytes,
    }


def _is_sqlite(path: Path) -> bool:
    if path.stat().st_size < 16:
        return False
    with path.open('rb') as handle:
        return handle.read(16) == b'SQLite format 3\x00'


def _copy_backup_entry(source: Path, destination: Path) -> None:
    stat = source.lstat()
    if source.is_symlink() or stat.st_uid != os.getuid():
        raise ActivationError('vault backup entry is unsafe')
    if source.is_dir():
        destination.mkdir(mode=0o700)
        for child in source.iterdir():
            if child.name.endswith(('-wal', '-shm')):
                continue
            _copy_backup_entry(child, destination / child.name)
        os.chmod(destination, 0o700)
        return
    if not source.is_file():
        raise ActivationError('vault backup entry has unsupported type')
    if _is_sqlite(source):
        source_connection = sqlite3.connect(f'{source.as_uri()}?mode=ro', uri=True)
        try:
            destination_connection = sqlite3.connect(destination)
            try:
                source_connection.backup(destination_connection)
            finally:
                destination_connection.close()
        finally:
            source_connection.close()
    else:
        shutil.copyfile(source, destination)
    os.chmod(destination, 0o600)


def _prune_upgrade_backups(parent: Path, *, keep: Path) -> int:
    candidates = [
        path for path in parent.iterdir()
        if path.is_dir() and not path.is_symlink() and not path.name.startswith('.')
    ]
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    retained = candidates[:MAX_UPGRADE_BACKUPS]
    if keep not in retained:
        retained = [keep, *[path for path in retained if path != keep]][:MAX_UPGRADE_BACKUPS]
    for path in candidates:
        if path not in retained:
            shutil.rmtree(path)
    return len(retained)


def create_upgrade_backup(
    vault: Path,
    install_root: Path,
    *,
    capacity: Mapping[str, int] | None = None,
    now: datetime | None = None,
    disk_usage: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    plan = dict(capacity or check_upgrade_capacity(vault, install_root))
    usage = disk_usage or shutil.disk_usage
    vault_id = hashlib.sha256(str(vault.resolve()).encode()).hexdigest()[:20]
    backups = install_root / 'backups'
    vault_backups = backups / vault_id
    _secure_directory(install_root)
    _secure_directory(backups)
    _secure_directory(vault_backups)
    current_free = usage(vault_backups).free
    if current_free < plan['backup_bytes'] + plan['reserve_bytes']:
        raise UpgradeCapacityError('insufficient free space for upgrade backup')
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    backup_id = timestamp.strftime('%Y%m%dT%H%M%S%fZ')
    temporary = vault_backups / f'.{backup_id}-{os.getpid()}.tmp'
    destination = vault_backups / backup_id
    if destination.exists() or temporary.exists():
        raise ActivationError('upgrade backup destination already exists')
    temporary.mkdir(mode=0o700)
    try:
        sources = _backup_sources(vault)
        for source in sources:
            _copy_backup_entry(source, temporary / source.name)
        manifest = {
            'schema_version': 1,
            'backup_id': backup_id,
            'vault_id': vault_id,
            'source_entries': [source.name for source in sources],
            'estimated_source_bytes': int(plan['backup_bytes']),
            'reserve_bytes': int(plan['reserve_bytes']),
            'private_paths_included': False,
            'secret_values_included': False,
        }
        manifest_path = temporary / 'backup-manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n',
            encoding='utf-8',
        )
        os.chmod(manifest_path, 0o600)
        remaining = usage(temporary).free
        if remaining < plan['reserve_bytes']:
            raise UpgradeCapacityError('upgrade backup exhausted free-space reserve')
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    retained = _prune_upgrade_backups(vault_backups, keep=destination)
    return {
        'created': True,
        'backup_id': backup_id,
        'retained_backups': retained,
        'estimated_source_bytes': int(plan['backup_bytes']),
        'private_paths_included': False,
        'secret_values_included': False,
    }


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def install_release(
    manifest_path: Path,
    install_root: Path,
    *,
    python: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    verify_distribution(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    release_id = manifest['distribution_set_sha256']
    releases = install_root / 'releases'
    _secure_directory(install_root)
    _secure_directory(releases)
    destination = releases / release_id
    installed_marker = destination / 'distribution-manifest.json'
    if installed_marker.is_file():
        installed = json.loads(installed_marker.read_text(encoding='utf-8'))
        if installed.get('distribution_set_sha256') != release_id:
            raise ActivationError('installed release marker mismatch')
        return destination
    if destination.exists():
        raise ActivationError('incomplete release installation exists')
    destination.mkdir(mode=0o700)
    try:
        venv = destination / 'venv'
        runner([python, '-m', 'venv', str(venv)], check=True, env=_runtime_env())
        root = manifest_path.resolve().parent
        wheels = [str(root / manifest[kind]['file']) for kind in ('runtime', 'provider')]
        runner([
            str(venv / 'bin/python'), '-m', 'pip', 'install',
            '--find-links', str(root), *wheels,
        ], check=True, env=_runtime_env(), stdout=subprocess.DEVNULL)
        for executable in ('trove', 'trove-mcp', 'troved'):
            if not os.access(venv / 'bin' / executable, os.X_OK):
                raise ActivationError('installed runtime executable is missing')
        marker = destination / 'distribution-manifest.json'
        marker.write_text(
            json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n',
            encoding='utf-8',
        )
        os.chmod(marker, 0o600)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _read_current(current: Path, releases: Path) -> Path | None:
    if not current.is_symlink():
        if current.exists():
            raise ActivationError('current activation pointer is not a symlink')
        return None
    resolved = current.resolve(strict=True)
    if resolved.parent != releases.resolve(strict=True):
        raise ActivationError('current activation pointer escapes release root')
    return resolved


def _switch(current: Path, release: Path) -> None:
    temporary = current.with_name(f'.current-{os.getpid()}')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(Path('releases') / release.name)
    os.replace(temporary, current)


def _prune_releases(releases: Path, *, keep: set[Path], maximum: int = MAX_RELEASES) -> int:
    candidates = [
        path for path in releases.iterdir()
        if path.is_dir() and not path.is_symlink()
        and path.name not in {item.name for item in keep}
    ]
    candidates.sort(
        key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True,
    )
    retained_extra = max(0, maximum - len(keep))
    for path in candidates[retained_extra:]:
        shutil.rmtree(path)
    return min(maximum, len(keep) + min(len(candidates), retained_extra))


def candidate_identity(
    release: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, str]:
    code = (
        'import json; '
        'from trove_daemon.lifecycle import build_identity,catalog_identity; '
        'from importlib.metadata import distributions; '
        'from pathlib import Path; '
        'p=[]; '
        '[(p.append(json.loads(Path(d.locate_file("trove_provider_wechat/manifest.json")).read_text())["package_sha256"])) '
        'for d in distributions() if any(e.group=="trove.providers" and e.name=="wechat-source" for e in d.entry_points)]; '
        'print(json.dumps({"runtime_build_hash":build_identity(),"catalog_hash":catalog_identity(),"provider_package_hash":p[0] if len(p)==1 else ""},sort_keys=True))'
    )
    completed = runner(
        [str(release / 'venv/bin/python'), '-c', code],
        check=True, capture_output=True, text=True, env=_runtime_env(), cwd=release,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ActivationError('candidate identity is invalid')
    return {str(key): str(item) for key, item in value.items()}


def _runtime_command(
    release: Path,
    vault: Path,
    action: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess],
) -> dict[str, Any]:
    completed = runner(
        [str(release / 'venv/bin/trove'), '--vault', str(vault), action],
        check=False, capture_output=True, text=True, timeout=20,
        env=_runtime_env(), cwd=release,
    )
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ActivationError('candidate lifecycle response is invalid') from exc
    if completed.returncode or not isinstance(payload, dict) or payload.get('ok') is not True:
        raise ActivationError('candidate lifecycle command failed')
    return payload


def activate_distribution(
    manifest_path: Path,
    install_root: Path,
    *,
    vault: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    installer: Callable[[Path, Path], Path] | None = None,
    health_check: Callable[[Path, Mapping[str, Any], Path | None], bool] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    report = verify_distribution(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    install = installer or (lambda source, root: install_release(source, root, runner=runner))
    release = install(manifest_path, install_root)
    releases = install_root / 'releases'
    current = install_root / 'current'
    previous = _read_current(current, releases)
    identity = candidate_identity(release, runner=runner) if health_check is None else {
        'runtime_build_hash': manifest['runtime_build_hash'],
        'catalog_hash': manifest['catalog_hash'],
        'provider_package_hash': manifest['provider_package_hash'],
    }
    expected_identity = {
        'runtime_build_hash': manifest['runtime_build_hash'],
        'catalog_hash': manifest['catalog_hash'],
        'provider_package_hash': manifest['provider_package_hash'],
    }
    if identity != expected_identity:
        raise ActivationError('candidate artifact identity mismatch')
    backup = None
    if vault is not None and previous != release:
        backup = create_upgrade_backup(vault, install_root)
    _switch(current, release)
    try:
        if health_check is not None:
            healthy = bool(health_check(release, manifest, vault))
        elif vault is not None:
            _runtime_command(release, vault, 'start', runner=runner)
            status = _runtime_command(release, vault, 'status', runner=runner)
            healthy = status.get('data', {}).get('state') == 'compatible'
        else:
            healthy = True
        if not healthy:
            raise ActivationError('candidate health check failed')
        if config_path is not None:
            if config_path.is_file():
                existing = load_product_config(config_path)
                config = ProductConfig(
                    vault_root=vault or existing.vault_root,
                    runtime_root=current.resolve(strict=True),
                    provider_id=manifest['provider']['provider_id'],
                    mcp_pack=existing.mcp_pack,
                    secret_names=existing.secret_names,
                    explicit=True,
                )
            else:
                config = ProductConfig(
                    vault_root=vault,
                    runtime_root=current.resolve(strict=True),
                    provider_id=manifest['provider']['provider_id'],
                    explicit=True,
                )
            write_product_config(config_path, config)
    except BaseException:
        if vault is not None:
            try:
                _runtime_command(release, vault, 'stop', runner=runner)
            except Exception:
                pass
        if previous is None:
            current.unlink(missing_ok=True)
        else:
            _switch(current, previous)
            if vault is not None:
                _runtime_command(previous, vault, 'start', runner=runner)
        raise
    retained = _prune_releases(
        releases,
        keep={item for item in (release, previous) if item is not None},
    )
    return {
        'ok': True,
        'distribution_set_sha256': report['distribution_set_sha256'],
        'activated': True,
        'previous_release_present': previous is not None,
        'runtime_health_checked': vault is not None or health_check is not None,
        'retained_releases': retained,
        'upgrade_backup_created': backup is not None,
        'retained_upgrade_backups': (
            int(backup['retained_backups']) if backup is not None else 0
        ),
        'private_paths_included': False,
        'secret_values_included': False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--install-root', type=Path, default=Path.home() / '.local/share/trove')
    parser.add_argument('--vault', type=Path)
    parser.add_argument('--config', type=Path, default=Path.home() / '.config/trove/config.json')
    args = parser.parse_args(argv)
    try:
        report = activate_distribution(
            args.manifest.expanduser(), args.install_root.expanduser(),
            vault=args.vault.expanduser().resolve(strict=True) if args.vault else None,
            config_path=args.config.expanduser(),
        )
    except UpgradeCapacityError:
        report = {
            'ok': False, 'error_code': 'insufficient_upgrade_capacity',
            'private_paths_included': False, 'secret_values_included': False,
        }
    except Exception:
        report = {
            'ok': False, 'error_code': 'distribution_activation_failed',
            'private_paths_included': False, 'secret_values_included': False,
        }
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
