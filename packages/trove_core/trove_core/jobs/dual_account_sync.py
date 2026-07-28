from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterator

from trove_core.vault.config import VaultConfig
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.mutations import coordinated_vault_mutation
from trove_core.wechat.decrypt import DecryptConfig, SelectedAccount, build_decrypt_plan, run_decrypt_plan
from trove_core.wechat.decrypt.manifest import MANIFEST_NAME
from trove_core.wechat.decrypt.path_safety import resolved_under
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter


CONFIG_VERSION = 1
DEFAULT_CONFIG_NAME = 'dual_account_sync.private.json'
STATE_NAME = 'dual_account_sync_state.redacted.json'
LOCK_NAME = '.dual_account_sync.lock'
OUTPUT_SOURCE_NAME = 'wechat-integrated-decrypted'


@dataclass(frozen=True)
class DualAccountSyncConfig:
    live_root: Path
    selected_accounts: tuple[SelectedAccount, ...]
    secret_name: str
    retained_runs: int = 2
    persist_account_identity: bool = True

    @classmethod
    def from_path(cls, path: Path) -> 'DualAccountSyncConfig':
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_mode & 0o077:
            raise ValueError('sync_config_not_private')
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('version') != CONFIG_VERSION:
            raise ValueError('sync_config_invalid')
        raw_accounts = payload.get('selected_accounts')
        if not isinstance(raw_accounts, list) or not 1 <= len(raw_accounts) <= 32:
            raise ValueError('sync_accounts_invalid')
        accounts: list[SelectedAccount] = []
        for item in raw_accounts:
            if not isinstance(item, dict):
                raise ValueError('sync_accounts_invalid')
            account_id = str(item.get('account_id') or '').strip()
            container_id = str(item.get('container_id') or '').strip()
            root_name = str(item.get('root_name') or '').strip()
            output_name = str(item.get('output_name') or '').strip()
            if not account_id or not container_id or not root_name or not output_name:
                raise ValueError('sync_accounts_invalid')
            accounts.append(SelectedAccount(
                account_id=account_id,
                container_id=container_id,
                root_name=root_name,
                output_name=output_name,
                secret_name=str(payload.get('secret_name') or '').strip(),
            ))
        secret_name = str(payload.get('secret_name') or '').strip()
        if not secret_name:
            raise ValueError('sync_secret_name_missing')
        retained = int(payload.get('retained_runs') or 2)
        persist_identity = payload.get('persist_account_identity', True)
        if type(persist_identity) is not bool:
            raise ValueError('sync_identity_policy_invalid')
        return cls(
            live_root=Path(str(payload['live_root'])).expanduser(),
            selected_accounts=tuple(accounts),
            secret_name=secret_name,
            retained_runs=max(1, min(3, retained)),
            persist_account_identity=persist_identity,
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | int(getattr(os, 'O_NOFOLLOW', 0)),
        0o600,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('short state write')
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    path.chmod(0o600)


@contextmanager
def _exclusive_job(lock_path: Path) -> Iterator[bool]:
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield acquired
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _source_digest(plan: Any) -> str:
    digest = hashlib.sha256()
    for item in sorted(plan.files, key=lambda value: str(value.source_path)):
        info = item.source_path.stat()
        digest.update(item.account_ref_hash.encode('ascii'))
        digest.update(item.source_path.name.encode('utf-8'))
        digest.update(str(info.st_size).encode('ascii'))
        digest.update(str(info.st_mtime_ns).encode('ascii'))
    return digest.hexdigest()


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='ascii'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sync_attempt(state: dict[str, Any]) -> int:
    value = state.get('sync_attempt', 0)
    return value if type(value) is int and 0 <= value <= 1_000_000 else 0


def _terminal_manifest(run: Path) -> dict[str, Any] | None:
    manifest = run / MANIFEST_NAME
    try:
        info = manifest.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > 1024 * 1024:
            return None
        payload = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get('run_id') != run.name:
        return None
    ok = payload.get('ok')
    status_value = payload.get('status')
    if (
        type(ok) is not bool
        or type(status_value) is not str
        or (
            ok
            and status_value not in {'completed', 'completed_with_account_gaps'}
        )
        or (not ok and status_value != 'failed')
    ):
        return None
    return payload


def _bound_run_identities(
    vault: VaultConfig,
    *,
    runs_root: Path,
    candidates: dict[str, tuple[Path, tuple[int, int]]],
) -> set[tuple[int, int]]:
    """Return decrypt generations still referenced by media bindings.

    A failed or partially completed sync may leave media bindings on the
    previous immutable snapshot.  Retention must preserve those generations
    without preserving every historical generation.
    """

    database = vault.paths.sqlite_path
    if not database.is_file():
        return set()
    uri = f'{database.resolve().as_uri()}?mode=ro'
    try:
        with sqlite3.connect(uri, uri=True, timeout=5.0) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    """SELECT name FROM sqlite_master
                         WHERE type='table'
                           AND name IN ('media_source_bindings','source_snapshots')"""
                )
            }
            if tables != {'media_source_bindings', 'source_snapshots'}:
                return set()
            root_refs = [
                str(row[0])
                for row in connection.execute(
                    """SELECT DISTINCT snapshot.root_ref
                         FROM media_source_bindings binding
                         JOIN source_snapshots snapshot
                           ON snapshot.snapshot_revision=binding.snapshot_revision
                        WHERE snapshot.root_ref IS NOT NULL
                          AND snapshot.root_ref!=''"""
                )
            ]
    except sqlite3.Error as exc:
        raise RuntimeError('snapshot_binding_scan_failed') from exc

    protected: set[tuple[int, int]] = set()
    vault_root = vault.root.resolve()
    for raw_ref in root_refs:
        relative = Path(raw_ref)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('snapshot_binding_ref_invalid')
        lexical = vault_root / relative
        try:
            within_runs = lexical.relative_to(runs_root)
        except ValueError:
            # Other registered source families are outside this retention lane.
            continue
        if not within_runs.parts:
            raise RuntimeError('snapshot_binding_ref_invalid')
        candidate = candidates.get(within_runs.parts[0])
        if candidate is None:
            # A binding may already describe a generation removed by an older
            # runtime.  There is no remaining local directory to protect.
            continue
        run_path, identity = candidate
        try:
            lexical.resolve(strict=False).relative_to(run_path.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError('snapshot_binding_ref_invalid') from exc
        protected.add(identity)
    return protected


def _nonterminal_run_count(vault: VaultConfig) -> int:
    """Count unpublished generations without deleting uncertain live work."""

    runs = vault.root / 'sources' / OUTPUT_SOURCE_NAME / 'runs'
    try:
        return sum(
            1
            for path in runs.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and _terminal_manifest(path) is None
        )
    except FileNotFoundError:
        return 0
    except OSError:
        return 1


def _prune_runs(vault: VaultConfig, *, retained: int) -> dict[str, int]:
    base = vault.root / 'sources' / OUTPUT_SOURCE_NAME
    runs = base / 'runs'
    try:
        with coordinated_vault_mutation(vault.root, operation='decrypt_snapshot'):
            runs_root = runs.resolve(strict=True)
            candidate_paths = sorted(
                (path for path in runs.iterdir() if path.is_dir() and not path.is_symlink()),
                key=lambda path: path.name,
            )
            candidates: list[tuple[Path, tuple[int, int]]] = []
            for path in candidate_paths:
                if _terminal_manifest(path) is None:
                    continue
                info = path.stat()
                candidates.append((path, (int(info.st_dev), int(info.st_ino))))
            candidates_by_name = {
                path.name: (path, identity)
                for path, identity in candidates
            }

            current_path = base / 'current'
            try:
                current_info = current_path.lstat()
            except FileNotFoundError:
                current_identity: tuple[int, int] | None = None
            else:
                if not stat.S_ISLNK(current_info.st_mode):
                    return {
                        'removed': 0,
                        'retained': len(candidates),
                        'locked': 0,
                        'invalid_current': 1,
                    }
                try:
                    current = current_path.resolve(strict=True)
                    target_info = current.lstat()
                except (OSError, RuntimeError):
                    return {
                        'removed': 0,
                        'retained': len(candidates),
                        'locked': 0,
                        'invalid_current': 1,
                    }
                current_identity = (int(target_info.st_dev), int(target_info.st_ino))
                matching = [
                    path
                    for path, identity in candidates
                    if identity == current_identity
                ]
                if (
                    current.parent != runs_root
                    or not stat.S_ISDIR(target_info.st_mode)
                    or stat.S_ISLNK(target_info.st_mode)
                    or len(matching) != 1
                    or (_terminal_manifest(current) or {}).get('ok') is not True
                ):
                    return {
                        'removed': 0,
                        'retained': len(candidates),
                        'locked': 0,
                        'invalid_current': 1,
                    }

            protected = _bound_run_identities(
                vault,
                runs_root=runs_root,
                candidates=candidates_by_name,
            )
            keep: set[tuple[int, int]] = protected | (
                {current_identity} if current_identity is not None else set()
            )
            for _path, identity in reversed(candidates):
                if len(keep) >= retained:
                    break
                keep.add(identity)
            removed = 0
            for path, identity in candidates:
                if identity in keep:
                    continue
                shutil.rmtree(path, ignore_errors=True)
                if not path.exists():
                    removed += 1
            return {
                'removed': removed,
                'retained': len(keep),
                'protected': len(protected),
                'locked': 0,
                'invalid_current': 0,
            }
    except VaultOperationLocked:
        # Retention is destructive and must not race a decrypt that is still
        # building outside its short publication writer window.
        return {'removed': 0, 'retained': 0, 'locked': 1, 'invalid_current': 0}
    except (OSError, RuntimeError):
        return {'removed': 0, 'retained': 0, 'locked': 0, 'invalid_current': 1}


def _account_ids(snapshot: Path) -> tuple[str, ...]:
    values: list[str] = []
    for child in sorted(snapshot.iterdir(), key=lambda path: path.name):
        if child.is_dir() and not child.is_symlink():
            try:
                values.append(WeChatDecryptedAccountImporter(child).account_id)
            except (OSError, ValueError):
                continue
    return tuple(dict.fromkeys(value for value in values if value))


def _run_trove_sync(
    vault: VaultConfig,
    *,
    account_ids: tuple[str, ...],
    idempotency_key: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    executable = Path(sys.executable).with_name('trove')
    arguments = [
        str(executable), '--vault', str(vault.root), '--timeout', '300',
        'sync', '--idempotency-key', idempotency_key,
    ]
    for account_id in account_ids:
        arguments.extend(('--account-ids', account_id))
    try:
        completed = runner(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=330,
        )
    except subprocess.TimeoutExpired:
        return {
            'ok': False,
            'status': 'transport_unknown',
            'returncode': None,
            'stdout_bytes': 0,
            'stderr_bytes': 0,
        }
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError):
        payload = {}
    if completed.returncode or not isinstance(payload, dict) or payload.get('ok') is not True:
        return {
            'ok': False,
            'status': 'sync_failed',
            'returncode': completed.returncode,
            'stdout_bytes': len((completed.stdout or '').encode('utf-8')),
            'stderr_bytes': len((completed.stderr or '').encode('utf-8')),
        }
    operation = (payload.get('data') or {}).get('operation') or {}
    result = operation.get('result') or {}
    return {
        'ok': operation.get('state') == 'completed',
        'status': operation.get('state'),
        'messages_imported': int(result.get('messages_imported') or 0),
        'sources_seen': int(result.get('sources_seen') or 0),
        'elapsed_ms': float(result.get('elapsed_ms') or 0.0),
    }


def run_once(
    vault_root: str | Path,
    *,
    config_path: str | Path,
    engine: Any | None = None,
    secret_resolver: Any | None = None,
    sync_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    started = time.monotonic()
    vault = VaultConfig.resolve(str(vault_root), env={})
    vault.ensure()
    private_path = Path(config_path).expanduser()
    if not resolved_under(private_path, vault.paths.jobs_dir):
        return {'ok': False, 'status': 'config_path_escape'}
    try:
        job = DualAccountSyncConfig.from_path(private_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {'ok': False, 'status': str(exc) if str(exc).startswith('sync_') else 'config_invalid'}
    with _exclusive_job(vault.paths.jobs_dir / LOCK_NAME) as acquired:
        if not acquired:
            return {'ok': True, 'status': 'already_running'}
        decrypt_config = DecryptConfig(
            live_root=job.live_root,
            vault_root=vault.root,
            selected_accounts=job.selected_accounts,
            secret_name=job.secret_name,
            output_source_name=OUTPUT_SOURCE_NAME,
            fail_on_unselected_snapshot_account=True,
            allow_partial_accounts=False,
            persist_account_identity=job.persist_account_identity,
        )
        plan = build_decrypt_plan(decrypt_config)
        if not plan.ok or not plan.files:
            return {
                'ok': False,
                'status': 'preflight_failed',
                'selected_accounts': len(job.selected_accounts),
                'planned_files': len(plan.files),
            }
        try:
            digest = _source_digest(plan)
        except OSError:
            return {'ok': False, 'status': 'source_changed_during_preflight'}
        state_path = vault.paths.jobs_dir / STATE_NAME
        state = _read_state(state_path)
        current = vault.root / 'sources' / OUTPUT_SOURCE_NAME / 'current'
        same_source = state.get('source_digest') == digest and current.exists()
        if same_source and state.get('last_status') == 'completed':
            current_account_ids = _account_ids(current)
            if len(current_account_ids) == len(job.selected_accounts):
                retention = _prune_runs(vault, retained=job.retained_runs)
                return {
                    'ok': True,
                    'status': 'unchanged',
                    'selected_accounts': len(job.selected_accounts),
                    'retention': retention,
                }
            # A stale state file must never bless a partial/corrupt snapshot.
            # Rebuild from the already-validated source plan instead.
            same_source = False
        decrypt: dict[str, Any]
        retention = {
            'removed': 0,
            'retained': 0,
            'locked': 0,
            'invalid_current': 0,
            'skipped': 1,
        }
        if same_source:
            decrypt = {
                'ok': True,
                'status': 'reused_pending_snapshot',
                'summary': {},
                'current_switched': True,
            }
        else:
            nonterminal_runs = _nonterminal_run_count(vault)
            if nonterminal_runs:
                # A normal Python exception removes its unpublished run.  A
                # remaining nonterminal directory therefore represents either
                # concurrent decrypt work or a hard-interrupted generation.
                # Never create another large snapshot beside uncertain work.
                return {
                    'ok': False,
                    'status': 'snapshot_build_present',
                    'nonterminal_runs': nonterminal_runs,
                    'raw_paths_included': False,
                    'secret_values_included': False,
                }
            try:
                decrypt = run_decrypt_plan(
                    plan,
                    engine=engine,
                    secret_resolver=secret_resolver,
                )
            except BaseException:
                # A failed publication never changed ``current``; retention is
                # therefore safe and still cleans older completed generations.
                _prune_runs(vault, retained=job.retained_runs)
                raise
            if not decrypt.get('ok') or not decrypt.get('current_switched'):
                retention = _prune_runs(vault, retained=job.retained_runs)
                return {
                    'ok': False,
                    'status': 'decrypt_failed',
                    'selected_accounts': len(job.selected_accounts),
                    'summary': dict(decrypt.get('summary') or {}),
                    'retention': retention,
                }
        account_ids = _account_ids(current)
        if len(account_ids) != len(job.selected_accounts):
            return {
                'ok': False,
                'status': 'account_set_mismatch',
                'expected_accounts': len(job.selected_accounts),
                'snapshot_accounts': len(account_ids),
                'retention': retention,
            }
        attempt = _sync_attempt(state) if same_source else 0
        _atomic_json(state_path, {
            'version': 1,
            'source_digest': digest,
            'selected_account_count': len(account_ids),
            'sync_attempt': attempt,
            'last_status': 'sync_pending',
            'raw_paths_included': False,
            'secret_values_included': False,
        })
        synced = _run_trove_sync(
            vault,
            account_ids=account_ids,
            idempotency_key=f'dual-sync-v2-{digest}-attempt-{attempt}',
            runner=sync_runner,
        )
        if synced.get('ok'):
            _atomic_json(state_path, {
                'version': 1,
                'source_digest': digest,
                'selected_account_count': len(account_ids),
                'sync_attempt': attempt,
                'last_status': 'completed',
                'raw_paths_included': False,
                'secret_values_included': False,
            })
        elif synced.get('status') == 'failed':
            _atomic_json(state_path, {
                'version': 1,
                'source_digest': digest,
                'selected_account_count': len(account_ids),
                'sync_attempt': attempt + 1,
                'last_status': 'sync_retry',
                'raw_paths_included': False,
                'secret_values_included': False,
            })
        # Failed syncs are safe to prune once every media-bound generation is
        # protected explicitly.  Keep at least current + one retry generation
        # when the sync result is not successful.
        retention = _prune_runs(
            vault,
            retained=(
                job.retained_runs
                if synced.get('ok')
                else max(2, job.retained_runs)
            ),
        )
        return {
            'ok': bool(synced.get('ok')),
            'status': 'completed' if synced.get('ok') else 'sync_failed',
            'selected_accounts': len(account_ids),
            'decrypt': {
                'status': decrypt.get('status'),
                'summary': dict(decrypt.get('summary') or {}),
                'elapsed_ms': float(decrypt.get('elapsed_ms') or 0.0),
            },
            'sync': synced,
            'retention': retention,
            'elapsed_ms': round((time.monotonic() - started) * 1000, 3),
            'raw_paths_included': False,
            'secret_values_included': False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--vault', required=True)
    parser.add_argument('--config')
    args = parser.parse_args(argv)
    vault = VaultConfig.resolve(args.vault, env={})
    config = Path(args.config).expanduser() if args.config else vault.paths.jobs_dir / DEFAULT_CONFIG_NAME
    try:
        report = run_once(vault.root, config_path=config)
    except BaseException as exc:
        report = {
            'ok': False,
            'status': 'job_failed',
            'failure_type': exc.__class__.__name__,
            'raw_paths_included': False,
            'secret_values_included': False,
        }
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(',', ':')))
    return 0 if report.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
