from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable

from trove_core.sync import SyncOptions, run_sync
from trove_core.vault.config import VaultConfig
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.mutations import coordinated_vault_mutation
from trove_core.watch import ManifestPollingBackend, WatchBackend

from .decrypt.config import DecryptConfig, DecryptFilePlan, DecryptPlan, SelectedAccount
from .decrypt.manifest import load_account_identity
from .decrypt.path_safety import resolved_under
from .decrypt.redaction import stable_hash
from .decrypt.runner import DecryptEngine, run_decrypt_plan


CONFIG_VERSION = 1
DEFAULT_OUTPUT_SOURCE_NAME = 'wechat-realtime-decrypted'
WATCH_MANIFEST_NAME = 'realtime_bridge_watch_manifest.redacted.json'
_SAFE_COMPONENT_RE = re.compile(r'[^A-Za-z0-9_.@-]+')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_COMPONENT_RE.sub('_', str(value or '').strip())[:180].strip('._')
    return cleaned or fallback


def _canonical_wxid(value: str) -> str:
    match = re.search(r'wxid_[A-Za-z0-9]+', str(value or ''))
    return match.group(0) if match else ''


@dataclass(frozen=True)
class RealtimeBridgeConfig:
    trusted_root: Path
    state_path: Path
    snapshot_root: Path
    contact_root: Path
    key_store_path: Path
    output_source_name: str = DEFAULT_OUTPUT_SOURCE_NAME
    debounce_seconds: float = 3.0
    retained_runs: int = 2

    @classmethod
    def from_path(cls, path: str | Path) -> RealtimeBridgeConfig:
        config_path = Path(path).expanduser()
        payload = json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or int(payload.get('version') or 0) != CONFIG_VERSION:
            raise ValueError('invalid_realtime_bridge_config')
        output_source_name = _safe_component(
            str(payload.get('output_source_name') or DEFAULT_OUTPUT_SOURCE_NAME),
            fallback=DEFAULT_OUTPUT_SOURCE_NAME,
        )
        return cls(
            trusted_root=Path(str(payload['trusted_root'])).expanduser(),
            state_path=Path(str(payload['state_path'])).expanduser(),
            snapshot_root=Path(str(payload['snapshot_root'])).expanduser(),
            contact_root=Path(str(payload['contact_root'])).expanduser(),
            key_store_path=Path(str(payload['key_store_path'])).expanduser(),
            output_source_name=output_source_name,
            debounce_seconds=max(0.0, min(300.0, float(payload.get('debounce_seconds') or 3.0))),
            retained_runs=max(1, min(10, int(payload.get('retained_runs') or 2))),
        )

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'version': CONFIG_VERSION,
            'trusted_root_configured': True,
            'state_path_configured': True,
            'snapshot_root_configured': True,
            'contact_root_configured': True,
            'key_store_configured': True,
            'output_source_name': self.output_source_name,
            'debounce_seconds': self.debounce_seconds,
            'retained_runs': self.retained_runs,
            'raw_paths_included': False,
        }


def _private_file(path: Path, *, trusted_root: Path) -> bool:
    try:
        return path.is_file() and resolved_under(path, trusted_root)
    except OSError:
        return False


def _canonical_account_name(
    cfg: VaultConfig,
    bridge: RealtimeBridgeConfig,
    *,
    account_id: str,
    container_name: str,
) -> str | None:
    account_id = _canonical_wxid(account_id)
    if not account_id:
        return None
    integrated_root = cfg.root / 'sources' / 'wechat-integrated-decrypted' / 'current'
    integrated_accounts: list[Path] = []
    try:
        integrated_accounts = [
            child for child in integrated_root.iterdir()
            if child.is_dir() and not child.is_symlink()
        ]
    except OSError:
        pass
    if integrated_accounts:
        matches = {
            child.name
            for child in integrated_accounts
            if account_id and (
                account_id in child.name
                or load_account_identity(child).get('own_wxid') == account_id
            )
        }
        # Once an integrated keyed-account set exists it is authoritative.
        # Realtime sources that cannot map to exactly one of those accounts are
        # ignored rather than creating a second or garbage account namespace.
        return next(iter(matches)) if len(matches) == 1 else None

    matches: set[str] = set()
    realtime_root = cfg.root / 'sources' / bridge.output_source_name / 'current'
    for root in (realtime_root,):
        try:
            matches.update(
                child.name
                for child in root.iterdir()
                if child.is_dir() and account_id and (
                    account_id in child.name
                    or load_account_identity(child).get('own_wxid') == account_id
                )
            )
        except OSError:
            continue
    if len(matches) == 1:
        return next(iter(matches))
    container = _safe_component(container_name, fallback='wechat')
    account = _safe_component(account_id, fallback='account-' + stable_hash(account_id or container, length=12))
    return f'{container}__{account}'


def _contact_for_source(
    bridge: RealtimeBridgeConfig,
    *,
    account_label: str,
    account_id: str,
) -> Path | None:
    direct = bridge.contact_root / account_label / 'contact.db'
    if _private_file(direct, trusted_root=bridge.trusted_root):
        return direct
    try:
        matches = [
            child / 'contact.db'
            for child in bridge.contact_root.iterdir()
            if child.is_dir() and account_id and account_id in child.name
        ]
    except OSError:
        return None
    ready = [path for path in matches if _private_file(path, trusted_root=bridge.trusted_root)]
    return ready[0] if len(ready) == 1 else None


def _load_live_message_sources(bridge: RealtimeBridgeConfig) -> list[dict[str, Any]]:
    if not _private_file(bridge.state_path, trusted_root=bridge.trusted_root):
        raise ValueError('realtime_state_unavailable')
    try:
        payload = json.loads(bridge.state_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('realtime_state_invalid') from exc
    fast = payload.get('fast_sync') if isinstance(payload, dict) else None
    raw_sources = fast.get('live_sources') if isinstance(fast, dict) else None
    if not isinstance(raw_sources, list):
        raise ValueError('realtime_sources_unavailable')
    return [
        item for item in raw_sources
        if isinstance(item, dict) and str(item.get('component') or '') == 'message'
    ]


def build_realtime_decrypt_plan(
    vault_root: str | Path,
    *,
    config: RealtimeBridgeConfig,
) -> tuple[DecryptPlan, dict[str, int]]:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    trusted_root = config.trusted_root.expanduser()
    required_paths = (
        config.state_path,
        config.snapshot_root,
        config.contact_root,
        config.key_store_path,
    )
    if not trusted_root.is_dir() or any(not resolved_under(path, trusted_root) for path in required_paths):
        raise ValueError('realtime_bridge_path_escape')
    if not _private_file(config.key_store_path, trusted_root=trusted_root):
        raise ValueError('realtime_key_store_unavailable')

    sources = _load_live_message_sources(config)
    grouped: dict[tuple[str, str, str, str], list[Path]] = {}
    skipped = 0
    for source in sources:
        raw_account_id = str(source.get('account_id') or '').strip()
        account_id = _canonical_wxid(raw_account_id)
        account_label = str(source.get('account_label') or '').strip()
        container_name = str(source.get('container_name') or '').strip()
        raw_source = str(source.get('path') or '').strip()
        source_name = Path(raw_source).name
        if not account_id or not account_label or not raw_source or not source_name.startswith('message_') or not source_name.endswith('.db'):
            skipped += 1
            continue
        snapshot_dir = config.snapshot_root / hashlib.sha1(raw_source.encode('utf-8')).hexdigest()[:16]
        snapshot_path = snapshot_dir / source_name
        if not _private_file(snapshot_path, trusted_root=trusted_root):
            skipped += 1
            continue
        canonical_name = _canonical_account_name(
            cfg,
            config,
            account_id=account_id,
            container_name=container_name,
        )
        if canonical_name is None:
            skipped += 1
            continue
        grouped.setdefault((account_id, account_label, container_name, canonical_name), []).append(snapshot_path)

    files: list[DecryptFilePlan] = []
    selected: list[SelectedAccount] = []
    ready_sources = 0
    ready_accounts = 0
    for (account_id, account_label, container_name, canonical_name), snapshots in sorted(grouped.items()):
        contact = _contact_for_source(config, account_label=account_label, account_id=account_id)
        if contact is None:
            skipped += len(snapshots)
            continue
        account_hash = stable_hash(account_id)
        selected.append(SelectedAccount(
            account_id=account_id,
            container_id=container_name or None,
            root_name=canonical_name,
        ))
        ready_accounts += 1
        files.append(DecryptFilePlan(
            account_ref_hash=account_hash,
            account_root=contact.parent,
            source_path=contact,
            file_family='contact',
            secret_name=None,
            output_relative=Path(canonical_name) / 'contact.db',
        ))
        for snapshot in sorted(snapshots, key=lambda path: path.name):
            files.append(DecryptFilePlan(
                account_ref_hash=account_hash,
                account_root=snapshot.parent,
                source_path=snapshot,
                file_family='message',
                secret_name=None,
                output_relative=Path(canonical_name) / snapshot.name,
            ))
            ready_sources += 1

    plan_config = DecryptConfig(
        live_root=trusted_root,
        vault_root=cfg.root,
        selected_accounts=tuple(selected),
        key_store_path=config.key_store_path,
        output_source_name=config.output_source_name,
        allowed_file_families=('message', 'contact'),
        fail_on_unselected_snapshot_account=False,
    )
    errors: tuple[str, ...] = () if files and selected else ('no_realtime_sources_ready',)
    plan = DecryptPlan(
        config=plan_config,
        files=tuple(files),
        errors=errors,
        generated_at=_now(),
    )
    return plan, {
        'sources_seen': len(sources),
        'sources_ready': ready_sources,
        'sources_skipped': skipped,
        'accounts_ready': ready_accounts,
    }


def _sync_summary(report: dict[str, Any]) -> dict[str, Any]:
    snapshot = report.get('snapshot') if isinstance(report.get('snapshot'), dict) else {}
    return {
        'ok': bool(report.get('ok')),
        'status': report.get('status'),
        'sources_seen': int(report.get('sources_seen') or 0),
        'sources_imported': int(report.get('sources_imported') or 0),
        'messages_imported': int(report.get('messages_imported') or 0),
        'conversations_changed': int(report.get('conversations_changed') or 0),
        'waterlines_updated': int(report.get('waterlines_updated') or 0),
        'snapshot_media': snapshot.get('media_cache') or {},
        'errors': list(report.get('errors') or [])[:20],
        'raw_paths_included': False,
        'raw_content_included': False,
    }


def _prune_realtime_runs(cfg: VaultConfig, bridge: RealtimeBridgeConfig) -> dict[str, Any]:
    try:
        with coordinated_vault_mutation(cfg, operation='decrypt_snapshot'):
            base = cfg.root / 'sources' / bridge.output_source_name
            runs_dir = base / 'runs'
            try:
                candidates = sorted(
                    path for path in runs_dir.iterdir()
                    if path.is_dir() and not path.is_symlink() and path.parent == runs_dir
                )
            except OSError:
                candidates = []
            keep: set[str] = set()
            current = base / 'current'
            try:
                current_target = current.resolve()
                if resolved_under(current_target, runs_dir):
                    keep.add(current_target.name)
            except OSError:
                pass
            for path in reversed(candidates):
                if len(keep) >= bridge.retained_runs:
                    break
                keep.add(path.name)
            removed = 0
            errors = 0
            for path in candidates:
                if path.name in keep:
                    continue
                try:
                    shutil.rmtree(path)
                    removed += 1
                except OSError:
                    errors += 1
            try:
                retained = sum(1 for path in runs_dir.iterdir() if path.is_dir() and not path.is_symlink())
            except OSError:
                retained = 0
    except VaultOperationLocked:
        return {
            'retained_runs': 0,
            'removed_runs': 0,
            'errors': 0,
            'skipped_reason': 'writer_lock',
            'raw_paths_included': False,
            'raw_content_included': False,
        }
    return {
        'retained_runs': retained,
        'removed_runs': removed,
        'errors': errors,
        'raw_paths_included': False,
        'raw_content_included': False,
    }


def run_realtime_bridge_once(
    vault_root: str | Path,
    *,
    config: RealtimeBridgeConfig,
    engine: DecryptEngine | None = None,
    sync_runner: Callable[..., dict[str, Any]] = run_sync,
) -> dict[str, Any]:
    started = time.time()
    try:
        plan, counts = build_realtime_decrypt_plan(vault_root, config=config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            'ok': False,
            'status': 'preflight_failed',
            'error_code': str(exc) if str(exc).startswith('realtime_') else exc.__class__.__name__,
            'elapsed_ms': round((time.time() - started) * 1000, 3),
            'raw_paths_included': False,
            'raw_content_included': False,
        }
    if not plan.ok or not plan.files:
        return {
            'ok': False,
            'status': 'no_sources_ready',
            **counts,
            'errors': list(plan.errors),
            'elapsed_ms': round((time.time() - started) * 1000, 3),
            'raw_paths_included': False,
            'raw_content_included': False,
        }
    decrypt = run_decrypt_plan(plan, engine=engine)
    decrypt_summary = {
        'ok': bool(decrypt.get('ok')),
        'status': decrypt.get('status'),
        'summary': dict(decrypt.get('summary') or {}),
        'errors': list(decrypt.get('errors') or [])[:20],
        'current_switched': bool(decrypt.get('current_switched')),
        'raw_paths_included': False,
        'raw_content_included': False,
    }
    if not decrypt_summary['ok'] or not decrypt_summary['current_switched']:
        return {
            'ok': False,
            'status': 'decrypt_failed',
            **counts,
            'decrypt': decrypt_summary,
            'elapsed_ms': round((time.time() - started) * 1000, 3),
            'raw_paths_included': False,
            'raw_content_included': False,
        }
    cfg = VaultConfig.resolve(str(vault_root), env={})
    snapshot_dir = cfg.root / 'sources' / config.output_source_name / 'current'
    sync = sync_runner(
        cfg.root,
        options=SyncOptions(
            snapshot_dir=snapshot_dir,
            snapshot_media_enabled=False,
            media_discovery_mode='message_delta',
        ),
    )
    sync_summary = _sync_summary(sync)
    status = 'completed' if sync_summary['ok'] and counts['sources_skipped'] == 0 else ('partial' if sync_summary['ok'] else 'sync_failed')
    # A failed projection does not make the decrypted snapshot runs valuable.
    # Keeping every failed attempt lets the watcher duplicate the full source
    # databases on each retry and can exhaust the volume in minutes.  Preserve
    # the current target plus the configured newest runs on both success and
    # failure; the dirty journal remains the authority for projection retry.
    retention = _prune_realtime_runs(cfg, config)
    return {
        'ok': status in {'completed', 'partial'},
        'status': status,
        **counts,
        'decrypt': decrypt_summary,
        'sync': sync_summary,
        'retention': retention,
        'elapsed_ms': round((time.time() - started) * 1000, 3),
        'raw_paths_included': False,
        'raw_content_included': False,
    }


def watch_realtime_bridge(
    vault_root: str | Path,
    *,
    config: RealtimeBridgeConfig,
    backend: WatchBackend | None = None,
) -> None:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    cfg.require_configured_for_write('realtime sync')
    initial = run_realtime_bridge_once(cfg.root, config=config)
    print(json.dumps(initial, ensure_ascii=False), flush=True)
    # The signed helper updates sibling state files every few seconds.  A
    # parent-directory kqueue subscription would therefore keep resetting the
    # debounce window even when the snapshot tree itself is stable.  This tree
    # is intentionally tiny, so bounded 1-2 second manifest polling is both
    # cheaper and more reliable than listening to the noisy parent directory.
    active_backend = backend or ManifestPollingBackend(
        config.snapshot_root,
        cfg.paths.jobs_dir / WATCH_MANIFEST_NAME,
        min_backoff_seconds=1.0,
        max_backoff_seconds=2.0,
    )
    pending_change = False
    last_change_at = 0.0
    stable_manifest_digest: str | None = None
    stable_scan_at = 0.0
    try:
        while True:
            tick = active_backend.poll(timeout=1.0)
            now = time.monotonic()
            if tick.changed or tick.event_loss:
                pending_change = True
                last_change_at = now
                stable_manifest_digest = None
                stable_scan_at = 0.0
            if (
                pending_change
                and tick.scan_complete
                and not tick.scan_discarded
                and tick.error_code is None
                and tick.manifest_digest
                and tick.manifest_digest != stable_manifest_digest
            ):
                stable_manifest_digest = tick.manifest_digest
                stable_scan_at = now
            stable_since = max(last_change_at, stable_scan_at)
            if (
                pending_change
                and stable_manifest_digest
                and not tick.scan_active
                and tick.error_code is None
                and now - stable_since >= config.debounce_seconds
            ):
                report = run_realtime_bridge_once(cfg.root, config=config)
                print(json.dumps(report, ensure_ascii=False), flush=True)
                pending_change = False
                stable_manifest_digest = None
                active_backend.request_repair(reason='post_realtime_sync')
    finally:
        active_backend.close()


__all__ = [
    'RealtimeBridgeConfig',
    'build_realtime_decrypt_plan',
    'run_realtime_bridge_once',
    'watch_realtime_bridge',
]
