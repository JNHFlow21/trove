from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Mapping

from trove_core.vault.config import VaultConfig

from .service import ReplyServiceConfig, _private_write_json
from .store import ReplyStore


WORK_CONTAINER = 'com.tencent.xinWeChat2'
DEFAULT_WORK_ROOT = (
    Path.home()
    / 'Library/Containers'
    / WORK_CONTAINER
    / 'Data/Documents/xwechat_files'
)


_DIGEST = re.compile(r'^[0-9a-f]{64}$')


class ReplyMigrationError(RuntimeError):
    code = 'reply_migration_failed'


def _read_private_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ReplyMigrationError('legacy reply input is unavailable')
    listed = path.stat()
    if listed.st_uid != os.getuid() or listed.st_mode & 0o022:
        raise ReplyMigrationError('legacy reply input permissions are unsafe')
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplyMigrationError('legacy reply input is invalid') from exc
    if not isinstance(payload, dict):
        raise ReplyMigrationError('legacy reply input must be an object')
    return payload, hashlib.sha256(data).hexdigest()


def _source_account(
    account_hash: str,
    *,
    work_root: Path,
) -> str:
    if _DIGEST.fullmatch(account_hash) is None or not work_root.is_dir():
        raise ReplyMigrationError('legacy source account binding is invalid')
    matches = [
        path.name
        for path in work_root.iterdir()
        if (
            path.is_dir()
            and not path.is_symlink()
            and hashlib.sha256(path.name.encode('utf-8')).hexdigest()
            == account_hash
        )
    ]
    if len(matches) != 1:
        raise ReplyMigrationError('legacy source account is ambiguous')
    return matches[0]


def _vault_account_binding(
    cfg: VaultConfig,
    *,
    source_account_id: str,
    container_name: str,
) -> tuple[str, str]:
    if container_name != WORK_CONTAINER:
        raise ReplyMigrationError('legacy client is not the supported Work client')
    namespace = f'{container_name}__{source_account_id}'
    snapshot = (
        cfg.root
        / 'sources'
        / 'wechat-kos-decrypted'
        / 'current'
        / namespace
    )
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ReplyMigrationError('canonical reply source snapshot is unavailable')
    account_id = (
        'acct-'
        + hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:12]
    )
    database = cfg.paths.sqlite_path
    if not database.is_file():
        raise ReplyMigrationError('Vault index is unavailable')
    uri = database.resolve().as_uri() + '?mode=ro'
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            account_rows = int(conn.execute(
                'SELECT COUNT(*) FROM accounts WHERE account_id=?',
                (account_id,),
            ).fetchone()[0])
            conversation_rows = int(conn.execute(
                'SELECT COUNT(*) FROM conversations WHERE account_id=?',
                (account_id,),
            ).fetchone()[0])
    except sqlite3.Error as exc:
        raise ReplyMigrationError('Vault account binding cannot be verified') from exc
    if account_rows != 1 or conversation_rows < 1:
        raise ReplyMigrationError('Vault account binding is not exact')
    return account_id, namespace


def _relative_style(value: Any) -> str:
    text = str(value or 'profile/style.md')
    path = Path(text)
    return (
        text
        if text and not path.is_absolute() and '..' not in path.parts
        else 'profile/style.md'
    )


def _legacy_config(
    payload: Mapping[str, Any],
    *,
    source_account_id: str,
    account_id: str,
    namespace: str,
) -> ReplyServiceConfig:
    backend = str(payload.get('reply_backend') or 'codex')
    model_value = (
        payload.get('codex_model')
        if backend == 'codex'
        else payload.get('api_model')
    )
    model = str(model_value or '')
    allowed = payload.get('allowed_target_refs') or []
    if not isinstance(allowed, list):
        raise ReplyMigrationError('legacy target allowlist is invalid')
    return ReplyServiceConfig(
        # Shadow is structurally unable to review or send. Enabling its poller
        # here does not grant delivery authority.
        armed=True,
        mode='shadow',
        provider_id='wechat-source',
        account_id=account_id,
        source_account_id=source_account_id,
        account_id_sha256=str(payload.get('account_id_sha256') or ''),
        conversation_namespace=namespace,
        key_store_secret=str(
            payload.get('key_store_secret') or 'TROVE_WECHAT_KEY_STORE',
        ),
        poll_seconds=float(payload.get('poll_seconds') or 1.0),
        send_shortcut=str(payload.get('send_shortcut') or 'unconfigured'),
        max_reply_chars=int(payload.get('max_reply_chars') or 500),
        cooldown_seconds=float(payload.get('cooldown_seconds') or 15.0),
        daily_send_limit=int(payload.get('daily_send_limit') or 300),
        target_scope=str(payload.get('target_scope') or 'allowlist'),
        allowed_target_refs=tuple(str(item) for item in allowed),
        agent_id='default-reply-agent',
        reply_backend=backend,
        model=model or 'gpt-5.6-terra',
        api_base_url=str(payload.get('api_base_url') or ''),
        api_key_secret=str(payload.get('api_key_secret') or ''),
        style_profile_path=_relative_style(payload.get('style_profile_path')),
        session_idle_days=float(
            payload.get('codex_session_idle_release_days') or 3.0,
        ),
        context_message_cap=int(payload.get('context_message_cap') or 50),
        generation_prestart_ms=int(
            payload.get('generation_prestart_ms') or 3_000,
        ),
        round_quiet_min_ms=int(
            payload.get('round_quiet_min_ms') or 6_000,
        ),
        round_quiet_default_ms=int(
            payload.get('round_quiet_default_ms') or 8_000,
        ),
        round_quiet_max_ms=int(
            payload.get('round_quiet_max_ms') or 15_000,
        ),
        round_max_collect_ms=int(
            payload.get('round_max_collect_ms') or 60_000,
        ),
        round_max_messages=int(
            payload.get('round_max_messages') or 50,
        ),
    )


def migrate_legacy_reply_runtime(
    vault_root: str | Path,
    *,
    legacy_config_path: str | Path,
    legacy_state_path: str | Path,
    work_root: str | Path = DEFAULT_WORK_ROOT,
    now: float | None = None,
) -> dict[str, Any]:
    """Idempotently seed the daemon-owned shadow runtime from legacy state."""

    cfg = VaultConfig.resolve(str(vault_root), env={})
    cfg.ensure()
    legacy_config, config_hash = _read_private_json(
        Path(legacy_config_path).expanduser(),
    )
    legacy_state, state_hash = _read_private_json(
        Path(legacy_state_path).expanduser(),
    )
    account_hash = str(legacy_config.get('account_id_sha256') or '')
    source_account_id = _source_account(
        account_hash,
        work_root=Path(work_root).expanduser(),
    )
    account_id, namespace = _vault_account_binding(
        cfg,
        source_account_id=source_account_id,
        container_name=str(
            legacy_config.get('container_name') or WORK_CONTAINER,
        ),
    )
    migrated = _legacy_config(
        legacy_config,
        source_account_id=source_account_id,
        account_id=account_id,
        namespace=namespace,
    )
    marker_path = cfg.root / 'jobs' / 'reply' / 'migration.json'
    config_path = ReplyServiceConfig.path_for_vault(cfg.root)
    replayed = marker_path.is_file()
    if replayed:
        marker, _marker_hash = _read_private_json(marker_path)
        if marker.get('legacy_config_sha256') != config_hash:
            raise ReplyMigrationError('legacy migration source changed')
        current = ReplyServiceConfig.load(cfg.root)
        if (
            current.account_id != migrated.account_id
            or current.source_account_id != migrated.source_account_id
            or current.account_id_sha256 != migrated.account_id_sha256
            or current.conversation_namespace
            != migrated.conversation_namespace
        ):
            raise ReplyMigrationError('existing reply account binding changed')
        if current.mode == 'shadow' and not current.armed:
            current = replace(current, armed=True)
            current.save(cfg.root)
    else:
        if config_path.exists():
            raise ReplyMigrationError('reply config already exists')
        migrated.save(cfg.root)

    cursors = legacy_state.get('cursors') or {}
    if not isinstance(cursors, dict):
        raise ReplyMigrationError('legacy reply cursors are invalid')
    store = ReplyStore.for_vault(cfg.root)
    timestamp = float(time.time() if now is None else now)
    accepted = 0
    for target_ref, position in cursors.items():
        if (
            not isinstance(target_ref, str)
            or _DIGEST.fullmatch(target_ref) is None
            or type(position) is not int
            or position < 0
        ):
            raise ReplyMigrationError('legacy reply cursor is invalid')
        store.advance_cursor(target_ref, position, now=timestamp)
        accepted += 1
    store.add_activity(
        'legacy_migration',
        state='replayed' if replayed else 'completed',
        now=timestamp,
    )
    _private_write_json(marker_path, {
        'schema_version': 1,
        'legacy_config_sha256': config_hash,
        'latest_legacy_state_sha256': state_hash,
        'account_ref': hashlib.sha256(
            account_id.encode('utf-8'),
        ).hexdigest()[:16],
        'cursor_count': accepted,
        'updated_at': timestamp,
        'source_values_included': False,
    })
    return {
        'ok': True,
        'status': 'replayed' if replayed else 'completed',
        'mode': 'shadow',
        'armed': True,
        'delivery_enabled': False,
        'cursor_count': accepted,
        'account_ref': hashlib.sha256(
            account_id.encode('utf-8'),
        ).hexdigest()[:16],
        'source_values_included': False,
        'secret_values_included': False,
    }


__all__ = [
    'DEFAULT_WORK_ROOT',
    'ReplyMigrationError',
    'migrate_legacy_reply_runtime',
]
