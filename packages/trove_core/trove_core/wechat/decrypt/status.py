from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import (
    coordinated_vault_mutation,
    mutation_entrypoint,
    record_vault_mutation_noop,
)

from .manifest import MANIFEST_NAME
from .redaction import redact_obj


def _base(vault_root: Path, output_source_name: str = 'wechat-integrated-decrypted') -> Path:
    return vault_root / 'sources' / output_source_name


def decrypt_status(vault_root: Path, *, run_id: str | None = None, output_source_name: str = 'wechat-integrated-decrypted') -> dict[str, Any]:
    base = _base(vault_root, output_source_name)
    runs_dir = base / 'runs'
    current = base / 'current'
    runs: list[dict[str, Any]] = []
    if run_id:
        candidates = [runs_dir / run_id]
    else:
        candidates = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)[:20] if runs_dir.exists() else []
    for run in candidates:
        manifest = run / MANIFEST_NAME
        payload: dict[str, Any]
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                payload = {'ok': False, 'status': 'manifest_invalid'}
        else:
            payload = {'ok': False, 'status': 'manifest_missing'}
        runs.append(redact_obj({'run_id': run.name, 'manifest': payload}))
    current_target = None
    if current.exists() or current.is_symlink():
        try:
            current_target = current.resolve().name
        except OSError:
            current_target = '<unresolved>'
    return {
        'ok': True,
        'output_source_name': output_source_name,
        'current_run_id': current_target,
        'runs': runs,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def known_keyed_account_refs(
    vault_root: Path,
    *,
    output_source_name: str = 'wechat-integrated-decrypted',
) -> set[str]:
    """Return opaque refs for accounts proven complete in the current run.

    The refs are hashes of local account-root paths.  No account names, paths,
    or key material are returned or persisted by this helper.
    """

    current = _base(vault_root, output_source_name) / 'current'
    manifest = current / MANIFEST_NAME
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return set()
    files = payload.get('files') if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return set()
    productive: set[str] = set()
    failed: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            continue
        account_ref = str(item.get('account_ref_hash') or '').strip()
        status = str(item.get('status') or '')
        if not account_ref:
            continue
        if status in {'decrypted', 'copied_plaintext', 'reused'}:
            productive.add(account_ref)
        elif status not in {'skipped'}:
            failed.add(account_ref)
    return productive - failed


@mutation_entrypoint('decrypt_snapshot')
def rollback_current(
    vault_root: Path,
    *,
    run_id: str,
    output_source_name: str = 'wechat-integrated-decrypted',
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    base = _base(vault_root, output_source_name)
    run_dir = base / 'runs' / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        record_vault_mutation_noop(operation='decrypt_snapshot')
        return {'ok': False, 'status': 'missing_run', 'run_id': run_id, 'raw_paths_included': False}
    with coordinated_vault_mutation(
        vault_root,
        operation='decrypt_snapshot',
        write_session=write_session,
    ):
        current = base / 'current'
        tmp = base / f'.current-rollback-{run_id}.tmp'
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(run_dir, target_is_directory=True)
        os.replace(tmp, current)
    return {'ok': True, 'status': 'current_switched', 'run_id': run_id, 'raw_paths_included': False}
