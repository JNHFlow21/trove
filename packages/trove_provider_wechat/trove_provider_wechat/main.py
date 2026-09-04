from __future__ import annotations

from contextlib import closing

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any, Mapping

from trove_protocol.provider import ProviderManifest, canonical_provider_schema_hash

from .reply import WeChatActionAdapter, WeChatLiveConfig
from .source import (
    WeChatDecryptedAccountImporter,
    current_account_records,
    current_accounts,
    is_current_account,
    load_account_records,
    source_accounts,
)


class WeChatProviderError(RuntimeError):
    code = 'provider_capability_unavailable'


class WeChatProvider:
    """In-process v1 Provider; every output record is already normalized."""

    def __init__(
        self,
        manifest: ProviderManifest,
        *,
        vault_root: str | Path | None = None,
        source_root: str | Path | None = None,
        media_assets: Mapping[str, str | Path] | None = None,
        live_config: WeChatLiveConfig | Mapping[str, Any] | None = None,
        live_source: Any | None = None,
        sender: Any | None = None,
        key_store: Mapping[str, Mapping[str, Any]] | None = None,
        runtime_root: str | Path | None = None,
    ):
        self.manifest = manifest
        self.vault_root = Path(vault_root).expanduser() if vault_root is not None else None
        self.source_root = Path(source_root).expanduser() if source_root is not None else None
        self.media_assets = {str(key): Path(value) for key, value in (media_assets or {}).items()}
        self.action: WeChatActionAdapter | None = None
        if live_config is not None:
            config = (
                live_config
                if isinstance(live_config, WeChatLiveConfig)
                else WeChatLiveConfig(**dict(live_config))
            )
            live_runtime = (
                Path(runtime_root)
                if runtime_root is not None
                else (
                    self.vault_root / 'jobs' / 'reply' / 'provider'
                    if self.vault_root is not None
                    else Path.home() / 'Library/Application Support/TROVE/reply/provider'
                )
            )
            if live_source is None:
                from .reply.live_source import WeChatLiveSource
                live_source = WeChatLiveSource(
                    config,
                    key_store or {},
                    runtime_root=live_runtime,
                )
            if sender is None:
                from .reply.sender import VerifiedSender
                sender = VerifiedSender(
                    config,
                    live_source,
                    runtime_root=live_runtime,
                )
            self.action = WeChatActionAdapter(config, live_source, sender)

    def hello(self) -> Mapping[str, Any]:
        return {
            'provider_id': self.manifest.provider_id,
            'version': self.manifest.version,
            'protocol': 'trove/1',
            'schema_sha256': canonical_provider_schema_hash(),
        }

    def capabilities(self) -> Mapping[str, Any]:
        return {
            'capabilities': list(self.manifest.capabilities),
            'source_types': list(self.manifest.source_types),
        }

    def health(self) -> Mapping[str, Any]:
        return {'ok': True, 'state': 'ready'}

    def accounts(self) -> list[Mapping[str, Any]]:
        if self.source_root is not None:
            discovered = [*source_accounts(self.source_root), *current_accounts(self.source_root)]
            return sorted(discovered, key=lambda item: str(item['account_id']))
        if self.vault_root is None:
            return []
        database = self.vault_root / 'index' / 'trove.sqlite'
        if not database.is_file():
            return []
        uri = database.resolve().as_uri() + '?mode=ro'
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                rows = connection.execute(
                    """SELECT a.account_id,a.label,COUNT(m.id),MAX(m.timestamp)
                         FROM accounts a LEFT JOIN messages m ON m.account_id=a.account_id
                        GROUP BY a.account_id,a.label ORDER BY a.account_id"""
                ).fetchall()
        except sqlite3.Error as exc:
            raise WeChatProviderError('provider account enumeration failed') from exc
        return [
            {
                'account_id': str(row[0]), 'label': str(row[1]),
                'message_count': int(row[2] or 0),
                **({'watermark': str(row[3])} if row[3] is not None else {}),
            }
            for row in rows
        ]

    def invoke(self, method: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if method == 'read':
            return self._read(payload)
        if method == 'media':
            return self._stage_media(payload)
        if method == 'action' and self.action is not None:
            return self.action.invoke(payload)
        raise WeChatProviderError('provider capability is unavailable')

    def _read(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        operation = payload.get('operation', 'accounts')
        if operation == 'accounts':
            return {'accounts': list(self.accounts())}
        if operation != 'changes' or self.source_root is None:
            raise WeChatProviderError('provider read operation is unavailable')
        account_id = payload.get('account_id')
        if not isinstance(account_id, str) or not account_id:
            raise WeChatProviderError('account_id is required')
        matching: list[tuple[Path, str]] = []
        for path in self.source_root.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            metadata_path = path / 'account.json'
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if metadata.get('account_id') == account_id:
                    matching.append((path, 'normalized'))
                continue
            if is_current_account(path):
                if WeChatDecryptedAccountImporter(path).account_id == account_id:
                    matching.append((path, 'current'))
        if len(matching) != 1:
            raise WeChatProviderError('provider account is unavailable')
        directory, source_kind = matching[0]
        if source_kind == 'current':
            records, watermark = current_account_records(directory)
        else:
            records, watermark = load_account_records(directory, account_id=account_id)
        cursor = payload.get('cursor')
        return {
            'account_id': account_id,
            'records': [] if cursor == watermark else records,
            'watermark': watermark,
            'change_cursor': watermark,
            'replayed': cursor == watermark,
        }

    def _stage_media(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        asset_id = payload.get('asset_id')
        staging_path = payload.get('staging_path')
        source = self.media_assets.get(str(asset_id))
        if source is None or not isinstance(staging_path, str):
            raise WeChatProviderError('media asset or staging grant is unavailable')
        listed = source.lstat()
        if not stat.S_ISREG(listed.st_mode) or stat.S_ISLNK(listed.st_mode):
            raise WeChatProviderError('media source is not a regular file')
        digest = hashlib.sha256()
        total = 0
        flags = os.O_WRONLY | int(getattr(os, 'O_NOFOLLOW', 0))
        fd = os.open(staging_path, flags)
        try:
            with source.open('rb') as input_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
                    digest.update(chunk)
                    total += len(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise OSError('short staging write')
                        view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        return {
            'size': total, 'sha256': digest.hexdigest(),
            'encoding': 'staging_path', 'blob_in_json': False,
        }


def create_provider(manifest: ProviderManifest, **kwargs: Any) -> WeChatProvider:
    return WeChatProvider(manifest, **kwargs)


__all__ = ['WeChatProvider', 'WeChatProviderError', 'create_provider']
