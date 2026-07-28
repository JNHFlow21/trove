from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from trove_client import TroveClient
from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.store.repositories import (
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MultimodalRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.fixture_factory import FixtureData
from trove_core.wechat.indexer import index_fixture_data, index_fixture_vault
from trove_daemon.lifecycle import RuntimeIdentity, catalog_identity
from trove_daemon.runtime_owner import RuntimeOwner
from trove_daemon.server import DaemonServer


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
)


def citations(value: Any) -> list[str]:
    if isinstance(value, str) and value.startswith('trove://'):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in citations(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in citations(nested)]
    return []


def create_fixture_vault(
    root: Path,
    *,
    with_media: bool = True,
    fixture_data: FixtureData | None = None,
) -> VaultConfig:
    if fixture_data is None:
        index_fixture_vault(root, reset=True)
    else:
        index_fixture_data(root, fixture_data, reset=True)
    config = VaultConfig.resolve(str(root), env={})
    if not with_media:
        return config
    source = root / 'sources/fixture-card.png'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PNG_1X1)
    source.chmod(0o600)
    store = SQLiteStore(config.paths.sqlite_path)
    message = next(
        row for row in store.all_messages()
        if row['account_id'] == 'acct-work'
        and row['conversation_id'] == 'conv-sales-review'
    )
    repo = MultimodalRepository(store)
    repo.upsert_media_asset(MediaAssetRecord(
        asset_id='asset-agent-flow-image',
        account_id='acct-work', source_type='message', source_id='fixture-card',
        modality='image', media_type='image', citation=message['citation'],
        path_ref='sources/fixture-card.png', cache_state='cached',
        processing_state='ready',
        metadata={
            'file_name': 'fixture-card.png',
            'message_citation': message['citation'],
            'conversation_id': 'conv-sales-review',
            'account_id': 'acct-work',
        },
    ))
    repo.upsert_media_asset_link(MediaAssetLinkRecord(
        'link-agent-flow-image', 'asset-agent-flow-image', 'acct-work',
        'message', message['citation'], 'group_chat', True, 'fixture',
    ))
    store.close()
    return config


class RuntimeHarness:
    def __init__(
        self,
        root: Path,
        *,
        with_media: bool = True,
        max_workers: int = 8,
        max_pending: int = 32,
        fixture_data: FixtureData | None = None,
    ):
        self.config = create_fixture_vault(
            root, with_media=with_media, fixture_data=fixture_data,
        )
        self.owner = RuntimeOwner(self.config, provider_factory=lambda: None)
        self.dispatcher = build_default_dispatcher(
            self.config, runtime_owner=self.owner,
        )
        self.identity = RuntimeIdentity.for_vault(
            root, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )
        self.server = DaemonServer(
            self.identity, self.dispatcher,
            max_workers=max_workers, max_pending=max_pending, idle_timeout=None,
        )
        self.client: TroveClient | None = None

    def __enter__(self) -> 'RuntimeHarness':
        self.server.start()
        self.client = TroveClient(self.identity, role='sdk', autostart=None)
        return self

    def call(self, capability: str, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        assert self.client is not None
        return self.client.call(capability, payload, request_id=request_id)

    def __exit__(self, *_args: object) -> None:
        if self.client is not None:
            self.client.close()
        self.server.stop(timeout=2.0)
        self.owner.close()


__all__ = ['PNG_1X1', 'RuntimeHarness', 'citations', 'create_fixture_vault']
