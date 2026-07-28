from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.media.source_registry import (
    SourceSnapshot,
    bind_account_assets,
    rebind_account_assets,
)


class MediaSourceRegistryTests(unittest.TestCase):
    def test_account_binding_is_linear_and_inherits_exact_moment_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            account_id = 'acct-registry'
            for index in range(150):
                citation = f'trove://fixture/moment/{index}'
                with store.connect() as conn:
                    conn.execute(
                        """INSERT INTO moment_items(
                               moment_id,account_id,author_id,citation,timestamp,text,link_json,
                               media_refs_json,comments_json,status,metadata_json)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            f'moment-{index}', account_id, 'author', citation, 'now', '', '{}',
                            '[]', '[]', 'active', json.dumps({'table': 'Moments', 'rowid': index}),
                        ),
                    )
                    conn.commit()
                repo.upsert_media_asset(MediaAssetRecord(
                    f'asset-{index}', account_id, 'moment', f'moment-{index}',
                    'image', 'image', citation + '#media-0', metadata={'media_idx': 0},
                ))
            for index in range(150, 300):
                repo.upsert_media_asset(MediaAssetRecord(
                    f'asset-{index}', account_id, 'message', f'message-{index}',
                    'voice', 'voice', f'trove://fixture/message/{index}',
                    metadata={'message_local_id': index},
                ))
            snapshot = SourceSnapshot('r' * 64, None, 'm' * 64, None, 'external_unbound')

            result = bind_account_assets(
                store, account_id=account_id, snapshot=snapshot, account_hash='a' * 64,
            )

            self.assertEqual(result, {'seen': 300, 'bound': 300})
            with store.connect() as conn:
                rows = conn.execute(
                    'SELECT source_coordinates_json FROM media_source_bindings'
                ).fetchall()
            self.assertEqual(len(rows), 300)
            moment_coordinates = json.loads(rows[0]['source_coordinates_json'])
            self.assertEqual(moment_coordinates['table'], 'Moments')
            self.assertIn('rowid', moment_coordinates)

    def test_rebind_moves_existing_coordinates_and_preserves_materialized_state(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-existing',
                'acct-registry',
                'message',
                'message-existing',
                'voice',
                'voice',
                'trove://fixture/message/existing',
                metadata={'message_local_id': 7},
            ))
            first = SourceSnapshot(
                'a' * 64,
                None,
                'b' * 64,
                None,
                'external_unbound',
            )
            bind_account_assets(
                store,
                account_id='acct-registry',
                snapshot=first,
                account_hash='c' * 64,
            )
            with store.connect() as conn:
                conn.execute(
                    """UPDATE media_source_bindings
                          SET locator_state='materialized'
                        WHERE asset_id='asset-existing'"""
                )
                conn.commit()
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-new',
                'acct-registry',
                'message',
                'message-new',
                'voice',
                'voice',
                'trove://fixture/message/new',
                metadata={'message_local_id': 8},
            ))
            second = SourceSnapshot(
                'd' * 64,
                None,
                'e' * 64,
                None,
                'external_unbound',
            )

            result = rebind_account_assets(
                store,
                account_id='acct-registry',
                snapshot=second,
                account_hash='f' * 64,
            )

            self.assertEqual(result, {'seen': 2, 'bound': 2})
            with store.connect() as conn:
                rows = list(conn.execute(
                    """SELECT asset_id,snapshot_revision,locator_state,
                              source_coordinates_json
                         FROM media_source_bindings
                        ORDER BY asset_id"""
                ))
            self.assertEqual(
                [row['snapshot_revision'] for row in rows],
                ['d' * 64, 'd' * 64],
            )
            self.assertEqual(
                [row['locator_state'] for row in rows],
                ['materialized', 'bound'],
            )
            self.assertEqual(
                [
                    json.loads(row['source_coordinates_json'])[
                        'message_local_id'
                    ]
                    for row in rows
                ],
                [7, 8],
            )


if __name__ == '__main__':
    unittest.main()
