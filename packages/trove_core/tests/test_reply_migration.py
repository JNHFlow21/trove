from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from trove_core.reply.migration import migrate_legacy_reply_runtime
from trove_core.reply.service import ReplyServiceConfig
from trove_core.reply.store import ReplyStore
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation


class ReplyMigrationTests(unittest.TestCase):
    def test_migration_is_private_shadow_only_and_cursor_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = VaultConfig.resolve(str(root / 'vault'), env={})
            cfg.ensure()
            source_account = 'wxid_fixture_1a2b'
            namespace = f'com.tencent.xinWeChat2__{source_account}'
            account_id = (
                'acct-'
                + hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:12]
            )
            snapshot = (
                cfg.root
                / 'sources/wechat-kos-decrypted/current'
                / namespace
            )
            snapshot.mkdir(parents=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()
            store.upsert_accounts([
                Account(account_id, 'Fixture', 'Fixture'),
            ])
            store.upsert_conversations([
                Conversation('conv-fixture', account_id, 'Private', 'private'),
            ])
            work_root = root / 'work'
            (work_root / source_account).mkdir(parents=True)
            legacy_config = root / 'legacy-config.json'
            legacy_state = root / 'legacy-state.json'
            target_a = 'a' * 64
            target_b = 'b' * 64
            legacy_config.write_text(json.dumps({
                'account_id_sha256': hashlib.sha256(
                    source_account.encode('utf-8'),
                ).hexdigest(),
                'container_name': 'com.tencent.xinWeChat2',
                'reply_backend': 'codex',
                'codex_model': 'gpt-5.6-terra',
                'key_store_secret': 'TROVE_WECHAT_KEY_STORE',
                'send_shortcut': 'return',
                'target_scope': 'all_private_except_official',
                'allowed_target_refs': [],
                'cooldown_seconds': 15,
                'daily_send_limit': 300,
            }), encoding='utf-8')
            legacy_state.write_text(json.dumps({
                'cursors': {target_a: 7},
            }), encoding='utf-8')
            os.chmod(legacy_config, 0o600)
            os.chmod(legacy_state, 0o600)

            first = migrate_legacy_reply_runtime(
                cfg.root,
                legacy_config_path=legacy_config,
                legacy_state_path=legacy_state,
                work_root=work_root,
                now=1_000.0,
            )
            loaded = ReplyServiceConfig.load(cfg.root)

            self.assertEqual(first['status'], 'completed')
            self.assertEqual(loaded.mode, 'shadow')
            self.assertTrue(loaded.armed)
            self.assertFalse(first['delivery_enabled'])
            self.assertEqual(loaded.account_id, account_id)
            self.assertEqual(loaded.source_account_id, source_account)
            self.assertEqual(loaded.conversation_namespace, namespace)
            self.assertNotIn(source_account, json.dumps(first))
            self.assertEqual(
                ReplyServiceConfig.path_for_vault(cfg.root).stat().st_mode
                & 0o777,
                0o600,
            )

            legacy_state.write_text(json.dumps({
                'cursors': {target_a: 6, target_b: 9},
            }), encoding='utf-8')
            os.chmod(legacy_state, 0o600)
            second = migrate_legacy_reply_runtime(
                cfg.root,
                legacy_config_path=legacy_config,
                legacy_state_path=legacy_state,
                work_root=work_root,
                now=1_001.0,
            )

            self.assertEqual(second['status'], 'replayed')
            self.assertEqual(
                ReplyStore.for_vault(cfg.root).cursor_map(),
                {target_a: 7, target_b: 9},
            )
            marker = cfg.root / 'jobs/reply/migration.json'
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(source_account, marker.read_text())


if __name__ == '__main__':
    unittest.main()
