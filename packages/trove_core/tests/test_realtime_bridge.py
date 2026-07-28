from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.decrypt.manifest import write_account_identity
from trove_core.wechat.decrypt.runner import CopyPlaintextEngine
from trove_core.wechat.importers.wechat_decrypted import msg_table_for
from trove_core.wechat.realtime_bridge import (
    RealtimeBridgeConfig,
    build_realtime_decrypt_plan,
    run_realtime_bridge_once,
)


class RealtimeBridgeTests(unittest.TestCase):
    @staticmethod
    def _write_contact(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
            conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
            conn.execute(
                'INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)',
                ('wxid_bridge_friend', 'Bridge Friend', '', ''),
            )
            conn.commit()

    @staticmethod
    def _write_message(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = msg_table_for('wxid_bridge_friend')
        with sqlite3.connect(path) as conn:
            conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
            conn.execute(
                'INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)',
                (1, 'wxid_bridge_friend', 1),
            )
            conn.execute(f'''CREATE TABLE {table} (
                local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
                real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
                download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
                message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
            )''')
            conn.execute(
                f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)',
                (1, 1, 1760000000, 'fixture realtime bridge token'),
            )
            conn.commit()

    def test_accessible_signed_helper_snapshot_decrypts_and_imports_without_live_root_scan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            external = root / 'signed-helper-private'
            state_path = external / 'auto_reply' / 'state.json'
            snapshot_root = external / 'auto_reply' / 'live_snapshots'
            contact_root = external / 'decrypted' / 'current'
            key_store = external / 'decrypted' / 'key_store.json'
            account_id = 'wxid_bridge_account'
            account_label = 'main-bot'
            canonical_name = f'com.tencent.xinWeChat__{account_id}'
            raw_source = root / 'protected-live-root' / account_id / 'db_storage' / 'message' / 'message_0.db'
            snapshot_dir = snapshot_root / hashlib.sha1(str(raw_source).encode('utf-8')).hexdigest()[:16]
            self._write_message(snapshot_dir / raw_source.name)
            self._write_contact(contact_root / account_label / 'contact.db')
            key_store.parent.mkdir(parents=True, exist_ok=True)
            key_store.write_text(json.dumps({'keys': {}}), encoding='utf-8')
            (vault / 'sources' / 'wechat-integrated-decrypted' / 'current' / canonical_name).mkdir(parents=True)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({
                'fast_sync': {
                    'live_sources': [{
                        'account_id': account_id,
                        'account_label': account_label,
                        'container_name': 'com.tencent.xinWeChat',
                        'component': 'message',
                        'path': str(raw_source),
                        'source_hash': 'fixture-source-hash',
                    }],
                },
            }), encoding='utf-8')
            old_runs = vault / 'sources' / 'wechat-realtime-decrypted' / 'runs'
            for index in range(3):
                (old_runs / f'20260101T00000{index}000000Z').mkdir(parents=True)
            config = RealtimeBridgeConfig(
                trusted_root=external,
                state_path=state_path,
                snapshot_root=snapshot_root,
                contact_root=contact_root,
                key_store_path=key_store,
                retained_runs=2,
            )

            report = run_realtime_bridge_once(vault, config=config, engine=CopyPlaintextEngine())

            self.assertTrue(report['ok'])
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['sources_seen'], 1)
            self.assertEqual(report['sources_ready'], 1)
            self.assertEqual(report['decrypt']['summary']['copied_plaintext'], 2)
            self.assertEqual(report['sync']['messages_imported'], 1)
            self.assertEqual(report['sync']['snapshot_media']['status'], 'skipped')
            self.assertEqual(report['sync']['snapshot_media']['reason'], 'disabled')
            self.assertEqual(report['retention']['retained_runs'], 2)
            self.assertEqual(report['retention']['removed_runs'], 2)
            self.assertEqual(len([path for path in old_runs.iterdir() if path.is_dir()]), 2)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertTrue(store.exact_search('realtime bridge token', limit=3))
            rendered = str(report)
            self.assertNotIn(str(external), rendered)
            self.assertNotIn(str(raw_source), rendered)
            self.assertNotIn(account_id, rendered)
            self.assertNotIn('realtime bridge token', rendered)
            self.assertFalse(report['raw_paths_included'])
            self.assertFalse(report['raw_content_included'])

    def test_failed_sync_still_prunes_decrypted_snapshot_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            external = root / 'signed-helper-private'
            state_path = external / 'auto_reply' / 'state.json'
            snapshot_root = external / 'auto_reply' / 'live_snapshots'
            contact_root = external / 'decrypted' / 'current'
            key_store = external / 'decrypted' / 'key_store.json'
            account_id = 'wxid_bridge_account'
            account_label = 'main-bot'
            raw_source = root / 'protected-live-root' / account_id / 'db_storage' / 'message' / 'message_0.db'
            snapshot_dir = snapshot_root / hashlib.sha1(str(raw_source).encode('utf-8')).hexdigest()[:16]
            self._write_message(snapshot_dir / raw_source.name)
            self._write_contact(contact_root / account_label / 'contact.db')
            key_store.parent.mkdir(parents=True, exist_ok=True)
            key_store.write_text(json.dumps({'keys': {}}), encoding='utf-8')
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({
                'fast_sync': {
                    'live_sources': [{
                        'account_id': account_id,
                        'account_label': account_label,
                        'container_name': 'com.tencent.xinWeChat',
                        'component': 'message',
                        'path': str(raw_source),
                        'source_hash': 'fixture-source-hash',
                    }],
                },
            }), encoding='utf-8')
            old_runs = vault / 'sources' / 'wechat-realtime-decrypted' / 'runs'
            for index in range(3):
                (old_runs / f'20260101T00000{index}000000Z').mkdir(parents=True)
            config = RealtimeBridgeConfig(
                trusted_root=external,
                state_path=state_path,
                snapshot_root=snapshot_root,
                contact_root=contact_root,
                key_store_path=key_store,
                retained_runs=2,
            )

            report = run_realtime_bridge_once(
                vault,
                config=config,
                engine=CopyPlaintextEngine(),
                sync_runner=lambda *_args, **_kwargs: {'ok': False, 'status': 'failed'},
            )

            self.assertFalse(report['ok'])
            self.assertEqual(report['status'], 'sync_failed')
            self.assertEqual(report['retention']['retained_runs'], 2)
            self.assertEqual(report['retention']['removed_runs'], 2)
            self.assertEqual(report['retention']['errors'], 0)
            self.assertEqual(len([path for path in old_runs.iterdir() if path.is_dir()]), 2)
            self.assertFalse(report['raw_paths_included'])
            self.assertFalse(report['raw_content_included'])

    def test_integrated_private_identity_is_authoritative_and_garbage_source_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            external = root / 'signed-helper-private'
            state_path = external / 'state.json'
            snapshot_root = external / 'snapshots'
            contact_root = external / 'contacts'
            key_store = external / 'key_store.json'
            valid_id = 'wxid_fixturec'
            garbage_id = 'wxid_fixturea'
            anonymous_name = 'account-0123456789abcdef'
            integrated = vault / 'sources' / 'wechat-integrated-decrypted' / 'current' / anonymous_name
            write_account_identity(integrated, account_ref_hash='0123456789abcdef', own_wxid=valid_id)
            key_store.parent.mkdir(parents=True, exist_ok=True)
            key_store.write_text(json.dumps({'keys': {}}), encoding='utf-8')
            sources = []
            for account_id, label in ((valid_id, 'valid'), (garbage_id, 'garbage')):
                state_account_id = account_id + '_1a2b'
                raw = root / 'protected' / state_account_id / 'message_0.db'
                snapshot = snapshot_root / hashlib.sha1(str(raw).encode('utf-8')).hexdigest()[:16] / raw.name
                self._write_message(snapshot)
                self._write_contact(contact_root / label / 'contact.db')
                sources.append({
                    'account_id': state_account_id,
                    'account_label': label,
                    'container_name': 'com.tencent.xinWeChat',
                    'component': 'message',
                    'path': str(raw),
                })
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({'fast_sync': {'live_sources': sources}}), encoding='utf-8')
            config = RealtimeBridgeConfig(
                trusted_root=external,
                state_path=state_path,
                snapshot_root=snapshot_root,
                contact_root=contact_root,
                key_store_path=key_store,
            )

            plan, counts = build_realtime_decrypt_plan(vault, config=config)

            self.assertTrue(plan.ok)
            self.assertEqual(counts, {
                'sources_seen': 2,
                'sources_ready': 1,
                'sources_skipped': 1,
                'accounts_ready': 1,
            })
            self.assertEqual({item.output_relative.parts[0] for item in plan.files}, {anonymous_name})
            rendered = json.dumps(plan.to_redacted_dict())
            self.assertNotIn(valid_id, rendered)
            self.assertNotIn(garbage_id, rendered)


if __name__ == '__main__':
    unittest.main()
