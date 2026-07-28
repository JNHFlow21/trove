from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sqlite3
from unittest.mock import patch

from trove_core.application.sensitive_commands import execute_files_archive
from trove_core.approvals import ApprovalManager, ApprovalValidationError
from trove_core.store.repositories import MediaAssetLinkRecord, MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.wechat.files import archive_approval_payload, list_conversation_files
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.media.hash_store import sha256_file
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.resources import discover_media_assets


class ConversationFilesTests(unittest.TestCase):
    def _approve_archive(self, cfg: VaultConfig, selection, dest: Path):
        payload = archive_approval_payload(cfg, selection=selection, dest_dir=dest)
        approval = ApprovalManager(cfg.root).request('files_archive', 'local-file-export', payload)
        manager = ApprovalManager(cfg.root)
        manager.decide(approval.approval_id, 'approved', note='fixture approval')
        return manager.require(
            'files_archive',
            'local-file-export',
            payload,
            approval_id=approval.approval_id,
        )

    def _fixture_with_file(self, root: Path) -> tuple[VaultConfig, Path, str]:
        index_fixture_vault(root, reset=True)
        cfg = VaultConfig.resolve(str(root), env={})
        source_dir = root.parent / 'wechat-cache-source'
        source_dir.mkdir(parents=True, exist_ok=True)
        source_file = source_dir / '报价单.pdf'
        source_file.write_bytes(b'fixture proposal pdf bytes')
        store = SQLiteStore(cfg.paths.sqlite_path)
        msg = next(row for row in store.all_messages() if row['conversation_id'] == 'conv-example_edu-private')
        repo = MultimodalRepository(store)
        repo.upsert_media_asset(MediaAssetRecord(
            asset_id='asset-file-fixture-1',
            account_id=msg['account_id'],
            source_type='message_file',
            source_id='file-fixture-1',
            modality='file',
            media_type='document',
            citation=f'{msg["citation"]}#file-1',
            content_hash=sha256_file(source_file),
            path_ref=str(source_file),
            cache_state='cached',
            processing_state='ready',
            metadata={'file_name': '报价单.pdf', 'size_bytes': source_file.stat().st_size},
        ))
        repo.upsert_media_asset_link(MediaAssetLinkRecord(
            link_id='mlink-file-fixture-1',
            asset_id='asset-file-fixture-1',
            account_id=msg['account_id'],
            source_type='message',
            source_citation=msg['citation'],
            scope_type='private_chat',
            accepted=True,
            reason='fixture',
        ))
        return cfg, source_file, msg['citation']

    def test_list_conversation_files_redacts_paths(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, source_file, _citation = self._fixture_with_file(Path(d) / 'vault')
            payload = list_conversation_files(SQLiteStore(cfg.paths.sqlite_path), contact='示例教育', file_name='报价')
            self.assertEqual(payload['raw_paths_included'], False)
            self.assertEqual(payload['count'], 1)
            item = payload['files'][0]
            self.assertEqual(item['asset_id'], 'asset-file-fixture-1')
            self.assertEqual(item['file_name'], '报价单.pdf')
            self.assertEqual(item['cache_state'], 'cached')
            self.assertNotIn(str(source_file), json.dumps(payload, ensure_ascii=False))
            self.assertNotIn('path_ref', json.dumps(payload, ensure_ascii=False))

    def test_archive_requires_approval_then_copies_and_dedupes(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, source_file, citation = self._fixture_with_file(Path(d) / 'vault')
            dest = Path(d) / 'archive-dest'
            selection = {'asset_ids': ['asset-file-fixture-1']}
            with self.assertRaises(ApprovalValidationError):
                execute_files_archive(cfg.root, selection=selection, dest_dir=dest, mode='copy', approval_grant=None)  # type: ignore[arg-type]
            grant = self._approve_archive(cfg, selection, dest)
            result = execute_files_archive(cfg.root, selection=selection, dest_dir=dest, mode='copy', approval_grant=grant)
            self.assertTrue(result['ok'])
            self.assertEqual(result['copied'], 1)
            self.assertEqual(result['skipped_missing'], 0)
            self.assertEqual(result['raw_paths_included'], False)
            manifest_path = dest / 'trove-archive-manifest.json'
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['raw_paths_included'], False)
            self.assertEqual(manifest['entries'][0]['hash'], sha256_file(source_file))
            self.assertEqual(manifest['entries'][0]['content_hash'], sha256_file(source_file))
            self.assertEqual(manifest['entries'][0]['source_citation'], citation)
            self.assertNotIn(str(source_file), json.dumps(manifest, ensure_ascii=False))

            second_grant = self._approve_archive(cfg, selection, dest)
            second = execute_files_archive(cfg.root, selection=selection, dest_dir=dest, mode='copy', approval_grant=second_grant)
            self.assertEqual(second['copied'], 0)
            self.assertEqual(second['skipped_duplicate'], 1)

    def test_archive_hash_and_copy_do_not_acquire_vault_writer(self):
        import trove_core.wechat.files as files_module

        with tempfile.TemporaryDirectory() as d:
            cfg, _source_file, _citation = self._fixture_with_file(Path(d) / 'vault')
            dest = Path(d) / 'archive-dest'
            selection = {'asset_ids': ['asset-file-fixture-1']}
            grant = self._approve_archive(cfg, selection, dest)
            with patch.object(
                VaultOperationCoordinator,
                'write',
                side_effect=AssertionError('pure export acquired Vault writer'),
            ), patch.object(
                files_module, 'sha256_file', wraps=files_module.sha256_file,
            ) as hashed, patch.object(
                files_module.shutil, 'copy2', wraps=files_module.shutil.copy2,
            ) as copied:
                result = execute_files_archive(
                    cfg.root,
                    selection=selection,
                    dest_dir=dest,
                    mode='copy',
                    approval_grant=grant,
                )

            self.assertEqual(result['copied'], 1)
            self.assertGreaterEqual(hashed.call_count, 1)
            self.assertEqual(copied.call_count, 1)

    def test_archive_e2e_with_contact_and_fuzzy_file_name(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, source_file, citation = self._fixture_with_file(Path(d) / 'vault')
            dest = Path(d) / 'archive-dest'
            store = SQLiteStore(cfg.paths.sqlite_path)
            listed = list_conversation_files(store, contact='示例教育', file_name='报价', limit=10)
            self.assertEqual(listed['count'], 1)
            selection = {'contact': '示例教育', 'file_name': '报价', 'limit': 10}
            with self.assertRaises(ApprovalValidationError):
                execute_files_archive(cfg.root, selection=selection, dest_dir=dest, mode='copy', approval_grant=None)  # type: ignore[arg-type]
            grant = self._approve_archive(cfg, selection, dest)
            result = execute_files_archive(cfg.root, selection=selection, dest_dir=dest, mode='copy', approval_grant=grant)
            self.assertEqual(result['copied'], 1)
            self.assertTrue((dest / '报价单.pdf').exists())
            manifest = json.loads((dest / 'trove-archive-manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['entries'][0]['source_citation'], citation)
            self.assertEqual(manifest['entries'][0]['content_hash'], sha256_file(source_file))
            self.assertFalse(manifest['raw_paths_included'])

    def test_archive_rejects_vault_destination(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _source_file, _citation = self._fixture_with_file(Path(d) / 'vault')
            with self.assertRaises(ValueError):
                execute_files_archive(cfg.root, selection={'asset_ids': ['asset-file-fixture-1']}, dest_dir=cfg.root / 'exports', mode='copy', approval_grant=None)  # type: ignore[arg-type]

    def test_archive_rejects_empty_selection_before_approval(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _source_file, _citation = self._fixture_with_file(Path(d) / 'vault')
            with self.assertRaisesRegex(ValueError, 'requires contact, conversation_id, or asset_id'):
                execute_files_archive(cfg.root, selection={}, dest_dir=Path(d) / 'archive', mode='copy', approval_grant=None)  # type: ignore[arg-type]

    def test_discovered_source_path_is_not_persisted_or_exported_without_materialization(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            account = Path(d) / 'decrypted' / 'current' / 'acct'
            (account / 'cache').mkdir(parents=True)
            source_file = account / 'cache' / 'photo.jpg'
            source_file.write_bytes(b'\xff\xd8fixture photo')
            conn = sqlite3.connect(account / 'message_resource.db')
            conn.execute('CREATE TABLE message_resource(local_id INTEGER, local_type TEXT, path TEXT)')
            conn.execute('INSERT INTO message_resource VALUES(?,?,?)', (1, '3', 'cache/photo.jpg'))
            conn.commit(); conn.close()

            refs = discover_media_assets(account, account_id='acct-a')
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].cache_state, 'source_available')
            self.assertIsNone(refs[0].path_ref)
            store = SQLiteStore(cfg.paths.sqlite_path)
            MediaLinker(MultimodalRepository(store)).link_references(refs)
            with store.connect() as conn:
                row = conn.execute('SELECT path_ref,cache_state FROM media_assets WHERE asset_id=?', (refs[0].asset_id,)).fetchone()
            self.assertIsNone(row['path_ref'])
            self.assertEqual(row['cache_state'], 'source_available')

            listed = list_conversation_files(store, limit=10)
            self.assertNotIn(str(source_file), json.dumps(listed, ensure_ascii=False))
            dest = Path(d) / 'archive'
            selection = {'asset_ids': [refs[0].asset_id]}
            grant = self._approve_archive(cfg, selection, dest)
            result = execute_files_archive(cfg.root, selection=selection, dest_dir=dest, mode='copy', approval_grant=grant)
            self.assertEqual(result['copied'], 0)
            self.assertEqual(result['skipped_missing'], 1)
            self.assertFalse((dest / 'photo.jpg').exists())


if __name__ == '__main__':
    unittest.main()
