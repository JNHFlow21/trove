from __future__ import annotations
import json
import shutil
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.store.change_journal import read_sync_commit_generation
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.sync import read_dirty_citations
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.vault.locks import VaultOperationLock
from trove_core.wechat.import_job import run_import_job
from trove_core.wechat.import_receipts import RECEIPT_FILE_NAME
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter, msg_table_for
from trove_core.wechat.process_config import (
    process_config_from_payload,
    read_latest_process_config,
    write_process_config,
)

class ImportJobResumeTests(unittest.TestCase):
    def test_full_import_source_preparation_runs_outside_writer(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            source_root = Path(d) / 'source'
            self._minimal_receipt_account(source_root)
            cfg = VaultConfig.resolve(str(vault), env={})
            phases: list[str] = []

            def probe(phase: str) -> None:
                with VaultOperationCoordinator(cfg).write(owner=f'probe-{phase}'):
                    phases.append(phase)

            from trove_core.wechat import import_job as import_module

            original_iter = import_module.iter_importable_files
            original_hash = import_module.strong_source_fingerprint
            original_stat = import_module.source_stat_token
            original_load = WeChatDecryptedAccountImporter.load
            original_aux = import_module.prepare_auxiliary_sources
            original_media = import_module.discover_media_assets_delta
            original_snapshot = import_module.inspect_source_snapshot

            def probed_iter(*args, **kwargs):
                probe('source-traversal')
                yield from original_iter(*args, **kwargs)

            def probed_hash(*args, **kwargs):
                probe('strong-hash')
                return original_hash(*args, **kwargs)

            def probed_stat(*args, **kwargs):
                probe('stat-tree')
                return original_stat(*args, **kwargs)

            def probed_load(importer, *args, **kwargs):
                probe('message-parse')
                return original_load(importer, *args, **kwargs)

            def probed_aux(*args, **kwargs):
                probe('auxiliary-parse')
                return original_aux(*args, **kwargs)

            def probed_media(*args, **kwargs):
                probe('media-discovery')
                return original_media(*args, **kwargs)

            def probed_snapshot(*args, **kwargs):
                probe('snapshot-hash')
                return original_snapshot(*args, **kwargs)

            with patch.object(import_module, 'iter_importable_files', probed_iter), \
                 patch.object(import_module, 'strong_source_fingerprint', probed_hash), \
                 patch.object(import_module, 'source_stat_token', probed_stat), \
                 patch.object(WeChatDecryptedAccountImporter, 'load', probed_load), \
                 patch.object(import_module, 'prepare_auxiliary_sources', probed_aux), \
                 patch.object(import_module, 'discover_media_assets_delta', probed_media), \
                 patch.object(import_module, 'inspect_source_snapshot', probed_snapshot):
                result = run_import_job(vault, [source_root], reset_index=False)

            self.assertEqual(result.status, 'completed')
            self.assertEqual(phases, [
                'source-traversal',
                'strong-hash',
                'stat-tree',
                'message-parse',
                'snapshot-hash',
                'auxiliary-parse',
                'media-discovery',
                'stat-tree',
            ])

    def test_stale_full_import_cannot_overwrite_newer_publication(self):
        with tempfile.TemporaryDirectory() as d:
            source_root = Path(d) / 'source'
            source_root.mkdir()
            source = source_root / 'messages.jsonl'
            base = {
                'account_id': 'acct-cas',
                'account_label': 'CAS',
                'conversation_id': 'conv-cas',
                'conversation_title': 'CAS',
                'sender_id': 'sender',
                'sender_name': 'Sender',
                'timestamp': '2026-06-21T00:00:00Z',
                'local_id': 1,
            }
            source.write_text(json.dumps({**base, 'content': 'stale prepared value'}) + '\n', encoding='utf-8')
            vault = Path(d) / 'vault'
            stale_ready = threading.Event()
            release_stale = threading.Event()
            reports: dict[str, object] = {}
            failures: list[BaseException] = []

            from trove_core.wechat import import_job as import_module

            original_prepare = import_module._prepare_import_job

            def ordered_prepare(*args, **kwargs):
                prepared = original_prepare(*args, **kwargs)
                if threading.current_thread().name == 'stale-full':
                    stale_ready.set()
                    if not release_stale.wait(10):
                        raise RuntimeError('timed out waiting to release stale full import')
                return prepared

            def run_stale() -> None:
                try:
                    reports['stale'] = run_import_job(vault, [source_root], reset_index=False)
                except BaseException as exc:  # pragma: no cover - asserted below
                    failures.append(exc)

            with patch.object(import_module, '_prepare_import_job', ordered_prepare):
                thread = threading.Thread(target=run_stale, name='stale-full')
                thread.start()
                self.assertTrue(stale_ready.wait(10), 'stale full import did not finish preparation')
                source.write_text(json.dumps({**base, 'content': 'newer published value'}) + '\n', encoding='utf-8')
                try:
                    reports['new'] = run_import_job(vault, [source_root], reset_index=False)
                finally:
                    release_stale.set()
                    thread.join(timeout=10)

            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(reports['new'].status, 'completed')
            self.assertEqual(reports['stale'].status, 'retry_required')
            self.assertEqual(reports['stale'].errors, ['sync_commit_generation_changed'])
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                content = str(conn.execute(
                    "SELECT content FROM messages WHERE account_id='acct-cas' AND local_id=1",
                ).fetchone()['content'])
            self.assertEqual(content, 'newer published value')

    def test_reset_import_preserves_monotonic_publication_generation(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'

            first = run_import_job(vault, [], reset_index=True)
            first_generation = read_sync_commit_generation(
                SQLiteStore(vault / 'index' / 'trove.sqlite', readonly=True),
            )
            second = run_import_job(vault, [], reset_index=True)
            second_generation = read_sync_commit_generation(
                SQLiteStore(vault / 'index' / 'trove.sqlite', readonly=True),
            )

            self.assertEqual(first.status, 'completed')
            self.assertEqual(second.status, 'completed')
            self.assertEqual(first_generation % 2, 0)
            self.assertGreater(second_generation, first_generation)

    def test_import_job_returns_typed_locked_without_mutating_job_state(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})

            with VaultOperationLock(cfg, owner='sync'):
                result = run_import_job(vault, [], reset_index=False)

            self.assertEqual(result.status, 'locked')
            self.assertEqual(result.errors, ['VaultOperationLocked'])
            self.assertFalse((vault / 'jobs' / 'last_import.json').exists())
            self.assertFalse((vault / 'index' / 'trove.sqlite').exists())

    def test_import_job_preserves_existing_incremental_process_config(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            write_process_config(
                vault,
                process_config_from_payload({'config_id': 'pcfg-existing-incremental', 'vector_index': 'incremental'}),
            )

            result = run_import_job(vault, [], reset_index=False)

            latest = read_latest_process_config(vault)
            self.assertEqual(result.status, 'completed')
            self.assertEqual(latest['status'], 'ok')
            self.assertEqual(latest['config']['config_id'], 'pcfg-existing-incremental')
            self.assertEqual(latest['config']['vector_index'], 'incremental')

    def test_import_job_uses_explicit_process_config_over_latest(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            write_process_config(
                vault,
                process_config_from_payload({'config_id': 'pcfg-existing-incremental', 'vector_index': 'incremental'}),
            )
            explicit = process_config_from_payload({'config_id': 'pcfg-explicit-diagnose', 'vector_index': 'diagnose_only'})

            result = run_import_job(vault, [], reset_index=False, process_config=explicit)

            latest = read_latest_process_config(vault)
            self.assertEqual(result.status, 'completed')
            self.assertEqual(latest['config']['config_id'], 'pcfg-explicit-diagnose')
            self.assertEqual(latest['config']['vector_index'], 'diagnose_only')

    def test_import_job_uses_default_process_config_when_latest_missing(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'

            result = run_import_job(vault, [], reset_index=False)

            latest = read_latest_process_config(vault)
            self.assertEqual(result.status, 'completed')
            self.assertEqual(latest['status'], 'ok')
            self.assertEqual(latest['config']['config_id'], 'pcfg-default')
            self.assertEqual(latest['config']['vector_index'], 'diagnose_only')

    def test_import_job_is_idempotent_for_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            msg = {'account_id':'a','account_label':'A','conversation_id':'c','conversation_title':'C','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'真实导入测试','local_id':1}
            (src / 'messages.jsonl').write_text(json.dumps(msg, ensure_ascii=False) + '\n', encoding='utf-8')
            vault = Path(d) / 'vault'
            first = run_import_job(vault, [src], reset_index=True)
            second = run_import_job(vault, [src], reset_index=False)
            self.assertEqual(first.status, 'completed')
            self.assertEqual(second.status, 'completed')
            self.assertEqual(SQLiteStore(vault / 'index' / 'trove.sqlite').counts()['messages'], 1)
            self.assertTrue((vault / 'jobs' / 'last_import.json').exists())

    def test_completed_strong_receipt_skips_only_exact_unchanged_source(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            path = src / 'messages.jsonl'
            first_row = {'account_id':'a','account_label':'A','conversation_id':'c','conversation_title':'C','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'receipt row one','local_id':1}
            path.write_text(json.dumps(first_row) + '\n', encoding='utf-8')
            vault = Path(d) / 'vault'

            first = run_import_job(vault, [src], reset_index=True)
            unchanged = run_import_job(vault, [src], reset_index=False)
            second_row = {**first_row, 'content':'receipt row two', 'local_id':2, 'timestamp':'2026-06-21T00:01:00Z'}
            path.write_text(json.dumps(first_row) + '\n' + json.dumps(second_row) + '\n', encoding='utf-8')
            changed = run_import_job(vault, [src], reset_index=False)

            self.assertEqual(first.status, 'completed')
            self.assertEqual(unchanged.status, 'completed')
            self.assertEqual(unchanged.sources_skipped_unchanged, 1)
            self.assertEqual(unchanged.sources_imported, 0)
            self.assertEqual(changed.sources_skipped_unchanged, 0)
            self.assertEqual(SQLiteStore(vault / 'index' / 'trove.sqlite').counts()['messages'], 2)

    def test_receipt_is_path_independent_for_stable_decrypted_account_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            first_root = vault / 'sources' / 'run-one'
            second_root = vault / 'sources' / 'run-two'
            account = self._minimal_receipt_account(first_root)
            shutil.copytree(account, second_root / account.name)

            first = run_import_job(vault, [first_root], reset_index=True)
            second = run_import_job(vault, [second_root], reset_index=False)

            self.assertEqual(first.status, 'completed')
            self.assertEqual(second.status, 'completed')
            self.assertEqual(second.sources_skipped_unchanged, 1)
            self.assertEqual(second.sources_imported, 0)

    def test_force_config_change_and_limited_import_cannot_false_skip(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            row = {'account_id':'a','account_label':'A','conversation_id':'c','conversation_title':'C','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'receipt controls','local_id':1}
            (src / 'messages.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
            vault = Path(d) / 'vault'
            run_import_job(vault, [src], reset_index=True)

            forced = run_import_job(vault, [src], force_rescan=True)
            changed_config = process_config_from_payload({'config_id':'pcfg-receipt-change','chunk_max_chars':700,'chunk_overlap_chars':70})
            configured = run_import_job(vault, [src], process_config=changed_config)
            limited_vault = Path(d) / 'limited-vault'
            limited = run_import_job(limited_vault, [src], limit_per_sqlite=1)

            self.assertEqual(forced.sources_skipped_unchanged, 0)
            self.assertEqual(configured.sources_skipped_unchanged, 0)
            self.assertEqual(limited.sources_skipped_unchanged, 0)
            self.assertFalse((limited_vault / 'jobs' / RECEIPT_FILE_NAME).exists())

    @staticmethod
    def _minimal_receipt_account(root: Path) -> Path:
        account = root / 'account-stable-fixture'
        account.mkdir(parents=True)
        with sqlite3.connect(account / 'contact.db') as conn:
            conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
            conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
        with sqlite3.connect(account / 'message_0.db') as conn:
            conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
        return account

    def test_import_job_never_reports_completed_when_chunk_projection_fails(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            msg = {'account_id':'a','account_label':'A','conversation_id':'c','conversation_title':'C','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'synthetic chunk failure fixture','local_id':1}
            (src / 'messages.jsonl').write_text(json.dumps(msg) + '\n', encoding='utf-8')
            vault = Path(d) / 'vault'

            with patch.object(SQLiteStore, '_rebuild_message_chunks_for_citations_conn', side_effect=RuntimeError('synthetic failure')):
                result = run_import_job(vault, [src], reset_index=True)

            persisted = json.loads((vault / 'jobs' / 'last_import.json').read_text(encoding='utf-8'))
            trace_rows = [json.loads(line) for line in (vault / 'logs' / 'trace-timeline.redacted.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(result.status, 'failed')
            self.assertEqual(persisted['status'], 'failed')
            self.assertTrue(any('RuntimeError' in error for error in result.errors))
            self.assertEqual(trace_rows[-1]['status'], 'fail')

    def test_import_job_never_reports_completed_when_dirty_journal_fails(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            msg = {'account_id':'a','account_label':'A','conversation_id':'c','conversation_title':'C','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'synthetic dirty failure fixture','local_id':1}
            (src / 'messages.jsonl').write_text(json.dumps(msg) + '\n', encoding='utf-8')
            vault = Path(d) / 'vault'

            with patch.object(SQLiteStore, '_record_dirty_refs_conn', side_effect=RuntimeError('synthetic failure')):
                result = run_import_job(vault, [src], reset_index=True)

            persisted = json.loads((vault / 'jobs' / 'last_import.json').read_text(encoding='utf-8'))
            self.assertEqual(result.status, 'failed')
            self.assertEqual(persisted['status'], 'failed')
            self.assertTrue(any('RuntimeError' in error for error in result.errors))

    def test_import_job_uses_bounded_chunk_rebuild_for_changed_messages(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            src.mkdir()
            rows = [
                {'account_id':'a','account_label':'A','conversation_id':'c1','conversation_title':'C1','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:00:00Z','content':'有界 chunk 重建 一','local_id':1},
                {'account_id':'a','account_label':'A','conversation_id':'c2','conversation_title':'C2','sender_id':'s','sender_name':'S','timestamp':'2026-06-21T00:01:00Z','content':'有界 chunk 重建 二','local_id':2},
            ]
            (src / 'messages.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
            vault = Path(d) / 'vault'

            with patch.object(SQLiteStore, 'rebuild_evidence_chunks', side_effect=AssertionError('full rebuild forbidden')):
                result = run_import_job(vault, [src], reset_index=True)

            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertEqual(result.status, 'completed')
            self.assertEqual(result.changed, 2)
            self.assertEqual(store.counts()['messages'], 2)
            self.assertEqual(store.counts()['chunks'], 2)

    def test_import_job_rebuilds_favorite_chunks_after_favorite_import(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            acct = src / 'com.tencent.xinWeChat__wxid_favorite_fixture'
            acct.mkdir(parents=True)
            with sqlite3.connect(acct / 'contact.db') as conn:
                conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
                conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
                conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_fav_friend', 'Favorite Friend', '', ''))
                conn.commit()
            with sqlite3.connect(acct / 'message_0.db') as conn:
                conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
                conn.commit()
            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('CREATE TABLE favorite_item (title TEXT, text TEXT, time TEXT)')
                conn.execute('INSERT INTO favorite_item(title,text,time) VALUES (?,?,?)', ('收藏方案', 'favuniquetoken 可检索收藏证据', '2026-06-21T00:00:00Z'))
                conn.commit()
            vault = Path(d) / 'vault'

            with patch.object(SQLiteStore, 'rebuild_evidence_chunks', side_effect=AssertionError('full rebuild forbidden')):
                result = run_import_job(vault, [src], reset_index=True)

            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertEqual(result.status, 'completed')
            self.assertEqual(result.favorites_imported, 1)
            rows = store.chunk_search('favuniquetoken', filters={'source_type': 'favorite'}, limit=5)
            self.assertTrue(rows)
            self.assertTrue(all(row['source_type'] == 'favorite' for row in rows))
            with store.connect() as conn:
                direct = conn.execute(
                    """SELECT e.chunk_citation FROM chunk_fts f JOIN evidence_chunks e ON e.rowid=f.rowid
                       WHERE chunk_fts MATCH ? AND e.source_type='favorite' LIMIT 1""",
                    ('favuniquetoken',),
                ).fetchone()
            self.assertIsNotNone(direct)

    def test_import_job_records_dirty_citations_for_messages_and_auxiliary(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'src'
            acct = src / 'com.tencent.xinWeChat__wxid_dirty_fixture'
            acct.mkdir(parents=True)
            with sqlite3.connect(acct / 'contact.db') as conn:
                conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
                conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
                conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_dirty_friend', 'Dirty Fixture', '', ''))
                conn.commit()
            table = msg_table_for('wxid_dirty_friend')
            with sqlite3.connect(acct / 'message_0.db') as conn:
                conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
                conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_dirty_friend', 1))
                conn.execute(f'''CREATE TABLE {table} (
                    local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
                    real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
                    download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
                    message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                    WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
                )''')
                conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (1, 1, 1710000000, 'fixture import dirty message token'))
                conn.commit()
            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('CREATE TABLE favorite_item (title TEXT, text TEXT, time TEXT)')
                conn.execute('INSERT INTO favorite_item(title,text,time) VALUES (?,?,?)', ('Fixture favorite', 'fixture import dirty favorite token', '2026-06-21T00:00:00Z'))
                conn.commit()
            vault = Path(d) / 'vault'

            result = run_import_job(vault, [src], reset_index=True)

            self.assertEqual(result.status, 'completed')
            dirty = read_dirty_citations(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            self.assertTrue(any('/message_0/1' in citation for citation in dirty))
            self.assertTrue(any('/contact/' in citation for citation in dirty))
            self.assertTrue(any('/favorite/' in citation for citation in dirty))
