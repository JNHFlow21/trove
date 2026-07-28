from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import json
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.knowledge.profile_automation import process_profile_refresh_queue
from trove_core.knowledge.profile_snapshots import list_profile_snapshots
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.sync import SyncOptions, clear_dirty_citations, maybe_index_vectors, read_dirty_citations, read_sync_config, read_waterlines, record_dirty_citations, run_sync
from trove_core.store.change_journal import clear_dirty_citation_batch, read_dirty_citation_batch
from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.runtime import index_vectors, vector_status_payload, zvec_collection_path
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.vector.zvec_store import ZVecStore
from trove_core.vector.ledger import VectorIndexLedger
from trove_core.wechat.importers.contacts import ContactIdentityImporter
from trove_core.wechat.importers.favorites import FavoritesImporter
from trove_core.wechat.importers.moments import MomentsImporter
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter
from trove_core.wechat.importers.wechat_decrypted import msg_table_for
from trove_core.wechat.process_config import process_config_from_payload, write_process_config
from trove_core.wechat.auxiliary_import import auxiliary_source_fingerprints
from trove_core.wechat.media.source_registry import inspect_source_snapshot


class SyncFixtureTests(unittest.TestCase):
    def test_sync_imports_only_explicitly_selected_account_ids(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, first_account, _table = self.make_snapshot(vault)
            second_account = current / 'com.tencent.xinWeChat2__wxid_other_fixture'
            shutil.copytree(first_account, second_account)
            first_id = WeChatDecryptedAccountImporter(first_account).account_id
            second_id = WeChatDecryptedAccountImporter(second_account).account_id
            self.assertNotEqual(first_id, second_id)

            report = run_sync(
                vault,
                options=SyncOptions(
                    account_ids=(second_id,),
                    snapshot_dir=current,
                    snapshot_media_enabled=False,
                    media_discovery_mode='message_delta',
                ),
            )

            self.assertTrue(report['ok'])
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as connection:
                account_ids = {
                    str(row['account_id'])
                    for row in connection.execute('SELECT account_id FROM accounts')
                }
            self.assertEqual(account_ids, {second_id})

    def test_sync_config_clamps_inline_dirty_work_and_preserves_explicit_zero(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.paths.jobs_dir.mkdir(parents=True)
            config_path = cfg.paths.jobs_dir / 'sync_config.redacted.json'
            config_path.write_text(json.dumps({'sync': {'vector_inline_dirty_limit': 100_000}}), encoding='utf-8')
            self.assertEqual(read_sync_config(cfg).vector_inline_dirty_limit, 512)
            config_path.write_text(json.dumps({'sync': {'vector_inline_dirty_limit': 0}}), encoding='utf-8')
            self.assertEqual(read_sync_config(cfg).vector_inline_dirty_limit, 0)

    def test_dirty_batch_clear_does_not_drop_row_redirtied_during_indexing(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ref = {'citation': 'trove://wechat/acct/conv/message_0/1', 'account_id': 'acct', 'conversation_id': 'conv'}
            record_dirty_citations(store, [ref])
            batch = read_dirty_citation_batch(store, limit=1)
            with patch('trove_core.store.change_journal._now', return_value='2999-01-01T00:00:00Z'):
                record_dirty_citations(store, [ref])

            self.assertEqual(clear_dirty_citation_batch(store, batch), 0)
            self.assertEqual(read_dirty_citations(store), [ref['citation']])

    def make_snapshot(self, vault: Path) -> tuple[Path, Path, str]:
        current = vault / 'sources' / 'wechat-kos-decrypted' / 'current'
        acct = current / 'com.tencent.xinWeChat__wxid_sync_fixture'
        acct.mkdir(parents=True)
        with sqlite3.connect(acct / 'contact.db') as conn:
            conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
            conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
            conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_sync_friend', 'Fixture Friend', '', ''))
            conn.commit()
        table = msg_table_for('wxid_sync_friend')
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
            conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_sync_friend', 1))
            conn.execute(f'''CREATE TABLE {table} (
                local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
                real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
                download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
                message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
            )''')
            conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (1, 1, 1710000000, 'fixture sync baseline token'))
            conn.commit()
        return current, acct, table

    def append_message(self, acct: Path, table: str, local_id: int, create_time: int, content: str) -> None:
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (local_id, 1, create_time, content))
            conn.commit()

    def add_auxiliary_sources(self, acct: Path) -> None:
        with sqlite3.connect(acct / 'sns.db') as conn:
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-m1', 'wxid_sync_friend', '<TimelineObject><id>m1</id><username>wxid_sync_friend</username><createTime>1760000000</createTime><contentDesc>fixture moment aux-token</contentDesc></TimelineObject>', b''))
            conn.commit()
        with sqlite3.connect(acct / 'favorite.db') as conn:
            conn.execute('CREATE TABLE favorite_item(fav_id TEXT, update_time TEXT, title TEXT, content TEXT)')
            conn.execute('INSERT INTO favorite_item VALUES(?,?,?,?)', ('f1', '2026-01-02', 'Fixture favorite', 'fixture favorite aux-token'))
            conn.commit()

    def test_sync_imports_append_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)

            first = run_sync(vault)
            self.assertTrue(first['ok'])
            self.assertEqual(first['messages_imported'], 1)
            self.assertEqual(first['conversations_changed'], 1)
            self.assertFalse(first['raw_content_included'])

            second = run_sync(vault)
            self.assertTrue(second['ok'])
            self.assertEqual(second['messages_imported'], 0)
            self.assertEqual(second['waterlines_updated'], 0)

            self.append_message(acct, table, 2, 1710000060, 'fixture sync incremental omega-token')
            third = run_sync(vault)
            self.assertTrue(third['ok'])
            self.assertEqual(third['messages_imported'], 1)
            self.assertEqual(third['chunks']['conversations'], 1)

            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertTrue(store.exact_search('omega-token', limit=3))
            waterlines = read_waterlines(store)
            self.assertEqual(max(state['max_local_id'] for state in waterlines.values()), 2)
            self.assertNotIn(str(vault), str(third))
            self.assertNotIn('omega-token', str(third))

    def test_sync_delta_queues_and_publishes_opted_in_profile_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            _current, acct, table = self.make_snapshot(vault)
            first_sync = run_sync(vault)
            self.assertTrue(first_sync['ok'])
            agent_tools.profile_automation_enable(
                vault, 'Fixture Friend', debounce_seconds=0,
            )
            initial = process_profile_refresh_queue(vault, limit=5)
            self.assertEqual(initial['created_snapshots'], 1)

            self.append_message(
                acct, table, 2, 1710000060,
                'fixture profile automation incremental token',
            )
            report = run_sync(vault)

            self.assertTrue(report['ok'])
            self.assertEqual(report['profiles']['refresh_queue']['queued'], 1)
            self.assertEqual(report['profiles']['worker']['created_snapshots'], 1)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            history = list_profile_snapshots(store, 'Fixture Friend')
            self.assertEqual([item['version'] for item in history['items']], [2, 1])
            self.assertNotIn('incremental token', json.dumps(report, ensure_ascii=False))

    def test_full_sync_deletion_refreshes_profile_even_without_imported_messages(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)
            first = run_sync(
                vault, options=SyncOptions(full=True, snapshot_dir=current),
            )
            self.assertTrue(first['ok'])
            agent_tools.profile_automation_enable(
                vault, 'Fixture Friend', debounce_seconds=0,
            )
            process_profile_refresh_queue(vault, limit=5)
            with sqlite3.connect(acct / 'message_0.db') as conn:
                conn.execute(f'DELETE FROM {table} WHERE local_id=1')
                conn.commit()

            report = run_sync(
                vault, options=SyncOptions(full=True, snapshot_dir=current),
            )

            self.assertTrue(report['ok'])
            self.assertEqual(report['messages_imported'], 0)
            self.assertEqual(report['profiles']['refresh_queue']['queued'], 1)
            self.assertEqual(report['profiles']['worker']['created_snapshots'], 1)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertEqual(
                [item['version'] for item in list_profile_snapshots(
                    store, 'Fixture Friend',
                )['items']],
                [2, 1],
            )

    def test_profile_worker_exception_preserves_committed_sync_report(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            self.make_snapshot(vault)

            with patch(
                'trove_core.sync.process_profile_refresh_queue',
                side_effect=RuntimeError('private fixture detail'),
            ):
                report = run_sync(vault)

            self.assertEqual(report['status'], 'partial')
            self.assertTrue(report['ok'])
            self.assertEqual(report['messages_imported'], 1)
            self.assertIsNotNone(report['chunks'])
            self.assertEqual(report['profiles']['worker']['error_code'], 'RuntimeError')
            self.assertNotIn('private fixture detail', str(report))

    def test_sync_since_filters_snapshot_rows(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)
            self.append_message(acct, table, 2, 1710003600, 'fixture sync since kept-token')
            report = run_sync(vault, options=SyncOptions(full=True, snapshot_dir=current, since=datetime.fromtimestamp(1710003000, tz=timezone.utc)))
            self.assertTrue(report['ok'])
            self.assertEqual(report['messages_imported'], 1)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertFalse(store.exact_search('baseline token', limit=3))
            self.assertTrue(store.exact_search('kept-token', limit=3))

    def test_sync_advances_waterline_for_empty_message_rows(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)
            first = run_sync(vault)
            self.assertTrue(first['ok'])
            self.assertEqual(first['messages_imported'], 1)

            self.append_message(acct, table, 2, 1710000060, '')
            second = run_sync(vault)
            self.assertTrue(second['ok'])
            self.assertEqual(second['messages_imported'], 0)
            self.assertEqual(second['waterlines_updated'], 1)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertEqual(max(state['max_local_id'] for state in read_waterlines(store).values()), 2)

            third = run_sync(vault)
            self.assertTrue(third['ok'])
            self.assertEqual(third['messages_imported'], 0)
            self.assertEqual(third['waterlines_updated'], 0)

    def test_sync_reports_dirty_count_and_indexes_only_dirty_citations(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            self.make_snapshot(vault)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))

            with patch('trove_core.sync.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.sync.vector_status_payload', return_value={'zvec': {'collection_exists': True, 'complete': True, 'metadata_complete': True, 'rebuild_required': False, 'provider_mismatch': False, 'stale': False, 'incomplete': False}}), \
                 patch('trove_core.sync.index_vectors', return_value={'backend': 'zvec', 'indexed': 1}) as index_vectors:
                report = run_sync(vault)

            self.assertTrue(report['ok'])
            self.assertEqual(report['dirty_count'], 2)
            self.assertEqual(report['vector']['status'], 'indexed')
            self.assertEqual(report['vector']['dirty_count'], 2)
            self.assertEqual(report['vector']['dirty_cleared'], 2)
            citations = index_vectors.call_args.kwargs['citations']
            self.assertEqual(len(citations), 2)
            self.assertTrue(all(citation.startswith('trove://wechat/') for citation in citations))
            self.assertTrue(any('/contact/' in citation for citation in citations))
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertEqual(read_dirty_citations(store), [])

    def test_sync_imports_auxiliary_sources_and_skips_unchanged_fingerprints(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            self.add_auxiliary_sources(acct)

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertTrue(first['ok'])
            self.assertEqual(first['auxiliary']['contacts_imported'], 1)
            self.assertEqual(first['auxiliary']['moments_imported'], 1)
            self.assertEqual(first['auxiliary']['favorites_imported'], 1)
            self.assertEqual(first['auxiliary']['dirty_citations'], 3)
            self.assertEqual(first['auxiliary']['changed_families']['contact'], 1)
            self.assertEqual(first['auxiliary']['changed_families']['moment'], 1)
            self.assertEqual(first['auxiliary']['changed_families']['favorite'], 1)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            self.assertTrue(store.chunk_search('aux-token', filters={'source_type': 'moment'}, limit=3))
            self.assertTrue(store.chunk_search('aux-token', filters={'source_type': 'favorite'}, limit=3))

            second = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertTrue(second['ok'])
            self.assertEqual(second['auxiliary']['sources_imported'], 0)
            self.assertEqual(second['auxiliary']['dirty_citations'], 0)

    def test_sync_snapshot_media_cache_copies_allowlisted_sns_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            live_account = root / 'xwechat_files' / 'wxid_sync_fixture'
            allowed = [
                live_account / 'cache' / '2026-01' / 'sns' / 'img' / 'aa' / 'image.dat',
                live_account / 'business' / 'sns' / 'bkg' / 'background.dat',
                live_account / 'business' / 'sns' / 'publish' / 'publish.dat',
            ]
            for idx, path in enumerate(allowed, start=1):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes([idx]) * idx)
            disallowed = live_account / 'cache' / 'not-a-month' / 'sns' / 'img' / 'skip.dat'
            disallowed.parent.mkdir(parents=True, exist_ok=True)
            disallowed.write_bytes(b'skip')

            with patch.dict('os.environ', {'TROVE_SYNC_SNAPSHOT_MEDIA_ROOT': str(root / 'xwechat_files')}):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            media = report['snapshot']['media_cache']
            self.assertEqual(media['status'], 'copied')
            self.assertEqual(media['copied_files'], 3)
            self.assertEqual(media['bytes_total'], 6)
            self.assertFalse(media['raw_paths_included'])
            self.assertEqual(media['d0_mapping_conclusion']['readable_sns_db'], 'no_mapping')
            self.assertEqual(media['d0_mapping_conclusion']['cache_files'], 'no_embedded_mapping')
            self.assertIn('media_0.db', media['d0_mapping_conclusion']['encrypted_candidates_not_inspected'])
            self.assertTrue((acct / 'cache' / '2026-01' / 'sns' / 'img' / 'aa' / 'image.dat').exists())
            self.assertTrue((acct / 'business' / 'sns' / 'bkg' / 'background.dat').exists())
            self.assertTrue((acct / 'business' / 'sns' / 'publish' / 'publish.dat').exists())
            self.assertFalse((acct / 'cache' / 'not-a-month' / 'sns' / 'img' / 'skip.dat').exists())

    def test_sync_snapshot_media_cache_skips_when_size_exceeds_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            live_file = root / 'xwechat_files' / 'wxid_sync_fixture' / 'cache' / '2026-01' / 'sns' / 'img' / 'aa' / 'image.dat'
            live_file.parent.mkdir(parents=True, exist_ok=True)
            live_file.write_bytes(b'1234')

            with patch.dict('os.environ', {
                'TROVE_SYNC_SNAPSHOT_MEDIA_ROOT': str(root / 'xwechat_files'),
                'TROVE_SYNC_SNAPSHOT_MEDIA_MAX_BYTES': '3',
            }):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            media = report['snapshot']['media_cache']
            self.assertEqual(media['status'], 'skipped')
            self.assertIn('encrypted_candidates_not_inspected', media['d0_mapping_conclusion'])
            self.assertEqual(media['reason'], 'copy_size_exceeds_limit')
            self.assertEqual(media['bytes_total'], 4)
            self.assertFalse((acct / 'cache' / '2026-01' / 'sns' / 'img' / 'aa' / 'image.dat').exists())

    def test_sync_can_disable_protected_live_media_scan_for_background_import(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, _acct, _table = self.make_snapshot(vault)

            with patch(
                'trove_core.sync.refresh_snapshot_media_cache',
                side_effect=PermissionError('protected live WeChat root'),
            ) as media_refresh:
                report = run_sync(
                    vault,
                    options=SyncOptions(snapshot_dir=current, snapshot_media_enabled=False),
                )

            self.assertTrue(report['ok'])
            self.assertEqual(report['messages_imported'], 1)
            self.assertEqual(report['snapshot']['media_cache']['status'], 'skipped')
            self.assertEqual(report['snapshot']['media_cache']['reason'], 'disabled')
            media_refresh.assert_not_called()
            self.assertNotIn('protected live WeChat root', str(report))

    def test_sync_treats_protected_optional_media_root_as_nonfatal(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, _acct, _table = self.make_snapshot(vault)

            with patch(
                'trove_core.sync.refresh_snapshot_media_cache',
                side_effect=PermissionError('protected live WeChat root'),
            ):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertTrue(report['ok'])
            self.assertEqual(report['messages_imported'], 1)
            self.assertEqual(report['snapshot']['media_cache']['status'], 'skipped')
            self.assertEqual(report['snapshot']['media_cache']['reason'], 'permission_denied')
            self.assertFalse(report['snapshot']['media_cache']['raw_paths_included'])
            self.assertNotIn('protected live WeChat root', str(report))

    def test_sync_message_delta_media_mode_registers_only_imported_media_without_full_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)
            with sqlite3.connect(acct / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
                    (2, 34, 1, 1710000060, 'voice/path/fixture.amr'),
                )
                conn.commit()

            with patch(
                'trove_core.sync.discover_media_assets_delta',
                side_effect=AssertionError('full media discovery must not run'),
            ):
                report = run_sync(
                    vault,
                    options=SyncOptions(
                        snapshot_dir=current,
                        snapshot_media_enabled=False,
                        media_discovery_mode='message_delta',
                    ),
                )

            self.assertTrue(report['ok'])
            self.assertEqual(report['messages_imported'], 2)
            self.assertEqual(report['media']['discovery_mode'], 'message_delta')
            self.assertEqual(report['media']['assets_seen'], 1)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM media_assets WHERE modality='voice'").fetchone()[0], 1)
                binding = conn.execute(
                    """SELECT b.snapshot_revision,s.root_ref,s.state
                         FROM media_source_bindings b
                         JOIN source_snapshots s
                           ON s.snapshot_revision=b.snapshot_revision
                         JOIN media_assets ma ON ma.asset_id=b.asset_id
                        WHERE ma.modality='voice'"""
                ).fetchone()
            self.assertIsNotNone(binding)
            self.assertEqual(binding['state'], 'available')
            self.assertEqual(
                (vault / binding['root_ref']).resolve(),
                current.resolve(),
            )

    def test_sync_rebinds_existing_message_media_when_snapshot_rotates(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)
            with sqlite3.connect(acct / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
                    (2, 34, 1, 1710000060, 'voice/path/fixture.amr'),
                )
                conn.commit()

            first = run_sync(
                vault,
                options=SyncOptions(
                    snapshot_dir=current,
                    snapshot_media_enabled=False,
                    media_discovery_mode='message_delta',
                ),
            )
            self.assertTrue(first['ok'])
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                first_binding = conn.execute(
                    """SELECT b.snapshot_revision
                         FROM media_source_bindings b
                         JOIN media_assets ma ON ma.asset_id=b.asset_id
                        WHERE ma.modality='voice'"""
                ).fetchone()

            replacement = current.parent / 'replacement'
            shutil.copytree(current, replacement)
            second = run_sync(
                vault,
                options=SyncOptions(
                    snapshot_dir=replacement,
                    snapshot_media_enabled=False,
                    media_discovery_mode='message_delta',
                ),
            )

            self.assertTrue(second['ok'])
            with store.connect() as conn:
                second_binding = conn.execute(
                    """SELECT b.snapshot_revision,s.root_ref
                         FROM media_source_bindings b
                         JOIN source_snapshots s
                           ON s.snapshot_revision=b.snapshot_revision
                         JOIN media_assets ma ON ma.asset_id=b.asset_id
                        WHERE ma.modality='voice'"""
                ).fetchone()
            self.assertNotEqual(
                first_binding['snapshot_revision'],
                second_binding['snapshot_revision'],
            )
            self.assertEqual(
                (vault / second_binding['root_ref']).resolve(),
                replacement.resolve(),
            )

    def test_sync_source_scans_run_before_final_writer_lock(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            self.add_auxiliary_sources(acct)
            cfg = VaultConfig.resolve(str(vault), env={})
            phases: list[str] = []

            def probe(phase: str) -> None:
                # A second writer can be acquired only when sync has released
                # its coordinator.  These are the potentially slow phases.
                with VaultOperationCoordinator(cfg).write(owner=f'probe-{phase}'):
                    phases.append(phase)

            from trove_core import sync as sync_module

            original_iter = sync_module.iter_importable_files
            original_load = WeChatDecryptedAccountImporter.load
            original_discover = sync_module.discover_media_assets_delta
            original_prepare_aux = sync_module.prepare_auxiliary_sources
            original_inspect = sync_module.inspect_source_snapshot

            def probed_iter(*args, **kwargs):
                probe('source-discovery')
                yield from original_iter(*args, **kwargs)

            def probed_load(importer, *args, **kwargs):
                probe('message-load')
                return original_load(importer, *args, **kwargs)

            def probed_discover(*args, **kwargs):
                probe('media-discovery')
                scan_store = kwargs['store']
                self.assertTrue(scan_store.readonly)
                with patch.object(
                    scan_store,
                    'initialize',
                    side_effect=AssertionError('read-only sync scan must not initialize a writable store'),
                ):
                    return original_discover(*args, **kwargs)

            def probed_prepare_aux(*args, **kwargs):
                probe('auxiliary-load')
                return original_prepare_aux(*args, **kwargs)

            def probed_inspect(*args, **kwargs):
                probe('snapshot-manifest')
                return original_inspect(*args, **kwargs)

            with patch('trove_core.sync.iter_importable_files', probed_iter), \
                 patch.object(WeChatDecryptedAccountImporter, 'load', probed_load), \
                 patch('trove_core.sync.discover_media_assets_delta', probed_discover), \
                 patch('trove_core.sync.prepare_auxiliary_sources', probed_prepare_aux), \
                 patch('trove_core.sync.inspect_source_snapshot', probed_inspect):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current, snapshot_media_enabled=False))

            self.assertTrue(report['ok'])
            self.assertEqual(
                phases,
                ['source-discovery', 'message-load', 'media-discovery', 'auxiliary-load', 'snapshot-manifest'],
            )

    def test_concurrent_stale_sync_cannot_prune_fingerprint_or_bind_over_newer_aux(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            old_xml = (
                '<TimelineObject><id>native-old</id><username>wxid-author</username>'
                '<createTime>1760000000</createTime><contentDesc>old prepared row</contentDesc>'
                '<ContentObject><mediaList><media><id>old-image</id><type>2</type>'
                '<url>https://cdn.invalid/old.jpg</url></media></mediaList></ContentObject>'
                '</TimelineObject>'
            )
            with sqlite3.connect(acct / 'sns.db') as conn:
                conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
                conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-old', 'wxid-author', old_xml, b''))
                conn.commit()

            cfg = VaultConfig.resolve(str(vault), env={})
            old_snapshot = inspect_source_snapshot(cfg, current)
            stale_ready = threading.Event()
            release_stale = threading.Event()
            reports: dict[str, dict] = {}
            failures: list[BaseException] = []

            from trove_core import sync as sync_module

            original_coordinated = sync_module.coordinated_vault_mutation
            mutation_calls: dict[str, int] = {}

            @contextmanager
            def ordered_coordinated(*args, **kwargs):
                name = threading.current_thread().name
                mutation_calls[name] = mutation_calls.get(name, 0) + 1
                if name == 'stale-sync' and mutation_calls[name] == 2:
                    # The old sync has completed every source scan, including
                    # snapshot hashing, but has not entered its final writer.
                    stale_ready.set()
                    if not release_stale.wait(10):
                        raise RuntimeError('timed out waiting to release stale sync')
                with original_coordinated(*args, **kwargs) as session:
                    yield session

            def run_stale() -> None:
                try:
                    reports['stale'] = run_sync(
                        vault,
                        options=SyncOptions(snapshot_dir=current, snapshot_media_enabled=False),
                    )
                except BaseException as exc:  # pragma: no cover - asserted below
                    failures.append(exc)

            with patch('trove_core.sync.coordinated_vault_mutation', ordered_coordinated):
                thread = threading.Thread(target=run_stale, name='stale-sync')
                thread.start()
                self.assertTrue(stale_ready.wait(10), 'stale sync did not finish preparation')

                new_xml = (
                    '<TimelineObject><id>native-new</id><username>wxid-author</username>'
                    '<createTime>1760000060</createTime><contentDesc>new committed row</contentDesc>'
                    '<ContentObject><mediaList><media><id>new-image</id><type>2</type>'
                    '<url>https://cdn.invalid/new.jpg</url></media></mediaList></ContentObject>'
                    '</TimelineObject>'
                )
                with sqlite3.connect(acct / 'sns.db') as conn:
                    conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-new', 'wxid-author', new_xml, b''))
                    conn.commit()
                new_snapshot = inspect_source_snapshot(cfg, current)
                self.assertNotEqual(new_snapshot.snapshot_revision, old_snapshot.snapshot_revision)
                new_fingerprints = auxiliary_source_fingerprints(
                    acct,
                    account_id=WeChatDecryptedAccountImporter(acct).account_id,
                )
                try:
                    reports['new'] = run_sync(
                        vault,
                        options=SyncOptions(snapshot_dir=current, snapshot_media_enabled=False),
                    )
                finally:
                    release_stale.set()
                    thread.join(timeout=10)

            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(reports['new']['ok'], reports['new'])
            self.assertEqual(reports['stale']['status'], 'retry_required')
            self.assertEqual(reports['stale']['errors'], ['sync_commit_generation_changed'])

            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                moment_texts = {
                    str(row['text'])
                    for row in conn.execute('SELECT text FROM moment_items')
                }
                persisted_fingerprints = {
                    str(row['source_key']): str(row['fingerprint'])
                    for row in conn.execute('SELECT source_key,fingerprint FROM sync_aux_state')
                }
                snapshot_revisions = {
                    str(row['snapshot_revision'])
                    for row in conn.execute('SELECT snapshot_revision FROM source_snapshots')
                }
                binding_revisions = {
                    str(row['snapshot_revision'])
                    for row in conn.execute('SELECT snapshot_revision FROM media_source_bindings')
                }
                binding_count = int(conn.execute('SELECT COUNT(*) FROM media_source_bindings').fetchone()[0])

            # The old prepared family contained only the first row.  Committing
            # it after the new sync would prune this second row and its asset.
            self.assertEqual(moment_texts, {'old prepared row', 'new committed row'})
            self.assertEqual(persisted_fingerprints, new_fingerprints)
            self.assertEqual(snapshot_revisions, {new_snapshot.snapshot_revision})
            self.assertNotIn(old_snapshot.snapshot_revision, snapshot_revisions)
            self.assertEqual(binding_count, 2)
            self.assertEqual(binding_revisions, {new_snapshot.snapshot_revision})

    def test_sync_reimports_only_changed_auxiliary_family(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            self.add_auxiliary_sources(acct)

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertTrue(first['ok'])
            self.assertEqual(first['auxiliary']['sources_imported'], 3)

            with sqlite3.connect(acct / 'favorite.db') as conn:
                conn.execute('INSERT INTO favorite_item VALUES(?,?,?,?)', ('f2', '2026-01-03', 'Fixture favorite 2', 'fixture favorite family-only token'))
                conn.commit()

            favorite_calls = {'count': 0}
            original_favorite = FavoritesImporter.load

            def counted_favorite(self, *args, **kwargs):
                favorite_calls['count'] += 1
                return original_favorite(self, *args, **kwargs)

            with patch.object(ContactIdentityImporter, 'load', side_effect=AssertionError('contact importer should not run')), \
                 patch.object(MomentsImporter, 'load', side_effect=AssertionError('moment importer should not run')), \
                 patch.object(FavoritesImporter, 'load', counted_favorite):
                second = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertTrue(second['ok'])
            self.assertEqual(favorite_calls['count'], 1)
            self.assertEqual(second['auxiliary']['sources_imported'], 1)
            self.assertEqual(second['auxiliary']['changed_families']['favorite'], 1)
            self.assertNotIn('contact', second['auxiliary']['changed_families'])
            self.assertNotIn('moment', second['auxiliary']['changed_families'])

    def test_sync_indexes_auxiliary_dirty_citations_incrementally(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self.make_snapshot(vault)
            self.add_auxiliary_sources(acct)
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))
            cfg = VaultConfig.resolve(str(vault), env={})

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertTrue(first['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)
            self.assertTrue(vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']['complete'])
            clear_dirty_citations(store, read_dirty_citations(store))

            self.append_message(acct, table, 2, 1710000060, 'fixture sync vector incremental message token')
            with sqlite3.connect(acct / 'sns.db') as conn:
                conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-m2', 'wxid_sync_friend', '<TimelineObject><id>m2</id><username>wxid_sync_friend</username><createTime>1760000060</createTime><contentDesc>fixture sync vector incremental moment token</contentDesc></TimelineObject>', b''))
                conn.commit()

            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertTrue(report['ok'])
            self.assertEqual(report['vector']['status'], 'indexed')
            self.assertGreater(report['vector']['dirty_cleared'], 0)
            zvec_status = vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']
            self.assertTrue(zvec_status['complete'])
            self.assertFalse(zvec_status['rebuild_required'])
            metadata = json.loads(Path(str(zvec_collection_path(cfg)) + '.trove-meta.json').read_text(encoding='utf-8'))
            with store.connect() as conn:
                new_chunks = [
                    row['chunk_citation']
                    for row in conn.execute(
                        """SELECT chunk_citation FROM evidence_chunks
                           WHERE parent_citation LIKE ? OR content LIKE ?""",
                        ('%/message_0/2', '%vector incremental moment token%'),
                    )
                ]
            self.assertGreaterEqual(len(new_chunks), 2)
            self.assertNotIn('content_hashes', metadata)
            ledger = VectorIndexLedger(store)
            indexed_hashes = ledger.hashes(metadata['generation_id'], new_chunks)
            for chunk_citation in new_chunks:
                self.assertIn(chunk_citation, indexed_hashes)
            self.assertEqual(read_dirty_citations(SQLiteStore(vault / 'index' / 'trove.sqlite')), [])

            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                second = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertEqual(second['vector']['status'], 'skipped')
            self.assertEqual(second['vector']['reason'], 'no_dirty_citations')

    def test_sync_links_cached_media_but_does_not_enqueue_image_precompute_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, _table = self.make_snapshot(vault)
            cache = acct / 'cache'
            cache.mkdir()
            (cache / 'sync-photo.jpg').write_bytes(b'\xff\xd8\xfffixture')
            with sqlite3.connect(acct / 'message_resource.db') as conn:
                conn.execute('CREATE TABLE resource(local_id INTEGER, local_type TEXT, file_path TEXT)')
                conn.execute('INSERT INTO resource VALUES(?,?,?)', (1, '3', 'cache/sync-photo.jpg'))
                conn.commit()

            report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertTrue(report['ok'])
            self.assertEqual(report['media']['accepted_links'], 1)
            self.assertEqual(report['media']['jobs']['queued'], 0)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM media_jobs WHERE job_type='image_observe'").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM media_assets WHERE modality='image'").fetchone()[0], 1)

    def test_sync_source_error_redacts_account_directory(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current = vault / 'sources' / 'wechat-kos-decrypted' / 'current'
            acct = current / 'com.tencent.xinWeChat__wxid_secret_fixture'
            acct.mkdir(parents=True)
            with sqlite3.connect(acct / 'contact.db') as conn:
                conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
                conn.commit()
            (acct / 'message_0.db').write_text('not a sqlite database', encoding='utf-8')

            report = run_sync(vault)

            self.assertFalse(report['ok'])
            self.assertEqual(report['errors'], ['source_1: DatabaseError'])
            combined = str(report)
            self.assertNotIn('wxid_secret_fixture', combined)
            self.assertNotIn(acct.name, combined)

    def test_sync_recommends_rebuild_instead_of_incremental_index_on_rebuild_required(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))

            with patch('trove_core.sync.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.sync.vector_status_payload', return_value={'reason_code': 'top-level', 'zvec': {'rebuild_required': True, 'reason_code': 'zvec_rebuild_required'}}), \
                 patch('trove_core.sync.index_vectors') as index_vectors:
                report = maybe_index_vectors(cfg, changed=True, backend='zvec')

            self.assertEqual(report['status'], 'recommend_rebuild')
            self.assertEqual(report['backend'], 'zvec')
            self.assertEqual(report['reason_code'], 'zvec_rebuild_required')
            self.assertFalse(report['auto_rebuild'])
            index_vectors.assert_not_called()

    def test_sync_indexes_dirty_zvec_catchup_pending_collection(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))

            with patch('trove_core.sync.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.sync.vector_status_payload', return_value={'zvec': {'collection_exists': True, 'complete': False, 'metadata_complete': True, 'catchup_pending': True, 'rebuild_required': False, 'reason_code': 'zvec_catchup_pending', 'provider_mismatch': False, 'stale': False, 'incomplete': False}}), \
                 patch('trove_core.sync.index_vectors', return_value={'backend': 'zvec', 'indexed': 1}) as index_vectors:
                report = maybe_index_vectors(
                    cfg,
                    changed=True,
                    backend='zvec',
                    citations=['trove://wechat/acct/conv/message_0/1'],
                )

            self.assertEqual(report['status'], 'indexed')
            self.assertEqual(report['backend'], 'zvec')
            self.assertEqual(report['dirty_count'], 1)
            index_vectors.assert_called_once()

    def test_maybe_index_vectors_processes_bounded_prefix_over_inline_limit(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))
            sync_config = cfg.root / 'jobs' / 'sync_config.redacted.json'
            sync_config.parent.mkdir(parents=True, exist_ok=True)
            sync_config.write_text(json.dumps({'sync': {'vector_inline_dirty_limit': 2}}), encoding='utf-8')
            store = SQLiteStore(cfg.paths.sqlite_path)
            refs = [
                {'citation': f'trove://wechat/acct/conv/message_0/{idx}', 'account_id': 'acct', 'conversation_id': 'conv'}
                for idx in range(3)
            ]
            record_dirty_citations(store, refs)

            with patch('trove_core.sync.configured_embedding_provider', return_value=FakeProvider()) as provider, \
                 patch('trove_core.sync.vector_status_payload', return_value={'zvec': {'collection_exists': True, 'complete': True, 'metadata_complete': True, 'rebuild_required': False, 'provider_mismatch': False, 'stale': False, 'incomplete': False}}) as vector_status, \
                 patch('trove_core.sync.index_vectors', return_value={'backend': 'zvec', 'indexed': 2}) as index_vectors:
                report = maybe_index_vectors(cfg, changed=True, backend='zvec', citations=read_dirty_citations(store))

            self.assertEqual(report['status'], 'indexed')
            self.assertEqual(report['dirty_count'], 3)
            self.assertEqual(report['inline_limit'], 2)
            self.assertEqual(report['processed_dirty_count'], 2)
            self.assertEqual(report['remaining_dirty_count'], 1)
            self.assertEqual(report['deferred_dirty_count'], 1)
            provider.assert_called_once()
            vector_status.assert_called_once()
            self.assertEqual(index_vectors.call_args.kwargs['citations'], read_dirty_citations(store)[:2])
            self.assertEqual(len(read_dirty_citations(store)), 3)

    def test_consecutive_syncs_drain_large_dirty_queue_outside_writer(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, _acct, _table = self.make_snapshot(vault)
            run_sync(vault, options=SyncOptions(snapshot_dir=current, snapshot_media_enabled=False))
            cfg = VaultConfig.resolve(str(vault), env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            clear_dirty_citations(store, read_dirty_citations(store))
            refs = [
                {'citation': f'trove://wechat/acct/conv/message_0/{idx}', 'account_id': 'acct', 'conversation_id': 'conv'}
                for idx in range(5)
            ]
            record_dirty_citations(store, refs)
            expected = read_dirty_citations(store)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))
            sync_config = cfg.root / 'jobs' / 'sync_config.redacted.json'
            sync_config.write_text(json.dumps({'sync': {'vector_inline_dirty_limit': 2}}), encoding='utf-8')

            provider_calls = 0

            def provider_without_writer(*_args, **_kwargs):
                nonlocal provider_calls
                provider_calls += 1
                with VaultOperationCoordinator(cfg).write(owner='sync-test-probe'):
                    pass
                return FakeProvider()

            with patch('trove_core.sync.configured_embedding_provider', side_effect=provider_without_writer), \
                 patch('trove_core.sync.vector_status_payload', return_value={'zvec': {'collection_exists': True, 'complete': True, 'metadata_complete': True, 'rebuild_required': False, 'provider_mismatch': False, 'stale': False, 'incomplete': False}}), \
                 patch('trove_core.sync.index_vectors', side_effect=lambda *_args, **kwargs: {'backend': 'zvec', 'indexed': len(kwargs['citations'])}) as index_vectors:
                reports = [
                    run_sync(vault, options=SyncOptions(snapshot_dir=current, snapshot_media_enabled=False))
                    for _ in range(3)
                ]

            self.assertEqual([report['vector']['processed_dirty_count'] for report in reports], [2, 2, 1])
            self.assertEqual([report['vector']['remaining_dirty_count'] for report in reports], [3, 1, 0])
            self.assertEqual([report['vector']['deferred_dirty_count'] for report in reports], [3, 1, 0])
            self.assertEqual([report['vector']['dirty_cleared'] for report in reports], [2, 2, 1])
            self.assertEqual(read_dirty_citations(store), [])
            self.assertEqual(provider_calls, 3)
            self.assertEqual(
                [call.kwargs['citations'] for call in index_vectors.call_args_list],
                [expected[:2], expected[2:4], expected[4:]],
            )
            self.assertTrue(all(call.kwargs['write_session'] is None for call in index_vectors.call_args_list))

    def test_dirty_queue_is_typed_deferred_when_config_or_provider_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            citations = ['trove://wechat/acct/conv/message_0/1']

            config_deferred = maybe_index_vectors(cfg, changed=True, citations=citations)
            self.assertEqual(config_deferred['status'], 'deferred')
            self.assertEqual(config_deferred['reason'], 'process_config_vector_index_not_incremental')
            self.assertEqual(config_deferred['processed_dirty_count'], 0)
            self.assertEqual(config_deferred['remaining_dirty_count'], 1)

            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))
            with patch('trove_core.sync.configured_embedding_provider', return_value=None), \
                 patch('trove_core.sync.index_vectors') as index_vectors:
                provider_deferred = maybe_index_vectors(cfg, changed=True, citations=citations)
            self.assertEqual(provider_deferred['status'], 'deferred')
            self.assertEqual(provider_deferred['reason'], 'embedding_provider_unavailable')
            self.assertEqual(provider_deferred['processed_dirty_count'], 0)
            self.assertEqual(provider_deferred['remaining_dirty_count'], 1)
            index_vectors.assert_not_called()

    def test_sync_recommends_rebuild_for_dirty_index_when_zvec_collection_missing(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental', 'vector_index': 'incremental'}))

            with patch('trove_core.sync.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.sync.vector_status_payload', return_value={'zvec': {'collection_exists': False, 'complete': False, 'rebuild_required': False, 'reason_code': 'zvec_collection_missing'}}), \
                 patch('trove_core.sync.index_vectors') as index_vectors:
                report = maybe_index_vectors(
                    cfg,
                    changed=True,
                    backend='zvec',
                    citations=['trove://wechat/acct/conv/message_0/1'],
                )

            self.assertEqual(report['status'], 'recommend_rebuild')
            self.assertEqual(report['backend'], 'zvec')
            self.assertEqual(report['reason_code'], 'zvec_collection_missing')
            self.assertEqual(report['dirty_count'], 1)
            self.assertFalse(report['auto_rebuild'])
            index_vectors.assert_not_called()


if __name__ == '__main__':
    unittest.main()
