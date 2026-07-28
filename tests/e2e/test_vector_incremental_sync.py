from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.maintain import MaintainOptions, maintain_vectors
from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.runtime import index_vectors, vector_status_payload, zvec_collection_path
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.sync import SyncOptions, clear_dirty_citations, read_dirty_citations, run_sync
from trove_core.vault.config import VaultConfig
from trove_core.vector.zvec_store import ZVecStore
from trove_core.vector.ledger import VectorIndexLedger
from trove_core.wechat.importers.wechat_decrypted import msg_table_for
from trove_core.wechat.process_config import process_config_from_payload, write_process_config


class VectorIncrementalSyncE2ETests(unittest.TestCase):
    def _make_snapshot(self, vault: Path) -> tuple[Path, Path, str]:
        current = vault / 'sources' / 'wechat-kos-decrypted' / 'current'
        acct = current / 'com.tencent.xinWeChat__wxid_vector_e2e_fixture'
        acct.mkdir(parents=True)
        with sqlite3.connect(acct / 'contact.db') as conn:
            conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
            conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
            conn.execute('INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)', ('wxid_vector_friend', 'Vector Fixture', '', ''))
            conn.commit()
        table = msg_table_for('wxid_vector_friend')
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
            conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_vector_friend', 1))
            conn.execute(f'''CREATE TABLE {table} (
                local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
                real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
                download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
                message_content TEXT, compress_content BLOB, packed_info_data BLOB,
                WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
            )''')
            conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (1, 1, 1710000000, 'fixture vector e2e baseline token'))
            conn.commit()
        with sqlite3.connect(acct / 'sns.db') as conn:
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-m1', 'wxid_vector_friend', '<TimelineObject><id>m1</id><username>wxid_vector_friend</username><createTime>1760000000</createTime><contentDesc>fixture vector e2e baseline moment token</contentDesc></TimelineObject>', b''))
            conn.commit()
        return current, acct, table

    def _append_message(self, acct: Path, table: str) -> None:
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute(
                f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)',
                (2, 1, 1710000060, 'fixture vector e2e incremental message token'),
            )
            conn.commit()

    def _append_messages(self, acct: Path, table: str, *, count: int) -> None:
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.executemany(
                f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)',
                [
                    (local_id, 1, 1710000000 + local_id * 60, f'fixture vector bounded catchup token {local_id}')
                    for local_id in range(2, 2 + count)
                ],
            )
            conn.commit()

    def _append_moment(self, acct: Path) -> None:
        with sqlite3.connect(acct / 'sns.db') as conn:
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-m2', 'wxid_vector_friend', '<TimelineObject><id>m2</id><username>wxid_vector_friend</username><createTime>1760000060</createTime><contentDesc>fixture vector e2e incremental moment token</contentDesc></TimelineObject>', b''))
            conn.commit()

    def _replace_message_content(self, acct: Path, table: str) -> None:
        with sqlite3.connect(acct / 'message_0.db') as conn:
            conn.execute(
                f'UPDATE {table} SET message_content=? WHERE local_id=?',
                ('fixture vector e2e updated same chunk token', 1),
            )
            conn.commit()

    def _write_inline_limit(self, vault: Path, limit: int) -> None:
        sync_config = vault / 'jobs' / 'sync_config.redacted.json'
        sync_config.parent.mkdir(parents=True, exist_ok=True)
        sync_config.write_text(json.dumps({'sync': {'vector_inline_dirty_limit': limit}}), encoding='utf-8')

    def test_full_rebuild_clears_the_historical_dirty_backlog(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, _acct, _table = self._make_snapshot(vault)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({
                'config_id': 'pcfg-incremental-e2e',
                'vector_index': 'incremental',
            }))

            self.assertTrue(run_sync(vault, options=SyncOptions(snapshot_dir=current))['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            dirty_before = len(read_dirty_citations(store))
            self.assertGreater(dirty_before, 0)

            rebuilt = index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)

            self.assertEqual(rebuilt['dirty_cleared'], dirty_before)
            self.assertEqual(read_dirty_citations(store), [])
            self.assertTrue(rebuilt['vector']['complete'])

    def test_zvec_incremental_sync_indexes_new_message_and_moment_without_status_mocks(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self._make_snapshot(vault)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental-e2e', 'vector_index': 'incremental'}))

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertTrue(first['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)
            baseline_status = vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']
            self.assertTrue(baseline_status['complete'])
            clear_dirty_citations(store, read_dirty_citations(store))

            self._append_message(acct, table)
            self._append_moment(acct)

            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertEqual(report['vector']['status'], 'indexed')
            self.assertGreater(report['vector']['dirty_cleared'], 0)
            status = vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']
            self.assertTrue(status['complete'])
            self.assertFalse(status['rebuild_required'])
            metadata = json.loads(Path(str(zvec_collection_path(cfg)) + '.trove-meta.json').read_text(encoding='utf-8'))
            with store.connect() as conn:
                new_chunk_citations = [
                    row['chunk_citation']
                    for row in conn.execute(
                        """SELECT chunk_citation FROM evidence_chunks
                           WHERE parent_citation LIKE ? OR content LIKE ?
                           ORDER BY chunk_citation""",
                        ('%/message_0/2', '%vector e2e incremental moment token%'),
                    )
                ]
            self.assertGreaterEqual(len(new_chunk_citations), 2)
            self.assertNotIn('content_hashes', metadata)
            indexed_hashes = VectorIndexLedger(store).hashes(metadata['generation_id'], new_chunk_citations)
            for citation in new_chunk_citations:
                self.assertIn(citation, indexed_hashes)

            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                second = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertEqual(second['vector']['reason'], 'no_dirty_citations')

    def test_sync_consumes_one_bounded_dirty_batch_without_status_mocks(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self._make_snapshot(vault)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental-e2e', 'vector_index': 'incremental'}))
            self._write_inline_limit(vault, 1)

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertTrue(first['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)
            clear_dirty_citations(store, read_dirty_citations(store))

            self._append_message(acct, table)
            self._append_moment(acct)
            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                first_batch = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertEqual(first_batch['vector']['status'], 'indexed')
            self.assertEqual(first_batch['vector']['processed_dirty_count'], 1)
            first_remaining = first_batch['vector']['remaining_dirty_count']
            self.assertGreater(first_remaining, 0)
            self.assertEqual(len(read_dirty_citations(store)), first_remaining)

    def test_sync_drains_real_zvec_catchup_across_multiple_bounded_batches(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self._make_snapshot(vault)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({
                'config_id': 'pcfg-incremental-e2e',
                'vector_index': 'incremental',
            }))
            self._write_inline_limit(vault, 2)

            self.assertTrue(run_sync(vault, options=SyncOptions(snapshot_dir=current))['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)
            clear_dirty_citations(store, read_dirty_citations(store))
            self._append_messages(acct, table, count=5)

            remaining = []
            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                for expected in (3, 1, 0):
                    report = run_sync(vault, options=SyncOptions(snapshot_dir=current))
                    self.assertEqual(report['vector']['status'], 'indexed')
                    remaining.append(report['vector']['remaining_dirty_count'])
                    self.assertEqual(remaining[-1], expected)
                    status = vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']
                    self.assertFalse(status['rebuild_required'])
                    self.assertEqual(status['catchup_pending'], expected > 0)

            self.assertEqual([5, *remaining], [5, 3, 1, 0])
            self.assertTrue(vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']['complete'])

    def test_maintain_consumes_dirty_queue_when_zvec_status_is_complete(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self._make_snapshot(vault)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental-e2e', 'vector_index': 'incremental'}))
            self._write_inline_limit(vault, 1)

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertTrue(first['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)
            clear_dirty_citations(store, read_dirty_citations(store))

            self._replace_message_content(acct, table)
            with patch.dict('os.environ', {'TROVE_SYNC_VECTOR_INLINE_DIRTY_LIMIT': '0'}), \
                 patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                deferred = run_sync(vault, options=SyncOptions(full=True, snapshot_dir=current))

            self.assertEqual(deferred['vector']['status'], 'deferred')
            before = vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']
            self.assertTrue(before['complete'])
            self.assertFalse(before['catchup_pending'])
            self.assertFalse(before['rebuild_required'])
            dirty = read_dirty_citations(store)
            self.assertEqual(len(dirty), 1)

            with patch('trove_core.maintain.configured_embedding_provider', return_value=provider):
                maintained = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            self.assertEqual(maintained['status'], 'indexed')
            self.assertEqual(maintained['dirty_cleared'], len(dirty))
            self.assertEqual(read_dirty_citations(store), [])

    def test_missing_zvec_collection_recommends_rebuild_before_dirty_deferral(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            current, acct, table = self._make_snapshot(vault)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            write_process_config(vault, process_config_from_payload({'config_id': 'pcfg-incremental-e2e', 'vector_index': 'incremental'}))
            self._write_inline_limit(vault, 1)

            first = run_sync(vault, options=SyncOptions(snapshot_dir=current))
            self.assertTrue(first['ok'])
            store = SQLiteStore(cfg.paths.sqlite_path)
            zvec = ZVecStore(zvec_collection_path(cfg), store=store)
            if not zvec.available:
                self.skipTest('ZVEC optional dependency is unavailable')
            index_vectors(cfg, provider, backend='zvec', purge=True, use_lock=False)
            clear_dirty_citations(store, read_dirty_citations(store))
            zvec.reset_collection()
            self.assertFalse(vector_status_payload(cfg, backend='zvec', provider=provider)['zvec']['collection_exists'])

            self._append_message(acct, table)
            self._append_moment(acct)
            with patch('trove_core.sync.configured_embedding_provider', return_value=provider):
                report = run_sync(vault, options=SyncOptions(snapshot_dir=current))

            self.assertEqual(report['vector']['status'], 'recommend_rebuild')
            self.assertEqual(report['vector']['reason_code'], 'zvec_collection_missing')
            self.assertGreater(report['vector']['dirty_count'], report['vector'].get('inline_limit', 1))
            self.assertGreater(len(read_dirty_citations(store)), 1)

            with patch('trove_core.maintain.configured_embedding_provider', return_value=provider):
                maintained = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            self.assertEqual(maintained['status'], 'recommend_rebuild')
            self.assertEqual(maintained['reason_code'], 'zvec_collection_missing')
            self.assertGreater(len(read_dirty_citations(store)), 1)


if __name__ == '__main__':
    unittest.main()
