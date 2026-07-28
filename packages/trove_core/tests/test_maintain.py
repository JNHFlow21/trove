from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.knowledge.profile_automation import process_profile_refresh_queue
from trove_core.knowledge.profile_snapshots import list_profile_snapshots
from trove_core.maintain import MaintainOptions, maintain_vectors, run_maintain
from trove_core.store.repositories import EntityRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import FTS_TOKENIZER_VERSION, SQLiteStore
from trove_core.sync import dirty_citation_count, record_dirty_citations
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.models import Account, Conversation, Message


class MaintainTests(unittest.TestCase):
    def test_maintain_reconciles_profile_changes_missed_by_sync_hooks(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-maintain-auto', entity_type='Customer',
                display_name='Maintain Auto Customer',
                identifiers={'wechat_id': 'wxid-maintain-auto'},
            ))
            repo = WeChatRepository(store)
            repo.replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-maintain-auto', 'acct', 'Maintain Auto Customer', 'private')],
                [Message(
                    'acct', 'A', 'wxid-maintain-auto', 'Maintain Auto Customer',
                    'private', 'wxid-maintain-auto', 'Maintain Auto Customer',
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    'fixture maintain profile baseline', 's', 1,
                )],
            )
            agent_tools.profile_automation_enable(
                vault, 'Maintain Auto Customer', debounce_seconds=180,
            )
            self.assertEqual(
                process_profile_refresh_queue(vault, limit=5)['created_snapshots'], 1,
            )
            repo.replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-maintain-auto', 'acct', 'Maintain Auto Customer', 'private')],
                [Message(
                    'acct', 'A', 'wxid-maintain-auto', 'Maintain Auto Customer',
                    'private', 'wxid-maintain-auto', 'Maintain Auto Customer',
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                    'fixture maintain profile changed', 's', 2,
                )],
            )

            report = run_maintain(vault)

            self.assertTrue(report['ok'])
            self.assertEqual(report['profiles']['reconcile_queue']['queued'], 1)
            self.assertEqual(report['profiles']['created_snapshots'], 1)
            self.assertEqual(
                list_profile_snapshots(store, 'Maintain Auto Customer')['count'], 2,
            )

    def test_profile_worker_exception_preserves_completed_maintenance_reports(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)

            with patch(
                'trove_core.maintain.process_profile_refresh_queue',
                side_effect=RuntimeError('private fixture detail'),
            ):
                report = run_maintain(vault)

            self.assertEqual(report['status'], 'action_required')
            self.assertFalse(report['ok'])
            self.assertTrue(report['schema'])
            self.assertTrue(report['storage'])
            self.assertEqual(report['profiles']['error_code'], 'RuntimeError')
            self.assertNotIn('private fixture detail', str(report))

    def test_vector_status_only_path_uses_readonly_store_and_skips_dirty_rows(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            SQLiteStore(cfg.paths.sqlite_path).initialize()

            def readonly_count(store):
                self.assertTrue(store.readonly)
                return 0

            with patch('trove_core.maintain.read_latest_process_config', return_value={'config': {}}), \
                 patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value={'state': 'available', 'zvec': {'state': 'available'}}), \
                 patch('trove_core.maintain.dirty_citation_count', side_effect=readonly_count), \
                 patch('trove_core.maintain.read_dirty_citation_batch') as read_dirty:
                report = maintain_vectors(cfg, options=MaintainOptions(), execute=False)

            self.assertEqual(report['status'], 'healthy')
            read_dirty.assert_not_called()

    def test_vector_maintenance_processes_a_bounded_dirty_batch(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            refs = [
                {'citation': f'fixture://dirty/{index:04d}', 'source_type': 'message'}
                for index in range(600)
            ]
            self.assertEqual(record_dirty_citations(store, refs), 600)
            status = {
                'state': 'available',
                'zvec': {
                    'state': 'available',
                    'stale': False,
                    'incomplete': False,
                    'catchup_pending': False,
                    'provider_mismatch': False,
                    'rebuild_required': False,
                    'reason_code': None,
                },
            }
            with patch('trove_core.maintain.read_latest_process_config', return_value={'config': {'vector_index': 'incremental'}}), \
                 patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value=status), \
                 patch('trove_core.maintain.index_vectors', return_value={'backend': 'zvec', 'indexed': 512}) as index_vectors:
                report = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            citations = index_vectors.call_args.kwargs['citations']
            self.assertEqual(len(citations), 512)
            self.assertEqual(report['dirty_count'], 600)
            self.assertEqual(report['dirty_batch_count'], 512)
            self.assertEqual(report['dirty_remaining'], 88)
            self.assertEqual(dirty_citation_count(store), 88)

    def test_vector_maintenance_does_not_clear_a_citation_redirtied_after_indexing(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            ref = {'citation': 'fixture://dirty/raced', 'source_type': 'message'}
            record_dirty_citations(store, [ref])
            status = {'state': 'available', 'zvec': {
                'state': 'available', 'stale': False, 'incomplete': False,
                'catchup_pending': False, 'provider_mismatch': False,
                'rebuild_required': False, 'reason_code': None,
            }}

            def index_then_redirty(*_args, **_kwargs):
                record_dirty_citations(store, [ref])
                return {'backend': 'zvec', 'indexed': 1}

            with patch('trove_core.maintain.read_latest_process_config', return_value={'config': {'vector_index': 'incremental'}}), \
                 patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value=status), \
                 patch('trove_core.maintain.index_vectors', side_effect=index_then_redirty):
                report = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            self.assertEqual(report['status'], 'indexed')
            self.assertEqual(report['dirty_cleared'], 0)
            self.assertEqual(dirty_citation_count(store), 1)

    def test_auto_rebuild_clears_only_the_dirty_snapshot_it_indexed(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            first = {'citation': 'fixture://dirty/first', 'source_type': 'message'}
            later = {'citation': 'fixture://dirty/later', 'source_type': 'message'}
            record_dirty_citations(store, [first])
            status = {'state': 'available', 'zvec': {
                'state': 'available', 'stale': False, 'incomplete': False,
                'catchup_pending': False, 'provider_mismatch': False,
                'rebuild_required': False, 'reason_code': None,
            }}

            def rebuild_then_new_dirty(*_args, **_kwargs):
                record_dirty_citations(store, [later])
                return {'backend': 'zvec', 'indexed': 1}

            with patch('trove_core.maintain.read_latest_process_config', return_value={'config': {'vector_index': 'incremental'}}), \
                 patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value=status), \
                 patch('trove_core.maintain.index_vectors', side_effect=rebuild_then_new_dirty):
                report = maintain_vectors(
                    cfg,
                    options=MaintainOptions(vector_backend='zvec', auto_rebuild=True),
                )

            self.assertEqual(report['status'], 'indexed')
            self.assertEqual(report['dirty_cleared'], 1)
            self.assertEqual(dirty_citation_count(store), 1)

    def test_maintain_repairs_drift_cleans_orphans_and_rotates_backups(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            sqlite_path = vault / 'index' / 'trove.sqlite'
            store = SQLiteStore(sqlite_path)
            with store.connect() as conn:
                conn.execute("INSERT OR REPLACE INTO schema_meta(key,value) VALUES('fts_tokenizer','legacy-tokenizer')")
                conn.execute("""INSERT OR REPLACE INTO evidence_chunks(chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                              VALUES('orphan','orphan','missing-parent','acct','acct','message','conv','title','actor','2026-01-01T00:00:00Z','orphan content',0,'{}','active','2026-01-01T00:00:00Z')""")
                conn.execute("""CREATE TABLE IF NOT EXISTS vector_entries (
                    citation TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    content_hash TEXT
                )""")
                conn.execute("INSERT OR REPLACE INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES('orphan','fake',1,'[0]','x')")
                conn.commit()
            for idx in range(4):
                backup = sqlite_path.with_name(f'trove.sqlite.bak-old-{idx}')
                backup.write_text('old', encoding='utf-8')
                os.utime(backup, (1 + idx, 1 + idx))

            report = run_maintain(vault, options=MaintainOptions(
                backup_retention=2,
                log_retention=2,
                full_scan=True,
            ))
            self.assertTrue(report['ok'])
            self.assertEqual(report['chunks']['removed_orphan_chunks'], 1)
            self.assertGreaterEqual(report['chunks']['removed_orphan_vectors'], 1)
            self.assertGreaterEqual(report['chunks']['dirty_recorded'], 1)
            self.assertTrue(report['schema']['fts_tokenizer_repaired'])
            self.assertEqual(report['schema']['fts_tokenizer'], FTS_TOKENIZER_VERSION)
            self.assertLessEqual(report['backups']['retained_count'], 2)
            self.assertFalse(report['raw_content_included'])
            self.assertNotIn(str(vault), str(report))
            with sqlite3.connect(sqlite_path) as conn:
                self.assertIsNone(conn.execute("SELECT 1 FROM evidence_chunks WHERE chunk_citation='orphan'").fetchone())
                self.assertIsNotNone(conn.execute("SELECT 1 FROM sync_dirty_citations WHERE citation='missing-parent'").fetchone())
                self.assertEqual(conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()[0], FTS_TOKENIZER_VERSION)

    def test_maintain_cli_style_missing_provider_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            report = run_maintain(vault)
            self.assertTrue(report['ok'])
            self.assertIn(report['vector']['status'], {'skipped', 'healthy', 'recommend_rebuild'})

    def test_maintain_skips_backup_without_destructive_repair(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            sqlite_path = vault / 'index' / 'trove.sqlite'

            report = run_maintain(vault, options=MaintainOptions(backup_retention=2))

            self.assertTrue(report['ok'])
            self.assertIsNone(report['backups']['created'])
            self.assertEqual(report['backups']['skipped_reason'], 'no_destructive_repair')
            self.assertEqual(report['backups']['destructive_reasons'], [])
            self.assertFalse(list(sqlite_path.parent.glob('trove.sqlite.bak-*')))

    def test_maintain_always_backup_preserves_legacy_behavior(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)

            report = run_maintain(vault, options=MaintainOptions(always_backup=True))

            self.assertTrue(report['ok'])
            self.assertIsNotNone(report['backups']['created'])
            self.assertIn('always_backup', report['backups']['destructive_reasons'])


    def test_maintain_backs_up_existing_db_missing_schema_meta(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            sqlite_path = vault / 'index' / 'trove.sqlite'
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute("DELETE FROM schema_meta WHERE key IN ('schema_version','fts_tokenizer')")
                conn.commit()

            report = run_maintain(vault, options=MaintainOptions(backup_retention=2))

            self.assertTrue(report['ok'])
            self.assertIsNotNone(report['backups']['created'])
            self.assertTrue(sqlite_path.with_name(report['backups']['created']).exists())
            with sqlite3.connect(sqlite_path) as conn:
                self.assertEqual(conn.execute("SELECT value FROM schema_meta WHERE key='fts_tokenizer'").fetchone()[0], FTS_TOKENIZER_VERSION)

    def test_maintain_skips_backup_for_new_empty_existing_db_file(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            sqlite_path = vault / 'index' / 'trove.sqlite'
            sqlite_path.parent.mkdir(parents=True)
            sqlite3.connect(sqlite_path).close()

            report = run_maintain(vault, options=MaintainOptions(backup_retention=2))

            self.assertTrue(report['ok'])
            self.assertIsNone(report['backups']['created'])
            self.assertEqual(report['backups']['skipped_reason'], 'no_destructive_repair')

    def test_maintain_reads_nested_stale_zvec_status_and_recommends_rebuild(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            nested_status = {
                'state': 'unavailable_fallback',
                'reason_code': 'top-level',
                'zvec': {
                    'state': 'unavailable_fallback',
                    'stale': True,
                    'incomplete': False,
                    'provider_mismatch': False,
                    'rebuild_required': True,
                    'reason_code': 'zvec_rebuild_required',
                },
            }

            with patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value=nested_status), \
                 patch('trove_core.maintain.index_vectors') as index_vectors:
                report = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            self.assertEqual(report['status'], 'recommend_rebuild')
            self.assertEqual(report['backend'], 'zvec')
            self.assertEqual(report['reason_code'], 'zvec_rebuild_required')
            index_vectors.assert_not_called()

    def test_maintain_requires_explicit_full_scan_for_incomplete_zvec_without_dirty_journal(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            nested_status = {
                'state': 'unavailable_fallback',
                'zvec': {
                    'state': 'unavailable_fallback',
                    'stale': False,
                    'incomplete': True,
                    'provider_mismatch': False,
                    'rebuild_required': True,
                    'reason_code': 'zvec_rebuild_required',
                },
            }

            with patch('trove_core.maintain.read_latest_process_config', return_value={'config': {'vector_index': 'incremental'}}), \
                 patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value=nested_status), \
                 patch('trove_core.maintain.index_vectors', return_value={'backend': 'zvec', 'indexed': 2}) as index_vectors:
                report = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            self.assertEqual(report['status'], 'full_scan_required')
            self.assertEqual(report['backend'], 'zvec')
            self.assertEqual(report['reason_code'], 'dirty_journal_gap')
            index_vectors.assert_not_called()

    def test_maintain_requires_explicit_full_scan_for_catchup_without_dirty_journal(self):
        class FakeProvider:
            provider_name = 'fake'
            model_id = 'fake-model'
            dimensions = 3
            request_format = ''

        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            nested_status = {
                'state': 'available',
                'zvec': {
                    'state': 'available',
                    'stale': False,
                    'incomplete': False,
                    'catchup_pending': True,
                    'provider_mismatch': False,
                    'rebuild_required': False,
                    'reason_code': 'zvec_catchup_pending',
                },
            }

            with patch('trove_core.maintain.read_latest_process_config', return_value={'config': {'vector_index': 'incremental'}}), \
                 patch('trove_core.maintain.configured_embedding_provider', return_value=FakeProvider()), \
                 patch('trove_core.maintain.vector_status_payload', return_value=nested_status), \
                 patch('trove_core.maintain.index_vectors', return_value={'backend': 'zvec', 'indexed': 2}) as index_vectors:
                report = maintain_vectors(cfg, options=MaintainOptions(vector_backend='zvec'))

            self.assertEqual(report['status'], 'full_scan_required')
            self.assertEqual(report['backend'], 'zvec')
            self.assertEqual(report['reason_code'], 'dirty_journal_gap')
            index_vectors.assert_not_called()

    def test_routine_maintain_skips_full_scans_models_and_provider_execution(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            with patch('trove_core.maintain._fts_count_report', side_effect=AssertionError('full FTS count used')), \
                 patch('trove_core.maintain.index_vectors', side_effect=AssertionError('provider execution used')):
                report = run_maintain(vault)

            self.assertTrue(report['ok'])
            self.assertEqual(report['chunks']['scan_mode'], 'dirty_only')
            self.assertEqual(report['fts']['scan_mode'], 'structural')
            self.assertEqual(report['storage']['fts_optimized'], [])
            self.assertFalse(report['storage']['pragma_optimize'])
            self.assertEqual(report['media']['status'], 'healthy')

    def test_explicit_model_work_is_not_silently_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            with patch('trove_core.maintain.index_vectors', side_effect=AssertionError('provider execution used')):
                report = run_maintain(
                    vault,
                    options=MaintainOptions(auto_rebuild=True, media_voice_budget=1),
                )

            self.assertFalse(report['ok'])
            self.assertEqual(report['status'], 'action_required')
            self.assertEqual(report['vector']['status'], 'deferred')
            self.assertEqual(report['vector']['reason_code'], 'explicit_vector_rebuild_required')
            self.assertEqual(report['media']['status'], 'deferred')
            self.assertEqual(report['media']['reason'], 'explicit_media_command_required')

    def test_readonly_integrity_planning_runs_outside_writer_window(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            active = 0
            from trove_core.maintain import coordinated_vault_mutation as original_mutation

            @contextmanager
            def tracked_mutation(*args, **kwargs):
                nonlocal active
                with original_mutation(*args, **kwargs) as session:
                    active += 1
                    try:
                        yield session
                    finally:
                        active -= 1

            def planned_scan(*_args, **_kwargs):
                self.assertEqual(active, 0)
                return False

            def provider_status(*_args, **_kwargs):
                self.assertEqual(active, 0)
                return {'status': 'healthy'}

            def media_status(*_args, **_kwargs):
                self.assertEqual(active, 0)
                return {'queue': {}}

            with patch('trove_core.maintain.coordinated_vault_mutation', side_effect=tracked_mutation), \
                 patch('trove_core.maintain.orphan_cleanup_needed', side_effect=planned_scan), \
                 patch('trove_core.maintain.fts_repair_needed', side_effect=planned_scan), \
                 patch('trove_core.maintain.storage_vacuum_needed', side_effect=planned_scan), \
                 patch('trove_core.maintain.maintain_vectors', side_effect=provider_status), \
                 patch('trove_core.maintain.media_status_payload', side_effect=media_status):
                report = run_maintain(vault, options=MaintainOptions(full_scan=True))

            self.assertTrue(report['ok'])
            self.assertEqual(active, 0)


if __name__ == '__main__':
    unittest.main()
