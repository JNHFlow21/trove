from __future__ import annotations

import hashlib
import re
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from trove_core.store import migrations as migrations_module
from trove_core.runtime import SearchRuntimeCache, build_search_engine
from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.search.query import SearchRequest
from trove_core.store.migrations import (
    EXPECTED_INDEXES,
    EXPECTED_TABLES,
    EXPECTED_TRIGGERS,
    SchemaMigrationRequired,
    SchemaPreflightUnavailable,
    SchemaVersionMismatch,
    SchemaVersionTooNew,
    foreign_key_compatibility_report,
    migrate_schema,
)
from trove_core.store.schema import FTS_TRIGGER_NAMES, MULTIMODAL_SCHEMA, SCHEMA_VERSION, TRIGRAM_FTS_SCHEMA
from trove_core.store.sqlite_store import ReadOnlyStoreError, SQLiteStore, open_store, schema_migration_required_payload
from trove_core.vault.config import VaultConfig
from trove_core.vector.sqlite_vector_store import SQLiteVectorStore


LEGACY_BASE = """
CREATE TABLE accounts(account_id TEXT PRIMARY KEY,label TEXT NOT NULL,display_name TEXT NOT NULL);
CREATE TABLE conversations(conversation_id TEXT NOT NULL,account_id TEXT NOT NULL,title TEXT NOT NULL,type TEXT NOT NULL,member_count INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(account_id,conversation_id));
CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT,citation TEXT NOT NULL UNIQUE,account_id TEXT NOT NULL,account_label TEXT NOT NULL,conversation_id TEXT NOT NULL,conversation_title TEXT NOT NULL,conversation_type TEXT NOT NULL,sender_id TEXT NOT NULL,sender_name TEXT NOT NULL,timestamp TEXT NOT NULL,content TEXT NOT NULL,shard_id TEXT NOT NULL,local_id INTEGER NOT NULL,sent_by_me INTEGER NOT NULL,source_type TEXT NOT NULL,direction TEXT NOT NULL,UNIQUE(account_id,conversation_id,shard_id,local_id));
CREATE VIRTUAL TABLE message_fts USING fts5(citation UNINDEXED,content,sender_name,conversation_title,tokenize='unicode61');
"""


class SchemaMigrationTests(unittest.TestCase):
    def _legacy(self, path: Path, version: int, *, with_row: bool = False, wrong_parent_index: bool = False) -> None:
        with sqlite3.connect(path) as conn:
            conn.executescript(LEGACY_BASE)
            if version:
                conn.execute('CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                conn.execute('INSERT INTO schema_meta VALUES(?,?)', ('schema_version', str(version)))
            if wrong_parent_index:
                evidence_ddl = next(
                    sql for sql in MULTIMODAL_SCHEMA
                    if sql.lstrip().startswith('CREATE TABLE IF NOT EXISTS evidence_chunks')
                )
                conn.execute(evidence_ddl)
                conn.execute(
                    'CREATE INDEX idx_evidence_chunks_parent '
                    'ON evidence_chunks(source_type,parent_citation)'
                )
            if with_row:
                conn.execute('INSERT INTO accounts VALUES(?,?,?)', ('acct-a', 'work', 'Work'))
                conn.execute('INSERT INTO conversations VALUES(?,?,?,?,?)', ('conv-a', 'acct-a', 'Fixture', 'private', 1))
                conn.execute(
                    """INSERT INTO messages(
                        citation,account_id,account_label,conversation_id,conversation_title,
                        conversation_type,sender_id,sender_name,timestamp,content,shard_id,
                        local_id,sent_by_me,source_type,direction
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    ('trove://fixture/1', 'acct-a', 'work', 'conv-a', 'Fixture', 'private',
                     'u1', 'Fixture', '2026-01-01T00:00:00Z', 'migration retained token',
                     's1', 1, 0, 'message', 'incoming'),
                )
            conn.execute(f'PRAGMA user_version={version}')
            conn.commit()

    @staticmethod
    def _semantic_manifest(path: Path) -> dict:
        with sqlite3.connect(path) as conn:
            tables = {
                name: tuple(sorted(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{name}")')))
                for name in sorted(EXPECTED_TABLES)
            }
            indexes = {
                name: tuple(str(row[2]) for row in conn.execute(f'PRAGMA index_info("{name}")'))
                for name in sorted(EXPECTED_INDEXES)
            }
            triggers = {
                name: ' '.join(str(conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
                ).fetchone()[0]).split())
                for name in sorted(EXPECTED_TRIGGERS)
            }
        return {'tables': tables, 'indexes': indexes, 'triggers': triggers}

    def test_empty_and_historical_schemas_converge_to_one_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            fresh = root / 'fresh.sqlite'
            SQLiteStore(fresh).initialize()
            paths.append(fresh)
            for version in range(SCHEMA_VERSION):
                path = root / f'legacy-{version}.sqlite'
                self._legacy(path, version, with_row=version == 1)
                SQLiteStore(path).initialize()
                paths.append(path)
            manifests = [self._semantic_manifest(path) for path in paths]
            self.assertTrue(all(manifest == manifests[0] for manifest in manifests[1:]))
            migrated = SQLiteStore(root / 'legacy-1.sqlite')
            self.assertEqual(migrated.schema_version(), SCHEMA_VERSION)
            self.assertTrue(migrated.exact_search('retained token'))

    def test_v15_vector_generation_retains_active_row_with_revision_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v15.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute(
                    """INSERT INTO vector_index_generations(
                           backend,generation_id,status,vector_text_version,
                           embedding_provider,embedding_model,dimensions,
                           expected_count,indexed_count,created_at,activated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    ('zvec', 'active-v15', 'active', 3, 'fixture', 'fixture', 16, 1, 1, 'now', 'now'),
                )
                conn.execute('ALTER TABLE vector_index_generations DROP COLUMN revision')
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version','15')"
                )
                conn.execute('PRAGMA user_version=15')
                conn.commit()

            migrated = SQLiteStore(path)
            migrated.initialize()
            with migrated.connect() as conn:
                columns = {str(row[1]) for row in conn.execute(
                    'PRAGMA table_info(vector_index_generations)'
                )}
                row = conn.execute(
                    """SELECT status,indexed_count,revision
                       FROM vector_index_generations
                       WHERE backend='zvec' AND generation_id='active-v15'"""
                ).fetchone()
            self.assertIn('revision', columns)
            self.assertIsNotNone(row)
            self.assertEqual((row['status'], row['indexed_count'], row['revision']), ('active', 1, 1))

    def test_v16_migration_backfills_normalized_entity_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v16.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute(
                    """INSERT INTO entities(
                           entity_id,entity_type,display_name,identifiers_json,status,
                           confidence,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        'customer-migration', 'Customer', '  Alice  ',
                        '{"wechat_username":"WXID-Alice","aliases":["Ａｌｉｃｅ", "Ally"]}',
                        'active', 0.9, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                    ),
                )
                conn.execute(
                    """INSERT INTO observations(
                           observation_id,entity_id,observation_type,value_json,status,
                           confidence,citation,source_type,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'obs-alice-remark', 'customer-migration', 'remark', '{"text":"  Friend Alice "}',
                        'active', 0.8, 'trove://fixture/contact/alice', 'contact',
                        '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                    ),
                )
                conn.execute('DROP TABLE entity_identifiers')
                conn.execute("UPDATE schema_meta SET value='16' WHERE key='schema_version'")
                conn.execute('PRAGMA user_version=16')
                conn.commit()

            migrated = SQLiteStore(path)
            migrated.initialize()
            with migrated.connect() as conn:
                rows = {
                    (row['identifier_type'], row['normalized_value'], row['source'], row['citation'])
                    for row in conn.execute(
                        'SELECT identifier_type,normalized_value,source,citation FROM entity_identifiers WHERE entity_id=?',
                        ('customer-migration',),
                    )
                }
            self.assertIn(('display_name', 'alice', 'entity', None), rows)
            self.assertIn(('user_id', 'wxid-alice', 'entity', None), rows)
            self.assertIn(('aliases', 'ally', 'entity', None), rows)
            self.assertIn(('remark', 'friend alice', 'observation', 'trove://fixture/contact/alice'), rows)

    def test_v16_legacy_image_columns_are_added_before_current_index_ddl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v16-legacy-image.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute('DROP INDEX idx_image_observations_projection_identity')
                for column in ('updated_at', 'prompt_version', 'model_id', 'content_sha256'):
                    conn.execute(f'ALTER TABLE image_observations DROP COLUMN {column}')
                conn.execute('DROP TABLE derived_data_purge_audit')
                conn.execute("UPDATE schema_meta SET value='16' WHERE key='schema_version'")
                conn.execute('PRAGMA user_version=16')
                conn.commit()

            migrated = SQLiteStore(path)
            migrated.initialize()
            with migrated.connect() as conn:
                columns = {str(row[1]) for row in conn.execute(
                    'PRAGMA table_info(image_observations)'
                )}
                projection_index = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_image_observations_projection_identity'"
                ).fetchone()
                audit_table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='derived_data_purge_audit'"
                ).fetchone()
            self.assertTrue({'content_sha256', 'model_id', 'prompt_version', 'updated_at'} <= columns)
            self.assertIsNotNone(projection_index)
            self.assertIsNotNone(audit_table)
            self.assertEqual(migrated.schema_version(), SCHEMA_VERSION)

    def test_latest_schema_repairs_missing_vector_generation_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'drifted-v16.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute('ALTER TABLE vector_index_generations DROP COLUMN revision')
                conn.commit()

            repaired = SQLiteStore(path)
            repaired.initialize()
            with repaired.connect() as conn:
                columns = {str(row[1]) for row in conn.execute(
                    'PRAGMA table_info(vector_index_generations)'
                )}
            self.assertIn('revision', columns)

    def test_fault_rolls_back_version_and_manifest_then_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite'
            self._legacy(path, 11)
            with sqlite3.connect(path) as conn:
                def fail(stage: str) -> None:
                    if stage == 'after_manifest_ddl':
                        raise RuntimeError('injected migration fault')

                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    migrate_schema(conn, fault_injector=fail)
                self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 11)
                self.assertIsNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='sync_state'"
                ).fetchone())
                migrate_schema(conn)
                self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)

    def test_faults_at_each_commit_boundary_leave_legacy_version_retryable(self):
        for stage in ('after_manifest_ddl', 'after_fts_drop', 'before_version_write', 'before_commit'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'legacy.sqlite'
                self._legacy(path, 11)
                with sqlite3.connect(path) as conn:
                    def fail(current_stage: str) -> None:
                        if current_stage == stage:
                            raise RuntimeError(stage)

                    with self.assertRaisesRegex(RuntimeError, stage):
                        migrate_schema(conn, fault_injector=fail)
                    self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 11)
                    self.assertIsNotNone(conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='message_fts'"
                    ).fetchone())
                    migrate_schema(conn)
                    self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], SCHEMA_VERSION)

    def test_concurrent_migrators_apply_manifest_once_after_cross_process_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite'
            self._legacy(path, 11)
            barrier = threading.Barrier(6)
            statements: list[str] = []
            statement_lock = threading.Lock()
            errors: list[BaseException] = []

            def run() -> None:
                try:
                    with sqlite3.connect(path, timeout=10) as conn:
                        conn.set_trace_callback(
                            lambda sql: (statement_lock.acquire(), statements.append(sql), statement_lock.release())
                        )
                        barrier.wait()
                        migrate_schema(conn)
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=run) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            manifest_applies = sum(
                'DROP INDEX IF EXISTS idx_media_assets_unique_ref' in statement
                for statement in statements
            )
            self.assertEqual(manifest_applies, 1)

    def test_future_version_is_never_downgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'future.sqlite'
            with sqlite3.connect(path) as conn:
                conn.execute(f'PRAGMA user_version={SCHEMA_VERSION + 1}')
                conn.commit()
            before = {
                item.name: (item.stat().st_mtime_ns, item.read_bytes())
                for item in path.parent.iterdir() if item.is_file()
            }
            with self.assertRaises(SchemaVersionTooNew):
                SQLiteStore(path).initialize()
            after = {
                item.name: (item.stat().st_mtime_ns, item.read_bytes())
                for item in path.parent.iterdir() if item.is_file()
            }
            self.assertEqual(after, before)

    def test_future_metadata_version_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'future-meta.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION + 1),),
                )
                conn.commit()
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                conn.execute('PRAGMA journal_mode=DELETE')
            before = {
                item.name: (item.stat().st_mtime_ns, item.read_bytes())
                for item in path.parent.iterdir() if item.is_file()
            }
            with self.assertRaises(SchemaVersionTooNew):
                open_store(path, readonly=True)
            with self.assertRaises(SchemaVersionTooNew):
                SQLiteStore(path).initialize()
            after = {
                item.name: (item.stat().st_mtime_ns, item.read_bytes())
                for item in path.parent.iterdir() if item.is_file()
            }
            self.assertEqual(after, before)

    def test_dual_version_mismatch_is_preflighted_without_persistent_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dual.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "UPDATE schema_meta SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION - 1),),
                )
                conn.commit()
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                conn.execute('PRAGMA journal_mode=DELETE')
            before = path.read_bytes()
            with self.assertRaises(SchemaVersionMismatch):
                SQLiteStore(path).initialize()
            self.assertEqual(path.read_bytes(), before)

    def test_live_wal_future_version_is_read_coherently_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'future-wal.sqlite'
            writer = SQLiteStore(path)
            writer.initialize()
            warm_reader = open_store(path, readonly=True)
            warm_reader.close()
            conn = writer.connect()
            before_main = path.read_bytes()
            conn.execute(f'PRAGMA user_version={SCHEMA_VERSION + 1}')
            conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION + 1),),
            )
            conn.commit()
            self.assertEqual(conn.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
            with self.assertRaises(SchemaVersionTooNew):
                open_store(path, readonly=True)
            with self.assertRaises(SchemaVersionTooNew):
                SQLiteStore(path).initialize()
            self.assertEqual(path.read_bytes(), before_main)
            self.assertEqual(conn.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
            writer.close()

    def test_readonly_manifest_validation_cache_tracks_schema_not_data_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cached-validation.sqlite'
            writer = SQLiteStore(path)
            writer.initialize()
            migrations_module._clear_schema_validation_cache_for_tests()
            try:
                with mock.patch.object(
                    migrations_module,
                    'validate_schema',
                    wraps=migrations_module.validate_schema,
                ) as full_validation:
                    first = open_store(path, readonly=True)
                    first.close()
                    self.assertEqual(full_validation.call_count, 1)

                    with writer.connect() as conn:
                        conn.execute('INSERT INTO accounts VALUES(?,?,?)', ('acct-cache', 'work', 'Cache'))
                        conn.commit()

                    second = open_store(path, readonly=True)
                    self.assertEqual(second.counts()['accounts'], 1)
                    second.close()
                    self.assertEqual(full_validation.call_count, 1)

                    with writer.connect() as conn:
                        conn.execute('DROP TRIGGER message_fts_ai')
                        conn.commit()

                    with self.assertRaises(SchemaMigrationRequired):
                        open_store(path, readonly=True)
                    self.assertEqual(full_validation.call_count, 2)
            finally:
                writer.close()
                migrations_module._clear_schema_validation_cache_for_tests()

    def test_readonly_manifest_validation_cache_rechecks_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'active.sqlite'
            replacement = root / 'replacement.sqlite'
            first_writer = SQLiteStore(path)
            first_writer.initialize()
            first_writer.close()
            second_writer = SQLiteStore(replacement)
            second_writer.initialize()
            second_writer.close()
            for database in (path, replacement):
                with sqlite3.connect(database) as conn:
                    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                    conn.execute('PRAGMA journal_mode=DELETE')

            migrations_module._clear_schema_validation_cache_for_tests()
            try:
                with mock.patch.object(
                    migrations_module,
                    'validate_schema',
                    wraps=migrations_module.validate_schema,
                ) as full_validation:
                    reader = open_store(path, readonly=True)
                    reader.close()
                    old_inode = path.stat().st_ino
                    replacement.replace(path)
                    self.assertNotEqual(path.stat().st_ino, old_inode)

                    replacement_reader = open_store(path, readonly=True)
                    replacement_reader.close()
                    self.assertEqual(full_validation.call_count, 2)
            finally:
                migrations_module._clear_schema_validation_cache_for_tests()

    def test_failed_manifest_validation_is_never_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.sqlite'
            writer = SQLiteStore(path)
            writer.initialize()
            with writer.connect() as conn:
                conn.execute('DROP TRIGGER message_fts_ai')
                conn.commit()
            writer.close()

            migrations_module._clear_schema_validation_cache_for_tests()
            try:
                with mock.patch.object(
                    migrations_module,
                    'validate_schema',
                    wraps=migrations_module.validate_schema,
                ) as full_validation:
                    for _ in range(2):
                        with self.assertRaises(SchemaMigrationRequired):
                            open_store(path, readonly=True)
                    self.assertEqual(full_validation.call_count, 2)
            finally:
                migrations_module._clear_schema_validation_cache_for_tests()

    def test_readonly_cache_never_binds_an_old_connection_to_a_replaced_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'active.sqlite'
            replacement = root / 'replacement.sqlite'
            writer = SQLiteStore(path)
            writer.initialize()
            writer.close()
            with sqlite3.connect(path) as conn:
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                conn.execute('PRAGMA journal_mode=DELETE')
                schema_cookie = int(conn.execute('PRAGMA schema_version').fetchone()[0])
            replacement.write_bytes(path.read_bytes())
            with sqlite3.connect(replacement) as conn:
                conn.execute('PRAGMA writable_schema=ON')
                conn.execute(
                    "DELETE FROM sqlite_master WHERE type='trigger' AND name='message_fts_ai'"
                )
                conn.execute('PRAGMA writable_schema=OFF')
                conn.commit()
                self.assertEqual(int(conn.execute('PRAGMA schema_version').fetchone()[0]), schema_cookie)
            with self.assertRaises(SchemaMigrationRequired):
                open_store(replacement, readonly=True)

            migrations_module._clear_schema_validation_cache_for_tests()
            stale_reader = SQLiteStore(path, readonly=True)
            original_preflight = stale_reader._preflight_readonly_connection
            swapped = False

            def replace_after_connection_preflight(conn: sqlite3.Connection) -> None:
                nonlocal swapped
                original_preflight(conn)
                replacement.replace(path)
                swapped = True

            try:
                with mock.patch.object(
                    stale_reader,
                    '_preflight_readonly_connection',
                    side_effect=replace_after_connection_preflight,
                ):
                    with self.assertRaises(SchemaPreflightUnavailable):
                        stale_reader.initialize()
                self.assertTrue(swapped)
                for _ in range(2):
                    with self.assertRaises(SchemaMigrationRequired):
                        open_store(path, readonly=True)
            finally:
                stale_reader.close()
                migrations_module._clear_schema_validation_cache_for_tests()

    def test_closed_store_repreflights_replaced_future_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'store.sqlite'
            replacement = root / 'replacement.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(replacement) as conn:
                conn.execute(f'PRAGMA user_version={SCHEMA_VERSION + 1}')
                conn.commit()
            replacement.replace(path)
            before = path.read_bytes()

            with self.assertRaises(SchemaVersionTooNew):
                store.initialize()

            self.assertEqual(path.read_bytes(), before)

    def test_schema_error_payload_prescribes_non_destructive_actions(self):
        future = schema_migration_required_payload(
            SchemaVersionTooNew(SCHEMA_VERSION + 1, SCHEMA_VERSION)
        )
        mismatch = schema_migration_required_payload(
            SchemaVersionMismatch(SCHEMA_VERSION, SCHEMA_VERSION)
        )
        unavailable = schema_migration_required_payload(ReadOnlyStoreError())

        self.assertEqual(future['error']['code'], 'schema_migration_required')
        self.assertEqual(future['error']['reason_code'], 'schema_version_too_new')
        self.assertEqual(future['error']['action'], 'upgrade_runtime')
        self.assertEqual(mismatch['error']['action'], 'repair_schema_versions')
        self.assertEqual(unavailable['error']['action'], 'checkpoint_or_repair')

    def test_readonly_old_schema_is_typed_and_byte_for_byte_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite'
            self._legacy(path, 1, with_row=True)
            before = (path.stat().st_mtime_ns, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            siblings = sorted(item.name for item in path.parent.iterdir())
            with self.assertRaises(SchemaMigrationRequired) as error:
                open_store(path, readonly=True)
            self.assertEqual(error.exception.code, 'schema_migration_required')
            after = (path.stat().st_mtime_ns, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(after, before)
            self.assertEqual(sorted(item.name for item in path.parent.iterdir()), siblings)

    def test_cached_search_on_old_schema_does_not_create_wal_or_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            path = vault / 'index' / 'trove.sqlite'
            path.parent.mkdir(parents=True)
            self._legacy(path, 1, with_row=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            before = (path.stat().st_mtime_ns, path.read_bytes())
            siblings = sorted(item.name for item in path.parent.iterdir())
            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)
            with self.assertRaises(SchemaMigrationRequired):
                cache.search(SearchRequest('migration retained token', limit=2))
            cache.close()
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), before)
            self.assertEqual(sorted(item.name for item in path.parent.iterdir()), siblings)

    def test_readonly_open_reads_active_wal_coherently_without_application_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'store.sqlite'
            writer = SQLiteStore(path)
            writer.initialize()
            with writer.connect() as conn:
                conn.execute('INSERT INTO accounts VALUES(?,?,?)', ('acct-a', 'work', 'Work'))
                conn.commit()
            reader = open_store(path, readonly=True)
            statements: list[str] = []
            reader.connect().set_trace_callback(statements.append)
            self.assertEqual(reader.counts()['accounts'], 1)
            self.assertEqual(reader.connect().total_changes, 0)
            self.assertEqual(reader.connect().execute('PRAGMA journal_mode').fetchone()[0], 'wal')
            self.assertEqual(writer.connect().execute('PRAGMA journal_mode').fetchone()[0], 'wal')
            reader.close()
            forbidden = ('CREATE ', 'ALTER ', 'DROP ', 'INSERT ', 'UPDATE ', 'DELETE ', 'COMMIT')
            self.assertFalse(any(statement.lstrip().upper().startswith(forbidden) for statement in statements))
            writer.close()

    def test_runtime_search_uses_readonly_connection_with_no_schema_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            cfg = VaultConfig.resolve(str(vault), env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()
            with store.connect() as conn:
                conn.execute('INSERT INTO accounts VALUES(?,?,?)', ('acct-a', 'work', 'Work'))
                conn.execute('INSERT INTO conversations VALUES(?,?,?,?,?)', ('conv-a', 'acct-a', 'Fixture', 'private', 1))
                conn.execute(
                    """INSERT INTO messages(
                        citation,account_id,account_label,conversation_id,conversation_title,
                        conversation_type,sender_id,sender_name,timestamp,content,content_kind,
                        shard_id,local_id,sent_by_me,source_type,direction
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    ('trove://fixture/1', 'acct-a', 'work', 'conv-a', 'Fixture', 'private',
                     'u1', 'Fixture', '2026-01-01T00:00:00Z', 'readonly search token', 'text',
                     's1', 1, 0, 'message', 'incoming'),
                )
                conn.commit()
            self.assertEqual(store.connect().execute('PRAGMA journal_mode').fetchone()[0], 'wal')

            engine = build_search_engine(cfg)
            statements: list[str] = []
            engine.store.connect().set_trace_callback(statements.append)
            self.assertTrue(engine.search(SearchRequest('readonly search token', limit=2)).results)
            SQLiteVectorStore(engine.store).search(
                'readonly search token',
                limit=2,
                provider=FakeEmbeddingProvider(),
            )
            with self.assertRaisesRegex(sqlite3.OperationalError, 'read-only'):
                engine.store.connect().commit()
            self.assertEqual(engine.store.connect().total_changes, 0)
            self.assertEqual(engine.store.connect().execute('PRAGMA journal_mode').fetchone()[0], 'wal')
            engine.store.close()
            store.close()
            forbidden = ('CREATE ', 'ALTER ', 'DROP ', 'INSERT ', 'UPDATE ', 'DELETE ', 'COMMIT', 'PRAGMA user_version =')
            self.assertFalse(any(statement.lstrip().upper().startswith(tuple(item.upper() for item in forbidden)) for statement in statements))

    def test_feature_modules_do_not_own_persistent_ddl(self):
        root = Path(__file__).resolve().parents[1] / 'trove_core'
        for relative in ('sync.py', 'vector/sqlite_vector_store.py'):
            text = (root / relative).read_text(encoding='utf-8')
            self.assertIsNone(
                re.search(r'\b(?:CREATE\s+(?:TABLE|INDEX|VIRTUAL)|ALTER\s+TABLE|DROP\s+(?:TABLE|INDEX))\b', text, re.I),
                relative,
            )

    def test_duplicate_index_is_repaired_and_query_plans_use_both_access_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite'
            self._legacy(path, 11, wrong_parent_index=True)
            SQLiteStore(path).initialize()
            with sqlite3.connect(path) as conn:
                parent_cols = tuple(row[2] for row in conn.execute('PRAGMA index_info(idx_evidence_chunks_parent)'))
                source_cols = tuple(row[2] for row in conn.execute('PRAGMA index_info(idx_evidence_chunks_source_parent)'))
                self.assertEqual(parent_cols, ('parent_citation', 'chunk_index'))
                self.assertEqual(source_cols, ('source_type', 'parent_citation'))
                parent_plan = ' '.join(str(row) for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT * FROM evidence_chunks WHERE parent_citation=? ORDER BY chunk_index',
                    ('trove://fixture',),
                ))
                source_plan = ' '.join(str(row) for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT * FROM evidence_chunks WHERE source_type=? AND parent_citation=?',
                    ('message', 'trove://fixture'),
                ))
            self.assertIn('idx_evidence_chunks_parent', parent_plan)
            self.assertIn('idx_evidence_chunks_source_parent', source_plan)

    def test_v27_resequences_duplicate_profile_versions_before_unique_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v26-profile-duplicates.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute('DROP INDEX idx_profile_snapshots_entity_version_unique')
                conn.execute(
                    """INSERT INTO entities(
                           entity_id,entity_type,display_name,created_at,updated_at)
                       VALUES('customer-duplicate-history','Customer','Fixture','now','now')"""
                )
                conn.executemany(
                    """INSERT INTO profile_snapshots(
                           profile_id,entity_id,version,projection_json,created_at)
                       VALUES(?,?,?,?,?)""",
                    [
                        ('profile-old', 'customer-duplicate-history', 1, '{}', '2026-01-01T00:00:00Z'),
                        ('profile-new', 'customer-duplicate-history', 1, '{}', '2026-01-02T00:00:00Z'),
                    ],
                )
                conn.execute("UPDATE schema_meta SET value='26' WHERE key='schema_version'")
                conn.execute('PRAGMA user_version=26')
                conn.commit()

            migrated = SQLiteStore(path)
            migrated.initialize()
            with migrated.connect() as conn:
                self.assertEqual(
                    [tuple(row) for row in conn.execute(
                        """SELECT profile_id,version FROM profile_snapshots
                             WHERE entity_id='customer-duplicate-history' ORDER BY version"""
                    )],
                    [
                        ('profile-old', 1),
                        ('profile-new', 2),
                    ],
                )
                index = conn.execute(
                    """SELECT [unique] FROM pragma_index_list('profile_snapshots')
                         WHERE name='idx_profile_snapshots_entity_version_unique'"""
                ).fetchone()
                self.assertEqual(index[0], 1)

    def test_v27_rebuilds_v26_profile_automation_foreign_keys_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v26-profile-automation.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute(
                    """INSERT INTO entities(
                           entity_id,entity_type,display_name,created_at,updated_at)
                       VALUES('customer-automation','Customer','Fixture','now','now')"""
                )
                conn.execute(
                    """INSERT INTO profile_snapshots(
                           profile_id,entity_id,version,projection_json,created_at)
                       VALUES('profile-automation','customer-automation',1,'{}','now')"""
                )
                conn.execute('DROP TABLE profile_refresh_queue')
                conn.execute('DROP TABLE profile_automation_subscriptions')
                conn.executescript(
                    """CREATE TABLE profile_automation_subscriptions (
                           entity_id TEXT PRIMARY KEY,
                           selector TEXT NOT NULL,
                           enabled INTEGER NOT NULL DEFAULT 1,
                           debounce_seconds INTEGER NOT NULL DEFAULT 180,
                           consent_scope TEXT NOT NULL DEFAULT 'explicit-profile-auto-maintenance-v1',
                           last_profile_id TEXT,
                           last_refresh_at TEXT,
                           last_error_code TEXT,
                           created_at TEXT NOT NULL,
                           updated_at TEXT NOT NULL,
                           FOREIGN KEY(entity_id) REFERENCES entities(entity_id),
                           FOREIGN KEY(last_profile_id) REFERENCES profile_snapshots(profile_id),
                           CHECK(enabled IN (0,1)),
                           CHECK(debounce_seconds BETWEEN 0 AND 3600)
                       );
                       CREATE TABLE profile_refresh_queue (
                           entity_id TEXT PRIMARY KEY,
                           generation INTEGER NOT NULL DEFAULT 1,
                           state TEXT NOT NULL DEFAULT 'pending',
                           reason TEXT NOT NULL,
                           available_at TEXT NOT NULL,
                           claimed_at TEXT,
                           attempt_count INTEGER NOT NULL DEFAULT 0,
                           last_error_code TEXT,
                           created_at TEXT NOT NULL,
                           updated_at TEXT NOT NULL,
                           FOREIGN KEY(entity_id) REFERENCES profile_automation_subscriptions(entity_id) ON DELETE CASCADE,
                           CHECK(state IN ('pending','processing','failed'))
                       );"""
                )
                conn.execute(
                    """INSERT INTO profile_automation_subscriptions(
                           entity_id,selector,enabled,debounce_seconds,consent_scope,
                           last_profile_id,last_refresh_at,last_error_code,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'customer-automation', 'Fixture', 1, 240,
                        'explicit-profile-auto-maintenance-v1', 'profile-automation',
                        'refreshed', 'fixture-error', 'created', 'updated',
                    ),
                )
                conn.execute(
                    """INSERT INTO profile_refresh_queue(
                           entity_id,generation,state,reason,available_at,claimed_at,
                           attempt_count,last_error_code,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'customer-automation', 3, 'failed', 'fixture', 'available',
                        'claimed', 2, 'fixture-error', 'created', 'updated',
                    ),
                )
                conn.execute("UPDATE schema_meta SET value='26' WHERE key='schema_version'")
                conn.execute('PRAGMA user_version=26')
                conn.commit()

            migrated = SQLiteStore(path)
            migrated.initialize()
            with migrated.connect() as conn:
                subscription_fks = {
                    (row['table'], row['from'], row['to'], row['on_update'], row['on_delete'])
                    for row in conn.execute('PRAGMA foreign_key_list(profile_automation_subscriptions)')
                }
                queue_fks = {
                    (row['table'], row['from'], row['to'], row['on_update'], row['on_delete'])
                    for row in conn.execute('PRAGMA foreign_key_list(profile_refresh_queue)')
                }
                subscription = tuple(conn.execute(
                    """SELECT entity_id,selector,enabled,debounce_seconds,consent_scope,
                              last_profile_id,last_refresh_at,last_error_code,created_at,updated_at
                         FROM profile_automation_subscriptions"""
                ).fetchone())
                queued = tuple(conn.execute(
                    """SELECT entity_id,generation,state,reason,available_at,claimed_at,
                              attempt_count,last_error_code,created_at,updated_at
                         FROM profile_refresh_queue"""
                ).fetchone())

            self.assertEqual(subscription_fks, {
                ('entities', 'entity_id', 'entity_id', 'CASCADE', 'CASCADE'),
                ('profile_snapshots', 'last_profile_id', 'profile_id', 'CASCADE', 'SET NULL'),
            })
            self.assertEqual(queue_fks, {
                ('profile_automation_subscriptions', 'entity_id', 'entity_id', 'CASCADE', 'CASCADE'),
            })
            self.assertEqual(subscription, (
                'customer-automation', 'Fixture', 1, 240,
                'explicit-profile-auto-maintenance-v1', 'profile-automation',
                'refreshed', 'fixture-error', 'created', 'updated',
            ))
            self.assertEqual(queued, (
                'customer-automation', 3, 'failed', 'fixture', 'available',
                'claimed', 2, 'fixture-error', 'created', 'updated',
            ))

    def test_v12_upgrades_to_v13_scoped_search_indexes_and_explain_uses_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v12.sqlite'
            self._legacy(path, 12, with_row=True)
            store = SQLiteStore(path)
            store.initialize()
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            with sqlite3.connect(path) as conn:
                conversation_plan = ' '.join(str(row) for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT * FROM conversations WHERE conversation_id=?',
                    ('conv-a',),
                ))
                message_plan = ' '.join(str(row) for row in conn.execute(
                    'EXPLAIN QUERY PLAN SELECT * FROM messages WHERE conversation_id=? ORDER BY timestamp DESC LIMIT ?',
                    ('conv-a', 10),
                ))
                chunk_plan = ' '.join(str(row) for row in conn.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM evidence_chunks WHERE source_id=? AND status='active' ORDER BY timestamp DESC LIMIT ?",
                    ('conv-a', 10),
                ))
            self.assertIn('idx_conversations_id_account', conversation_plan)
            self.assertIn('idx_messages_conversation_time', message_plan)
            self.assertIn('idx_evidence_chunks_source_id_status_time', chunk_plan)

    def test_latest_schema_repairs_drifted_named_index_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'drifted-index.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute('DROP INDEX idx_messages_timestamp')
                conn.execute('CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC, account_id)')
                conn.commit()

            repair = SQLiteStore(path)
            repair.initialize()
            repair.close()

            with sqlite3.connect(path) as conn:
                columns = tuple(
                    str(row[2]) for row in conn.execute('PRAGMA index_info(idx_messages_timestamp)')
                )
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_messages_timestamp'"
                ).fetchone()[0]
            self.assertEqual(columns, ('timestamp',))
            self.assertNotIn('DESC', str(sql).upper())

    def test_fts_virtual_table_sql_is_manifested_and_wrong_tokenizer_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong-fts.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                for name in FTS_TRIGGER_NAMES:
                    conn.execute(f'DROP TRIGGER IF EXISTS "{name}"')
                conn.execute('DROP TABLE chunk_fts')
                conn.execute('DROP TABLE message_fts')
                conn.execute(
                    """CREATE VIRTUAL TABLE message_fts USING fts5(
                        citation UNINDEXED,content,sender_name,conversation_title,
                        tokenize='unicode61',content='messages',content_rowid='id'
                    )"""
                )
                conn.execute(
                    """CREATE VIRTUAL TABLE chunk_fts USING fts5(
                        chunk_citation UNINDEXED,content,title,actor,
                        tokenize='unicode61',content='evidence_chunks',content_rowid='rowid'
                    )"""
                )
                for sql in TRIGRAM_FTS_SCHEMA:
                    if 'CREATE TRIGGER' in sql:
                        conn.execute(sql)
                conn.commit()
            with self.assertRaises(SchemaMigrationRequired) as mismatch:
                open_store(path, readonly=True)
            self.assertIn('virtual-table-definition:message_fts', mismatch.exception.missing_objects)
            repair = SQLiteStore(path)
            repair.initialize()
            repair.close()
            reader = open_store(path, readonly=True)
            with reader.connect() as conn:
                sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='message_fts'"
                ).fetchone()[0]
            reader.close()
            self.assertIn("tokenize='trigram'", sql)

    def test_force_fts_rebuild_is_transactional_and_preserves_schema_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fts-rebuild.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            with store.connect() as conn:
                conn.execute('INSERT INTO accounts VALUES(?,?,?)', ('acct-a', 'work', 'Work'))
                conn.execute('INSERT INTO conversations VALUES(?,?,?,?,?)', ('conv-a', 'acct-a', 'Fixture', 'private', 1))
                conn.execute(
                    """INSERT INTO messages(
                        citation,account_id,account_label,conversation_id,conversation_title,
                        conversation_type,sender_id,sender_name,timestamp,content,content_kind,
                        shard_id,local_id,sent_by_me,source_type,direction
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    ('trove://fixture/fts', 'acct-a', 'work', 'conv-a', 'Fixture', 'private',
                     'u1', 'Fixture', '2026-01-01T00:00:00Z', 'atomic fts token', 'text',
                     's1', 1, 0, 'message', 'incoming'),
                )
                conn.commit()
            before_version = store.schema_version()

            with self.assertRaisesRegex(RuntimeError, 'after_fts_drop'):
                store.rebuild_fts(
                    fault_injector=lambda stage: (_ for _ in ()).throw(RuntimeError(stage))
                    if stage == 'after_fts_drop' else None
                )
            self.assertEqual(store.schema_version(), before_version)
            self.assertTrue(store.exact_search('atomic fts token'))
            with store.connect() as conn:
                from trove_core.store.migrations import validate_schema
                validate_schema(conn)

            report = store.rebuild_fts()
            self.assertEqual(report['message_rows'], 1)
            self.assertEqual(store.schema_version(), before_version)

    def test_table_sql_manifest_rejects_missing_autoincrement_and_check(self):
        with self.subTest(feature='autoincrement'), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'no-autoincrement.sqlite'
            with sqlite3.connect(path) as conn:
                conn.executescript(LEGACY_BASE.replace(' AUTOINCREMENT', ''))
                conn.execute('CREATE TABLE schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
                conn.execute('INSERT INTO schema_meta VALUES(?,?)', ('schema_version', '11'))
                conn.execute('PRAGMA user_version=11')
                conn.commit()
            with self.assertRaises(SchemaMigrationRequired) as missing_auto:
                SQLiteStore(path).initialize()
            self.assertIn('table-autoincrement:messages', missing_auto.exception.missing_objects)
            with sqlite3.connect(path) as conn:
                self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0], 11)

        with self.subTest(feature='check'), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'no-check.sqlite'
            store = SQLiteStore(path)
            store.initialize()
            store.close()
            with sqlite3.connect(path) as conn:
                conn.execute('DROP TABLE observations')
                conn.execute(
                    """CREATE TABLE observations (
                        observation_id TEXT PRIMARY KEY, entity_id TEXT NOT NULL,
                        observation_type TEXT NOT NULL, value_json TEXT NOT NULL,
                        status TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0,
                        citation TEXT NOT NULL, source_type TEXT NOT NULL, valid_from TEXT,
                        supersedes_observation_id TEXT, created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL, FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
                    )"""
                )
                conn.execute(
                    'CREATE INDEX idx_observations_entity_status '
                    'ON observations(entity_id,status,observation_type)'
                )
                conn.execute('CREATE INDEX idx_observations_citation ON observations(citation)')
                conn.commit()
            with self.assertRaises(SchemaMigrationRequired) as missing_check:
                open_store(path, readonly=True)
            self.assertIn('table-check-definition:observations', missing_check.exception.missing_objects)

    def test_foreign_key_report_contains_only_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'store.sqlite')
            store.initialize()
            with store.connect() as conn:
                report = foreign_key_compatibility_report(conn)
            self.assertFalse(report['foreign_keys_enabled'])
            self.assertTrue(report['counts_only'])
            self.assertIn('orphan_count', report)
            self.assertNotIn('rows', report)
            self.assertNotIn('citations', report)


if __name__ == '__main__':
    unittest.main()
