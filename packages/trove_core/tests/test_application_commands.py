from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

from trove_core.application.commands import (
    AuxiliaryImportCommand,
    FullImportCommand,
    MaintainCommand,
    SyncCommand,
    TroveCommands,
    VectorCommand,
)
from trove_core.store.change_journal import read_dirty_citations
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.wechat.import_job import run_import_job


class _Provider:
    name = 'fixture-local'
    dimensions = 3
    egress_kind = None


class ApplicationCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / 'vault'
        self.commands = TroveCommands(self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_none_config_preserves_default_vault_discovery(self) -> None:
        resolved = VaultConfig.resolve(str(self.root), env={})
        with mock.patch(
            'trove_core.application.commands.VaultConfig.resolve', return_value=resolved,
        ) as resolve:
            commands = TroveCommands(None)
        resolve.assert_called_once_with(None)
        self.assertEqual(commands.config, resolved)

    def test_full_import_passes_the_exact_approval_grant_to_sensitive_leaf(self) -> None:
        grant = object()
        command = FullImportCommand(('one', 'two'), reset_index_cache=True, limit_per_sqlite=7)
        with mock.patch(
            'trove_core.application.sensitive_commands.execute_full_import',
            return_value={'ok': True},
        ) as execute:
            self.assertEqual(self.commands.full_import(command, approval_grant=grant), {'ok': True})  # type: ignore[arg-type]
        self.assertIs(execute.call_args.kwargs['approval_grant'], grant)
        self.assertEqual(execute.call_args.kwargs['limit_per_sqlite'], 7)
        self.assertTrue(execute.call_args.kwargs['reset_index_cache'])

    def test_reset_and_scope_pass_the_exact_approval_grant(self) -> None:
        grant = object()
        with mock.patch(
            'trove_core.application.sensitive_commands.execute_reset_index_cache',
            return_value={'ok': True},
        ) as reset:
            self.commands.reset_index_cache(approval_grant=grant)  # type: ignore[arg-type]
        with mock.patch(
            'trove_core.application.sensitive_commands.execute_scope_rebuild',
            return_value={'ok': True},
        ) as scope:
            self.commands.scope_rebuild(approval_grant=grant)  # type: ignore[arg-type]
        self.assertIs(reset.call_args.kwargs['approval_grant'], grant)
        self.assertIs(scope.call_args.kwargs['approval_grant'], grant)

    def test_sync_and_maintain_construct_one_leaf_options_object(self) -> None:
        with mock.patch('trove_core.sync.run_sync', return_value={'ok': True}) as run_sync:
            self.commands.sync(SyncCommand(
                full=True,
                since='2026-01-02T03:04:05Z',
                snapshot_dir='snapshot',
                limit_per_shard=9,
                vector_backend='sqlite',
            ))
        options = run_sync.call_args.kwargs['options']
        self.assertTrue(options.full)
        self.assertEqual(options.limit_per_shard, 9)
        self.assertEqual(options.vector_backend, 'sqlite')
        self.assertIsNotNone(options.since)

        with mock.patch('trove_core.maintain.run_maintain', return_value={'ok': True}) as run_maintain:
            self.commands.maintain(MaintainCommand(
                auto_rebuild=True,
                backup_retention=4,
                media_image_budget=2,
            ))
        options = run_maintain.call_args.kwargs['options']
        self.assertTrue(options.auto_rebuild)
        self.assertEqual(options.backup_retention, 4)
        self.assertEqual(options.media_image_budget, 2)

        defaults = MaintainCommand()
        self.assertEqual(defaults.media_voice_budget, 0)
        self.assertFalse(defaults.full_scan)

    def test_vector_selection_uses_provider_factory_and_preserves_grant(self) -> None:
        grant = object()
        with mock.patch('trove_core.runtime.configured_embedding_provider', return_value=_Provider()) as factory:
            prepared = self.commands.prepare_vector(VectorCommand(
                action='rebuild',
                model_path='fixture-model',
                backend='sqlite',
                batch_size=5,
                max_messages=11,
            ))
            factory.assert_called_once_with(
                'fixture-model',
                strict=True,
                vault_root=self.commands.config.root,
                prefer_cloud=False,
            )
        self.assertTrue(prepared.requires_approval)
        self.assertEqual(prepared.approval_action, 'vector_rebuild')

        with mock.patch(
            'trove_core.application.sensitive_commands.execute_vector_mutation',
            return_value={'ok': True},
        ) as execute:
            self.commands.vector(prepared, approval_grant=grant)  # type: ignore[arg-type]
        self.assertIs(execute.call_args.kwargs['approval_grant'], grant)
        self.assertEqual(execute.call_args.kwargs['action'], 'vector_rebuild')

    def test_local_non_destructive_vector_index_needs_no_approval(self) -> None:
        with mock.patch('trove_core.runtime.configured_embedding_provider', return_value=_Provider()):
            prepared = self.commands.prepare_vector(VectorCommand(action='index', backend='zvec'))
        self.assertFalse(prepared.requires_approval)
        with mock.patch('trove_core.runtime.index_vectors', return_value={'ok': True}) as index:
            self.commands.vector(prepared)
        self.assertIsNone(index.call_args.kwargs['approval_grant'])
        self.assertIsNone(index.call_args.kwargs['approval_payload'])

    def test_command_dtos_reject_adapter_coercion(self) -> None:
        for build in (
            lambda: FullImportCommand((), reset_index_cache=1),
            lambda: SyncCommand(full='yes'),
            lambda: MaintainCommand(vacuum=1),
            lambda: VectorCommand(batch_size=0),
            lambda: AuxiliaryImportCommand('contacts', 'x', '', None),
        ):
            with self.subTest(build=build), self.assertRaises((TypeError, ValueError)):
                build()

    def test_auxiliary_imports_project_only_exact_changed_citations(self) -> None:
        source_root = Path(self.tempdir.name) / 'sources'
        source_root.mkdir()
        contact_db = source_root / 'people.sqlite'
        with sqlite3.connect(contact_db) as conn:
            conn.execute(
                'CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, signature TEXT, big_head_url TEXT)'
            )
            conn.execute(
                'INSERT INTO contact VALUES(?,?,?,?,?,?)',
                ('wxid-delta', 'Delta Contact', '', '', 'one changed contact', ''),
            )
        moment_db = source_root / 'timeline.sqlite'
        with sqlite3.connect(moment_db) as conn:
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute(
                'INSERT INTO SnsTimeLine VALUES(?,?,?,?)',
                (
                    'feed-delta',
                    'wxid-delta',
                    '<TimelineObject><id>moment-delta</id><username>wxid-delta</username>'
                    '<createTime>1760000000</createTime><contentDesc>one changed moment</contentDesc></TimelineObject>',
                    b'',
                ),
            )
        favorite_db = source_root / 'saved.sqlite'
        with sqlite3.connect(favorite_db) as conn:
            conn.execute('CREATE TABLE favorite_item(fav_id TEXT, update_time TEXT, title TEXT, content TEXT)')
            conn.execute(
                'INSERT INTO favorite_item VALUES(?,?,?,?)',
                ('favorite-delta', '2026-01-02', 'Delta Favorite', 'one changed favorite'),
            )

        bounded_calls: list[tuple[str, tuple[str, ...]]] = []
        original = SQLiteStore.rebuild_evidence_chunks_for_source_citations

        def bounded(store, source_type, citations, **kwargs):
            bounded_calls.append((source_type, tuple(citations)))
            return original(store, source_type, citations, **kwargs)

        with (
            mock.patch.object(
                SQLiteStore,
                'rebuild_evidence_chunks',
                side_effect=AssertionError('full corpus rebuild forbidden'),
            ),
            mock.patch.object(
                SQLiteStore,
                'rebuild_evidence_chunks_for_source_types',
                side_effect=AssertionError('family-wide rebuild forbidden'),
            ),
            mock.patch.object(
                SQLiteStore,
                'rebuild_evidence_chunks_for_source_citations',
                new=bounded,
            ),
        ):
            for kind, path in (
                ('contacts', contact_db),
                ('moments', moment_db),
                ('favorites', favorite_db),
            ):
                result = self.commands.auxiliary_import(
                    AuxiliaryImportCommand(kind, path, 'acct-delta'),
                )
                self.assertEqual(result[f'imported_{kind}'], 1)

        self.assertEqual([family for family, _ in bounded_calls], ['contact', 'moment', 'favorite'])
        self.assertTrue(all(len(citations) == 1 for _, citations in bounded_calls))
        store = SQLiteStore(self.commands.config.paths.sqlite_path)
        try:
            dirty = read_dirty_citations(store)
            self.assertEqual(len(dirty), 3)
            self.assertEqual(set(dirty), {citation for _, citations in bounded_calls for citation in citations})
        finally:
            store.close_all()

    def test_auxiliary_source_load_runs_outside_writer(self) -> None:
        source_root = Path(self.tempdir.name) / 'outside-writer-source'
        source_root.mkdir()
        contact_db = source_root / 'contacts.sqlite'
        with sqlite3.connect(contact_db) as conn:
            conn.execute(
                'CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, signature TEXT)'
            )
            conn.execute(
                'INSERT INTO contact VALUES(?,?,?,?,?)',
                ('wxid-outside-writer', 'Outside Writer', '', '', 'prepared without Vault lock'),
            )

        from trove_core.wechat import auxiliary_import as auxiliary_module

        original_prepare = auxiliary_module.prepare_auxiliary_sources
        observed: list[str] = []

        def prepare_with_probe(*args, **kwargs):
            with VaultOperationCoordinator(self.commands.config).write(owner='probe-aux-prepare'):
                observed.append('prepare')
            return original_prepare(*args, **kwargs)

        with mock.patch.object(auxiliary_module, 'prepare_auxiliary_sources', prepare_with_probe):
            result = self.commands.auxiliary_import(
                AuxiliaryImportCommand('contacts', contact_db, 'acct-outside-writer'),
            )

        self.assertEqual(observed, ['prepare'])
        self.assertEqual(result['imported_contacts'], 1)

    def test_stale_auxiliary_prepare_retries_after_full_import_publication(self) -> None:
        source_root = Path(self.tempdir.name) / 'stale-aux-source'
        source_root.mkdir()
        contact_db = source_root / 'contacts.sqlite'
        with sqlite3.connect(contact_db) as conn:
            conn.execute(
                'CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, signature TEXT)'
            )
            conn.execute(
                'INSERT INTO contact VALUES(?,?,?,?,?)',
                ('wxid-stale', 'Stale Prepared Contact', '', '', 'must not publish after full import'),
            )

        from trove_core.wechat import auxiliary_import as auxiliary_module

        original_prepare = auxiliary_module.prepare_auxiliary_sources
        stale_ready = threading.Event()
        release_stale = threading.Event()
        results: dict[str, dict] = {}
        failures: list[BaseException] = []

        def ordered_prepare(*args, **kwargs):
            prepared = original_prepare(*args, **kwargs)
            if threading.current_thread().name == 'stale-aux':
                stale_ready.set()
                if not release_stale.wait(10):
                    raise RuntimeError('timed out waiting to release stale auxiliary import')
            return prepared

        def run_stale() -> None:
            try:
                results['stale'] = self.commands.auxiliary_import(
                    AuxiliaryImportCommand('contacts', contact_db, 'acct-stale'),
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        with mock.patch.object(auxiliary_module, 'prepare_auxiliary_sources', ordered_prepare):
            thread = threading.Thread(target=run_stale, name='stale-aux')
            thread.start()
            self.assertTrue(stale_ready.wait(10), 'auxiliary import did not finish preparation')
            published = run_import_job(self.root, [], reset_index=False)
            release_stale.set()
            thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(published.status, 'completed')
        self.assertEqual(results['stale']['status'], 'retry_required')
        self.assertEqual(results['stale']['errors'], ['sync_commit_generation_changed'])
        store = SQLiteStore(self.commands.config.paths.sqlite_path)
        with store.connect() as conn:
            stale_rows = int(conn.execute(
                "SELECT COUNT(*) FROM observations WHERE citation LIKE 'trove://wechat/acct-stale/contact/%'",
            ).fetchone()[0])
        self.assertEqual(stale_rows, 0)


if __name__ == '__main__':
    unittest.main()
