from __future__ import annotations

from contextlib import closing, contextmanager
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.sync import SyncOptions, run_sync
from trove_core.vault.locks import VaultOperationLocked
from trove_core.wechat.decrypt import DecryptConfig, build_decrypt_plan, run_decrypt_plan
from trove_core.wechat.decrypt.config import SelectedAccount, selected_accounts_from_strings
from trove_core.wechat.decrypt.manifest import (
    INTERNAL_ACCOUNT_IDENTITY_NAME,
    INTERNAL_GUARD_NAME,
    load_account_identity,
    load_snapshot_guard,
)
from trove_core.wechat.decrypt.runner import CopyPlaintextEngine, DecryptFileResult, MESSAGE_CREATE_TIME_INDEX_PREFIX
from trove_core.wechat.decrypt.status import known_keyed_account_refs
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter, msg_table_for


def make_account(root: Path, name: str, *, message_text: str = 'hello selected') -> Path:
    acct = root / name
    acct.mkdir(parents=True)
    with sqlite3.connect(acct / 'contact.db') as conn:
        conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT, extra_buffer BLOB)')
        conn.execute('INSERT INTO contact(username,remark,nick_name,alias,extra_buffer) VALUES (?,?,?,?,?)', ('wxid_friend', 'Friend', '', '', b''))
        conn.commit()
    table = msg_table_for('wxid_friend')
    with sqlite3.connect(acct / 'message_0.db') as conn:
        conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_friend', 1))
        conn.execute(f'''CREATE TABLE {table} (
            local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
            real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
            download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
            message_content TEXT, compress_content BLOB, packed_info_data BLOB,
            WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
        )''')
        conn.execute(f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)', (1, 1, 1710000000, message_text))
        conn.commit()
    with sqlite3.connect(acct / 'message_fts.db') as conn:
        conn.execute('CREATE TABLE sensitive_fts(content TEXT)')
        conn.commit()
    return acct


class WeChatDecryptPipelineTests(unittest.TestCase):
    def test_writer_preflight_rejects_before_snapshot_creation(self):
        import trove_core.wechat.decrypt.runner as runner_module

        class UnexpectedEngine(CopyPlaintextEngine):
            def decrypt(self, source, dest, *, key, file_family):
                raise AssertionError('decrypt must not start while the writer is unavailable')

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(
                    ['wxid_keep:com.tencent.xinWeChat__wxid_keep']
                ),
            )

            @contextmanager
            def reject_writer(*_args, **_kwargs):
                raise VaultOperationLocked()
                yield  # pragma: no cover

            with (
                patch.object(runner_module, 'coordinated_vault_mutation', reject_writer),
                self.assertRaises(VaultOperationLocked),
            ):
                run_decrypt_plan(
                    build_decrypt_plan(cfg),
                    engine=UnexpectedEngine(),
                )

            self.assertFalse(
                (
                    vault / 'sources' / 'wechat-integrated-decrypted' / 'runs'
                ).exists()
            )

    def test_engine_exception_discards_run_created_before_publication(self):
        class BoomEngine(CopyPlaintextEngine):
            def decrypt(self, source, dest, *, key, file_family):
                raise RuntimeError('boom during decrypt')

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            account = make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(
                    ['wxid_keep:com.tencent.xinWeChat__wxid_keep']
                ),
            )
            first = run_decrypt_plan(
                build_decrypt_plan(cfg),
                engine=CopyPlaintextEngine(),
            )
            base = vault / 'sources' / 'wechat-integrated-decrypted'
            first_current = (base / 'current').resolve(strict=True)
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {msg_table_for("wxid_friend")}'
                    '(local_id,real_sender_id,create_time,message_content) '
                    'VALUES (?,?,?,?)',
                    (2, 1, 1710000001, 'engine failure input'),
                )
                conn.commit()

            with self.assertRaisesRegex(RuntimeError, 'boom during decrypt'):
                run_decrypt_plan(
                    build_decrypt_plan(cfg),
                    engine=BoomEngine(),
                )

            self.assertEqual((base / 'current').resolve(strict=True), first_current)
            self.assertEqual(
                {
                    path.name
                    for path in (base / 'runs').iterdir()
                    if path.is_dir() and not path.is_symlink()
                },
                {first['run_ref']},
            )
            self.assertEqual(list(base.glob('.current-*.tmp')), [])

    def test_stage_exception_discards_run_and_partial_staged_current(self):
        import trove_core.wechat.decrypt.runner as runner_module

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            account = make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(
                    ['wxid_keep:com.tencent.xinWeChat__wxid_keep']
                ),
            )
            first = run_decrypt_plan(
                build_decrypt_plan(cfg),
                engine=CopyPlaintextEngine(),
            )
            base = vault / 'sources' / 'wechat-integrated-decrypted'
            first_current = (base / 'current').resolve(strict=True)
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {msg_table_for("wxid_friend")}'
                    '(local_id,real_sender_id,create_time,message_content) '
                    'VALUES (?,?,?,?)',
                    (2, 1, 1710000001, 'stage failure input'),
                )
                conn.commit()
            original_stage = runner_module._stage_current

            def stage_then_fail(*args, **kwargs):
                original_stage(*args, **kwargs)
                raise OSError('boom during stage')

            with (
                patch.object(runner_module, '_stage_current', stage_then_fail),
                self.assertRaisesRegex(OSError, 'boom during stage'),
            ):
                run_decrypt_plan(
                    build_decrypt_plan(cfg),
                    engine=CopyPlaintextEngine(),
                )

            self.assertEqual((base / 'current').resolve(strict=True), first_current)
            self.assertEqual(
                {
                    path.name
                    for path in (base / 'runs').iterdir()
                    if path.is_dir() and not path.is_symlink()
                },
                {first['run_ref']},
            )
            self.assertEqual(list(base.glob('.current-*.tmp')), [])

    def test_publication_failures_cleanup_only_unpublished_runs(self):
        import trove_core.wechat.decrypt.runner as runner_module

        phases = {
            'entry': (False, VaultOperationLocked('entry failure')),
            'body': (False, OSError('manifest failure')),
            'exit_after_switch': (True, RuntimeError('exit failure')),
            'publish_after_switch': (True, ValueError('post-switch failure')),
        }
        for phase, (switched, failure) in phases.items():
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                live = root / 'live'
                vault = root / 'vault'
                account = make_account(live, 'com.tencent.xinWeChat__wxid_keep')
                cfg = DecryptConfig(
                    live_root=live,
                    vault_root=vault,
                    selected_accounts=selected_accounts_from_strings(
                        ['wxid_keep:com.tencent.xinWeChat__wxid_keep']
                    ),
                )
                first = run_decrypt_plan(
                    build_decrypt_plan(cfg),
                    engine=CopyPlaintextEngine(),
                )
                base = vault / 'sources' / 'wechat-integrated-decrypted'
                runs = base / 'runs'
                first_run = runs / first['run_ref']
                first_current = (base / 'current').resolve(strict=True)
                with sqlite3.connect(account / 'message_0.db') as conn:
                    conn.execute(
                        f'INSERT INTO {msg_table_for("wxid_friend")}'
                        '(local_id,real_sender_id,create_time,message_content) '
                        'VALUES (?,?,?,?)',
                        (2, 1, 1710000001, f'unpublished {phase}'),
                    )
                    conn.commit()

                original_coordinated = runner_module.coordinated_vault_mutation
                original_publish = runner_module._publish_current
                lock_calls = 0

                @contextmanager
                def fail_at_publication(*args, **kwargs):
                    nonlocal lock_calls
                    lock_calls += 1
                    if lock_calls == 1:
                        with original_coordinated(*args, **kwargs) as session:
                            yield session
                        return
                    if phase == 'entry':
                        raise failure
                    with original_coordinated(*args, **kwargs) as session:
                        yield session
                        if phase == 'exit_after_switch':
                            raise failure

                def fail_manifest(*_args, **_kwargs):
                    raise failure

                def publish_then_fail(*args, **kwargs):
                    original_publish(*args, **kwargs)
                    raise failure

                manifest_patch = (
                    patch.object(runner_module, 'write_manifest', fail_manifest)
                    if phase == 'body'
                    else patch.object(runner_module, 'write_manifest', wraps=runner_module.write_manifest)
                )
                publish_patch = (
                    patch.object(runner_module, '_publish_current', publish_then_fail)
                    if phase == 'publish_after_switch'
                    else patch.object(runner_module, '_publish_current', wraps=runner_module._publish_current)
                )
                with (
                    patch.object(
                        runner_module,
                        'coordinated_vault_mutation',
                        fail_at_publication,
                    ),
                    manifest_patch,
                    publish_patch,
                    self.assertRaises(type(failure)) as raised,
                ):
                    run_decrypt_plan(
                        build_decrypt_plan(cfg),
                        engine=CopyPlaintextEngine(),
                    )

                self.assertIs(raised.exception, failure)
                remaining_runs = {
                    path.name
                    for path in runs.iterdir()
                    if path.is_dir() and not path.is_symlink()
                }
                current_after = (base / 'current').resolve(strict=True)
                self.assertTrue(first_run.is_dir())
                self.assertEqual(list(base.glob('.current-*.tmp')), [])
                if switched:
                    self.assertNotEqual(current_after, first_current)
                    self.assertIn(current_after.name, remaining_runs)
                    self.assertEqual(len(remaining_runs), 2)
                else:
                    self.assertEqual(current_after, first_current)
                    self.assertEqual(remaining_runs, {first['run_ref']})

    def test_unchanged_files_are_hardlinked_from_the_published_run(self):
        class CountingEngine(CopyPlaintextEngine):
            def __init__(self):
                self.calls: list[str] = []

            def decrypt(self, source, dest, *, key, file_family):
                self.calls.append(source.name)
                return super().decrypt(source, dest, key=key, file_family=file_family)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            account = make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(['wxid_keep:com.tencent.xinWeChat__wxid_keep']),
            )
            first_engine = CountingEngine()
            first = run_decrypt_plan(build_decrypt_plan(cfg), engine=first_engine)
            previous_contact = (
                vault / 'sources' / 'wechat-integrated-decrypted' / 'runs' /
                first['run_ref'] / 'com.tencent.xinWeChat__wxid_keep' / 'contact.db'
            )
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(
                    f'INSERT INTO {msg_table_for("wxid_friend")}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)',
                    (2, 1, 1710000001, 'incremental'),
                )
                conn.commit()

            second_engine = CountingEngine()
            second = run_decrypt_plan(build_decrypt_plan(cfg), engine=second_engine)
            current_contact = (
                vault / 'sources' / 'wechat-integrated-decrypted' / 'current' /
                'com.tencent.xinWeChat__wxid_keep' / 'contact.db'
            )

            self.assertTrue(first['ok'])
            self.assertTrue(second['ok'])
            self.assertEqual(set(first_engine.calls), {'contact.db', 'message_0.db'})
            self.assertEqual(second_engine.calls, ['message_0.db'])
            self.assertEqual(second['summary']['reused'], 1)
            self.assertEqual(second['summary']['copied_plaintext'], 1)
            self.assertEqual(previous_contact.stat().st_ino, current_contact.stat().st_ino)
            self.assertEqual(len(known_keyed_account_refs(vault)), 1)
            self.assertNotIn(str(live), json.dumps(second))

    def test_message_outputs_gain_create_time_delta_indexes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(['wxid_keep:com.tencent.xinWeChat__wxid_keep']),
            )
            report = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())

            self.assertTrue(report['ok'])
            snapshot_db = (
                vault / 'sources' / 'wechat-integrated-decrypted' / 'current' /
                'com.tencent.xinWeChat__wxid_keep' / 'message_0.db'
            )
            table = msg_table_for('wxid_friend')
            with closing(sqlite3.connect(f'file:{snapshot_db}?mode=ro', uri=True)) as conn:
                index_names = {
                    str(row[0])
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
                }
                rows = conn.execute(f'SELECT create_time FROM {table}').fetchall()
            self.assertIn(f'{MESSAGE_CREATE_TIME_INDEX_PREFIX}{table}', index_names)
            self.assertEqual([row[0] for row in rows], [1710000000])

    def test_reused_message_output_is_indexed_without_touching_the_published_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(['wxid_keep:com.tencent.xinWeChat__wxid_keep']),
            )
            first = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())
            self.assertTrue(first['ok'])
            runs = vault / 'sources' / 'wechat-integrated-decrypted' / 'runs'
            table = msg_table_for('wxid_friend')
            index_name = f'{MESSAGE_CREATE_TIME_INDEX_PREFIX}{table}'
            first_db = runs / first['run_ref'] / 'com.tencent.xinWeChat__wxid_keep' / 'message_0.db'

            def has_index(path: Path) -> bool:
                with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as conn:
                    return bool(conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                        (index_name,),
                    ).fetchone())

            self.assertTrue(has_index(first_db))
            # Simulate a published run from before delta indexes existed.
            with sqlite3.connect(first_db) as conn:
                conn.execute(f'DROP INDEX {index_name}')

            second = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())
            self.assertTrue(second['ok'])
            self.assertEqual(second['summary']['reused'], 2)
            second_db = runs / second['run_ref'] / 'com.tencent.xinWeChat__wxid_keep' / 'message_0.db'
            self.assertFalse(has_index(first_db))
            self.assertTrue(has_index(second_db))
            self.assertNotEqual(first_db.stat().st_ino, second_db.stat().st_ino)

            third = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())
            self.assertTrue(third['ok'])
            third_db = runs / third['run_ref'] / 'com.tencent.xinWeChat__wxid_keep' / 'message_0.db'
            self.assertTrue(has_index(third_db))
            self.assertEqual(second_db.stat().st_ino, third_db.stat().st_ino)

    def test_explicit_container_and_root_use_direct_layout_without_global_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            container = 'com.tencent.xinWeChat'
            root_name = 'wxid_selected_fixture'
            source = live / container / 'Data/Documents/xwechat_files'
            make_account(source, root_name)
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=(SelectedAccount(
                    account_id=root_name,
                    container_id=container,
                    root_name=root_name,
                    output_name='account-fixture',
                ),),
            )

            with patch(
                'trove_core.wechat.decrypt.preflight.discover_account_roots',
                side_effect=AssertionError('global discovery must not run'),
            ):
                plan = build_decrypt_plan(cfg)

            self.assertTrue(plan.ok)
            self.assertEqual(len(plan.files), 2)
            self.assertEqual(plan.skipped_accounts, ())

    def test_db_storage_scan_is_shallow_and_ignores_unreviewed_subtrees(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            container = 'com.tencent.xinWeChat'
            root_name = 'wxid_selected_fixture'
            account = live / container / 'Data/Documents/xwechat_files' / root_name
            flat = make_account(root, 'flat-fixture')
            for family, name in (('contact', 'contact.db'), ('message', 'message_0.db')):
                target = account / 'db_storage' / family
                target.mkdir(parents=True, exist_ok=True)
                flat.joinpath(name).replace(target / name)
            noisy = account / 'db_storage' / 'MMKV' / 'unreviewed' / 'deep'
            noisy.mkdir(parents=True)
            (noisy / 'contact.db').write_bytes(b'not a reviewed location')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=(SelectedAccount(
                    account_id=root_name,
                    container_id=container,
                    root_name=root_name,
                    output_name='account-fixture',
                ),),
            )

            original_iterdir = Path.iterdir

            def guarded_iterdir(path):
                if path == account / 'db_storage':
                    raise AssertionError('db_storage root enumeration is forbidden')
                return original_iterdir(path)

            with (
                patch.object(Path, 'rglob', side_effect=AssertionError('recursive scan is forbidden')),
                patch.object(Path, 'iterdir', guarded_iterdir),
            ):
                plan = build_decrypt_plan(cfg)

            self.assertTrue(plan.ok)
            self.assertEqual(len(plan.files), 2)
            self.assertEqual({item.file_family for item in plan.files}, {'contact', 'message'})

    def test_secret_decrypt_and_snapshot_staging_run_before_short_publish_writer(self):
        import trove_core.wechat.decrypt.runner as runner_module

        class TrackingResolver:
            def __init__(self, states):
                self.states = states

            def get_secret(self, _name):
                self.states.append(lock_depth > 0)
                return 'fixture-key'

        class TrackingEngine(CopyPlaintextEngine):
            def __init__(self, states):
                self.states = states

            def decrypt(self, source, dest, *, key, file_family):
                self.states.append(lock_depth > 0)
                return super().decrypt(source, dest, key=key, file_family=file_family)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(
                    ['wxid_keep:com.tencent.xinWeChat__wxid_keep'],
                    secret_name='WECHAT_KEY',
                ),
                secret_name='WECHAT_KEY',
            )
            lock_depth = 0
            secret_states: list[bool] = []
            decrypt_states: list[bool] = []
            stage_states: list[bool] = []
            manifest_states: list[bool] = []
            publish_states: list[bool] = []
            original_coordinated = runner_module.coordinated_vault_mutation
            original_stage = runner_module._stage_current
            original_manifest = runner_module.write_manifest
            original_publish = runner_module._publish_current

            @contextmanager
            def tracked_coordinated(*args, **kwargs):
                nonlocal lock_depth
                with original_coordinated(*args, **kwargs) as session:
                    lock_depth += 1
                    try:
                        yield session
                    finally:
                        lock_depth -= 1

            def tracked_stage(*args, **kwargs):
                stage_states.append(lock_depth > 0)
                return original_stage(*args, **kwargs)

            def tracked_manifest(*args, **kwargs):
                manifest_states.append(lock_depth > 0)
                return original_manifest(*args, **kwargs)

            def tracked_publish(*args, **kwargs):
                publish_states.append(lock_depth > 0)
                return original_publish(*args, **kwargs)

            with patch.object(runner_module, 'coordinated_vault_mutation', tracked_coordinated), patch.object(
                runner_module, '_stage_current', side_effect=tracked_stage,
            ), patch.object(
                runner_module, 'write_manifest', side_effect=tracked_manifest,
            ), patch.object(
                runner_module, '_publish_current', side_effect=tracked_publish,
            ):
                report = run_decrypt_plan(
                    build_decrypt_plan(cfg),
                    engine=TrackingEngine(decrypt_states),
                    secret_resolver=TrackingResolver(secret_states),
                )

            self.assertTrue(report['ok'])
            self.assertTrue(secret_states)
            self.assertTrue(decrypt_states)
            self.assertEqual(set(secret_states), {False})
            self.assertEqual(set(decrypt_states), {False})
            self.assertEqual(stage_states, [False])
            self.assertEqual(manifest_states, [True])
            self.assertEqual(publish_states, [True])

    def test_preflight_allows_selected_accounts_and_skips_fts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            make_account(live, 'com.tencent.xinWeChat__wxid_skip')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(['wxid_keep:com.tencent.xinWeChat__wxid_keep'], secret_name='WECHAT_KEY'),
                secret_name='WECHAT_KEY',
            )

            plan = build_decrypt_plan(cfg)
            payload = plan.to_redacted_dict()

            self.assertTrue(plan.ok)
            self.assertEqual(payload['config']['selected_account_count'], 1)
            self.assertEqual(payload['skipped_account_count'], 1)
            self.assertEqual(payload['by_family']['message'], 1)
            self.assertEqual(payload['by_family']['contact'], 1)
            self.assertTrue(any(item['file_family'] == 'message_fts' and item['reason'] == 'out_of_scope' for item in payload['skipped_files']))
            self.assertNotIn(str(live), json.dumps(payload))
            self.assertNotIn('WECHAT_KEY_VALUE', json.dumps(payload))

    def test_run_writes_guard_and_sync_rejects_unselected_snapshot_account(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            selected = make_account(live, 'com.tencent.xinWeChat__wxid_keep', message_text='selected sync needle')
            unselected = make_account(live, 'com.tencent.xinWeChat__wxid_skip', message_text='skip sync needle')
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(['wxid_keep:com.tencent.xinWeChat__wxid_keep']),
            )

            report = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())

            self.assertTrue(report['ok'])
            current = vault / 'sources' / 'wechat-integrated-decrypted' / 'current'
            self.assertTrue((current / INTERNAL_GUARD_NAME).exists())
            self.assertTrue(load_snapshot_guard(current).allows(current / selected.name))
            # Simulate a polluted snapshot. The import-time hard gate must reject it.
            polluted = current / unselected.name
            polluted.mkdir(parents=True)
            for child in unselected.iterdir():
                if child.suffix == '.db':
                    polluted.joinpath(child.name).write_bytes(child.read_bytes())

            sync = run_sync(vault, options=SyncOptions(snapshot_dir=current, full=True))

            self.assertTrue(sync['ok'])
            self.assertIn('not_selected_account', ' '.join(sync['errors']))
            self.assertGreaterEqual(sync['sources_seen'], 2)
            self.assertEqual(sync['messages_imported'], 1)
            self.assertNotIn('selected sync needle', str(sync))
            self.assertNotIn(str(live), str(sync))

    def test_anonymous_output_preserves_private_own_identity_for_direction_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            source_name = 'com.tencent.xinWeChat__opaque-owner-root'
            make_account(live, source_name)
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=(SelectedAccount(
                    'wxid_ownerfixture', root_name=source_name, output_name='account-0123456789abcdef',
                ),),
            )

            report = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())
            account = vault / 'sources' / 'wechat-integrated-decrypted' / 'current' / 'account-0123456789abcdef'
            identity = load_account_identity(account)
            _, _, messages = WeChatDecryptedAccountImporter(account).load()

            self.assertTrue(report['ok'])
            self.assertTrue((account / INTERNAL_ACCOUNT_IDENTITY_NAME).is_file())
            self.assertEqual(identity['own_wxid'], 'wxid_ownerfixture')
            self.assertEqual({message.direction for message in messages}, {'incoming'})
            self.assertNotIn('wxid_ownerfixture', json.dumps(report))

    def test_path_escape_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            acct = make_account(live, 'com.tencent.xinWeChat__wxid_keep')
            outside = root / 'outside.db'
            outside.write_bytes(b'not sqlite')
            (acct / 'hardlink.db').symlink_to(outside)
            cfg = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=selected_accounts_from_strings(['wxid_keep:com.tencent.xinWeChat__wxid_keep']),
            )

            report = run_decrypt_plan(build_decrypt_plan(cfg), engine=CopyPlaintextEngine())

            self.assertFalse(report['ok'])
            self.assertIn('path_escape', report['errors'])
            self.assertNotIn(str(outside), json.dumps(report))

    def test_explicit_partial_mode_keeps_only_complete_accounts_and_declares_key_gap(self):
        class OneAccountFailsEngine(CopyPlaintextEngine):
            def decrypt(self, source, dest, *, key, file_family):
                if 'missing-key' in source.parent.name:
                    return DecryptFileResult('', source.name, file_family, 'failed', error_code='decrypt_failed')
                return super().decrypt(source, dest, key=key, file_family=file_family)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live, 'wechat-available')
            make_account(live, 'wechat-missing-key')
            config = DecryptConfig(
                live_root=live,
                vault_root=vault,
                selected_accounts=(
                    SelectedAccount('available', root_name='wechat-available', output_name='account-a'),
                    SelectedAccount('missing', root_name='wechat-missing-key', output_name='account-b'),
                ),
                allow_partial_accounts=True,
            )

            report = run_decrypt_plan(build_decrypt_plan(config), engine=OneAccountFailsEngine())

            self.assertTrue(report['ok'])
            self.assertEqual(report['status'], 'completed_with_account_gaps')
            self.assertEqual(report['summary']['complete_accounts'], 1)
            self.assertEqual(report['summary']['unavailable_accounts'], 1)
            self.assertEqual(report['terminal_gaps'][0]['kind'], 'account_key_unavailable')
            current = vault / 'sources' / 'wechat-integrated-decrypted' / 'current'
            self.assertTrue((current / 'account-a' / 'contact.db').exists())
            self.assertFalse((current / 'account-b').exists())
            self.assertEqual(len(known_keyed_account_refs(vault)), 1)
            self.assertNotIn('wechat-available', json.dumps(report))
            self.assertNotIn('wechat-missing-key', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
