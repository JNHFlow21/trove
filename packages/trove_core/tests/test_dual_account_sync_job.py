from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.jobs.dual_account_sync import (
    SYNC_CLIENT_TIMEOUT_SECONDS,
    SYNC_PROCESS_TIMEOUT_SECONDS,
    SYNC_STATUS_CLIENT_TIMEOUT_SECONDS,
    SYNC_STATUS_PROCESS_TIMEOUT_SECONDS,
    _prune_runs,
    run_once,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.decrypt.runner import CopyPlaintextEngine
from trove_core.wechat.importers.wechat_decrypted import msg_table_for


class _SecretResolver:
    def get_secret(self, _name):
        return '{}'


def _account(root: Path, name: str, message: str) -> None:
    account = root / name / 'db_storage'
    account.mkdir(parents=True)
    with sqlite3.connect(account / 'contact.db') as connection:
        connection.execute(
            'CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)'
        )
        connection.execute(
            'INSERT INTO contact VALUES (?,?,?,?)',
            ('wxid_friend', 'Friend', '', ''),
        )
    table = msg_table_for('wxid_friend')
    with sqlite3.connect(account / 'message_0.db') as connection:
        connection.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
        connection.execute(
            'INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)',
            (1, 'wxid_friend', 1),
        )
        connection.execute(f'''CREATE TABLE {table} (
            local_id INTEGER, server_id INTEGER, local_type INTEGER, sort_seq INTEGER,
            real_sender_id INTEGER, create_time INTEGER, status INTEGER, upload_status INTEGER,
            download_status INTEGER, server_seq INTEGER, origin_source INTEGER, source INTEGER,
            message_content TEXT, compress_content BLOB, packed_info_data BLOB,
            WCDB_CT_message_content BLOB, WCDB_CT_source BLOB
        )''')
        connection.execute(
            f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)',
            (1, 1, 1710000000, message),
        )


def _terminal_run(path: Path) -> None:
    path.mkdir(parents=True)
    (path / 'decrypt_manifest.redacted.json').write_text(
        json.dumps({'ok': True, 'status': 'completed', 'run_id': path.name}),
        encoding='utf-8',
    )


class DualAccountSyncJobTests(unittest.TestCase):
    def test_retention_fails_closed_when_current_targets_nested_directory(self):
        with tempfile.TemporaryDirectory() as d:
            vault = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            vault.ensure()
            base = vault.root / 'sources' / 'wechat-integrated-decrypted'
            runs = base / 'runs'
            first = runs / '20260723T010000000000Z'
            second = runs / '20260723T020000000000Z'
            _terminal_run(first)
            _terminal_run(second)
            nested = first / 'nested'
            _terminal_run(nested)
            (base / 'current').symlink_to(nested, target_is_directory=True)

            report = _prune_runs(vault, retained=1)

            self.assertEqual(report['invalid_current'], 1)
            self.assertEqual(report['removed'], 0)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_retention_fails_closed_when_current_symlink_is_broken(self):
        with tempfile.TemporaryDirectory() as d:
            vault = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            vault.ensure()
            base = vault.root / 'sources' / 'wechat-integrated-decrypted'
            runs = base / 'runs'
            first = runs / '20260723T010000000000Z'
            second = runs / '20260723T020000000000Z'
            _terminal_run(first)
            _terminal_run(second)
            (base / 'current').symlink_to(runs / 'missing', target_is_directory=True)

            report = _prune_runs(vault, retained=1)

            self.assertEqual(report['invalid_current'], 1)
            self.assertEqual(report['removed'], 0)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_decrypt_exception_still_runs_safe_retention(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            jobs = vault / 'jobs'
            source = live / 'com.tencent.xinWeChat' / 'Data/Documents/xwechat_files'
            _account(source, 'wxid_first1_suffix', 'first account message')
            jobs.mkdir(parents=True)
            config = jobs / 'dual_account_sync.private.json'
            config.write_text(json.dumps({
                'version': 1,
                'live_root': str(live),
                'secret_name': 'TROVE_WECHAT_KEY_STORE',
                'retained_runs': 1,
                'selected_accounts': [{
                    'account_id': 'wxid_first1_suffix',
                    'container_id': 'com.tencent.xinWeChat',
                    'root_name': 'wxid_first1_suffix',
                    'output_name': 'account-first',
                }],
            }), encoding='utf-8')
            config.chmod(0o600)

            base = vault / 'sources' / 'wechat-integrated-decrypted'
            runs = base / 'runs'
            current_run = runs / '20260723T010000000000Z'
            obsolete_run = runs / '20260723T020000000000Z'
            current_run.mkdir(parents=True)
            obsolete_run.mkdir()
            for run in (current_run, obsolete_run):
                (run / 'decrypt_manifest.redacted.json').write_text(
                    json.dumps({
                        'ok': True,
                        'status': 'completed',
                        'run_id': run.name,
                    }),
                    encoding='utf-8',
                )
            base.mkdir(parents=True, exist_ok=True)
            (base / 'current').symlink_to(current_run, target_is_directory=True)

            def fail_during_publish(*_args, **_kwargs):
                raise RuntimeError('writer unavailable')

            with patch(
                'trove_core.jobs.dual_account_sync.run_decrypt_plan',
                side_effect=fail_during_publish,
            ):
                with self.assertRaisesRegex(RuntimeError, 'writer unavailable'):
                    run_once(
                        vault,
                        config_path=config,
                        engine=CopyPlaintextEngine(),
                        secret_resolver=_SecretResolver(),
                    )

            self.assertEqual((base / 'current').resolve(), current_run.resolve())
            self.assertEqual(
                {path.name for path in runs.iterdir() if path.is_dir()},
                {current_run.name},
            )

    def test_retention_keeps_bound_generation_without_accumulating_history(self):
        with tempfile.TemporaryDirectory() as d:
            vault = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            vault.ensure()
            base = vault.root / 'sources' / 'wechat-integrated-decrypted'
            runs = base / 'runs'
            generations = [
                runs / f'20260723T0{hour}0000000000Z'
                for hour in range(1, 5)
            ]
            for run in generations:
                _terminal_run(run)
                (run / 'account-first').mkdir()
            (base / 'current').symlink_to(generations[-1], target_is_directory=True)

            store = SQLiteStore(vault.paths.sqlite_path)
            store.initialize()
            with store.connect() as connection:
                connection.execute(
                    """INSERT INTO media_assets(
                           asset_id,account_id,source_type,source_id,modality,media_type,
                           citation,cache_state,processing_state,metadata_json,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'asset-bound', 'account-1', 'private_chat', 'source-1',
                        'voice', 'voice', 'citation-1', 'unknown', 'pending',
                        '{}', '2026-07-23T00:00:00Z', '2026-07-23T00:00:00Z',
                    ),
                )
                connection.execute(
                    """INSERT INTO source_snapshots(
                           snapshot_revision,root_ref,manifest_hash,guard_run_id_hash,
                           state,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        'snapshot-bound',
                        str(
                            (generations[0] / 'account-first').relative_to(vault.root)
                        ),
                        'manifest-hash', None, 'available',
                        '2026-07-23T00:00:00Z', '2026-07-23T00:00:00Z',
                    ),
                )
                connection.execute(
                    """INSERT INTO media_source_bindings(
                           asset_id,snapshot_revision,account_dir_hash,
                           source_coordinates_json,locator_state,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        'asset-bound', 'snapshot-bound', 'account-hash', '{}',
                        'bound', '2026-07-23T00:00:00Z', '2026-07-23T00:00:00Z',
                    ),
                )
                connection.commit()

            report = _prune_runs(vault, retained=1)

            self.assertEqual(report['removed'], 2)
            self.assertEqual(report['retained'], 2)
            self.assertEqual(report['protected'], 1)
            self.assertTrue(generations[0].is_dir())
            self.assertFalse(generations[1].exists())
            self.assertFalse(generations[2].exists())
            self.assertTrue(generations[3].is_dir())

    def test_nonterminal_generation_blocks_another_large_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            jobs = vault / 'jobs'
            source = live / 'com.tencent.xinWeChat' / 'Data/Documents/xwechat_files'
            _account(source, 'wxid_first1_suffix', 'first account message')
            jobs.mkdir(parents=True)
            config = jobs / 'dual_account_sync.private.json'
            config.write_text(json.dumps({
                'version': 1,
                'live_root': str(live),
                'secret_name': 'TROVE_WECHAT_KEY_STORE',
                'retained_runs': 1,
                'selected_accounts': [{
                    'account_id': 'wxid_first1_suffix',
                    'container_id': 'com.tencent.xinWeChat',
                    'root_name': 'wxid_first1_suffix',
                    'output_name': 'account-first',
                }],
            }), encoding='utf-8')
            config.chmod(0o600)
            unfinished = (
                vault / 'sources' / 'wechat-integrated-decrypted'
                / 'runs' / '20260723T010000000000Z'
            )
            unfinished.mkdir(parents=True)

            with patch(
                'trove_core.jobs.dual_account_sync.run_decrypt_plan',
            ) as decrypt:
                report = run_once(
                    vault,
                    config_path=config,
                    engine=CopyPlaintextEngine(),
                    secret_resolver=_SecretResolver(),
                )

            self.assertFalse(report['ok'])
            self.assertEqual(report['status'], 'snapshot_build_present')
            self.assertEqual(report['nonterminal_runs'], 1)
            decrypt.assert_not_called()
            self.assertEqual(
                len([path for path in unfinished.parent.iterdir() if path.is_dir()]),
                1,
            )

    def test_two_selected_containers_sync_once_then_skip_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            jobs = vault / 'jobs'
            first_root = live / 'com.tencent.xinWeChat' / 'Data/Documents/xwechat_files'
            second_root = live / 'com.tencent.xinWeChat2' / 'Data/Documents/xwechat_files'
            _account(first_root, 'wxid_first1_suffix', 'first account message')
            _account(second_root, 'wxid_second2_suffix', 'second account message')
            jobs.mkdir(parents=True)
            config = jobs / 'dual_account_sync.private.json'
            config.write_text(json.dumps({
                'version': 1,
                'live_root': str(live),
                'secret_name': 'TROVE_WECHAT_KEY_STORE',
                'retained_runs': 1,
                'selected_accounts': [
                    {
                        'account_id': 'wxid_first1_suffix',
                        'container_id': 'com.tencent.xinWeChat',
                        'root_name': 'wxid_first1_suffix',
                        'output_name': 'account-first',
                    },
                    {
                        'account_id': 'wxid_second2_suffix',
                        'container_id': 'com.tencent.xinWeChat2',
                        'root_name': 'wxid_second2_suffix',
                        'output_name': 'account-second',
                    },
                ],
            }), encoding='utf-8')
            config.chmod(0o600)
            calls: list[list[str]] = []

            def sync_runner(arguments, **_kwargs):
                calls.append(list(arguments))
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=json.dumps({
                        'ok': True,
                        'data': {
                            'operation': {
                                'state': 'completed',
                                'result': {
                                    'messages_imported': 2,
                                    'sources_seen': 2,
                                },
                            },
                        },
                    }),
                    stderr='',
                )

            first = run_once(
                vault,
                config_path=config,
                engine=CopyPlaintextEngine(),
                secret_resolver=_SecretResolver(),
                sync_runner=sync_runner,
            )
            second = run_once(
                vault,
                config_path=config,
                engine=CopyPlaintextEngine(),
                secret_resolver=_SecretResolver(),
                sync_runner=sync_runner,
            )

            self.assertTrue(first['ok'])
            self.assertEqual(first['selected_accounts'], 2)
            self.assertEqual(first['sync']['messages_imported'], 2)
            self.assertEqual(second['status'], 'unchanged')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].count('--account-ids'), 2)
            self.assertFalse((jobs / 'dual_account_sync_state.redacted.json').stat().st_mode & 0o077)

            current = (vault / 'sources' / 'wechat-integrated-decrypted' / 'current').resolve()
            shutil.rmtree(current / 'account-second')
            repaired = run_once(
                vault,
                config_path=config,
                engine=CopyPlaintextEngine(),
                secret_resolver=_SecretResolver(),
                sync_runner=sync_runner,
            )
            self.assertTrue(repaired['ok'])
            self.assertEqual(repaired['status'], 'completed')
            self.assertEqual(len(calls), 2)
            self.assertEqual(len([path for path in (vault / 'sources' / 'wechat-integrated-decrypted' / 'current').iterdir() if path.is_dir()]), 2)

    def test_retention_waits_for_successful_sync_before_removing_bound_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            jobs = vault / 'jobs'
            source = live / 'com.tencent.xinWeChat' / 'Data/Documents/xwechat_files'
            _account(source, 'wxid_first1_suffix', 'first account message')
            jobs.mkdir(parents=True)
            config = jobs / 'dual_account_sync.private.json'
            config.write_text(json.dumps({
                'version': 1,
                'live_root': str(live),
                'secret_name': 'TROVE_WECHAT_KEY_STORE',
                'retained_runs': 1,
                'selected_accounts': [{
                    'account_id': 'wxid_first1_suffix',
                    'container_id': 'com.tencent.xinWeChat',
                    'root_name': 'wxid_first1_suffix',
                    'output_name': 'account-first',
                }],
            }), encoding='utf-8')
            config.chmod(0o600)

            def completed(arguments, **_kwargs):
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=json.dumps({
                        'ok': True,
                        'data': {'operation': {'state': 'completed', 'result': {}}},
                    }),
                    stderr='',
                )

            first = run_once(
                vault,
                config_path=config,
                engine=CopyPlaintextEngine(),
                secret_resolver=_SecretResolver(),
                sync_runner=completed,
            )
            self.assertTrue(first['ok'])
            base = vault / 'sources' / 'wechat-integrated-decrypted'
            previous = (base / 'current').resolve(strict=True)

            message_db = (
                source / 'wxid_first1_suffix' / 'db_storage' / 'message_0.db'
            )
            table = msg_table_for('wxid_friend')
            with sqlite3.connect(message_db) as connection:
                connection.execute(
                    f'INSERT INTO {table}(local_id,real_sender_id,create_time,message_content) VALUES (?,?,?,?)',
                    (2, 1, 1710000001, 'second account message'),
                )

            def failed(arguments, **_kwargs):
                self.assertTrue(previous.is_dir())
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=json.dumps({
                        'ok': True,
                        'data': {'operation': {'state': 'failed', 'result': {}}},
                    }),
                    stderr='',
                )

            second = run_once(
                vault,
                config_path=config,
                engine=CopyPlaintextEngine(),
                secret_resolver=_SecretResolver(),
                sync_runner=failed,
            )

            self.assertFalse(second['ok'])
            self.assertTrue(previous.is_dir())

            def completed_after_retry(arguments, **_kwargs):
                self.assertTrue(previous.is_dir())
                return completed(arguments, **_kwargs)

            third = run_once(
                vault,
                config_path=config,
                engine=CopyPlaintextEngine(),
                secret_resolver=_SecretResolver(),
                sync_runner=completed_after_retry,
            )

            self.assertTrue(third['ok'])
            self.assertFalse(previous.exists())

    def test_terminal_failure_advances_attempt_and_reuses_pending_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            jobs = vault / 'jobs'
            source = live / 'com.tencent.xinWeChat' / 'Data/Documents/xwechat_files'
            _account(source, 'wxid_first1_suffix', 'first account message')
            jobs.mkdir(parents=True)
            config = jobs / 'dual_account_sync.private.json'
            config.write_text(json.dumps({
                'version': 1,
                'live_root': str(live),
                'secret_name': 'TROVE_WECHAT_KEY_STORE',
                'retained_runs': 1,
                'selected_accounts': [{
                    'account_id': 'wxid_first1_suffix',
                    'container_id': 'com.tencent.xinWeChat',
                    'root_name': 'wxid_first1_suffix',
                    'output_name': 'account-first',
                }],
            }), encoding='utf-8')
            config.chmod(0o600)
            calls: list[list[str]] = []

            def sync_runner(arguments, **_kwargs):
                calls.append(list(arguments))
                self.assertEqual(
                    arguments[arguments.index('--timeout') + 1],
                    str(SYNC_CLIENT_TIMEOUT_SECONDS),
                )
                self.assertEqual(
                    _kwargs['timeout'],
                    SYNC_PROCESS_TIMEOUT_SECONDS,
                )
                state = 'failed' if len(calls) == 1 else 'completed'
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=json.dumps({
                        'ok': True,
                        'data': {'operation': {'state': state, 'result': {}}},
                    }),
                    stderr='',
                )

            reports = [
                run_once(
                    vault,
                    config_path=config,
                    engine=CopyPlaintextEngine(),
                    secret_resolver=_SecretResolver(),
                    sync_runner=sync_runner,
                )
                for _ in range(2)
            ]

            keys = [call[call.index('--idempotency-key') + 1] for call in calls]
            self.assertFalse(reports[0]['ok'])
            self.assertEqual(reports[1]['decrypt']['status'], 'reused_pending_snapshot')
            self.assertTrue(reports[1]['ok'])
            self.assertNotEqual(keys[0], keys[1])

    def test_timeout_replays_same_operation_then_polls_to_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'Containers'
            vault = root / 'vault'
            jobs = vault / 'jobs'
            source = live / 'com.tencent.xinWeChat' / 'Data/Documents/xwechat_files'
            _account(source, 'wxid_first1_suffix', 'first account message')
            jobs.mkdir(parents=True)
            config = jobs / 'dual_account_sync.private.json'
            config.write_text(json.dumps({
                'version': 1,
                'live_root': str(live),
                'secret_name': 'TROVE_WECHAT_KEY_STORE',
                'retained_runs': 1,
                'selected_accounts': [{
                    'account_id': 'wxid_first1_suffix',
                    'container_id': 'com.tencent.xinWeChat',
                    'root_name': 'wxid_first1_suffix',
                    'output_name': 'account-first',
                }],
            }), encoding='utf-8')
            config.chmod(0o600)
            calls: list[list[str]] = []
            sync_calls = 0
            status_calls = 0

            def sync_runner(arguments, **kwargs):
                nonlocal sync_calls, status_calls
                calls.append(list(arguments))
                if 'sync' in arguments:
                    sync_calls += 1
                    self.assertEqual(
                        arguments[arguments.index('--timeout') + 1],
                        str(SYNC_CLIENT_TIMEOUT_SECONDS),
                    )
                    self.assertEqual(kwargs['timeout'], SYNC_PROCESS_TIMEOUT_SECONDS)
                    if sync_calls == 1:
                        raise subprocess.TimeoutExpired(
                            arguments,
                            SYNC_PROCESS_TIMEOUT_SECONDS,
                        )
                    state = 'running'
                else:
                    status_calls += 1
                    self.assertEqual(
                        arguments[arguments.index('--timeout') + 1],
                        str(SYNC_STATUS_CLIENT_TIMEOUT_SECONDS),
                    )
                    self.assertEqual(
                        kwargs['timeout'],
                        SYNC_STATUS_PROCESS_TIMEOUT_SECONDS,
                    )
                    state = 'running' if status_calls == 1 else 'completed'
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=json.dumps({
                        'ok': True,
                        'data': {
                            'operation': {
                                'operation_id': 'op_test_replay',
                                'state': state,
                                'result': {},
                            },
                        },
                    }),
                    stderr='',
                )

            with patch(
                'trove_core.jobs.dual_account_sync.time.sleep',
                return_value=None,
            ):
                report = run_once(
                    vault,
                    config_path=config,
                    engine=CopyPlaintextEngine(),
                    secret_resolver=_SecretResolver(),
                    sync_runner=sync_runner,
                )

            keys = [
                call[call.index('--idempotency-key') + 1]
                for call in calls
                if 'sync' in call
            ]
            self.assertTrue(report['ok'])
            self.assertEqual(sync_calls, 2)
            self.assertEqual(status_calls, 2)
            self.assertEqual(len(set(keys)), 1)
            state = json.loads((jobs / 'dual_account_sync_state.redacted.json').read_text())
            self.assertEqual(state['last_status'], 'completed')
            self.assertEqual(state['sync_attempt'], 0)


if __name__ == '__main__':
    unittest.main()
