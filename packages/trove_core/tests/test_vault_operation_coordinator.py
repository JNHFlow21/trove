from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from trove_core.vault import coordinator as coordinator_module
from trove_core.vault import locks as lock_module
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import (
    MutationOutsideCoordinator,
    VaultOperationCoordinator,
    VaultWriteSession,
    require_vault_write_session,
)
from trove_core.vault.locks import VaultOperationLock, VaultOperationLocked


class VaultOperationCoordinatorTests(unittest.TestCase):
    def _cfg(self, directory: str, name: str = 'vault') -> VaultConfig:
        return VaultConfig.resolve(str(Path(directory) / name), env={})

    def test_stable_flock_inode_is_authoritative_and_diagnostics_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory, 'synthetic-private-vault-marker')
            coordinator = VaultOperationCoordinator(cfg)

            with coordinator.write(owner='sync') as session:
                lock_path = cfg.paths.logs_dir / 'trove-index-writer.flock'
                info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
                inode = lock_path.stat().st_ino
                diagnostics = json.loads(info_path.read_text(encoding='utf-8'))

                self.assertTrue(session.active)
                self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(info_path.stat().st_mode), 0o600)
                self.assertEqual(diagnostics['vault_hash'], session.vault_hash)
                self.assertEqual(diagnostics['owner'], 'sync')
                self.assertNotIn(str(cfg.root), info_path.read_text(encoding='utf-8'))
                self.assertFalse(list(cfg.paths.logs_dir.glob('*.tmp')))

            self.assertTrue(lock_path.exists())
            self.assertEqual(lock_path.stat().st_ino, inode)
            self.assertFalse(info_path.exists())

            # A forged/stale diagnostic must never block an unlocked stable
            # flock.  Acquisition replaces it rather than trusting its PID.
            info_path.write_text(
                json.dumps({'pid': os.getpid(), 'owner': 'stale', 'owner_nonce': 'stale'}),
                encoding='utf-8',
            )
            with VaultOperationCoordinator(cfg).write(owner='maintain') as next_session:
                self.assertTrue(next_session.active)
                self.assertEqual(lock_path.stat().st_ino, inode)

    def test_crash_releases_flock_but_marker_requires_explicit_offline_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            child_code = "\n".join(
                [
                    'import os, sys',
                    'from trove_core.vault.coordinator import VaultOperationCoordinator',
                    f"coordinator = VaultOperationCoordinator({str(cfg.root)!r})",
                    "coordinator.acquire(owner='crash-test')",
                    "print('READY', flush=True)",
                    'sys.stdin.buffer.read(1)',
                    'os._exit(23)',
                ]
            )
            child = subprocess.Popen(
                [sys.executable, '-c', child_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), 'READY')  # type: ignore[union-attr]
                lock_path = cfg.paths.logs_dir / 'trove-index-writer.flock'
                info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
                inode = lock_path.stat().st_ino

                with self.assertRaises(VaultOperationLocked):
                    VaultOperationCoordinator(cfg).acquire(owner='contender')

                child.stdin.close()  # type: ignore[union-attr]
                self.assertEqual(child.wait(timeout=5), 23)
                self.assertEqual(child.stderr.read(), '')  # type: ignore[union-attr]
                self.assertTrue(info_path.exists(), 'crash should leave only non-authoritative diagnostics')

                with self.assertRaises(VaultOperationLocked) as blocked:
                    VaultOperationCoordinator(cfg).acquire(owner='recovery')
                self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')

                # Simulate a future explicit offline repair after all old
                # binaries are known stopped; acquisition itself never unlinks.
                (cfg.paths.logs_dir / 'trove-index-writer.lock.json').unlink()
                (cfg.paths.logs_dir / 'trove-index-writer.pid').unlink()
                with VaultOperationCoordinator(cfg).write(owner='recovery') as recovered:
                    self.assertTrue(recovered.active)
                    self.assertEqual(lock_path.stat().st_ino, inode)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                for stream in (child.stdout, child.stderr):
                    if stream is not None:
                        stream.close()

    def test_initialized_flock_still_respects_live_legacy_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with VaultOperationCoordinator(cfg).write(owner='sync'):
                pass
            lock_path = cfg.paths.logs_dir / 'trove-index-writer.flock'
            inode = lock_path.stat().st_ino
            cfg.paths.logs_dir.joinpath('trove-index-writer.pid').write_text(
                f'{os.getpid()}\n',
                encoding='utf-8',
            )
            cfg.paths.logs_dir.joinpath('trove-index-writer.lock.json').write_text(
                json.dumps(
                    {
                        'pid': os.getpid(),
                        'owner': 'legacy-writer',
                        'created_at': 1,
                        'process_start_time': lock_module._process_start_time(os.getpid()),
                    }
                ),
                encoding='utf-8',
            )

            with self.assertRaises(VaultOperationLocked):
                VaultOperationCoordinator(cfg).acquire(owner='maintain')
            self.assertEqual(lock_path.stat().st_ino, inode)
            self.assertEqual(
                cfg.paths.logs_dir.joinpath('trove-index-writer.pid').read_text(encoding='utf-8').strip(),
                str(os.getpid()),
            )

    def test_new_writer_diagnostics_block_the_legacy_algorithm(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with VaultOperationCoordinator(cfg).write(owner='sync'):
                pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
                info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
                observed = lock_module._read_lock_info(pid_path, info_path)
                self.assertTrue(lock_module._lock_owner_running(observed))
                with self.assertRaises(FileExistsError):
                    fd = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    os.close(fd)

    def test_schema1_crash_diagnostic_does_not_follow_reused_live_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            with coordinator.write(owner='sync'):
                pass

            progress = cfg.paths.vector_dir / 'zvec' / 'messages.trove-progress.json'
            progress.parent.mkdir(parents=True)
            progress.write_text(json.dumps({'state': 'running', 'updated_at': 9_999_999_999}), encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text(f'{os.getpid()}\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps(
                    {
                        'schema': 1,
                        'pid': os.getpid(),
                        'owner': 'other',
                        'created_at': 1,
                        'process_birth_hash': '0' * 64,
                        'owner_nonce': 'a' * 32,
                        'vault_hash': coordinator.vault_hash,
                    }
                ),
                encoding='utf-8',
            )

            # The stable flock correctly reports no active vector writer, but
            # acquisition never auto-deletes even a recognizable crash marker.
            self.assertIsNone(lock_module.active_vector_progress(cfg))
            with self.assertRaises(VaultOperationLocked) as blocked:
                VaultOperationCoordinator(cfg).acquire(owner='maintain')
            self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').unlink()
            (cfg.paths.logs_dir / 'trove-index-writer.pid').unlink()
            with VaultOperationCoordinator(cfg).write(owner='maintain') as recovered:
                self.assertTrue(recovered.active)

    def test_active_vector_progress_honors_live_legacy_writer_without_stable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            progress = cfg.paths.vector_dir / 'zvec' / 'messages.trove-progress.json'
            progress.parent.mkdir(parents=True)
            progress.write_text(json.dumps({'state': 'running', 'updated_at': 9_999_999_999}), encoding='utf-8')
            cfg.paths.logs_dir.mkdir(parents=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text(f'{os.getpid()}\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps(
                    {
                        'pid': os.getpid(),
                        'owner': 'legacy-vector',
                        'created_at': 1,
                        'process_start_time': lock_module._process_start_time(os.getpid()),
                    }
                ),
                encoding='utf-8',
            )

            self.assertEqual(lock_module.active_vector_progress(cfg)['state'], 'running')

    def test_session_requires_explicit_parent_and_rejects_release_out_of_order(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            outer = coordinator.acquire(owner='outer')
            try:
                with self.assertRaises(MutationOutsideCoordinator) as implicit:
                    coordinator.acquire(owner='implicit')
                self.assertEqual(implicit.exception.code, 'parent_session_required')

                child = coordinator.acquire(owner='child', parent=outer)
                try:
                    self.assertEqual(child.parent_nonce, outer.session_nonce)
                    self.assertIs(require_vault_write_session(cfg, child), child)
                    with self.assertRaises(MutationOutsideCoordinator) as suspended_parent:
                        require_vault_write_session(cfg, outer)
                    self.assertEqual(suspended_parent.exception.code, 'non_leaf_write_session')
                    with self.assertRaises(MutationOutsideCoordinator) as wrong_parent:
                        coordinator.acquire(owner='sibling', parent=outer)
                    self.assertEqual(wrong_parent.exception.code, 'invalid_parent_session')
                    with self.assertRaises(MutationOutsideCoordinator) as active_child:
                        coordinator.release(outer)
                    self.assertEqual(active_child.exception.code, 'active_child_write_session')
                finally:
                    coordinator.release(child)

                self.assertTrue(outer.active)
                self.assertIs(require_vault_write_session(cfg, outer), outer)
            finally:
                coordinator.release(outer)

            self.assertTrue(outer.released)
            self.assertFalse(outer.active)

    def test_missing_forged_cross_vault_process_mismatch_and_reuse_are_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory, 'first')
            other_cfg = self._cfg(directory, 'second')

            with self.assertRaises(MutationOutsideCoordinator) as missing:
                require_vault_write_session(cfg, None)
            self.assertEqual(missing.exception.code, 'missing_write_session')

            coordinator = VaultOperationCoordinator(cfg)
            with coordinator.write(owner='sync') as session:
                with self.assertRaises(MutationOutsideCoordinator) as forged_constructor:
                    VaultWriteSession(
                        owner='forged',
                        vault_hash=session.vault_hash,
                        pid=session.pid,
                        process_birth=session.process_birth,
                        coordinator_nonce=session.coordinator_nonce,
                        session_nonce='forged',
                        _coordinator=coordinator,
                        _seal=object(),
                    )
                self.assertEqual(forged_constructor.exception.code, 'forged_write_session')

                forged_copy = replace(session, session_nonce='forged-nonce')
                with self.assertRaises(MutationOutsideCoordinator) as forged_nonce:
                    coordinator.validate(forged_copy)
                self.assertEqual(forged_nonce.exception.code, 'inactive_write_session')

                with self.assertRaises(MutationOutsideCoordinator) as cross_vault:
                    require_vault_write_session(other_cfg, session)
                self.assertEqual(cross_vault.exception.code, 'cross_vault_write_session')

                with patch.object(
                    coordinator_module,
                    '_current_process_identity',
                    return_value=(session.pid + 1, session.process_birth),
                ):
                    with self.assertRaises(MutationOutsideCoordinator) as cross_process:
                        coordinator.validate(session)
                self.assertEqual(cross_process.exception.code, 'cross_process_write_session')

                with patch.object(
                    coordinator_module,
                    '_current_process_identity',
                    return_value=(session.pid, f'{session.process_birth}-different'),
                ):
                    with self.assertRaises(MutationOutsideCoordinator) as stale_process:
                        coordinator.validate(session)
                self.assertEqual(stale_process.exception.code, 'stale_process_write_session')

            with self.assertRaises(MutationOutsideCoordinator) as reused:
                require_vault_write_session(cfg, session)
            self.assertEqual(reused.exception.code, 'released_write_session')

    def test_compatibility_facade_exposes_sealed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            lock = VaultOperationLock(cfg, owner='import')
            with lock:
                session = lock.write_session
                self.assertIsNotNone(session)
                self.assertIs(require_vault_write_session(cfg, session), session)
            self.assertIsNone(lock.write_session)
            self.assertTrue((cfg.paths.logs_dir / 'trove-index-writer.flock').exists())

    def test_invalid_owner_is_rejected_before_diagnostics_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            with self.assertRaises(MutationOutsideCoordinator) as invalid:
                VaultOperationCoordinator(cfg).acquire(owner='raw secret with spaces')
            self.assertEqual(invalid.exception.code, 'invalid_writer_owner')
            self.assertFalse(cfg.paths.logs_dir.exists())

    def test_unconfigured_default_cannot_create_writer_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = VaultConfig.resolve(None, env={'HOME': directory})
            self.assertEqual(cfg.source, 'unconfigured')
            with self.assertRaises(ValueError):
                VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertFalse(cfg.root.exists())

    def test_logs_symlink_cannot_redirect_writer_files_outside_vault(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self._cfg(directory)
            cfg.root.mkdir()
            outside = root / 'outside'
            outside.mkdir()
            cfg.paths.logs_dir.symlink_to(outside, target_is_directory=True)

            with self.assertRaises(VaultOperationLocked) as rejected:
                VaultOperationCoordinator(cfg).acquire(owner='sync')

            self.assertEqual(rejected.exception.code, 'vault_writer_lock_unavailable')
            self.assertEqual(list(outside.iterdir()), [])

    def test_release_recovers_when_non_authoritative_json_was_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            session = coordinator.acquire(owner='sync')
            info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            info_path.unlink()

            coordinator.release(session)

            self.assertFalse(pid_path.exists())
            with VaultOperationCoordinator(cfg).write(owner='maintain') as recovered:
                self.assertTrue(recovered.active)

    def test_unknown_valid_owner_is_redacted_in_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            synthetic_owner = 'syntheticsecretlikeowner'
            with VaultOperationCoordinator(cfg).write(owner=synthetic_owner):
                text = (cfg.paths.logs_dir / 'trove-index-writer.lock.json').read_text(encoding='utf-8')
                payload = json.loads(text)
                self.assertEqual(payload['owner'], 'other')
                self.assertNotIn(synthetic_owner, text)

    def test_empty_or_corrupt_legacy_marker_never_ages_out_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            cfg.paths.logs_dir.mkdir(parents=True)
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            pid_path.write_bytes(b'')

            # Reproduce a legacy writer paused after O_EXCL for longer than the
            # former publication grace.  It must still retain exclusivity.
            time.sleep(2.1)
            with self.assertRaises(VaultOperationLocked) as empty:
                VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(empty.exception.code, 'vault_writer_marker_recovery_required')
            self.assertEqual(pid_path.read_bytes(), b'')

            pid_path.write_bytes(b'123')
            with self.assertRaises(VaultOperationLocked):
                VaultOperationCoordinator(cfg, stale_seconds=0).acquire(owner='sync')

            pid_path.write_text('untrusted-diagnostic-content', encoding='utf-8')
            with self.assertRaises(VaultOperationLocked) as corrupt:
                VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(corrupt.exception.code, 'vault_writer_marker_recovery_required')
            self.assertNotIn('untrusted', str(corrupt.exception))

            old = time.time() - (10 * 24 * 60 * 60)
            os.utime(pid_path, (old, old))
            with self.assertRaises(VaultOperationLocked) as still_blocked:
                VaultOperationCoordinator(cfg, stale_seconds=0).acquire(owner='sync')
            self.assertEqual(still_blocked.exception.code, 'vault_writer_marker_recovery_required')
            self.assertEqual(pid_path.read_text(encoding='utf-8'), 'untrusted-diagnostic-content')

    def test_dead_marker_is_untouched_for_a_paused_legacy_stale_cleaner(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            cfg.paths.logs_dir.mkdir(parents=True)
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
            pid_path.write_text('99999999\n', encoding='utf-8')
            info_path.write_text(
                json.dumps({'pid': 99999999, 'created_at': 1, 'owner': 'legacy'}),
                encoding='utf-8',
            )
            pid_identity = (pid_path.stat().st_dev, pid_path.stat().st_ino)
            info_bytes = info_path.read_bytes()

            # A legacy contender may already be paused immediately before an
            # unconditional stale unlink.  New code must not replace its inode.
            with patch.object(lock_module, '_pid_running', return_value=False):
                with self.assertRaises(VaultOperationLocked) as blocked:
                    VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')
            self.assertEqual((pid_path.stat().st_dev, pid_path.stat().st_ino), pid_identity)
            self.assertEqual(info_path.read_bytes(), info_bytes)

    def test_pid_marker_is_prefilled_and_info_failure_cannot_leave_self_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            observed: list[bytes] = []

            def fail_info(dir_fd, name, payload):
                fd = os.open('trove-index-writer.pid', os.O_RDONLY, dir_fd=dir_fd)
                try:
                    observed.append(os.read(fd, 64))
                finally:
                    os.close(fd)
                raise OSError('synthetic info failure')

            with patch.object(lock_module, '_atomic_write_json_at', side_effect=fail_info):
                with self.assertRaises(VaultOperationLocked) as failed:
                    VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(failed.exception.code, 'vault_writer_diagnostics_unavailable')
            self.assertEqual(observed, [f'{os.getpid()}\n'.encode('ascii')])
            self.assertFalse((cfg.paths.logs_dir / 'trove-index-writer.pid').exists())
            with VaultOperationCoordinator(cfg).write(owner='sync') as recovered:
                self.assertTrue(recovered.active)

    def test_post_publish_pid_fault_cleans_only_the_published_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            original = lock_module._fsync_directory
            calls = 0

            def fail_first_directory_sync(fd):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError('synthetic durability failure')
                return original(fd)

            with patch.object(lock_module, '_fsync_directory', side_effect=fail_first_directory_sync):
                with self.assertRaises(VaultOperationLocked) as failed:
                    VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(failed.exception.code, 'vault_writer_diagnostics_unavailable')
            self.assertFalse((cfg.paths.logs_dir / 'trove-index-writer.pid').exists())

            real_unlink = lock_module.os.unlink
            failed_temp_unlink = False

            def fail_published_temp_unlink(path, *args, **kwargs):
                nonlocal failed_temp_unlink
                if not failed_temp_unlink and str(path).startswith('.trove-index-writer.pid.'):
                    failed_temp_unlink = True
                    raise OSError('synthetic post-link unlink failure')
                return real_unlink(path, *args, **kwargs)

            with patch.object(lock_module.os, 'unlink', side_effect=fail_published_temp_unlink):
                with self.assertRaises(VaultOperationLocked):
                    VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertTrue(failed_temp_unlink)
            self.assertFalse((cfg.paths.logs_dir / 'trove-index-writer.pid').exists())
            self.assertFalse(list(cfg.paths.logs_dir.glob('*.tmp')))
            with VaultOperationCoordinator(cfg).write(owner='sync') as recovered:
                self.assertTrue(recovered.active)

    def test_malformed_diagnostic_numbers_never_escape_parser_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            cfg.paths.logs_dir.mkdir(parents=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text('99999999\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({'pid': 'not-a-number', 'created_at': 'not-a-float', 'schema': []}),
                encoding='utf-8',
            )
            with self.assertRaises(VaultOperationLocked) as blocked:
                VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')
            self.assertEqual(
                (cfg.paths.logs_dir / 'trove-index-writer.pid').read_text(encoding='utf-8'),
                '99999999\n',
            )

            progress = cfg.paths.vector_dir / 'zvec' / 'messages.trove-progress.json'
            progress.parent.mkdir(parents=True)
            progress.write_text(json.dumps({'state': 'running', 'updated_at': {'bad': 'value'}}), encoding='utf-8')
            self.assertIsNone(lock_module.active_vector_progress(cfg))

    def test_hardlinked_flock_is_rejected_before_chmod_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            cfg.paths.logs_dir.mkdir(parents=True)
            outside = Path(directory) / 'outside-file'
            outside.write_bytes(b'outside-content')
            outside.chmod(0o640)
            os.link(outside, cfg.paths.logs_dir / 'trove-index-writer.flock')

            with self.assertRaises(VaultOperationLocked) as rejected:
                VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(rejected.exception.code, 'vault_writer_lock_unavailable')
            self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o640)
            self.assertEqual(outside.read_bytes(), b'outside-content')

    def test_release_survives_info_directory_replacement_and_removes_owned_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            session = coordinator.acquire(owner='sync')
            info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            info_path.unlink()
            info_path.mkdir()

            coordinator.release(session)

            self.assertTrue(info_path.is_dir())
            self.assertFalse(pid_path.exists())
            info_path.rmdir()
            with VaultOperationCoordinator(cfg).write(owner='maintain') as recovered:
                self.assertTrue(recovered.active)

    def test_release_never_deletes_replaced_pid_even_with_same_numeric_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            session = coordinator.acquire(owner='sync')
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            pid_path.unlink()
            pid_path.write_text(f'{os.getpid()}\n', encoding='utf-8')

            coordinator.release(session)

            self.assertEqual(pid_path.read_text(encoding='utf-8'), f'{os.getpid()}\n')

    def test_root_retarget_invalidates_session_and_release_uses_bound_dirfd(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first_root = base / 'first'
            second_root = base / 'second'
            first_root.mkdir()
            second_root.mkdir()
            first_cfg = VaultConfig.resolve(str(first_root), env={})
            second_cfg = VaultConfig.resolve(str(second_root), env={})
            first = VaultOperationCoordinator(first_cfg)
            first_session = first.acquire(owner='sync')

            moved_root = base / 'first-moved'
            first_root.rename(moved_root)
            first_root.symlink_to(second_root, target_is_directory=True)
            second = VaultOperationCoordinator(second_cfg)
            second_session = second.acquire(owner='maintain')
            second_info = second_root / 'logs' / 'trove-index-writer.lock.json'
            second_nonce = json.loads(second_info.read_text(encoding='utf-8'))['owner_nonce']
            try:
                with self.assertRaises(MutationOutsideCoordinator) as changed:
                    require_vault_write_session(first_cfg, first_session)
                self.assertEqual(changed.exception.code, 'vault_writer_path_changed')

                first.release(first_session)
                self.assertEqual(json.loads(second_info.read_text(encoding='utf-8'))['owner_nonce'], second_nonce)
                self.assertFalse((moved_root / 'logs' / 'trove-index-writer.pid').exists())
            finally:
                second.release(second_session)

    def test_final_root_symlink_is_rejected_without_writing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / 'target'
            target.mkdir()
            link = base / 'vault-link'
            link.symlink_to(target, target_is_directory=True)
            cfg = VaultConfig.resolve(str(link), env={})

            with self.assertRaises(VaultOperationLocked) as rejected:
                VaultOperationCoordinator(cfg).acquire(owner='sync')
            self.assertEqual(rejected.exception.code, 'vault_writer_lock_unavailable')
            self.assertFalse((target / 'logs').exists())

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires fork')
    def test_forked_child_invalidates_inherited_session_without_releasing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            session = coordinator.acquire(owner='sync')
            read_fd, write_fd = os.pipe()
            child = os.fork()
            if child == 0:
                try:
                    os.close(read_fd)
                    os.write(write_fd, b'inactive' if not session.active else b'active')
                finally:
                    os._exit(0)
            os.close(write_fd)
            try:
                self.assertEqual(os.read(read_fd, 32), b'inactive')
                os.waitpid(child, 0)
                self.assertTrue(session.active)
            finally:
                os.close(read_fd)
                coordinator.release(session)

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires fork')
    def test_child_does_not_keep_parent_flock_alive_after_parent_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            code = "\n".join(
                [
                    'import os, time',
                    'from trove_core.vault.coordinator import VaultOperationCoordinator',
                    f"c = VaultOperationCoordinator({str(cfg.root)!r})",
                    "c.acquire(owner='sync')",
                    'child = os.fork()',
                    "if child == 0:\n    time.sleep(5)\n    os._exit(0)",
                    "print(child, flush=True)",
                    'os._exit(23)',
                ]
            )
            holder = subprocess.Popen(
                [sys.executable, '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            sleeper = int(holder.stdout.readline().strip())  # type: ignore[union-attr]
            try:
                self.assertEqual(holder.wait(timeout=5), 23)
                lock_fd = os.open(cfg.paths.logs_dir / 'trove-index-writer.flock', os.O_RDWR)
                try:
                    # The still-running fork child did not inherit ownership of
                    # the parent's open-file description.
                    lock_module.fcntl.flock(lock_fd, lock_module.fcntl.LOCK_EX | lock_module.fcntl.LOCK_NB)
                    lock_module.fcntl.flock(lock_fd, lock_module.fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
                with self.assertRaises(VaultOperationLocked) as blocked:
                    VaultOperationCoordinator(cfg).acquire(owner='maintain')
                self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')
            finally:
                try:
                    os.kill(sleeper, 9)
                except ProcessLookupError:
                    pass
                for stream in (holder.stdout, holder.stderr):
                    if stream is not None:
                        stream.close()
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').unlink()
            (cfg.paths.logs_dir / 'trove-index-writer.pid').unlink()
            with VaultOperationCoordinator(cfg).write(owner='maintain') as recovered:
                self.assertTrue(recovered.active)

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires fork')
    def test_multithreaded_fork_reinitializes_inherited_global_mutexes(self):
        code = "\n".join(
            [
                'import os, tempfile, threading',
                'from trove_core.vault import coordinator as m',
                'c = m.VaultOperationCoordinator(tempfile.mkdtemp())',
                'ready = threading.Event()',
                'release = threading.Event()',
                'def hold():',
                '    with c._mutex:',
                '        with m._PROCESS_IDENTITY_LOCK:',
                '            with m._PROCESS_COORDINATORS_LOCK:',
                '                ready.set()',
                '                release.wait(5)',
                't = threading.Thread(target=hold, daemon=True)',
                't.start()',
                'assert ready.wait(2)',
                'child = os.fork()',
                'if child == 0:',
                '    with c._mutex:',
                "        print('CHILD', m._current_process_identity()[0], flush=True)",
                '    os._exit(0)',
                'release.set()',
                'os.waitpid(child, 0)',
                't.join(2)',
                "print('DONE', flush=True)",
            ]
        )
        completed = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('CHILD', completed.stdout)
        self.assertIn('DONE', completed.stdout)


if __name__ == '__main__':
    unittest.main()
