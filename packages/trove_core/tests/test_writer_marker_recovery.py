from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trove_core.application import writer_recovery as recovery_app
from trove_core.application.writer_recovery import (
    WRITER_MARKER_RECOVERY_ACTION,
    WRITER_MARKER_RECOVERY_DANGER_CLASS,
    recover_writer_marker,
    writer_marker_recovery_payload,
)
from trove_core.approvals import (
    ApprovalGrant,
    ApprovalManager,
    ApprovalRequired,
    ApprovalValidationError,
    SENSITIVE_CAPABILITY_INVENTORY,
)
from trove_core.vault import locks as lock_module
from trove_core.vault import writer_recovery as recovery_protocol
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.writer_recovery import WriterMarkerRecoveryError


class WriterMarkerRecoveryTests(unittest.TestCase):
    def _cfg(self, directory: str) -> VaultConfig:
        return VaultConfig.resolve(str(Path(directory) / 'synthetic-vault'), env={})

    def _write_marker(
        self,
        cfg: VaultConfig,
        *,
        pid: int,
        info: dict | None = None,
    ) -> tuple[Path, Path]:
        cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
        info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
        pid_path.write_text(f'{pid}\n', encoding='ascii')
        if info is not None:
            info_path.write_text(json.dumps(info), encoding='utf-8')
        return pid_path, info_path

    def _approved(self, cfg: VaultConfig) -> tuple[ApprovalManager, str, ApprovalGrant]:
        payload = writer_marker_recovery_payload(legacy_writers_stopped=True)
        manager = ApprovalManager(cfg.root)
        record = manager.request(
            WRITER_MARKER_RECOVERY_ACTION,
            WRITER_MARKER_RECOVERY_DANGER_CLASS,
            payload,
        )
        manager.decide(record.approval_id, 'approved')
        grant = manager.consume(
            WRITER_MARKER_RECOVERY_ACTION,
            WRITER_MARKER_RECOVERY_DANGER_CLASS,
            payload,
            approval_id=record.approval_id,
        )
        return manager, record.approval_id, grant

    def test_crashed_writer_requires_explicit_recovery_then_can_reacquire(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            code = "\n".join(
                [
                    'import os',
                    'from trove_core.vault.coordinator import VaultOperationCoordinator',
                    f"VaultOperationCoordinator({str(cfg.root)!r}).acquire(owner='crash-test')",
                    "print('READY', flush=True)",
                    'os._exit(23)',
                ]
            )
            crashed = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(crashed.returncode, 23, crashed.stderr)
            self.assertEqual(crashed.stdout.strip(), 'READY')

            with self.assertRaises(VaultOperationLocked) as ordinary:
                VaultOperationCoordinator(cfg).acquire(owner='maintain')
            self.assertEqual(ordinary.exception.code, 'vault_writer_marker_recovery_required')

            _, _, grant = self._approved(cfg)
            report = recover_writer_marker(
                cfg,
                legacy_writers_stopped=True,
                approval_grant=grant,
            )
            self.assertEqual(
                report.to_dict(),
                {
                    'ok': True,
                    'code': 'writer_marker_recovered',
                    'recovered': True,
                    'pid_marker_removed': True,
                    'info_marker_removed': True,
                    'temporary_markers_removed': 0,
                    'paths_included': False,
                },
            )
            self.assertNotIn(str(cfg.root), json.dumps(report.to_dict()))
            with VaultOperationCoordinator(cfg).write(owner='maintain') as session:
                self.assertTrue(session.active)

    def test_live_pid_is_never_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            birth = lock_module._process_start_time(os.getpid())
            if birth is None:
                self.skipTest('process birth is unavailable on this platform')
            pid_path, info_path = self._write_marker(
                cfg,
                pid=os.getpid(),
                info={
                    'pid': os.getpid(),
                    'process_start_time': birth,
                },
            )
            pid_identity = (pid_path.stat().st_dev, pid_path.stat().st_ino)
            info_bytes = info_path.read_bytes()
            _, _, grant = self._approved(cfg)

            with self.assertRaises(WriterMarkerRecoveryError) as active:
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(active.exception.code, 'writer_marker_recovery_writer_active')
            self.assertNotIn(str(cfg.root), str(active.exception))
            self.assertEqual((pid_path.stat().st_dev, pid_path.stat().st_ino), pid_identity)
            self.assertEqual(info_path.read_bytes(), info_bytes)

    def test_stable_flock_blocks_recovery_before_marker_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            coordinator = VaultOperationCoordinator(cfg)
            session = coordinator.acquire(owner='sync')
            try:
                pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
                pid_identity = (pid_path.stat().st_dev, pid_path.stat().st_ino)
                _, _, grant = self._approved(cfg)
                with self.assertRaises(WriterMarkerRecoveryError) as active:
                    recover_writer_marker(
                        cfg,
                        legacy_writers_stopped=True,
                        approval_grant=grant,
                    )
                self.assertEqual(active.exception.code, 'writer_marker_recovery_writer_active')
                self.assertEqual((pid_path.stat().st_dev, pid_path.stat().st_ino), pid_identity)
                self.assertTrue(session.active)
            finally:
                coordinator.release(session)

    def test_out_of_range_pid_is_typed_and_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            pid_path.write_text('99999999999999999999\n', encoding='ascii')
            original = pid_path.read_bytes()
            _, _, grant = self._approved(cfg)

            with self.assertRaises(WriterMarkerRecoveryError) as rejected:
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(rejected.exception.code, 'writer_marker_recovery_owner_unverifiable')
            self.assertEqual(pid_path.read_bytes(), original)

    def test_live_reused_pid_with_canonical_birth_mismatch_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            birth = lock_module._process_start_time(os.getpid())
            if birth is None:
                self.skipTest('process birth is unavailable on this platform')
            vault_hash = hashlib.sha256(str(cfg.root.resolve()).encode('utf-8')).hexdigest()
            self._write_marker(
                cfg,
                pid=os.getpid(),
                info={
                    'schema': 1,
                    'pid': os.getpid(),
                    'process_birth_hash': '0' * 64,
                    'owner_nonce': 'a' * 32,
                    'vault_hash': vault_hash,
                },
            )
            _, _, grant = self._approved(cfg)
            report = recover_writer_marker(
                cfg,
                legacy_writers_stopped=True,
                approval_grant=grant,
            )
            self.assertTrue(report.recovered)

    def test_marker_inode_swap_after_grant_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, _ = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            _, _, grant = self._approved(cfg)
            original_path = cfg.paths.logs_dir / 'original-marker.pid'
            original_claim = recovery_app._claim_recovery_grant

            def validate_then_swap(*args, **kwargs):
                original_claim(*args, **kwargs)
                pid_path.rename(original_path)
                pid_path.write_text('99999999\n', encoding='ascii')

            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                patch.object(recovery_app, '_claim_recovery_grant', side_effect=validate_then_swap),
                self.assertRaises(WriterMarkerRecoveryError) as changed,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(changed.exception.code, 'writer_marker_recovery_path_changed')
            self.assertTrue(original_path.exists())
            self.assertTrue(pid_path.exists())

    def test_authority_lock_inode_swap_after_claim_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, _ = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            _, _, grant = self._approved(cfg)
            original_claim = recovery_app._claim_recovery_grant
            saved_lock = cfg.paths.logs_dir / 'saved-authority.flock'

            def validate_then_swap(*args, **kwargs):
                original_claim(*args, **kwargs)
                lock_path = cfg.paths.logs_dir / 'trove-index-writer.flock'
                lock_path.rename(saved_lock)
                lock_path.write_bytes(b'replacement')

            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                patch.object(recovery_app, '_claim_recovery_grant', side_effect=validate_then_swap),
                self.assertRaises(WriterMarkerRecoveryError) as changed,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(changed.exception.code, 'writer_marker_recovery_path_changed')
            self.assertTrue(saved_lock.exists())
            self.assertTrue(pid_path.exists())

    def test_root_retarget_after_claim_fails_closed_on_bound_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            _, _, grant = self._approved(cfg)
            original_claim = recovery_app._claim_recovery_grant
            moved_root = Path(directory) / 'moved-vault'
            replacement_root = Path(directory) / 'replacement-vault'
            replacement_root.mkdir()

            def validate_then_retarget(*args, **kwargs):
                original_claim(*args, **kwargs)
                cfg.root.rename(moved_root)
                cfg.root.symlink_to(replacement_root, target_is_directory=True)

            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                patch.object(recovery_app, '_claim_recovery_grant', side_effect=validate_then_retarget),
                self.assertRaises(WriterMarkerRecoveryError) as changed,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(changed.exception.code, 'writer_marker_recovery_path_changed')
            self.assertTrue((moved_root / 'logs' / 'trove-index-writer.pid').exists())
            self.assertFalse((replacement_root / 'logs').exists())

    def test_only_recognized_two_link_publication_shape_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid = 99_999_999
            pid_path, info_path = self._write_marker(cfg, pid=pid, info={'pid': pid})
            nonce = 'a' * 32
            temp_path = cfg.paths.logs_dir / f'.trove-index-writer.pid.{pid}.{nonce}.tmp'
            os.link(pid_path, temp_path)
            self.assertEqual(pid_path.stat().st_nlink, 2)
            _, _, grant = self._approved(cfg)
            with patch.object(recovery_protocol, '_pid_running', return_value=False):
                report = recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(report.temporary_markers_removed, 1)
            self.assertTrue(report.info_marker_removed)
            self.assertFalse(pid_path.exists())
            self.assertFalse(info_path.exists())
            self.assertFalse(temp_path.exists())

    def test_unrecognized_hardlink_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, info_path = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            outside = Path(directory) / 'outside-marker-link'
            os.link(pid_path, outside)
            _, _, grant = self._approved(cfg)
            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                self.assertRaises(WriterMarkerRecoveryError) as unsafe,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(unsafe.exception.code, 'writer_marker_recovery_marker_unsafe')
            self.assertTrue(pid_path.exists())
            self.assertTrue(info_path.exists())
            self.assertTrue(outside.exists())

    def test_recognized_two_link_info_publication_shape_is_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid = 99_999_999
            pid_path, info_path = self._write_marker(cfg, pid=pid, info={'pid': pid})
            nonce = 'b' * 32
            temp_path = cfg.paths.logs_dir / f'.trove-index-writer.lock.json.{pid}.{nonce}.tmp'
            os.link(info_path, temp_path)
            self.assertEqual(info_path.stat().st_nlink, 2)
            _, _, grant = self._approved(cfg)
            with patch.object(recovery_protocol, '_pid_running', return_value=False):
                report = recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(report.temporary_markers_removed, 1)
            self.assertTrue(report.info_marker_removed)
            self.assertFalse(pid_path.exists())
            self.assertFalse(info_path.exists())
            self.assertFalse(temp_path.exists())

    def test_unrecognized_info_hardlink_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, info_path = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            outside = Path(directory) / 'outside-info-link'
            os.link(info_path, outside)
            _, _, grant = self._approved(cfg)
            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                self.assertRaises(WriterMarkerRecoveryError) as unsafe,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(unsafe.exception.code, 'writer_marker_recovery_marker_unsafe')
            self.assertTrue(pid_path.exists())
            self.assertTrue(info_path.exists())
            self.assertTrue(outside.exists())

    def test_cleanup_fault_leaves_pid_for_safe_retry_and_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, info_path = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            _, _, first_grant = self._approved(cfg)
            original_unlink = recovery_protocol._unlink_identity_at
            failed = False

            def fail_first_info(dir_fd, name, expected, *, require_regular):
                nonlocal failed
                if name == 'trove-index-writer.lock.json' and not failed:
                    failed = True
                    return False
                return original_unlink(
                    dir_fd,
                    name,
                    expected,
                    require_regular=require_regular,
                )

            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                patch.object(recovery_protocol, '_unlink_identity_at', side_effect=fail_first_info),
                self.assertRaises(WriterMarkerRecoveryError) as interrupted,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=first_grant,
                )
            self.assertEqual(interrupted.exception.code, 'writer_marker_recovery_cleanup_incomplete')
            self.assertTrue(pid_path.exists())
            self.assertTrue(info_path.exists())

            _, _, retry_grant = self._approved(cfg)
            with patch.object(recovery_protocol, '_pid_running', return_value=False):
                retried = recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=retry_grant,
                )
            self.assertTrue(retried.recovered)

            _, _, idempotent_grant = self._approved(cfg)
            idempotent = recover_writer_marker(
                cfg,
                legacy_writers_stopped=True,
                approval_grant=idempotent_grant,
            )
            self.assertEqual(idempotent.code, 'writer_marker_absent')
            self.assertFalse(idempotent.recovered)

    def test_fault_after_info_cleanup_still_leaves_pid_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, info_path = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            _, _, first_grant = self._approved(cfg)
            original_unlink = recovery_protocol._unlink_identity_at
            failed_pid = False

            def fail_first_pid(dir_fd, name, expected, *, require_regular):
                nonlocal failed_pid
                if name == 'trove-index-writer.pid' and not failed_pid:
                    failed_pid = True
                    return False
                return original_unlink(
                    dir_fd,
                    name,
                    expected,
                    require_regular=require_regular,
                )

            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                patch.object(recovery_protocol, '_unlink_identity_at', side_effect=fail_first_pid),
                self.assertRaises(WriterMarkerRecoveryError) as interrupted,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=first_grant,
                )
            self.assertEqual(interrupted.exception.code, 'writer_marker_recovery_cleanup_incomplete')
            self.assertTrue(pid_path.exists())
            self.assertFalse(info_path.exists())

            _, _, retry_grant = self._approved(cfg)
            with patch.object(recovery_protocol, '_pid_running', return_value=False):
                retried = recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=retry_grant,
                )
            self.assertTrue(retried.recovered)
            self.assertFalse(pid_path.exists())

    def test_exact_bool_exact_grant_and_validation_precede_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                SENSITIVE_CAPABILITY_INVENTORY[WRITER_MARKER_RECOVERY_ACTION],
                WRITER_MARKER_RECOVERY_DANGER_CLASS,
            )
            cfg = self._cfg(directory)
            pid_path, info_path = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            _, _, grant = self._approved(cfg)

            for invalid in (False, 1, 'true', None):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ApprovalValidationError) as rejected:
                        recover_writer_marker(
                            cfg,
                            legacy_writers_stopped=invalid,  # type: ignore[arg-type]
                            approval_grant=grant,
                        )
                    self.assertEqual(
                        rejected.exception.code,
                        'writer_marker_recovery_confirmation_required',
                    )
            with self.assertRaises(ApprovalValidationError) as lookalike:
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant.to_dict(),  # type: ignore[arg-type]
                )
            self.assertEqual(lookalike.exception.code, 'invalid_grant')

            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                patch.object(
                    ApprovalGrant,
                    'validate_for',
                    side_effect=ApprovalValidationError('replayed grant', code='approval_replayed'),
                ),
                self.assertRaises(ApprovalValidationError) as replayed,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(replayed.exception.code, 'approval_replayed')
            self.assertTrue(pid_path.exists())
            self.assertTrue(info_path.exists())

    def test_consumed_approval_record_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            manager, approval_id, grant = self._approved(cfg)
            payload = writer_marker_recovery_payload(legacy_writers_stopped=True)
            self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            with patch.object(recovery_protocol, '_pid_running', return_value=False):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            replay_pid, _ = self._write_marker(
                cfg,
                pid=99_999_999,
                info={'pid': 99_999_999},
            )
            with (
                patch.object(recovery_protocol, '_pid_running', return_value=False),
                self.assertRaises(ApprovalValidationError) as grant_replayed,
            ):
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=grant,
                )
            self.assertEqual(grant_replayed.exception.code, 'approval_grant_replayed')
            self.assertTrue(replay_pid.exists())

            with self.assertRaises(ApprovalRequired) as replayed:
                manager.consume(
                    WRITER_MARKER_RECOVERY_ACTION,
                    WRITER_MARKER_RECOVERY_DANGER_CLASS,
                    payload,
                    approval_id=approval_id,
                )
            self.assertEqual(replayed.exception.code, 'approval_replayed')

    def test_mismatched_grant_is_rejected_before_authority_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path, _ = self._write_marker(cfg, pid=99_999_999, info={'pid': 99_999_999})
            manager = ApprovalManager(cfg.root)
            wrong = manager.request('reset_index_cache', 'delete_or_purge', {'scope': 'synthetic'})
            manager.decide(wrong.approval_id, 'approved')
            wrong_grant = manager.consume(
                'reset_index_cache',
                'delete_or_purge',
                {'scope': 'synthetic'},
                approval_id=wrong.approval_id,
            )

            with self.assertRaises(ApprovalValidationError) as rejected:
                recover_writer_marker(
                    cfg,
                    legacy_writers_stopped=True,
                    approval_grant=wrong_grant,
                )
            self.assertEqual(rejected.exception.code, 'grant_mismatch')
            self.assertTrue(pid_path.exists())
            self.assertFalse((cfg.paths.logs_dir / 'trove-index-writer.flock').exists())


if __name__ == '__main__':
    unittest.main()
