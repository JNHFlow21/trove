from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trove_cli.main import main
from trove_core.application import writer_recovery as recovery_app
from trove_core.application.writer_recovery import (
    WRITER_MARKER_RECOVERY_ACTION,
    WRITER_MARKER_RECOVERY_DANGER_CLASS,
    writer_marker_recovery_payload,
)
from trove_core.approvals import ApprovalGrant, ApprovalManager
from trove_core.vault.config import VaultConfig
from trove_core.vault.writer_recovery import WriterMarkerRecoveryResult


class WriterMarkerRecoveryCliTests(unittest.TestCase):
    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = int(exc.code)
        return code, stdout.getvalue(), stderr.getvalue()

    def _cfg(self, directory: str) -> VaultConfig:
        cfg = VaultConfig.resolve(str(Path(directory) / 'synthetic-vault'), env={})
        cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def _approved_id(self, cfg: VaultConfig) -> str:
        payload = writer_marker_recovery_payload(legacy_writers_stopped=True)
        manager = ApprovalManager(cfg.root)
        record = manager.request(
            WRITER_MARKER_RECOVERY_ACTION,
            WRITER_MARKER_RECOVERY_DANGER_CLASS,
            payload,
        )
        manager.decide(record.approval_id, 'approved')
        return record.approval_id

    def test_cli_requires_confirmation_flag_and_approval_id(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            pid_path.write_text('99999999\n', encoding='ascii')

            for argv in (
                ['--vault', str(cfg.root), 'writer-marker-recovery', '--legacy-writers-stopped'],
                ['--vault', str(cfg.root), 'writer-marker-recovery', '--approval-id', 'appr-0000000000000000'],
                [
                    '--vault',
                    str(cfg.root),
                    'writer-marker-recovery',
                    '--legacy-writers-stopped=true',
                    '--approval-id',
                    'appr-0000000000000000',
                ],
            ):
                with self.subTest(argv=argv):
                    code, out, _ = self.run_cli(list(argv))
                    self.assertEqual(code, 2)
                    self.assertEqual(out, '')
                    self.assertTrue(pid_path.exists())

    def test_cli_requests_exact_payload_then_recovers_without_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            pid_path = cfg.paths.logs_dir / 'trove-index-writer.pid'
            info_path = cfg.paths.logs_dir / 'trove-index-writer.lock.json'
            pid_path.write_text('99999999\n', encoding='ascii')
            info_path.write_text(json.dumps({'pid': 99_999_999}), encoding='utf-8')
            payload = {'legacy_writers_stopped': True}

            code, out, _ = self.run_cli(
                [
                    '--vault',
                    str(cfg.root),
                    'approval-request',
                    '--action',
                    WRITER_MARKER_RECOVERY_ACTION,
                    '--danger-class',
                    WRITER_MARKER_RECOVERY_DANGER_CLASS,
                    '--payload-json',
                    json.dumps(payload),
                    '--json',
                ]
            )
            self.assertEqual(code, 3)
            approval_id = json.loads(out)['approval']['approval_id']
            code, _, _ = self.run_cli(
                [
                    '--vault',
                    str(cfg.root),
                    'approval-decision',
                    approval_id,
                    '--status',
                    'approved',
                    '--json',
                ]
            )
            self.assertEqual(code, 0)
            code, out, _ = self.run_cli(
                [
                    '--vault',
                    str(cfg.root),
                    'writer-marker-recovery',
                    '--approval-id',
                    approval_id,
                    '--legacy-writers-stopped',
                    '--json',
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(out)
            self.assertEqual(report['code'], 'writer_marker_recovered')
            self.assertFalse(report['paths_included'])
            self.assertNotIn(str(cfg.root), out)
            self.assertFalse(pid_path.exists())
            self.assertFalse(info_path.exists())

    def test_cli_invokes_only_the_application_command_with_authentic_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            approval_id = self._approved_id(cfg)
            expected = WriterMarkerRecoveryResult(
                code='writer_marker_absent',
                recovered=False,
                pid_marker_removed=False,
                info_marker_removed=False,
                temporary_markers_removed=0,
            )
            with patch.object(recovery_app, 'recover_writer_marker', return_value=expected) as command:
                code, out, _ = self.run_cli(
                    [
                        '--vault',
                        str(cfg.root),
                        'writer-marker-recovery',
                        '--approval-id',
                        approval_id,
                        '--legacy-writers-stopped',
                        '--json',
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)['code'], 'writer_marker_absent')
            command.assert_called_once()
            call = command.call_args
            self.assertEqual(call.args[0].root, cfg.root)
            self.assertIs(call.kwargs['legacy_writers_stopped'], True)
            self.assertIs(type(call.kwargs['approval_grant']), ApprovalGrant)

    def test_approval_request_rejects_non_object_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._cfg(directory)
            code, out, _ = self.run_cli(
                [
                    '--vault',
                    str(cfg.root),
                    'approval-request',
                    '--action',
                    WRITER_MARKER_RECOVERY_ACTION,
                    '--danger-class',
                    WRITER_MARKER_RECOVERY_DANGER_CLASS,
                    '--payload-json',
                    'true',
                    '--json',
                ]
            )
            self.assertEqual(code, 2)
            self.assertEqual(out, '')


if __name__ == '__main__':
    unittest.main()
