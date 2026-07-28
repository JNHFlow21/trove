from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.schedule import MAINTAIN_LABEL, SAFETY_NOTE, SYNC_LABEL, DualAccountScheduleOptions, ScheduleInstallOptions, bootstrap_launch_agents, install_dual_account_schedule, install_schedule, parse_interval_seconds, uninstall_schedule


class ScheduleTests(unittest.TestCase):
    def test_dual_account_schedule_is_short_lived_and_has_no_watcher_or_maintainer(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            out = root / 'launchd'
            runtime = root / 'current' / 'venv' / 'bin' / 'python'
            report = install_dual_account_schedule(
                vault,
                options=DualAccountScheduleOptions(
                    sync_interval='5m',
                    runtime_python=runtime,
                    dry_run=True,
                    output_dir=out,
                ),
            )
            self.assertTrue(report['ok'])
            self.assertEqual(report['labels'], [SYNC_LABEL])
            self.assertEqual(report['files'], [f'{SYNC_LABEL}.plist'])
            payload = plistlib.loads((out / f'{SYNC_LABEL}.plist').read_bytes())
            self.assertEqual(payload['StartInterval'], 300)
            self.assertNotIn('KeepAlive', payload)
            self.assertIn('trove_core.jobs.dual_account_sync', payload['ProgramArguments'])
            self.assertFalse((out / f'{MAINTAIN_LABEL}.plist').exists())

    def test_install_dry_run_writes_readonly_launchd_plists(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            out = Path(d) / 'plists'
            report = install_schedule(vault, options=ScheduleInstallOptions(sync_interval='2h', maintain_at='04:15', dry_run=True, output_dir=out))
            self.assertTrue(report['ok'])
            self.assertTrue(report['dry_run'])
            self.assertFalse(report['installed'])
            self.assertEqual(set(report['labels']), {SYNC_LABEL, MAINTAIN_LABEL})
            sync_text = (out / f'{SYNC_LABEL}.plist').read_text(encoding='utf-8')
            maintain_text = (out / f'{MAINTAIN_LABEL}.plist').read_text(encoding='utf-8')
            self.assertIn(SAFETY_NOTE, sync_text)
            self.assertIn(SAFETY_NOTE, maintain_text)
            self.assertNotIn('wechat_sender', sync_text + maintain_text)
            self.assertNotIn('auto_reply', sync_text + maintain_text)
            sync = plistlib.loads(sync_text.encode('utf-8'))
            maintain = plistlib.loads(maintain_text.encode('utf-8'))
            self.assertEqual(sync['Label'], SYNC_LABEL)
            self.assertEqual(sync['StartInterval'], 7200)
            self.assertIn('sync', sync['ProgramArguments'])
            self.assertEqual(maintain['StartCalendarInterval'], {'Hour': 4, 'Minute': 15})
            self.assertIn('maintain', maintain['ProgramArguments'])
            self.assertNotIn(str(vault), str(report))

    def test_watch_mode_uses_keepalive_not_interval(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            out = Path(d) / 'plists'
            install_schedule(vault, options=ScheduleInstallOptions(watch=True, dry_run=True, output_dir=out))
            sync = plistlib.loads((out / f'{SYNC_LABEL}.plist').read_bytes())
            self.assertTrue(sync['KeepAlive'])
            self.assertNotIn('StartInterval', sync)
            self.assertIn('--watch', sync['ProgramArguments'])

    def test_realtime_bridge_schedule_uses_private_config_and_disables_live_media_scan(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            out = Path(d) / 'plists'
            config = vault / 'jobs' / 'realtime_bridge.private.json'
            install_schedule(
                vault,
                options=ScheduleInstallOptions(
                    watch=True,
                    realtime_config=config,
                    dry_run=True,
                    output_dir=out,
                ),
            )
            sync = plistlib.loads((out / f'{SYNC_LABEL}.plist').read_bytes())
            self.assertIn('realtime-sync', sync['ProgramArguments'])
            self.assertIn('--config', sync['ProgramArguments'])
            self.assertIn(str(config), sync['ProgramArguments'])
            self.assertIn('--watch', sync['ProgramArguments'])
            self.assertEqual(sync['EnvironmentVariables']['TROVE_SYNC_SNAPSHOT_MEDIA_ENABLED'], '0')
            self.assertTrue(sync['KeepAlive'])

    def test_interval_parser(self):
        self.assertEqual(parse_interval_seconds('1h'), 3600)
        self.assertEqual(parse_interval_seconds('90m'), 5400)
        with self.assertRaises(ValueError):
            parse_interval_seconds('10s')

    def test_uninstall_reports_launchctl_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / f'{SYNC_LABEL}.plist').write_text('sync', encoding='utf-8')
            with patch('trove_core.schedule.bootout_launch_agents', return_value={'ok': False, 'ran': True, 'commands': [{'ok': False, 'returncode': 1}]}):
                report = uninstall_schedule(dry_run=False, output_dir=out)

            self.assertFalse(report['ok'])
            self.assertTrue(report['launchctl']['ran'])
            self.assertFalse(report['installed'])

    def test_bootstrap_reenables_and_reloads_existing_launch_agents(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f'{SYNC_LABEL}.plist'
            path.write_text('fixture', encoding='utf-8')
            completed = [
                __import__('subprocess').CompletedProcess([], 0, '', ''),
                __import__('subprocess').CompletedProcess([], 3, '', ''),
                __import__('subprocess').CompletedProcess([], 0, '', ''),
            ]
            with patch('trove_core.schedule.subprocess.run', side_effect=completed) as run:
                report = bootstrap_launch_agents([path])

            self.assertTrue(report['ok'])
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(commands[0][:2], ['launchctl', 'enable'])
            self.assertEqual(commands[1][:2], ['launchctl', 'bootout'])
            self.assertEqual(commands[2][:2], ['launchctl', 'bootstrap'])

    def test_bootstrap_retries_transient_launchd_io_error_after_bootout(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f'{SYNC_LABEL}.plist'
            path.write_text('fixture', encoding='utf-8')
            completed = [
                __import__('subprocess').CompletedProcess([], 0, '', ''),
                __import__('subprocess').CompletedProcess([], 0, '', ''),
                __import__('subprocess').CompletedProcess([], 5, '', 'transient'),
                __import__('subprocess').CompletedProcess([], 0, '', ''),
            ]
            with patch('trove_core.schedule.subprocess.run', side_effect=completed) as run, \
                 patch('trove_core.schedule.time.sleep') as sleep:
                report = bootstrap_launch_agents([path])

            self.assertTrue(report['ok'])
            self.assertEqual(run.call_count, 4)
            sleep.assert_called_once()
            self.assertEqual(report['commands'][0]['returncode'], 0)
            self.assertEqual(report['commands'][0]['bootstrap_attempts'], 2)


if __name__ == '__main__':
    unittest.main()
