from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.migrate_installed_consumers import (
    EXPECTED_TOOL,
    LEGACY_TOOL_ID,
    TOOL_ID,
    apply,
    audit,
    discover_installed_consumers,
    migrated_config,
)


def _legacy_config() -> dict:
    return {
        'version': 1,
        'tools': [
            {'id': 'agent-other', 'command': 'other', 'args': [], 'requiredSecrets': []},
            {
                'id': LEGACY_TOOL_ID,
                'name': 'Legacy source adapter',
                'command': 'bash',
                'args': ['-lc', 'cd /private/source && python -m legacy.server'],
                'requiredSecrets': ['LEGACY_SECRET'],
            },
        ],
    }


class InstalledConsumerMigrationTests(unittest.TestCase):
    def test_audit_is_redacted_and_requires_legacy_migration(self):
        report = audit(_legacy_config())
        self.assertFalse(report['ok'])
        self.assertTrue(report['requires_migration'])
        self.assertEqual(report['legacy_entrypoint_references'], 1)
        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn('/private/source', encoded)
        self.assertNotIn('LEGACY_SECRET', encoded)

    def test_migration_is_exact_and_preserves_unrelated_tools(self):
        migrated = migrated_config(_legacy_config())
        self.assertEqual(migrated['tools'][0]['id'], 'agent-other')
        replacement = migrated['tools'][1]
        self.assertEqual(replacement, EXPECTED_TOOL)
        self.assertEqual(audit(migrated)['legacy_entrypoint_references'], 0)

    def test_installed_consumer_discovery_returns_counts_not_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            launch = home / 'Library/LaunchAgents'
            launch.mkdir(parents=True)
            (launch / 'com.trove.wechat.sync.plist').write_text('private source path', encoding='utf-8')
            skill = home / '.agents/skills/trove-chat-recall'
            skill.mkdir(parents=True)

            def runner(argv, **_kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout='0 * * * * trove_cli legacy\n')

            counts, paths = discover_installed_consumers(home=home, runner=runner)
            self.assertEqual(counts, {
                'legacy_launch_agents': 1,
                'legacy_schedule_references': 1,
                'legacy_generated_skill_links': 1,
            })
            report = audit(migrated_config(_legacy_config()), installed_counts=counts)
            self.assertEqual(report['legacy_entrypoint_references'], 3)
            self.assertNotIn(str(home), json.dumps(report))
            self.assertEqual(paths, (launch / 'com.trove.wechat.sync.plist',))

    def test_apply_runs_doctor_before_atomic_write_then_reconciles_and_syncs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps(_legacy_config()), encoding='utf-8')
            executable = root / 'trove-mcp'
            executable.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            executable.chmod(0o700)
            skillctl = root / 'skillctl'
            skillctl.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            skillctl.chmod(0o700)
            calls: list[tuple[str, ...]] = []

            def runner(argv, **_kwargs):
                calls.append(tuple(argv))
                if argv[1] == 'doctor':
                    self.assertEqual(json.loads(config.read_text())['tools'][1]['command'], 'bash')
                return subprocess.CompletedProcess(argv, 0)

            report = apply(config, runner=runner, executable=str(executable), skillctl=skillctl)
            self.assertTrue(report['ok'])
            self.assertEqual(calls[0], ('agent-switch', 'doctor'))
            self.assertEqual(calls[1], ('agent-switch', 'reconcile'))
            self.assertEqual(calls[2], (str(skillctl), 'sync', 'trove', '--prune'))
            installed = json.loads(config.read_text(encoding='utf-8'))
            tool = next(item for item in installed['tools'] if item['id'] == TOOL_ID)
            self.assertEqual(tool['command'], str(executable.resolve()))
            self.assertEqual(os.stat(config).st_mode & 0o777, 0o600)

    def test_failed_executable_validation_does_not_change_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            before = json.dumps(_legacy_config(), sort_keys=True)
            config.write_text(before, encoding='utf-8')
            calls = []

            def runner(argv, **_kwargs):
                calls.append(tuple(argv))
                return subprocess.CompletedProcess(argv, 0)

            with self.assertRaises(RuntimeError):
                apply(config, runner=runner, executable=str(root / 'missing'))
            self.assertEqual(calls, [('agent-switch', 'doctor')])
            self.assertEqual(json.dumps(json.loads(config.read_text()), sort_keys=True), before)

    def test_external_schedule_blocks_apply_before_config_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            before = json.dumps(_legacy_config(), sort_keys=True)
            config.write_text(before, encoding='utf-8')
            executable = root / 'trove-mcp'
            executable.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            executable.chmod(0o700)

            def runner(argv, **_kwargs):
                return subprocess.CompletedProcess(argv, 0)

            with self.assertRaises(RuntimeError):
                apply(
                    config, runner=runner, executable=str(executable),
                    legacy_schedule_references=1,
                )
            self.assertEqual(json.dumps(json.loads(config.read_text()), sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
