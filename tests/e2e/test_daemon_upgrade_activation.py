from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from scripts.activate_distribution import (
    ActivationError, activate_distribution, check_upgrade_capacity,
    create_upgrade_backup, install_release,
)
from trove_core.product_config import load_product_config
from tests.distribution_support import distribution_dir


def _second_distribution(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    path = destination / 'distribution-manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['source_git_sha'] = 'f' * 40
    core = {
        key: manifest[key]
        for key in (
            'source_git_sha', 'source_dirty', 'protocol', 'runtime_build_hash',
            'catalog_hash', 'provider_package_hash', 'runtime', 'provider',
        )
    }
    manifest['distribution_set_sha256'] = hashlib.sha256(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode()).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n')
    return path


class DaemonUpgradeActivationTests(unittest.TestCase):
    def test_upgrade_backup_fails_closed_before_consuming_swap_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            index = vault / 'index'
            index.mkdir(parents=True)
            (index / 'trove.sqlite').write_bytes(b'x' * 1024)
            usage = namedtuple('usage', 'total used free')
            with self.assertRaisesRegex(ActivationError, 'insufficient free space'):
                check_upgrade_capacity(
                    vault, root / 'install',
                    disk_usage=lambda _path: usage(100 * 1024**3, 95 * 1024**3, 5 * 1024**3),
                )

    def test_upgrade_backup_is_owner_only_atomic_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            index = vault / 'index'
            manifests = vault / 'manifests'
            index.mkdir(parents=True)
            manifests.mkdir()
            database = index / 'trove.sqlite'
            connection = sqlite3.connect(database)
            connection.execute('CREATE TABLE proof(value TEXT NOT NULL)')
            connection.execute('INSERT INTO proof VALUES (?)', ('preserved',))
            connection.commit()
            connection.close()
            (manifests / 'source.json').write_text('{}\n')
            install = root / 'install'
            usage = namedtuple('usage', 'total used free')
            capacity = check_upgrade_capacity(
                vault, install,
                disk_usage=lambda _path: usage(100 * 1024**3, 10 * 1024**3, 90 * 1024**3),
            )
            for second in range(3):
                report = create_upgrade_backup(
                    vault, install, capacity=capacity,
                    now=datetime(2026, 7, 18, 12, 0, second, tzinfo=timezone.utc),
                    disk_usage=lambda _path: usage(
                        100 * 1024**3, 10 * 1024**3, 90 * 1024**3,
                    ),
                )
                self.assertTrue(report['created'])
            vault_id = hashlib.sha256(str(vault.resolve()).encode()).hexdigest()[:20]
            backups = [path for path in (install / 'backups' / vault_id).iterdir() if path.is_dir()]
            self.assertEqual(len(backups), 2)
            self.assertTrue(all(path.stat().st_mode & 0o777 == 0o700 for path in backups))
            for backup in backups:
                self.assertEqual((backup / 'backup-manifest.json').stat().st_mode & 0o777, 0o600)
                payload = json.loads((backup / 'backup-manifest.json').read_text())
                self.assertFalse(payload['private_paths_included'])
                self.assertFalse(payload['secret_values_included'])
                copied = sqlite3.connect(backup / 'index/trove.sqlite')
                self.assertEqual(copied.execute('SELECT value FROM proof').fetchone(), ('preserved',))
                copied.close()

    def test_same_verified_artifact_install_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def runner(argv, **_kwargs):
                calls.append(tuple(argv))
                if argv[1:3] == ['-m', 'venv']:
                    venv = Path(argv[3])
                    (venv / 'bin').mkdir(parents=True)
                    (venv / 'bin/python').write_text('#!/bin/sh\n')
                    (venv / 'bin/python').chmod(0o700)
                elif 'pip' in argv:
                    venv = Path(argv[0]).parents[1]
                    for name in ('trove', 'trove-mcp', 'troved'):
                        path = venv / 'bin' / name
                        path.write_text('#!/bin/sh\n')
                        path.chmod(0o700)
                from subprocess import CompletedProcess
                return CompletedProcess(argv, 0)

            manifest = distribution_dir() / 'distribution-manifest.json'
            first = install_release(manifest, root, runner=runner)
            call_count = len(calls)
            second = install_release(manifest, root, runner=runner)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), call_count)
            self.assertEqual((first / 'distribution-manifest.json').stat().st_mode & 0o777, 0o600)

    def test_failed_candidate_restores_previous_release_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_manifest = distribution_dir() / 'distribution-manifest.json'
            second_manifest = _second_distribution(distribution_dir(), root / 'candidate')
            install_root = root / 'install'
            config_path = root / 'config/config.json'

            def installer(manifest_path, target):
                manifest = json.loads(manifest_path.read_text())
                release = target / 'releases' / manifest['distribution_set_sha256']
                (release / 'venv/bin').mkdir(parents=True, exist_ok=True)
                return release

            first = activate_distribution(
                first_manifest, install_root, installer=installer,
                health_check=lambda *_args: True, config_path=config_path,
            )
            self.assertTrue(first['ok'])
            previous = (install_root / 'current').resolve()
            config_before = config_path.read_bytes()
            vault = root / 'vault'
            vault.mkdir()
            lifecycle_calls = []

            def runner(argv, **_kwargs):
                lifecycle_calls.append(tuple(argv))
                return subprocess.CompletedProcess(
                    argv, 0,
                    stdout=json.dumps({'ok': True, 'data': {'state': 'compatible'}}) + '\n',
                    stderr='',
                )

            usage = namedtuple('usage', 'total used free')
            capacity = {
                'backup_bytes': 0,
                'free_before_bytes': 90 * 1024**3,
                'reserve_bytes': 15 * 1024**3,
            }
            with (
                patch('scripts.activate_distribution.check_upgrade_capacity', return_value=capacity),
                patch(
                    'scripts.activate_distribution.shutil.disk_usage',
                    return_value=usage(100 * 1024**3, 10 * 1024**3, 90 * 1024**3),
                ),
            ):
                with self.assertRaisesRegex(ActivationError, 'health check failed'):
                    activate_distribution(
                        second_manifest, install_root, installer=installer,
                        health_check=lambda *_args: False, config_path=config_path,
                        vault=vault, runner=runner,
                    )
            self.assertEqual((install_root / 'current').resolve(), previous)
            self.assertEqual(config_path.read_bytes(), config_before)
            self.assertEqual([call[-1] for call in lifecycle_calls], ['stop', 'start'])
            self.assertIn(str(previous), lifecycle_calls[-1][0])

    def test_healthy_candidate_activates_exact_hash_and_owner_only_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _second_distribution(distribution_dir(), root / 'candidate')
            manifest = json.loads(manifest_path.read_text())
            install_root = root / 'install'
            config_path = root / 'config/config.json'
            for index in range(5):
                old = install_root / 'releases' / f'{index:064x}'
                old.mkdir(parents=True)

            def installer(source, target):
                value = json.loads(source.read_text())
                release = target / 'releases' / value['distribution_set_sha256']
                (release / 'venv/bin').mkdir(parents=True, exist_ok=True)
                return release

            report = activate_distribution(
                manifest_path, install_root, installer=installer,
                health_check=lambda _release, candidate, _vault: (
                    candidate['distribution_set_sha256'] == manifest['distribution_set_sha256']
                ),
                config_path=config_path,
            )
            self.assertTrue(report['ok'])
            self.assertEqual((install_root / 'current').resolve().name, manifest['distribution_set_sha256'])
            config = load_product_config(config_path)
            self.assertEqual(config.runtime_root, (install_root / 'current').resolve())
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)
            self.assertLessEqual(len(list((install_root / 'releases').iterdir())), 3)
            self.assertLessEqual(report['retained_releases'], 3)


if __name__ == '__main__':
    unittest.main()
