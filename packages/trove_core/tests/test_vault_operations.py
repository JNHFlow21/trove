from __future__ import annotations
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.vault.operations import reset_index_cache
from trove_core.vault.config import VaultConfig
from trove_core.vault import locks as lock_module
from trove_core.vault.locks import VaultOperationLock, VaultOperationLocked, active_vector_progress

class VaultOperationsTests(unittest.TestCase):
    def test_reset_index_cache_only_removes_target_cache(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            sqlite = vault / 'index' / 'trove.sqlite'
            vector = vault / 'vectors'
            sqlite.parent.mkdir(parents=True)
            vector.mkdir(parents=True)
            sqlite.write_text('x')
            (vector / 'v').write_text('x')
            out = reset_index_cache(vault)
            self.assertIn('trove.sqlite', out['removed'])
            self.assertIn('vectors/', out['removed'])
            self.assertFalse(sqlite.exists())
            self.assertTrue((vault / 'index').exists())

    def test_writer_lock_never_auto_reclaims_dead_pid(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text('99999999\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({'pid': 99999999, 'owner': 'maintain', 'created_at': time.time()}),
                encoding='utf-8',
            )
            with self.assertRaises(VaultOperationLocked) as blocked:
                with VaultOperationLock(cfg, owner='sync'):
                    pass
            self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')
            self.assertEqual((cfg.paths.logs_dir / 'trove-index-writer.pid').read_text(), '99999999\n')

    def test_writer_lock_does_not_reclaim_live_owner(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            current_start = lock_module._process_start_time(os.getpid())
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text(f'{os.getpid()}\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({
                    'pid': os.getpid(),
                    'owner': 'maintain',
                    'created_at': time.time(),
                    'process_start_time': current_start,
                    'owner_nonce': 'live-owner',
                }),
                encoding='utf-8',
            )

            with self.assertRaises(VaultOperationLocked):
                with VaultOperationLock(cfg, owner='sync'):
                    pass
            self.assertEqual(
                (cfg.paths.logs_dir / 'trove-index-writer.lock.json')
                .read_text(encoding='utf-8')
                .count('live-owner'),
                1,
            )

    def test_writer_lock_treats_permission_error_pid_as_running(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text('424242\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({'pid': 424242, 'owner': 'maintain', 'created_at': time.time()}),
                encoding='utf-8',
            )
            with patch.object(lock_module.os, 'kill', side_effect=PermissionError):
                with self.assertRaises(VaultOperationLocked):
                    with VaultOperationLock(cfg, owner='sync'):
                        pass
            self.assertEqual(
                (cfg.paths.logs_dir / 'trove-index-writer.pid').read_text(encoding='utf-8').strip(),
                '424242',
            )

    def test_writer_lock_never_auto_reclaims_reused_pid(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text(f'{os.getpid()}\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({
                    'pid': os.getpid(),
                    'owner': 'old-writer',
                    'created_at': time.time(),
                    'process_start_time': 'old-process-start',
                    'owner_nonce': 'old-owner',
                }),
                encoding='utf-8',
            )
            with patch.object(lock_module, '_process_start_time', return_value='new-process-start'):
                with self.assertRaises(VaultOperationLocked) as blocked:
                    with VaultOperationLock(cfg, owner='sync', owner_nonce='a' * 32):
                        pass
            self.assertEqual(blocked.exception.code, 'vault_writer_marker_recovery_required')
            self.assertEqual(
                json.loads(
                    (cfg.paths.logs_dir / 'trove-index-writer.lock.json').read_text(encoding='utf-8')
                )['owner_nonce'],
                'old-owner',
            )

    def test_writer_lock_facade_transmits_protocol_valid_owner_nonce(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            nonce = 'a' * 32
            with VaultOperationLock(cfg, owner='sync', owner_nonce=nonce) as lock:
                payload = json.loads((cfg.paths.logs_dir / 'trove-index-writer.lock.json').read_text(encoding='utf-8'))
                self.assertEqual(payload['owner_nonce'], nonce)
                self.assertEqual(payload['owner_nonce'], lock.owner_nonce)

    def test_writer_lock_release_requires_owner_nonce(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            first = VaultOperationLock(cfg, owner='sync', owner_nonce='1' * 32)
            first.__enter__()
            try:
                VaultOperationLock(cfg, owner='sync', owner_nonce='2' * 32).release()
                self.assertTrue((cfg.paths.logs_dir / 'trove-index-writer.pid').exists())
            finally:
                first.release()
            self.assertFalse((cfg.paths.logs_dir / 'trove-index-writer.pid').exists())

    def test_active_vector_progress_ignores_dead_writer_pid(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            progress = cfg.paths.vector_dir / 'zvec' / 'messages.trove-progress.json'
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps({'state': 'running', 'updated_at': time.time()}), encoding='utf-8')
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text('99999999\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({'pid': 99999999, 'owner': 'maintain', 'created_at': time.time()}),
                encoding='utf-8',
            )
            self.assertIsNone(active_vector_progress(cfg))

    def test_active_vector_progress_ignores_reused_writer_pid(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = VaultConfig.resolve(str(Path(d) / 'vault'), env={})
            cfg.ensure()
            progress = cfg.paths.vector_dir / 'zvec' / 'messages.trove-progress.json'
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(json.dumps({'state': 'running', 'updated_at': time.time()}), encoding='utf-8')
            cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            (cfg.paths.logs_dir / 'trove-index-writer.pid').write_text(f'{os.getpid()}\n', encoding='utf-8')
            (cfg.paths.logs_dir / 'trove-index-writer.lock.json').write_text(
                json.dumps({
                    'pid': os.getpid(),
                    'owner': 'maintain',
                    'created_at': time.time(),
                    'process_start_time': 'old-process-start',
                    'owner_nonce': 'old-owner',
                }),
                encoding='utf-8',
            )
            with patch.object(lock_module, '_process_start_time', return_value='new-process-start'):
                self.assertIsNone(active_vector_progress(cfg))
