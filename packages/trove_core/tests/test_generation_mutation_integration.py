from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from trove_core.runtime import SearchRuntimeCache
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.repositories import EntityRecord, MultimodalRepository
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import VaultGenerationUnavailable, vault_generation_read
from trove_core.vault.operations import initialize_index, purge_derived_data, rebuild_scope, reset_index_cache
from trove_core.wechat.models import Account, Conversation, Message


class GenerationMutationIntegrationTests(unittest.TestCase):
    @staticmethod
    def _wait_for_writer(cfg: VaultConfig, done: threading.Event) -> bool:
        marker = cfg.paths.logs_dir / 'trove-index-writer.pid'
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if marker.exists():
                return True
            if done.wait(0.01):
                return False
        return marker.exists()

    def test_reset_waits_for_active_reader_then_publishes_complete_empty_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'vault'
            initialize_index(root)
            cfg = VaultConfig.resolve(str(root), env={})
            sqlite_path = cfg.paths.sqlite_path.resolve()
            old_identity = (sqlite_path.stat().st_dev, sqlite_path.stat().st_ino)
            done = threading.Event()
            reports: list[dict] = []
            errors: list[BaseException] = []

            with vault_generation_read(cfg) as old_token:
                def reset() -> None:
                    try:
                        reports.append(reset_index_cache(root))
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=reset, daemon=True)
                worker.start()
                self.assertTrue(self._wait_for_writer(cfg, done), errors)
                time.sleep(0.05)
                self.assertFalse(done.is_set(), 'reset crossed an active generation read')
                self.assertEqual((sqlite_path.stat().st_dev, sqlite_path.stat().st_ino), old_identity)
                self.assertEqual(old_token.sqlite[1:3], old_identity)

            self.assertTrue(done.wait(3.0))
            worker.join(3.0)
            self.assertFalse(errors)
            self.assertIn('trove.sqlite', reports[0]['removed'])
            self.assertFalse((root / '.trove-generation-publish.json').exists())
            with vault_generation_read(cfg) as current:
                self.assertEqual(current.sqlite[0], 'missing')

    def test_reset_fault_blocks_reads_until_idempotent_same_operation_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'vault'
            initialize_index(root)
            cfg = VaultConfig.resolve(str(root), env={})
            sqlite_path = cfg.paths.sqlite_path.resolve()
            original_unlink = Path.unlink
            faulted = False

            def crash_after_sqlite_unlink(path: Path, *args, **kwargs):
                nonlocal faulted
                result = original_unlink(path, *args, **kwargs)
                if not faulted and path == sqlite_path:
                    faulted = True
                    raise RuntimeError('simulated reset publication crash')
                return result

            with patch.object(Path, 'unlink', new=crash_after_sqlite_unlink):
                with self.assertRaisesRegex(RuntimeError, 'simulated reset publication crash'):
                    reset_index_cache(root)

            self.assertTrue(faulted)
            self.assertTrue((root / '.trove-generation-publish.json').exists())
            with self.assertRaises(VaultGenerationUnavailable) as blocked:
                with vault_generation_read(cfg):
                    pass
            self.assertEqual(blocked.exception.code, 'vault_generation_recovery_required')
            with self.assertRaises(VaultGenerationUnavailable) as wrong_retry:
                rebuild_scope(root)
            self.assertEqual(wrong_retry.exception.code, 'vault_generation_recovery_required')

            reset_index_cache(root)
            self.assertFalse((root / '.trove-generation-publish.json').exists())
            with vault_generation_read(cfg) as current:
                self.assertEqual(current.sqlite[0], 'missing')
            self.assertEqual(reset_index_cache(root)['removed'], [])

    def test_scope_rebuild_waits_for_reader_and_runtime_invalidates_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'vault'
            initialize_index(root)
            cfg = VaultConfig.resolve(str(root), env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            try:
                store.upsert_accounts([Account('acct-generation', 'Generation', 'Generation')])
                store.upsert_conversations(
                    [Conversation('gh_generation', 'acct-generation', 'Excluded source', 'private')]
                )
                store.upsert_messages(
                    [
                        Message(
                            'acct-generation',
                            'Generation',
                            'gh_generation',
                            'Excluded source',
                            'private',
                            'gh_generation',
                            'Excluded',
                            datetime(2026, 1, 1, tzinfo=timezone.utc),
                            'scopepurgesentinel',
                            'generation',
                            1,
                        )
                    ]
                )
            finally:
                store.close()

            cache = SearchRuntimeCache(cfg, provider_factory=lambda: None)
            request = SearchRequest('scopepurgesentinel', limit=5, semantic='off')
            self.assertTrue(cache.search(request).results)
            initial_generation = cache.generation
            done = threading.Event()
            reports: list[dict] = []
            errors: list[BaseException] = []

            with vault_generation_read(cfg):
                def rebuild() -> None:
                    try:
                        reports.append(rebuild_scope(root))
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=rebuild, daemon=True)
                worker.start()
                self.assertTrue(self._wait_for_writer(cfg, done), errors)
                time.sleep(0.05)
                self.assertFalse(done.is_set(), 'scope rebuild crossed an active generation read')
                with sqlite3.connect(cfg.paths.sqlite_path) as connection:
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM messages WHERE content='scopepurgesentinel'"
                        ).fetchone()[0],
                        1,
                    )

            self.assertTrue(done.wait(3.0))
            worker.join(3.0)
            self.assertFalse(errors)
            self.assertEqual(reports[0]['purged_conversations'], 1)
            with sqlite3.connect(cfg.paths.sqlite_path) as connection:
                self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

            self.assertFalse(cache.search(request).results)
            self.assertEqual(cache.generation, initial_generation + 1)
            self.assertFalse(cache.search(request).results)
            self.assertEqual(cache.generation, initial_generation + 1)
            cache.close()

    def test_derived_data_purge_waits_for_active_generation_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'vault'
            initialize_index(root)
            cfg = VaultConfig.resolve(str(root), env={})
            MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path)).upsert_entity(EntityRecord(
                'customer-generation-purge', 'Customer', 'Generation Purge',
                {'wechat_id': 'wxid-generation-purge'},
            ))
            done = threading.Event()
            reports: list[dict] = []
            errors: list[BaseException] = []

            with vault_generation_read(cfg):
                def purge() -> None:
                    try:
                        reports.append(purge_derived_data(
                            root, scope_type='entity', scope_id='customer-generation-purge',
                        ))
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=purge, daemon=True)
                worker.start()
                self.assertTrue(self._wait_for_writer(cfg, done), errors)
                time.sleep(0.05)
                self.assertFalse(done.is_set(), 'derived purge crossed an active generation read')
                with sqlite3.connect(cfg.paths.sqlite_path) as connection:
                    self.assertEqual(connection.execute(
                        "SELECT COUNT(*) FROM entities WHERE entity_id='customer-generation-purge'"
                    ).fetchone()[0], 1)

            self.assertTrue(done.wait(3.0))
            worker.join(3.0)
            self.assertFalse(errors)
            self.assertTrue(reports[0]['ok'])
            with sqlite3.connect(cfg.paths.sqlite_path) as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM entities WHERE entity_id='customer-generation-purge'"
                ).fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
