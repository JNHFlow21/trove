from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import (
    VaultGenerationLease,
    VaultGenerationUnavailable,
    coordinated_vault_generation_publish,
    vault_generation_publish,
    vault_generation_read,
)
from trove_core.wechat.indexer import index_fixture_vault


class VaultGenerationLeaseTests(unittest.TestCase):
    def test_publisher_waits_for_old_reader_then_exposes_complete_new_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(str(root), env={})
            sqlite_path = cfg.paths.sqlite_path
            old_identity = (sqlite_path.stat().st_dev, sqlite_path.stat().st_ino)
            started = threading.Event()
            published = threading.Event()
            errors: list[BaseException] = []

            with vault_generation_read(cfg) as old_token:
                def replace_generation() -> None:
                    try:
                        started.set()
                        with vault_generation_publish(cfg) as lease:
                            replacement = cfg.paths.index_dir / "replacement.sqlite"
                            with sqlite3.connect(replacement) as conn:
                                conn.execute("CREATE TABLE generation(value TEXT NOT NULL)")
                                conn.execute("INSERT INTO generation VALUES('new-complete')")
                                conn.commit()
                                self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                            os.replace(replacement, sqlite_path)
                            lease.refresh_token()
                        published.set()
                    except BaseException as exc:  # pragma: no cover - surfaced below
                        errors.append(exc)
                        published.set()

                worker = threading.Thread(target=replace_generation, daemon=True)
                worker.start()
                self.assertTrue(started.wait(2.0))
                time.sleep(0.05)
                self.assertFalse(published.is_set(), "publisher replaced a generation with a read lease active")
                self.assertEqual((sqlite_path.stat().st_dev, sqlite_path.stat().st_ino), old_identity)
                self.assertEqual(old_token.sqlite[1:3], old_identity)

            self.assertTrue(published.wait(2.0))
            worker.join(2.0)
            self.assertFalse(errors)
            with vault_generation_read(cfg) as new_token:
                self.assertNotEqual(new_token.sqlite[1:3], old_identity)
                with sqlite3.connect(sqlite_path) as conn:
                    self.assertEqual(conn.execute("SELECT value FROM generation").fetchone()[0], "new-complete")
                    self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_read_lease_is_metadata_only_and_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            with vault_generation_read(VaultConfig.resolve(str(root), env={})) as token:
                self.assertEqual(token.sqlite[0], "file")
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            self.assertEqual(after, before)

    def test_nested_read_reuses_outer_lease_until_complete_logical_read_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(str(root), env={})
            with vault_generation_read(cfg) as outer:
                with vault_generation_read(cfg) as inner:
                    self.assertIs(inner, outer)
                    self.assertEqual(inner.cache_key(), outer.cache_key())
                # An inner helper exiting must not release the outer barrier.
                with vault_generation_read(cfg) as after_inner:
                    self.assertIs(after_inner, outer)

    def test_nested_publication_reuses_outer_lease_and_one_recovery_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            cfg = VaultConfig.resolve(str(root), env={})
            marker = root / ".trove-generation-publish.json"
            with coordinated_vault_generation_publish(cfg, operation="full-import") as outer:
                self.assertTrue(marker.exists())
                with coordinated_vault_generation_publish(cfg, operation="reset-index") as nested:
                    self.assertIs(nested, outer)
                    self.assertTrue(marker.exists())
            self.assertFalse(marker.exists())

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork semantics")
    def test_crashed_reader_releases_kernel_lease_for_idempotent_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(str(root), env={})
            ready_read, ready_write = os.pipe()
            exit_read, exit_write = os.pipe()
            pid = os.fork()
            if pid == 0:  # pragma: no cover - child process
                try:
                    os.close(ready_read)
                    os.close(exit_write)
                    lease = VaultGenerationLease(cfg, mode="read").acquire()
                    os.write(ready_write, b"1")
                    os.read(exit_read, 1)
                    # Simulate SIGKILL-style cleanup: no Python __exit__ runs.
                    os._exit(77)
                except BaseException:
                    os._exit(78)

            os.close(ready_write)
            os.close(exit_read)
            try:
                self.assertEqual(os.read(ready_read, 1), b"1")
                acquired = threading.Event()

                def publish_after_crash() -> None:
                    with vault_generation_publish(cfg):
                        acquired.set()

                worker = threading.Thread(target=publish_after_crash, daemon=True)
                worker.start()
                time.sleep(0.05)
                self.assertFalse(acquired.is_set())
                os.write(exit_write, b"x")
                _, status = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 77)
                self.assertTrue(acquired.wait(2.0))
                worker.join(2.0)
                with sqlite3.connect(cfg.paths.sqlite_path) as conn:
                    self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            finally:
                for fd in (ready_read, exit_write):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    waited, _ = os.waitpid(pid, os.WNOHANG)
                    if waited == 0:
                        os.kill(pid, 9)
                        os.waitpid(pid, 0)
                except (ChildProcessError, ProcessLookupError):
                    pass

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork semantics")
    def test_crashed_publisher_blocks_reads_until_same_operation_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_fixture_vault(root, reset=True)
            cfg = VaultConfig.resolve(str(root), env={})
            pid = os.fork()
            if pid == 0:  # pragma: no cover - child process
                try:
                    with vault_generation_publish(cfg, operation="reset-index"):
                        (root / "index" / "partial-reset").write_text("pending", encoding="ascii")
                        os._exit(79)
                except BaseException:
                    os._exit(80)

            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 79)
            marker = root / ".trove-generation-publish.json"
            self.assertTrue(marker.exists())
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(VaultGenerationUnavailable) as blocked:
                with vault_generation_read(cfg):
                    pass
            self.assertEqual(blocked.exception.code, "vault_generation_recovery_required")
            with self.assertRaises(VaultGenerationUnavailable) as wrong_retry:
                with vault_generation_publish(cfg, operation="scope-rebuild"):
                    pass
            self.assertEqual(wrong_retry.exception.code, "vault_generation_recovery_required")

            with vault_generation_publish(cfg, operation="reset-index"):
                (root / "index" / "partial-reset").unlink(missing_ok=True)
                with sqlite3.connect(cfg.paths.sqlite_path) as connection:
                    self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

            self.assertFalse(marker.exists())
            with vault_generation_read(cfg) as token:
                self.assertEqual(token.sqlite[0], "file")

    def test_symlink_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            real = parent / "real"
            real.mkdir()
            link = parent / "link"
            link.symlink_to(real, target_is_directory=True)
            cfg = VaultConfig.resolve(str(link), env={})
            with self.assertRaises(VaultGenerationUnavailable) as error:
                VaultGenerationLease(cfg).acquire()
            self.assertEqual(error.exception.code, "vault_generation_root_unsafe")

if __name__ == "__main__":
    unittest.main()
