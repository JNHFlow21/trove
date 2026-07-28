from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.wechat import fixture_guard as fixture_guard_module
from trove_core.wechat.fixture_guard import (
    FIXTURE_MARKER_BYTES,
    FIXTURE_MARKER_NAME,
    FIXTURE_GENERATION_STATE_NAME,
    FIXTURE_READY_NAME,
    FixtureVaultGuardError,
    prepare_fixture_vault,
)
from trove_core.wechat.fixture_factory import generate_fixture
from trove_core.wechat.indexer import index_fixture_data, index_fixture_vault
from trove_core.store.repositories import WeChatRepository


class FixtureVaultGuardTests(unittest.TestCase):
    def test_active_fixture_claim_fails_closed_without_waiting_or_mutating(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'fixture'
            vault.mkdir()
            root_fd = os.open(vault, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(FixtureVaultGuardError) as caught:
                    index_fixture_vault(vault, reset=True)
                self.assertEqual(caught.exception.reason_code, 'fixture_claim_locked')
                self.assertEqual(list(vault.iterdir()), [])
            finally:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
                os.close(root_fd)

    def test_nonempty_unmarked_root_fails_before_store_or_file_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "real-vault"
            sqlite_path = vault / "index" / "trove.sqlite"
            sqlite_path.parent.mkdir(parents=True)
            sqlite_path.write_bytes(b"real-vault-sentinel")
            note = vault / "private-note.txt"
            note.write_bytes(b"private-sentinel")
            before = self._snapshot(vault)

            for reset in (False, True):
                with self.subTest(reset=reset):
                    with patch("trove_core.wechat.indexer.SQLiteStore") as store:
                        with self.assertRaises(FixtureVaultGuardError) as caught:
                            index_fixture_vault(vault, reset=reset)
                    store.assert_not_called()
                    self.assertEqual(caught.exception.reason_code, "fixture_marker_missing_nonempty_root")
                    self.assertEqual(self._snapshot(vault), before)
                    self.assertFalse((vault / FIXTURE_MARKER_NAME).exists())

    def test_empty_claim_race_removes_only_its_marker_and_never_opens_store(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "claim-race"
            original = fixture_guard_module._write_marker_once

            def inject_nonempty(root_fd: int):
                fd = os.open('private.bin', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=root_fd)
                try:
                    os.write(fd, b'private-sentinel')
                finally:
                    os.close(fd)
                return original(root_fd)

            with patch.object(fixture_guard_module, '_write_marker_once', side_effect=inject_nonempty):
                with patch('trove_core.wechat.indexer.SQLiteStore') as store:
                    with self.assertRaises(FixtureVaultGuardError) as caught:
                        index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, 'fixture_claim_raced_with_nonempty_root')
            store.assert_not_called()
            self.assertEqual((vault / 'private.bin').read_bytes(), b'private-sentinel')
            self.assertNotEqual((vault / FIXTURE_MARKER_NAME).read_bytes(), FIXTURE_MARKER_BYTES)
            with self.assertRaises(FixtureVaultGuardError) as invalidated:
                prepare_fixture_vault(vault)
            self.assertEqual(invalidated.exception.reason_code, 'fixture_marker_invalid_content')

    def test_rename_swap_after_marker_validation_preserves_unmarked_real_db(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            fixture = parent / 'fixture'
            index_fixture_vault(fixture, reset=True)
            real = parent / 'real'
            real_sqlite = real / 'index' / 'trove.sqlite'
            real_sqlite.parent.mkdir(parents=True)
            real_sqlite.write_bytes(b'real-db-sentinel')
            parked = parent / 'parked-fixture'
            original_replace = WeChatRepository.replace_fixture
            staged_paths: list[Path] = []

            def build_stage_then_swap(repository, *args, **kwargs):
                staged_paths.append(repository.store.path)
                fixture.rename(parked)
                real.rename(fixture)
                return original_replace(repository, *args, **kwargs)

            with patch.object(WeChatRepository, 'replace_fixture', new=build_stage_then_swap):
                with self.assertRaises(FixtureVaultGuardError) as caught:
                    index_fixture_vault(fixture, reset=True, write_jsonl=True)
            self.assertIn(caught.exception.reason_code, {'fixture_root_changed_during_operation', 'fixture_marker_missing'})
            self.assertTrue(staged_paths)
            self.assertTrue(all(not str(path).startswith(str(fixture)) for path in staged_paths))
            self.assertEqual((fixture / 'index' / 'trove.sqlite').read_bytes(), b'real-db-sentinel')
            self.assertFalse((fixture / FIXTURE_MARKER_NAME).exists())
            self.assertFalse((fixture / 'fixtures').exists())

    def test_post_publish_snapshot_private_db_is_never_adopted_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'claim-race'
            original_list = fixture_guard_module._list_names
            calls = 0

            def inject_after_snapshot(root_fd: int):
                nonlocal calls
                calls += 1
                names = original_list(root_fd)
                if calls == 2:
                    os.mkdir('index', mode=0o700, dir_fd=root_fd)
                    index_fd = os.open('index', os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0), dir_fd=root_fd)
                    try:
                        fd = os.open('trove.sqlite', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=index_fd)
                        try:
                            os.write(fd, b'private-db-after-snapshot')
                        finally:
                            os.close(fd)
                    finally:
                        os.close(index_fd)
                return names

            with patch.object(fixture_guard_module, '_list_names', side_effect=inject_after_snapshot):
                with self.assertRaises(FixtureVaultGuardError) as caught:
                    index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, 'fixture_uncertified_sqlite_present')
            self.assertEqual((vault / 'index' / 'trove.sqlite').read_bytes(), b'private-db-after-snapshot')
            self.assertEqual((vault / FIXTURE_MARKER_NAME).read_bytes(), FIXTURE_MARKER_BYTES)
            self.assertFalse((vault / FIXTURE_READY_NAME).exists())

    def test_marker_rollback_never_deletes_a_replacement_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'marker-race'
            original = fixture_guard_module._write_marker_once
            replacement_identity: list[tuple[int, int]] = []

            def replace_published_marker(root_fd: int):
                publication = original(root_fd)
                assert publication is not None
                replacement = '.replacement-marker'
                fd = os.open(replacement, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=root_fd)
                try:
                    os.write(fd, FIXTURE_MARKER_BYTES)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(replacement, FIXTURE_MARKER_NAME, src_dir_fd=root_fd, dst_dir_fd=root_fd)
                current = os.stat(FIXTURE_MARKER_NAME, dir_fd=root_fd, follow_symlinks=False)
                replacement_identity.append((current.st_dev, current.st_ino))
                return publication

            with patch.object(fixture_guard_module, '_write_marker_once', side_effect=replace_published_marker):
                with patch('trove_core.wechat.indexer.SQLiteStore') as store:
                    with self.assertRaises(FixtureVaultGuardError) as caught:
                        index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, 'fixture_marker_changed_during_publication')
            store.assert_not_called()
            current = (vault / FIXTURE_MARKER_NAME).stat()
            self.assertEqual((current.st_dev, current.st_ino), replacement_identity[0])
            self.assertEqual((vault / FIXTURE_MARKER_NAME).read_bytes(), FIXTURE_MARKER_BYTES)

    def test_empty_and_missing_roots_receive_strict_path_free_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            for vault in (Path(directory) / "empty", Path(directory) / "missing"):
                with self.subTest(vault=vault.name):
                    if vault.name == "empty":
                        vault.mkdir()
                    report = index_fixture_vault(vault, reset=True)
                    marker = vault / FIXTURE_MARKER_NAME
                    self.assertEqual(marker.read_bytes(), FIXTURE_MARKER_BYTES)
                    self.assertNotIn(str(vault).encode(), marker.read_bytes())
                    ready = vault / FIXTURE_READY_NAME
                    ready_payload = json.loads(ready.read_text(encoding="ascii"))
                    sqlite_path = vault / "index" / "trove.sqlite"
                    sqlite_stat = sqlite_path.stat()
                    self.assertEqual(ready_payload["format"], "trove-fixture-ready")
                    self.assertEqual(ready_payload["version"], 2)
                    self.assertEqual(
                        ready_payload["sqlite_sha256"],
                        hashlib.sha256(sqlite_path.read_bytes()).hexdigest(),
                    )
                    self.assertEqual(
                        (ready_payload["sqlite_device"], ready_payload["sqlite_inode"]),
                        (sqlite_stat.st_dev, sqlite_stat.st_ino),
                    )
                    self.assertNotIn(str(vault).encode(), ready.read_bytes())
                    self.assertGreaterEqual(report["counts"]["messages"], 12)

    def test_only_valid_marker_can_authorize_later_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "fixture"
            first = index_fixture_vault(vault, reset=True)
            keep = vault / "fixture-note.txt"
            keep.write_bytes(b"synthetic-only")
            second = index_fixture_vault(vault, reset=True)
            self.assertEqual(first["counts"], second["counts"])
            self.assertEqual(keep.read_bytes(), b"synthetic-only")

            marker = vault / FIXTURE_MARKER_NAME
            marker.write_bytes(b"{}\n")
            marker.chmod(0o600)
            before = self._snapshot(vault)
            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_marker_invalid_content")
            self.assertEqual(self._snapshot(vault), before)

    def test_active_sqlite_sidecars_are_never_deleted_or_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "fixture"
            index_fixture_vault(vault, reset=True)
            wal = vault / 'index' / 'trove.sqlite-wal'
            shm = vault / 'index' / 'trove.sqlite-shm'
            wal.write_bytes(b'stale-wal')
            shm.write_bytes(b'stale-shm')
            before = self._snapshot(vault)
            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_sqlite_sidecars_active")
            self.assertEqual(self._snapshot(vault), before)
            self.assertEqual(wal.read_bytes(), b'stale-wal')
            self.assertEqual(shm.read_bytes(), b'stale-shm')
            self.assertFalse((vault / FIXTURE_GENERATION_STATE_NAME).exists())

    def test_empty_read_sidecars_allow_only_exact_generation_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "fixture"
            index_fixture_vault(vault, reset=True)
            target = vault / "index" / "trove.sqlite"
            original_identity = (target.stat().st_dev, target.stat().st_ino)
            wal = vault / "index" / "trove.sqlite-wal"
            shm = vault / "index" / "trove.sqlite-shm"
            wal.write_bytes(b"")
            shm.write_bytes(b"read-only-shm-sentinel")

            index_fixture_vault(vault, reset=True)
            self.assertEqual((target.stat().st_dev, target.stat().st_ino), original_identity)
            self.assertEqual(wal.read_bytes(), b"")
            self.assertEqual(shm.read_bytes(), b"read-only-shm-sentinel")

            before = self._snapshot(vault)
            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_data(vault, generate_fixture(), reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_sqlite_sidecars_active")
            self.assertEqual(self._snapshot(vault), before)

    def test_target_inode_replaced_after_publish_is_never_certified_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "fixture"
            original = fixture_guard_module.FixtureVaultGuardSession.mark_generation_ready
            replacement_identity: list[tuple[int, int]] = []

            def replace_target_before_ready(session, artifact):
                target = vault / "index" / "trove.sqlite"
                replacement = vault / "index" / ".attacker-replacement.sqlite"
                replacement.write_bytes(target.read_bytes())
                replacement.chmod(0o600)
                os.replace(replacement, target)
                current = target.stat()
                replacement_identity.append((current.st_dev, current.st_ino))
                self.assertNotEqual(replacement_identity[-1], artifact.identity)
                return original(session, artifact)

            with patch.object(
                fixture_guard_module.FixtureVaultGuardSession,
                "mark_generation_ready",
                new=replace_target_before_ready,
            ):
                with self.assertRaises(FixtureVaultGuardError) as caught:
                    index_fixture_vault(vault, reset=True)

            self.assertEqual(caught.exception.reason_code, "fixture_published_artifact_replaced")
            self.assertTrue((vault / FIXTURE_GENERATION_STATE_NAME).exists())
            self.assertFalse((vault / FIXTURE_READY_NAME).exists())
            current = (vault / "index" / "trove.sqlite").stat()
            self.assertEqual((current.st_dev, current.st_ino), replacement_identity[0])

    def test_target_inode_replaced_after_ready_write_fails_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "fixture"
            original = fixture_guard_module._replace_ready_generation

            def replace_target_after_ready(*args, **kwargs):
                publication = original(*args, **kwargs)
                target = vault / "index" / "trove.sqlite"
                replacement = vault / "index" / ".post-ready-replacement.sqlite"
                replacement.write_bytes(target.read_bytes())
                replacement.chmod(0o600)
                os.replace(replacement, target)
                return publication

            with patch.object(
                fixture_guard_module,
                "_replace_ready_generation",
                new=replace_target_after_ready,
            ):
                with self.assertRaises(FixtureVaultGuardError) as caught:
                    index_fixture_vault(vault, reset=True)

            self.assertEqual(caught.exception.reason_code, "fixture_published_artifact_replaced")
            ready = json.loads((vault / FIXTURE_READY_NAME).read_text(encoding="ascii"))
            current = (vault / "index" / "trove.sqlite").stat()
            self.assertNotEqual(
                (ready["sqlite_device"], ready["sqlite_inode"]),
                (current.st_dev, current.st_ino),
            )
            self.assertTrue((vault / FIXTURE_GENERATION_STATE_NAME).exists())

    def test_publish_fsync_and_ready_failures_are_retryable_and_integral(self):
        failpoints = (
            "_copy_generation_candidate",
            "_sync_generation_directory",
            "_replace_ready_generation",
        )
        for failpoint in failpoints:
            with self.subTest(failpoint=failpoint), tempfile.TemporaryDirectory() as directory:
                vault = Path(directory) / "fixture"
                original = getattr(fixture_guard_module, failpoint)
                calls = 0

                def fail_once(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise FixtureVaultGuardError(f"injected_{failpoint}")
                    return original(*args, **kwargs)

                with patch.object(fixture_guard_module, failpoint, new=fail_once):
                    with self.assertRaises(FixtureVaultGuardError) as caught:
                        index_fixture_vault(vault, reset=True)
                self.assertEqual(caught.exception.reason_code, f"injected_{failpoint}")
                self.assertTrue((vault / FIXTURE_GENERATION_STATE_NAME).exists())

                index_fixture_vault(vault, reset=True)
                self._assert_ready_generation(vault)

    def test_switch_failure_preserves_old_complete_generation_until_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "fixture"
            index_fixture_vault(vault, reset=True)
            target = vault / "index" / "trove.sqlite"
            old_bytes = target.read_bytes()
            old_identity = (target.stat().st_dev, target.stat().st_ino)
            original = fixture_guard_module._switch_generation

            def fail_before_replace(*args, **kwargs):
                raise FixtureVaultGuardError("injected_switch_failure")

            with patch.object(fixture_guard_module, "_switch_generation", new=fail_before_replace):
                with self.assertRaises(FixtureVaultGuardError) as caught:
                    index_fixture_data(vault, generate_fixture(), reset=True)
            self.assertEqual(caught.exception.reason_code, "injected_switch_failure")

            state = json.loads((vault / FIXTURE_GENERATION_STATE_NAME).read_text(encoding="ascii"))
            self.assertEqual(state["phase"], "prepared")
            previous = vault / "index" / f".trove.sqlite.fixture-{state['nonce']}.previous"
            self.assertEqual(target.read_bytes(), old_bytes)
            self.assertEqual(previous.read_bytes(), old_bytes)
            self.assertEqual((target.stat().st_dev, target.stat().st_ino), old_identity)
            self.assertEqual(
                (previous.stat().st_dev, previous.stat().st_ino),
                old_identity,
            )

            index_fixture_data(vault, generate_fixture(), reset=True)
            self._assert_ready_generation(vault)

    def test_root_symlink_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "target"
            target.mkdir()
            link = parent / "fixture-link"
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(link, reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_root_is_symlink")
            self.assertEqual(list(target.iterdir()), [])

    def test_parent_symlink_cannot_redirect_new_fixture_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            external = parent / "external"
            external.mkdir()
            redirect = parent / "redirect"
            redirect.symlink_to(external, target_is_directory=True)

            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(redirect / "new-fixture", reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_path_contains_symlink")
            self.assertEqual(list(external.iterdir()), [])

    def test_marker_symlink_is_not_fixture_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            marker_source = parent / "copied-marker"
            marker_source.write_bytes(FIXTURE_MARKER_BYTES)
            marker_source.chmod(0o600)
            vault = parent / "vault"
            vault.mkdir()
            (vault / FIXTURE_MARKER_NAME).symlink_to(marker_source)

            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_marker_invalid_type")
            self.assertEqual(marker_source.read_bytes(), FIXTURE_MARKER_BYTES)

    def test_fixture_tree_symlink_cannot_escape_reset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            vault = parent / "fixture"
            prepare_fixture_vault(vault)
            external = parent / "external"
            external.mkdir()
            sentinel = external / "trove.sqlite"
            sentinel.write_bytes(b"external-sentinel")
            (vault / "index").symlink_to(external, target_is_directory=True)

            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(vault, reset=True)
            self.assertEqual(caught.exception.reason_code, "fixture_tree_contains_symlink")
            self.assertEqual(sentinel.read_bytes(), b"external-sentinel")

    def test_fixture_tree_hardlink_cannot_escape_non_reset_write(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            vault = parent / "fixture"
            prepare_fixture_vault(vault)
            (vault / "index").mkdir()
            external = parent / "external.sqlite"
            external.write_bytes(b"external-sentinel")
            (vault / "index" / "trove.sqlite").hardlink_to(external)

            with self.assertRaises(FixtureVaultGuardError) as caught:
                index_fixture_vault(vault, reset=False)
            self.assertEqual(caught.exception.reason_code, "fixture_tree_contains_hardlink")
            self.assertEqual(external.read_bytes(), b"external-sentinel")

    @staticmethod
    def _assert_ready_generation(vault: Path) -> None:
        target = vault / "index" / "trove.sqlite"
        ready = json.loads((vault / FIXTURE_READY_NAME).read_text(encoding="ascii"))
        current = target.stat()
        assert ready["sqlite_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert (ready["sqlite_device"], ready["sqlite_inode"]) == (current.st_dev, current.st_ino)
        assert not (vault / FIXTURE_GENERATION_STATE_NAME).exists()
        assert not list((vault / "index").glob(".trove.sqlite.fixture-*"))
        with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]

    @staticmethod
    def _snapshot(root: Path) -> list[tuple[str, str, bytes]]:
        snapshot: list[tuple[str, str, bytes]] = []
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            if path.is_symlink():
                snapshot.append((relative, "symlink", str(path.readlink()).encode()))
            elif path.is_dir():
                snapshot.append((relative, "directory", b""))
            else:
                snapshot.append((relative, "file", path.read_bytes()))
        return snapshot


if __name__ == "__main__":
    unittest.main()
