from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.sync import SyncConfig, SyncOptions, watch_sync
from trove_core.watch import (
    KqueueWatchBackend,
    MAX_SCAN_ENTRIES_PER_TICK,
    ManifestPollingBackend,
    WatchTick,
    load_watch_manifest,
)


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class _TickSequenceBackend:
    name = 'fixture-backend'

    def __init__(self, ticks: list[WatchTick]):
        self.ticks = list(ticks)
        self.poll_count = 0
        self.closed = False

    def poll(self, timeout: float = 1.0) -> WatchTick:
        del timeout
        self.poll_count += 1
        if not self.ticks:
            raise AssertionError('watch_sync polled past the stable fixture tick')
        return self.ticks.pop(0)

    def request_repair(self, *, reason: str = 'manual') -> None:
        pass

    def close(self) -> None:
        self.closed = True


class WatchBackendTests(unittest.TestCase):
    def _finish_scan(self, backend: ManifestPollingBackend, *, maximum_ticks: int = 100):
        ticks = []
        for _ in range(maximum_ticks):
            tick = backend.poll(timeout=0.0)
            ticks.append(tick)
            if tick.scan_complete and not tick.scan_discarded:
                return tick, ticks
        self.fail('bounded manifest scan did not finish')

    def test_streaming_manifest_scan_is_bounded_and_publishes_only_when_complete(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / 'snapshot'
            root.mkdir()
            for index in range(10_000):
                (root / f'fixture-{index:05d}.db').touch()
            manifest_path = base / 'jobs' / 'watch.redacted.json'
            clock = _Clock()
            backend = ManifestPollingBackend(
                root,
                manifest_path,
                max_entries_per_tick=MAX_SCAN_ENTRIES_PER_TICK,
                clock=clock,
                sleep=clock.sleep,
            )
            try:
                first = backend.poll(timeout=0.0)
                second = backend.poll(timeout=0.0)
                self.assertLessEqual(first.entries_processed, MAX_SCAN_ENTRIES_PER_TICK)
                self.assertLessEqual(second.entries_processed, MAX_SCAN_ENTRIES_PER_TICK)
                self.assertFalse(first.scan_complete)
                self.assertFalse(second.scan_complete)
                self.assertFalse(manifest_path.exists())

                final, ticks = self._finish_scan(backend)
                self.assertTrue(final.changed)
                self.assertTrue(final.scan_complete)
                self.assertTrue(all(tick.entries_processed <= MAX_SCAN_ENTRIES_PER_TICK for tick in ticks))
                manifest = load_watch_manifest(manifest_path)
                self.assertIsNotNone(manifest)
                self.assertEqual(manifest.entry_count, 10_001)
                payload = manifest_path.read_text(encoding='utf-8')
                self.assertNotIn('fixture-00000.db', payload)
                self.assertLess(len(payload.encode('utf-8')), 1024)
                self.assertFalse(json.loads(payload)['raw_paths_included'])
            finally:
                backend.close()

    def test_idle_poll_does_no_scan_work_and_backoff_grows_to_bound(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / 'snapshot'
            root.mkdir()
            source = root / 'fixture.db'
            source.write_text('one', encoding='utf-8')
            clock = _Clock()
            backend = ManifestPollingBackend(
                root,
                base / 'manifest.json',
                min_backoff_seconds=1.0,
                max_backoff_seconds=4.0,
                clock=clock,
                sleep=clock.sleep,
            )
            try:
                initial, _ = self._finish_scan(backend)
                self.assertTrue(initial.changed)
                idle = backend.poll(timeout=0.0)
                self.assertEqual(idle.entries_processed, 0)
                self.assertFalse(idle.scan_complete)

                source.write_text('changed-size', encoding='utf-8')
                clock.advance(1.0)
                changed, _ = self._finish_scan(backend)
                self.assertTrue(changed.changed)

                clock.advance(1.0)
                unchanged, _ = self._finish_scan(backend)
                self.assertFalse(unchanged.changed)
                clock.advance(2.0)
                unchanged_again, _ = self._finish_scan(backend)
                self.assertFalse(unchanged_again.changed)
                self.assertEqual(backend._backoff, 4.0)
                idle_again = backend.poll(timeout=0.0)
                self.assertEqual(idle_again.entries_processed, 0)
            finally:
                backend.close()

    def test_interrupted_or_invalidated_scan_keeps_prior_complete_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / 'snapshot'
            root.mkdir()
            (root / 'baseline.db').touch()
            manifest_path = base / 'manifest.json'
            clock = _Clock()
            baseline = ManifestPollingBackend(
                root,
                manifest_path,
                max_entries_per_tick=2,
                clock=clock,
                sleep=clock.sleep,
            )
            first, _ = self._finish_scan(baseline)
            self.assertTrue(first.changed)
            prior_bytes = manifest_path.read_bytes()
            prior_generation = load_watch_manifest(manifest_path).scan_generation

            for index in range(20):
                (root / f'new-{index:02d}.db').touch()
            baseline.request_repair(reason='fixture-change')
            partial = baseline.poll(timeout=0.0)
            self.assertFalse(partial.scan_complete)
            self.assertEqual(manifest_path.read_bytes(), prior_bytes)
            baseline.request_repair(reason='event-during-scan')
            discarded = None
            for _ in range(20):
                tick = baseline.poll(timeout=0.0)
                if tick.scan_discarded:
                    discarded = tick
                    break
            self.assertIsNotNone(discarded)
            self.assertEqual(manifest_path.read_bytes(), prior_bytes)
            baseline.close()

            recovered = ManifestPollingBackend(
                root,
                manifest_path,
                max_entries_per_tick=2,
                clock=clock,
                sleep=clock.sleep,
            )
            try:
                complete, ticks = self._finish_scan(recovered)
                self.assertTrue(complete.changed)
                self.assertTrue(all(tick.entries_processed <= 2 for tick in ticks))
                self.assertEqual(load_watch_manifest(manifest_path).scan_generation, prior_generation + 1)
            finally:
                recovered.close()

    def test_manifest_fallback_follows_only_the_configured_root_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            target = base / 'runs' / 'fixture-run'
            target.mkdir(parents=True)
            source = target / 'fixture.db'
            source.write_text('one', encoding='utf-8')
            root = base / 'current'
            root.symlink_to(target, target_is_directory=True)
            manifest_path = base / 'manifest.json'
            clock = _Clock()
            backend = ManifestPollingBackend(
                root,
                manifest_path,
                clock=clock,
                sleep=clock.sleep,
            )
            try:
                initial, _ = self._finish_scan(backend)
                initial_digest = initial.manifest_digest
                self.assertEqual(load_watch_manifest(manifest_path).entry_count, 3)

                source.write_text('changed-through-current-symlink', encoding='utf-8')
                backend.request_repair(reason='fixture-target-change')
                changed, _ = self._finish_scan(backend)
                self.assertTrue(changed.changed)
                self.assertNotEqual(changed.manifest_digest, initial_digest)
            finally:
                backend.close()

    @unittest.skipUnless(sys.platform == 'darwin', 'native kqueue test is macOS-only')
    def test_native_kqueue_backend_bounds_descriptors_and_reports_events(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / 'snapshot'
            (root / 'a' / 'b' / 'c').mkdir(parents=True)
            backend = KqueueWatchBackend(
                root,
                base / 'manifest.json',
                max_directory_watches=2,
                max_entries_per_tick=32,
            )
            try:
                initial = backend.poll(timeout=0.0)
                self.assertTrue(initial.scan_complete)
                self.assertTrue(initial.descriptor_overflow)
                self.assertLessEqual(len(backend._fd_to_path), 2)

                (root / 'event.db').touch()
                event_tick = None
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    tick = backend.poll(timeout=0.1)
                    if tick.change_source == 'event':
                        event_tick = tick
                        break
                self.assertIsNotNone(event_tick)
                self.assertLessEqual(event_tick.entries_processed, 32)
            finally:
                backend.close()

    def test_watch_sync_uses_backend_instead_of_recursive_mtime_polling(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            snapshot = Path(d) / 'snapshot'
            snapshot.mkdir()
            backend = _TickSequenceBackend([
                WatchTick(backend='fixture-backend', changed=True, change_source='event', repair_pending=True),
                WatchTick(
                    backend='fixture-backend',
                    scan_complete=True,
                    manifest_digest='a' * 64,
                ),
            ])
            options = SyncOptions(snapshot_dir=snapshot)
            with patch('trove_core.sync.run_sync', return_value={'ok': True}) as run, \
                 patch.object(Path, 'rglob', side_effect=AssertionError('legacy scan used')), \
                 patch('trove_core.sync.read_sync_config', return_value=SyncConfig(debounce_seconds=0.0)):
                watch_sync(vault, options=options, once=True, backend=backend)

            run.assert_called_once_with(vault, options=options)
            self.assertEqual(backend.poll_count, 2)
            self.assertTrue(backend.closed)

    def test_profile_worker_failure_does_not_terminate_or_leak_detail_from_watcher(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            snapshot = Path(d) / 'snapshot'
            snapshot.mkdir()
            backend = _TickSequenceBackend([
                WatchTick(
                    backend='fixture-backend',
                    scan_complete=True,
                    manifest_digest='c' * 64,
                ),
            ])
            emitted: list[str] = []

            with patch(
                'trove_core.sync.process_profile_refresh_queue',
                side_effect=RuntimeError('private fixture detail'),
            ), patch('builtins.print', side_effect=lambda value, **_: emitted.append(str(value))):
                watch_sync(
                    vault,
                    options=SyncOptions(snapshot_dir=snapshot),
                    once=True,
                    backend=backend,
                )

            self.assertTrue(backend.closed)
            self.assertTrue(any('RuntimeError' in value for value in emitted))
            self.assertFalse(any('private fixture detail' in value for value in emitted))

    def test_watch_sync_waits_for_stable_repair_after_partial_and_discarded_event_scan(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            snapshot = Path(d) / 'snapshot'
            snapshot.mkdir()
            backend = _TickSequenceBackend([
                WatchTick(backend='fixture-backend', changed=True, change_source='event', repair_pending=True),
                WatchTick(backend='fixture-backend', entries_processed=4096, repair_pending=True),
                WatchTick(
                    backend='fixture-backend',
                    scan_discarded=True,
                    event_loss=True,
                    repair_pending=True,
                ),
                WatchTick(backend='fixture-backend', entries_processed=4096, repair_pending=True),
                WatchTick(
                    backend='fixture-backend',
                    scan_complete=True,
                    manifest_digest='b' * 64,
                    descriptor_overflow=True,
                ),
            ])
            options = SyncOptions(snapshot_dir=snapshot)

            def stable_only(*_args, **_kwargs):
                self.assertEqual(backend.poll_count, 5)
                return {'ok': True}

            with patch('trove_core.sync.run_sync', side_effect=stable_only) as run, \
                 patch('trove_core.sync.read_sync_config', return_value=SyncConfig(debounce_seconds=0.0)):
                watch_sync(vault, options=options, once=True, backend=backend)

            run.assert_called_once()
            self.assertTrue(backend.closed)

    def test_watch_sync_debounce_is_not_starved_by_unchanged_completed_scans(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            snapshot = vault / 'sources' / 'wechat-kos-decrypted' / 'current'
            snapshot.mkdir(parents=True)
            backend = _TickSequenceBackend([
                WatchTick(
                    backend='fixture',
                    changed=True,
                    scan_complete=True,
                    manifest_digest='a' * 64,
                ),
                WatchTick(
                    backend='fixture',
                    changed=False,
                    scan_complete=True,
                    manifest_digest='a' * 64,
                ),
            ])
            options = SyncOptions(snapshot_dir=snapshot)
            with patch('trove_core.sync.run_sync', return_value={'ok': True}) as run, \
                 patch('trove_core.sync.read_sync_config', return_value=SyncConfig(debounce_seconds=3.0)), \
                 patch('trove_core.sync.time.monotonic', side_effect=[0.0, 3.0]), \
                 patch('trove_core.sync.time.sleep'):
                watch_sync(vault, options=options, once=True, backend=backend)

            run.assert_called_once()
            self.assertTrue(backend.closed)


if __name__ == '__main__':
    unittest.main()
