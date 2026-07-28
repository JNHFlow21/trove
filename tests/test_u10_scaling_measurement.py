from __future__ import annotations

import unittest

from scripts.benchmark_u10_scaling import measure


class U10ScalingMeasurementTests(unittest.TestCase):
    def test_measurement_is_synthetic_bounded_and_exposes_legacy_linear_work(self):
        report = measure(128, 64)
        self.assertTrue(report['ok'])
        self.assertTrue(report['fixture']['synthetic_only'])
        self.assertEqual(report['dirty_backlog']['legacy_materialized_rows'], 128)
        self.assertEqual(report['dirty_backlog']['bounded_batch_rows'], 128)
        self.assertGreater(report['vector_metadata']['legacy_content_hash_bytes'], 128)
        self.assertEqual(report['vector_metadata']['authoritative_rows'], 128)
        self.assertEqual(report['vector_metadata']['delta_candidate_rows'], 1)
        self.assertLessEqual(report['vector_metadata']['constant_sidecar_bytes'], 4096)
        self.assertEqual(report['watcher']['legacy_stat_calls'], 66)
        self.assertEqual(report['watcher']['production_probe']['idle_entries_processed'], 0)
        self.assertTrue(report['watcher']['production_probe']['scan_completed'])
        self.assertEqual(report['watcher']['bounded_fallback_max_entries_per_tick'], 4096)
        self.assertEqual(report['watcher']['bounded_fallback_ticks'], 1)
        self.assertFalse(report['raw_content_included'])

    def test_one_million_logical_watch_fixture_counts_every_legacy_stat_without_inodes(self):
        report = measure(32, 1_000_000, logical_watch=True)
        self.assertEqual(report['watcher']['fixture_files'], 1_000_000)
        self.assertEqual(report['watcher']['legacy_stat_calls'], 1_000_257)
        self.assertEqual(report['watcher']['bounded_fallback_ticks'], 245)
        self.assertEqual(report['watcher']['native_directory_watch_limit'], 1024)
        self.assertEqual(report['watcher']['measurement'], 'logical_exact_operation_count')


if __name__ == '__main__':
    unittest.main()
