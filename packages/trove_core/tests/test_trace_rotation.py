from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.vault import tracing
from trove_core.vault.tracing import TraceTimeline


class TraceRotationTests(unittest.TestCase):
    def test_trace_rotates_to_five_private_files_and_tails_across_boundary(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tracing, '_MAX_TRACE_BYTES', 700):
            timeline = TraceTimeline(directory)
            for index in range(40):
                timeline.append('sync', 'complete', {'count': index})
            log_dir = Path(directory) / 'logs'
            family = sorted(log_dir.glob('trace-timeline.redacted.jsonl*'))
            data_files = [path for path in family if path.name != 'trace-timeline.redacted.jsonl' or path.is_file()]
            # Dedicated coordination file has another name and is not counted.
            self.assertLessEqual(len(data_files), 5)
            self.assertGreaterEqual(len(data_files), 2)
            self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in data_files))
            rows = timeline.list(limit=10)
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[-1]['payload']['count'], 39)

    def test_tail_reads_only_limit_proportional_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            timeline = TraceTimeline(directory)
            timeline.append('sync', 'complete', {'count': 0})
            path = Path(directory) / 'logs' / 'trace-timeline.redacted.jsonl'
            row = json.dumps({
                'trace_id': 'trace-' + 'a' * 16,
                'stage': 'sync',
                'status': 'complete',
                'created_at': '2026-01-01T00:00:00Z',
                'payload': {'count': 1},
            }).encode('utf-8') + b'\n'
            with path.open('ab') as handle:
                while handle.tell() < 1024 * 1024:
                    handle.write(row)
            original_pread = os.pread
            observed = {'bytes': 0}

            def measured(fd, amount, offset):
                observed['bytes'] += amount
                return original_pread(fd, amount, offset)

            with patch.object(os, 'pread', side_effect=measured):
                rows = timeline.list(limit=2)
            self.assertEqual(len(rows), 2)
            self.assertLess(observed['bytes'], 64 * 1024)

    def test_unknown_telemetry_markers_and_payload_content_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            canary = 'privatecanarymarker'
            timeline = TraceTimeline(directory)
            timeline.append(canary, canary, {'query': canary, canary: canary})
            persisted = (Path(directory) / 'logs' / 'trace-timeline.redacted.jsonl').read_text(encoding='utf-8')
            self.assertNotIn(canary, persisted)
            self.assertIn('redacted-stage-', persisted)
            self.assertIn('redacted-status-', persisted)


if __name__ == '__main__':
    unittest.main()
