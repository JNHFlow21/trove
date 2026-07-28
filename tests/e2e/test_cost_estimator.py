from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import MediaAssetLinkRecord, MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore


class CostEstimatorTests(unittest.TestCase):
    def test_estimator_counts_accepted_media_and_stays_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            repo = MultimodalRepository(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            repo.upsert_media_asset(MediaAssetRecord('asset-v', 'acct-a', 'private_chat', 'msg-1', 'voice', 'voice', 'trove://wechat/acct-a/chat/c/s/1', metadata={'duration_seconds': 30}))
            repo.upsert_media_asset_link(MediaAssetLinkRecord('link-v', 'asset-v', 'acct-a', 'private_chat', 'trove://wechat/acct-a/chat/c/s/1', 'private_chat', True, 'accepted'))
            repo.upsert_media_asset(MediaAssetRecord('asset-i', 'acct-a', 'moment', 'm1', 'image', 'image', 'trove://wechat/acct-a/moment/m1'))
            repo.upsert_media_asset_link(MediaAssetLinkRecord('link-i', 'asset-i', 'acct-a', 'moment', 'trove://wechat/acct-a/moment/m1', 'moment', True, 'accepted'))
            out = vault / 'proof' / 'cost' / 'estimate.redacted.json'
            proc = subprocess.run([sys.executable, 'scripts/estimate_cloud_import_cost.py', '--vault', str(vault), '--out', str(out)], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(payload['accepted_audio_duration_seconds'], 30)
            self.assertEqual(payload['accepted_image_count'], 1)
            self.assertIsNotNone(payload['estimated_cost_rmb'])
            text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn('/Users/', text)
            self.assertFalse(payload['raw_paths_included'])


if __name__ == '__main__':
    unittest.main()
