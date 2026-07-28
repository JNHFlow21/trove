from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vision.base import ImageObservationResult, VisionProvider, VisionRequest, VisionUsage
from trove_core.vision.jobs import run_image_observation_job


class FakeVisionProvider(VisionProvider):
    name = 'fake-vision'
    model = 'doubao-seed-2-0-lite-260215'
    def __init__(self, fail=None, confidence=0.8):
        self.fail = fail
        self.confidence = confidence
        self.calls = 0
    def observe(self, request: VisionRequest) -> ImageObservationResult:
        self.calls += 1
        if self.fail:
            raise self.fail
        return ImageObservationResult('fixture image', '报价', ['document'], ['pricing'], ['示例教育'], self.confidence, VisionUsage(input_tokens=100, output_tokens=50, estimated_cost_rmb=0.001), citations=[request.citation] if request.citation else [])


class ImageObservationJobsTests(unittest.TestCase):
    def test_success_writes_observation_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-i', account_id='acct-a', source_type='moment', source_id='m1', modality='image', media_type='image', citation='trove://wechat/acct-a/moment/m1'))
            img = root / 'img.jpg'; img.write_bytes(b'\xff\xd8\xfffixture')
            provider = FakeVisionProvider()
            first = run_image_observation_job(repo, asset_id='asset-i', image_path=img, provider=provider, citation='trove://wechat/acct-a/moment/m1')
            second = run_image_observation_job(repo, asset_id='asset-i', image_path=img, provider=provider, citation='trove://wechat/acct-a/moment/m1')
            self.assertEqual(first['observation_status'], 'active')
            self.assertTrue(second['idempotent'])
            self.assertEqual(provider.calls, 1)
            with repo.store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM image_observations').fetchone()[0], 1)

    def test_malformed_provider_json_records_needs_review(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            img = root / 'img.jpg'; img.write_bytes(b'\xff\xd8\xfffixture')
            result = run_image_observation_job(repo, asset_id='asset-i', image_path=img, provider=FakeVisionProvider(ValueError('bad json')), citation='trove://c')
            self.assertEqual(result['status'], 'needs_review')
            with repo.store.connect() as conn:
                self.assertEqual(conn.execute('SELECT status FROM provider_jobs').fetchone()[0], 'needs_review')

    def test_low_confidence_observation_is_review_needed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            img = root / 'img.jpg'; img.write_bytes(b'\xff\xd8\xfffixture')
            result = run_image_observation_job(repo, asset_id='asset-i', image_path=img, provider=FakeVisionProvider(confidence=0.4), citation='trove://c')
            self.assertEqual(result['observation_status'], 'needs_review')


if __name__ == '__main__':
    unittest.main()
