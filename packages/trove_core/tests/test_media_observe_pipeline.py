from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.media_pipeline import run_image_observation_budget
from trove_core.local_vlm.fake import FakeLocalVLMCaptionProvider
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.vision.base import ImageObservationResult, VisionProvider, VisionRequest, VisionUsage
from trove_core.wechat.media.dat_decoder import decode_wechat_dat_file


class LocalFakeVisionProvider(VisionProvider):
    name = 'local-fake-vision'
    model = 'fixture-ocr'

    def __init__(self):
        self.calls = 0

    def observe(self, request: VisionRequest) -> ImageObservationResult:
        self.calls += 1
        return ImageObservationResult(
            caption='',
            visible_text='图片OCR试点needle',
            objects=[],
            business_signals=[],
            entity_mentions=[],
            confidence=0.99,
            usage=VisionUsage(estimated_cost_rmb=0.0),
            citations=[request.citation] if request.citation else [],
        )


class MediaObservePipelineTests(unittest.TestCase):
    def test_ocr_provider_runs_without_global_writer(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'cached.jpg'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'\xff\xd8\xfffixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                asset_id='asset-lock-probe', account_id='acct-a', source_type='message',
                source_id='msg-lock-probe', modality='image', media_type='image',
                citation='trove://wechat/acct-a/conv-a/message_0/99',
                path_ref='sources/cached.jpg', cache_state='cached',
            ))

            class ProbeProvider(LocalFakeVisionProvider):
                def observe(self, request):
                    with VaultOperationCoordinator(cfg).write(owner='probe-ocr'):
                        pass
                    return super().observe(request)

            result = run_image_observation_budget(
                vault, budget=1, provider=ProbeProvider(), include_images=True,
            )
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['completed'], 1)

    def test_budgeted_local_image_pipeline_decodes_dat_and_writes_searchable_chunk_once(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'image.dat'
            image.parent.mkdir(parents=True, exist_ok=True)
            raw = b'\xff\xd8\xfffixture'
            key = 0x42
            image.write_bytes(bytes(b ^ key for b in raw))
            self.assertEqual(decode_wechat_dat_file(image).status, 'decoded')
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image-1',
                account_id='acct-a',
                source_type='message',
                source_id='msg-1',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv-a/message_0/1',
                path_ref='sources/image.dat',
                cache_state='cached',
            ))
            provider = LocalFakeVisionProvider()

            first = run_image_observation_budget(vault, budget=1, provider=provider, include_images=True)
            second = run_image_observation_budget(vault, budget=1, provider=provider, include_images=True)

            self.assertTrue(first['ok'])
            self.assertEqual(first['completed'], 1)
            self.assertEqual(second['processed'], 0)
            self.assertEqual(provider.calls, 1)
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM image_observations').fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE source_type='image_observation'").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT status FROM media_jobs WHERE job_type='image_observe'").fetchone()[0], 'done')
            hits = store.chunk_search('图片OCR试点needle', filters={'source_type': 'image_observation'}, limit=3)
            self.assertEqual(len(hits), 1)
            self.assertFalse(first['raw_content_included'])
            self.assertFalse(first['cloud_calls_made'])

    def test_caption_budget_merges_with_ocr_and_refreshes_search_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'poster.jpg'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'\xff\xd8\xfffixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-caption-1',
                account_id='acct-a',
                source_type='message',
                source_id='msg-caption',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv-a/message_0/9',
                path_ref='sources/poster.jpg',
                cache_state='cached',
            ))
            ocr_provider = LocalFakeVisionProvider()
            caption_provider = FakeLocalVLMCaptionProvider(caption='中文海报描述needle', labels=['海报', '预算'])

            result = run_image_observation_budget(
                vault,
                budget=1,
                provider=ocr_provider,
                caption=True,
                caption_budget=1,
                caption_provider=caption_provider,
                include_images=True,
            )

            self.assertTrue(result['ok'])
            self.assertEqual(result['completed'], 1)
            self.assertEqual(result['caption']['completed'], 1)
            self.assertEqual(ocr_provider.calls, 1)
            self.assertEqual(caption_provider.calls, 1)
            with store.connect() as conn:
                row = conn.execute('SELECT caption,visible_text,objects_json FROM image_observations WHERE asset_id=?', ('asset-caption-1',)).fetchone()
                self.assertEqual(row['caption'], '中文海报描述needle')
                self.assertIn('图片OCR试点needle', row['visible_text'])
                self.assertIn('海报', row['objects_json'])
                chunk = conn.execute("SELECT content FROM evidence_chunks WHERE source_type='image_observation'").fetchone()
                self.assertIn('中文海报描述needle', chunk['content'])
                self.assertIn('图片OCR试点needle', chunk['content'])
            hits = store.chunk_search('中文海报描述needle', filters={'source_type': 'image_observation'}, limit=3)
            self.assertEqual(len(hits), 1)

    def test_budget_prioritizes_cached_images_before_missing_cache(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'cached.jpg'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'\xff\xd8\xfffixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image-missing',
                account_id='acct-a',
                source_type='message',
                source_id='msg-missing',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv-a/message_0/1',
                path_ref='sources/missing.jpg',
                cache_state='missing_local_cache',
            ))
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image-cached',
                account_id='acct-a',
                source_type='message',
                source_id='msg-cached',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv-a/message_0/2',
                path_ref='sources/cached.jpg',
                cache_state='cached',
            ))
            provider = LocalFakeVisionProvider()

            result = run_image_observation_budget(vault, budget=1, provider=provider, include_images=True)

            self.assertTrue(result['ok'])
            self.assertEqual(result['completed'], 1)
            self.assertEqual(result['skipped'], 0)
            with store.connect() as conn:
                statuses = {
                    row['asset_id']: row['status']
                    for row in conn.execute('SELECT asset_id,status FROM media_jobs ORDER BY asset_id')
                }
            self.assertEqual(statuses['asset-image-cached'], 'done')
            # Explicit image work enqueues only the bounded batch; it must not
            # rescan and publish jobs for the whole corpus under the writer.
            self.assertNotIn('asset-image-missing', statuses)

    def test_image_precompute_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'cached.jpg'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'\xff\xd8\xfffixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image-cached',
                account_id='acct-a',
                source_type='message',
                source_id='msg-cached',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv-a/message_0/2',
                path_ref='sources/cached.jpg',
                cache_state='cached',
            ))
            provider = LocalFakeVisionProvider()

            result = run_image_observation_budget(vault, budget=1, provider=provider)

            self.assertTrue(result['ok'])
            self.assertEqual(result['action'], 'media_observe_disabled')
            self.assertEqual(result['completed'], 0)
            self.assertEqual(provider.calls, 0)
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM media_jobs WHERE job_type='image_observe'").fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
