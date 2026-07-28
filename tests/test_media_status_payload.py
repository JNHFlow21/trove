from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.media_pipeline import media_status_payload
from trove_core.store.repositories import ImageObservationRecord, MediaAssetRecord, MultimodalRepository, ProviderJobRecord, TranscriptRecord
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig


class MediaStatusPayloadTests(unittest.TestCase):
    def test_voice_coverage_counts_only_current_cloud_transcripts(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-cloud', 'acct-a', 'message', 'm1', 'voice', 'voice',
                'trove://voice/cloud', content_hash='a' * 64,
            ))
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-fake', 'acct-a', 'message', 'm2', 'voice', 'voice',
                'trove://voice/fake', content_hash='b' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-cloud', asset_id='asset-cloud', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='a' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-fake', asset_id='asset-fake', provider='fake-asr',
                model='fixture', job_type='asr', status='completed',
                request_hash='b' * 64,
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-cloud', 'asset-cloud', 'trove://voice/cloud#voice',
                'valid cloud transcript', job_id='job-cloud',
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-fake', 'asset-fake', 'trove://voice/fake#voice',
                'fake transcript', job_id='job-fake',
            ))

            payload = media_status_payload(vault)

            self.assertEqual(payload['coverage']['voice_transcripts'], {
                'done': 1,
                'total': 2,
                'ratio': 0.5,
            })

    def test_empty_vault_status_does_not_create_database_files(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            vault.mkdir()

            payload = media_status_payload(vault)

            self.assertTrue(payload['ok'])
            self.assertEqual(payload['media_assets'], {'voice': 0, 'image': 0})
            self.assertEqual(payload['queue'], {})
            self.assertEqual(payload['backlog'], 0)
            self.assertEqual(list(vault.rglob('*')), [])

    def test_caption_coverage_counts_captioned_images(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            repo.upsert_media_asset(MediaAssetRecord('asset-1', 'acct-a', 'message', 'm1', 'image', 'image', 'trove://image/1'))
            repo.upsert_media_asset(MediaAssetRecord('asset-2', 'acct-a', 'message', 'm2', 'image', 'image', 'trove://image/2'))
            repo.insert_image_observation(ImageObservationRecord('obs-1', 'asset-1', 'trove://image/1#image', '中文描述'))
            repo.insert_image_observation(ImageObservationRecord('obs-2', 'asset-2', 'trove://image/2#image', '', visible_text='OCR'))

            payload = media_status_payload(vault)

            self.assertEqual(payload['coverage']['image_captions'], {'done': 1, 'total': 2, 'ratio': 0.5})


if __name__ == '__main__':
    unittest.main()
