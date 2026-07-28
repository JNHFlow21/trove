from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import (
    EntityRecord,
    ImageObservationRecord,
    MediaAssetRecord,
    MultimodalRepository,
    ObservationRecord,
    ProviderJobRecord,
    RelationshipRecord,
    TranscriptRecord,
)
from trove_core.store.sqlite_store import SQLiteStore


class ObservationSchemaTests(unittest.TestCase):
    def test_multimodal_repository_preserves_citations_and_usage(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'trove.sqlite'))
            asset = repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-1', account_id='acct-a', source_type='message', source_id='msg-1',
                modality='voice', media_type='voice', local_type='34', citation='trove://wechat/acct-a/conv/s1/1',
                content_hash='hash-1', cache_state='cached', processing_state='pending', metadata={'duration_hint': 3.2},
            ))
            self.assertEqual(asset['citation'], 'trove://wechat/acct-a/conv/s1/1')
            job = repo.record_provider_job(ProviderJobRecord(
                job_id='job-1', asset_id='asset-1', provider='volcengine-asr-flash', model='bigmodel:volc.bigasr.auc_turbo',
                job_type='asr', status='completed', usage={'duration_seconds': 3.0}, cost_rmb=0.004,
                citation='trove://wechat/acct-a/conv/s1/1',
            ))
            self.assertEqual(json.loads(job['usage_json'])['duration_seconds'], 3.0)
            transcript = repo.insert_transcript(TranscriptRecord(
                transcript_id='tr-1', asset_id='asset-1', job_id='job-1', citation='trove://wechat/acct-a/conv/s1/1#voice',
                text='fixture transcript', confidence=0.91, duration_seconds=3.0,
            ))
            self.assertEqual(transcript['citation'], 'trove://wechat/acct-a/conv/s1/1#voice')

            entity = repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育', identifiers={'wechat_id': 'wxid-fixture'}))
            self.assertEqual(entity['entity_type'], 'Customer')
            obs = repo.add_observation(ObservationRecord(
                observation_id='obs-1', entity_id='customer-1', observation_type='Need', value={'text': 'needs budget approval'},
                status='active', confidence=0.8, citation='trove://wechat/acct-a/conv/s1/1#voice', source_type='transcript',
            ))
            self.assertEqual(obs['status'], 'active')
            rel = repo.add_relationship(RelationshipRecord(
                relationship_id='rel-1', subject_entity_id='customer-1', predicate='supports_claim', object_ref='obs-1',
                citation='trove://wechat/acct-a/conv/s1/1#voice', confidence=0.8,
            ))
            self.assertEqual(rel['predicate'], 'supports_claim')
            self.assertEqual(len(repo.active_observations('customer-1')), 1)

    def test_image_observation_and_validation_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'trove.sqlite'))
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-img', account_id='acct-a', source_type='moment', source_id='sns-1', modality='image', media_type='image', citation='trove://wechat/acct-a/moment/sns-1'))
            row = repo.insert_image_observation(ImageObservationRecord(
                observation_id='img-obs-1', asset_id='asset-img', citation='trove://wechat/acct-a/moment/sns-1#image',
                caption='fixture product screenshot', visible_text='报价单', objects=[{'label': 'screenshot'}],
                business_signals=[{'type': 'pricing'}], confidence=0.77, status='needs_review',
            ))
            self.assertEqual(json.loads(row['objects_json'])[0]['label'], 'screenshot')
            repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育'))
            with self.assertRaises(ValueError):
                repo.add_observation(ObservationRecord(
                    observation_id='bad', entity_id='customer-1', observation_type='Need', value={},
                    status='deleted', confidence=0.1, citation='trove://x', source_type='message',
                ))
            with self.assertRaises(ValueError):
                repo.record_provider_job(ProviderJobRecord(job_id='bad-job', provider='x', model='y', job_type='asr', status='lost'))


if __name__ == '__main__':
    unittest.main()
