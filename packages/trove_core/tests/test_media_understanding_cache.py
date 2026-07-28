from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import MediaUnderstandingRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore


SHA_A = 'a' * 64


class MediaUnderstandingCacheTests(unittest.TestCase):
    def test_upsert_keys_by_content_sha_and_appends_citations(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            first = repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='fixture caption',
                visible_text='fixture text',
                objects=[{'label': 'poster'}],
                business_signals=[{'type': 'pricing'}],
                model_id='agent-fixture-v1',
                prompt_version='p1',
                confidence=0.8,
                source_citations=['trove://fixture/private#image-0'],
            ))
            self.assertEqual(first['content_sha256'], SHA_A)
            self.assertEqual(json.loads(first['source_citations_json']), ['trove://fixture/private#image-0'])

            second = repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='fixture caption',
                visible_text='fixture text',
                objects=[{'label': 'poster'}],
                business_signals=[{'type': 'pricing'}],
                model_id='agent-fixture-v1',
                prompt_version='p1',
                confidence=0.8,
                source_citations=['trove://fixture/moment#image-0'],
            ))
            self.assertEqual(json.loads(second['source_citations_json']), [
                'trove://fixture/private#image-0',
                'trove://fixture/moment#image-0',
            ])
            self.assertEqual(json.loads(second['metadata_json']).get('history'), None)

    def test_model_upgrade_overwrites_current_and_records_history(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='old caption',
                model_id='agent-fixture-v1',
                prompt_version='p1',
                source_citations=['trove://fixture/private#image-0'],
            ))
            row = repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='new caption',
                model_id='agent-fixture-v2',
                prompt_version='p2',
                source_citations=['trove://fixture/private#image-0'],
            ))
            self.assertEqual(row['caption'], 'new caption')
            self.assertEqual(row['model_id'], 'agent-fixture-v2')
            history = json.loads(row['metadata_json'])['history']
            self.assertEqual(history[-1]['caption'], 'old caption')
            self.assertEqual(history[-1]['model_id'], 'agent-fixture-v1')


    def test_replace_flag_clears_explicit_empty_fields_by_hash(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='old caption',
                visible_text='old text',
                objects=[{'label': 'poster'}],
                business_signals=[{'type': 'pricing'}],
                keyframes=[{'time_seconds': 1, 'description': 'old'}],
                audio_transcript='old audio',
                confidence=0.6,
                model_id='agent-fixture-v1',
                prompt_version='p1',
                source_citations=['trove://fixture/private#image-0'],
            ))
            preserved = repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='',
                visible_text='',
                objects=[],
                business_signals=[],
                keyframes=[],
                audio_transcript='',
                confidence=0.0,
                model_id='agent-fixture-v1',
                prompt_version='p1',
                source_citations=['trove://fixture/private#image-0'],
            ))
            self.assertEqual(preserved['caption'], 'old caption')
            self.assertEqual(json.loads(preserved['objects_json'])[0]['label'], 'poster')
            self.assertEqual(float(preserved['confidence']), 0.6)

            cleared = repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=SHA_A,
                modality='image',
                caption='',
                visible_text='',
                objects=[],
                business_signals=[],
                keyframes=[],
                audio_transcript='',
                confidence=0.0,
                model_id='agent-fixture-v1',
                prompt_version='p1',
                source_citations=['trove://fixture/private#image-0'],
                replace=True,
            ))
            self.assertEqual(cleared['caption'], '')
            self.assertEqual(cleared['visible_text'], '')
            self.assertEqual(json.loads(cleared['objects_json']), [])
            self.assertEqual(json.loads(cleared['business_signals_json']), [])
            self.assertEqual(json.loads(cleared['keyframes_json']), [])
            self.assertEqual(cleared['audio_transcript'], '')
            self.assertEqual(float(cleared['confidence']), 0.0)

    def test_requires_content_hash_model_and_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            with self.assertRaises(ValueError):
                repo.upsert_media_understanding(MediaUnderstandingRecord(
                    content_sha256='not-a-sha',
                    modality='image',
                    model_id='agent-fixture-v1',
                    prompt_version='p1',
                ))
            with self.assertRaises(ValueError):
                repo.upsert_media_understanding(MediaUnderstandingRecord(
                    content_sha256=SHA_A,
                    modality='image',
                    model_id='',
                    prompt_version='p1',
                ))
            with self.assertRaises(ValueError):
                repo.upsert_media_understanding(MediaUnderstandingRecord(
                    content_sha256=SHA_A,
                    modality='image',
                    model_id='agent-fixture-v1',
                    prompt_version='',
                ))


if __name__ == '__main__':
    unittest.main()
