from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.search.context import ContextService
from trove_core.store.repositories import ImageObservationRecord, MediaAssetRecord, MultimodalRepository, ProviderJobRecord, TranscriptRecord
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.models import Account, Conversation, Message


class MultimodalChunkRefreshTests(unittest.TestCase):
    def test_full_rebuild_and_reads_never_revive_non_cloud_transcripts(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            repo = MultimodalRepository(store)
            store.upsert_accounts([Account('acct-a', 'A', 'A')])
            store.upsert_conversations([Conversation('conv-a', 'acct-a', 'A private', 'private')])
            store.upsert_messages([
                Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 's', 1, content_kind='voice'),
                Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 2, tzinfo=timezone.utc), '[voice]', 's', 2, content_kind='voice'),
            ])
            valid_parent = 'trove://wechat/acct-a/conv-a/s/1#voice'
            fake_parent = 'trove://wechat/acct-a/conv-a/s/2#voice'
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-cloud', 'acct-a', 'message', 'm1', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/s/1', cache_state='cached', content_hash='a' * 64,
            ))
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-fake', 'acct-a', 'message', 'm2', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/s/2', cache_state='cached', content_hash='b' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-cloud', asset_id='asset-cloud', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='a' * 64, citation='trove://wechat/acct-a/conv-a/s/1',
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-fake', asset_id='asset-fake', provider='fake-asr',
                model='fixture', job_type='asr', status='completed',
                request_hash='b' * 64, citation='trove://wechat/acct-a/conv-a/s/2',
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-cloud', 'asset-cloud', valid_parent, 'validcloudunique transcript',
                job_id='job-cloud',
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-fake', 'asset-fake', fake_parent, 'fakeasrunique transcript',
                job_id='job-fake',
            ))
            fake_chunk = f'{fake_parent}#chunk-0'
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO evidence_chunks(
                           chunk_id,chunk_citation,parent_citation,account_id,account_label,
                           source_type,source_id,title,actor,timestamp,content,chunk_index,
                           metadata_json,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        fake_chunk, fake_chunk, fake_parent, '', 'Vault', 'transcript',
                        'transcript-fake', 'Voice transcript', 'Transcript', '',
                        'fakeasrunique transcript', 0, '{"family":"transcript"}', 'active',
                        '2026-01-01T00:00:00Z',
                    ),
                )
                conn.execute(
                    'INSERT INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)',
                    (fake_chunk, 'test', 1, '[0.0]', 'fake-hash'),
                )
                conn.commit()

            self.assertIsNone(store.evidence_by_citation(fake_parent))
            self.assertIsNone(store.evidence_by_citation(fake_chunk))
            self.assertNotIn(fake_parent, store.evidence_by_citations([fake_parent]))
            self.assertFalse(store.chunk_search('fakeasrunique', {'source_type': 'transcript'}, limit=5))
            self.assertFalse(store.vector_entries_for_search({'source_type': 'transcript'}, limit=10))
            self.assertNotIn(fake_chunk, {row['citation'] for row in store.iter_vector_documents()})
            hints = store.media_hints_for_citations([
                'trove://wechat/acct-a/conv-a/s/1',
                'trove://wechat/acct-a/conv-a/s/2',
            ])
            self.assertEqual(hints['trove://wechat/acct-a/conv-a/s/1']['transcript_state'], 'cached')
            self.assertEqual(hints['trove://wechat/acct-a/conv-a/s/2']['transcript_state'], 'pending')
            self.assertEqual(store.scope_status()['families']['transcript'], 1)

            store.rebuild_evidence_chunks()

            self.assertTrue(store.chunk_search('validcloudunique', {'source_type': 'transcript'}, limit=5))
            self.assertFalse(store.chunk_search('fakeasrunique', {'source_type': 'transcript'}, limit=5))
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    'SELECT COUNT(*) FROM evidence_chunks WHERE parent_citation=?', (fake_parent,),
                ).fetchone()[0], 0)
                self.assertEqual(conn.execute(
                    'SELECT COUNT(*) FROM vector_entries WHERE citation=?', (fake_chunk,),
                ).fetchone()[0], 0)

    def test_transcript_and_image_chunk_refresh_record_incremental_dirty_refs(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-v',
                account_id='acct-a',
                source_type='message',
                source_id='m1',
                modality='voice',
                media_type='voice',
                citation='trove://wechat/acct-a/conv-a/s1/1',
                content_hash='e' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-replace-cloud', asset_id='asset-v', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation='trove://wechat/acct-a/conv-a/s1/1',
            ))
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-i',
                account_id='acct-a',
                source_type='moment',
                source_id='moment-1',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/moment/moment-1',
            ))
            transcript_citation = 'trove://wechat/acct-a/conv-a/s1/1#voice'
            image_citation = 'trove://wechat/acct-a/moment/moment-1#image'

            repo.insert_transcript(TranscriptRecord(
                transcript_id='transcript-dirty',
                asset_id='asset-v',
                citation=transcript_citation,
                text='incremental transcript fixture',
                job_id='job-replace-cloud',
            ))
            repo.insert_image_observation(ImageObservationRecord(
                observation_id='image-dirty',
                asset_id='asset-i',
                citation=image_citation,
                caption='incremental image fixture',
            ))

            with store.connect() as conn:
                dirty = {
                    str(row['citation']): str(row['source_type'])
                    for row in conn.execute(
                        'SELECT citation,source_type FROM sync_dirty_citations WHERE citation IN (?,?)',
                        (transcript_citation, image_citation),
                    )
                }
            self.assertEqual(dirty, {
                transcript_citation: 'transcript',
                image_citation: 'image_observation',
            })

    def test_replacing_transcript_citation_removes_old_chunks_and_vectors(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-v',
                account_id='acct-a',
                source_type='message',
                source_id='m1',
                modality='voice',
                media_type='voice',
                citation='trove://wechat/acct-a/conv-a/s1/1',
                content_hash='e' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-replace-cloud', asset_id='asset-v', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation='trove://wechat/acct-a/conv-a/s1/1',
            ))
            old_parent = 'trove://wechat/acct-a/conv-a/s1/1#voice-old'
            new_parent = 'trove://wechat/acct-a/conv-a/s1/1#voice-new'
            repo.insert_transcript(TranscriptRecord(
                transcript_id='transcript-stable-id',
                asset_id='asset-v',
                citation=old_parent,
                text='olduniquealpha transcript before replacement',
                job_id='job-replace-cloud',
                confidence=0.8,
            ))
            old_chunk = f'{old_parent}#chunk-0'
            self.assertTrue(store.chunk_search('olduniquealpha', {'source_type': 'transcript'}, limit=5))
            with store.connect() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS vector_entries (
                        citation TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        vector_json TEXT NOT NULL,
                        content_hash TEXT
                    )"""
                )
                conn.execute(
                    'INSERT INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)',
                    (old_chunk, 'test', 1, '[0.0]', 'old-hash'),
                )
                conn.commit()

            repo.insert_transcript(TranscriptRecord(
                transcript_id='transcript-stable-id',
                asset_id='asset-v',
                citation=new_parent,
                text='newuniquebeta transcript after replacement',
                job_id='job-replace-cloud',
                confidence=0.9,
            ))

            self.assertFalse(store.chunk_search('olduniquealpha', {'source_type': 'transcript'}, limit=5))
            self.assertIsNone(ContextService(store).fetch(old_chunk)['evidence'])
            new_hits = store.chunk_search('newuniquebeta', {'source_type': 'transcript'}, limit=5)
            self.assertEqual([hit['parent_citation'] for hit in new_hits], [new_parent])
            with store.connect() as conn:
                self.assertEqual(
                    conn.execute('SELECT COUNT(*) FROM evidence_chunks WHERE parent_citation=?', (old_parent,)).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    conn.execute('SELECT COUNT(*) FROM vector_entries WHERE citation=?', (old_chunk,)).fetchone()[0],
                    0,
                )


if __name__ == '__main__':
    unittest.main()
