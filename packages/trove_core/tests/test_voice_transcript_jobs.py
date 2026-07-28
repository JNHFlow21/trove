from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.asr.base import ASRProvider, ASRRequest, ASRResult, ASRUsage
from trove_core.asr.jobs import run_voice_transcript_job
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore


class FakeASRProvider(ASRProvider):
    name = 'fake-asr'
    model_name = 'bigmodel'
    resource_id = 'volc.bigasr.auc_turbo'
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = 0
    def transcribe(self, request: ASRRequest) -> ASRResult:
        self.calls += 1
        if self.fail:
            raise self.fail
        return ASRResult('fixture transcript', 'zh', 0.9, ASRUsage(duration_seconds=8, estimated_cost_rmb=0.01), citations=[request.citation] if request.citation else [])


class VoiceTranscriptJobsTests(unittest.TestCase):
    def test_success_writes_job_usage_and_transcript_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            repo.upsert_media_asset(MediaAssetRecord(asset_id='asset-v', account_id='acct-a', source_type='message', source_id='msg1', modality='voice', media_type='voice', citation='trove://wechat/acct-a/conv/s1/1'))
            audio = root / 'voice.wav'; audio.write_bytes(b'RIFFfixture')
            provider = FakeASRProvider()
            first = run_voice_transcript_job(repo, asset_id='asset-v', audio_path=audio, provider=provider, citation='trove://wechat/acct-a/conv/s1/1')
            second = run_voice_transcript_job(repo, asset_id='asset-v', audio_path=audio, provider=provider, citation='trove://wechat/acct-a/conv/s1/1')
            self.assertEqual(first['status'], 'completed')
            self.assertTrue(second['idempotent'])
            self.assertEqual(provider.calls, 1)
            with repo.store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 1)
                self.assertEqual(conn.execute('SELECT cost_rmb FROM provider_jobs').fetchone()[0], 0.01)

    def test_timeout_records_retryable_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            audio = root / 'voice.wav'; audio.write_bytes(b'RIFFfixture')
            result = run_voice_transcript_job(repo, asset_id='asset-v', audio_path=audio, provider=FakeASRProvider(TimeoutError()), citation='trove://c')
            self.assertEqual(result['status'], 'retryable_failure')
            with repo.store.connect() as conn:
                self.assertEqual(conn.execute('SELECT status FROM provider_jobs').fetchone()[0], 'retryable_failure')


if __name__ == '__main__':
    unittest.main()
