from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from trove_core.approvals import ApprovalManager
from trove_core.application.cloud_commands import cloud_voice_transcript_payload, execute_cloud_voice_transcript
from trove_core.asr.fake import FakeASRProvider
from trove_core.media_pipeline import ensure_voice_transcript, enqueue_media_jobs, media_status_payload, run_voice_transcription_budget, voice_transcription_plan
from trove_core.providers.config import ProviderConfig
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository, ProviderJobRecord, TranscriptRecord, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.media.materializer import MaterializationResult
from trove_core.wechat.models import Account, Conversation, Message


class _CloudASRProvider(FakeASRProvider):
    name = 'volcengine-asr-flash'
    egress_kind = 'cloud_asr_upload'

    def __init__(self, text: str = 'fixture transcript', duration_seconds: float = 1.0):
        super().__init__(text=text, duration_seconds=duration_seconds)
        cfg = ProviderConfig.resolve()
        self.model_name = cfg.asr_model_name
        self.resource_id = cfg.asr_resource_id
        self.endpoint = cfg.asr_endpoint


def _execute_approved_cloud(
    vault: Path,
    citation: str,
    provider: _CloudASRProvider,
    *,
    allow_group_voice: bool = False,
):
    payload = cloud_voice_transcript_payload(citation)
    grant = ApprovalManager(vault).require(
        'voice_cloud_asr', 'cloud_asr_upload', payload, one_step_approval=True,
    )
    with patch(
        'trove_core.application.cloud_commands._cloud_asr_provider_from_runtime',
        return_value=(provider, None),
    ):
        return execute_cloud_voice_transcript(
            vault,
            citation=citation,
            approval_grant=grant,
            allow_group_voice=allow_group_voice,
        )


class MediaTranscribePipelineTests(unittest.TestCase):
    def test_voice_materialization_file_work_runs_before_writer_publication(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-materialize', 'acct-a', 'message', 'msg', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/message_0/1', path_ref='', cache_state='metadata_only',
            ))
            prepared = vault / 'media' / 'materialized' / 'fixture.wav'
            prepared.parent.mkdir(parents=True)
            prepared.write_bytes(b'RIFFxxxxWAVEfixture')
            lock_depth = 0
            materialize_lock_states: list[bool] = []

            import trove_core.media_pipeline as pipeline
            original_mutation = pipeline.coordinated_vault_mutation

            @contextmanager
            def tracked_mutation(*args, **kwargs):
                nonlocal lock_depth
                with original_mutation(*args, **kwargs) as session:
                    lock_depth += 1
                    try:
                        yield session
                    finally:
                        lock_depth -= 1

            def prepared_result(*_args, **_kwargs):
                materialize_lock_states.append(lock_depth > 0)
                return MaterializationResult(
                    True, 'materialized', 'asset-materialize', prepared,
                    str(prepared.relative_to(vault)), 'fixture-hash', 'audio/wav', 'fixture',
                    bytes_written=prepared.stat().st_size,
                )

            with patch.object(pipeline, 'coordinated_vault_mutation', tracked_mutation), patch.object(
                pipeline, 'materialize_media_asset', side_effect=prepared_result,
            ):
                result = ensure_voice_transcript(
                    vault,
                    citation='trove://wechat/acct-a/conv-a/message_0/1',
                    allow_local_asr=False,
                )

            self.assertEqual(materialize_lock_states, [False])
            self.assertEqual(result['status'], 'pending_transcript')
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT path_ref FROM media_assets WHERE asset_id='asset-materialize'",
                ).fetchone()[0], str(prepared.relative_to(vault)))

    def test_concurrent_voice_claim_calls_provider_once(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-concurrent', 'acct-a', 'message', 'msg-concurrent', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/message_0/1', path_ref='sources/voice.wav', cache_state='cached',
            ))
            entered = threading.Event()
            release = threading.Event()

            class BlockingProvider(_CloudASRProvider):
                def __init__(self):
                    super().__init__(text='并发转写')
                    self.calls = 0

                def transcribe(self, request):
                    self.calls += 1
                    entered.set()
                    self.assert_release = release.wait(timeout=5)
                    return super().transcribe(request)

            provider = BlockingProvider()
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(_execute_approved_cloud, vault, citation, provider)
                self.assertTrue(entered.wait(timeout=5))
                second = executor.submit(_execute_approved_cloud, vault, citation, provider).result(timeout=5)
                release.set()
                first = first_future.result(timeout=5)

            self.assertTrue(provider.assert_release)
            self.assertEqual(provider.calls, 1)
            self.assertEqual(first['status'], 'completed')
            self.assertEqual(second['status'], 'in_progress')
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-concurrent'").fetchone()[0], 1)

    def test_budget_reclaims_a_stale_running_voice_job_after_worker_crash(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-stale-running', 'acct-a', 'message', 'msg-stale', 'voice', 'voice',
                citation, path_ref='sources/voice.wav', cache_state='cached',
            ))
            enqueue_media_jobs(store, modalities={'voice'}, asset_ids=['asset-stale-running'])

            with patch('trove_core.media_pipeline._commit_voice_inference', side_effect=SystemExit('simulated crash')):
                with self.assertRaises(SystemExit):
                    _execute_approved_cloud(vault, citation, _CloudASRProvider(text='discarded'))
            with store.connect() as conn:
                conn.execute("UPDATE provider_jobs SET updated_at='2020-01-01T00:00:00Z' WHERE asset_id='asset-stale-running'")
                conn.execute("UPDATE media_jobs SET updated_at='2020-01-01T00:00:00Z' WHERE asset_id='asset-stale-running'")
                conn.commit()
            self.assertEqual(media_status_payload(vault)['backlog'], 1)

            result = _execute_approved_cloud(
                vault, citation, _CloudASRProvider(text='reclaimed transcript'),
            )

            self.assertEqual(result['status'], 'completed')
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-stale-running'",
                ).fetchone()[0], 1)
                self.assertEqual(conn.execute(
                    "SELECT status FROM media_jobs WHERE asset_id='asset-stale-running'",
                ).fetchone()[0], 'done')

    def test_voice_commit_cas_discards_result_when_source_changes(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            replacement = vault / 'sources' / 'replacement.wav'
            replacement.write_bytes(b'RIFFreplacement')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-cas', 'acct-a', 'message', 'msg-cas', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/message_0/1', path_ref='sources/voice.wav', cache_state='cached',
            ))

            class MutatingProvider(_CloudASRProvider):
                def transcribe(self, request):
                    with store.connect() as conn:
                        conn.execute(
                            "UPDATE media_assets SET path_ref='sources/replacement.wav',updated_at=datetime('now') WHERE asset_id='asset-cas'",
                        )
                        conn.commit()
                    return super().transcribe(request)

            result = _execute_approved_cloud(
                vault, 'trove://wechat/acct-a/conv-a/message_0/1',
                MutatingProvider(text='不得提交'),
            )

            self.assertEqual(result['status'], 'retryable_failure')
            self.assertEqual(result['reason'], 'voice_source_changed')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 0)

    def test_profile_approval_does_not_upload_bytes_changed_after_approval(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture-before-approval')
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-approved-source', 'acct-a', 'message', 'msg-approved', 'voice', 'voice',
                citation, path_ref='sources/voice.wav', cache_state='cached',
            ))
            prepared = ensure_voice_transcript(
                vault, citation=citation, estimate_cloud_asr_cost=True,
            )
            scope = {
                'profile_run_hash': 'run-hash',
                'task_set_hash': 'task-hash',
                'citation_set_hash': 'citation-hash',
                'source_revision_hash': 'source-revision-hash',
                'content_hash': prepared['content_sha256'],
                'actor_hash': 'actor-hash',
                'session_hash': 'session-hash',
                'purpose': 'person_relationship_profile_enrichment',
                'cost_ceiling_rmb': None,
            }
            payload = cloud_voice_transcript_payload(citation, profile_scope=scope)
            grant = ApprovalManager(vault).require(
                'voice_cloud_asr', 'cloud_asr_upload', payload, one_step_approval=True,
            )
            audio.write_bytes(b'RIFFfixture-after-approval')

            class TrackingProvider(_CloudASRProvider):
                def __init__(self):
                    super().__init__('must not upload')
                    self.calls = 0

                def transcribe(self, request):
                    self.calls += 1
                    return super().transcribe(request)

            provider = TrackingProvider()
            with patch(
                'trove_core.application.cloud_commands._cloud_asr_provider_from_runtime',
                return_value=(provider, None),
            ):
                result = execute_cloud_voice_transcript(
                    vault, citation=citation, approval_grant=grant, profile_scope=scope,
                )

            self.assertEqual(result['status'], 'retryable_failure')
            self.assertEqual(result['reason'], 'voice_source_changed')
            self.assertFalse(result['cloud_calls_made'])
            self.assertEqual(provider.calls, 0)

    def test_budget_path_never_constructs_or_calls_local_asr(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-no-local', 'acct-a', 'message', 'msg-no-local', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/message_0/1', path_ref='sources/voice.wav', cache_state='cached',
            ))
            enqueue_media_jobs(store, modalities={'voice'}, asset_ids=['asset-no-local'])

            result = run_voice_transcription_budget(vault, budget=1)

            self.assertFalse(result['ok'])
            self.assertEqual(result['processed'], 0)
            self.assertEqual(result['errors'], {'cloud_asr_approval_required': 1})
            self.assertTrue(result['provider']['cloud_only'])
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM provider_jobs').fetchone()[0], 0)

    def test_lazy_voice_does_not_cross_match_ambiguous_local_id(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('conv-private', 'acct-a', 'private', 'private'),
                    Conversation('conv-group', 'acct-a', 'group', 'group', member_count=3),
                ],
                [
                    Message('acct-a', 'A', 'conv-private', 'private', 'private', 'u1', 'one', datetime(2026, 1, 1, tzinfo=timezone.utc), 'private', 'message_0', 7),
                    Message('acct-a', 'A', 'conv-group', 'group', 'group', 'u2', 'two', datetime(2026, 1, 1, tzinfo=timezone.utc), 'group', 'message_0', 7),
                ],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'voice-ambiguous', 'acct-a', 'message', 'legacy', 'voice', 'voice',
                'trove://wechat/acct-a/media/legacy/7',
                metadata={'message_local_id': 7},
            ))

            result = ensure_voice_transcript(
                vault,
                citation='trove://wechat/acct-a/media/legacy/7',
                provider=FakeASRProvider(text='must not run'),
            )

            self.assertEqual(result['status'], 'skipped')
            self.assertEqual(result['reason'], 'out_of_scope')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 0)

    def test_active_transcript_is_unique_per_asset_and_refreshes_only_affected_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-one-active', 'acct', 'message', 'msg', 'voice', 'voice', 'trove://voice/1',
                content_hash='e' * 64,
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-old-cloud', asset_id='asset-one-active', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation='trove://voice/1',
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-new-cloud', asset_id='asset-one-active', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation='trove://voice/1',
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-old', 'asset-one-active', 'trove://voice/1#voice', '旧转写',
                job_id='job-old-cloud', status='active',
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-new', 'asset-one-active', 'trove://voice/1#voice', '新转写',
                job_id='job-new-cloud', status='active',
            ))
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-one-active' AND status='active'").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT status FROM transcripts WHERE transcript_id='transcript-old'").fetchone()[0], 'superseded')
                chunks = list(conn.execute("SELECT content FROM evidence_chunks WHERE source_type='transcript' AND status='active'"))
            self.assertEqual([row['content'] for row in chunks], ['新转写'])

    def test_invalid_cloud_response_preserves_existing_local_projection(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-preserve-local', 'acct-a', 'message', 'msg-preserve', 'voice', 'voice',
                citation, path_ref='sources/voice.wav', cache_state='cached',
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-preserve-local', asset_id='asset-preserve-local', provider='local-faster-whisper',
                model='base:local', job_type='asr', status='completed', citation=citation,
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-preserve-local', 'asset-preserve-local', citation + '#voice',
                '旧本地转写仅用于失败保留', job_id='job-preserve-local',
            ))

            result = _execute_approved_cloud(
                vault, citation, _CloudASRProvider(text=''),
            )

            self.assertEqual(result['status'], 'terminal_failure')
            self.assertEqual(result['error_code'], 'cloud_asr_invalid_response')
            with store.connect() as conn:
                active = conn.execute(
                    "SELECT transcript_id FROM transcripts WHERE asset_id=? AND status='active'",
                    ('asset-preserve-local',),
                ).fetchone()
                self.assertEqual(active['transcript_id'], 'transcript-preserve-local')
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM transcripts WHERE asset_id=?",
                    ('asset-preserve-local',),
                ).fetchone()[0], 1)
                evidence = list(conn.execute(
                    "SELECT content FROM evidence_chunks WHERE source_type='transcript' AND status='active'",
                ))
                self.assertEqual(evidence, [])
            self.assertEqual(
                store.chunk_search('旧本地转写仅用于失败保留', filters={'source_type': 'transcript'}, limit=3),
                [],
            )

    def test_approved_cloud_voice_pipeline_writes_searchable_chunk_once(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '语音消息', 'message_0', 1)],
            )
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-voice-1',
                account_id='acct-a',
                source_type='message',
                source_id='msg-1',
                modality='voice',
                media_type='voice',
                citation='trove://wechat/acct-a/conv-a/message_0/1',
                path_ref='sources/voice.wav',
                cache_state='cached',
            ))
            enqueue_media_jobs(store, modalities={'voice'}, asset_ids=['asset-voice-1'])

            first = _execute_approved_cloud(
                vault,
                'trove://wechat/acct-a/conv-a/message_0/1',
                _CloudASRProvider(text='语音试点转写needle'),
            )
            second = ensure_voice_transcript(
                vault, citation='trove://wechat/acct-a/conv-a/message_0/1',
            )

            self.assertEqual(first['status'], 'completed')
            self.assertEqual(second['status'], 'cached')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE source_type='transcript'").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT status FROM media_jobs WHERE job_type='voice_transcribe'").fetchone()[0], 'done')
            hits = store.chunk_search('语音试点转写needle', filters={'source_type': 'transcript'}, limit=3)
            self.assertEqual(len(hits), 1)
            self.assertFalse(first['raw_content_included'])
            self.assertTrue(first['cloud_calls_made'])

    def test_zero_budget_returns_without_queue_scan_or_writer(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            provider = FakeASRProvider(text='must not run')
            with patch('trove_core.media_pipeline.enqueue_media_jobs', side_effect=AssertionError('queue scan used')), patch(
                'trove_core.media_pipeline.coordinated_vault_mutation',
                side_effect=AssertionError('writer used'),
            ):
                result = run_voice_transcription_budget(vault, budget=0, provider=provider)

            self.assertEqual(result['processed'], 0)
            self.assertEqual(result['requested_budget'], 0)

    def test_voice_transcription_can_be_limited_to_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            (vault / 'sources').mkdir(parents=True, exist_ok=True)
            (vault / 'sources' / 'a.wav').write_bytes(b'RIFFfixture-a')
            (vault / 'sources' / 'b.wav').write_bytes(b'RIFFfixture-b')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [
                    Conversation('conv-a', 'acct-a', 'A private', 'private'),
                    Conversation('conv-b', 'acct-a', 'B private', 'private'),
                ],
                [
                    Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户A', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice'),
                    Message('acct-a', 'A', 'conv-b', 'B private', 'private', 'u2', '客户B', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 2, content_kind='voice'),
                ],
            )
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord('asset-a', 'acct-a', 'message', 'msg-a', 'voice', 'voice', 'trove://wechat/acct-a/conv-a/message_0/1', path_ref='sources/a.wav', cache_state='cached'))
            repo.upsert_media_asset(MediaAssetRecord('asset-b', 'acct-a', 'message', 'msg-b', 'voice', 'voice', 'trove://wechat/acct-a/conv-b/message_0/2', path_ref='sources/b.wav', cache_state='cached'))

            plan = voice_transcription_plan(vault, conversation_id='conv-a')
            with patch('trove_core.media_pipeline.enqueue_media_jobs', wraps=enqueue_media_jobs) as enqueue:
                result = run_voice_transcription_budget(vault, budget=5, conversation_id='conv-a')

            self.assertEqual(plan['pending'], 1)
            self.assertEqual(enqueue.call_args.kwargs['asset_ids'], ['asset-a'])
            self.assertEqual(result['conversation_id'], 'conv-a')
            self.assertEqual(result['processed'], 0)
            self.assertEqual(result['errors'], {'cloud_asr_approval_required': 1})
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 0)

    def test_lazy_voice_transcript_does_not_upload_without_explicit_allow(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord('asset-lazy', 'acct-a', 'message', 'msg-lazy', 'voice', 'voice', 'trove://wechat/acct-a/conv-a/message_0/1', path_ref='sources/voice.wav', cache_state='cached'))

            from trove_core.media_pipeline import ensure_voice_transcript
            pending = ensure_voice_transcript(vault, citation='trove://wechat/acct-a/conv-a/message_0/1')
            done = _execute_approved_cloud(
                vault, 'trove://wechat/acct-a/conv-a/message_0/1',
                _CloudASRProvider(text='懒加载转写needle'),
            )
            cached = ensure_voice_transcript(vault, citation='trove://wechat/acct-a/conv-a/message_0/1')

            self.assertTrue(pending['ok'])
            self.assertEqual(pending['status'], 'pending_transcript')
            self.assertFalse(pending['cloud_calls_made'])
            self.assertTrue(done['ok'])
            self.assertEqual(done['status'], 'completed')
            self.assertEqual(cached['status'], 'cached')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts').fetchone()[0], 1)

    def test_lazy_voice_ignores_local_asr_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', 'A private', 'private')],
                [Message('acct-a', 'A', 'conv-a', 'A private', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                'asset-local-lazy', 'acct-a', 'message', 'msg-local', 'voice', 'voice',
                'trove://wechat/acct-a/conv-a/message_0/1', path_ref='sources/voice.wav', cache_state='cached',
            ))
            from trove_core.media_pipeline import ensure_voice_transcript
            result = ensure_voice_transcript(
                vault,
                citation='trove://wechat/acct-a/conv-a/message_0/1',
                allow_local_asr=True,
                env={'TROVE_ENABLE_LOCAL_ASR': '1'},
            )

            self.assertEqual(result['status'], 'pending_transcript')
            self.assertEqual(result['reason'], 'cloud_asr_not_requested')
            self.assertFalse(result['cloud_calls_made'])
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-local-lazy'").fetchone()[0], 0)

    def test_group_voice_requires_explicit_scoped_local_path_and_never_enters_global_queue(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'group.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFfixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('room@chatroom', 'acct-a', '群聊', 'group')],
                [Message('acct-a', 'A', 'room@chatroom', '群聊', 'group', 'u1', '群友', datetime(2026, 1, 1, tzinfo=timezone.utc), '[voice]', 'message_0', 1, content_kind='voice')],
            )
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord('asset-group', 'acct-a', 'message', 'msg-group', 'voice', 'voice', 'trove://wechat/acct-a/room@chatroom/message_0/1', path_ref='sources/group.wav', cache_state='cached'))
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO media_jobs(
                           job_id,asset_id,job_type,status,retry_count,error_code,last_duration_ms,created_at,updated_at
                       ) VALUES(
                           'job-group','asset-group','voice_transcribe','pending',0,NULL,0,datetime('now'),datetime('now')
                       )"""
                )
                conn.commit()

            from trove_core.media_pipeline import ensure_voice_transcript
            citation = 'trove://wechat/acct-a/room@chatroom/message_0/1'
            queued_before = enqueue_media_jobs(store, modalities={'voice'}, asset_ids=['asset-group'])
            result = ensure_voice_transcript(
                vault,
                citation=citation,
            )

            self.assertFalse(result['ok'])
            self.assertEqual(result['reason'], 'out_of_scope')
            self.assertFalse(result['cloud_calls_made'])
            self.assertEqual(queued_before['seen'], 0)
            self.assertEqual(queued_before['queued'], 0)
            self.assertEqual(queued_before['skipped_out_of_scope'], 1)

            scoped = _execute_approved_cloud(
                vault, citation, _CloudASRProvider(text='人物画像群聊语音'),
                allow_group_voice=True,
            )
            still_scoped_out = ensure_voice_transcript(vault, citation=citation)
            queued_after = enqueue_media_jobs(store, modalities={'voice'}, asset_ids=['asset-group'])

            self.assertEqual(scoped['status'], 'completed')
            self.assertTrue(scoped['cloud_calls_made'])
            self.assertEqual(still_scoped_out['reason'], 'out_of_scope')
            self.assertEqual(queued_after['seen'], 0)
            self.assertEqual(queued_after['queued'], 0)
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-group' AND status='active'",
                ).fetchone()[0], 1)
                job = conn.execute(
                    "SELECT status,error_code FROM media_jobs WHERE asset_id='asset-group'",
                ).fetchone()
                self.assertEqual((job['status'], job['error_code']), ('skipped', 'out_of_scope'))


if __name__ == '__main__':
    unittest.main()
