from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trove_core.application.cloud_commands import (
    cloud_voice_transcript_payload,
    execute_cloud_vector_index,
    execute_cloud_voice_transcript,
)
from trove_core.approvals import ApprovalManager, ApprovalValidationError, claim_approval_grant
from trove_core.asr.base import ASRProvider, ASRRequest, ASRResult, ASRUsage
from trove_core.asr.jobs import run_voice_transcript_job
from trove_core.asr.volcengine_flash import VolcengineASRFlashProvider
from trove_core.providers.config import (
    DEFAULT_ASR_ENDPOINT,
    DEFAULT_ASR_MODEL_NAME,
    DEFAULT_ASR_RESOURCE_ID,
)
from trove_core.runtime import index_vectors, vector_cloud_approval_payload
from trove_core.security.egress import cloud_asr_payload, cloud_embedding_payload, cloud_vision_payload
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vision.base import ImageObservationResult, VisionProvider, VisionRequest, VisionUsage
from trove_core.vision.jobs import run_image_observation_job
from trove_core.vision.volcengine_ark import VolcengineArkVisionProvider
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.models import Account, Conversation, Message


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_count(store: SQLiteStore, table: str) -> int:
    with store.connect() as conn:
        return int(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])


class FakeCloudASRProvider(ASRProvider):
    name = 'volcengine-asr-flash'
    model_name = DEFAULT_ASR_MODEL_NAME
    resource_id = DEFAULT_ASR_RESOURCE_ID
    egress_kind = 'cloud_asr_upload'

    def __init__(self) -> None:
        self.endpoint = DEFAULT_ASR_ENDPOINT
        self.calls = 0

    def transcribe(self, request: ASRRequest) -> ASRResult:
        self.calls += 1
        return ASRResult(
            text='synthetic approved transcript',
            language='zh',
            confidence=1.0,
            usage=ASRUsage(duration_seconds=1.0, estimated_cost_rmb=0.0),
            citations=[request.citation] if request.citation else [],
        )


class FakeCloudVisionProvider(VisionProvider):
    name = 'synthetic-cloud-vision'
    model = 'synthetic-vision-v1'
    egress_kind = 'cloud_vision_upload'

    def __init__(self) -> None:
        self.endpoint = 'https://example.invalid/vision'
        self.calls = 0

    def observe(self, request: VisionRequest) -> ImageObservationResult:
        self.calls += 1
        return ImageObservationResult(
            caption='synthetic',
            visible_text='',
            objects=[],
            business_signals=[],
            entity_mentions=[],
            confidence=1.0,
            usage=VisionUsage(),
            citations=[request.citation] if request.citation else [],
        )


class FakeCloudEmbeddingProvider:
    name = 'synthetic-cloud:embedding-v1'
    provider_name = 'synthetic-cloud'
    model = 'embedding-v1'
    dimensions = 4
    endpoint = 'https://example.invalid/embeddings'
    egress_kind = 'cloud_embedding_upload'

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0, 0.0, 0.0]


class CloudEgressApprovalTests(unittest.TestCase):
    def test_cloud_vector_full_digest_never_runs_under_writer_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeCloudEmbeddingProvider()
            payload = vector_cloud_approval_payload(
                cfg, provider, backend='sqlite', batch_size=16,
                max_messages=2, purge=False,
            )
            grant = ApprovalManager(cfg.root).require(
                'cloud_vector_index', 'cloud_embedding_upload', payload,
                one_step_approval=True,
            )

            import trove_core.runtime as runtime
            original_mutation = runtime.coordinated_vault_mutation
            original_digest = runtime._vector_cloud_approval_payload_and_snapshot
            lock_depth = 0
            digest_lock_states: list[bool] = []
            provider_lock_states: list[bool] = []
            original_embed = provider.embed

            def tracked_embed(text: str):
                provider_lock_states.append(lock_depth > 0)
                return original_embed(text)

            provider.embed = tracked_embed  # type: ignore[method-assign]

            @contextmanager
            def tracked_mutation(*args, **kwargs):
                nonlocal lock_depth
                with original_mutation(*args, **kwargs) as session:
                    lock_depth += 1
                    try:
                        yield session
                    finally:
                        lock_depth -= 1

            def tracked_digest(*args, **kwargs):
                digest_lock_states.append(lock_depth > 0)
                return original_digest(*args, **kwargs)

            with patch.object(runtime, 'coordinated_vault_mutation', tracked_mutation), patch.object(
                runtime, '_vector_cloud_approval_payload_and_snapshot', tracked_digest,
            ):
                result = execute_cloud_vector_index(
                    cfg.root,
                    provider=provider,
                    approval_grant=grant,
                    backend='sqlite',
                    batch_size=16,
                    max_messages=2,
                )

            self.assertTrue(result['ok'])
            self.assertEqual(digest_lock_states, [False, False])
            self.assertTrue(provider_lock_states)
            self.assertEqual(set(provider_lock_states), {False})

    def test_cloud_vector_snapshot_cas_rejects_change_before_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeCloudEmbeddingProvider()
            payload = vector_cloud_approval_payload(
                cfg, provider, backend='sqlite', batch_size=16,
                max_messages=2, purge=False,
            )
            grant = ApprovalManager(cfg.root).require(
                'cloud_vector_index', 'cloud_embedding_upload', payload,
                one_step_approval=True,
            )
            claim_approval_grant(
                grant,
                cfg.root,
                action='cloud_vector_index',
                danger_class='cloud_embedding_upload',
                payload=payload,
            )

            import trove_core.runtime as runtime
            original_mutation = runtime.coordinated_vault_mutation

            @contextmanager
            def mutate_before_commit(*args, **kwargs):
                with original_mutation(*args, **kwargs) as session:
                    with SQLiteStore(cfg.paths.sqlite_path).connect() as conn:
                        conn.execute("UPDATE messages SET content=content || ' changed' WHERE id=(SELECT MIN(id) FROM messages)")
                        conn.commit()
                    yield session

            with patch.object(runtime, 'coordinated_vault_mutation', mutate_before_commit):
                with self.assertRaises(ApprovalValidationError) as raised:
                    index_vectors(
                        cfg,
                        provider,
                        backend='sqlite',
                        batch_size=16,
                        max_messages=2,
                        approval_grant=grant,
                        approval_payload=payload,
                    )

            self.assertEqual(raised.exception.code, 'vector_source_snapshot_changed')
            self.assertEqual(provider.calls, 0)

    def test_provider_request_response_logging_is_disabled_and_raw_extras_are_not_emitted(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode('utf-8')

        log_output = io.StringIO()
        handler = logging.StreamHandler(log_output)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            asr = VolcengineASRFlashProvider(
                api_key='fixture',
                urlopen=lambda *_args, **_kwargs: Response({
                    'text': 'synthetic parsed transcript',
                    'unused_raw_marker': 'provider-response-marker-asr',
                }),
            )
            vision = VolcengineArkVisionProvider(
                api_key='fixture',
                urlopen=lambda *_args, **_kwargs: Response({
                    'output_text': json.dumps({
                        'caption': 'synthetic', 'visible_text': '', 'objects': [],
                        'business_signals': [], 'entity_mentions': [], 'confidence': 1.0,
                    }),
                    'unused_raw_marker': 'provider-response-marker-vision',
                }),
            )
            with tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / 'voice.wav'; audio.write_bytes(b'RIFFsynthetic')
                image = Path(directory) / 'image.jpg'; image.write_bytes(b'\xff\xd8\xffsynthetic')
                asr.transcribe(ASRRequest(asset_id='a', audio_path=audio, citation='trove://fixture/asr'))
                vision.observe(VisionRequest(asset_id='i', image_path=image, citation='trove://fixture/vision'))
        finally:
            root_logger.removeHandler(handler)
        self.assertFalse(asr.request_response_logging)
        self.assertFalse(vision.request_response_logging)
        self.assertNotIn('provider-response-marker-asr', log_output.getvalue())
        self.assertNotIn('provider-response-marker-vision', log_output.getvalue())

    def test_direct_cloud_asr_sink_rejects_unclaimed_grant_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = VaultConfig.resolve(str(Path(directory) / 'vault'), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            audio = cfg.root / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFsynthetic')
            provider = FakeCloudASRProvider()
            citation = 'trove://wechat/synthetic/private/message_0/1'
            payload = cloud_asr_payload(
                citation=citation,
                provider=provider.name,
                model=provider.model_name,
                resource_id=provider.resource_id,
                endpoint=provider.endpoint,
            )
            grant = ApprovalManager(cfg.root).require(
                'voice_cloud_asr',
                'cloud_asr_upload',
                payload,
                one_step_approval=True,
            )
            before_db = _sha256(cfg.paths.sqlite_path)
            before_audio = _sha256(audio)

            with self.assertRaises(ApprovalValidationError) as raised:
                run_voice_transcript_job(
                    repo,
                    asset_id='synthetic-audio',
                    audio_path=audio,
                    provider=provider,
                    citation=citation,
                    approval_grant=grant,
                    approval_payload=payload,
                )

            self.assertEqual(raised.exception.code, 'approval_grant_unclaimed')
            self.assertEqual(provider.calls, 0)
            self.assertEqual(_sha256(audio), before_audio)
            self.assertEqual(_sha256(cfg.paths.sqlite_path), before_db)
            self.assertEqual(_table_count(store, 'provider_jobs'), 0)
            self.assertEqual(_table_count(store, 'transcripts'), 0)

    def test_direct_cloud_vision_sink_rejects_unclaimed_grant_without_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = VaultConfig.resolve(str(Path(directory) / 'vault'), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            image = cfg.root / 'sources' / 'image.jpg'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'\xff\xd8\xffsynthetic')
            provider = FakeCloudVisionProvider()
            citation = 'trove://wechat/synthetic/private/message_0/2#image-0'
            payload = cloud_vision_payload(
                citation=citation,
                provider=provider.name,
                model=provider.model,
                endpoint=provider.endpoint,
            )
            grant = ApprovalManager(cfg.root).require(
                'image_cloud_vision',
                'cloud_vision_upload',
                payload,
                one_step_approval=True,
            )
            before_db = _sha256(cfg.paths.sqlite_path)
            before_image = _sha256(image)

            with self.assertRaises(ApprovalValidationError) as raised:
                run_image_observation_job(
                    repo,
                    asset_id='synthetic-image',
                    image_path=image,
                    provider=provider,
                    citation=citation,
                    approval_grant=grant,
                    approval_payload=payload,
                )

            self.assertEqual(raised.exception.code, 'approval_grant_unclaimed')
            self.assertEqual(provider.calls, 0)
            self.assertEqual(_sha256(image), before_image)
            self.assertEqual(_sha256(cfg.paths.sqlite_path), before_db)
            self.assertEqual(_table_count(store, 'provider_jobs'), 0)
            self.assertEqual(_table_count(store, 'image_observations'), 0)

    def test_direct_cloud_vector_sink_rejects_unclaimed_grant_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeCloudEmbeddingProvider()
            payload = vector_cloud_approval_payload(
                cfg,
                provider,
                backend='sqlite',
                batch_size=16,
                max_messages=2,
                purge=False,
            )
            grant = ApprovalManager(cfg.root).require(
                'cloud_vector_index',
                'cloud_embedding_upload',
                payload,
                one_step_approval=True,
            )
            before_db = _sha256(cfg.paths.sqlite_path)

            with self.assertRaises(ApprovalValidationError) as raised:
                index_vectors(
                    cfg,
                    provider,
                    backend='sqlite',
                    batch_size=16,
                    max_messages=2,
                    approval_grant=grant,
                    approval_payload=payload,
                )

            self.assertEqual(raised.exception.code, 'approval_grant_unclaimed')
            self.assertEqual(provider.calls, 0)
            self.assertEqual(_sha256(cfg.paths.sqlite_path), before_db)
            self.assertEqual(_table_count(SQLiteStore(cfg.paths.sqlite_path), 'vector_entries'), 0)

    def test_cloud_asr_application_claims_once_and_replay_never_constructs_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            citation = 'trove://wechat/acct-synthetic/conv-private/message_0/1'
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-synthetic', 'Synthetic', 'Synthetic')],
                [Conversation('conv-private', 'acct-synthetic', 'Synthetic private', 'private')],
                [Message(
                    'acct-synthetic',
                    'Synthetic',
                    'conv-private',
                    'Synthetic private',
                    'private',
                    'u1',
                    'Synthetic user',
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    '[voice]',
                    'message_0',
                    1,
                    content_kind='voice',
                )],
            )
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b'RIFFsynthetic')
            MultimodalRepository(store).upsert_media_asset(MediaAssetRecord(
                asset_id='asset-cloud-asr',
                account_id='acct-synthetic',
                source_type='message',
                source_id='synthetic-message',
                modality='voice',
                media_type='voice',
                citation=citation,
                path_ref='sources/voice.wav',
                cache_state='cached',
            ))
            payload = cloud_voice_transcript_payload(citation, env={})
            grant = ApprovalManager(cfg.root).require(
                'voice_cloud_asr',
                'cloud_asr_upload',
                payload,
                one_step_approval=True,
            )
            provider = FakeCloudASRProvider()

            with patch(
                'trove_core.application.cloud_commands._cloud_asr_provider_from_runtime',
                return_value=(provider, None),
            ) as resolver:
                result = execute_cloud_voice_transcript(
                    cfg.root,
                    citation=citation,
                    approval_grant=grant,
                    env={},
                )
                with self.assertRaises(ApprovalValidationError) as replay:
                    execute_cloud_voice_transcript(
                        cfg.root,
                        citation=citation,
                        approval_grant=grant,
                        env={},
                    )

            self.assertTrue(result['ok'])
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(provider.calls, 1)
            self.assertEqual(resolver.call_count, 1)
            self.assertEqual(replay.exception.code, 'approval_grant_replayed')
            self.assertEqual(_table_count(store, 'transcripts'), 1)

    def test_cloud_vector_application_claims_once_and_replay_never_calls_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeCloudEmbeddingProvider()
            payload = vector_cloud_approval_payload(
                cfg,
                provider,
                backend='sqlite',
                batch_size=16,
                max_messages=2,
                purge=False,
            )
            grant = ApprovalManager(cfg.root).require(
                'cloud_vector_index',
                'cloud_embedding_upload',
                payload,
                one_step_approval=True,
            )

            result = execute_cloud_vector_index(
                cfg.root,
                provider=provider,
                approval_grant=grant,
                backend='sqlite',
                batch_size=16,
                max_messages=2,
            )
            calls_after_success = provider.calls
            with self.assertRaises(ApprovalValidationError) as replay:
                execute_cloud_vector_index(
                    cfg.root,
                    provider=provider,
                    approval_grant=grant,
                    backend='sqlite',
                    batch_size=16,
                    max_messages=2,
                )

            self.assertTrue(result['ok'])
            self.assertEqual(result['indexed'], 2)
            self.assertEqual(calls_after_success, 2)
            self.assertEqual(provider.calls, calls_after_success)
            self.assertEqual(replay.exception.code, 'approval_grant_replayed')

    def test_probe_without_approval_never_constructs_cloud_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            repo_root = Path(__file__).resolve().parents[3]
            script = repo_root / 'scripts' / 'probe_cloud_embedding_text.py'
            spec = importlib.util.spec_from_file_location('_trove_cloud_probe_test', script)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader if spec is not None else None)
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            output = Path(directory) / 'probe.json'
            constructor_calls: list[tuple[tuple, dict]] = []

            def forbidden_provider(*args, **kwargs):
                constructor_calls.append((args, kwargs))
                raise AssertionError('provider construction happened before approval')

            stdout = io.StringIO()
            with patch.object(module, 'OpenAICompatibleEmbeddingProvider', forbidden_provider), redirect_stdout(stdout):
                code = module.main([
                    '--vault', str(vault),
                    '--cases', str(repo_root / 'tests' / 'golden' / 'retrieval_core.jsonl'),
                    '--out', str(output),
                    '--max-cases', '1',
                ])

            self.assertEqual(code, 3)
            self.assertEqual(constructor_calls, [])
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(stdout.getvalue())['error']['code'], 'approval_required')

    def test_egress_payloads_reject_coercible_control_types(self):
        with self.assertRaises(TypeError):
            cloud_embedding_payload(
                operation='cloud_vector_index',
                provider='synthetic',
                model='v1',
                dimensions=True,  # type: ignore[arg-type]
                endpoint='https://example.invalid/embedding',
                input_digest='0' * 64,
                item_count=1,
            )


if __name__ == '__main__':
    unittest.main()
