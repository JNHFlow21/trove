from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import base64
import tempfile
import threading
import unittest
import wave
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.approvals import ApprovalManager, ApprovalRequired
from trove_core.asr.base import ASRProvider, ASRRequest, ASRResult, ASRUsage
from trove_core.asr.fake import FakeASRProvider
from trove_core.knowledge.customer_profile import build_customer_profile
from trove_core.knowledge.logical_evidence import deduplicate_logical_rows, logical_moment_media_key
from trove_core.knowledge.profile_enrichment import ProfileEnrichmentError, ProfileEnrichmentService
from trove_core.providers.pricing import estimate_asr_flash_rmb
from trove_core.store.repositories import (
    EntityRecord,
    ImageObservationRecord,
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MediaUnderstandingRecord,
    MultimodalRepository,
    ProviderJobRecord,
    TranscriptRecord,
    WeChatRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.wechat.models import Account, Conversation, Message


class _ProfileCloudASRProvider(ASRProvider):
    name = 'volcengine-asr-flash'
    egress_kind = 'cloud_asr_upload'

    def __init__(self, *, model_name: str, resource_id: str, endpoint: str, duration_seconds: float = 1.0):
        self.model_name = model_name
        self.resource_id = resource_id
        self.endpoint = endpoint
        self.duration_seconds = duration_seconds
        self.calls = 0

    def transcribe(self, request: ASRRequest) -> ASRResult:
        self.calls += 1
        return ASRResult(
            '审批后的云端语音转写', 'zh', 1.0,
            ASRUsage(
                duration_seconds=self.duration_seconds,
                estimated_cost_rmb=estimate_asr_flash_rmb(self.duration_seconds),
            ),
            citations=[request.citation] if request.citation else [],
        )


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
)


def _write_silent_wav(path: Path, *, duration_seconds: float) -> None:
    rate = 8000
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(1)
        audio.setframerate(rate)
        audio.writeframes(b'\x80' * int(rate * duration_seconds))


class ProfileEnrichmentTests(unittest.TestCase):
    def test_moment_media_dedup_uses_projected_evidence_timestamp(self):
        rows = [
            {
                'citation': 'trove://wechat/acct-a/moment/one#image-0',
                'source_id': 'moment:one#image-0',
                'author_id': 'wxid-contact',
                'text': '',
                'evidence_timestamp': '2026-01-01T00:00:00Z',
                'modality': 'image',
            },
            {
                'citation': 'trove://wechat/acct-a/moment/two#image-0',
                'source_id': 'moment:two#image-0',
                'author_id': 'wxid-contact',
                'text': '',
                'evidence_timestamp': '2026-01-02T00:00:00Z',
                'modality': 'image',
            },
            {
                'citation': 'trove://wechat/acct-b/moment/one-copy#image-0',
                'source_id': 'moment:one-copy#image-0',
                'author_id': 'wxid-contact',
                'text': '',
                'evidence_timestamp': '2026-01-01T00:00:00Z',
                'modality': 'image',
            },
        ]

        deduplicated = deduplicate_logical_rows(rows, key=logical_moment_media_key)

        self.assertEqual([row['citation'] for row in deduplicated], [
            'trove://wechat/acct-a/moment/one#image-0',
            'trove://wechat/acct-a/moment/two#image-0',
        ])

    def test_agent_plan_discovers_complete_scope_without_global_writer(self):
        with tempfile.TemporaryDirectory() as root:
            vault, _store, _citation = self._group_voice_fixture(root)
            original = ProfileEnrichmentService.discover

            def discover_without_writer(service, *args, **kwargs):
                with VaultOperationCoordinator(vault).write(owner='probe-profile-discovery'):
                    pass
                return original(service, *args, **kwargs)

            with patch.object(ProfileEnrichmentService, 'discover', discover_without_writer):
                manifest = agent_tools.profile_enrichment_plan(
                    vault, '群聊语音人物', actor='operator', session='lock-probe', item_budget=5,
                    purpose='person_relationship_profile_enrichment',
                )
            self.assertTrue(manifest['ok'])

    def _fixture(self, root: str) -> tuple[SQLiteStore, ProfileEnrichmentService, dict[str, str]]:
        store = SQLiteStore(Path(root) / 'vault.sqlite')
        repo = MultimodalRepository(store)
        repo.upsert_entity(EntityRecord(
            entity_id='customer-jia', entity_type='Customer', display_name='示例客户甲',
            identifiers={'wechat_id': 'wxid-jia', 'remark': '示例客户甲'},
        ))
        messages = [
            Message('acct', 'A', 'wxid-jia', '示例客户甲', 'private', 'wxid-jia', '示例客户甲', datetime(2026, 1, 4, tzinfo=timezone.utc), '[语音]', 's', 1, content_kind='voice'),
            Message('acct', 'A', 'wxid-jia', '示例客户甲', 'private', 'wxid-jia', '示例客户甲', datetime(2026, 1, 3, tzinfo=timezone.utc), '[图片]', 's', 2, content_kind='image'),
            Message('acct', 'A', 'group', '群', 'group', 'wxid-jia', '示例客户甲', datetime(2026, 1, 2, tzinfo=timezone.utc), '[图片]', 's', 3, content_kind='image'),
            Message('acct', 'A', 'wxid-jia', '示例客户甲', 'private', 'wxid-jia', '示例客户甲', datetime(2026, 1, 1, tzinfo=timezone.utc), '[应用消息]', 's', 4, content_kind='appmsg'),
            Message('acct', 'A', 'group', '群', 'group', 'wxid-jia', '示例客户甲', datetime(2025, 12, 31, tzinfo=timezone.utc), '[应用消息]', 's', 5, content_kind='appmsg'),
            Message('acct', 'A', 'group', '群', 'group', 'wxid-jia', '示例客户甲', datetime(2025, 12, 30, tzinfo=timezone.utc), '[语音]', 's', 6, content_kind='voice'),
        ]
        WeChatRepository(store).replace_fixture(
            [Account('acct', 'A', 'A')],
            [Conversation('wxid-jia', 'acct', '示例客户甲', 'private'), Conversation('group', 'acct', '群', 'group', 3)],
            messages,
        )
        refs = {
            'voice': messages[0].citation,
            'image': messages[1].citation,
            'group': messages[2].citation,
            'moment': 'trove://wechat/acct/moment/m1#image-0',
            'appmsg': messages[3].citation,
            'group_appmsg': messages[4].citation,
            'group_voice': messages[5].citation,
        }
        for name, modality in [('voice', 'voice'), ('image', 'image'), ('group', 'image')]:
            asset_id = f'asset-{name}'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id=asset_id, account_id='acct', source_type='message', source_id=name,
                modality=modality, media_type=modality, citation=refs[name],
                content_hash=('a' * 64 if name == 'image' else 'e' * 64 if name == 'voice' else None),
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                link_id=f'link-{name}', asset_id=asset_id, account_id='acct', source_type='message',
                source_citation=refs[name], scope_type='private_chat' if name != 'group' else 'group_chat',
                accepted=True, reason='fixture',
            ))
        repo.insert_moment_item(
            moment_id='m1', account_id='acct', citation='trove://wechat/acct/moment/m1',
            author_id='wxid-jia', timestamp='2026-01-05T00:00:00Z', text='动态',
        )
        repo.upsert_media_asset(MediaAssetRecord(
            asset_id='asset-moment', account_id='acct', source_type='moment', source_id='m1#image-0',
            modality='image', media_type='image', citation=refs['moment'], content_hash='b' * 64,
        ))
        repo.upsert_media_asset_link(MediaAssetLinkRecord(
            link_id='link-moment', asset_id='asset-moment', account_id='acct', source_type='moment',
            source_citation=refs['moment'], scope_type='moment_authored', accepted=True, reason='fixture',
        ))
        with store.connect() as conn:
            conn.execute(
                """INSERT INTO message_payloads(citation,appmsg_type,normalized_type,parse_status,normalized_json,
                       display_text,source_hash,parser_version,unsupported_reason,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (refs['appmsg'], 999, 'unsupported', 'unsupported', '{}', '[应用消息]', 'c' * 64,
                 'fixture/v1', 'unsupported_type', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'),
            )
            conn.commit()
        return store, ProfileEnrichmentService(store), refs

    def _group_voice_fixture(self, root: str) -> tuple[Path, SQLiteStore, str]:
        vault = Path(root) / 'vault'
        cfg = VaultConfig.resolve(str(vault), env={})
        cfg.ensure()
        audio = vault / 'sources' / 'group.wav'
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b'RIFFfixture')
        store = SQLiteStore(cfg.paths.sqlite_path)
        repo = MultimodalRepository(store)
        repo.upsert_entity(EntityRecord(
            entity_id='person-group-voice', entity_type='Customer', display_name='群聊语音人物',
            identifiers={'wechat_id': 'wxid-group-person', 'remark': '群聊语音人物'},
        ))
        message = Message(
            'acct', 'A', 'profile-room@chatroom', '人物群', 'group',
            'wxid-group-person', '群聊语音人物',
            datetime(2026, 1, 1, tzinfo=timezone.utc), '[语音]', 'message_0', 1,
            content_kind='voice',
        )
        WeChatRepository(store).replace_fixture(
            [Account('acct', 'A', 'A')],
            [Conversation('profile-room@chatroom', 'acct', '人物群', 'group', 3)],
            [message],
        )
        repo.upsert_media_asset(MediaAssetRecord(
            'asset-profile-group-voice', 'acct', 'message', 'group-voice-1',
            'voice', 'voice', message.citation,
            path_ref='sources/group.wav', cache_state='cached',
        ))
        repo.upsert_media_asset_link(MediaAssetLinkRecord(
            'link-profile-group-voice', 'asset-profile-group-voice', 'acct', 'message',
            message.citation, 'group_chat', True, 'fixture',
        ))
        return vault, store, message.citation

    def test_complete_manifest_includes_private_and_authored_media_but_excludes_group(self):
        with tempfile.TemporaryDirectory() as root:
            _, service, refs = self._fixture(root)
            result = service.plan('示例客户甲', actor='operator', session='session-1', item_budget=20)

            citations = {item['citation'] for item in result['items']}
            self.assertIn(refs['voice'], citations)
            self.assertIn(refs['image'], citations)
            self.assertIn(refs['moment'], citations)
            self.assertIn(refs['appmsg'], citations)
            self.assertNotIn(refs['group'], citations)
            self.assertNotIn(refs['group_appmsg'], citations)
            self.assertNotIn(refs['group_voice'], citations)
            self.assertTrue(all('next_tool' in item and 'relevance_reason' in item for item in result['items']))
            self.assertFalse(result['raw_content_included'])
            self.assertFalse(result['raw_paths_included'])

    def test_discovery_summary_matches_full_manifest_counts_without_materializing_it(self):
        with tempfile.TemporaryDirectory() as root:
            _, service, _ = self._fixture(root)
            _resolved, items, _revision = service.discover(
                '示例客户甲', purpose='person_relationship_profile_enrichment',
            )
            _resolved, deferred, summary_revision = service.summarize_discovery(
                '示例客户甲', purpose='person_relationship_profile_enrichment',
            )
            expected: dict[str, int] = {}
            for item in items:
                if not item.complete:
                    expected[item.modality] = expected.get(item.modality, 0) + 1

            self.assertEqual(deferred, expected)
            self.assertTrue(summary_revision.startswith('src-summary-'))

    def test_discovery_deduplicates_same_events_from_duplicate_archives(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            duplicate_messages = [
                Message(
                    'acct-copy', 'Copy', 'wxid-jia', '示例客户甲', 'private',
                    'wxid-jia', '示例客户甲', datetime(2026, 1, 4, tzinfo=timezone.utc),
                    '[不同版本的语音占位]', 's', 1, content_kind='voice',
                ),
                Message(
                    'acct-copy', 'Copy', 'wxid-jia', '示例客户甲', 'private',
                    'wxid-jia', '示例客户甲', datetime(2026, 1, 1, tzinfo=timezone.utc),
                    '[不同版本的应用消息占位]', 's', 4, content_kind='appmsg',
                ),
            ]
            WeChatRepository(store).replace_fixture(
                [Account('acct-copy', 'Copy', 'Copy')],
                [Conversation('wxid-jia', 'acct-copy', '示例客户甲', 'private')],
                duplicate_messages,
            )

            _resolved, items, _revision = service.discover(
                '示例客户甲', purpose='person_relationship_profile_enrichment',
            )
            citations = {item.citation for item in items}

            self.assertIn(refs['voice'], citations)
            self.assertIn(refs['appmsg'], citations)
            self.assertNotIn(duplicate_messages[0].citation, citations)
            self.assertNotIn(duplicate_messages[1].citation, citations)

    def test_appmsg_completion_keeps_discovery_revision_stable(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            _resolved, before, before_revision = service.discover(
                '示例客户甲', purpose='person_relationship_profile_enrichment',
            )
            before_appmsg = next(item for item in before if item.citation == refs['appmsg'])
            self.assertFalse(before_appmsg.complete)

            with store.connect() as conn:
                conn.execute(
                    "UPDATE message_payloads SET parse_status='parsed' WHERE citation=?",
                    (refs['appmsg'],),
                )
                conn.commit()

            _resolved, after, after_revision = service.discover(
                '示例客户甲', purpose='person_relationship_profile_enrichment',
            )
            after_appmsg = next(item for item in after if item.citation == refs['appmsg'])

            self.assertTrue(after_appmsg.complete)
            self.assertEqual(after_revision, before_revision)

    def test_claim_uses_exact_task_query_when_manifest_page_omits_pending_task(self):
        with tempfile.TemporaryDirectory() as root:
            vault, store, _citation = self._group_voice_fixture(root)
            service = ProfileEnrichmentService(store)
            result = service.plan(
                '群聊语音人物', actor='operator', session='claim-page-session',
                mode='complete', item_budget=20,
                purpose='person_relationship_profile_enrichment',
            )
            expected = next(item for item in result['items'] if item['state'] == 'pending')
            original_manifest = ProfileEnrichmentService.manifest

            def manifest_page_without_pending_task(service, *args, **kwargs):
                manifest = original_manifest(service, *args, **kwargs)
                manifest['items'] = []
                return manifest

            with patch.object(
                ProfileEnrichmentService,
                'manifest',
                manifest_page_without_pending_task,
            ):
                claimed = agent_tools.profile_enrichment_claim(
                    vault, result['run_id'], actor='operator', session='claim-page-session',
                    worker='local-agent', execution_location='local',
                )

            self.assertEqual(claimed['task']['task_id'], expected['task_id'])

    def test_standard_manifest_stays_in_direct_private_scope(self):
        with tempfile.TemporaryDirectory() as root:
            _, service, refs = self._fixture(root)
            result = service.plan(
                '示例客户甲', actor='operator', session='standard-session', mode='standard', item_budget=20,
            )
            citations = {item['citation'] for item in result['items']}
            self.assertIn(refs['voice'], citations)
            self.assertIn(refs['image'], citations)
            self.assertIn(refs['appmsg'], citations)
            self.assertNotIn(refs['moment'], citations)
            self.assertNotIn(refs['group'], citations)

    def test_person_relationship_manifest_includes_group_speech_and_missing_appmsg_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            _, service, refs = self._fixture(root)
            result = service.plan(
                '示例客户甲', actor='operator', session='person-session', mode='complete', item_budget=20,
                purpose='person_relationship_profile_enrichment',
            )

            items = {item['citation']: item for item in result['items']}
            self.assertIn(refs['group'], items)
            self.assertIn(refs['group_voice'], items)
            self.assertIn(refs['group_appmsg'], items)
            self.assertEqual(items[refs['group']]['relevance_reason'], 'contact_group_speech')
            self.assertEqual(items[refs['group_voice']]['relevance_reason'], 'contact_group_speech')
            self.assertIsNone(items[refs['group_voice']]['content_hash'])
            self.assertEqual(items[refs['group_appmsg']]['modality'], 'appmsg')
            self.assertEqual(items[refs['group_appmsg']]['relevance_reason'], 'contact_group_speech_appmsg')
            self.assertEqual(items[refs['group_appmsg']]['next_tool'], 'trove_profile_enrichment_appmsg_execute')

    def test_ordinary_profile_read_creates_no_enrichment_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            store, _, _ = self._fixture(root)

            build_customer_profile(store, '示例客户甲', limit=3)

            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM profile_enrichment_runs').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM profile_enrichment_tasks').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM provider_jobs').fetchone()[0], 0)

    def test_claim_is_owner_bound_reclaimable_and_completion_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            manifest = service.plan('示例客户甲', actor='operator', session='session-1', item_budget=20)
            voice = next(item for item in manifest['items'] if item['citation'] == refs['voice'])
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            claim = service.claim(
                manifest['run_id'], voice['task_id'], actor='operator', session='session-1',
                lease_owner='worker-a', execution_location='local', lease_seconds=30, now=start,
            )
            self.assertTrue(claim['task']['worker_bound'])
            self.assertTrue(claim['task']['lease_active'])
            with self.assertRaises(ProfileEnrichmentError) as cross_owner:
                service.manifest(manifest['run_id'], actor='other', session='session-1')
            self.assertEqual(cross_owner.exception.code, 'enrichment_owner_mismatch')
            self.assertEqual(service.reclaim_expired(
                manifest['run_id'], actor='operator', session='session-1', now=start + timedelta(seconds=31),
            ), 1)
            reclaimed = service.claim(
                manifest['run_id'], voice['task_id'], actor='operator', session='session-1',
                lease_owner='worker-b', execution_location='local', lease_seconds=60, now=start + timedelta(seconds=32),
            )
            repo = MultimodalRepository(store)
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-transcript-voice', asset_id='asset-voice', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation=refs['voice'],
            ))
            repo.insert_transcript(TranscriptRecord(
                transcript_id='transcript-voice', asset_id='asset-voice', citation=refs['voice'] + '#voice',
                text='已转写', job_id='job-transcript-voice',
            ))
            with self.assertRaises(ProfileEnrichmentError) as wrong_worker:
                service.complete(
                    manifest['run_id'], voice['task_id'], actor='operator', session='session-1',
                    lease_owner='worker-a', claim_token=reclaimed['agent_action']['claim_token'],
                    completion_key='wrong-worker', now=start + timedelta(seconds=33),
                )
            self.assertEqual(wrong_worker.exception.code, 'task_lease_owner_mismatch')
            completed = service.complete(
                manifest['run_id'], voice['task_id'], actor='operator', session='session-1',
                lease_owner='worker-b', claim_token=reclaimed['agent_action']['claim_token'], completion_key='complete-voice-1',
                now=start + timedelta(seconds=33),
            )
            self.assertEqual(completed['task']['state'], 'completed')
            repeated = service.complete(
                manifest['run_id'], voice['task_id'], actor='operator', session='session-1',
                lease_owner='worker-b', claim_token='already-consumed', completion_key='complete-voice-1', now=start + timedelta(seconds=34),
            )
            self.assertTrue(repeated['idempotent'])
            with self.assertRaises(ProfileEnrichmentError) as replay:
                service.complete(
                    manifest['run_id'], voice['task_id'], actor='operator', session='session-1',
                    lease_owner='worker-a', claim_token=claim['agent_action']['claim_token'], completion_key='different-key',
                    now=start + timedelta(seconds=34),
                )
            self.assertEqual(replay.exception.code, 'completion_replay_rejected')

    def test_local_image_capability_is_one_time_and_annotation_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            manifest = service.plan('示例客户甲', actor='operator', session='session-1', item_budget=20)
            image = next(item for item in manifest['items'] if item['citation'] == refs['image'])
            claim = service.claim(
                manifest['run_id'], image['task_id'], actor='operator', session='session-1',
                lease_owner='local-mcp', execution_location='local', lease_seconds=300,
            )
            action = claim['agent_action']
            redeemed = service.redeem_local_media(
                manifest['run_id'], image['task_id'], actor='operator', session='session-1',
                lease_owner='local-mcp', claim_token=action['claim_token'], media_capability=action['media_capability'],
            )
            self.assertEqual(redeemed['next_tool'], 'trove_media_fetch')
            with self.assertRaises(ProfileEnrichmentError) as replay:
                service.redeem_local_media(
                    manifest['run_id'], image['task_id'], actor='operator', session='session-1',
                    lease_owner='local-mcp', claim_token=action['claim_token'], media_capability=action['media_capability'],
                )
            self.assertEqual(replay.exception.code, 'media_capability_replayed')
            with self.assertRaises(ProfileEnrichmentError) as missing:
                service.complete(
                    manifest['run_id'], image['task_id'], actor='operator', session='session-1',
                    lease_owner='local-mcp', claim_token=action['claim_token'], completion_key='image-complete',
                )
            self.assertEqual(missing.exception.code, 'matching_media_annotation_required')

            repo = MultimodalRepository(store)
            with store.connect() as conn:
                conn.execute("UPDATE media_assets SET content_hash=? WHERE asset_id='asset-image'", ('d' * 64,))
                conn.commit()
            repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256='d' * 64, modality='image', model_id='local-agent', prompt_version='v1',
                caption='结构化视觉理解', source_citations=[refs['image']],
            ))
            repo.insert_image_observation(ImageObservationRecord(
                observation_id='obs-image', asset_id='asset-image', citation=refs['image'] + '#image',
                caption='结构化视觉理解', status='active',
            ))
            done = service.complete(
                manifest['run_id'], image['task_id'], actor='operator', session='session-1',
                lease_owner='local-mcp', claim_token=action['claim_token'], completion_key='image-complete',
            )
            self.assertEqual(done['task']['state'], 'completed')
            self.assertEqual(done['task']['content_hash'], 'd' * 64)

    def test_budget_pause_and_resume_does_not_repeat_completed_work(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            first = service.plan('示例客户甲', actor='operator', session='session-1', item_budget=1)
            self.assertEqual(first['counts'].get('paused_budget'), 3)
            active = next(item for item in first['items'] if item['state'] == 'pending')
            if active['modality'] == 'voice':
                repo = MultimodalRepository(store)
                repo.record_provider_job(ProviderJobRecord(
                    job_id='job-budget', asset_id='asset-voice', provider='volcengine-asr-flash',
                    model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                    request_hash='e' * 64, citation=refs['voice'],
                ))
                repo.insert_transcript(TranscriptRecord(
                    transcript_id='tr-budget', asset_id='asset-voice', citation=refs['voice'] + '#voice',
                    text='转写', job_id='job-budget',
                ))
            else:
                self.skipTest('fixture order should select direct voice first')
            claim = service.claim(
                first['run_id'], active['task_id'], actor='operator', session='session-1',
                lease_owner='worker', execution_location='local',
            )
            service.complete(
                first['run_id'], active['task_id'], actor='operator', session='session-1',
                lease_owner='worker', claim_token=claim['agent_action']['claim_token'], completion_key='budget-first',
            )
            resumed = service.resume_budget(
                first['run_id'], actor='operator', session='session-1', additional_items=2,
            )
            completed = [item for item in resumed['items'] if item['task_id'] == active['task_id']]
            self.assertEqual(completed[0]['state'], 'completed')
            self.assertEqual(resumed['counts'].get('pending'), 2)
            self.assertEqual(resumed['counts'].get('paused_budget'), 1)

    def test_remote_run_cannot_be_claimed_as_local_and_waits_for_approval(self):
        with tempfile.TemporaryDirectory() as root:
            _, service, _ = self._fixture(root)
            manifest = service.plan(
                '示例客户甲', actor='operator', session='remote-session', execution_location='remote',
                processor_identity='remote-agent/model', item_budget=20, cost_budget_rmb=10,
            )
            task = next(item for item in manifest['items'] if item['modality'] == 'image')
            self.assertEqual(task['state'], 'awaiting_approval')
            self.assertTrue(task['approval_required'])
            with self.assertRaises(ProfileEnrichmentError) as mismatch:
                service.claim(
                    manifest['run_id'], task['task_id'], actor='operator', session='remote-session',
                    lease_owner='worker', execution_location='local',
                )
            self.assertIn(mismatch.exception.code, {'enrichment_task_not_claimable', 'execution_attestation_mismatch'})

    def test_zero_cost_budget_means_unlimited_cloud_asr(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            with store.connect() as conn:
                conn.execute("UPDATE media_assets SET content_hash=? WHERE asset_id='asset-voice'", ('e' * 64,))
                conn.commit()
            manifest = service.plan('示例客户甲', actor='operator', session='budgetless', item_budget=20, cost_budget_rmb=0)
            task = next(item for item in manifest['items'] if item['citation'] == refs['voice'])
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='budgetless',
                lease_owner='worker', execution_location='local',
            )
            scope = service.voice_cloud_scope(
                manifest['run_id'], task['task_id'], actor='operator', session='budgetless',
                lease_owner='worker', claim_token=claim['agent_action']['claim_token'],
            )
            self.assertIsNone(scope['cost_ceiling_rmb'])

    def test_60_second_cloud_voice_is_rejected_before_approval_when_budget_is_005(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            with store.connect() as conn:
                conn.execute("UPDATE media_assets SET content_hash=? WHERE asset_id='asset-voice'", ('e' * 64,))
                conn.commit()
            manifest = service.plan(
                '示例客户甲', actor='operator', session='short-cloud-budget',
                item_budget=20, cost_budget_rmb=0.05,
            )
            task = next(item for item in manifest['items'] if item['citation'] == refs['voice'])
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='short-cloud-budget',
                lease_owner='worker', execution_location='local',
            )

            with self.assertRaises(ProfileEnrichmentError) as budget:
                service.voice_cloud_scope(
                    manifest['run_id'], task['task_id'], actor='operator', session='short-cloud-budget',
                    lease_owner='worker', claim_token=claim['agent_action']['claim_token'],
                    estimated_cost_rmb=estimate_asr_flash_rmb(60),
                )

            self.assertEqual(estimate_asr_flash_rmb(60), 0.075)
            self.assertEqual(budget.exception.code, 'enrichment_cost_budget_exhausted')
            with store.connect() as conn:
                reserved = conn.execute(
                    'SELECT estimated_cost_rmb FROM profile_enrichment_tasks WHERE task_id=?',
                    (task['task_id'],),
                ).fetchone()[0]
            self.assertEqual(reserved, 0.0)

    def test_cloud_voice_reservations_are_atomic_idempotent_and_released_on_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store, service, refs = self._fixture(root)
            repo = MultimodalRepository(store)
            with store.connect() as conn:
                conn.execute("UPDATE media_assets SET content_hash=? WHERE asset_id='asset-voice'", ('e' * 64,))
                conn.commit()
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-group-voice', 'acct', 'message', 'group-voice', 'voice', 'voice',
                refs['group_voice'], content_hash='f' * 64,
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-group-voice', 'asset-group-voice', 'acct', 'message', refs['group_voice'],
                'group_chat', True, 'fixture',
            ))
            manifest = service.plan(
                '示例客户甲', actor='operator', session='concurrent-budget', item_budget=20,
                cost_budget_rmb=0.05, purpose='person_relationship_profile_enrichment',
            )
            voices = [item for item in manifest['items'] if item['modality'] == 'voice']
            claims = [
                service.claim(
                    manifest['run_id'], task['task_id'], actor='operator', session='concurrent-budget',
                    lease_owner=f'worker-{index}', execution_location='local',
                )
                for index, task in enumerate(voices)
            ]
            barrier = threading.Barrier(2)

            def reserve(index: int) -> tuple[int, str]:
                barrier.wait()
                try:
                    service.voice_cloud_scope(
                        manifest['run_id'], voices[index]['task_id'], actor='operator', session='concurrent-budget',
                        lease_owner=f'worker-{index}', claim_token=claims[index]['agent_action']['claim_token'],
                    )
                    return index, 'reserved'
                except ProfileEnrichmentError as exc:
                    return index, exc.code

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(reserve, range(2)))
            self.assertEqual(sorted(value for _, value in results), ['enrichment_cost_budget_exhausted', 'reserved'])
            winner = next(index for index, value in results if value == 'reserved')
            loser = 1 - winner

            # Repeating the same scope reuses, rather than doubles, its reservation.
            repeated = service.voice_cloud_scope(
                manifest['run_id'], voices[winner]['task_id'], actor='operator', session='concurrent-budget',
                lease_owner=f'worker-{winner}', claim_token=claims[winner]['agent_action']['claim_token'],
            )
            self.assertEqual(repeated['cost_ceiling_rmb'], 0.05)
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    'SELECT SUM(estimated_cost_rmb) FROM profile_enrichment_tasks WHERE run_id=?',
                    (manifest['run_id'],),
                ).fetchone()[0], 0.05)

            service.fail(
                manifest['run_id'], voices[winner]['task_id'], actor='operator', session='concurrent-budget',
                lease_owner=f'worker-{winner}', claim_token=claims[winner]['agent_action']['claim_token'],
                reason='provider_failed', terminal=True,
            )
            released = service.voice_cloud_scope(
                manifest['run_id'], voices[loser]['task_id'], actor='operator', session='concurrent-budget',
                lease_owner=f'worker-{loser}', claim_token=claims[loser]['agent_action']['claim_token'],
            )
            self.assertEqual(released['cost_ceiling_rmb'], 0.05)

    def test_cached_cloud_transcript_retry_settles_provider_job_cost_and_uses_long_lease(self):
        with tempfile.TemporaryDirectory() as root:
            vault = Path(root) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            _write_silent_wav(audio, duration_seconds=1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-cached-cloud', entity_type='Customer', display_name='缓存云语音客户',
                identifiers={'wechat_id': 'wxid-cached-cloud', 'remark': '缓存云语音客户'},
            ))
            message = Message(
                'acct', 'A', 'wxid-cached-cloud', '缓存云语音客户', 'private', 'wxid-cached-cloud', '缓存云语音客户',
                datetime(2026, 1, 1, tzinfo=timezone.utc), '[语音]', 's', 1, content_kind='voice',
            )
            WeChatRepository(store).replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-cached-cloud', 'acct', '缓存云语音客户', 'private')],
                [message],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-cached-cloud', 'acct', 'message', 'voice-cached-cloud', 'voice', 'voice', message.citation,
                content_hash='c' * 64, path_ref='sources/voice.wav', cache_state='cached',
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-cached-cloud', 'asset-cached-cloud', 'acct', 'message', message.citation,
                'private_chat', True, 'fixture',
            ))
            manifest = agent_tools.profile_enrichment_plan(
                vault, '缓存云语音客户', actor='operator', session='cached-cloud-session',
                item_budget=5, cost_budget_rmb=0.05,
            )
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='cached-cloud-session',
                worker='local-mcp', execution_location='local',
            )
            service = ProfileEnrichmentService(store)
            service.voice_cloud_scope(
                manifest['run_id'], claim['task']['task_id'], actor='operator', session='cached-cloud-session',
                lease_owner='local-mcp', claim_token=claim['agent_action']['claim_token'],
            )
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-cached-cloud', asset_id='asset-cached-cloud', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='c' * 64, cost_rmb=0.03, citation=message.citation,
            ))
            repo.insert_transcript(TranscriptRecord(
                transcript_id='transcript-cached-cloud', asset_id='asset-cached-cloud',
                job_id='job-cached-cloud', citation=message.citation + '#voice', text='缓存的云端转写',
            ))

            with patch.object(ProfileEnrichmentService, 'heartbeat', autospec=True) as heartbeat:
                result = agent_tools.profile_enrichment_voice_execute(
                    vault, claim['task']['task_id'], actor='operator', session='cached-cloud-session',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                )

            self.assertEqual(heartbeat.call_args.kwargs['lease_seconds'], 1800)
            self.assertEqual(result['status'], 'cached')
            self.assertEqual(result['profile_enrichment']['task']['actual_cost_rmb'], 0.03)
            self.assertEqual(result['profile_enrichment']['task']['estimated_cost_rmb'], 0.0)

    def test_revocation_invalidates_manifest_and_outstanding_capabilities(self):
        with tempfile.TemporaryDirectory() as root:
            _, service, _ = self._fixture(root)
            manifest = service.plan('示例客户甲', actor='operator', session='session-1', item_budget=20)
            task = next(item for item in manifest['items'] if item['state'] == 'pending')
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='session-1',
                lease_owner='worker', execution_location='local',
            )
            self.assertTrue(claim['agent_action']['claim_token'])
            service.revoke(manifest['run_id'], actor='operator', session='session-1')
            with self.assertRaises(ProfileEnrichmentError) as revoked:
                service.manifest(manifest['run_id'], actor='operator', session='session-1')
            self.assertEqual(revoked.exception.code, 'enrichment_run_revoked')

    def test_cached_local_asr_is_replaced_by_approved_cloud_transcript(self):
        from trove_core.providers.config import ProviderConfig

        with tempfile.TemporaryDirectory() as root:
            vault = Path(root) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            _write_silent_wav(audio, duration_seconds=1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-local-voice', entity_type='Customer', display_name='本地语音客户',
                identifiers={'wechat_id': 'wxid-local-voice', 'remark': '本地语音客户'},
            ))
            message = Message(
                'acct', 'A', 'wxid-local-voice', '本地语音客户', 'private', 'wxid-local-voice', '本地语音客户',
                datetime(2026, 1, 1, tzinfo=timezone.utc), '[语音]', 's', 1, content_kind='voice',
            )
            WeChatRepository(store).replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-local-voice', 'acct', '本地语音客户', 'private')],
                [message],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-local-profile', 'acct', 'message', 'voice-1', 'voice', 'voice', message.citation,
                path_ref='sources/voice.wav', cache_state='cached',
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-local-profile', 'asset-local-profile', 'acct', 'message', message.citation,
                'private_chat', True, 'fixture',
            ))
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-local-profile', asset_id='asset-local-profile', provider='local-faster-whisper',
                model='base:local', job_type='asr', status='completed', citation=message.citation,
            ))
            repo.insert_transcript(TranscriptRecord(
                'transcript-local-profile', 'asset-local-profile', message.citation + '#voice',
                '低质量本地转写', job_id='job-local-profile',
            ))

            manifest = agent_tools.profile_enrichment_plan(
                vault, '本地语音客户', actor='operator', session='voice-cloud-session', item_budget=20,
            )
            task = next(item for item in manifest['items'] if item['modality'] == 'voice')
            self.assertEqual(task['state'], 'pending')
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='voice-cloud-session', worker='local-mcp',
                execution_location='local',
            )
            with self.assertRaises(ApprovalRequired) as approval:
                agent_tools.profile_enrichment_voice_execute(
                    vault, task['task_id'], actor='operator', session='voice-cloud-session',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                )
            ApprovalManager(vault).decide(approval.exception.record.approval_id, 'approved')
            provider_cfg = ProviderConfig.resolve()
            provider = _ProfileCloudASRProvider(
                model_name=provider_cfg.asr_model_name,
                resource_id=provider_cfg.asr_resource_id,
                endpoint=provider_cfg.asr_endpoint,
            )
            with patch(
                'trove_core.application.cloud_commands._cloud_asr_provider_from_runtime',
                return_value=(provider, None),
            ):
                result = agent_tools.profile_enrichment_voice_execute(
                    vault, task['task_id'], actor='operator', session='voice-cloud-session',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                    approval_id=approval.exception.record.approval_id,
                )

            self.assertEqual(result['execution_path'], 'approved_cloud_asr')
            self.assertEqual(result['profile_enrichment']['task']['state'], 'completed')
            with store.connect() as conn:
                rows = list(conn.execute(
                    "SELECT t.status,pj.provider FROM transcripts t LEFT JOIN provider_jobs pj ON pj.job_id=t.job_id WHERE t.asset_id=? ORDER BY pj.provider",
                    ('asset-local-profile',),
                ))
            self.assertEqual(
                {(row['provider'], row['status']) for row in rows},
                {('local-faster-whisper', 'superseded'), ('volcengine-asr-flash', 'active')},
            )

    def test_person_profile_group_voice_uses_approved_cloud_without_global_queue(self):
        from trove_core.providers.config import ProviderConfig

        with tempfile.TemporaryDirectory() as root:
            vault, store, citation = self._group_voice_fixture(root)
            manifest = agent_tools.profile_enrichment_plan(
                vault, '群聊语音人物', actor='operator', session='group-voice-cloud',
                item_budget=5, purpose='person_relationship_profile_enrichment',
            )
            task = next(item for item in manifest['items'] if item['citation'] == citation)
            self.assertEqual(task['relevance_reason'], 'contact_group_speech')
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], task_id=task['task_id'], actor='operator',
                session='group-voice-cloud', worker='local-mcp', execution_location='local',
            )
            with self.assertRaises(ApprovalRequired) as approval:
                agent_tools.profile_enrichment_voice_execute(
                    vault, task['task_id'], actor='operator', session='group-voice-cloud',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                )
            ApprovalManager(vault).decide(approval.exception.record.approval_id, 'approved')
            provider_cfg = ProviderConfig.resolve()
            provider = _ProfileCloudASRProvider(
                model_name=provider_cfg.asr_model_name,
                resource_id=provider_cfg.asr_resource_id,
                endpoint=provider_cfg.asr_endpoint,
            )
            with patch(
                'trove_core.application.cloud_commands._cloud_asr_provider_from_runtime',
                return_value=(provider, None),
            ):
                result = agent_tools.profile_enrichment_voice_execute(
                    vault, task['task_id'], actor='operator', session='group-voice-cloud',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                    approval_id=approval.exception.record.approval_id,
                )

            self.assertEqual(result['execution_path'], 'approved_cloud_asr')
            self.assertTrue(result['cloud_calls_made'])
            self.assertEqual(result['profile_enrichment']['task']['state'], 'completed')
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-profile-group-voice' AND status='active'",
                ).fetchone()[0], 1)
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM media_jobs WHERE asset_id='asset-profile-group-voice'",
                ).fetchone()[0], 0)

    def test_agent_voice_cloud_fallback_is_run_scoped_budgeted_and_approval_bound(self):
        from trove_core.providers.config import ProviderConfig

        with tempfile.TemporaryDirectory() as root:
            vault = Path(root) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            audio = vault / 'sources' / 'voice.wav'
            audio.parent.mkdir(parents=True, exist_ok=True)
            _write_silent_wav(audio, duration_seconds=60)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-cloud-voice', entity_type='Customer', display_name='云端语音客户',
                identifiers={'wechat_id': 'wxid-cloud-voice', 'remark': '云端语音客户'},
            ))
            message = Message(
                'acct', 'A', 'wxid-cloud-voice', '云端语音客户', 'private', 'wxid-cloud-voice', '云端语音客户',
                datetime(2026, 1, 1, tzinfo=timezone.utc), '[语音]', 's', 1, content_kind='voice',
            )
            WeChatRepository(store).replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-cloud-voice', 'acct', '云端语音客户', 'private')],
                [message],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-cloud-profile', 'acct', 'message', 'voice-cloud', 'voice', 'voice', message.citation,
                path_ref='sources/voice.wav', cache_state='cached',
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-cloud-profile', 'asset-cloud-profile', 'acct', 'message', message.citation,
                'private_chat', True, 'fixture',
            ))
            low_budget_manifest = agent_tools.profile_enrichment_plan(
                vault, '云端语音客户', actor='operator', session='cloud-budget-too-low', item_budget=20,
                cost_budget_rmb=0.05,
            )
            low_budget_claim = agent_tools.profile_enrichment_claim(
                vault, low_budget_manifest['run_id'], actor='operator', session='cloud-budget-too-low',
                worker='local-mcp-low-budget', execution_location='local',
            )
            paused = agent_tools.profile_enrichment_voice_execute(
                vault, low_budget_claim['task']['task_id'], actor='operator',
                session='cloud-budget-too-low', worker='local-mcp-low-budget',
                claim_token=low_budget_claim['agent_action']['claim_token'],
            )
            self.assertEqual(paused['status'], 'paused_budget')
            self.assertEqual(paused['estimated_cost_rmb'], 0.075)
            self.assertEqual(paused['profile_enrichment']['state'], 'paused_budget')

            manifest = agent_tools.profile_enrichment_plan(
                vault, '云端语音客户', actor='operator', session='cloud-session', item_budget=20,
            )
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='cloud-session', worker='local-mcp',
                execution_location='local',
            )
            with self.assertRaises(ApprovalRequired) as approval:
                agent_tools.profile_enrichment_voice_execute(
                    vault, claim['task']['task_id'], actor='operator', session='cloud-session',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                )
            record = approval.exception.record
            self.assertEqual(record.danger_class, 'cloud_asr_upload')
            self.assertNotIn(message.citation, str(record.payload))
            task_status = agent_tools.profile_enrichment_status(
                vault, manifest['run_id'], actor='operator', session='cloud-session',
            )
            self.assertEqual(task_status['items'][0]['state'], 'awaiting_approval')
            self.assertEqual(task_status['budget']['cost_rmb'], 0.0)
            with store.connect() as conn:
                task_row = conn.execute(
                    'SELECT approval_scope_hash,content_hash FROM profile_enrichment_tasks WHERE task_id=?',
                    (claim['task']['task_id'],),
                ).fetchone()
                self.assertEqual(task_row['approval_scope_hash'], record.request_hash)
                self.assertEqual(len(task_row['content_hash']), 64)
            ApprovalManager(vault).decide(record.approval_id, 'approved')
            provider_cfg = ProviderConfig.resolve()
            provider = _ProfileCloudASRProvider(
                model_name=provider_cfg.asr_model_name,
                resource_id=provider_cfg.asr_resource_id,
                endpoint=provider_cfg.asr_endpoint,
                duration_seconds=60,
            )
            with patch(
                'trove_core.application.cloud_commands._cloud_asr_provider_from_runtime', return_value=(provider, None),
            ):
                result = agent_tools.profile_enrichment_voice_execute(
                    vault, claim['task']['task_id'], actor='operator', session='cloud-session',
                    worker='local-mcp', claim_token=claim['agent_action']['claim_token'],
                    approval_id=record.approval_id,
                )

            self.assertEqual(result['execution_path'], 'approved_cloud_asr')
            self.assertEqual(result['profile_enrichment']['task']['state'], 'completed')
            self.assertEqual(provider.calls, 1)
            self.assertEqual(result['profile_enrichment']['task']['actual_cost_rmb'], 0.075)
            with store.connect() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM transcripts WHERE asset_id='asset-cloud-profile' AND status='active'").fetchone()[0], 1)

    def test_agent_image_annotation_projects_searchable_versioned_evidence_and_completes_task(self):
        with tempfile.TemporaryDirectory() as root:
            vault = Path(root) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'image.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-image', entity_type='Customer', display_name='视觉客户',
                identifiers={'wechat_id': 'wxid-image', 'remark': '视觉客户'},
            ))
            message = Message(
                'acct', 'A', 'wxid-image', '视觉客户', 'private', 'wxid-image', '视觉客户',
                datetime(2026, 1, 1, tzinfo=timezone.utc), '[图片]', 's', 1, content_kind='image',
            )
            WeChatRepository(store).replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-image', 'acct', '视觉客户', 'private')],
                [message],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-image-profile', 'acct', 'message', 'image-1', 'image', 'image', message.citation,
                path_ref='sources/image.png', cache_state='cached',
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-image-profile', 'asset-image-profile', 'acct', 'message', message.citation,
                'private_chat', True, 'fixture',
            ))
            manifest = agent_tools.profile_enrichment_plan(
                vault, '视觉客户', actor='operator', session='image-session', item_budget=20,
                processor_identity='agent-vision-v1', prompt_version='profile-image/p1',
            )
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='image-session', worker='local-mcp',
                execution_location='local',
            )
            action = claim['agent_action']
            preview = agent_tools.profile_enrichment_redeem_media(
                vault, claim['task']['task_id'], actor='operator', session='image-session', worker='local-mcp',
                claim_token=action['claim_token'], media_capability=action['media_capability'],
            )
            self.assertTrue(preview['ok'])
            self.assertEqual(preview['next_tool'], 'trove_profile_enrichment_image_annotate')
            with self.assertRaises(ProfileEnrichmentError) as processor_mismatch:
                agent_tools.profile_enrichment_image_annotate(
                    vault, claim['task']['task_id'], actor='operator', session='image-session', worker='local-mcp',
                    claim_token=action['claim_token'], model_id='different-agent', prompt_version='profile-image/p1',
                    caption='不应写入',
                )
            self.assertEqual(processor_mismatch.exception.code, 'annotation_processor_mismatch')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_understanding').fetchone()[0], 0)
            result = agent_tools.profile_enrichment_image_annotate(
                vault, claim['task']['task_id'], actor='operator', session='image-session', worker='local-mcp',
                claim_token=action['claim_token'], model_id='agent-vision-v1', prompt_version='profile-image/p1',
                caption='视觉客户预算海报', visible_text='独特视觉搜索needle',
                objects=[{'label': 'poster'}], business_signals=[{'type': 'pricing'}], confidence=0.9,
            )

            self.assertEqual(result['execution_path'], 'local_agent_vision')
            self.assertEqual(result['profile_enrichment']['task']['state'], 'completed')
            hits = store.chunk_search('独特视觉搜索needle', filters={'source_type': 'image_observation'}, limit=3)
            self.assertEqual(len(hits), 1)
            profile = build_customer_profile(store, '视觉客户', limit=5)
            self.assertEqual(len(profile['sections']['image_observations']), 1)
            self.assertFalse(profile['sections']['image_observations'][0]['auto_approved_personal_fact'])
            with store.connect() as conn:
                row = conn.execute("SELECT * FROM image_observations WHERE asset_id='asset-image-profile' AND status='active'").fetchone()
                self.assertEqual(row['model_id'], 'agent-vision-v1')
                self.assertEqual(row['prompt_version'], 'profile-image/p1')
                self.assertEqual(row['content_sha256'], preview['content_sha256'])
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM observations WHERE entity_id='customer-image'").fetchone()[0], 0)

    def test_image_redeem_records_terminal_locator_gap_instead_of_sticking_awaiting_agent(self):
        with tempfile.TemporaryDirectory() as root:
            vault = Path(root) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(
                entity_id='customer-image-gap', entity_type='Customer', display_name='视觉缺口客户',
                identifiers={'wechat_id': 'wxid-image-gap', 'remark': '视觉缺口客户'},
            ))
            message = Message(
                'acct', 'A', 'wxid-image-gap', '视觉缺口客户', 'private',
                'wxid-image-gap', '视觉缺口客户', datetime(2026, 1, 1, tzinfo=timezone.utc),
                '[图片]', 's', 1, content_kind='image',
            )
            WeChatRepository(store).replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-image-gap', 'acct', '视觉缺口客户', 'private')],
                [message],
            )
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-image-gap', 'acct', 'message', 'image-gap', 'image', 'image', message.citation,
                cache_state='source_available',
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-image-gap', 'asset-image-gap', 'acct', 'message', message.citation,
                'private_chat', True, 'fixture',
            ))
            manifest = agent_tools.profile_enrichment_plan(
                vault, '视觉缺口客户', actor='operator', session='image-gap-session', item_budget=5,
                processor_identity='agent-vision-v1', prompt_version='profile-image/p1',
            )
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='image-gap-session',
                worker='local-mcp', execution_location='local',
            )
            action = claim['agent_action']
            original_manifest = ProfileEnrichmentService.manifest

            def manifest_page_without_claimed_task(service, *args, **kwargs):
                result = original_manifest(service, *args, **kwargs)
                result['items'] = []
                return result

            with patch('trove_core.agent_tools.tools.fetch_media', return_value={
                'ok': False,
                'code': 'media_unavailable',
                'status': 'unavailable',
                'reason': 'locator_routes_exhausted',
                'raw_paths_included': False,
                'raw_content_included': False,
            }), patch.object(
                ProfileEnrichmentService,
                'manifest',
                manifest_page_without_claimed_task,
            ):
                result = agent_tools.profile_enrichment_redeem_media(
                    vault, claim['task']['task_id'], actor='operator', session='image-gap-session',
                    worker='local-mcp', claim_token=action['claim_token'],
                    media_capability=action['media_capability'],
                )

            self.assertEqual(result['profile_enrichment']['task']['state'], 'unavailable')
            self.assertEqual(result['profile_enrichment']['task']['terminal_reason'], 'locator_routes_exhausted')
            self.assertEqual(result['profile_enrichment']['state'], 'complete_with_terminal_gaps')
            self.assertTrue(result['profile_enrichment']['profile_snapshot']['created'])
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    'SELECT COUNT(*) FROM profile_snapshots WHERE run_id=?',
                    (manifest['run_id'],),
                ).fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
