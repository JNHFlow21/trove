from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trove_core.knowledge.profile_enrichment import ProfileEnrichmentService
from trove_core.knowledge.profile_snapshots import _evidence_digests, finalize_profile_snapshot, profile_snapshot_status
from trove_core.agent_tools.tools import media_annotate
from trove_core.store.repositories import (
    EntityRecord,
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MultimodalRepository,
    ProviderJobRecord,
    TranscriptRecord,
    WeChatRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
)
PRIVATE_FIXTURE_PATH = '/' + 'Us' + 'ers' + '/' + 'private' + '/' + 'raw.db'


class ProfileSnapshotTests(unittest.TestCase):
    def test_voice_evidence_digests_use_one_batch_query(self):
        with tempfile.TemporaryDirectory() as root:
            store = SQLiteStore(Path(root) / 'vault.sqlite')
            repo = MultimodalRepository(store)
            tasks: list[dict[str, str]] = []
            for index in range(12):
                asset_id = f'asset-batch-{index}'
                citation = f'trove://fixture/voice/{index}'
                repo.upsert_media_asset(MediaAssetRecord(
                    asset_id, 'acct', 'message', f'voice-{index}', 'voice', 'voice', citation,
                    content_hash=f'{index:064x}',
                ))
                repo.record_provider_job(ProviderJobRecord(
                    job_id=f'job-batch-{index}', asset_id=asset_id, provider='volcengine-asr-flash',
                    model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                    request_hash=f'{index:064x}', citation=citation,
                ))
                repo.insert_transcript(TranscriptRecord(
                    f'transcript-batch-{index}', asset_id, citation + '#voice', f'批量转写 {index}',
                    job_id=f'job-batch-{index}',
                ))
                tasks.append({'asset_id': asset_id, 'citation': citation, 'modality': 'voice'})
            statements: list[str] = []
            original_connect = store.connect

            def traced_connect():
                conn = original_connect()
                conn.set_trace_callback(statements.append)
                return conn

            store.connect = traced_connect  # type: ignore[method-assign]
            try:
                digests = _evidence_digests(store, tasks)
            finally:
                store.connect = original_connect  # type: ignore[method-assign]

            reads = [sql for sql in statements if sql.lstrip().upper().startswith(('SELECT', 'WITH'))]
            self.assertEqual(len(digests), 12)
            self.assertEqual(len(reads), 1)

    def test_missing_snapshot_status_skips_full_enrichment_discovery(self):
        with tempfile.TemporaryDirectory() as root:
            store = SQLiteStore(Path(root) / 'vault.sqlite')
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-no-snapshot', entity_type='Customer', display_name='无快照客户',
                identifiers={'wechat_id': 'wxid-no-snapshot'},
            ))

            with patch.object(ProfileEnrichmentService, 'discover', side_effect=AssertionError('unexpected full discovery')):
                status = profile_snapshot_status(
                    store,
                    '无快照客户',
                    resolved_entity={'entity_id': 'customer-no-snapshot'},
                )

            self.assertEqual(status['completeness_state'], 'missing')

    def _voice_run(self, root: str, *, session: str = 'snapshot-session'):
        vault = Path(root) / 'vault'
        cfg = VaultConfig.resolve(str(vault), env={})
        cfg.ensure()
        store = SQLiteStore(cfg.paths.sqlite_path)
        repo = MultimodalRepository(store)
        repo.upsert_entity(EntityRecord(
            entity_id='customer-snapshot', entity_type='Customer', display_name='快照客户',
            identifiers={'wechat_id': 'wxid-snapshot', 'remark': '快照客户'},
        ))
        voice = Message(
            'acct', 'A', 'wxid-snapshot', '快照客户', 'private', 'wxid-snapshot', '快照客户',
            datetime(2026, 1, 2, tzinfo=timezone.utc), '[语音]', 's', 1, content_kind='voice',
        )
        text = Message(
            'acct', 'A', 'wxid-snapshot', '快照客户', 'private', 'wxid-snapshot', '快照客户',
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            f'快照客户参考 https://signed.invalid/item?token=secret {PRIVATE_FIXTURE_PATH}', 's', 2,
        )
        WeChatRepository(store).replace_fixture(
            [Account('acct', 'A', 'A')],
            [Conversation('wxid-snapshot', 'acct', '快照客户', 'private')],
            [voice, text],
        )
        repo.upsert_media_asset(MediaAssetRecord(
            'asset-snapshot-voice', 'acct', 'message', 'voice-1', 'voice', 'voice', voice.citation,
            content_hash='e' * 64,
        ))
        repo.upsert_media_asset_link(MediaAssetLinkRecord(
            'link-snapshot-voice', 'asset-snapshot-voice', 'acct', 'message', voice.citation,
            'private_chat', True, 'fixture',
        ))
        service = ProfileEnrichmentService(store)
        manifest = service.plan('快照客户', actor='operator', session=session, item_budget=20)
        return vault, store, repo, service, manifest

    def test_unchanged_finalize_is_noop_and_new_transcript_creates_one_version(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, repo, service, manifest = self._voice_run(root)
            task = manifest['items'][0]
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', execution_location='local',
            )
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-snapshot-cloud', asset_id='asset-snapshot-voice', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation=task['citation'],
            ))
            repo.insert_transcript(TranscriptRecord(
                'snapshot-transcript-v1', 'asset-snapshot-voice', task['citation'] + '#voice',
                '快照客户第一版语音证据', job_id='job-snapshot-cloud', status='active',
            ))
            service.complete(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', claim_token=claim['agent_action']['claim_token'], completion_key='snapshot-v1',
            )

            first = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='snapshot-session')
            repeated = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='snapshot-session')

            self.assertTrue(first['created'])
            self.assertEqual(first['version'], 1)
            self.assertTrue(repeated['cache_hit'])
            self.assertEqual(repeated['profile_id'], first['profile_id'])
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM profile_snapshots').fetchone()[0], 1)

            # Usage accounting is operational metadata, not profile meaning.
            # A later cost reconciliation must not manufacture a new profile.
            with store.connect() as conn:
                conn.execute(
                    'UPDATE profile_enrichment_runs SET actual_cost_rmb=? WHERE run_id=?',
                    (3.25, manifest['run_id']),
                )
                conn.commit()
            cost_reconciled = finalize_profile_snapshot(
                store, manifest['run_id'], actor='operator', session='snapshot-session',
            )
            self.assertTrue(cost_reconciled['cache_hit'])
            self.assertEqual(cost_reconciled['profile_id'], first['profile_id'])

            repo.insert_transcript(TranscriptRecord(
                'snapshot-transcript-v2', 'asset-snapshot-voice', task['citation'] + '#voice',
                '快照客户第二版语音证据', job_id='job-snapshot-cloud', status='active',
            ))
            stale = profile_snapshot_status(store, '快照客户')
            self.assertTrue(stale['stale'])
            second = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='snapshot-session')
            self.assertTrue(second['created'])
            self.assertEqual(second['version'], 2)
            self.assertNotEqual(second['content_hash'], first['content_hash'])
            self.assertFalse(profile_snapshot_status(store, '快照客户')['stale'])

    def test_terminal_gaps_are_explicit_and_nonterminal_runs_do_not_finalize(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, _, service, manifest = self._voice_run(root)
            pending = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='snapshot-session')
            self.assertFalse(pending['finalized'])
            self.assertEqual(pending['completeness_state'], 'pending')
            task = manifest['items'][0]
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', execution_location='local',
            )
            service.fail(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', claim_token=claim['agent_action']['claim_token'],
                reason='locator_routes_exhausted', terminal=True,
            )
            snapshot = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='snapshot-session')
            self.assertEqual(snapshot['completeness_state'], 'complete_with_terminal_gaps')
            self.assertEqual(snapshot['unresolved_gaps'][0]['reason'], 'locator_routes_exhausted')
            self.assertEqual(snapshot['enrichment_summary']['completeness']['families']['voice']['terminal_gaps'], 1)

    def test_snapshot_freshness_reuses_the_formal_runs_discovery_purpose(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, _, service, _default_manifest = self._voice_run(root)
            purpose = 'person_relationship_profile_enrichment'
            manifest = service.plan(
                '快照客户', actor='operator', session='person-purpose-session',
                item_budget=20, purpose=purpose,
            )
            task = manifest['items'][0]
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator',
                session='person-purpose-session', lease_owner='worker',
                execution_location='local',
            )
            service.fail(
                manifest['run_id'], task['task_id'], actor='operator',
                session='person-purpose-session', lease_owner='worker',
                claim_token=claim['agent_action']['claim_token'],
                reason='locator_routes_exhausted', terminal=True,
            )
            finalize_profile_snapshot(
                store, manifest['run_id'], actor='operator',
                session='person-purpose-session',
            )
            calls: list[str] = []
            original_discover = ProfileEnrichmentService.discover

            def record_purpose(service, customer, *, purpose='customer_profile_enrichment'):
                calls.append(purpose)
                return original_discover(service, customer, purpose=purpose)

            with patch.object(ProfileEnrichmentService, 'discover', record_purpose):
                status = profile_snapshot_status(store, '快照客户')

            self.assertFalse(status['stale'])
            self.assertEqual(calls, [purpose])

    def test_formal_finalize_retry_reuses_its_snapshot_after_newer_automatic_version(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, repo, service, manifest = self._voice_run(root)
            task = manifest['items'][0]
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', execution_location='local',
            )
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-formal-retry', asset_id='asset-snapshot-voice', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation=task['citation'],
            ))
            repo.insert_transcript(TranscriptRecord(
                'formal-retry-transcript', 'asset-snapshot-voice', task['citation'] + '#voice',
                '正式画像重试证据', job_id='job-formal-retry', status='active',
            ))
            service.complete(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', claim_token=claim['agent_action']['claim_token'],
                completion_key='formal-retry-complete',
            )
            formal = finalize_profile_snapshot(
                store, manifest['run_id'], actor='operator', session='snapshot-session',
            )
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO profile_snapshots(
                           profile_id,entity_id,version,projection_json,content_hash,source_revision,
                           schema_version,completeness_state,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        'profile-auto-between-retries', formal['entity_id'], 2, '{}',
                        'automatic-between-retries', 'auto-source', 'customer-profile/auto-v1',
                        'current', '2026-01-03T00:00:00Z',
                    ),
                )
                conn.commit()

            retried = finalize_profile_snapshot(
                store, manifest['run_id'], actor='operator', session='snapshot-session',
            )

            self.assertTrue(retried['cache_hit'])
            self.assertEqual(retried['profile_id'], formal['profile_id'])
            self.assertEqual(retried['version'], 1)
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM profile_snapshots').fetchone()[0], 2)

    def test_new_image_observation_version_creates_one_new_snapshot(self):
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
                'customer-image-snapshot', 'Customer', '图片快照客户',
                {'wechat_id': 'wxid-image-snapshot', 'remark': '图片快照客户'},
            ))
            message = Message(
                'acct', 'A', 'wxid-image-snapshot', '图片快照客户', 'private', 'wxid-image-snapshot', '图片快照客户',
                datetime(2026, 1, 1, tzinfo=timezone.utc), '[图片]', 's', 1, content_kind='image',
            )
            WeChatRepository(store).replace_fixture(
                [Account('acct', 'A', 'A')], [Conversation('wxid-image-snapshot', 'acct', '图片快照客户', 'private')], [message],
            )
            sha = hashlib.sha256(PNG_1X1).hexdigest()
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-image-snapshot', 'acct', 'message', 'image-1', 'image', 'image', message.citation,
                content_hash=sha, path_ref='sources/image.png', cache_state='cached',
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-image-snapshot', 'asset-image-snapshot', 'acct', 'message', message.citation,
                'private_chat', True, 'fixture',
            ))
            service = ProfileEnrichmentService(store)
            manifest = service.plan(
                '图片快照客户', actor='operator', session='image-snapshot', item_budget=20,
                processor_identity='agent-v1', prompt_version='p1',
            )
            task = manifest['items'][0]
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='image-snapshot',
                lease_owner='worker', execution_location='local',
            )
            service.redeem_local_media(
                manifest['run_id'], task['task_id'], actor='operator', session='image-snapshot', lease_owner='worker',
                claim_token=claim['agent_action']['claim_token'], media_capability=claim['agent_action']['media_capability'],
            )
            media_annotate(
                vault, citation=message.citation, caption='图片快照客户旧版视觉',
                model_id='agent-v1', prompt_version='p1',
            )
            service.complete(
                manifest['run_id'], task['task_id'], actor='operator', session='image-snapshot', lease_owner='worker',
                claim_token=claim['agent_action']['claim_token'], completion_key='image-snapshot-v1',
            )
            first = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='image-snapshot')
            media_annotate(
                vault, citation=message.citation, caption='图片快照客户新版视觉', visible_text='新版快照OCR',
                model_id='agent-v2', prompt_version='p2',
            )
            self.assertTrue(profile_snapshot_status(store, '图片快照客户')['stale'])
            second = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='image-snapshot')
            self.assertEqual(second['version'], first['version'] + 1)
            self.assertNotEqual(second['content_hash'], first['content_hash'])

    def test_snapshot_json_is_cited_redacted_and_contains_no_provider_payload(self):
        with tempfile.TemporaryDirectory() as root:
            _, store, repo, service, manifest = self._voice_run(root)
            task = manifest['items'][0]
            claim = service.claim(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', execution_location='local',
            )
            repo.record_provider_job(ProviderJobRecord(
                job_id='job-snapshot-safety-cloud', asset_id='asset-snapshot-voice', provider='volcengine-asr-flash',
                model='bigmodel:volc.bigasr.auc_turbo', job_type='asr', status='completed',
                request_hash='e' * 64, citation=task['citation'],
            ))
            repo.insert_transcript(TranscriptRecord(
                'snapshot-private-safety', 'asset-snapshot-voice', task['citation'] + '#voice',
                '快照客户安全证据', job_id='job-snapshot-safety-cloud', status='active',
            ))
            service.complete(
                manifest['run_id'], task['task_id'], actor='operator', session='snapshot-session',
                lease_owner='worker', claim_token=claim['agent_action']['claim_token'], completion_key='snapshot-safety',
            )
            finalized = finalize_profile_snapshot(store, manifest['run_id'], actor='operator', session='snapshot-session')
            with store.connect() as conn:
                raw = conn.execute('SELECT projection_json FROM profile_snapshots WHERE profile_id=?', (finalized['profile_id'],)).fetchone()[0]
            projection = json.loads(raw)
            serialized = json.dumps(projection, ensure_ascii=False)
            self.assertNotIn('https://signed.invalid', serialized)
            self.assertNotIn(PRIVATE_FIXTURE_PATH, serialized)
            self.assertNotIn('token=secret', serialized)
            self.assertFalse(projection['provider_payloads_included'])
            self.assertFalse(projection['raw_paths_included'])
            for rows in projection['profile']['sections'].values():
                for row in rows:
                    self.assertTrue(row['citations'])


if __name__ == '__main__':
    unittest.main()
