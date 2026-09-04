from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest
from pathlib import Path

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers import media_plan
from trove_core.providers.pricing import estimate_asr_flash_rmb
from trove_core.store.repositories import (
    ImageObservationRecord,
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MultimodalRepository,
    ProviderJobRecord,
    TranscriptRecord,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.fixture_factory import FixtureData
from trove_core.wechat.indexer import index_fixture_data
from trove_core.wechat.models import Account, Conversation, Message


_BASE = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
_CLOUD_PROVIDER = 'volcengine-asr-flash'
_CLOUD_MODEL = 'bigmodel:volc.bigasr.auc_turbo'


def _citation(account_id: str, conversation_id: str, local_id: int) -> str:
    return f'trove://wechat/{account_id}/{conversation_id}/message_0/{local_id}'


def _message(conversation: Conversation, local_id: int, *, minutes: int, sender: str = 'sender-alice', kind: str = 'text') -> Message:
    return Message(
        conversation.account_id,
        'Work',
        conversation.conversation_id,
        conversation.title,
        conversation.type,
        sender,
        'Alice' if sender == 'sender-alice' else '我',
        _BASE + timedelta(minutes=minutes),
        f'[{kind}]',
        'message_0',
        local_id,
        direction_hint='incoming' if sender == 'sender-alice' else 'outgoing',
        content_kind=kind,
    )


def _fixture() -> FixtureData:
    accounts = [Account('acct-work', 'Work', '工作')]
    alice = Conversation('conv-alice', 'acct-work', 'Alice', 'private', 2)
    team = Conversation('conv-team', 'acct-work', '产品群', 'group', 4)
    messages = [
        _message(alice, 1, minutes=10, kind='voice'),
        _message(alice, 2, minutes=20, kind='voice'),
        _message(alice, 3, minutes=30, kind='voice'),
        _message(alice, 4, minutes=40, kind='voice'),
        _message(alice, 5, minutes=50, kind='image'),
        _message(alice, 6, minutes=60, kind='image'),
        _message(alice, 7, minutes=70, kind='image'),
        _message(alice, 8, minutes=80, kind='file'),
        _message(alice, 9, minutes=90, sender='me-work', kind='image'),
        _message(team, 10, minutes=100, sender='sender-gina', kind='voice'),
    ]
    return FixtureData(accounts, [alice, team], messages)


def _asset(asset_id: str, conversation_id: str, local_id: int, modality: str, *, cache_state: str = 'metadata_only', path_ref: str | None = None, content_hash: str | None = None) -> MediaAssetRecord:
    return MediaAssetRecord(
        asset_id,
        'acct-work',
        'private_chat' if conversation_id == 'conv-alice' else 'group_chat',
        f'msg-{local_id}',
        modality,
        modality,
        _citation('acct-work', conversation_id, local_id),
        content_hash=content_hash,
        path_ref=path_ref,
        cache_state=cache_state,
    )


def _seed_media(vault: Path) -> None:
    cfg = VaultConfig.resolve(str(vault))
    repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
    repo.upsert_media_asset(_asset('asset-voice-cloud', 'conv-alice', 1, 'voice', cache_state='cached', path_ref='media/voice/1.wav', content_hash='a' * 64))
    repo.upsert_media_asset(_asset('asset-voice-local', 'conv-alice', 2, 'voice', cache_state='cached', path_ref='media/voice/2.wav', content_hash='b' * 64))
    repo.upsert_media_asset(_asset('asset-voice-stale', 'conv-alice', 3, 'voice', cache_state='cached', path_ref='media/voice/3.wav', content_hash='c' * 64))
    repo.upsert_media_asset(_asset('asset-voice-nosrc', 'conv-alice', 4, 'voice'))
    repo.upsert_media_asset(_asset('asset-image-ocr', 'conv-alice', 5, 'image', cache_state='cached', path_ref='media/image/5.jpg'))
    repo.upsert_media_asset(_asset('asset-image-caption', 'conv-alice', 6, 'image', cache_state='cached', path_ref='media/image/6.jpg'))
    repo.upsert_media_asset(_asset('asset-image-pending', 'conv-alice', 7, 'image', cache_state='cached', path_ref='media/image/7.jpg'))
    repo.upsert_media_asset(_asset('asset-file', 'conv-alice', 8, 'file'))
    repo.upsert_media_asset(_asset('asset-image-outgoing', 'conv-alice', 9, 'image'))
    repo.upsert_media_asset(_asset('asset-voice-group', 'conv-team', 10, 'voice', cache_state='cached', path_ref='media/voice/10.wav'))
    # Orphan cache media rejected by the link rule stays out of every plan.
    repo.upsert_media_asset(MediaAssetRecord(
        'asset-orphan', 'acct-work', 'excluded_orphan_cache_media', 'orphan-1', 'image', 'image',
        'trove://wechat/acct-work/orphan-cache/img-1', cache_state='cached', path_ref='media/orphan/1.jpg',
    ))
    repo.upsert_media_asset_link(MediaAssetLinkRecord(
        'link-orphan', 'asset-orphan', 'acct-work', 'excluded_orphan_cache_media',
        'trove://wechat/acct-work/orphan-cache/img-1', 'excluded_orphan_cache_media', False, 'orphan_cache_media',
    ))
    # Cloud-valid transcript: current hash, completed cloud job.
    repo.record_provider_job(ProviderJobRecord(
        job_id='job-cloud', asset_id='asset-voice-cloud', provider=_CLOUD_PROVIDER,
        model=_CLOUD_MODEL, job_type='asr', status='completed', request_hash='a' * 64,
    ))
    repo.insert_transcript(TranscriptRecord(
        'transcript-cloud', 'asset-voice-cloud', _citation('acct-work', 'conv-alice', 1),
        '云端转写正文', job_id='job-cloud', duration_seconds=4.0,
    ))
    # Local-only transcript: never a valid cache under the cloud-only voice policy.
    repo.record_provider_job(ProviderJobRecord(
        job_id='job-local', asset_id='asset-voice-local', provider='local-faster-whisper',
        model='base:local', job_type='asr', status='completed', request_hash='b' * 64,
    ))
    repo.insert_transcript(TranscriptRecord(
        'transcript-local', 'asset-voice-local', _citation('acct-work', 'conv-alice', 2),
        '本地转写正文', job_id='job-local', duration_seconds=6.0,
    ))
    # Stale cloud transcript: the asset content moved on (request hash mismatch).
    repo.record_provider_job(ProviderJobRecord(
        job_id='job-stale', asset_id='asset-voice-stale', provider=_CLOUD_PROVIDER,
        model=_CLOUD_MODEL, job_type='asr', status='completed', request_hash='d' * 64,
    ))
    repo.insert_transcript(TranscriptRecord(
        'transcript-stale', 'asset-voice-stale', _citation('acct-work', 'conv-alice', 3),
        '过期转写正文', job_id='job-stale', duration_seconds=2.0,
    ))
    repo.insert_image_observation(ImageObservationRecord(
        'obs-ocr', 'asset-image-ocr', _citation('acct-work', 'conv-alice', 5), '', visible_text='发票 总计 100 元',
    ))
    repo.insert_image_observation(ImageObservationRecord(
        'obs-caption', 'asset-image-caption', _citation('acct-work', 'conv-alice', 6), '一张白板照片',
    ))


class _PlanVaultCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_data(self.vault, _fixture(), reset=True)
        _seed_media(self.vault)
        self.config = VaultConfig.resolve(str(self.vault))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dispatcher(self):
        return build_default_dispatcher(self.vault)

    def _plan(self, payload=None, request_id='req-plan'):
        return self.dispatcher().dispatch('trove.media_enrich_plan', payload or {'account_id': 'acct-work'}, request_id=request_id)


class MediaEnrichPlanTests(_PlanVaultCase):
    def test_conversation_scope_counts_and_plan(self):
        response = self._plan({'conversation_id': 'conv-alice'})
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['scope'], {
            'account_id': 'acct-work', 'conversation_id': 'conv-alice',
            'author_id': None, 'since': None, 'until': None,
        })
        self.assertEqual(data['filters'], {'media_types': ['image', 'voice', 'file'], 'kinds': ['ocr', 'caption', 'transcribe'], 'execution': 'auto'})
        totals = data['scope_totals']
        self.assertEqual(totals['state'], 'exact')
        self.assertEqual(totals['media_assets'], 9)
        self.assertEqual(totals['by_modality'], {'file': 1, 'image': 4, 'voice': 4})
        self.assertFalse(data['truncated'])
        self.assertEqual(response['coverage'], {'state': 'complete', 'returned': 8, 'remaining': 0})

        plan = {entry['kind']: entry for entry in data['plan']}
        ocr = plan['ocr']
        self.assertEqual(ocr['execution'], 'local')
        self.assertEqual(ocr['provider'], 'local-macos-vision')
        self.assertEqual(ocr['media_enrich_kind'], 'annotate')
        self.assertEqual((ocr['candidates'], ocr['understood'], ocr['pending'], ocr['pending_no_source']), (4, 1, 2, 1))
        self.assertEqual(ocr['estimated_cost_rmb'], 0.0)
        self.assertFalse(ocr['approval_required'])
        caption = plan['caption']
        self.assertEqual((caption['candidates'], caption['understood'], caption['pending'], caption['pending_no_source']), (4, 1, 2, 1))
        self.assertEqual(caption['provider'], 'local-vlm-qwen25-vl')

        transcribe = plan['transcribe']
        self.assertEqual(transcribe['execution'], 'cloud')
        self.assertEqual(transcribe['provider'], _CLOUD_PROVIDER)
        self.assertEqual(transcribe['media_enrich_kind'], 'transcribe')
        # asset-voice-local (local transcript) and asset-voice-stale (hash
        # mismatch) stay pending; only the current cloud projection counts.
        self.assertEqual((transcribe['candidates'], transcribe['understood']), (4, 1))
        self.assertEqual((transcribe['pending'], transcribe['pending_no_source']), (2, 1))
        self.assertTrue(transcribe['approval_required'])
        self.assertEqual(transcribe['required_approval'], {
            'grant_action': 'voice_cloud_asr',
            'danger_class': 'cloud_asr_upload',
            'granularity': 'per_citation',
        })
        self.assertEqual(transcribe['audio_seconds_basis'], 'transcript_duration_sample_average')
        self.assertEqual(transcribe['audio_seconds_sample'], 3)
        # Average of the 4s/6s/2s sampled transcript durations.
        self.assertEqual(transcribe['estimated_audio_seconds'], 8.0)
        self.assertEqual(transcribe['estimated_cost_rmb'], round(2 * estimate_asr_flash_rmb(4.0), 6))
        self.assertEqual(transcribe['estimated_seconds'], 16.0)
        self.assertEqual(transcribe['duration_tier'], 'under_a_minute')
        self.assertFalse(transcribe['cloud_calls_made'])
        self.assertIsNone(transcribe['estimated_tokens'])
        self.assertTrue(any('cloud ASR' in note for note in data['notes']))
        self.assertTrue(any('media_fetch' in note for note in data['notes']))

    def test_group_voice_is_out_of_scope_for_transcribe(self):
        response = self._plan({'conversation_id': 'conv-team', 'kinds': ['transcribe']})
        self.assertTrue(response['ok'], response.get('error'))
        entry = response['data']['plan'][0]
        self.assertEqual(entry['kind'], 'transcribe')
        self.assertEqual((entry['candidates'], entry['out_of_scope'], entry['pending']), (1, 1, 0))
        self.assertEqual(response['data']['candidates']['excluded'], 1)

    def test_time_window_filters_by_message_timestamp(self):
        response = self._plan({
            'conversation_id': 'conv-alice',
            'since': '2026-08-20T12:45:00Z',
            'until': '2026-08-20T13:25:00Z',
        })
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        # Minutes 50/60/70 (incoming images) and 80 (file) fall inside
        # 12:45-13:25; the minute-90 outgoing image does not.
        self.assertEqual(data['scope_totals']['by_modality'], {'file': 1, 'image': 3})
        plan = {entry['kind']: entry for entry in data['plan']}
        self.assertEqual(plan['ocr']['candidates'], 3)
        self.assertEqual(plan['transcribe']['candidates'], 0)
        self.assertEqual(data['scope']['since'], '2026-08-20T12:45:00Z')

    def test_account_scope_aggregates_and_evaluates_cached_first(self):
        response = self._plan({'account_id': 'acct-work', 'limit': 3})
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        totals = data['scope_totals']
        self.assertEqual(totals['state'], 'exact')
        self.assertEqual(totals['media_assets'], 11)
        self.assertEqual(totals['by_modality'], {'file': 1, 'image': 5, 'voice': 5})
        # limit=3 gathers the cached tier first: three of the cached assets.
        self.assertEqual(data['candidates']['evaluated'], 3)
        self.assertEqual(data['candidates']['with_source'], 3)
        self.assertTrue(data['truncated'])
        self.assertEqual(response['coverage']['state'], 'partial')
        self.assertEqual(response['coverage']['remaining'], 7)
        self.assertTrue(any('limit=3' in note for note in data['notes']))

    def test_account_scope_includes_orphan_exclusion_in_classification(self):
        response = self._plan({'account_id': 'acct-work', 'kinds': ['ocr'], 'limit': 1000})
        self.assertTrue(response['ok'], response.get('error'))
        entry = response['data']['plan'][0]
        # Five images total; the rejected orphan-cache asset is excluded.
        self.assertEqual(entry['candidates'], 5)
        self.assertEqual(entry['out_of_scope'], 1)
        self.assertEqual(entry['understood'], 1)
        self.assertEqual(entry['pending'], 2)
        self.assertEqual(entry['pending_no_source'], 1)

    def test_author_scope_scans_sender_window(self):
        response = self._plan({'author_id': 'sender-alice'})
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        totals = data['scope_totals']
        self.assertEqual(totals['state'], 'scan_capped')
        self.assertIsNone(totals['media_assets'])
        self.assertEqual(totals['messages_total'], 8)
        self.assertEqual(totals['messages_scanned'], 8)
        self.assertEqual(data['candidates']['by_modality'], {'image': 3, 'voice': 4})
        plan = {entry['kind']: entry for entry in data['plan']}
        self.assertEqual(plan['transcribe']['understood'], 1)

        outgoing = self._plan({'author_id': 'me-work'})
        self.assertEqual(outgoing['data']['candidates']['by_modality'], {'image': 1})

        missing = self._plan({'author_id': 'sender-nobody'})
        self.assertFalse(missing['ok'])
        self.assertEqual(missing['error']['code'], 'no_results')

    def test_media_type_and_kind_filters(self):
        files = self._plan({'conversation_id': 'conv-alice', 'media_types': ['file']})
        self.assertTrue(files['ok'])
        self.assertEqual(files['data']['scope_totals']['by_modality'], {'file': 1})
        self.assertEqual(files['data']['plan'], [])
        self.assertEqual(files['data']['candidates']['evaluated'], 0)
        self.assertTrue(any('no understanding pipeline' in note for note in files['data']['notes']))

        mismatched = self._plan({'conversation_id': 'conv-alice', 'kinds': ['transcribe'], 'media_types': ['image']})
        self.assertTrue(mismatched['ok'])
        self.assertEqual(mismatched['data']['plan'], [])
        self.assertTrue(any('outside media_types' in note for note in mismatched['data']['notes']))

        images_only = self._plan({'conversation_id': 'conv-alice', 'media_types': ['image'], 'kinds': ['ocr']})
        self.assertEqual([entry['kind'] for entry in images_only['data']['plan']], ['ocr'])
        self.assertEqual(images_only['data']['scope_totals']['by_modality'], {'image': 4})

    def test_local_only_excludes_transcribe(self):
        response = self._plan({'conversation_id': 'conv-alice', 'execution': 'local_only'})
        self.assertTrue(response['ok'], response.get('error'))
        plan = {entry['kind']: entry for entry in response['data']['plan']}
        transcribe = plan['transcribe']
        self.assertFalse(transcribe['included'])
        self.assertEqual(transcribe['exclusion_reason'], 'cloud_only_kind_excluded_by_execution_filter')
        self.assertFalse(transcribe['approval_required'])
        # Counts and estimates stay informative under the exclusion.
        self.assertEqual(transcribe['pending'], 2)
        self.assertGreater(transcribe['estimated_cost_rmb'], 0)
        self.assertTrue(all(entry['included'] for kind, entry in plan.items() if kind != 'transcribe'))
        self.assertTrue(any('cloud-only' in note for note in response['data']['notes']))

    def test_scope_validation_fails_typed(self):
        dispatcher = self.dispatcher()
        both = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'conversation_id': 'conv-alice', 'author_id': 'sender-alice'},
            request_id='req-both',
        )
        self.assertEqual(both['error']['code'], 'invalid_request')
        missing_scope = dispatcher.dispatch('trove.media_enrich_plan', {}, request_id='req-none')
        self.assertEqual(missing_scope['error']['code'], 'invalid_request')
        account_time = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'account_id': 'acct-work', 'since': '2026-08-01T00:00:00Z'},
            request_id='req-acct-time',
        )
        self.assertEqual(account_time['error']['code'], 'invalid_request')
        bad_execution = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'account_id': 'acct-work', 'execution': 'cloud_first'},
            request_id='req-exec',
        )
        self.assertEqual(bad_execution['error']['code'], 'invalid_request')
        bad_kind = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'account_id': 'acct-work', 'kinds': ['summarize']},
            request_id='req-kind',
        )
        self.assertEqual(bad_kind['error']['code'], 'invalid_request')
        bad_media = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'account_id': 'acct-work', 'media_types': ['video']},
            request_id='req-media',
        )
        self.assertEqual(bad_media['error']['code'], 'invalid_request')
        inverted = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'conversation_id': 'conv-alice', 'since': '2026-09-01T00:00:00Z', 'until': '2026-08-01T00:00:00Z'},
            request_id='req-inverted',
        )
        self.assertEqual(inverted['error']['code'], 'invalid_request')
        for limit in (0, 1001):
            response = dispatcher.dispatch(
                'trove.media_enrich_plan', {'account_id': 'acct-work', 'limit': limit}, request_id=f'req-limit-{limit}',
            )
            self.assertEqual(response['error']['code'], 'invalid_request')

    def test_conversation_scope_resolution_is_exact_and_typed(self):
        missing = self._plan({'conversation_id': 'conv-missing'})
        self.assertFalse(missing['ok'])
        self.assertEqual(missing['error']['code'], 'no_results')
        ambiguous = self._plan({'conversation_id': 'conv-dup'})
        self.assertEqual(ambiguous['error']['code'], 'no_results')

    def test_ambiguous_conversation_across_accounts_fails_typed(self):
        vault = Path(self.temp.name) / 'vault-dup'
        accounts = [Account('acct-a', 'A', '甲'), Account('acct-b', 'B', '乙')]
        conv_a = Conversation('conv-dup', 'acct-a', 'Dup A', 'private', 2)
        conv_b = Conversation('conv-dup', 'acct-b', 'Dup B', 'private', 2)
        index_fixture_data(vault, FixtureData(accounts, [conv_a, conv_b], [
            _message(conv_a, 1, minutes=10, kind='image'),
            _message(conv_b, 1, minutes=10, kind='image'),
        ]), reset=True)
        repo = MultimodalRepository(SQLiteStore(VaultConfig.resolve(str(vault)).paths.sqlite_path))
        for account_id in ('acct-a', 'acct-b'):
            repo.upsert_media_asset(MediaAssetRecord(
                f'asset-{account_id}', account_id, 'private_chat', 'msg-1', 'image', 'image',
                _citation(account_id, 'conv-dup', 1), cache_state='cached', path_ref='media/dup/1.jpg',
            ))
        dispatcher = build_default_dispatcher(vault)
        ambiguous = dispatcher.dispatch('trove.media_enrich_plan', {'conversation_id': 'conv-dup'}, request_id='req-dup')
        self.assertFalse(ambiguous['ok'])
        self.assertEqual(ambiguous['error']['code'], 'ambiguous_target')
        resolved = dispatcher.dispatch(
            'trove.media_enrich_plan',
            {'conversation_id': 'conv-dup', 'account_id': 'acct-b'},
            request_id='req-dup-b',
        )
        self.assertTrue(resolved['ok'], resolved.get('error'))
        self.assertEqual(resolved['data']['scope']['account_id'], 'acct-b')
        self.assertEqual(resolved['data']['scope_totals']['by_modality'], {'image': 1})

    def test_preview_queries_use_bounded_indexes(self):
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    # Conversation totals: constant-prefix citation range.
                    (
                        'SELECT ma.modality, ma.cache_state, COUNT(*) AS n'
                        ' FROM media_assets ma INDEXED BY idx_media_assets_citation'
                        ' WHERE ma.citation GLOB ? AND ma.modality IN (?,?)'
                        ' GROUP BY ma.modality, ma.cache_state',
                        ('trove://wechat/acct-work/conv-alice/*', 'image', 'voice'),
                    ),
                    # Conversation candidates with a message-time window:
                    # pinned per-asset citation seek into messages.
                    (
                        'SELECT ma.asset_id FROM media_assets ma INDEXED BY idx_media_assets_citation'
                        ' CROSS JOIN messages m INDEXED BY idx_messages_citation ON m.citation = ma.citation'
                        ' WHERE ma.citation GLOB ? AND ma.modality IN (?) AND m.timestamp>=? AND m.timestamp<?'
                        ' ORDER BY ma.citation LIMIT ?',
                        ('trove://wechat/acct-work/conv-alice/*', 'image', '2026-08-20T12:45:00Z', '2026-08-20T13:20:00Z', 21),
                    ),
                    # Account totals: covering (account, modality, cache_state).
                    (
                        'SELECT ma.modality, ma.cache_state, COUNT(*) AS n'
                        ' FROM media_assets ma INDEXED BY idx_media_assets_account_modality'
                        ' WHERE ma.account_id=? AND ma.modality IN (?,?)'
                        ' GROUP BY ma.modality, ma.cache_state',
                        ('acct-work', 'image', 'voice'),
                    ),
                    # Account candidate ids: one covering tier seek.
                    (
                        'SELECT ma.asset_id FROM media_assets ma INDEXED BY idx_media_assets_account_modality'
                        ' WHERE ma.account_id=? AND ma.modality=? AND ma.cache_state=? LIMIT ?',
                        ('acct-work', 'image', 'cached', 21),
                    ),
                    # Author window: sender-leading bounded enumeration.
                    (
                        'SELECT m.citation, m.timestamp FROM messages m INDEXED BY idx_messages_sender_time'
                        ' WHERE m.sender_id=? ORDER BY m.timestamp DESC, m.citation DESC LIMIT ?',
                        ('sender-alice', 5001),
                    ),
                    # Eligibility probe: per-asset link seek.
                    (
                        'SELECT l.asset_id, MAX(l.accepted) AS any_accepted'
                        ' FROM media_asset_links l INDEXED BY idx_media_asset_links_asset'
                        ' WHERE l.asset_id IN (?,?) GROUP BY l.asset_id',
                        ('asset-orphan', 'asset-image-ocr'),
                    ),
                    # Private-chat proof: citation seek into messages.
                    (
                        "SELECT m.citation FROM messages m INDEXED BY idx_messages_citation"
                        " WHERE m.citation IN (?,?) AND m.conversation_type='private'",
                        (_citation('acct-work', 'conv-alice', 1), _citation('acct-work', 'conv-team', 10)),
                    ),
                ):
                    plans.extend(
                        str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params)
                    )
        finally:
            store.close()
        plan_text = ' | '.join(plans)
        self.assertNotIn('SCAN media_assets', plan_text)
        self.assertNotIn('SCAN messages', plan_text)
        self.assertNotIn('SCAN l', plan_text)
        self.assertIn('SEARCH ma USING INDEX idx_media_assets_citation (citation>? AND citation<?)', plan_text)
        self.assertIn('SEARCH m USING INDEX idx_messages_citation (citation=?)', plan_text)
        self.assertIn('SEARCH ma USING COVERING INDEX idx_media_assets_account_modality (account_id=? AND modality=', plan_text)
        self.assertIn('SEARCH m USING INDEX idx_messages_sender_time (sender_id=?)', plan_text)
        self.assertIn('SEARCH l USING COVERING INDEX idx_media_asset_links_asset (asset_id=?)', plan_text)

    def test_response_never_carries_content_or_paths(self):
        response = self._plan({'account_id': 'acct-work'})
        self.assertTrue(response['ok'])
        self.assertFalse(response['data']['raw_content_included'])
        self.assertFalse(response['data']['raw_paths_included'])
        encoded = json.dumps(response, ensure_ascii=False)
        for forbidden in ('云端转写正文', '本地转写正文', '过期转写正文', '发票 总计', '一张白板照片', 'media/voice/1.wav', 'media/orphan/1.jpg'):
            self.assertNotIn(forbidden, encoded)

    def test_missing_vault_returns_bounded_empty_result(self):
        missing = Path(self.temp.name) / 'missing-vault'
        config = VaultConfig.resolve(str(missing), env={})
        outcome = media_plan.media_enrich_plan(config, {'account_id': 'acct-work'})
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.data['scope_totals']['media_assets'], 0)
        self.assertEqual(outcome.data['plan'], [])
        self.assertEqual(outcome.coverage['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
