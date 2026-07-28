from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trove_core.application.sensitive_commands import derived_data_purge_payload, execute_derived_data_purge
from trove_core.approvals import ApprovalManager, ApprovalValidationError
from trove_core.knowledge.profile_enrichment import ProfileEnrichmentService
from trove_core.maintain import rotate_sqlite_backups
from trove_core.store.repositories import (
    EntityRecord,
    ImageObservationRecord,
    MediaAssetLinkRecord,
    MediaAssetRecord,
    MediaUnderstandingRecord,
    MultimodalRepository,
    ObservationRecord,
    ProviderJobRecord,
    TranscriptRecord,
    WeChatRepository,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.operations import derived_data_lifecycle
from trove_core.wechat.media.locator import MediaLocatorResult
from trove_core.wechat.media.materializer import materialize_media_asset
from trove_core.wechat.models import Account, Conversation, Message


class DerivedDataLifecycleTests(unittest.TestCase):
    def _approve_purge(self, vault: Path, *, scope_type: str, scope_id: str):
        payload = derived_data_purge_payload(scope_type=scope_type, scope_id=scope_id)
        return ApprovalManager(vault).require(
            'derived_data_purge', 'delete_or_purge', payload, one_step_approval=True,
        )

    def _entity_fixture(self, root: Path):
        vault = root / 'vault'
        cfg = VaultConfig.resolve(str(vault), env={})
        cfg.ensure()
        store = SQLiteStore(cfg.paths.sqlite_path)
        repo = MultimodalRepository(store)
        entity_id = 'customer-purge-fixture'
        conversation_id = 'conv-purge-fixture'
        repo.upsert_entity(EntityRecord(
            entity_id, 'Customer', 'Purge Fixture Customer',
            {'wechat_id': 'wxid-purge-fixture', 'conversation_ids': [conversation_id]},
        ))
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        voice = Message(
            'acct-purge', 'A', conversation_id, 'Purge Fixture Customer', 'private',
            'wxid-purge-fixture', 'Purge Fixture Customer', timestamp,
            'privacy purge voice needle', 'message_0', 1, content_kind='voice',
        )
        image = Message(
            'acct-purge', 'A', conversation_id, 'Purge Fixture Customer', 'private',
            'wxid-purge-fixture', 'Purge Fixture Customer', timestamp,
            'privacy purge image needle', 'message_0', 2, content_kind='image',
        )
        WeChatRepository(store).replace_fixture(
            [Account('acct-purge', 'A', 'A')],
            [Conversation(conversation_id, 'acct-purge', 'Purge Fixture Customer', 'private')],
            [voice, image],
        )
        materialized = vault / 'media' / 'materialized' / 'aa' / 'voice.wav'
        decoded = vault / 'media' / 'decoded' / 'bb' / 'image.png'
        for path, data in ((materialized, b'RIFFfixture'), (decoded, b'PNGfixture')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o600)
        assets = [
            MediaAssetRecord(
                'asset-purge-voice', 'acct-purge', 'message', voice.citation,
                'voice', 'voice', voice.citation, content_hash='a' * 64,
                path_ref=str(materialized.relative_to(vault)), cache_state='cached',
            ),
            MediaAssetRecord(
                'asset-purge-image', 'acct-purge', 'message', image.citation,
                'image', 'image', image.citation, content_hash='b' * 64,
                path_ref=str(decoded.relative_to(vault)), cache_state='cached',
            ),
        ]
        for asset in assets:
            repo.upsert_media_asset(asset)
            repo.upsert_media_asset_link(MediaAssetLinkRecord(
                'link-' + asset.asset_id, asset.asset_id, 'acct-purge', 'message',
                asset.citation, 'private_chat', True, 'fixture',
            ))
        repo.record_provider_job(ProviderJobRecord(
            'job-purge-voice', 'fixture-provider', 'fixture-model', 'asr', 'completed',
            asset_id='asset-purge-voice', citation=voice.citation,
        ))
        repo.record_provider_job(ProviderJobRecord(
            'job-purge-image', 'fixture-provider', 'fixture-model', 'vision', 'completed',
            asset_id='asset-purge-image', citation=image.citation,
        ))
        repo.insert_transcript(TranscriptRecord(
            'transcript-purge', 'asset-purge-voice', voice.citation + '#voice',
            'privacy purge transcript needle', job_id='job-purge-voice',
        ))
        repo.merge_image_observation(ImageObservationRecord(
            'imageobs-purge', 'asset-purge-image', image.citation + '#image',
            'privacy purge image observation needle', job_id='job-purge-image',
            content_sha256='b' * 64, model_id='fixture-model', prompt_version='p1', status='active',
        ))
        repo.upsert_media_understanding(MediaUnderstandingRecord(
            'b' * 64, 'image', 'fixture-model', 'p1',
            caption='privacy purge understanding needle', source_citations=[image.citation],
        ))
        repo.record_decode_result(
            decode_id='decode-purge', asset_id='asset-purge-image', status='decoded',
            derivative_ref=str(decoded.relative_to(vault)),
        )
        repo.add_observation(ObservationRecord(
            'observation-purge', entity_id, 'Need', {'text': 'privacy purge ontology needle'},
            'active', 0.9, voice.citation, 'transcript',
        ))
        service = ProfileEnrichmentService(store)
        manifest = service.plan(
            'Purge Fixture Customer', actor='operator', session='purge-fixture', item_budget=20,
            processor_identity='fixture-model', prompt_version='p1',
        )
        related_approval = ApprovalManager(vault).request(
            'voice_cloud_asr', 'cloud_asr_upload', {'scope': 'fixture-task'},
        )
        with store.connect() as conn:
            task_id = conn.execute(
                "SELECT task_id FROM profile_enrichment_tasks WHERE run_id=? ORDER BY task_id LIMIT 1",
                (manifest['run_id'],),
            ).fetchone()[0]
            conn.execute(
                "UPDATE profile_enrichment_tasks SET approval_id=?,approval_required=1 WHERE task_id=?",
                (related_approval.approval_id, task_id),
            )
            conn.execute(
                """INSERT INTO profile_snapshots(
                       profile_id,entity_id,version,projection_json,content_hash,source_revision,run_id,
                       schema_version,completeness_state,evidence_citations_json,enrichment_summary_json,gaps_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    'profile-purge', entity_id, 1, '{}', 'c' * 64, manifest['source_revision'],
                    manifest['run_id'], 'fixture/v1', 'complete',
                    json.dumps([voice.citation, image.citation]), '{}', '[]',
                    '2026-01-01T00:00:00Z',
                ),
            )
            conn.execute(
                """INSERT INTO profile_automation_subscriptions(
                       entity_id,selector,enabled,debounce_seconds,consent_scope,last_profile_id,
                       created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    entity_id, 'wxid-purge-fixture', 1, 180,
                    'explicit-profile-auto-maintenance-v1', 'profile-purge',
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                ),
            )
            conn.execute(
                """INSERT INTO profile_refresh_queue(
                       entity_id,generation,state,reason,available_at,created_at,updated_at)
                   VALUES(?,1,'pending','fixture',?,?,?)""",
                (
                    entity_id, '2026-01-01T00:03:00Z',
                    '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                ),
            )
            conn.execute(
                """INSERT OR REPLACE INTO evidence_chunks(
                       chunk_id,chunk_citation,parent_citation,account_id,account_label,source_type,source_id,
                       title,actor,timestamp,content,chunk_index,metadata_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    'chunk-purge', voice.citation + '#voice#chunk-0', voice.citation + '#voice',
                    'acct-purge', 'A', 'transcript', conversation_id, '', '', '',
                    'privacy purge evidence needle', 0, '{}', 'active', '2026-01-01T00:00:00Z',
                ),
            )
            conn.execute(
                'INSERT INTO vector_entries(citation,provider,dimensions,vector_json,content_hash) VALUES(?,?,?,?,?)',
                (voice.citation + '#voice#chunk-0', 'fixture', 2, '[1,0]', 'd' * 64),
            )
            conn.commit()
        preview = vault / 'media' / 'previews' / 'fixture.png'
        temp_path = vault / 'media' / 'tmp' / '.materialize-fixture'
        for path in (preview, temp_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
            path.chmod(0o600)
        rotate_sqlite_backups(cfg.paths.sqlite_path, retention=3, create=True)
        return vault, store, entity_id, manifest, related_approval.approval_id, materialized, decoded

    def test_entity_purge_removes_database_files_approvals_and_old_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            vault, store, entity_id, manifest, approval_id, materialized, decoded = self._entity_fixture(Path(directory))

            result = execute_derived_data_purge(
                vault,
                scope_type='entity',
                scope_id=entity_id,
                approval_grant=self._approve_purge(vault, scope_type='entity', scope_id=entity_id),
            )

            self.assertTrue(result['ok'])
            serialized = json.dumps(result, ensure_ascii=False)
            self.assertNotIn(entity_id, serialized)
            self.assertFalse(result['raw_content_included'])
            self.assertFalse(result['raw_paths_included'])
            self.assertFalse(materialized.exists())
            self.assertFalse(decoded.exists())
            self.assertFalse((vault / 'media' / 'previews').exists())
            self.assertFalse((vault / 'media' / 'tmp').exists())
            with self.assertRaises(ApprovalValidationError):
                ApprovalManager(vault).load(approval_id)
            with store.connect() as conn:
                for table in (
                    'entities', 'entity_identifiers', 'observations', 'relationships',
                    'profile_snapshots', 'profile_enrichment_runs', 'profile_enrichment_tasks',
                    'profile_automation_subscriptions', 'profile_refresh_queue',
                    'media_assets', 'media_asset_links', 'provider_jobs', 'transcripts',
                    'image_observations', 'media_understanding', 'evidence_chunks', 'vector_entries',
                ):
                    self.assertEqual(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0, table)
                audit = conn.execute('SELECT * FROM derived_data_purge_audit').fetchone()
                self.assertEqual(audit['scope_type'], 'entity')
                self.assertNotIn(entity_id, audit['counts_json'])
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)
            backups = list((vault / 'index').glob('trove.sqlite.bak-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(os.stat(backups[0]).st_mode & 0o077, 0)
            self.assertEqual(store.exact_search('privacy purge needle'), [])
            self.assertEqual(result['backups']['pre_purge_removed'], 1)
            self.assertEqual(result['backups']['post_purge_retained'], 1)
            self.assertEqual(manifest['run_id'].startswith('enrichrun-'), True)

    def test_task_purge_removes_reusable_derivatives_but_keeps_reset_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            vault, store, _entity_id, manifest, _approval_id, materialized, _decoded = self._entity_fixture(Path(directory))
            with store.connect() as conn:
                task = conn.execute(
                    "SELECT task_id,asset_id FROM profile_enrichment_tasks WHERE run_id=? AND asset_id='asset-purge-voice'",
                    (manifest['run_id'],),
                ).fetchone()

            execute_derived_data_purge(
                vault,
                scope_type='task',
                scope_id=task['task_id'],
                approval_grant=self._approve_purge(vault, scope_type='task', scope_id=task['task_id']),
            )

            with store.connect() as conn:
                self.assertIsNone(conn.execute(
                    'SELECT 1 FROM profile_enrichment_tasks WHERE task_id=?', (task['task_id'],),
                ).fetchone())
                asset = conn.execute('SELECT * FROM media_assets WHERE asset_id=?', (task['asset_id'],)).fetchone()
                self.assertIsNotNone(asset)
                self.assertIsNone(asset['path_ref'])
                self.assertIsNone(asset['content_hash'])
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM transcripts WHERE asset_id=?', (task['asset_id'],)).fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT state FROM profile_enrichment_runs WHERE run_id=?', (manifest['run_id'],)).fetchone()[0], 'cancelled')
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM profile_snapshots WHERE run_id=?', (manifest['run_id'],)).fetchone()[0], 0)
                subscription = conn.execute(
                    'SELECT last_profile_id FROM profile_automation_subscriptions'
                ).fetchone()
                self.assertIsNone(subscription['last_profile_id'])
                queue = conn.execute(
                    'SELECT state,reason FROM profile_refresh_queue'
                ).fetchone()
                self.assertEqual((queue['state'], queue['reason']), ('pending', 'derived_data_purged'))
            self.assertFalse(materialized.exists())

    def test_source_purge_removes_bound_source_root_and_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            source = vault / 'sources' / 'purge-source'
            source.mkdir(parents=True)
            (source / 'fixture.bin').write_bytes(b'fixture')
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-source-purge', 'acct', 'message', 'source', 'image', 'image',
                'trove://fixture/source-purge', cache_state='source_available',
            ))
            revision = 'e' * 64
            with store.connect() as conn:
                conn.execute(
                    "INSERT INTO source_snapshots VALUES(?,?,?,?,?,?,?)",
                    (revision, str(source.relative_to(vault)), 'f' * 64, None, 'available', 'now', 'now'),
                )
                conn.execute(
                    "INSERT INTO media_source_bindings VALUES(?,?,?,?,?,?,?)",
                    ('asset-source-purge', revision, 'a' * 64, '{}', 'bound', 'now', 'now'),
                )
                conn.commit()

            result = execute_derived_data_purge(
                vault,
                scope_type='source',
                scope_id=revision,
                approval_grant=self._approve_purge(vault, scope_type='source', scope_id=revision),
            )

            self.assertFalse(source.exists())
            self.assertEqual(result['filesystem']['source_roots_removed'], 1)
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM source_snapshots').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_assets').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_source_bindings').fetchone()[0], 0)

    def test_source_purge_unlinks_root_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            outside = root / 'outside-source'
            outside.mkdir()
            protected = outside / 'keep.bin'
            protected.write_bytes(b'keep')
            source_link = vault / 'sources' / 'linked-source'
            source_link.parent.mkdir(parents=True, exist_ok=True)
            source_link.symlink_to(outside, target_is_directory=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            store.initialize()
            revision = '1' * 64
            with store.connect() as conn:
                conn.execute(
                    "INSERT INTO source_snapshots VALUES(?,?,?,?,?,?,?)",
                    (revision, str(source_link.relative_to(vault)), '2' * 64, None, 'available', 'now', 'now'),
                )
                conn.commit()

            result = execute_derived_data_purge(
                vault,
                scope_type='source',
                scope_id=revision,
                approval_grant=self._approve_purge(vault, scope_type='source', scope_id=revision),
            )

            self.assertEqual(result['filesystem']['source_roots_removed'], 1)
            self.assertFalse(source_link.exists())
            self.assertTrue(protected.exists())

    def test_post_database_cleanup_failure_marks_purge_audit_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            vault, store, entity_id, _manifest, _approval_id, _materialized, _decoded = self._entity_fixture(Path(directory))
            with patch(
                'trove_core.vault.operations._clear_runtime_cache_dir',
                side_effect=RuntimeError('synthetic cleanup failure'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'synthetic cleanup failure'):
                    execute_derived_data_purge(
                        vault,
                        scope_type='entity',
                        scope_id=entity_id,
                        approval_grant=self._approve_purge(vault, scope_type='entity', scope_id=entity_id),
                    )
            with store.connect() as conn:
                audit = conn.execute(
                    'SELECT status FROM derived_data_purge_audit ORDER BY created_at DESC LIMIT 1',
                ).fetchone()
            self.assertEqual(audit['status'], 'failed')

    def test_materialization_temp_is_private_and_removed_on_baseexception(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            source = vault / 'sources' / 'voice.wav'
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b'RIFFfixture')
            asset = {'asset_id': 'asset-interrupt', 'modality': 'voice'}

            def interrupted(*_args, **_kwargs):
                temp_files = list((vault / 'media' / 'tmp').glob('.materialize-*'))
                self.assertEqual(len(temp_files), 1)
                self.assertEqual(os.stat(temp_files[0]).st_mode & 0o077, 0)
                raise KeyboardInterrupt()

            with patch(
                'trove_core.wechat.media.materializer.locate_media_asset',
                return_value=MediaLocatorResult(
                    'local', route='source_snapshot', path=source,
                    snapshot_revision='fixture-revision',
                ),
            ), patch(
                'trove_core.wechat.media.materializer.resolve_snapshot_root',
                return_value=(source.parent, None),
            ), patch(
                'trove_core.wechat.media.materializer._copy_reader',
                side_effect=interrupted,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    materialize_media_asset(cfg, store, asset, citation='trove://fixture/interrupt')

            self.assertEqual(list((vault / 'media' / 'tmp').glob('.materialize-*')), [])

    def test_lifecycle_matrix_declares_provider_and_audit_contracts(self):
        matrix = derived_data_lifecycle()
        self.assertFalse(matrix['provider_request_response_logging'])
        artifacts = {item['artifact'] for item in matrix['artifacts']}
        self.assertIn('sqlite_backups', artifacts)
        self.assertIn('redacted_purge_audit', artifacts)
        self.assertFalse(matrix['raw_content_included'])


if __name__ == '__main__':
    unittest.main()
