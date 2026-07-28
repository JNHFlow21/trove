from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.knowledge.wiki import write_wiki_page
from trove_core.maintain import MaintainOptions, run_maintain
from trove_core.media_fetch import fetch_media
from trove_core.media_pipeline import (
    ensure_voice_transcript,
    run_image_caption_budget,
    run_image_observation_budget,
    run_media_maintenance,
    run_voice_transcription_budget,
    voice_transcription_plan,
)
from trove_core.media_understanding import annotate_media_understanding, invalidate_media_understanding
from trove_core.runtime import index_vectors, rebuild_vectors_atomic
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.sync import SyncOptions, run_sync
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import MutationOutsideCoordinator, VaultOperationCoordinator
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.mutations import VAULT_MUTATION_INVENTORY, mutation_entrypoint, mutation_operation
from trove_core.vault.operations import (
    backfill_content_kind,
    initialize_index,
    purge_derived_data,
    rebuild_chunks,
    rebuild_fts,
    rebuild_scope,
    reset_index_cache,
)
from trove_core.wechat.decrypt.runner import run_decrypt_plan
from trove_core.wechat.decrypt.status import rollback_current
from trove_core.wechat.files import archive_files
from trove_core.wechat.import_job import run_import_job
from trove_core.wechat.appmsg_backfill import backfill_appmsg_payloads
from trove_core.wechat.media.backfill import backfill_message_media_references
from trove_core.wechat.indexer import index_fixture_data, index_fixture_vault
from trove_core.wechat.process_config import process_config_from_payload, write_process_config


class _UnusedEmbeddingProvider:
    dimensions = 3
    provider_name = 'unused'
    model_id = 'unused'


class VaultMutationCoverageTests(unittest.TestCase):
    def test_registered_entrypoint_cannot_return_after_bypassing_coordinator(self):
        @mutation_entrypoint('sync')
        def bypass():
            return {'ok': True}

        with self.assertRaises(MutationOutsideCoordinator) as caught:
            bypass()
        self.assertEqual(caught.exception.code, 'mutation_entrypoint_without_coordinator')

    def test_inventory_covers_every_public_mutation_family(self):
        expected = {
            'decrypt_snapshot',
            'full_import',
            'auxiliary_import',
            'reset_index_cache',
            'scope_rebuild',
            'entity_reconcile',
            'profile_enrichment',
            'profile_automation',
            'derived_data_purge',
            'sync',
            'maintain',
            'vector_index',
            'vector_rebuild',
            'content_kind_backfill',
            'appmsg_backfill',
            'message_media_backfill',
            'rebuild_chunks',
            'rebuild_fts',
            'initialize_index',
            'files_archive',
            'media_fetch',
            'media_annotate',
            'media_invalidate',
            'media_transcribe',
            'media_observe',
            'observation_write',
            'process_config_write',
            'source_manifest_write',
            'wiki_write',
            'fixture_generation',
        }
        self.assertEqual(set(VAULT_MUTATION_INVENTORY), expected)
        self.assertEqual(VAULT_MUTATION_INVENTORY['sync'].approval_policy, 'unattended')
        self.assertEqual(VAULT_MUTATION_INVENTORY['maintain'].approval_policy, 'unattended')

    def test_public_mutation_entrypoints_are_registered(self):
        entrypoints = {
            'decrypt_snapshot': [run_decrypt_plan, rollback_current],
            'full_import': [run_import_job, agent_tools.start_import],
            'auxiliary_import': [agent_tools.import_contacts, agent_tools.import_moments, agent_tools.import_favorites],
            'reset_index_cache': [reset_index_cache, agent_tools.reset_index_cache],
            'scope_rebuild': [rebuild_scope, agent_tools.scope_rebuild],
            'entity_reconcile': [agent_tools.identity_reconcile],
            'profile_enrichment': [
                agent_tools.profile_enrichment_plan,
                agent_tools.profile_enrichment_claim,
                agent_tools.profile_enrichment_redeem_media,
                agent_tools.profile_enrichment_image_annotate,
                agent_tools.profile_enrichment_heartbeat,
                agent_tools.profile_enrichment_complete,
                agent_tools.profile_enrichment_fail,
                agent_tools.profile_enrichment_resume,
                agent_tools.profile_enrichment_revoke,
                agent_tools.profile_enrichment_voice_execute,
                agent_tools.profile_enrichment_finalize,
            ],
            'profile_automation': [
                agent_tools.profile_automation_enable,
                agent_tools.profile_automation_disable,
                agent_tools.profile_automation_refresh_now,
                agent_tools.profile_automation_run_due,
            ],
            'derived_data_purge': [purge_derived_data],
            'sync': [run_sync, agent_tools.sync],
            'maintain': [run_maintain, run_media_maintenance, agent_tools.maintain],
            'vector_index': [index_vectors, agent_tools.vector_index],
            'vector_rebuild': [rebuild_vectors_atomic],
            'content_kind_backfill': [backfill_content_kind],
            'appmsg_backfill': [backfill_appmsg_payloads, agent_tools.appmsg_repair],
            'message_media_backfill': [backfill_message_media_references, agent_tools.message_media_repair],
            'rebuild_chunks': [rebuild_chunks, agent_tools.rebuild_chunks],
            'rebuild_fts': [rebuild_fts, agent_tools.rebuild_fts],
            'initialize_index': [initialize_index],
            'files_archive': [archive_files, agent_tools.files_archive],
            'media_fetch': [fetch_media, agent_tools.media_fetch],
            'media_annotate': [annotate_media_understanding, agent_tools.media_annotate],
            'media_invalidate': [invalidate_media_understanding, agent_tools.media_understanding_invalidate],
            'media_transcribe': [
                ensure_voice_transcript,
                run_voice_transcription_budget,
                agent_tools.voice_transcribe_fixture,
                agent_tools.voice_transcribe_lazy,
                agent_tools.media_transcribe_budget,
            ],
            'media_observe': [
                run_image_caption_budget,
                run_image_observation_budget,
                agent_tools.image_observe_fixture,
                agent_tools.media_observe_budget,
            ],
            'observation_write': [
                agent_tools.observe_add,
                agent_tools.observe_propose,
                agent_tools.observe_retire,
                agent_tools.observe_approve,
            ],
            'process_config_write': [write_process_config],
            'source_manifest_write': [agent_tools.source_manifest],
            'wiki_write': [write_wiki_page],
            'fixture_generation': [index_fixture_vault, index_fixture_data],
        }
        self.assertEqual(set(entrypoints), set(VAULT_MUTATION_INVENTORY))
        for operation, functions in entrypoints.items():
            with self.subTest(operation=operation):
                self.assertTrue(functions)
                self.assertEqual({mutation_operation(function) for function in functions}, {operation})

    def test_import_lock_conflicts_are_typed_for_other_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            with VaultOperationCoordinator(cfg).write(owner='import'):
                sync = run_sync(vault, options=SyncOptions(snapshot_dir=vault / 'missing'))
                self.assertEqual(sync['status'], 'locked')
                self.assertEqual(sync['errors'], ['VaultOperationLocked'])

                maintain = run_maintain(vault, options=MaintainOptions(media_voice_budget=0))
                self.assertEqual(maintain['status'], 'locked')
                self.assertEqual(maintain['errors'], ['VaultOperationLocked'])

                imported = run_import_job(vault, [])
                self.assertEqual(imported.status, 'locked')
                self.assertEqual(imported.errors, ['VaultOperationLocked'])

                with self.assertRaises(VaultOperationLocked):
                    index_vectors(cfg, _UnusedEmbeddingProvider(), backend='sqlite')
                with self.assertRaises(VaultOperationLocked):
                    reset_index_cache(vault)
                with self.assertRaises(VaultOperationLocked):
                    agent_tools.scope_rebuild(vault, yes=True)

    def test_unattended_sync_and_maintain_never_request_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            with patch('trove_core.approvals.ApprovalManager.require', side_effect=AssertionError('approval requested')):
                sync = run_sync(vault, options=SyncOptions(snapshot_dir=vault / 'missing'))
                self.assertEqual(sync['status'], 'no_snapshot')
                maintain = run_maintain(vault, options=MaintainOptions(media_voice_budget=0))
                self.assertTrue(maintain['ok'])

    def test_read_entrypoints_do_not_acquire_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            initialize_index(vault)
            with patch(
                'trove_core.vault.mutations.VaultOperationCoordinator.acquire',
                side_effect=AssertionError('read acquired writer'),
            ) as acquire:
                agent_tools.vault_status(vault)
                agent_tools.scope_status(vault)
                agent_tools.list_contacts(vault)
                agent_tools.list_moments(vault)
                agent_tools.list_favorites(vault)
                agent_tools.files_list(vault, limit=1)
                agent_tools.import_status(vault)
                agent_tools.media_understanding_status_tool(vault)
                voice_transcription_plan(vault)
            acquire.assert_not_called()

    def test_agent_query_helpers_validate_readonly_and_close_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            initialized_modes: list[bool] = []
            original_initialize = SQLiteStore.initialize

            def audited_initialize(store: SQLiteStore) -> None:
                initialized_modes.append(store.readonly)
                original_initialize(store)

            with patch.object(SQLiteStore, 'initialize', audited_initialize):
                agent_tools.provider_jobs(vault, limit=1)
                agent_tools.scope_status(vault)
                agent_tools.list_contacts(vault, limit=1)
                agent_tools.list_moments(vault, limit=1)
                agent_tools.list_favorites(vault, limit=1)
                agent_tools.files_list(vault, contact='示例教育', limit=1)
                agent_tools.customer_profile(vault, '示例教育', limit=1)
                agent_tools.fetch_context(
                    vault,
                    'trove://wechat/acct-work/conv-northstar/message_0/1',
                    before=0,
                    after=0,
                )

            self.assertTrue(initialized_modes)
            self.assertTrue(all(initialized_modes))

    def test_agent_query_on_missing_vault_never_creates_or_migrates_it(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'missing-vault'
            profile = agent_tools.customer_profile(vault, 'synthetic customer', limit=1)
            jobs = agent_tools.provider_jobs(vault, limit=1)
            files = agent_tools.files_list(vault, contact='synthetic customer', limit=1)
            self.assertIsNone(profile['resolved_entity'])
            self.assertEqual(jobs['jobs'], [])
            self.assertEqual(files['files'], [])
            self.assertFalse(vault.exists())

    def test_explicit_session_reuses_parent_without_second_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            with VaultOperationCoordinator(cfg).write(owner='import') as session:
                with patch(
                    'trove_core.vault.mutations.VaultOperationCoordinator.acquire',
                    side_effect=AssertionError('nested mutation reacquired writer'),
                ) as acquire:
                    path = write_process_config(
                        vault,
                        process_config_from_payload({'config_id': 'pcfg-reentrant'}),
                        write_session=session,
                    )
                    self.assertTrue(path.exists())
                acquire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
