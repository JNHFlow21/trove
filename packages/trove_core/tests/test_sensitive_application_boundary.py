from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trove_core.application.inventory import (
    APPLICATION_COMMAND_INVENTORY,
    validate_application_command_inventory,
)
from trove_core.application.cloud_commands import (
    execute_cloud_image_observation,
    execute_cloud_rerank,
)
from trove_core.application.sensitive_commands import (
    execute_full_import,
    execute_reset_index_cache,
    full_import_payload,
    reset_index_cache_payload,
)
from trove_core.approvals import (
    ApprovalManager,
    ApprovalValidationError,
    SENSITIVE_CAPABILITY_INVENTORY,
)
from trove_core.vault.config import VaultConfig
from trove_core.security.egress import cloud_rerank_payload, cloud_vision_payload
from trove_core.wechat.files import archive_approval_payload, archive_files


class SensitiveApplicationBoundaryTests(unittest.TestCase):
    @staticmethod
    def _grant(vault: Path, action: str, danger_class: str, payload: dict):
        manager = ApprovalManager(vault)
        record = manager.request(action, danger_class, payload)
        manager.decide(record.approval_id, 'approved')
        return manager.require(action, danger_class, payload, approval_id=record.approval_id)

    def test_inventory_has_one_auditable_boundary_for_every_sensitive_action(self):
        validate_application_command_inventory()
        self.assertEqual(set(APPLICATION_COMMAND_INVENTORY), set(SENSITIVE_CAPABILITY_INVENTORY))

    def test_reset_claims_exact_grant_before_sink_and_replay_has_zero_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            grant = self._grant(
                vault,
                'reset_index_cache',
                'destructive_rebuild',
                reset_index_cache_payload(),
            )
            with mock.patch(
                'trove_core.application.sensitive_commands.reset_index_cache',
                return_value={'removed': 3},
            ) as sink:
                first = execute_reset_index_cache(vault, approval_grant=grant)
                self.assertEqual(first['removed'], 3)
                with self.assertRaises(ApprovalValidationError) as replayed:
                    execute_reset_index_cache(vault, approval_grant=grant)
            self.assertEqual(replayed.exception.code, 'approval_grant_replayed')
            sink.assert_called_once_with(vault)

    def test_full_import_cross_payload_is_blocked_before_import_sink(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            approved_sources = [Path(directory) / 'approved-source']
            payload = full_import_payload(
                approved_sources,
                reset_index_cache=False,
                limit_per_sqlite=None,
                process_config=None,
            )
            grant = self._grant(vault, 'full_import', 'full_import', payload)
            with mock.patch('trove_core.application.sensitive_commands.run_import_job') as sink:
                with self.assertRaises(ApprovalValidationError) as mismatch:
                    execute_full_import(
                        vault,
                        [Path(directory) / 'different-source'],
                        reset_index_cache=False,
                        limit_per_sqlite=None,
                        process_config=None,
                        approval_grant=grant,
                    )
            self.assertEqual(mismatch.exception.code, 'grant_payload_mismatch')
            sink.assert_not_called()

    def test_low_level_file_export_rejects_consumed_but_unclaimed_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            selection = {'asset_ids': ['fixture-asset']}
            dest = Path(directory) / 'export'
            payload = archive_approval_payload(cfg, selection=selection, dest_dir=dest)
            grant = self._grant(vault, 'files_archive', 'local-file-export', payload)
            with self.assertRaises(ApprovalValidationError) as unclaimed:
                archive_files(
                    cfg,
                    selection=selection,
                    dest_dir=dest,
                    approval_grant=grant,
                    approval_payload=payload,
                )
            self.assertEqual(unclaimed.exception.code, 'approval_grant_unclaimed')
            self.assertFalse(dest.exists())

    def test_cloud_vision_application_claims_before_provider_job_and_replay(self):
        class Provider:
            egress_kind = 'cloud_vision_upload'
            name = 'fixture-cloud-vision'
            model = 'fixture-model'
            endpoint = 'https://example.invalid/vision'

        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            image = Path(directory) / 'image.jpg'
            image.write_bytes(b'fixture-image')
            payload = cloud_vision_payload(
                citation='trove://fixture/image',
                provider=Provider.name,
                model=Provider.model,
                endpoint=Provider.endpoint,
            )
            grant = self._grant(vault, 'image_cloud_vision', 'cloud_vision_upload', payload)
            with mock.patch(
                'trove_core.application.cloud_commands.run_image_observation_job',
                return_value={'status': 'completed'},
            ) as sink:
                first = execute_cloud_image_observation(
                    vault,
                    asset_id='asset-1',
                    image_path=image,
                    citation='trove://fixture/image',
                    provider=Provider(),
                    approval_grant=grant,
                )
                self.assertEqual(first['status'], 'completed')
                with self.assertRaises(ApprovalValidationError) as replayed:
                    execute_cloud_image_observation(
                        vault,
                        asset_id='asset-1',
                        image_path=image,
                        citation='trove://fixture/image',
                        provider=Provider(),
                        approval_grant=grant,
                    )
            self.assertEqual(replayed.exception.code, 'approval_grant_replayed')
            self.assertEqual(sink.call_count, 1)

    def test_cloud_rerank_application_claims_before_transport_and_replay(self):
        class Provider:
            egress_kind = 'cloud_rerank_upload'
            provider_name = 'fixture-cloud-rerank'
            model = 'fixture-model'
            endpoint = 'https://example.invalid/rerank'

            def __init__(self):
                self.calls = 0

            def rerank(self, _query, _documents, *, top_n, **_kwargs):
                self.calls += 1
                return [(0, 1.0)][:top_n]

        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            provider = Provider()
            documents = ['synthetic document']
            payload = cloud_rerank_payload(
                query='synthetic query',
                documents=documents,
                top_n=1,
                provider=provider.provider_name,
                model=provider.model,
                endpoint=provider.endpoint,
            )
            grant = self._grant(vault, 'cloud_rerank', 'cloud_rerank_upload', payload)
            result = execute_cloud_rerank(
                vault,
                query='synthetic query',
                documents=documents,
                top_n=1,
                provider=provider,
                approval_grant=grant,
            )
            self.assertTrue(result['ok'])
            self.assertIn('usage', result)
            self.assertEqual(result['usage']['output_tokens_billed'], 0)
            with self.assertRaises(ApprovalValidationError) as replayed:
                execute_cloud_rerank(
                    vault,
                    query='synthetic query',
                    documents=documents,
                    top_n=1,
                    provider=provider,
                    approval_grant=grant,
                )
            self.assertEqual(replayed.exception.code, 'approval_grant_replayed')
            self.assertEqual(provider.calls, 1)

    def test_cloud_rerank_bound_rejects_before_approval_consumption(self):
        from trove_core.bounds import BoundedInputError

        class Provider:
            egress_kind = 'cloud_rerank_upload'
            provider_name = 'fixture-cloud-rerank'
            model = 'fixture-model'
            endpoint = 'https://example.invalid/rerank'

            def rerank(self, *_args, **_kwargs):
                raise AssertionError('transport touched')

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'trove_core.application.cloud_commands.claim_approval_grant',
            side_effect=AssertionError('approval consumed'),
        ) as claim:
            with self.assertRaises(BoundedInputError) as raised:
                execute_cloud_rerank(
                    Path(directory) / 'vault',
                    query='synthetic query',
                    documents=['synthetic'] * 201,
                    top_n=201,
                    provider=Provider(),
                    approval_grant=None,  # type: ignore[arg-type]
                )
        self.assertEqual(raised.exception.code, 'invalid_limit')
        self.assertEqual(raised.exception.field, 'reranker_candidate_limit')
        claim.assert_not_called()

    def test_adapters_do_not_call_low_level_sensitive_sinks(self):
        repo = Path(__file__).resolve().parents[3]
        packages = repo / 'packages'
        adapters = [
            packages / 'trove_cli' / 'trove_cli' / 'v1_main.py',
            packages / 'trove_mcp' / 'trove_mcp' / 'v1_server.py',
            packages / 'trove_core' / 'trove_core' / 'agent_tools' / 'tools.py',
        ]
        prohibited = {
            'run_import_job',
            'purge_excluded_scope',
            'purge_derived_data',
            'archive_files',
            'invalidate_media_understanding',
            'set_observation_status',
            'rebuild_vectors_atomic',
            'backfill_message_content_kind',
        }
        violations: list[str] = []
        for path in adapters:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in prohibited:
                            violations.append(f'{path.name}:{node.lineno}:import:{alias.name}')
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name in prohibited:
                        violations.append(f'{path.name}:{node.lineno}:call:{name}')
                    if name == 'index_vectors':
                        purge = next((kw.value for kw in node.keywords if kw.arg == 'purge'), None)
                        if isinstance(purge, ast.Constant) and purge.value is True:
                            violations.append(f'{path.name}:{node.lineno}:call:index_vectors(purge=True)')
        self.assertEqual(violations, [])


if __name__ == '__main__':
    unittest.main()
