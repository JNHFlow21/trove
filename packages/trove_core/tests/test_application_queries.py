from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from trove_core.approvals import ApprovalRequired
from trove_core.application.queries import (
    ContextQuery,
    FilesQuery,
    ListQuery,
    QueryInputError,
    SearchQuery,
    TroveQueries,
    validation_error_payload,
)
from trove_core.bounds import BoundedInputError
from trove_core.runtime import SearchRuntimeCache
from trove_core.search.query import SearchResponse
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import VaultGenerationUnavailable, vault_generation_publish
from trove_core.wechat.indexer import index_fixture_vault
from trove_core.wechat.models import Evidence
from trove_core.wechat.process_config import process_config_from_payload, write_process_config


class _Runtime:
    def __init__(self, response: SearchResponse):
        self.response = response
        self.requests = []

    def search_with_metrics(self, request):
        self.requests.append(request)
        return self.response, {'cache_hit': True, 'candidate_count': len(self.response.results), 'duration_ms': 7}


class _BlockingRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__(SearchResponse('fixture', [], 0, {'vector': {}}, 1.0))
        self.entered = threading.Event()
        self.release = threading.Event()

    def search_with_metrics(self, request):
        self.entered.set()
        if not self.release.wait(3.0):
            raise TimeoutError('test search release was not signalled')
        return super().search_with_metrics(request)


class ApplicationQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / 'vault'
        index_fixture_vault(self.root, reset=True)
        self.queries = TroveQueries(VaultConfig.resolve(str(self.root)))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_search_resolves_contact_and_owns_search_contract(self) -> None:
        result = self.queries.search(SearchQuery(
            '预算审批',
            contact='示例教育',
            since='2026-06-20T09:00:00Z',
            until='2026-06-20T09:20:00Z',
            semantic='off',
        ))
        payload = result.to_dict()
        self.assertTrue(payload['ok'])
        self.assertTrue(payload['results'])
        filters = payload['retrieval_status']['metadata_filters']
        self.assertEqual(filters['account_id'], 'acct-work')
        self.assertEqual(filters['conversation_id'], 'conv-example_edu-private')

    def test_search_and_context_share_contact_time_and_no_result_errors(self) -> None:
        search = self.queries.search(SearchQuery('预算审批', contact='示例教育', semantic='off'))
        citation = search.to_dict()['results'][0]['citation']
        context = self.queries.context(ContextQuery(
            citation,
            contact='示例教育',
            since='2026-06-20T09:00:00Z',
            until='2026-06-20T09:20:00Z',
            before=2,
            after=2,
        )).to_dict()
        self.assertTrue(context['ok'])
        self.assertTrue(context['messages'])

        rejected = self.queries.context(ContextQuery(citation, conversation_id='wrong')).to_dict()
        self.assertEqual(rejected['code'], 'no_results')
        self.assertFalse(rejected['ok'])

    def test_lists_and_files_use_typed_result_dto(self) -> None:
        self.assertIn('contacts', self.queries.list_contacts(ListQuery(limit=2)).to_dict())
        self.assertIn('moments', self.queries.list_moments(ListQuery(limit=2)).to_dict())
        self.assertIn('favorites', self.queries.list_favorites(ListQuery(limit=2)).to_dict())
        self.assertIn('conversations', self.queries.list_conversations(ListQuery(limit=2)).to_dict())
        files = self.queries.list_files(FilesQuery(contact='示例教育', limit=2)).to_dict()
        self.assertTrue(files['ok'])
        self.assertLessEqual(files['count'], 2)

    def test_ambiguous_and_missing_contacts_are_typed(self) -> None:
        # The fixture deliberately has no exact conversation for this value.
        missing = self.queries.search(SearchQuery('x', contact='does-not-exist', semantic='off')).to_dict()
        self.assertEqual(missing['code'], 'no_results')
        self.assertEqual(missing['error']['candidates'], [])

    def test_bounds_and_time_errors_have_one_protocol_payload(self) -> None:
        for constructor, code, field in (
            (lambda: SearchQuery('x', limit='3'), 'invalid_limit', 'limit'),
            (lambda: ContextQuery('x', before=201), 'invalid_limit', 'before'),
            (lambda: SearchQuery('x', since='not-a-date'), 'invalid_timestamp', 'since'),
            (
                lambda: SearchQuery('x', since='2026-02-01T00:00:00Z', until='2026-01-01T00:00:00Z'),
                'invalid_time_range',
                'since',
            ),
        ):
            with self.subTest(code=code, field=field):
                with self.assertRaises((BoundedInputError, QueryInputError)) as raised:
                    constructor()
                payload = validation_error_payload(raised.exception)
                self.assertEqual(payload['code'], code)
                self.assertEqual(payload['error']['field'], field)

    def test_runtime_metrics_survive_application_boundary(self) -> None:
        response = SearchResponse('fixture', [], 0, {'vector': {}}, 1.0)
        runtime = _Runtime(response)
        def forbidden_uow(*_args, **_kwargs):
            raise AssertionError('ordinary runtime search opened a second repository store')

        result = TroveQueries(
            self.root,
            runtime=runtime,
            uow_factory=forbidden_uow,
        ).search(SearchQuery('fixture'))
        self.assertEqual(result.code, 'no_results')
        self.assertEqual(len(runtime.requests), 1)

    def test_cloud_rerank_requires_exact_approval_before_provider_construction(self) -> None:
        cfg = SimpleNamespace(
            cloud_rerank_top_k=20,
            cloud_rerank_provider='aliyun',
            cloud_rerank_model='qwen3-rerank',
            cloud_rerank_endpoint='https://example.invalid/reranks',
        )
        factory = Mock()
        factory.readiness.return_value = SimpleNamespace(ready=True, reason_code=None)
        response = SearchResponse('fixture', [
            Evidence(
                f'trove://fixture/{index}', 'acct', 'fixture', 'conv', 'fixture conversation',
                'private', 'fixture sender', '2026-01-01T00:00:00Z', f'fixture snippet {index}',
                'message', 'incoming', float(index), ['fts'], f'trove://fixture/{index}',
            )
            for index in range(3)
        ], 3, {'vector': {}, 'phase_latency_ms': {}}, 1.0)
        queries = TroveQueries(self.root, runtime=_Runtime(response))
        with (
            patch('trove_core.providers.config.ProviderConfig.resolve', return_value=cfg),
            patch('trove_core.providers.factory.ProviderFactory.resolve', return_value=factory),
            self.assertRaises(ApprovalRequired),
        ):
            queries.search(SearchQuery(
                '预算审批',
                reranker_mode='cloud-qwen3',
                allow_cloud_rerank=True,
                semantic='off',
            ))
        factory.create_cloud_reranker.assert_not_called()

    def test_one_step_cloud_rerank_reorders_bounded_results_and_records_status(self) -> None:
        cfg = SimpleNamespace(
            cloud_rerank_top_k=20,
            cloud_rerank_provider='aliyun',
            cloud_rerank_model='qwen3-rerank',
            cloud_rerank_endpoint='https://example.invalid/reranks',
        )
        provider = SimpleNamespace(
            provider_name='aliyun',
            model='qwen3-rerank',
            endpoint='https://example.invalid/reranks',
        )
        factory = Mock()
        factory.readiness.return_value = SimpleNamespace(ready=True, reason_code=None)
        factory.create_cloud_reranker.return_value = provider
        response = SearchResponse('fixture', [
            Evidence(
                f'trove://fixture/{index}', 'acct', 'fixture', 'conv', 'fixture conversation',
                'private', 'fixture sender', '2026-01-01T00:00:00Z', f'fixture snippet {index}',
                'message', 'incoming', float(index), ['fts'], f'trove://fixture/{index}',
            )
            for index in range(3)
        ], 3, {'vector': {}, 'phase_latency_ms': {}}, 1.0)
        queries = TroveQueries(self.root, runtime=_Runtime(response))

        def execute(_vault, *, documents, **_kwargs):
            return {
                'ok': True,
                'results': [
                    {'index': index, 'score': float(index)}
                    for index in reversed(range(len(documents)))
                ],
            }

        with (
            patch('trove_core.providers.config.ProviderConfig.resolve', return_value=cfg),
            patch('trove_core.providers.factory.ProviderFactory.resolve', return_value=factory),
            patch('trove_core.application.cloud_commands.execute_cloud_rerank', side_effect=execute),
        ):
            result = queries.search(SearchQuery(
                '预算审批',
                limit=2,
                reranker_mode='cloud-qwen3',
                allow_cloud_rerank=True,
                cloud_rerank_one_step_approval=True,
                semantic='off',
            )).to_dict()

        status = result['retrieval_status']['reranker']
        self.assertTrue(result['ok'])
        self.assertEqual(status['state'], 'available')
        self.assertEqual(status['mode'], 'cloud-qwen3')
        self.assertTrue(status['invoked'])
        self.assertEqual(len(result['results']), 2)
        self.assertNotIn('documents', str(status))
        factory.create_cloud_reranker.assert_called_once()

    def test_cloud_rerank_uses_episode_bundles_as_complete_evidence_candidates(self) -> None:
        cfg = SimpleNamespace(
            cloud_rerank_top_k=20,
            cloud_rerank_provider='aliyun',
            cloud_rerank_model='qwen3-rerank',
            cloud_rerank_endpoint='https://example.invalid/reranks',
        )
        provider = SimpleNamespace(
            provider_name='aliyun', model='qwen3-rerank', endpoint='https://example.invalid/reranks',
        )
        factory = Mock()
        factory.readiness.return_value = SimpleNamespace(ready=True, reason_code=None)
        factory.create_cloud_reranker.return_value = provider
        messages = [
            Evidence(
                f'trove://fixture/{index}', 'acct', 'fixture', 'conv', 'fixture conversation',
                'private', 'fixture sender', '2026-01-01T00:00:00Z', f'fixture snippet {index}',
                'message', 'incoming', float(index), ['fts'], f'trove://fixture/{index}',
            )
            for index in range(3)
        ]
        bundle = Evidence(
            'trove://episode/representative', 'acct', 'fixture', 'conv', 'fixture conversation',
            'private', 'fixture sender', '2026-01-01T00:00:00Z', 'bounded episode evidence',
            'message', 'incoming', 1.0, ['episode-bundle'], 'trove://episode/representative',
            supporting_citations=('support-a', 'support-b'), evidence_kind='episode',
        )
        alternate_bundle = Evidence(
            'trove://episode/alternate', 'acct', 'fixture', 'conv', 'fixture conversation',
            'private', 'fixture sender', '2026-01-01T00:00:00Z', 'alternate bounded episode',
            'message', 'incoming', 0.5, ['episode-bundle'], 'trove://episode/alternate',
            supporting_citations=('support-c',), evidence_kind='episode',
        )
        response = SearchResponse(
            'fixture', messages, 3, {'vector': {}, 'phase_latency_ms': {}}, 1.0,
            episode_bundles=(bundle, alternate_bundle),
        )
        queries = TroveQueries(self.root, runtime=_Runtime(response))
        captured = {}

        def execute(_vault, *, documents, **_kwargs):
            captured['documents'] = documents
            return {
                'ok': True,
                'results': [{'index': index, 'score': 1.0 - index / 10} for index in range(len(documents))],
            }

        with (
            patch('trove_core.providers.config.ProviderConfig.resolve', return_value=cfg),
            patch('trove_core.providers.factory.ProviderFactory.resolve', return_value=factory),
            patch('trove_core.application.cloud_commands.execute_cloud_rerank', side_effect=execute),
        ):
            result = queries.search(SearchQuery(
                '关联脉络', limit=2, reranker_mode='cloud-qwen3', allow_cloud_rerank=True,
                cloud_rerank_one_step_approval=True, semantic='off',
            )).to_dict()

        self.assertEqual(result['results'][0]['evidence_kind'], 'episode')
        self.assertEqual(tuple(result['results'][0]['supporting_citations']), ('support-a', 'support-b'))
        self.assertEqual(result['retrieval_status']['reranker']['episode_bundle_candidates'], 2)
        self.assertEqual(result['retrieval_status']['reranker']['episode_bundle_results'], 2)
        self.assertIn('bounded episode evidence', captured['documents'][0])
        self.assertIn('instruct', factory.create_cloud_reranker.call_args.kwargs)

    def test_vault_cloud_policy_makes_bounded_rerank_default_without_exact_approval(self) -> None:
        write_process_config(self.root, process_config_from_payload({
            'config_id': 'pcfg-query-cloud-policy',
            'cloud_retrieval': 'enabled',
        }))
        cfg = SimpleNamespace(
            cloud_rerank_top_k=20,
            cloud_rerank_provider='aliyun',
            cloud_rerank_model='qwen3-rerank',
            cloud_rerank_endpoint='https://example.invalid/reranks',
        )
        provider = SimpleNamespace(
            provider_name='aliyun', model='qwen3-rerank', endpoint='https://example.invalid/reranks',
        )
        factory = Mock()
        factory.readiness.return_value = SimpleNamespace(ready=True, reason_code=None)
        factory.create_cloud_reranker.return_value = provider
        response = SearchResponse('fixture', [
            Evidence(
                f'trove://fixture/{index}', 'acct', 'fixture', 'conv', 'fixture conversation',
                'private', 'fixture sender', '2026-01-01T00:00:00Z', f'fixture snippet {index}',
                'message', 'incoming', float(index), ['fts'], f'trove://fixture/{index}',
            )
            for index in range(3)
        ], 3, {'vector': {}, 'phase_latency_ms': {}}, 1.0)
        runtime = _Runtime(response)
        queries = TroveQueries(self.root, runtime=runtime)

        with (
            patch('trove_core.providers.config.ProviderConfig.resolve', return_value=cfg),
            patch('trove_core.providers.factory.ProviderFactory.resolve', return_value=factory),
            patch('trove_core.application.cloud_commands.execute_policy_cloud_rerank', return_value={
                'ok': True,
                'results': [{'index': 2, 'score': 1.0}, {'index': 1, 'score': 0.5}],
                'usage': {'input_tokens': 10, 'estimated_cost_usd': 0.000001},
            }),
        ):
            result = queries.search(SearchQuery('预算审批', limit=2, semantic='off')).to_dict()

        status = result['retrieval_status']['reranker']
        self.assertEqual(status['state'], 'available')
        self.assertEqual(status['authorization'], 'vault-continuous-retrieval-v1')
        self.assertIsNone(status['approval_id'])
        self.assertEqual(runtime.requests[0].reranker_mode, 'features')

    def test_vault_cloud_policy_reuses_successful_rerank_within_generation(self) -> None:
        write_process_config(self.root, process_config_from_payload({
            'config_id': 'pcfg-query-cloud-rerank-cache',
            'cloud_retrieval': 'enabled',
        }))
        cfg = SimpleNamespace(
            cloud_rerank_top_k=20,
            cloud_rerank_provider='aliyun',
            cloud_rerank_model='qwen3-rerank',
            cloud_rerank_endpoint='https://example.invalid/reranks',
        )
        provider = SimpleNamespace(
            provider_name='aliyun', model='qwen3-rerank', endpoint='https://example.invalid/reranks',
        )
        factory = Mock()
        factory.readiness.return_value = SimpleNamespace(ready=True, reason_code=None)
        factory.create_cloud_reranker.return_value = provider
        runtime = SearchRuntimeCache(
            VaultConfig.resolve(str(self.root)), provider_factory=lambda: None
        )
        queries = TroveQueries(self.root, runtime=runtime)

        try:
            with (
                patch('trove_core.providers.config.ProviderConfig.resolve', return_value=cfg),
                patch('trove_core.providers.factory.ProviderFactory.resolve', return_value=factory),
                patch(
                    'trove_core.application.cloud_commands.execute_policy_cloud_rerank',
                    return_value={
                        'ok': True,
                        'results': [{'index': 1, 'score': 1.0}, {'index': 0, 'score': 0.5}],
                        'usage': {'input_tokens': 10, 'estimated_cost_usd': 0.000001},
                    },
                ) as execute,
            ):
                first = queries.search(SearchQuery('预算审批', limit=2, semantic='off'))
                second = queries.search(SearchQuery('预算审批', limit=2, semantic='off'))

            self.assertTrue(first.ok)
            self.assertTrue(second.ok)
            self.assertEqual(execute.call_count, 1)
            self.assertFalse(first.metrics['cloud_rerank_cache_hit'])
            self.assertTrue(second.metrics['cloud_rerank_cache_hit'])
            self.assertEqual(runtime.status()['memo_cache_entries'], 2)
        finally:
            runtime.close()

    def test_recovery_marker_blocks_every_public_query_before_sqlite_is_opened(self) -> None:
        marker = self.root / '.trove-generation-publish.json'
        marker.write_text(json.dumps({
            'format': 'trove-generation-publish',
            'version': 1,
            'operation': 'query-test',
            'nonce': '1' * 32,
        }, sort_keys=True, separators=(',', ':')) + '\n', encoding='ascii')
        marker.chmod(0o600)
        with self.assertRaises(VaultGenerationUnavailable) as blocked:
            self.queries.evidence('trove://fixture/missing')
        self.assertEqual(blocked.exception.code, 'vault_generation_recovery_required')

    @unittest.skipUnless(hasattr(os, 'fork'), 'requires cross-process flock semantics')
    def test_exclusive_publish_waits_for_complete_application_search(self) -> None:
        runtime = _BlockingRuntime()
        queries = TroveQueries(self.root, runtime=runtime)
        go_read, go_write = os.pipe()
        attempted_read, attempted_write = os.pipe()
        done_read, done_write = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child process
            try:
                os.close(go_write)
                os.close(attempted_read)
                os.close(done_read)
                if os.read(go_read, 1) != b'1':
                    os._exit(81)
                os.write(attempted_write, b'1')
                with vault_generation_publish(self.queries.config, operation='query-test'):
                    pass
                os.write(done_write, b'1')
                os._exit(0)
            except BaseException:
                os._exit(82)

        os.close(go_read)
        os.close(attempted_write)
        os.close(done_write)
        errors: list[BaseException] = []

        def run_search() -> None:
            try:
                queries.search(SearchQuery('fixture', semantic='off'))
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        worker = threading.Thread(target=run_search, daemon=True)
        worker.start()
        try:
            self.assertTrue(runtime.entered.wait(2.0))
            os.write(go_write, b'1')
            self.assertEqual(os.read(attempted_read, 1), b'1')
            ready, _, _ = select.select([done_read], [], [], 0.15)
            self.assertEqual(ready, [], 'publisher crossed an in-flight application query')
            runtime.release.set()
            worker.join(3.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            ready, _, _ = select.select([done_read], [], [], 3.0)
            self.assertEqual(ready, [done_read])
            self.assertEqual(os.read(done_read, 1), b'1')
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        finally:
            runtime.release.set()
            for fd in (go_write, attempted_read, done_read):
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                waited, _ = os.waitpid(pid, os.WNOHANG)
                if waited == 0:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
            except (ChildProcessError, ProcessLookupError):
                pass


if __name__ == '__main__':
    unittest.main()
