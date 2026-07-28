from __future__ import annotations

import copy
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts import benchmark_u10_scaling
from scripts.build_release_closeout_manifest import (
    CUTOVER_CHECK_IDS,
    CUTOVER_COUNT_KEYS,
    CloseoutManifestError,
    V1_RELEASE_PRIVACY,
    build_cutover_acceptance,
    build_closeout_manifest,
    build_v1_release_manifest,
    main as closeout_main,
    validate_v1_release_manifest,
)
from scripts.release_gate_contracts import (
    CONTRACT_SCHEMA_VERSION,
    RELEASE_ASSURANCE,
    U15_GATE_CONTRACTS,
    U15_REQUIRED_GATE_IDS,
    release_gate_contract_sha256,
    release_soak_contract_valid,
)
from scripts.run_release_gate import main as release_gate_main, run_release_gate
from scripts.run_release_soak import (
    DOGFOOD_PRIVACY,
    DOGFOOD_REQUIRED_SECONDS,
    DOGFOOD_SKILLS,
    _open_fd_count,
    _resource_summary,
    classify_soak_error,
    finalize_dogfood_record,
    record_dogfood_client,
    run_release_soak,
    run_release_soak_isolated,
    validate_dogfood_record,
)
from trove_core.runtime import RuntimeOverloaded, RuntimeTimedOut
from trove_core.search.evidence_provenance import stable_payload_sha256, verify_evidence_manifest, write_evidence_artifact
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.generation import VaultGenerationLease


_SHA_A = 'a' * 40
_SHA_B = 'b' * 40
_PRIVACY = {
    'synthetic_or_redacted': True,
    'redacted': True,
    'sensitive_content_included': False,
    'private_paths_included': False,
    'raw_command_output_included': False,
    'credential_values_included': False,
}


def _dogfood_start(started: datetime) -> dict:
    return {
        'schema_version': 1,
        'artifact_type': 'trove_v1_dogfood_start_redacted',
        'ok': True,
        'subject_sha': _SHA_A,
        'distribution_set_sha256': 'd' * 64,
        'started_at_utc': started.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        'required_duration_seconds': DOGFOOD_REQUIRED_SECONDS,
        'required_clients': 2,
        'required_skills': list(DOGFOOD_SKILLS),
        'privacy': dict(DOGFOOD_PRIVACY),
    }


def _dogfood_metrics() -> dict:
    skills = {}
    for skill in DOGFOOD_SKILLS:
        calls = 2 if skill == 'trove-search' else 1
        skills[skill] = {
            'tasks': 1,
            'tasks_succeeded': 1,
            'cited_outcomes': 1,
            'calls': calls,
            'operator_repairs': 0,
        }
    return {
        'sessions': 3,
        'active_days': 3,
        'source_free_install': True,
        'mcp_connected': True,
        'recall_succeeded': True,
        'provider_status_succeeded': True,
        'skills': skills,
    }


def _stuck_isolated_worker(_connection, _options) -> None:
    time.sleep(30.0)


class DogfoodGateTests(unittest.TestCase):
    def test_same_candidate_requires_three_elapsed_days_and_two_distinct_clients(self):
        started = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
        start = _dogfood_start(started)
        first = record_dogfood_client(
            start, 'client-one', _dogfood_metrics(), now=started + timedelta(days=2),
        )
        second = record_dogfood_client(
            start, 'client-two', _dogfood_metrics(), now=started + timedelta(days=2),
        )
        with self.assertRaisesRegex(ValueError, 'three-day'):
            finalize_dogfood_record(
                start, [first, second], now=started + timedelta(seconds=DOGFOOD_REQUIRED_SECONDS - 1),
            )
        with self.assertRaisesRegex(ValueError, 'bound to the candidate'):
            finalize_dogfood_record(
                start, [first, first], now=started + timedelta(days=3),
            )

        record = finalize_dogfood_record(
            start, [second, first], now=started + timedelta(days=3),
        )
        validate_dogfood_record(record)
        self.assertTrue(record['ok'])
        self.assertEqual(record['duration_seconds'], DOGFOOD_REQUIRED_SECONDS)
        self.assertEqual(record['summary']['clients'], 2)
        self.assertEqual(record['summary']['skills_per_client'], 6)
        self.assertEqual(record['summary']['operator_repairs'], 0)
        self.assertTrue(record['summary']['ordinary_call_budget_passed'])
        serialized = json.dumps(record, sort_keys=True)
        self.assertNotIn('client-one', serialized)
        self.assertNotIn('client-two', serialized)

    def test_skill_failure_operator_repair_and_call_budget_fail_closed(self):
        started = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
        start = _dogfood_start(started)
        for skill, field, value, error in (
            ('trove-profile', 'operator_repairs', 1, 'outcome'),
            ('trove-recall', 'calls', 2, 'recall call budget'),
            ('trove-search', 'calls', 3, 'search call budget'),
        ):
            with self.subTest(skill=skill, field=field):
                metrics = _dogfood_metrics()
                metrics['skills'][skill][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    record_dogfood_client(start, skill, metrics, now=started + timedelta(days=3))


def _receipt_artifact(
    commit_sha: str = _SHA_A,
    *,
    dirty: bool = False,
    privacy: dict | None = None,
    command_id: str = 'synthetic-check-v1',
    ok: bool = True,
    failures: int = 0,
    gate_id: str = 'synthetic-check',
    profile: str = 'custom',
) -> dict:
    return {
        'schema_version': 1,
        'artifact_type': 'synthetic_check_redacted',
        'ok': ok,
        'release_receipt': {
            'schema_version': 1,
            'subject': {'commit_sha': commit_sha, 'dirty': dirty},
            'fixture': {'kind': 'synthetic_or_redacted'},
            'privacy': dict(_PRIVACY if privacy is None else privacy),
            'counts': {'checks_completed': 1, 'failures': failures},
            'platform': {
                'system': 'TestOS',
                'release': '1.0',
                'machine': 'test64',
                'python_version': '3.14.0',
            },
            'version': '1.0.0',
            'command_id': command_id,
            'gate_id': gate_id,
            'profile': profile,
        },
        'privacy': dict(_PRIVACY if privacy is None else privacy),
    }


def _standard_provenance_artifact(commit_sha: str = _SHA_A) -> dict:
    digest = stable_payload_sha256('synthetic')
    return {
        'schema_version': 2,
        'artifact_type': 'search_benchmark_redacted',
        'provenance': {
            'schema_version': 1,
            'git': {'commit_sha': commit_sha, 'dirty': False},
            'platform': {
                'system': 'TestOS',
                'release': '1.0',
                'machine': 'test64',
                'python_implementation': 'CPython',
                'python_version': '3.14.0',
                'cpu_count': 4,
                'processor_sha256': digest,
                'memory_bytes': 1024,
            },
            'fixture': {'kind': 'synthetic_or_redacted', 'sha256': digest},
            'seed': 1,
            'case_pack_sha256': digest,
            'store': {
                'schema_version': 1,
                'schema_manifest_sha256': digest,
                'content_identity_sha256': digest,
                'index_generation_sha256': digest,
                'document_count': 1,
            },
            'provider': {'provider_sha256': digest, 'model_sha256': digest, 'dimensions': 0},
            'execution': {
                'temperature': 'cold',
                'warmups': 0,
                'rounds': 1,
                'includes_engine_build': True,
            },
            'privacy': {
                'raw_fixture_identity_included': False,
                'raw_case_pack_included': False,
                'private_paths_included': False,
                'provider_names_included': False,
                'model_names_included': False,
            },
        },
    }


def _release_soak_gate_artifact(gate_id: str = 'concurrency-10m') -> dict:
    duration = 600.0 if gate_id == 'concurrency-10m' else 3600.0
    sample_count = 120 if gate_id == 'concurrency-10m' else 720
    heartbeat_windows = sample_count // 5
    artifact = _receipt_artifact(
        gate_id=gate_id,
        profile='release',
        command_id=U15_GATE_CONTRACTS[gate_id]['command_id'],
    )
    counts = {
        'searches_completed': heartbeat_windows * 10,
        'contacts_completed': heartbeat_windows * 10,
        'contexts_completed': heartbeat_windows * 10,
        'evidence_completed': heartbeat_windows * 10,
        'writes_completed': heartbeat_windows * 2,
        'writer_generation_contentions': heartbeat_windows * 2,
        'runtime_overloaded': 0,
        'runtime_timeouts': 0,
        'lock_errors': 0,
        'unhandled_errors': 0,
        'resource_samples': sample_count,
    }
    artifact.update({
        'artifact_type': 'release_soak_redacted',
        'elapsed_seconds': duration,
        'counts': counts,
        'configuration': {
            'readers': 32,
            'writers': 1,
            'warmup_searches': 8,
            'duration_seconds': duration,
            'writer_interval_seconds': 5.0,
            'sample_interval_seconds': 5.0,
            'search_mode': 'exact_off',
            'read_workload': 'exact+contacts+context+evidence',
            'process_isolated': True,
        },
        'cleanup': {
            'readers_joined': True,
            'writer_joined': True,
            'runtime_closed': True,
            'temp_vault_removed': True,
        },
        'resources': {
            'complete': True,
            'sample_count': sample_count,
            'required_sample_count': 20,
            'full_window_count': heartbeat_windows,
            'window_samples': 5,
            'sustained_windows': 3,
            'expected_sample_count': sample_count,
            'minimum_sample_count': sample_count - 1,
            'missed_deadlines': 0,
            'max_sample_gap_seconds': 5.25,
            'max_allowed_sample_gap_seconds': 7.5,
            'heartbeat_window_count': heartbeat_windows,
            'writer_heartbeat_required': True,
            'window_heartbeats': [
                {
                    'searches_completed': 10,
                    'contacts_completed': 10,
                    'contexts_completed': 10,
                    'evidence_completed': 10,
                    'writes_completed': 2,
                    'complete': True,
                }
                for _ in range(heartbeat_windows)
            ],
            'any_sustained_growth': False,
            'metrics': {
                'rss_bytes': {
                    'first_window_median': 1.0, 'last_window_median': 1.0,
                    'last_minus_first': 0.0, 'peak': 1,
                    'sustained_growth_threshold': 64 * 1024 * 1024, 'sustained_growth': False,
                },
                'fd_count': {
                    'first_window_median': 1.0, 'last_window_median': 1.0,
                    'last_minus_first': 0.0, 'peak': 1,
                    'sustained_growth_threshold': 8, 'sustained_growth': False,
                },
                'thread_count': {
                    'first_window_median': 1.0, 'last_window_median': 1.0,
                    'last_minus_first': 0.0, 'peak': 1,
                    'sustained_growth_threshold': 4, 'sustained_growth': False,
                },
            },
        },
    })
    artifact['release_receipt']['counts'] = counts
    artifact['assurance'] = dict(RELEASE_ASSURANCE)
    return artifact


def _source_entry(artifact_type: str, index: int = 0, *, gate_id: str | None = None) -> dict:
    entry = {
        'artifact_file': f'source-{index}.redacted.json',
        'artifact_sha256': f'{index + 1:064x}',
        'artifact_bytes': 100 + index,
        'artifact_type': artifact_type,
    }
    if gate_id is not None:
        entry['gate_id'] = gate_id
    return entry


def _locked_gate_artifact(gate_id: str) -> dict:
    if gate_id in {'concurrency-10m', 'soak-1h'}:
        return _release_soak_gate_artifact(gate_id)
    contract = U15_GATE_CONTRACTS[gate_id]
    artifact = _receipt_artifact(
        gate_id=gate_id,
        profile='release',
        command_id=contract['command_id'],
    )
    artifact['artifact_type'] = contract['artifact_type']
    artifact['assurance'] = dict(RELEASE_ASSURANCE)
    artifact['release_receipt']['counts'] = dict(contract['success_counts'])
    if contract['runner_kind'] == 'verified-evidence':
        source_evidence = [_source_entry(contract['source_artifact_types'][0])]
        verified_gate_ids: list[str] = []
    elif contract['runner_kind'] == 'verified-receipt-set':
        source_evidence = [
            _source_entry(
                U15_GATE_CONTRACTS[source_gate]['artifact_type'],
                index,
                gate_id=source_gate,
            )
            for index, source_gate in enumerate(contract['source_gate_ids'])
        ]
        verified_gate_ids = list(contract['source_gate_ids'])
    else:
        source_evidence = []
        verified_gate_ids = []
    artifact['gate_execution'] = {
        'contract_schema_version': CONTRACT_SCHEMA_VERSION,
        'contract_sha256': release_gate_contract_sha256(gate_id),
        'runner_kind': contract['runner_kind'],
        'checks_planned': contract['success_counts']['checks_completed'],
        'checks_completed': contract['success_counts']['checks_completed'],
        'source_artifacts': contract['success_counts'].get('supporting_artifacts', 0),
    }
    artifact['source_evidence'] = source_evidence
    artifact['verified_gate_ids'] = verified_gate_ids
    return artifact


def _retrieval_gate_source() -> dict:
    profiles = {
        'exact_cold': {'ok': True, 'route': 'exact', 'temperature': 'cold', 'semantic_mode': 'off'},
        'exact_warm': {'ok': True, 'route': 'exact', 'temperature': 'warm', 'semantic_mode': 'off'},
        'rewrite_cold': {'ok': True, 'route': 'rewrite', 'temperature': 'cold', 'semantic_mode': 'on'},
        'rewrite_warm': {'ok': True, 'route': 'rewrite', 'temperature': 'warm', 'semantic_mode': 'on'},
    }
    return {
        'schema_version': 1,
        'artifact_type': 'retrieval_eval_gate_redacted',
        'ok': True,
        'release_mode': True,
        'candidate_complete': True,
        'evidence_gate': {
            'ok': True,
            'manifests_required': True,
            'provenance_pair': {'candidate_commit': _SHA_A},
        },
        'frozen_pack_gate': {'ok': True},
        'relative_quality_gate': {'ok': True},
        'category_quality_gate': {'ok': True},
        'case_quality_gate': {'ok': True},
        'candidate_metrics': {'negative_only_queries': 1, 'negative_pass_rate': 1.0},
        'negative_regressions': 0,
        'negative_pass_rate_floor_miss': False,
        'latency_gate': {
            'ok': True,
            'required_profiles': sorted(profiles),
            'profiles': profiles,
        },
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
            'token_values_included': False,
        },
    }


class ReleaseSoakHarnessTests(unittest.TestCase):
    def test_slow_no_lock_generation_acquire_cannot_fake_contention(self):
        production_acquire = VaultGenerationLease.acquire

        for fake_active in (False, True):
            def slow_no_lock(lease):
                if lease.mode == 'publish':
                    time.sleep(0.05)
                    if fake_active:
                        lease._fd = os.open(lease.cfg.root, os.O_RDONLY)
                        lease._owner_pid = os.getpid()
                    return lease
                return production_acquire(lease)

            with self.subTest(fake_active=fake_active), patch.object(
                VaultGenerationLease, 'acquire', slow_no_lock,
            ):
                report = run_release_soak(
                    duration_seconds=1.0,
                    readers=8,
                    warmup_searches=1,
                    writer_interval_seconds=0.2,
                    sample_interval_seconds=0.1,
                    resource_window_samples=2,
                    sustained_windows=2,
                    sampler=lambda: {'rss_bytes': 1, 'fd_count': 1, 'thread_count': 1},
                    subject={'commit_sha': _SHA_A, 'dirty': False},
                )

                self.assertFalse(report['ok'])
                self.assertEqual(report['counts']['writes_completed'], 0)
                self.assertEqual(report['counts']['writer_generation_contentions'], 0)
                self.assertGreater(report['counts']['unhandled_errors'], 0)

    def test_stuck_worker_is_killed_and_returns_typed_failure_receipt(self):
        options = {
            'duration_seconds': 0.05,
            'readers': 2,
            'warmup_searches': 1,
            'writer_interval_seconds': 0.01,
            'sample_interval_seconds': 0.01,
            'resource_window_samples': 1,
            'sustained_windows': 1,
            'rss_growth_threshold_bytes': 1,
            'fd_growth_threshold': 1,
            'thread_growth_threshold': 1,
        }
        report = run_release_soak_isolated(
            options,
            hard_timeout_seconds=0.1,
            worker_target=_stuck_isolated_worker,
        )
        self.assertFalse(report['ok'])
        self.assertEqual(report['failure_code'], 'worker_hard_timeout')
        self.assertTrue(report['cleanup']['worker_process_reaped'])
        self.assertEqual(report['release_receipt']['profile'], 'test-only')

    def test_thirty_two_readers_writer_are_bounded_and_cleanup(self):
        baseline_threads = threading.active_count()
        baseline_fds = _open_fd_count()
        started = time.monotonic()

        def injected_clock() -> float:
            return time.monotonic() - started

        report = run_release_soak(
            duration_seconds=3.6,
            readers=32,
            warmup_searches=2,
            writer_interval_seconds=0.8,
            sample_interval_seconds=0.3,
            resource_window_samples=3,
            sustained_windows=2,
            clock=injected_clock,
            sampler=lambda: {'rss_bytes': 10_000_000, 'fd_count': 12, 'thread_count': 42},
            subject={'commit_sha': _SHA_A, 'dirty': False},
        )

        self.assertTrue(report['ok'], report)
        self.assertEqual(report['configuration']['readers'], 32)
        self.assertEqual(report['configuration']['writers'], 1)
        self.assertEqual(report['configuration']['search_mode'], 'exact_off')
        self.assertEqual(report['release_receipt']['profile'], 'test-only')
        self.assertGreater(report['counts']['searches_completed'], 0)
        self.assertGreater(report['counts']['contacts_completed'], 0)
        self.assertGreater(report['counts']['contexts_completed'], 0)
        self.assertGreater(report['counts']['evidence_completed'], 0)
        self.assertGreater(report['counts']['writes_completed'], 0)
        self.assertGreater(report['counts']['writer_generation_contentions'], 0)
        self.assertEqual(
            report['counts']['writer_generation_contentions'],
            report['counts']['writes_completed'],
        )
        self.assertEqual(report['counts']['runtime_overloaded'], 0)
        self.assertEqual(report['counts']['runtime_timeouts'], 0)
        self.assertEqual(report['counts']['lock_errors'], 0)
        self.assertEqual(report['counts']['unhandled_errors'], 0)
        self.assertFalse(report['resources']['any_sustained_growth'])
        self.assertEqual(report['resources']['missed_deadlines'], 0)
        self.assertLessEqual(
            report['resources']['max_sample_gap_seconds'],
            report['resources']['max_allowed_sample_gap_seconds'],
        )
        self.assertTrue(all(report['cleanup'].values()))
        self.assertLessEqual(threading.active_count(), baseline_threads + 1)
        after_fds = _open_fd_count()
        if baseline_fds is not None and after_fds is not None:
            self.assertLessEqual(after_fds, baseline_fds + 2)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(tempfile.gettempdir() + '/', serialized)
        self.assertNotIn('示例教育', serialized)

    def test_injected_sampler_detects_sustained_growth(self):
        samples = [
            {'rss_bytes': 100, 'fd_count': 10, 'thread_count': 5},
            {'rss_bytes': 100, 'fd_count': 10, 'thread_count': 5},
            {'rss_bytes': 200, 'fd_count': 20, 'thread_count': 10},
            {'rss_bytes': 200, 'fd_count': 20, 'thread_count': 10},
            {'rss_bytes': 210, 'fd_count': 21, 'thread_count': 11},
            {'rss_bytes': 210, 'fd_count': 21, 'thread_count': 11},
        ]
        summary = _resource_summary(
            samples,
            window_samples=2,
            sustained_windows=2,
            growth_thresholds={'rss_bytes': 50, 'fd_count': 5, 'thread_count': 2},
        )
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['any_sustained_growth'])
        self.assertEqual(summary['metrics']['rss_bytes']['first_window_median'], 100.0)
        self.assertEqual(summary['metrics']['rss_bytes']['last_window_median'], 210.0)

    def test_insufficient_windows_never_claim_complete_or_bounded(self):
        summary = _resource_summary(
            [{'rss_bytes': 100, 'fd_count': 10, 'thread_count': 5}],
            window_samples=5,
            sustained_windows=3,
            growth_thresholds={'rss_bytes': 50, 'fd_count': 5, 'thread_count': 2},
        )
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['required_sample_count'], 20)
        self.assertIsNone(summary['metrics']['rss_bytes']['first_window_median'])

    def test_release_contract_requires_120_or_720_continuous_samples_and_heartbeats(self):
        for sample_count, duration in ((120, 600.0), (720, 3600.0)):
            samples = [
                {'rss_bytes': 100, 'fd_count': 10, 'thread_count': 5}
                for _ in range(sample_count)
            ]
            heartbeat_samples = [
                {
                    'searches_completed': index + 1,
                    'contacts_completed': index + 1,
                    'contexts_completed': index + 1,
                    'evidence_completed': index + 1,
                    'writes_completed': index + 1,
                }
                for index in range(sample_count)
            ]
            summary = _resource_summary(
                samples,
                window_samples=5,
                sustained_windows=3,
                growth_thresholds={'rss_bytes': 50, 'fd_count': 5, 'thread_count': 2},
                sample_offsets=[index * 5.0 for index in range(sample_count)],
                heartbeat_samples=heartbeat_samples,
                heartbeat_baseline={key: 0 for key in heartbeat_samples[0]},
                sample_interval_seconds=5.0,
                duration_seconds=duration,
                missed_deadlines=0,
                require_writer_heartbeat_per_window=True,
            )
            self.assertTrue(summary['complete'], summary)
            self.assertEqual(summary['expected_sample_count'], sample_count)
            self.assertEqual(summary['heartbeat_window_count'], sample_count // 5)

        ten_minutes = _release_soak_gate_artifact('concurrency-10m')
        one_hour = _release_soak_gate_artifact('soak-1h')
        self.assertTrue(release_soak_contract_valid(ten_minutes, 'concurrency-10m'))
        self.assertTrue(release_soak_contract_valid(one_hour, 'soak-1h'))
        self.assertEqual(ten_minutes['resources']['sample_count'], 120)
        self.assertEqual(one_hour['resources']['sample_count'], 720)

        too_few = _release_soak_gate_artifact('concurrency-10m')
        too_few['resources']['sample_count'] = 20
        too_few['resources']['full_window_count'] = 4
        too_few['resources']['heartbeat_window_count'] = 4
        too_few['resources']['window_heartbeats'] = too_few['resources']['window_heartbeats'][:4]
        too_few['counts']['resource_samples'] = 20
        too_few['release_receipt']['counts']['resource_samples'] = 20
        self.assertFalse(release_soak_contract_valid(too_few, 'concurrency-10m'))

        paused_then_caught_up = _release_soak_gate_artifact('concurrency-10m')
        paused_then_caught_up['resources']['missed_deadlines'] = 1
        paused_then_caught_up['resources']['max_sample_gap_seconds'] = 25.0
        self.assertFalse(release_soak_contract_valid(paused_then_caught_up, 'concurrency-10m'))

        missing_heartbeat = _release_soak_gate_artifact('soak-1h')
        missing_heartbeat['resources']['window_heartbeats'][71]['writes_completed'] = 0
        missing_heartbeat['resources']['window_heartbeats'][71]['complete'] = False
        self.assertFalse(release_soak_contract_valid(missing_heartbeat, 'soak-1h'))

    def test_errors_are_classified_without_message_output(self):
        self.assertEqual(classify_soak_error(RuntimeOverloaded()), 'runtime_overloaded')
        self.assertEqual(classify_soak_error(RuntimeTimedOut()), 'runtime_timeouts')
        self.assertEqual(classify_soak_error(VaultOperationLocked()), 'lock_errors')
        self.assertEqual(classify_soak_error(sqlite3.OperationalError('database is busy')), 'lock_errors')
        self.assertEqual(classify_soak_error(ValueError('synthetic failure')), 'unhandled_errors')

    def test_abnormal_runtime_still_joins_closes_and_removes_temp_vault(self):
        closed = threading.Event()

        class FailingRuntime:
            def search(self, _request):
                raise RuntimeTimedOut()

            def invalidate(self, _reason):
                return None

            def close(self):
                closed.set()

        report = run_release_soak(
            duration_seconds=0.08,
            readers=4,
            warmup_searches=1,
            writer_interval_seconds=0.02,
            sample_interval_seconds=0.01,
            resource_window_samples=1,
            sustained_windows=2,
            sampler=lambda: {'rss_bytes': 1, 'fd_count': 1, 'thread_count': 1},
            subject={'commit_sha': _SHA_A, 'dirty': False},
            runtime_factory=lambda _cfg, _readers: FailingRuntime(),  # type: ignore[arg-type]
        )
        self.assertFalse(report['ok'])
        self.assertGreater(report['counts']['runtime_timeouts'], 0)
        self.assertTrue(closed.is_set())
        self.assertTrue(all(report['cleanup'].values()))


class U10ComplexityProducerTests(unittest.TestCase):
    def test_production_ledger_sidecar_and_bounded_watcher_are_exercised(self):
        report = benchmark_u10_scaling.measure(128, 256, logical_watch=True)
        self.assertTrue(report['ok'], report)
        self.assertTrue(all(report['checks'].values()))
        self.assertEqual(report['vector_metadata']['authoritative_rows'], 128)
        self.assertEqual(report['vector_metadata']['delta_candidate_rows'], 1)
        self.assertLessEqual(report['vector_metadata']['constant_sidecar_bytes'], 4096)
        self.assertTrue(report['watcher']['production_probe']['scan_completed'])
        self.assertLessEqual(
            report['watcher']['production_probe']['max_entries_processed_per_tick'],
            report['thresholds']['fallback_rescan_max_entries_per_tick'],
        )

    def test_threshold_regression_and_production_path_fault_fail_closed(self):
        with patch.object(benchmark_u10_scaling, 'DIRTY_BATCH_LIMIT', 1024):
            oversized_batch = benchmark_u10_scaling.measure(600, 32, logical_watch=True)
        self.assertFalse(oversized_batch['ok'])
        self.assertFalse(oversized_batch['checks']['dirty_backlog_is_bounded'])

        production_apply = benchmark_u10_scaling.VectorIndexLedger.apply_delta

        def over_budget_apply(ledger, *args, **kwargs):
            result = production_apply(ledger, *args, **kwargs)
            return {**result, 'sql_statements': 11}

        with patch.object(benchmark_u10_scaling.VectorIndexLedger, 'apply_delta', over_budget_apply):
            over_budget = benchmark_u10_scaling.measure(64, 32, logical_watch=True)
        self.assertFalse(over_budget['ok'])
        self.assertFalse(over_budget['checks']['ledger_delta_is_constant'])

        with (
            patch.object(benchmark_u10_scaling, 'measure', return_value={'ok': False}),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(benchmark_u10_scaling.main([]), 2)


class ReleaseGateRunnerTests(unittest.TestCase):
    @staticmethod
    def _write(directory: Path, name: str, artifact: dict) -> Path:
        path = directory / name
        write_evidence_artifact(artifact, path)
        return path

    def test_locked_command_bundle_has_no_arbitrary_command_seam(self):
        executed: list[tuple[tuple[str, ...], int]] = []

        def executor(argv, timeout):
            executed.append((tuple(argv), timeout))
            return True

        report = run_release_gate(
            'privacy-security',
            executor=executor,
            subject={'commit_sha': _SHA_A, 'dirty': False},
            system='TestOS',
        )
        expected = [
            (tuple(step['argv']), step['timeout_seconds'])
            for step in U15_GATE_CONTRACTS['privacy-security']['steps']
        ]
        self.assertTrue(report['ok'])
        self.assertEqual(executed, expected)
        self.assertEqual(report['release_receipt']['profile'], 'test-only')
        self.assertEqual(report['gate_execution']['contract_sha256'], release_gate_contract_sha256('privacy-security'))
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn('privacy:scan', serialized)
        self.assertNotIn('./scripts/trove-python', serialized)

    def test_evidence_backed_quality_and_latency_are_independently_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retrieval = self._write(root, 'retrieval.redacted.json', _retrieval_gate_source())
            for gate_id, sources in (
                ('retrieval-quality-negative', [retrieval]),
                ('exact-rewrite-latency', [retrieval]),
            ):
                report = run_release_gate(
                    gate_id,
                    source_paths=sources,
                    executor=lambda _argv, _timeout: True,
                    subject={'commit_sha': _SHA_A, 'dirty': False},
                    system='TestOS',
                )
                self.assertTrue(report['ok'], report)
                self.assertEqual(report['release_receipt']['counts'], U15_GATE_CONTRACTS[gate_id]['success_counts'])
                self.assertEqual(len(report['source_evidence']), 1)

            unsafe = _retrieval_gate_source()
            unsafe['negative_pass_rate_floor_miss'] = True
            failed_path = self._write(root, 'retrieval-failed.redacted.json', unsafe)
            failed = run_release_gate(
                'retrieval-quality-negative',
                source_paths=[failed_path],
                executor=lambda _argv, _timeout: True,
                subject={'commit_sha': _SHA_A, 'dirty': False},
                system='TestOS',
            )
            self.assertFalse(failed['ok'])
            self.assertEqual(failed['failure_code'], 'retrieval_quality_gate_failed')

    def test_complexity_gate_is_locked_to_real_one_million_scaling_producer(self):
        executed = []

        def executor(argv, timeout):
            executed.append((tuple(argv), timeout))
            return True

        report = run_release_gate(
            'complexity-1m',
            executor=executor,
            subject={'commit_sha': _SHA_A, 'dirty': False},
            system='TestOS',
        )
        self.assertTrue(report['ok'])
        self.assertEqual(executed, [
            (tuple(step['argv']), step['timeout_seconds'])
            for step in U15_GATE_CONTRACTS['complexity-1m']['steps']
        ])
        commands = [entry[0] for entry in executed]
        self.assertIn('scripts/benchmark_u10_scaling.py', commands[0])
        self.assertEqual(commands[0].count('1000000'), 2)
        self.assertIn('--logical-watch', commands[0])
        self.assertIn('scripts/benchmark_u9_delta.py', commands[1])
        self.assertIn('1000000', commands[1])
        self.assertIn('scripts/benchmark_u7_bounds.py', commands[2])
        self.assertIn('10000', commands[2])
        self.assertEqual(report['release_receipt']['counts']['checks_completed'], 3)

    def test_provenance_runner_requires_exact_other_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for gate_id in U15_GATE_CONTRACTS['provenance']['source_gate_ids']:
                sources.append(self._write(root, f'{gate_id}.redacted.json', _locked_gate_artifact(gate_id)))
            report = run_release_gate(
                'provenance',
                source_paths=sources,
                executor=lambda _argv, _timeout: True,
                subject={'commit_sha': _SHA_A, 'dirty': False},
                system='TestOS',
            )
            self.assertTrue(report['ok'], report)
            self.assertEqual(report['verified_gate_ids'], list(U15_GATE_CONTRACTS['provenance']['source_gate_ids']))
            self.assertEqual(
                report['release_receipt']['counts']['supporting_artifacts'],
                len(U15_GATE_CONTRACTS['provenance']['source_gate_ids']),
            )

            missing = run_release_gate(
                'provenance',
                source_paths=sources[:-1],
                executor=lambda _argv, _timeout: True,
                subject={'commit_sha': _SHA_A, 'dirty': False},
                system='TestOS',
            )
            self.assertFalse(missing['ok'])
            self.assertEqual(missing['failure_code'], 'source_artifact_count_mismatch')

            invalid_soak = _locked_gate_artifact('concurrency-10m')
            invalid_soak['resources']['full_window_count'] = 23
            invalid_path = self._write(root, 'invalid-concurrency.redacted.json', invalid_soak)
            invalid_sources = [
                invalid_path if path.name == 'concurrency-10m.redacted.json' else path
                for path in sources
            ]
            invalid = run_release_gate(
                'provenance',
                source_paths=invalid_sources,
                executor=lambda _argv, _timeout: True,
                subject={'commit_sha': _SHA_A, 'dirty': False},
                system='TestOS',
            )
            self.assertFalse(invalid['ok'])
            self.assertEqual(invalid['failure_code'], 'source_receipt_invalid')

    def test_macos_gate_fails_before_steps_on_non_macos(self):
        called = False

        def executor(_argv, _timeout):
            nonlocal called
            called = True
            return True

        report = run_release_gate(
            'macos-contracts',
            executor=executor,
            subject={'commit_sha': _SHA_A, 'dirty': False},
            system='Linux',
        )
        self.assertFalse(report['ok'])
        self.assertEqual(report['failure_code'], 'platform_contract_mismatch')
        self.assertFalse(called)

    def test_output_sidecar_cannot_overwrite_source_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'gate.redacted.json'
            source = root / 'gate.redacted.json.manifest.json'
            source.write_text('must-not-change', encoding='utf-8')
            with redirect_stdout(io.StringIO()):
                status = release_gate_main([
                    '--gate-id', 'complexity-1m',
                    '--source', str(source),
                    '--out', str(out),
                ])
            self.assertEqual(status, 2)
            self.assertEqual(source.read_text(encoding='utf-8'), 'must-not-change')
            self.assertFalse(out.exists())


class ReleaseCloseoutManifestTests(unittest.TestCase):
    @staticmethod
    def _write(directory: Path, name: str, artifact: dict) -> Path:
        path = directory / name
        write_evidence_artifact(artifact, path)
        return path

    def test_manifest_is_reproducible_redacted_and_has_independent_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            second = self._write(root, 'b.redacted.json', _receipt_artifact(command_id='check-b-v1'))
            first = self._write(root, 'a.redacted.json', _receipt_artifact(command_id='check-a-v1'))
            manifest = build_closeout_manifest([second, first], required_gate_ids={'synthetic-check'})
            self.assertEqual([item['artifact_file'] for item in manifest['inputs']], [
                'a.redacted.json',
                'b.redacted.json',
            ])
            self.assertEqual(manifest['subject_sha'], _SHA_A)
            serialized = json.dumps(manifest, sort_keys=True)
            self.assertNotIn(str(root), serialized)
            for forbidden in ('"query"', '"citation"', '"provider"', '"model"', '"secret"'):
                self.assertNotIn(forbidden, serialized)

            out = root / 'closeout.redacted.json'
            write_evidence_artifact(manifest, out)
            sidecar = verify_evidence_manifest(out, required=True)
            self.assertEqual(sidecar['artifact_file'], out.name)

    def test_default_closeout_requires_all_locked_runner_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = []
            gate_paths: dict[str, Path] = {}
            for gate_id in sorted(U15_REQUIRED_GATE_IDS - {'provenance'}):
                path = self._write(root, f'{gate_id}.redacted.json', _locked_gate_artifact(gate_id))
                artifacts.append(path)
                gate_paths[gate_id] = path
            provenance = _locked_gate_artifact('provenance')
            provenance['source_evidence'] = []
            for gate_id in U15_GATE_CONTRACTS['provenance']['source_gate_ids']:
                path = gate_paths[gate_id]
                sidecar = verify_evidence_manifest(path, required=True)
                provenance['source_evidence'].append({
                    'artifact_file': path.name,
                    'artifact_sha256': sidecar['artifact_sha256'],
                    'artifact_bytes': sidecar['artifact_bytes'],
                    'artifact_type': _locked_gate_artifact(gate_id)['artifact_type'],
                    'gate_id': gate_id,
                })
            artifacts.append(self._write(root, 'provenance.redacted.json', provenance))
            manifest = build_closeout_manifest(artifacts)
            self.assertTrue(manifest['ok'])
            self.assertEqual(set(manifest['required_gate_ids']), set(U15_REQUIRED_GATE_IDS))
            self.assertEqual(manifest['counts']['required_gates'], len(U15_REQUIRED_GATE_IDS))
            self.assertEqual(manifest['counts']['passed_gates'], len(U15_REQUIRED_GATE_IDS))

            provenance['source_evidence'][0]['artifact_sha256'] = 'f' * 64
            forged = self._write(root, 'provenance-forged.redacted.json', provenance)
            with self.assertRaisesRegex(CloseoutManifestError, 'provenance_receipt_not_bound'):
                build_closeout_manifest([*artifacts[:-1], forged])

    def test_generic_self_report_and_gate_contract_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic = _receipt_artifact(
                gate_id='privacy-security',
                profile='release',
                command_id='self-reported-pass-v1',
            )
            generic_path = self._write(root, 'generic.redacted.json', generic)
            with self.assertRaisesRegex(CloseoutManifestError, 'invalid_release_gate_contract'):
                build_closeout_manifest([generic_path], required_gate_ids={'privacy-security'})

            valid = _locked_gate_artifact('privacy-security')
            valid['gate_execution']['contract_sha256'] = '0' * 64
            forged_path = self._write(root, 'forged.redacted.json', valid)
            with self.assertRaisesRegex(CloseoutManifestError, 'invalid_release_gate_execution'):
                build_closeout_manifest([forged_path], required_gate_ids={'privacy-security'})

            count_forgery = _locked_gate_artifact('functional-minimal')
            count_forgery['release_receipt']['counts']['checks_completed'] = 3
            count_forgery['gate_execution']['checks_completed'] = 3
            count_path = self._write(root, 'count-forged.redacted.json', count_forgery)
            with self.assertRaisesRegex(CloseoutManifestError, 'invalid_release_gate_contract'):
                build_closeout_manifest([count_path], required_gate_ids={'functional-minimal'})

    def test_output_sidecar_cannot_overwrite_input_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'closeout.redacted.json'
            input_artifact = self._write(
                root,
                'closeout.redacted.json.manifest.json',
                _receipt_artifact(),
            )
            before = input_artifact.read_bytes()
            with redirect_stdout(io.StringIO()):
                status = closeout_main([
                    str(input_artifact),
                    '--out', str(out),
                    '--require-gate', 'synthetic-check',
                ])
            self.assertEqual(status, 2)
            self.assertEqual(input_artifact.read_bytes(), before)
            self.assertFalse(out.exists())

    def test_tampered_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._write(root, 'tamper.redacted.json', _receipt_artifact())
            artifact.write_text(artifact.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_sidecar_invalid'):
                build_closeout_manifest([artifact])

    def test_standard_provenance_is_verified_but_sensitive_identity_fields_are_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = self._write(root, 'gate.redacted.json', _receipt_artifact())
            artifact = self._write(root, 'standard.redacted.json', _standard_provenance_artifact())
            manifest = build_closeout_manifest([artifact, gate], required_gate_ids={'synthetic-check'})
            serialized = json.dumps(manifest, sort_keys=True)
            self.assertEqual(manifest['subject_sha'], _SHA_A)
            roles = {item['artifact_file']: item['role'] for item in manifest['inputs']}
            self.assertEqual(roles['gate.redacted.json'], 'gate')
            self.assertEqual(roles['standard.redacted.json'], 'supporting')
            self.assertNotIn('provider_sha256', serialized)
            self.assertNotIn('model_sha256', serialized)
            self.assertNotIn('case_pack_sha256', serialized)

    def test_failed_typed_soak_and_failed_gate_counts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed_soak = self._write(root, 'failed-soak.redacted.json', _receipt_artifact(ok=False))
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_gate_failed'):
                build_closeout_manifest([failed_soak])

            failed_gate = self._write(root, 'failed-gate.redacted.json', _receipt_artifact(failures=1))
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_gate_failed'):
                build_closeout_manifest([failed_gate])

            inconsistent = _receipt_artifact()
            inconsistent['release_receipt']['counts']['unhandled_errors'] = 1
            inconsistent_gate = self._write(root, 'inconsistent-gate.redacted.json', inconsistent)
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_gate_failed'):
                build_closeout_manifest([inconsistent_gate])

    def test_standard_evidence_cannot_claim_pass_and_quality_failed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supporting = self._write(root, 'supporting.redacted.json', _standard_provenance_artifact())
            with self.assertRaisesRegex(CloseoutManifestError, 'missing_required_gate_receipts'):
                build_closeout_manifest([supporting], required_gate_ids={'synthetic-check'})

            failed = _standard_provenance_artifact()
            failed['ok'] = False
            failed_quality = self._write(root, 'quality-failed.redacted.json', failed)
            gate = self._write(root, 'passed-gate.redacted.json', _receipt_artifact())
            with self.assertRaisesRegex(CloseoutManifestError, 'supporting_artifact_failed'):
                build_closeout_manifest([gate, failed_quality])

    def test_mismatched_subject_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write(root, 'one.redacted.json', _receipt_artifact(_SHA_A))
            second = self._write(root, 'two.redacted.json', _receipt_artifact(_SHA_B))
            with self.assertRaisesRegex(CloseoutManifestError, 'subject_sha_mismatch'):
                build_closeout_manifest([first, second])

    def test_privacy_mismatch_and_dirty_subject_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe_privacy = dict(_PRIVACY)
            unsafe_privacy['sensitive_content_included'] = True
            unsafe = self._write(root, 'unsafe.redacted.json', _receipt_artifact(privacy=unsafe_privacy))
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_privacy_mismatch'):
                build_closeout_manifest([unsafe])

            dirty = self._write(root, 'dirty.redacted.json', _receipt_artifact(dirty=True))
            with self.assertRaisesRegex(CloseoutManifestError, 'dirty_subject'):
                build_closeout_manifest([dirty])

    def test_whole_artifact_redaction_is_enforced_not_self_asserted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = _receipt_artifact()
            unsafe['query'] = 'RAW PRIVATE QUERY'
            unsafe['private_path'] = '/Users/' + 'example/SecretVault'
            artifact = self._write(root, 'unsafe-body.redacted.json', unsafe)
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_not_strictly_redacted'):
                build_closeout_manifest([artifact], required_gate_ids={'synthetic-check'})

            standard = _standard_provenance_artifact()
            standard['provenance']['privacy']['credential_values_included'] = True
            standard_artifact = self._write(root, 'unsafe-standard.redacted.json', standard)
            with self.assertRaisesRegex(CloseoutManifestError, 'artifact_privacy_mismatch'):
                build_closeout_manifest([standard_artifact], required_gate_ids={'synthetic-check'})

    def test_test_only_receipt_cannot_satisfy_required_release_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._write(
                root,
                'test-only.redacted.json',
                _receipt_artifact(gate_id='concurrency-10m', profile='test-only'),
            )
            with self.assertRaisesRegex(CloseoutManifestError, 'missing_required_gate_receipts'):
                build_closeout_manifest([artifact], required_gate_ids={'concurrency-10m'})

    def test_soak_gate_contract_is_independently_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self._write(root, 'valid-soak.redacted.json', _release_soak_gate_artifact())
            manifest = build_closeout_manifest([valid], required_gate_ids={'concurrency-10m'})
            self.assertTrue(manifest['ok'])

            unsafe = _release_soak_gate_artifact()
            unsafe['cleanup']['runtime_closed'] = False
            unsafe['resources']['any_sustained_growth'] = True
            failed = self._write(root, 'failed-soak-contract.redacted.json', unsafe)
            with self.assertRaisesRegex(CloseoutManifestError, 'invalid_soak_gate_receipt'):
                build_closeout_manifest([failed], required_gate_ids={'concurrency-10m'})

    def test_v1_manifest_binds_distribution_and_every_release_evidence_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution = {
                'source_git_sha': _SHA_A,
                'source_dirty': False,
                'distribution_set_sha256': 'd' * 64,
                'runtime_build_hash': '1' * 64,
                'catalog_hash': '2' * 64,
                'provider_package_hash': '3' * 64,
                'runtime': {'sha256': '4' * 64},
                'provider': {'sha256': '5' * 64},
            }
            distribution_path = root / 'distribution-manifest.json'
            distribution_path.write_text(json.dumps(distribution), encoding='utf-8')

            closeout = {
                'schema_version': 1,
                'artifact_type': 'release_closeout_manifest_redacted',
                'ok': True,
                'subject_sha': _SHA_A,
                'required_gate_ids': sorted(U15_REQUIRED_GATE_IDS),
                'counts': {
                    'required_gates': len(U15_REQUIRED_GATE_IDS),
                    'passed_gates': len(U15_REQUIRED_GATE_IDS),
                },
                'privacy': dict(_PRIVACY),
                'assurance': dict(RELEASE_ASSURANCE),
            }
            closeout_path = self._write(root, 'closeout.redacted.json', closeout)

            perf = copy.deepcopy(json.loads(Path('docs/perf/agent-runtime-budgets.json').read_text()))
            perf['current_fixture_baseline']['git_sha'] = _SHA_A
            perf_path = root / 'perf.redacted.json'
            perf_path.write_text(json.dumps(perf), encoding='utf-8')

            acceptance_privacy = {
                'content_included': False,
                'queries_included': False,
                'citations_included': False,
                'account_identifiers_included': False,
                'private_paths_included': False,
                'raw_errors_included': False,
                'secret_values_included': False,
            }
            clients = [
                {
                    'client': name,
                    'tasks': 6,
                    'tasks_succeeded': 6,
                    'task_success_rate': 1.0,
                    'calls': 9,
                    'wrong_tool_calls': 0,
                    'operator_interventions': 0,
                    'citation_count': 6,
                }
                for name in ('first', 'second')
            ]
            agent = {
                'schema_version': 1,
                'artifact_type': 'agent_product_acceptance_redacted',
                'ok': True,
                'fixture_sha256': '6' * 64,
                'clients': clients,
                'summary': {
                    'clients': 2,
                    'tasks': 12,
                    'tasks_succeeded': 12,
                    'task_success_rate': 1.0,
                    'calls': 18,
                    'wrong_tool_calls': 0,
                    'operator_interventions': 0,
                    'citation_count': 12,
                },
                'privacy': acceptance_privacy,
            }
            agent_path = root / 'agent.redacted.json'
            agent_path.write_text(json.dumps(agent), encoding='utf-8')

            case = {
                'ok': True,
                'runs': 1,
                'cold_latency_ms': 1.0,
                'warm_latency_ms': None,
                'result_count': 1,
                'citation_count': 1,
                'coverage_complete': True,
                'cursor_present': False,
                'error_code': None,
            }
            real = {
                'schema_version': 1,
                'artifact_type': 'agent_runtime_real_vault_acceptance_redacted',
                'ok': True,
                'input_sha256': '7' * 64,
                'cases': {name: dict(case) for name in ('capabilities', 'accounts', 'recall', 'search')},
                'legacy_citations': {'requested': 1, 'readable': 1, 'max_latency_ms': 1.0, 'ok': True},
                'quality': {
                    'cases': 2,
                    'cases_succeeded': 2,
                    'citations_found': 2,
                    'complete_coverage_cases': 2,
                    'ok': True,
                },
                'warm_latency_ms': {'p50': 1.0, 'max': 1.0},
                'baseline': {'provided': False, 'checks': 0, 'passed': 0, 'ok': True},
                'privacy': acceptance_privacy,
            }
            real_path = root / 'real.redacted.json'
            real_path.write_text(json.dumps(real), encoding='utf-8')

            controls = {
                'checks': {key: True for key in CUTOVER_CHECK_IDS},
                'counts': {key: 0 for key in CUTOVER_COUNT_KEYS},
            }
            with patch('verify_distribution.verify_distribution', return_value={'ok': True}):
                cutover = build_cutover_acceptance(distribution_path, controls)
            cutover_path = self._write(root, 'cutover.redacted.json', cutover)

            with patch('verify_distribution.verify_distribution', return_value={'ok': True}):
                manifest = build_v1_release_manifest(
                    distribution_manifest_path=distribution_path,
                    test_summary_path=closeout_path,
                    perf_summary_path=perf_path,
                    agent_acceptance_path=agent_path,
                    real_vault_acceptance_path=real_path,
                    cutover_acceptance_path=cutover_path,
                )
            validate_v1_release_manifest(manifest)
            self.assertEqual(manifest['source_git_sha'], _SHA_A)
            self.assertEqual(manifest['distribution_set_sha256'], 'd' * 64)
            self.assertEqual(set(manifest['evidence_sha256']), {
                'agent_acceptance_sha256', 'cutover_acceptance_sha256',
                'perf_summary_sha256', 'real_vault_acceptance_sha256',
                'test_summary_sha256',
            })
            self.assertEqual(manifest['privacy'], V1_RELEASE_PRIVACY)

            perf['current_fixture_baseline']['git_sha'] = _SHA_B
            perf_path.write_text(json.dumps(perf), encoding='utf-8')
            with patch('verify_distribution.verify_distribution', return_value={'ok': True}):
                with self.assertRaisesRegex(CloseoutManifestError, 'perf_subject_sha_mismatch'):
                    build_v1_release_manifest(
                        distribution_manifest_path=distribution_path,
                        test_summary_path=closeout_path,
                        perf_summary_path=perf_path,
                        agent_acceptance_path=agent_path,
                        real_vault_acceptance_path=real_path,
                        cutover_acceptance_path=cutover_path,
                    )


if __name__ == '__main__':
    unittest.main()
