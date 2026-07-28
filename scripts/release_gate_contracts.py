"""Immutable U15 gate/runner contracts shared by emitters and closeout."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


AGENT_RUNTIME_TARGETS = {
    'warm_exact_recall_p50_ms_max': 50,
    'warm_exact_recall_p95_ms_max': 150,
    'warm_lexical_search_p95_ms_max': 250,
    'warm_context_p95_ms_max': 100,
    'protocol_adapter_noop_p95_ms_max': 10,
    'daemon_cold_ready_p95_ms_max': 1500,
    'daemon_idle_rss_mib_max': 96,
    'mcp_idle_rss_mib_max': 72,
    'idle_cpu_percent_max': 0.5,
    'standard_tools_list_bytes_max': 24 * 1024,
    'standard_tools_list_tokens_max': 6000,
    'compact_response_soft_bytes_max': 64 * 1024,
    'response_hard_bytes_max': 256 * 1024,
}


def agent_runtime_budget_contract_valid(payload: Mapping[str, Any]) -> bool:
    """Validate the frozen U1 baseline and immutable absolute target set."""

    try:
        if payload.get('schema_version') != 1:
            return False
        if payload.get('artifact_type') != 'agent_runtime_release_budgets':
            return False
        if dict(payload.get('targets') or {}) != AGENT_RUNTIME_TARGETS:
            return False
        baseline = payload.get('current_fixture_baseline')
        if not isinstance(baseline, dict):
            return False
        from scripts.benchmark_agent_runtime import evaluate_absolute_budgets, validate_benchmark_artifact

        validate_benchmark_artifact(baseline)
        if evaluate_absolute_budgets(baseline).get('ok') is not True:
            return False
        policy = payload.get('relative_regression_policy')
        if not isinstance(policy, dict):
            return False
        return (
            policy.get('same_hardware_required') is True
            and policy.get('same_fixture_required') is True
            and policy.get('same_rounds_required') is True
            and 1.0 <= float(policy.get('latency_p95_max_ratio')) <= 1.25
        )
    except (KeyError, TypeError, ValueError):
        return False


CONTRACT_SCHEMA_VERSION = 1
RELEASE_ASSURANCE = {
    'contract_locked': True,
    'tamper_evident': True,
    'clean_sha_reproducible': True,
    'attestation_out_of_scope': True,
}

SOAK_COUNT_KEYS = (
    'contacts_completed',
    'contexts_completed',
    'evidence_completed',
    'lock_errors',
    'resource_samples',
    'runtime_overloaded',
    'runtime_timeouts',
    'searches_completed',
    'unhandled_errors',
    'writer_generation_contentions',
    'writes_completed',
)


def _step(*argv: str, timeout_seconds: int = 1800) -> dict[str, Any]:
    return {'argv': tuple(argv), 'timeout_seconds': timeout_seconds}


# These are deliberately exact command bundles, not user-supplied shell text.
# The receipt contains only the contract hash and aggregate counts; stdout,
# stderr, argv, paths, environment values, and exception messages are discarded.
_PRIVACY_SECURITY_STEPS = (
    _step('./scripts/trove-python', 'scripts/privacy_scan.py', '.', timeout_seconds=600),
    _step(
        './scripts/trove-python',
        'scripts/run_unittest_no_skips.py',
        'packages.trove_core.tests.test_approval_integrity',
        'packages.trove_core.tests.test_cloud_egress_approvals',
        'packages.trove_core.tests.test_cloud_reranker',
        'packages.trove_core.tests.test_embedding_provider_contract',
        'packages.trove_core.tests.test_local_token_manager',
        'packages.trove_core.tests.test_provider_factory_contract',
        'packages.trove_core.tests.test_provider_config',
        'packages.trove_core.tests.test_managed_process',
        'packages.trove_core.tests.test_sensitive_application_boundary',
        'packages.trove_core.tests.test_wechat_key_capture',
        'tests.e2e.test_untrusted_evidence',
        'tests.e2e.test_human_approval_control',
        timeout_seconds=900,
    ),
)

_PRODUCT_EXTRAS_STEPS = (
    _step('./scripts/trove-python', 'scripts/check.py', 'e2e', timeout_seconds=1800),
    _step(
        './scripts/trove-python',
        'scripts/run_agent_product_acceptance.py',
        timeout_seconds=300,
    ),
    _step(
        './scripts/trove-python',
        'scripts/run_unittest_no_skips.py',
        'tests.e2e.test_vector_incremental_sync',
        timeout_seconds=900,
    ),
)

_MACOS_CONTRACT_STEPS = (
    _step('./scripts/trove-python', 'scripts/check.py', 'contract', timeout_seconds=1800),
    _step('./scripts/trove-python', 'scripts/check.py', 'package', timeout_seconds=900),
    _step('./scripts/trove-python', 'scripts/runtime_doctor.py', '--json', timeout_seconds=300),
)

_CRASH_RECOVERY_STEPS = (
    _step(
        './scripts/trove-python',
        'scripts/run_unittest_no_skips.py',
        'packages.trove_core.tests.test_writer_marker_recovery',
        'packages.trove_cli.tests.test_writer_marker_recovery_cli',
        'packages.trove_core.tests.test_import_job_resume',
        'packages.trove_core.tests.test_generation_mutation_integration',
        'packages.trove_core.tests.test_vault_generation_lease',
        'packages.trove_core.tests.test_schema_migrations',
        'packages.trove_core.tests.test_vector_index_ledger',
        'packages.trove_core.tests.test_zvec_real_adapter',
        'tests.e2e.test_vector_incremental_sync',
        'tests.e2e.test_operation_crash_recovery',
        'tests.e2e.test_provider_fault_recovery',
        'tests.e2e.test_agent_concurrency',
        'tests.e2e.test_multi_vault_isolation',
        'tests.e2e.test_multi_account_scope',
        timeout_seconds=1200,
    ),
)


U15_GATE_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    'privacy-security': {
        'command_id': 'run-u15-privacy-security-v1',
        'artifact_type': 'release_privacy_security_gate_redacted',
        'runner_kind': 'locked-command-bundle',
        'steps': _PRIVACY_SECURITY_STEPS,
        'count_keys': ('checks_completed', 'failures'),
        'success_counts': {'checks_completed': len(_PRIVACY_SECURITY_STEPS), 'failures': 0},
    },
    'functional-minimal': {
        'command_id': 'run-u15-functional-minimal-v1',
        'artifact_type': 'release_functional_minimal_gate_redacted',
        'runner_kind': 'locked-command-bundle',
        'steps': (
            _step('./scripts/trove-python', 'scripts/check.py', 'unit', timeout_seconds=1800),
            _step('./scripts/trove-python', 'scripts/check.py', 'contract', timeout_seconds=1800),
        ),
        'count_keys': ('checks_completed', 'failures'),
        'success_counts': {'checks_completed': 2, 'failures': 0},
    },
    'product-extras': {
        'command_id': 'run-u15-product-extras-v1',
        'artifact_type': 'release_product_extras_gate_redacted',
        'runner_kind': 'locked-command-bundle',
        'steps': _PRODUCT_EXTRAS_STEPS,
        'count_keys': ('checks_completed', 'failures'),
        'success_counts': {'checks_completed': len(_PRODUCT_EXTRAS_STEPS), 'failures': 0},
    },
    'macos-contracts': {
        'command_id': 'run-u15-macos-contracts-v1',
        'artifact_type': 'release_macos_contracts_gate_redacted',
        'runner_kind': 'locked-command-bundle',
        'required_system': 'Darwin',
        'steps': _MACOS_CONTRACT_STEPS,
        'count_keys': ('checks_completed', 'failures'),
        'success_counts': {'checks_completed': len(_MACOS_CONTRACT_STEPS), 'failures': 0},
    },
    'crash-recovery': {
        'command_id': 'run-u15-crash-recovery-v1',
        'artifact_type': 'release_crash_recovery_gate_redacted',
        'runner_kind': 'locked-command-bundle',
        'steps': _CRASH_RECOVERY_STEPS,
        'count_keys': ('checks_completed', 'failures'),
        'success_counts': {'checks_completed': len(_CRASH_RECOVERY_STEPS), 'failures': 0},
    },
    'complexity-1m': {
        'command_id': 'run-u15-complexity-1m-v1',
        'artifact_type': 'release_complexity_1m_gate_redacted',
        'runner_kind': 'locked-command-bundle',
        'steps': (
            _step(
                './scripts/trove-python',
                'scripts/benchmark_u10_scaling.py',
                '--rows',
                '1000000',
                '--watch-files',
                '1000000',
                '--logical-watch',
                timeout_seconds=3600,
            ),
            _step(
                './scripts/trove-python',
                'scripts/benchmark_u9_delta.py',
                '--rows',
                '1000000',
                timeout_seconds=3600,
            ),
            _step(
                './scripts/trove-python',
                'scripts/benchmark_u7_bounds.py',
                '--rows',
                '10000',
                '--rounds',
                '3',
                timeout_seconds=1800,
            ),
        ),
        'count_keys': ('checks_completed', 'failures'),
        'success_counts': {'checks_completed': 3, 'failures': 0},
    },
    'concurrency-10m': {
        'command_id': 'run-release-soak-v1',
        'artifact_type': 'release_soak_redacted',
        'runner_kind': 'isolated-soak',
        'count_keys': SOAK_COUNT_KEYS,
    },
    'soak-1h': {
        'command_id': 'run-release-soak-v1',
        'artifact_type': 'release_soak_redacted',
        'runner_kind': 'isolated-soak',
        'count_keys': SOAK_COUNT_KEYS,
    },
    'retrieval-quality-negative': {
        'command_id': 'verify-u15-retrieval-quality-negative-v1',
        'artifact_type': 'release_retrieval_quality_negative_gate_redacted',
        'runner_kind': 'verified-evidence',
        'source_artifact_types': ('retrieval_eval_gate_redacted',),
        'count_keys': ('checks_completed', 'failures', 'supporting_artifacts'),
        'success_counts': {'checks_completed': 8, 'failures': 0, 'supporting_artifacts': 1},
    },
    'exact-rewrite-latency': {
        'command_id': 'verify-u15-exact-rewrite-latency-v1',
        'artifact_type': 'release_exact_rewrite_latency_gate_redacted',
        'runner_kind': 'verified-evidence',
        'source_artifact_types': ('retrieval_eval_gate_redacted',),
        'count_keys': ('checks_completed', 'failures', 'supporting_artifacts'),
        'success_counts': {'checks_completed': 4, 'failures': 0, 'supporting_artifacts': 1},
    },
    'provenance': {
        'command_id': 'verify-u15-provenance-v1',
        'artifact_type': 'release_provenance_gate_redacted',
        'runner_kind': 'verified-receipt-set',
        'source_gate_ids': tuple(sorted({
            'privacy-security',
            'functional-minimal',
            'product-extras',
            'macos-contracts',
            'crash-recovery',
            'complexity-1m',
            'concurrency-10m',
            'soak-1h',
            'retrieval-quality-negative',
            'exact-rewrite-latency',
        })),
        'count_keys': ('checks_completed', 'failures', 'supporting_artifacts'),
        'success_counts': {'checks_completed': 10, 'failures': 0, 'supporting_artifacts': 10},
    },
}

U15_REQUIRED_GATE_IDS = frozenset(U15_GATE_CONTRACTS)


def _canonical_contract(gate_id: str) -> dict[str, Any]:
    contract = U15_GATE_CONTRACTS[gate_id]
    return {
        'schema_version': CONTRACT_SCHEMA_VERSION,
        'gate_id': gate_id,
        'command_id': contract['command_id'],
        'artifact_type': contract['artifact_type'],
        'runner_kind': contract['runner_kind'],
        'required_system': contract.get('required_system'),
        'steps': [
            {
                'argv': list(step['argv']),
                'timeout_seconds': step['timeout_seconds'],
            }
            for step in contract.get('steps', ())
        ],
        'source_artifact_types': list(contract.get('source_artifact_types', ())),
        'source_gate_ids': list(contract.get('source_gate_ids', ())),
        'count_keys': sorted(contract['count_keys']),
        'success_counts': dict(sorted((contract.get('success_counts') or {}).items())),
        'assurance': dict(sorted(RELEASE_ASSURANCE.items())),
    }


def release_gate_contract_sha256(gate_id: str) -> str:
    payload = json.dumps(
        _canonical_contract(gate_id),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def release_soak_contract_valid(artifact: Mapping[str, Any], gate_id: str) -> bool:
    """Independently recompute every release-soak success invariant."""

    if gate_id not in {'concurrency-10m', 'soak-1h'}:
        return False
    receipt = artifact.get('release_receipt')
    configuration = artifact.get('configuration')
    resources = artifact.get('resources')
    cleanup = artifact.get('cleanup')
    counts = artifact.get('counts')
    expected_duration = 600.0 if gate_id == 'concurrency-10m' else 3600.0
    expected_configuration = {
        'readers': 32,
        'writers': 1,
        'warmup_searches': 8,
        'duration_seconds': expected_duration,
        'sample_interval_seconds': 5.0,
        'writer_interval_seconds': 5.0,
        'search_mode': 'exact_off',
        'read_workload': 'exact+contacts+context+evidence',
        'process_isolated': True,
    }
    expected_cleanup = {
        'readers_joined': True,
        'writer_joined': True,
        'runtime_closed': True,
        'temp_vault_removed': True,
    }
    if (
        artifact.get('schema_version') != 1
        or artifact.get('artifact_type') != 'release_soak_redacted'
        or artifact.get('ok') is not True
        or not isinstance(receipt, dict)
        or receipt.get('gate_id') != gate_id
        or receipt.get('profile') != 'release'
        or receipt.get('command_id') != U15_GATE_CONTRACTS[gate_id]['command_id']
        or configuration != expected_configuration
        or cleanup != expected_cleanup
        or not isinstance(counts, dict)
        or counts != receipt.get('counts')
        or set(counts) != set(SOAK_COUNT_KEYS)
        or any(type(value) is not int or value < 0 for value in counts.values())
        or any(counts.get(key, 0) < 1 for key in (
            'searches_completed', 'contacts_completed', 'contexts_completed',
            'evidence_completed', 'writes_completed', 'writer_generation_contentions',
        ))
        or counts.get('writer_generation_contentions') != counts.get('writes_completed')
        or any(counts.get(key, 0) != 0 for key in (
            'runtime_overloaded', 'runtime_timeouts', 'lock_errors', 'unhandled_errors',
        ))
    ):
        return False

    elapsed = artifact.get('elapsed_seconds')
    if (
        type(elapsed) not in {int, float}
        or not math.isfinite(float(elapsed))
        or not expected_duration * 0.99 <= float(elapsed) <= expected_duration * 1.01
    ):
        return False
    if not isinstance(resources, dict) or set(resources) != {
        'sample_count',
        'required_sample_count',
        'full_window_count',
        'window_samples',
        'sustained_windows',
        'complete',
        'any_sustained_growth',
        'metrics',
        'expected_sample_count',
        'minimum_sample_count',
        'missed_deadlines',
        'max_sample_gap_seconds',
        'max_allowed_sample_gap_seconds',
        'heartbeat_window_count',
        'writer_heartbeat_required',
        'window_heartbeats',
    }:
        return False
    sample_count = resources.get('sample_count')
    expected_samples = 120 if gate_id == 'concurrency-10m' else 720
    minimum_samples = expected_samples - 1
    max_gap = resources.get('max_sample_gap_seconds')
    if (
        type(sample_count) is not int
        or not minimum_samples <= sample_count <= expected_samples
        or sample_count != counts.get('resource_samples')
        or resources.get('required_sample_count') != 20
        or resources.get('expected_sample_count') != expected_samples
        or resources.get('minimum_sample_count') != minimum_samples
        or resources.get('full_window_count') != sample_count // 5
        or resources.get('window_samples') != 5
        or resources.get('sustained_windows') != 3
        or resources.get('missed_deadlines') != 0
        or type(max_gap) not in {int, float}
        or not math.isfinite(float(max_gap))
        or not 0 <= float(max_gap) <= 7.5
        or resources.get('max_allowed_sample_gap_seconds') != 7.5
        or resources.get('heartbeat_window_count') != sample_count // 5
        or resources.get('writer_heartbeat_required') is not True
        or resources.get('complete') is not True
        or resources.get('any_sustained_growth') is not False
    ):
        return False
    metrics = resources.get('metrics')
    thresholds = {'rss_bytes': 64 * 1024 * 1024, 'fd_count': 8, 'thread_count': 4}
    if not isinstance(metrics, dict) or set(metrics) != set(thresholds):
        return False
    for name, threshold in thresholds.items():
        metric = metrics.get(name)
        if not isinstance(metric, dict) or set(metric) != {
            'first_window_median',
            'last_window_median',
            'last_minus_first',
            'peak',
            'sustained_growth_threshold',
            'sustained_growth',
        }:
            return False
        first = metric.get('first_window_median')
        last = metric.get('last_window_median')
        delta = metric.get('last_minus_first')
        peak = metric.get('peak')
        if any(
            type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0
            for value in (first, last, peak)
        ):
            return False
        if (
            type(delta) not in {int, float}
            or not math.isfinite(float(delta))
            or float(delta) != float(last) - float(first)
            or float(peak) < max(float(first), float(last))
            or metric.get('sustained_growth_threshold') != threshold
            or metric.get('sustained_growth') is not False
        ):
            return False
    window_heartbeats = resources.get('window_heartbeats')
    heartbeat_keys = {
        'searches_completed', 'contacts_completed', 'contexts_completed',
        'evidence_completed', 'writes_completed', 'complete',
    }
    if (
        not isinstance(window_heartbeats, list)
        or len(window_heartbeats) != sample_count // 5
        or not window_heartbeats
    ):
        return False
    heartbeat_totals = {key: 0 for key in heartbeat_keys - {'complete'}}
    for window in window_heartbeats:
        if not isinstance(window, dict) or set(window) != heartbeat_keys or window.get('complete') is not True:
            return False
        for key in heartbeat_totals:
            value = window.get(key)
            if type(value) is not int or value < 1:
                return False
            heartbeat_totals[key] += value
    if any(heartbeat_totals[key] > counts.get(key, 0) for key in heartbeat_totals):
        return False
    return True
