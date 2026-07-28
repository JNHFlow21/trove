#!/usr/bin/env python3
"""Build a fail-closed, redacted release closeout manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)

from release_gate_contracts import (  # noqa: E402
    CONTRACT_SCHEMA_VERSION,
    RELEASE_ASSURANCE,
    U15_GATE_CONTRACTS,
    U15_REQUIRED_GATE_IDS,
    release_gate_contract_sha256,
    release_soak_contract_valid,
)

from trove_core.search.evidence_provenance import (  # noqa: E402
    EvidenceProvenanceError,
    evidence_manifest_path,
    validate_artifact_provenance,
    verify_evidence_manifest,
    write_evidence_artifact,
)
from trove_core.search.eval_schema import RedactionError, validate_redacted_artifact  # noqa: E402


CLOSEOUT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
COMMAND_ID = 'build-release-closeout-manifest-v1'
V1_RELEASE_MANIFEST_SCHEMA_VERSION = 1
V1_RELEASE_PRIVACY = {
    'content_included': False,
    'queries_included': False,
    'citations_included': False,
    'account_identifiers_included': False,
    'private_paths_included': False,
    'raw_command_output_included': False,
    'secret_values_included': False,
    'vault_content_included': False,
}
CUTOVER_CHECK_IDS = frozenset({
    'agent_switch_mcp_clean',
    'artifact_inventory_clean',
    'automatic_recovery_verified',
    'bounded_drain_verified',
    'candidate_health_verified',
    'existing_vault_upgrade_without_reindex',
    'external_schedules_clean',
    'hash_mismatch_detected',
    'launch_agents_clean',
    'metadata_index_backup_verified',
    'rollback_vault_intact',
    'skill_hub_links_clean',
    'source_free_install_verified',
})
CUTOVER_COUNT_KEYS = frozenset({
    'failed_checks',
    'forbidden_artifact_members',
    'legacy_consumer_references',
    'vault_full_reindexes',
})
_HEX = frozenset('0123456789abcdef')
_IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
_BASENAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
_WINDOWS_ABSOLUTE = re.compile(r'^[A-Za-z]:[\\/]')
_COUNT_KEYS = frozenset({
    'searches_completed',
    'contacts_completed',
    'contexts_completed',
    'evidence_completed',
    'writes_completed',
    'writer_generation_contentions',
    'runtime_overloaded',
    'runtime_timeouts',
    'lock_errors',
    'unhandled_errors',
    'resource_samples',
    'artifacts',
    'artifact_bytes',
    'checks_completed',
    'tests_completed',
    'failures',
    'skips',
    'gates_passed',
    'supporting_artifacts',
    'required_gates',
    'passed_gates',
})
_TYPED_PRIVACY = {
    'synthetic_or_redacted': True,
    'redacted': True,
    'sensitive_content_included': False,
    'private_paths_included': False,
    'raw_command_output_included': False,
    'credential_values_included': False,
}
_FAILURE_COUNT_KEYS = frozenset({
    'failures',
    'runtime_overloaded',
    'runtime_timeouts',
    'lock_errors',
    'unhandled_errors',
})
_STANDARD_PRIVACY_FALSE = frozenset({
    'raw_fixture_identity_included',
    'raw_case_pack_included',
    'private_paths_included',
    'provider_names_included',
    'model_names_included',
})
_FORBIDDEN_FINAL_KEYS = frozenset({
    'query',
    'queries',
    'citation',
    'citations',
    'provider',
    'provider_name',
    'model',
    'model_name',
    'secret',
    'secret_value',
    'raw_command_output',
    'command',
    'path',
    'absolute_path',
})


class CloseoutManifestError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise CloseoutManifestError(code) from exc
    if not isinstance(value, dict):
        raise CloseoutManifestError(code)
    return value


def _verified_json_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        verify_evidence_manifest(path, required=True)
    except Exception as exc:
        raise CloseoutManifestError(code) from exc
    return _load_json_object(path, code=code)


def _is_commit_sha(value: Any) -> bool:
    text = str(value or '').lower()
    return len(text) in {40, 64} and all(character in _HEX for character in text)


def _safe_text(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise CloseoutManifestError(code)
    if any(character in value for character in ('/', '\\', '\n', '\r', '\x00')):
        raise CloseoutManifestError(code)
    return value


def _safe_identifier(value: Any, *, code: str) -> str:
    text = _safe_text(value, code=code)
    if not _IDENTIFIER.fullmatch(text):
        raise CloseoutManifestError(code)
    return text


def _safe_basename(path: Path) -> str:
    name = path.name
    if not _BASENAME.fullmatch(name):
        raise CloseoutManifestError('unsafe_artifact_basename')
    return name


def _validate_subject(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {'commit_sha', 'dirty'}:
        raise CloseoutManifestError('invalid_subject')
    commit_sha = str(value.get('commit_sha') or '').lower()
    if not _is_commit_sha(commit_sha):
        raise CloseoutManifestError('invalid_subject_sha')
    if value.get('dirty') is not False:
        raise CloseoutManifestError('dirty_subject')
    return {'commit_sha': commit_sha, 'dirty': False}


def _validate_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise CloseoutManifestError('invalid_receipt_counts')
    counts: dict[str, int] = {}
    for key, count in value.items():
        if key not in _COUNT_KEYS or type(count) is not int or count < 0:
            raise CloseoutManifestError('invalid_receipt_counts')
        counts[key] = count
    return dict(sorted(counts.items()))


def _validate_platform(value: Any) -> dict[str, str]:
    required = {'system', 'release', 'machine', 'python_version'}
    if not isinstance(value, dict) or set(value) != required:
        raise CloseoutManifestError('invalid_receipt_platform')
    return {
        key: _safe_text(value[key], code='invalid_receipt_platform')
        for key in sorted(required)
    }


def _validate_typed_receipt(value: Any) -> dict[str, Any]:
    required = {
        'schema_version', 'subject', 'fixture', 'privacy', 'counts',
        'platform', 'version', 'command_id', 'gate_id', 'profile',
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CloseoutManifestError('invalid_release_receipt')
    if value.get('schema_version') != RECEIPT_SCHEMA_VERSION:
        raise CloseoutManifestError('unsupported_release_receipt')
    fixture = value.get('fixture')
    if fixture != {'kind': 'synthetic_or_redacted'}:
        raise CloseoutManifestError('artifact_not_synthetic_or_redacted')
    if value.get('privacy') != _TYPED_PRIVACY:
        raise CloseoutManifestError('artifact_privacy_mismatch')
    if value.get('profile') not in {'release', 'custom', 'test-only'}:
        raise CloseoutManifestError('invalid_gate_profile')
    return {
        'subject': _validate_subject(value.get('subject')),
        'counts': _validate_counts(value.get('counts')),
        'platform': _validate_platform(value.get('platform')),
        'version': _safe_identifier(value.get('version'), code='invalid_receipt_version'),
        'command_id': _safe_identifier(value.get('command_id'), code='invalid_command_id'),
        'gate_id': _safe_identifier(value.get('gate_id'), code='invalid_gate_id'),
        'profile': _safe_identifier(value.get('profile'), code='invalid_gate_profile'),
    }


def _validate_source_evidence(value: Any, *, gate_id: str, contract: Mapping[str, Any]) -> None:
    if not isinstance(value, list):
        raise CloseoutManifestError('invalid_release_gate_execution')
    kind = contract['runner_kind']
    if kind == 'locked-command-bundle':
        expected_count = 0
    elif kind == 'verified-evidence':
        expected_count = 1
    elif kind == 'verified-receipt-set':
        expected_count = len(contract['source_gate_ids'])
    else:
        raise CloseoutManifestError('invalid_release_gate_execution')
    if len(value) != expected_count:
        raise CloseoutManifestError('invalid_release_gate_execution')

    seen_files: set[str] = set()
    seen_gates: set[str] = set()
    for item in value:
        required = {'artifact_file', 'artifact_sha256', 'artifact_bytes', 'artifact_type'}
        if kind == 'verified-receipt-set':
            required.add('gate_id')
        if not isinstance(item, dict) or set(item) != required:
            raise CloseoutManifestError('invalid_release_gate_execution')
        artifact_file = _safe_basename(Path(str(item.get('artifact_file') or '')))
        if artifact_file in seen_files:
            raise CloseoutManifestError('invalid_release_gate_execution')
        seen_files.add(artifact_file)
        artifact_sha = str(item.get('artifact_sha256') or '').lower()
        if len(artifact_sha) != 64 or not _is_commit_sha(artifact_sha):
            raise CloseoutManifestError('invalid_release_gate_execution')
        if type(item.get('artifact_bytes')) is not int or item['artifact_bytes'] <= 0:
            raise CloseoutManifestError('invalid_release_gate_execution')
        artifact_type = _safe_identifier(
            item.get('artifact_type'),
            code='invalid_release_gate_execution',
        )
        if 'redacted' not in artifact_type:
            raise CloseoutManifestError('invalid_release_gate_execution')
        if kind == 'verified-evidence' and artifact_type not in contract['source_artifact_types']:
            raise CloseoutManifestError('invalid_release_gate_execution')
        if kind == 'verified-receipt-set':
            source_gate = _safe_identifier(item.get('gate_id'), code='invalid_release_gate_execution')
            if source_gate in seen_gates or source_gate not in contract['source_gate_ids']:
                raise CloseoutManifestError('invalid_release_gate_execution')
            if artifact_type != U15_GATE_CONTRACTS[source_gate]['artifact_type']:
                raise CloseoutManifestError('invalid_release_gate_execution')
            seen_gates.add(source_gate)
    if kind == 'verified-receipt-set' and seen_gates != set(contract['source_gate_ids']):
        raise CloseoutManifestError('invalid_release_gate_execution')


def _validate_locked_gate_contract(
    artifact: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    gate_id = str(receipt['gate_id'])
    contract = U15_GATE_CONTRACTS.get(gate_id)
    if contract is None:
        raise CloseoutManifestError('invalid_release_gate_contract')
    counts = receipt['counts']
    if (
        artifact.get('artifact_type') != contract['artifact_type']
        or receipt.get('command_id') != contract['command_id']
        or set(counts) != set(contract['count_keys'])
        or artifact.get('privacy') != _TYPED_PRIVACY
        or artifact.get('assurance') != RELEASE_ASSURANCE
    ):
        raise CloseoutManifestError('invalid_release_gate_contract')
    if contract.get('success_counts') is not None and counts != contract['success_counts']:
        raise CloseoutManifestError('invalid_release_gate_contract')

    if contract['runner_kind'] == 'isolated-soak':
        return contract

    execution = artifact.get('gate_execution')
    expected_execution_keys = {
        'contract_schema_version',
        'contract_sha256',
        'runner_kind',
        'checks_planned',
        'checks_completed',
        'source_artifacts',
    }
    if not isinstance(execution, dict) or set(execution) != expected_execution_keys:
        raise CloseoutManifestError('invalid_release_gate_execution')
    if (
        execution.get('contract_schema_version') != CONTRACT_SCHEMA_VERSION
        or execution.get('contract_sha256') != release_gate_contract_sha256(gate_id)
        or execution.get('runner_kind') != contract['runner_kind']
        or execution.get('checks_planned') != contract['success_counts']['checks_completed']
        or execution.get('checks_completed') != counts.get('checks_completed')
        or execution.get('source_artifacts') != counts.get('supporting_artifacts', 0)
    ):
        raise CloseoutManifestError('invalid_release_gate_execution')
    _validate_source_evidence(artifact.get('source_evidence'), gate_id=gate_id, contract=contract)
    verified_gate_ids = artifact.get('verified_gate_ids')
    expected_verified = list(contract.get('source_gate_ids', ()))
    if verified_gate_ids != expected_verified:
        raise CloseoutManifestError('invalid_release_gate_execution')
    return contract


def _validate_standard_provenance(value: Any) -> dict[str, Any]:
    try:
        validate_artifact_provenance(value, release=True)
    except EvidenceProvenanceError as exc:
        raise CloseoutManifestError('invalid_standard_provenance') from exc
    if value.get('fixture') != {
        'kind': 'synthetic_or_redacted',
        'sha256': value.get('fixture', {}).get('sha256'),
    }:
        raise CloseoutManifestError('artifact_not_synthetic_or_redacted')
    privacy = value.get('privacy')
    if not isinstance(privacy, dict):
        raise CloseoutManifestError('artifact_privacy_mismatch')
    if set(privacy) != set(_STANDARD_PRIVACY_FALSE):
        raise CloseoutManifestError('artifact_privacy_mismatch')
    for key in _STANDARD_PRIVACY_FALSE:
        if privacy[key] is not False:
            raise CloseoutManifestError('artifact_privacy_mismatch')
    platform_value = value.get('platform') or {}
    platform_receipt = _validate_platform({
        key: platform_value.get(key)
        for key in ('system', 'release', 'machine', 'python_version')
    })
    return {
        'subject': _validate_subject(value.get('git')),
        'counts': {},
        'platform': platform_receipt,
        'version': None,
        'command_id': None,
    }


def _load_verified_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise CloseoutManifestError('missing_artifact')
    try:
        sidecar = verify_evidence_manifest(path, required=True)
        artifact = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise CloseoutManifestError('artifact_sidecar_invalid') from exc
    if not isinstance(sidecar, dict) or not isinstance(artifact, dict):
        raise CloseoutManifestError('artifact_sidecar_invalid')
    try:
        validate_redacted_artifact(artifact)
    except RedactionError as exc:
        raise CloseoutManifestError('artifact_not_strictly_redacted') from exc
    def validate_privacy_flags(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).endswith('_included') and nested is not False:
                    raise CloseoutManifestError('artifact_privacy_mismatch')
                validate_privacy_flags(nested)
        elif isinstance(value, list):
            for nested in value:
                validate_privacy_flags(nested)
    validate_privacy_flags(artifact)
    if sidecar.get('privacy') != {
        'artifact_content_included': False,
        'private_paths_included': False,
        'token_values_included': False,
    }:
        raise CloseoutManifestError('sidecar_privacy_mismatch')
    artifact_type = artifact.get('artifact_type')
    if not isinstance(artifact_type, str) or 'redacted' not in artifact_type:
        raise CloseoutManifestError('artifact_not_redacted')
    return artifact, sidecar


def _artifact_receipt(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if 'release_receipt' in artifact:
        if artifact.get('ok') is not True:
            raise CloseoutManifestError('artifact_gate_failed')
        receipt = _validate_typed_receipt(artifact.get('release_receipt'))
        if any(receipt['counts'].get(key, 0) != 0 for key in _FAILURE_COUNT_KEYS):
            raise CloseoutManifestError('artifact_gate_failed')
        if receipt['profile'] == 'release':
            _validate_locked_gate_contract(artifact, receipt)
            if receipt['gate_id'] == 'provenance':
                receipt['_verified_sources'] = [dict(item) for item in artifact['source_evidence']]
        if receipt['profile'] == 'release' and receipt['gate_id'] in {'concurrency-10m', 'soak-1h'}:
            if not release_soak_contract_valid(artifact, receipt['gate_id']):
                raise CloseoutManifestError('invalid_soak_gate_receipt')
        if receipt['profile'] == 'release':
            receipt['role'] = 'gate'
        elif receipt['profile'] == 'custom' and receipt['gate_id'] not in U15_REQUIRED_GATE_IDS:
            receipt['role'] = 'gate'
        else:
            receipt['role'] = 'test-only'
        return receipt
    if 'provenance' in artifact:
        if artifact.get('ok') is False or artifact.get('complete') is False:
            raise CloseoutManifestError('supporting_artifact_failed')
        receipt = _validate_standard_provenance(artifact.get('provenance'))
        receipt['role'] = 'supporting'
        return receipt
    raise CloseoutManifestError('missing_release_receipt')


def _assert_no_forbidden_final_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_FINAL_KEYS:
                raise CloseoutManifestError('forbidden_final_field')
            _assert_no_forbidden_final_fields(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_final_fields(nested)
        return
    if isinstance(value, str):
        lowered = value.lower()
        if value.startswith(('/', '~/', '\\')) or lowered.startswith('file:') or _WINDOWS_ABSOLUTE.match(value):
            raise CloseoutManifestError('absolute_path_in_final_manifest')


def build_closeout_manifest(
    artifact_paths: list[str | Path],
    *,
    required_gate_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    if not artifact_paths:
        raise CloseoutManifestError('missing_artifact')
    required = set(U15_REQUIRED_GATE_IDS if required_gate_ids is None else required_gate_ids)
    if not required:
        raise CloseoutManifestError('missing_required_gate_set')
    required = {_safe_identifier(value, code='invalid_gate_id') for value in required}
    inputs: list[dict[str, Any]] = []
    subject_sha: str | None = None
    basenames: set[str] = set()
    total_bytes = 0
    gate_count = 0
    supporting_count = 0
    passed_gate_ids: set[str] = set()
    gate_inputs: dict[str, dict[str, Any]] = {}
    provenance_sources: list[dict[str, Any]] | None = None
    for raw_path in artifact_paths:
        path = Path(raw_path).expanduser()
        name = _safe_basename(path)
        if name in basenames:
            raise CloseoutManifestError('duplicate_artifact_basename')
        basenames.add(name)
        artifact, sidecar = _load_verified_artifact(path)
        receipt = _artifact_receipt(artifact)
        if receipt['role'] == 'gate':
            if receipt['gate_id'] in gate_inputs and receipt['gate_id'] in U15_REQUIRED_GATE_IDS:
                raise CloseoutManifestError('duplicate_gate_receipt')
            gate_count += 1
            passed_gate_ids.add(receipt['gate_id'])
        else:
            supporting_count += 1
        current_sha = receipt['subject']['commit_sha']
        if subject_sha is None:
            subject_sha = current_sha
        elif subject_sha != current_sha:
            raise CloseoutManifestError('subject_sha_mismatch')

        artifact_bytes = int(sidecar['artifact_bytes'])
        total_bytes += artifact_bytes
        item: dict[str, Any] = {
            'artifact_file': name,
            'artifact_sha256': str(sidecar['artifact_sha256']),
            'artifact_bytes': artifact_bytes,
            'platform': receipt['platform'],
            'role': receipt['role'],
        }
        if receipt.get('gate_id') is not None:
            item['gate_id'] = receipt['gate_id']
        if receipt['counts']:
            item['counts'] = receipt['counts']
        if receipt['version'] is not None:
            item['version'] = receipt['version']
        if receipt['command_id'] is not None:
            item['command_id'] = receipt['command_id']
        inputs.append(item)
        if receipt['role'] == 'gate':
            gate_inputs[receipt['gate_id']] = {
                'artifact_file': name,
                'artifact_sha256': str(sidecar['artifact_sha256']),
                'artifact_bytes': artifact_bytes,
                'artifact_type': str(artifact['artifact_type']),
                'gate_id': receipt['gate_id'],
            }
            if receipt['gate_id'] == 'provenance':
                provenance_sources = receipt.get('_verified_sources')

    assert subject_sha is not None
    missing = required - passed_gate_ids
    if missing:
        raise CloseoutManifestError('missing_required_gate_receipts')
    if 'provenance' in passed_gate_ids:
        expected_source_gates = set(U15_GATE_CONTRACTS['provenance']['source_gate_ids'])
        if not isinstance(provenance_sources, list):
            raise CloseoutManifestError('provenance_receipt_not_bound')
        claimed_by_gate = {
            str(item.get('gate_id')): item
            for item in provenance_sources
            if isinstance(item, dict)
        }
        if set(claimed_by_gate) != expected_source_gates:
            raise CloseoutManifestError('provenance_receipt_not_bound')
        for gate_id in expected_source_gates:
            if gate_inputs.get(gate_id) != claimed_by_gate[gate_id]:
                raise CloseoutManifestError('provenance_receipt_not_bound')
    inputs.sort(key=lambda item: item['artifact_file'])
    aggregate_counts = {
        'artifacts': len(inputs),
        'artifact_bytes': total_bytes,
        'gates_passed': gate_count,
        'supporting_artifacts': supporting_count,
        'required_gates': len(required),
        'passed_gates': len(passed_gate_ids & required),
    }
    locked_u15 = required == set(U15_REQUIRED_GATE_IDS)
    manifest = {
        'schema_version': CLOSEOUT_SCHEMA_VERSION,
        'artifact_type': (
            'release_closeout_manifest_redacted'
            if locked_u15
            else 'release_closeout_manifest_custom_redacted'
        ),
        'ok': True,
        'subject_sha': subject_sha,
        'required_gate_ids': sorted(required),
        'inputs': inputs,
        'counts': aggregate_counts,
        'release_receipt': {
            'schema_version': RECEIPT_SCHEMA_VERSION,
            'subject': {'commit_sha': subject_sha, 'dirty': False},
            'fixture': {'kind': 'synthetic_or_redacted'},
            'privacy': dict(_TYPED_PRIVACY),
            'counts': aggregate_counts,
            'platform': inputs[0]['platform'],
            'version': '1',
            'command_id': COMMAND_ID,
            'gate_id': 'u15-closeout',
            'profile': 'release' if locked_u15 else 'custom',
        },
        'privacy': dict(_TYPED_PRIVACY),
        'assurance': {
            **RELEASE_ASSURANCE,
            'contract_locked': locked_u15,
        },
    }
    _assert_no_forbidden_final_fields(manifest)
    return manifest


def build_cutover_acceptance(
    distribution_manifest_path: str | Path,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a content-free receipt for the real install/upgrade/cutover drill."""

    from verify_distribution import verify_distribution

    distribution_path = Path(distribution_manifest_path).expanduser()
    try:
        verify_distribution(distribution_path)
    except Exception as exc:
        raise CloseoutManifestError('distribution_verification_failed') from exc
    distribution = _load_json_object(distribution_path, code='invalid_distribution_manifest')
    if distribution.get('source_dirty') is not False:
        raise CloseoutManifestError('dirty_distribution')
    if not isinstance(controls, Mapping) or set(controls) != {'checks', 'counts'}:
        raise CloseoutManifestError('invalid_cutover_controls')
    checks = controls.get('checks')
    counts = controls.get('counts')
    if (
        not isinstance(checks, Mapping)
        or set(checks) != CUTOVER_CHECK_IDS
        or any(checks[key] is not True for key in CUTOVER_CHECK_IDS)
    ):
        raise CloseoutManifestError('cutover_check_failed')
    if (
        not isinstance(counts, Mapping)
        or set(counts) != CUTOVER_COUNT_KEYS
        or any(type(counts[key]) is not int or counts[key] != 0 for key in CUTOVER_COUNT_KEYS)
    ):
        raise CloseoutManifestError('cutover_count_failed')
    report = {
        'schema_version': 1,
        'artifact_type': 'trove_v1_cutover_acceptance_redacted',
        'ok': True,
        'subject_sha': distribution['source_git_sha'],
        'distribution_set_sha256': distribution['distribution_set_sha256'],
        'checks': {key: True for key in sorted(CUTOVER_CHECK_IDS)},
        'counts': {key: 0 for key in sorted(CUTOVER_COUNT_KEYS)},
        'privacy': dict(V1_RELEASE_PRIVACY),
    }
    validate_cutover_acceptance(report)
    return report


def validate_cutover_acceptance(report: Mapping[str, Any]) -> None:
    required = {
        'schema_version', 'artifact_type', 'ok', 'subject_sha',
        'distribution_set_sha256', 'checks', 'counts', 'privacy',
    }
    if set(report) != required:
        raise CloseoutManifestError('invalid_cutover_acceptance')
    if (
        report.get('schema_version') != 1
        or report.get('artifact_type') != 'trove_v1_cutover_acceptance_redacted'
        or report.get('ok') is not True
        or not _is_commit_sha(report.get('subject_sha'))
        or len(str(report.get('subject_sha'))) != 40
        or not _is_commit_sha(report.get('distribution_set_sha256'))
        or len(str(report.get('distribution_set_sha256'))) != 64
        or report.get('checks') != {key: True for key in sorted(CUTOVER_CHECK_IDS)}
        or report.get('counts') != {key: 0 for key in sorted(CUTOVER_COUNT_KEYS)}
        or report.get('privacy') != V1_RELEASE_PRIVACY
    ):
        raise CloseoutManifestError('invalid_cutover_acceptance')
    _assert_no_forbidden_final_fields(report)


def _validate_final_closeout(report: Mapping[str, Any], subject_sha: str) -> None:
    required = set(U15_REQUIRED_GATE_IDS)
    counts = report.get('counts')
    if (
        report.get('schema_version') != CLOSEOUT_SCHEMA_VERSION
        or report.get('artifact_type') != 'release_closeout_manifest_redacted'
        or report.get('ok') is not True
        or report.get('subject_sha') != subject_sha
        or set(report.get('required_gate_ids') or ()) != required
        or not isinstance(counts, Mapping)
        or counts.get('required_gates') != len(required)
        or counts.get('passed_gates') != len(required)
        or report.get('privacy') != _TYPED_PRIVACY
        or report.get('assurance') != {**RELEASE_ASSURANCE, 'contract_locked': True}
    ):
        raise CloseoutManifestError('invalid_final_closeout')
    _assert_no_forbidden_final_fields(report)


def build_v1_release_manifest(
    *,
    distribution_manifest_path: str | Path,
    test_summary_path: str | Path,
    perf_summary_path: str | Path,
    agent_acceptance_path: str | Path,
    real_vault_acceptance_path: str | Path,
    cutover_acceptance_path: str | Path,
) -> dict[str, Any]:
    """Bind one clean frozen distribution to all required redacted evidence."""

    from benchmark_agent_runtime import validate_benchmark_artifact
    from release_gate_contracts import agent_runtime_budget_contract_valid
    from run_agent_product_acceptance import validate_report as validate_agent_acceptance
    from run_real_vault_acceptance import validate_report as validate_real_vault_acceptance
    from verify_distribution import verify_distribution

    distribution_path = Path(distribution_manifest_path).expanduser()
    try:
        verify_distribution(distribution_path)
    except Exception as exc:
        raise CloseoutManifestError('distribution_verification_failed') from exc
    distribution = _load_json_object(distribution_path, code='invalid_distribution_manifest')
    subject_sha = str(distribution.get('source_git_sha') or '')
    if distribution.get('source_dirty') is not False:
        raise CloseoutManifestError('dirty_distribution')

    test_path = Path(test_summary_path).expanduser()
    test_summary = _verified_json_object(test_path, code='invalid_test_summary')
    _validate_final_closeout(test_summary, subject_sha)

    perf_path = Path(perf_summary_path).expanduser()
    perf_summary = _load_json_object(perf_path, code='invalid_perf_summary')
    try:
        if not agent_runtime_budget_contract_valid(perf_summary):
            raise ValueError
        baseline = perf_summary['current_fixture_baseline']
        validate_benchmark_artifact(baseline)
    except Exception as exc:
        raise CloseoutManifestError('invalid_perf_summary') from exc
    if baseline.get('git_sha') != subject_sha:
        raise CloseoutManifestError('perf_subject_sha_mismatch')

    agent_path = Path(agent_acceptance_path).expanduser()
    agent_acceptance = _load_json_object(agent_path, code='invalid_agent_acceptance')
    try:
        validate_agent_acceptance(agent_acceptance)
    except Exception as exc:
        raise CloseoutManifestError('invalid_agent_acceptance') from exc
    agent_summary = agent_acceptance.get('summary') or {}
    if (
        agent_acceptance.get('ok') is not True
        or agent_summary.get('clients') != 2
        or agent_summary.get('tasks') != 12
        or agent_summary.get('tasks_succeeded') != 12
        or agent_summary.get('wrong_tool_calls') != 0
        or agent_summary.get('operator_interventions') != 0
        or int(agent_summary.get('citation_count') or 0) < 12
    ):
        raise CloseoutManifestError('agent_acceptance_failed')

    real_path = Path(real_vault_acceptance_path).expanduser()
    real_acceptance = _load_json_object(real_path, code='invalid_real_vault_acceptance')
    try:
        validate_real_vault_acceptance(real_acceptance)
    except Exception as exc:
        raise CloseoutManifestError('invalid_real_vault_acceptance') from exc
    if real_acceptance.get('ok') is not True:
        raise CloseoutManifestError('real_vault_acceptance_failed')

    cutover_path = Path(cutover_acceptance_path).expanduser()
    cutover = _verified_json_object(cutover_path, code='invalid_cutover_acceptance')
    validate_cutover_acceptance(cutover)
    if (
        cutover.get('subject_sha') != subject_sha
        or cutover.get('distribution_set_sha256') != distribution['distribution_set_sha256']
    ):
        raise CloseoutManifestError('cutover_candidate_mismatch')

    evidence_paths = {
        'agent_acceptance_sha256': agent_path,
        'cutover_acceptance_sha256': cutover_path,
        'perf_summary_sha256': perf_path,
        'real_vault_acceptance_sha256': real_path,
        'test_summary_sha256': test_path,
    }
    manifest = {
        'schema_version': V1_RELEASE_MANIFEST_SCHEMA_VERSION,
        'artifact_type': 'trove_v1_release_manifest_redacted',
        'version': '1.0.0',
        'release_status': 'frozen',
        'source_git_sha': subject_sha,
        'distribution_manifest_sha256': _sha256_file(distribution_path),
        'distribution_set_sha256': distribution['distribution_set_sha256'],
        'runtime_artifact_sha256': distribution['runtime']['sha256'],
        'provider_artifact_sha256': distribution['provider']['sha256'],
        'runtime_build_hash': distribution['runtime_build_hash'],
        'catalog_hash': distribution['catalog_hash'],
        'provider_package_hash': distribution['provider_package_hash'],
        'evidence_sha256': {
            key: _sha256_file(path)
            for key, path in sorted(evidence_paths.items())
        },
        'release_gate_summary': {
            'required_gates': test_summary['counts']['required_gates'],
            'passed_gates': test_summary['counts']['passed_gates'],
            'cutover_checks': len(CUTOVER_CHECK_IDS),
            'cutover_failures': cutover['counts']['failed_checks'],
        },
        'privacy': dict(V1_RELEASE_PRIVACY),
    }
    validate_v1_release_manifest(manifest)
    return manifest


def validate_v1_release_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        'schema_version', 'artifact_type', 'version', 'release_status',
        'source_git_sha', 'distribution_manifest_sha256',
        'distribution_set_sha256', 'runtime_artifact_sha256',
        'provider_artifact_sha256', 'runtime_build_hash', 'catalog_hash',
        'provider_package_hash', 'evidence_sha256',
        'release_gate_summary', 'privacy',
    }
    if set(manifest) != required:
        raise CloseoutManifestError('invalid_v1_release_manifest')
    if (
        manifest.get('schema_version') != V1_RELEASE_MANIFEST_SCHEMA_VERSION
        or manifest.get('artifact_type') != 'trove_v1_release_manifest_redacted'
        or manifest.get('version') != '1.0.0'
        or manifest.get('release_status') != 'frozen'
        or not _is_commit_sha(manifest.get('source_git_sha'))
        or len(str(manifest.get('source_git_sha'))) != 40
        or manifest.get('privacy') != V1_RELEASE_PRIVACY
    ):
        raise CloseoutManifestError('invalid_v1_release_manifest')
    for key in (
        'distribution_manifest_sha256', 'distribution_set_sha256',
        'runtime_artifact_sha256', 'provider_artifact_sha256',
        'runtime_build_hash', 'catalog_hash', 'provider_package_hash',
    ):
        value = manifest.get(key)
        if not _is_commit_sha(value) or len(str(value)) != 64:
            raise CloseoutManifestError('invalid_v1_release_manifest')
    evidence = manifest.get('evidence_sha256')
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != {
            'agent_acceptance_sha256', 'cutover_acceptance_sha256',
            'perf_summary_sha256', 'real_vault_acceptance_sha256',
            'test_summary_sha256',
        }
        or any(not _is_commit_sha(value) or len(str(value)) != 64 for value in evidence.values())
    ):
        raise CloseoutManifestError('invalid_v1_release_manifest')
    gate = manifest.get('release_gate_summary')
    if (
        not isinstance(gate, Mapping)
        or set(gate) != {
            'required_gates', 'passed_gates', 'cutover_checks', 'cutover_failures',
        }
        or gate.get('required_gates') != len(U15_REQUIRED_GATE_IDS)
        or gate.get('passed_gates') != len(U15_REQUIRED_GATE_IDS)
        or gate.get('cutover_checks') != len(CUTOVER_CHECK_IDS)
        or gate.get('cutover_failures') != 0
    ):
        raise CloseoutManifestError('invalid_v1_release_manifest')
    _assert_no_forbidden_final_fields(manifest)


def _read_bounded_json_fd(fd: int, *, limit: int = 64 * 1024) -> dict[str, Any]:
    payload = bytearray()
    while len(payload) <= limit:
        chunk = os.read(fd, min(8192, limit + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > limit:
        raise CloseoutManifestError('cutover_input_too_large')
    try:
        value = json.loads(payload.decode('utf-8')) if payload else None
    except Exception as exc:
        raise CloseoutManifestError('invalid_cutover_controls') from exc
    if not isinstance(value, dict):
        raise CloseoutManifestError('invalid_cutover_controls')
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.release-manifest-', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Build a verified redacted release closeout manifest.')
    parser.add_argument('artifacts', nargs='*', help='Redacted artifacts with independent .manifest.json sidecars.')
    parser.add_argument('--out', required=True, help='Final redacted manifest path.')
    parser.add_argument(
        '--require-gate',
        action='append',
        default=[],
        help='Required typed release gate id. Repeat; defaults to the locked U15 set.',
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--build-v1-release-manifest', action='store_true',
        help='Bind the frozen distribution to automated test, perf, acceptance, and cutover evidence.',
    )
    mode.add_argument(
        '--build-cutover-acceptance', action='store_true',
        help='Build a redacted real install/upgrade/cutover receipt from a bounded FD.',
    )
    parser.add_argument('--distribution-manifest')
    parser.add_argument('--test-summary')
    parser.add_argument('--perf-summary')
    parser.add_argument('--agent-acceptance')
    parser.add_argument('--real-vault-acceptance')
    parser.add_argument('--cutover-acceptance')
    parser.add_argument('--cutover-input-fd', type=int)
    args = parser.parse_args(argv)
    try:
        out = Path(args.out).expanduser()
        if args.build_v1_release_manifest:
            required_paths = (
                args.distribution_manifest, args.test_summary, args.perf_summary,
                args.agent_acceptance, args.real_vault_acceptance,
                args.cutover_acceptance,
            )
            if args.artifacts or args.require_gate or not all(required_paths) or args.cutover_input_fd is not None:
                raise CloseoutManifestError('invalid_v1_release_arguments')
            manifest = build_v1_release_manifest(
                distribution_manifest_path=args.distribution_manifest,
                test_summary_path=args.test_summary,
                perf_summary_path=args.perf_summary,
                agent_acceptance_path=args.agent_acceptance,
                real_vault_acceptance_path=args.real_vault_acceptance,
                cutover_acceptance_path=args.cutover_acceptance,
            )
            _atomic_write_json(out, manifest)
            print(json.dumps({
                'ok': True,
                'artifact_file': out.name,
                'release_status': manifest['release_status'],
                'private_paths_printed': False,
                'raw_content_printed': False,
            }, sort_keys=True))
            return 0
        if args.build_cutover_acceptance:
            if (
                args.artifacts or args.require_gate or not args.distribution_manifest
                or args.cutover_input_fd is None
                or any((
                    args.test_summary, args.perf_summary, args.agent_acceptance,
                    args.real_vault_acceptance, args.cutover_acceptance,
                ))
            ):
                raise CloseoutManifestError('invalid_cutover_arguments')
            manifest = build_cutover_acceptance(
                args.distribution_manifest,
                _read_bounded_json_fd(args.cutover_input_fd),
            )
            write_evidence_artifact(manifest, out)
            print(json.dumps({
                'ok': True,
                'artifact_file': out.name,
                'manifest_file': evidence_manifest_path(out).name,
                'private_paths_printed': False,
                'raw_content_printed': False,
            }, sort_keys=True))
            return 0
        if any((
            args.distribution_manifest, args.test_summary, args.perf_summary,
            args.agent_acceptance, args.real_vault_acceptance, args.cutover_acceptance,
        )) or args.cutover_input_fd is not None:
            raise CloseoutManifestError('invalid_closeout_arguments')
        manifest = build_closeout_manifest(
            args.artifacts,
            required_gate_ids=set(args.require_gate) if args.require_gate else None,
        )
        output_resolved = out.resolve()
        protected = {
            candidate
            for path in args.artifacts
            for candidate in (
                Path(path).expanduser().resolve(),
                evidence_manifest_path(Path(path).expanduser()).resolve(),
            )
        }
        output_files = {output_resolved, evidence_manifest_path(out).resolve()}
        if output_files & protected:
            raise CloseoutManifestError('output_collides_with_input')
        write_evidence_artifact(manifest, out)
    except CloseoutManifestError as exc:
        print(json.dumps({'ok': False, 'error_code': exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({'ok': False, 'error_code': 'closeout_artifact_write_failed'}, sort_keys=True))
        return 2
    output = {
        'ok': True,
        'artifact_file': out.name,
        'manifest_file': evidence_manifest_path(out).name,
        'counts': manifest['counts'],
        'private_paths_printed': False,
        'raw_content_printed': False,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
