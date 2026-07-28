#!/usr/bin/env python3
"""Run one immutable local-runtime gate and emit a content-free typed receipt."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)

from release_gate_contracts import (  # noqa: E402
    CONTRACT_SCHEMA_VERSION,
    RELEASE_ASSURANCE,
    U15_GATE_CONTRACTS,
    release_gate_contract_sha256,
    release_soak_contract_valid,
)
from trove_core.search.eval_schema import RedactionError, validate_redacted_artifact  # noqa: E402
from trove_core.search.evidence_provenance import (  # noqa: E402
    EvidenceProvenanceError,
    collect_git_provenance,
    evidence_manifest_path,
    verify_evidence_manifest,
    write_evidence_artifact,
)


GATE_ARTIFACT_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
_HEX = frozenset('0123456789abcdef')
_BASENAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
_TYPED_PRIVACY = {
    'synthetic_or_redacted': True,
    'redacted': True,
    'sensitive_content_included': False,
    'private_paths_included': False,
    'raw_command_output_included': False,
    'credential_values_included': False,
}
_SIDECAR_PRIVACY = {
    'artifact_content_included': False,
    'private_paths_included': False,
    'token_values_included': False,
}
_FAILURE_COUNT_KEYS = frozenset({
    'failures',
    'runtime_overloaded',
    'runtime_timeouts',
    'lock_errors',
    'unhandled_errors',
})
RUNTIME_CONTRACT = {
    'protocol': 'trove/1',
    'product_surface': 'mcp_cli_skills',
    'legacy_surface_present': False,
    'network_listener_present': False,
}


class ReleaseGateRunnerError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _is_sha(value: Any) -> bool:
    text = str(value or '').lower()
    return len(text) in {40, 64} and all(character in _HEX for character in text)


def _normalized_subject(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    sha = str(source.get('commit_sha') or '').lower()
    dirty = source.get('dirty')
    return {
        'commit_sha': sha if _is_sha(sha) else '0' * 40,
        'dirty': dirty is not False,
    }


def _subject_is_clean(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {'commit_sha', 'dirty'}
        and _is_sha(value.get('commit_sha'))
        and value.get('dirty') is False
    )


def _platform_receipt() -> dict[str, str]:
    return {
        'system': platform.system() or 'unknown',
        'release': platform.release() or 'unknown',
        'machine': platform.machine() or 'unknown',
        'python_version': platform.python_version(),
    }


def _version() -> str:
    for distribution in ('trove-runtime',):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return 'unknown'


def _validate_all_privacy_flags(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).endswith('_included') and nested is not False:
                raise ReleaseGateRunnerError('source_privacy_mismatch')
            _validate_all_privacy_flags(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_all_privacy_flags(nested)


def _load_verified_source(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not path.exists() or not path.is_file() or not _BASENAME.fullmatch(path.name):
        raise ReleaseGateRunnerError('source_artifact_missing')
    try:
        sidecar = verify_evidence_manifest(path, required=True)
        artifact = json.loads(path.read_text(encoding='utf-8'))
        validate_redacted_artifact(artifact)
    except (OSError, json.JSONDecodeError, EvidenceProvenanceError, RedactionError) as exc:
        raise ReleaseGateRunnerError('source_artifact_invalid') from exc
    if not isinstance(artifact, dict) or not isinstance(sidecar, dict):
        raise ReleaseGateRunnerError('source_artifact_invalid')
    if sidecar.get('privacy') != _SIDECAR_PRIVACY:
        raise ReleaseGateRunnerError('source_privacy_mismatch')
    _validate_all_privacy_flags(artifact)
    artifact_type = artifact.get('artifact_type')
    if not isinstance(artifact_type, str) or 'redacted' not in artifact_type:
        raise ReleaseGateRunnerError('source_artifact_not_redacted')
    metadata = {
        'artifact_file': path.name,
        'artifact_sha256': str(sidecar.get('artifact_sha256') or ''),
        'artifact_bytes': int(sidecar.get('artifact_bytes') or 0),
        'artifact_type': artifact_type,
    }
    if not _is_sha(metadata['artifact_sha256']) or len(metadata['artifact_sha256']) != 64:
        raise ReleaseGateRunnerError('source_artifact_invalid')
    return artifact, sidecar, metadata


def _retrieval_candidate_subject(artifact: Mapping[str, Any]) -> dict[str, Any]:
    evidence = artifact.get('evidence_gate')
    pair = evidence.get('provenance_pair') if isinstance(evidence, dict) else None
    candidate_sha = pair.get('candidate_commit') if isinstance(pair, dict) else None
    subject = {'commit_sha': str(candidate_sha or '').lower(), 'dirty': False}
    if not _subject_is_clean(subject):
        raise ReleaseGateRunnerError('source_subject_invalid')
    return subject


def _verify_retrieval_quality(artifact: Mapping[str, Any]) -> dict[str, Any]:
    evidence = artifact.get('evidence_gate') or {}
    metrics = artifact.get('candidate_metrics') or {}
    checks = (
        artifact.get('schema_version') == 1 and artifact.get('artifact_type') == 'retrieval_eval_gate_redacted',
        artifact.get('ok') is True and artifact.get('release_mode') is True,
        artifact.get('candidate_complete') is True,
        evidence.get('ok') is True and evidence.get('manifests_required') is True,
        (artifact.get('frozen_pack_gate') or {}).get('ok') is True,
        (artifact.get('relative_quality_gate') or {}).get('ok') is True,
        (artifact.get('category_quality_gate') or {}).get('ok') is True,
        (
            (artifact.get('case_quality_gate') or {}).get('ok') is True
            and type(metrics.get('negative_only_queries')) is int
            and metrics.get('negative_only_queries', 0) > 0
            and metrics.get('negative_pass_rate') is not None
            and artifact.get('negative_regressions') == 0
            and artifact.get('negative_pass_rate_floor_miss') is False
        ),
    )
    if not all(checks):
        raise ReleaseGateRunnerError('retrieval_quality_gate_failed')
    return _retrieval_candidate_subject(artifact)


def _verify_retrieval_latency(artifact: Mapping[str, Any]) -> dict[str, Any]:
    evidence = artifact.get('evidence_gate') or {}
    latency = artifact.get('latency_gate') or {}
    profiles = latency.get('profiles') or {}
    expected = {
        'exact_cold': ('exact', 'cold', 'off'),
        'exact_warm': ('exact', 'warm', 'off'),
        'rewrite_cold': ('rewrite', 'cold', 'on'),
        'rewrite_warm': ('rewrite', 'warm', 'on'),
    }
    profile_ok = (
        set(latency.get('required_profiles') or ()) == set(expected)
        and set(profiles) == set(expected)
        and all(
            isinstance(profiles.get(name), dict)
            and profiles[name].get('ok') is True
            and (
                profiles[name].get('route'),
                profiles[name].get('temperature'),
                profiles[name].get('semantic_mode'),
            ) == controls
            for name, controls in expected.items()
        )
    )
    checks = (
        artifact.get('schema_version') == 1 and artifact.get('artifact_type') == 'retrieval_eval_gate_redacted',
        artifact.get('ok') is True and artifact.get('release_mode') is True,
        evidence.get('ok') is True and evidence.get('manifests_required') is True,
        latency.get('ok') is True and profile_ok,
    )
    if not all(checks):
        raise ReleaseGateRunnerError('retrieval_latency_gate_failed')
    return _retrieval_candidate_subject(artifact)


def _validate_source_release_receipt(
    artifact: Mapping[str, Any],
    *,
    expected_gate_id: str,
) -> dict[str, Any]:
    contract = U15_GATE_CONTRACTS[expected_gate_id]
    receipt = artifact.get('release_receipt')
    if not isinstance(receipt, dict):
        raise ReleaseGateRunnerError('source_receipt_invalid')
    counts = receipt.get('counts')
    if (
        artifact.get('ok') is not True
        or artifact.get('artifact_type') != contract['artifact_type']
        or receipt.get('schema_version') != RECEIPT_SCHEMA_VERSION
        or receipt.get('gate_id') != expected_gate_id
        or receipt.get('command_id') != contract['command_id']
        or receipt.get('profile') != 'release'
        or receipt.get('fixture') != {'kind': 'synthetic_or_redacted'}
        or receipt.get('privacy') != _TYPED_PRIVACY
        or artifact.get('assurance') != RELEASE_ASSURANCE
        or not _subject_is_clean(receipt.get('subject'))
        or not isinstance(counts, dict)
        or set(counts) != set(contract['count_keys'])
        or any(type(value) is not int or value < 0 for value in counts.values())
        or any(counts.get(key, 0) != 0 for key in _FAILURE_COUNT_KEYS)
    ):
        raise ReleaseGateRunnerError('source_receipt_invalid')
    if contract.get('success_counts') is not None and counts != contract['success_counts']:
        raise ReleaseGateRunnerError('source_receipt_invalid')
    if contract['runner_kind'] == 'isolated-soak':
        if not release_soak_contract_valid(artifact, expected_gate_id):
            raise ReleaseGateRunnerError('source_receipt_invalid')
    else:
        execution = artifact.get('gate_execution')
        if (
            not isinstance(execution, dict)
            or execution.get('contract_schema_version') != CONTRACT_SCHEMA_VERSION
            or execution.get('contract_sha256') != release_gate_contract_sha256(expected_gate_id)
            or execution.get('runner_kind') != contract['runner_kind']
            or execution.get('checks_completed') != counts.get('checks_completed')
        ):
            raise ReleaseGateRunnerError('source_receipt_invalid')
    return dict(receipt['subject'])


def _default_executor(argv: Sequence[str], timeout_seconds: int) -> bool:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=float(timeout_seconds),
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _failure_counts(contract: Mapping[str, Any], *, checks_completed: int, supporting: int) -> dict[str, int]:
    counts = {key: 0 for key in contract['count_keys']}
    if 'checks_completed' in counts:
        counts['checks_completed'] = max(0, int(checks_completed))
    if 'supporting_artifacts' in counts:
        counts['supporting_artifacts'] = max(0, int(supporting))
    if 'failures' in counts:
        counts['failures'] = 1
    return counts


def run_release_gate(
    gate_id: str,
    *,
    source_paths: Sequence[str | Path] = (),
    executor: Callable[[Sequence[str], int], bool] | None = None,
    subject: Mapping[str, Any] | None = None,
    system: str | None = None,
) -> dict[str, Any]:
    """Execute or verify one fixed gate without accepting arbitrary commands."""

    if gate_id not in U15_GATE_CONTRACTS or gate_id in {'concurrency-10m', 'soak-1h'}:
        raise ValueError('unsupported gate id; use run_release_soak.py for soak gates')
    contract = U15_GATE_CONTRACTS[gate_id]
    production_seams = executor is None and subject is None and system is None
    executor_fn = executor or _default_executor
    start_subject = _normalized_subject(subject if subject is not None else collect_git_provenance(_ROOT))
    current_system = system if system is not None else platform.system()
    source_entries: list[dict[str, Any]] = []
    verified_gate_ids: list[str] = []
    checks_completed = 0
    failure_code: str | None = None

    try:
        if not _subject_is_clean(start_subject):
            raise ReleaseGateRunnerError('subject_not_clean')
        if contract.get('required_system') and current_system != contract['required_system']:
            raise ReleaseGateRunnerError('platform_contract_mismatch')

        kind = contract['runner_kind']
        if kind == 'locked-command-bundle':
            if source_paths:
                raise ReleaseGateRunnerError('unexpected_source_artifact')
            for step in contract['steps']:
                if not executor_fn(step['argv'], int(step['timeout_seconds'])):
                    raise ReleaseGateRunnerError('locked_step_failed')
                checks_completed += 1
        elif kind in {'verified-evidence', 'verified-receipt-set'}:
            expected_sources = 1 if kind == 'verified-evidence' else len(contract['source_gate_ids'])
            if len(source_paths) != expected_sources:
                raise ReleaseGateRunnerError('source_artifact_count_mismatch')
            loaded = [_load_verified_source(Path(path).expanduser()) for path in source_paths]
            if len({metadata['artifact_file'] for _, _, metadata in loaded}) != len(loaded):
                raise ReleaseGateRunnerError('duplicate_source_artifact')

            if gate_id == 'retrieval-quality-negative':
                artifact, _, metadata = loaded[0]
                if artifact.get('artifact_type') not in contract['source_artifact_types']:
                    raise ReleaseGateRunnerError('source_artifact_type_mismatch')
                source_subject = _verify_retrieval_quality(artifact)
                checks_completed = contract['success_counts']['checks_completed']
                source_entries.append(metadata)
            elif gate_id == 'exact-rewrite-latency':
                artifact, _, metadata = loaded[0]
                if artifact.get('artifact_type') not in contract['source_artifact_types']:
                    raise ReleaseGateRunnerError('source_artifact_type_mismatch')
                source_subject = _verify_retrieval_latency(artifact)
                checks_completed = contract['success_counts']['checks_completed']
                source_entries.append(metadata)
            elif gate_id == 'provenance':
                by_gate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
                for artifact, _, metadata in loaded:
                    receipt = artifact.get('release_receipt')
                    source_gate = receipt.get('gate_id') if isinstance(receipt, dict) else None
                    if source_gate not in contract['source_gate_ids'] or source_gate in by_gate:
                        raise ReleaseGateRunnerError('source_gate_set_mismatch')
                    by_gate[source_gate] = (artifact, metadata)
                if set(by_gate) != set(contract['source_gate_ids']):
                    raise ReleaseGateRunnerError('source_gate_set_mismatch')
                source_subject = None
                for source_gate in contract['source_gate_ids']:
                    artifact, metadata = by_gate[source_gate]
                    receipt_subject = _validate_source_release_receipt(
                        artifact,
                        expected_gate_id=source_gate,
                    )
                    if source_subject is None:
                        source_subject = receipt_subject
                    elif source_subject != receipt_subject:
                        raise ReleaseGateRunnerError('source_subject_mismatch')
                    source_entries.append({**metadata, 'gate_id': source_gate})
                    verified_gate_ids.append(source_gate)
                assert source_subject is not None
                checks_completed = contract['success_counts']['checks_completed']
            else:  # pragma: no cover - contract table makes this unreachable.
                raise ReleaseGateRunnerError('unsupported_gate_contract')

            if source_subject != start_subject:
                raise ReleaseGateRunnerError('source_subject_mismatch')
        else:  # pragma: no cover - contract table makes this unreachable.
            raise ReleaseGateRunnerError('unsupported_gate_contract')

        if production_seams:
            end_subject = _normalized_subject(collect_git_provenance(_ROOT))
            if end_subject != start_subject or not _subject_is_clean(end_subject):
                raise ReleaseGateRunnerError('subject_changed_during_gate')
    except ReleaseGateRunnerError as exc:
        failure_code = exc.code
    except Exception:
        failure_code = 'gate_runner_failed'

    if failure_code is None:
        counts = dict(contract['success_counts'])
        ok = True
    else:
        counts = _failure_counts(
            contract,
            checks_completed=checks_completed,
            supporting=len(source_entries),
        )
        ok = False

    source_entries.sort(key=lambda item: item['artifact_file'])
    verified_gate_ids.sort()
    execution = {
        'contract_schema_version': CONTRACT_SCHEMA_VERSION,
        'contract_sha256': release_gate_contract_sha256(gate_id),
        'runner_kind': contract['runner_kind'],
        'checks_planned': int(contract['success_counts']['checks_completed']),
        'checks_completed': counts.get('checks_completed', 0),
        'source_artifacts': counts.get('supporting_artifacts', 0),
    }
    receipt = {
        'schema_version': RECEIPT_SCHEMA_VERSION,
        'subject': start_subject,
        'fixture': {'kind': 'synthetic_or_redacted'},
        'privacy': dict(_TYPED_PRIVACY),
        'counts': counts,
        'platform': _platform_receipt(),
        'version': _version(),
        'command_id': contract['command_id'],
        'gate_id': gate_id,
        'profile': 'release' if production_seams else 'test-only',
    }
    report: dict[str, Any] = {
        'schema_version': GATE_ARTIFACT_SCHEMA_VERSION,
        'artifact_type': contract['artifact_type'],
        'ok': ok,
        'gate_execution': execution,
        'release_receipt': receipt,
        'source_evidence': source_entries,
        'verified_gate_ids': verified_gate_ids,
        'assurance': {
            **RELEASE_ASSURANCE,
            'clean_sha_reproducible': bool(production_seams and _subject_is_clean(start_subject)),
        },
        'privacy': dict(_TYPED_PRIVACY),
        'runtime_contract': dict(RUNTIME_CONTRACT),
    }
    if failure_code is not None:
        report['failure_code'] = failure_code
    validate_redacted_artifact(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one locked TROVE release gate.')
    parser.add_argument('--gate-id', required=True, choices=sorted(
        set(U15_GATE_CONTRACTS) - {'concurrency-10m', 'soak-1h'},
    ))
    parser.add_argument(
        '--source',
        action='append',
        default=[],
        help='Verified redacted source artifact; accepted only by evidence-backed gates.',
    )
    parser.add_argument('--out', required=True, help='Redacted receipt artifact path.')
    args = parser.parse_args(argv)
    try:
        out = Path(args.out).expanduser()
        protected = {
            candidate
            for source in args.source
            for candidate in (
                Path(source).expanduser().resolve(),
                evidence_manifest_path(Path(source).expanduser()).resolve(),
            )
        }
        output_files = {out.resolve(), evidence_manifest_path(out).resolve()}
        if output_files & protected:
            raise ReleaseGateRunnerError('output_collides_with_source')
        report = run_release_gate(args.gate_id, source_paths=args.source)
        write_evidence_artifact(report, out)
    except (ValueError, ReleaseGateRunnerError):
        print(json.dumps({'ok': False, 'error_code': 'invalid_release_gate_configuration'}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({'ok': False, 'error_code': 'release_gate_artifact_write_failed'}, sort_keys=True))
        return 2
    print(json.dumps({
        'ok': report['ok'],
        'artifact_file': out.name,
        'manifest_file': evidence_manifest_path(out).name,
        'gate_id': args.gate_id,
        'counts': report['release_receipt']['counts'],
        'private_paths_printed': False,
        'raw_command_output_printed': False,
    }, sort_keys=True))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
