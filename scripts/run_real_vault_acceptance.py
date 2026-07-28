#!/usr/bin/env python3
"""Exercise a real Vault while emitting only whitelisted metrics and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)

from trove_client import TroveClient, TroveClientError  # noqa: E402
from trove_daemon.lifecycle import RuntimeIdentity  # noqa: E402


MAX_PRIVATE_INPUT = 64 * 1024
MAX_LEGACY_CITATIONS = 32
REPORT_KEYS = frozenset({
    'schema_version', 'artifact_type', 'ok', 'input_sha256', 'cases',
    'legacy_citations', 'quality', 'warm_latency_ms', 'baseline', 'privacy',
})
CASE_KEYS = frozenset({
    'ok', 'runs', 'cold_latency_ms', 'warm_latency_ms', 'result_count',
    'citation_count', 'coverage_complete', 'cursor_present', 'error_code',
})
PRIVACY = {
    'content_included': False,
    'queries_included': False,
    'citations_included': False,
    'account_identifiers_included': False,
    'private_paths_included': False,
    'raw_errors_included': False,
    'secret_values_included': False,
}


def _private_input(fd: int | None) -> dict[str, Any]:
    if fd is None:
        raise ValueError('private input fd is required')
    chunks = bytearray()
    while len(chunks) <= MAX_PRIVATE_INPUT:
        chunk = os.read(fd, min(8192, MAX_PRIVATE_INPUT + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
    if len(chunks) > MAX_PRIVATE_INPUT:
        raise ValueError('private input exceeds bound')
    value = json.loads(chunks.decode('utf-8')) if chunks else None
    if not isinstance(value, dict) or set(value) - {'recall', 'search', 'legacy_citations', 'minimums'}:
        raise ValueError('private input has an invalid shape')
    if not isinstance(value.get('recall'), dict) or not isinstance(value.get('search'), dict):
        raise ValueError('private input requires recall and search objects')
    citations = value.get('legacy_citations')
    if (
        not isinstance(citations, list)
        or not 1 <= len(citations) <= MAX_LEGACY_CITATIONS
        or any(
            not isinstance(item, str)
            or not item.startswith('trove://')
            or not 1 <= len(item) <= 2048
            for item in citations
        )
    ):
        raise ValueError('private input requires bounded legacy citations')
    minimums = value.get('minimums', {})
    if not isinstance(minimums, dict) or set(minimums) - {'capabilities', 'accounts', 'recall', 'search'}:
        raise ValueError('private minimums have an invalid shape')
    allowed_minimums = {'result_count_min', 'citation_count_min', 'warm_latency_ms_max'}
    for case, controls in minimums.items():
        if not isinstance(controls, dict) or set(controls) - allowed_minimums:
            raise ValueError(f'invalid minimum controls for {case}')
        if any(type(number) not in {int, float} or number < 0 for number in controls.values()):
            raise ValueError(f'invalid minimum value for {case}')
    return value


def _count_citations(value: Any) -> int:
    if isinstance(value, str):
        return int(value.startswith('trove://'))
    if isinstance(value, Mapping):
        return sum(_count_citations(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_citations(item) for item in value)
    return 0


def _result_count(data: Mapping[str, Any]) -> int:
    for key in ('total', 'count'):
        value = data.get(key)
        if type(value) is int and value >= 0:
            return value
    for key in ('items', 'accounts', 'capabilities', 'results', 'timeline', 'messages'):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _call(
    client: TroveClient,
    capability: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, float, str | None]:
    started = time.perf_counter()
    try:
        response = client.call(
            capability, payload,
            request_id=f'acceptance-{secrets.token_urlsafe(12)}', timeout=60.0,
        )
        return response, round((time.perf_counter() - started) * 1000.0, 3), None
    except TroveClientError as exc:
        return None, round((time.perf_counter() - started) * 1000.0, 3), exc.code


def _case(
    client: TroveClient,
    capability: str,
    payload: Mapping[str, Any],
    *,
    repeat: bool,
) -> dict[str, Any]:
    first, cold_latency, error = _call(client, capability, payload)
    second = None
    warm_latency = None
    if first is not None and repeat:
        second, warm_latency, error = _call(client, capability, payload)
    result = second or first
    if result is None:
        return {
            'ok': False, 'runs': 1, 'cold_latency_ms': cold_latency,
            'warm_latency_ms': None, 'result_count': 0, 'citation_count': 0,
            'coverage_complete': False, 'cursor_present': False,
            'error_code': error or 'acceptance_call_failed',
        }
    data = result.get('data') if isinstance(result.get('data'), Mapping) else {}
    coverage = result.get('coverage') if isinstance(result.get('coverage'), Mapping) else {}
    return {
        'ok': result.get('ok') is True,
        'runs': 2 if repeat else 1,
        'cold_latency_ms': cold_latency,
        'warm_latency_ms': warm_latency,
        'result_count': _result_count(data),
        'citation_count': _count_citations(data),
        'coverage_complete': coverage.get('state') == 'complete',
        'cursor_present': isinstance(coverage.get('cursor'), str),
        'error_code': error,
    }


def _baseline(cases: Mapping[str, Mapping[str, Any]], minimums: Mapping[str, Any]) -> dict[str, Any]:
    checks = passed = 0
    for name, controls in minimums.items():
        case = cases[name]
        for key, threshold in controls.items():
            checks += 1
            if key == 'result_count_min':
                passed += int(case['result_count'] >= threshold)
            elif key == 'citation_count_min':
                passed += int(case['citation_count'] >= threshold)
            elif key == 'warm_latency_ms_max':
                passed += int(
                    case['warm_latency_ms'] is not None
                    and case['warm_latency_ms'] <= threshold
                )
    return {
        'provided': bool(minimums),
        'checks': checks,
        'passed': passed,
        'ok': checks == passed,
    }


def run(vault: Path, private: Mapping[str, Any]) -> dict[str, Any]:
    identity = RuntimeIdentity.for_vault(vault)
    cases: dict[str, dict[str, Any]] = {}
    legacy_latencies: list[float] = []
    legacy_readable = 0
    with TroveClient(identity, role='sdk') as client:
        cases['capabilities'] = _case(client, 'trove.capabilities', {}, repeat=False)
        cases['accounts'] = _case(client, 'trove.resolve', {'kind': 'account'}, repeat=False)
        cases['recall'] = _case(client, 'trove.recall', private['recall'], repeat=True)
        cases['search'] = _case(client, 'trove.search', private['search'], repeat=True)
        for citation in private['legacy_citations']:
            result, latency, _error = _call(
                client, 'trove.context',
                {'citation': citation, 'before': 1, 'after': 1},
            )
            legacy_latencies.append(latency)
            if result is not None and result.get('ok') is True and _count_citations(result.get('data')) > 0:
                legacy_readable += 1

    minimums = private.get('minimums') if isinstance(private.get('minimums'), Mapping) else {}
    baseline = _baseline(cases, minimums)
    evidence_cases = (cases['recall'], cases['search'])
    quality = {
        'cases': len(evidence_cases),
        'cases_succeeded': sum(int(case['ok']) for case in evidence_cases),
        'citations_found': sum(case['citation_count'] for case in evidence_cases),
        'complete_coverage_cases': sum(int(case['coverage_complete']) for case in evidence_cases),
        'ok': all(
            case['ok']
            and case['result_count'] > 0
            and case['citation_count'] > 0
            and (case['coverage_complete'] or case['cursor_present'])
            for case in evidence_cases
        ),
    }
    warm = [case['warm_latency_ms'] for case in evidence_cases if case['warm_latency_ms'] is not None]
    legacy = {
        'requested': len(private['legacy_citations']),
        'readable': legacy_readable,
        'max_latency_ms': round(max(legacy_latencies), 3) if legacy_latencies else 0.0,
        'ok': legacy_readable == len(private['legacy_citations']),
    }
    report = {
        'schema_version': 1,
        'artifact_type': 'agent_runtime_real_vault_acceptance_redacted',
        'ok': bool(all(case['ok'] for case in cases.values()) and quality['ok'] and legacy['ok'] and baseline['ok']),
        'input_sha256': hashlib.sha256(json.dumps(
            private, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')).hexdigest(),
        'cases': cases,
        'legacy_citations': legacy,
        'quality': quality,
        'warm_latency_ms': {
            'p50': round(statistics.median(warm), 3) if warm else 0.0,
            'max': round(max(warm), 3) if warm else 0.0,
        },
        'baseline': baseline,
        'privacy': dict(PRIVACY),
    }
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if (
        set(report) != REPORT_KEYS
        or report.get('schema_version') != 1
        or report.get('artifact_type') != 'agent_runtime_real_vault_acceptance_redacted'
        or report.get('privacy') != PRIVACY
    ):
        raise ValueError('invalid real Vault acceptance report')
    cases = report.get('cases')
    if (
        not isinstance(cases, dict)
        or set(cases) != {'capabilities', 'accounts', 'recall', 'search'}
        or any(not isinstance(case, dict) or set(case) != CASE_KEYS for case in cases.values())
    ):
        raise ValueError('invalid real Vault case metrics')
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if any(value in serialized for value in ('trove://', '/Users/', 'traceback', 'query_text')):
        raise ValueError('real Vault acceptance report contains private data')


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.acceptance-', dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', required=True, type=Path)
    parser.add_argument('--input-fd', required=True, type=int)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        private = _private_input(args.input_fd)
        report = run(args.vault.expanduser().resolve(), private)
        if args.out:
            _atomic_write(args.out.expanduser(), report)
            print(json.dumps({'ok': report['ok'], 'artifact_file': args.out.name}, sort_keys=True))
        else:
            print(json.dumps(report, sort_keys=True, separators=(',', ':')))
        return 0 if report['ok'] else 2
    except Exception:
        print(json.dumps({'ok': False, 'error_code': 'acceptance_failed'}, sort_keys=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
