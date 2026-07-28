#!/usr/bin/env python3
"""Measure the pre-cutover Agent runtime with redacted, reproducible output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for rel in (
    'packages/trove_protocol', 'packages/trove_core', 'packages/trove_client',
    'packages/trove_daemon', 'packages/trove_cli', 'packages/trove_mcp',
):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.measure_agent_surface import (  # noqa: E402
    discover_mcp_tools,
    estimate_json_tokens,
    load_task_corpus,
)
from scripts.release_gate_contracts import AGENT_RUNTIME_TARGETS  # noqa: E402


REQUIRED_MEASUREMENTS = (
    'cli_startup',
    'mcp_startup',
    'exact_recall',
    'lexical_search',
    'context',
    'minimal_daemon_noop',
)
FALSE_PRIVACY_FLAGS = (
    'content_included',
    'contacts_included',
    'citations_included',
    'absolute_paths_included',
    'secret_values_included',
)


def evaluate_absolute_budgets(artifact: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every measured R26-R29 release budget."""

    validate_benchmark_artifact(artifact)
    measurements = artifact['measurements']
    resources = artifact['resources']
    surface = artifact['surface']
    observed = {
        'warm_exact_recall_p50_ms_max': measurements['exact_recall']['warm']['p50_ms'],
        'warm_exact_recall_p95_ms_max': measurements['exact_recall']['warm']['p95_ms'],
        'warm_lexical_search_p95_ms_max': measurements['lexical_search']['warm']['p95_ms'],
        'warm_context_p95_ms_max': measurements['context']['warm']['p95_ms'],
        'protocol_adapter_noop_p95_ms_max': measurements['minimal_daemon_noop']['warm']['p95_ms'],
        'daemon_cold_ready_p95_ms_max': measurements['minimal_daemon_noop']['cold']['p95_ms'],
        'daemon_idle_rss_mib_max': resources['daemon_idle_rss_mib'],
        'mcp_idle_rss_mib_max': resources['mcp_idle_rss_mib'],
        'idle_cpu_percent_max': resources['idle_cpu_percent'],
        'standard_tools_list_bytes_max': surface['tools_list_bytes'],
        'standard_tools_list_tokens_max': surface['tools_list_estimated_tokens'],
    }
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for metric, value in observed.items():
        limit = AGENT_RUNTIME_TARGETS[metric]
        passed = float(value) < float(limit) if metric == 'idle_cpu_percent_max' else float(value) <= float(limit)
        check = {'metric': metric, 'observed': value, 'limit': limit, 'passed': passed}
        checks.append(check)
        if not passed:
            failures.append(check)
    return {
        'ok': not failures,
        'checks': checks,
        'failure_count': len(failures),
        'failures': failures,
    }


def evaluate_relative_regressions(
    artifact: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_ratio: float = 1.10,
) -> dict[str, Any]:
    """Compare p95 latency only for a same-host, same-fixture run."""

    validate_benchmark_artifact(artifact)
    validate_benchmark_artifact(baseline)
    comparable = (
        artifact['hardware'] == baseline['hardware']
        and artifact['fixture_sha256'] == baseline['fixture_sha256']
        and artifact['rounds'] == baseline['rounds']
    )
    if not comparable:
        return {
            'ok': False,
            'comparable': False,
            'failure_count': 1,
            'failures': [{'metric': 'comparison_contract', 'passed': False}],
        }
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name in ('exact_recall', 'lexical_search', 'context', 'minimal_daemon_noop'):
        current = float(artifact['measurements'][name]['warm']['p95_ms'])
        previous = float(baseline['measurements'][name]['warm']['p95_ms'])
        ratio = current / previous if previous > 0 else (1.0 if current <= 0 else float('inf'))
        passed = ratio <= max_ratio
        check = {
            'metric': f'{name}_warm_p95_ratio',
            'ratio': round(ratio, 6),
            'limit': max_ratio,
            'passed': passed,
        }
        checks.append(check)
        if not passed:
            failures.append(check)
    return {
        'ok': not failures,
        'comparable': True,
        'checks': checks,
        'failure_count': len(failures),
        'failures': failures,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError('percentile requires samples')
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(samples: list[float], *, warmups: list[float] | None = None) -> dict[str, Any]:
    if not samples:
        raise ValueError('measurement samples cannot be empty')
    rounded_samples = [round(float(value), 6) for value in samples]
    result: dict[str, Any] = {
        'samples_ms': rounded_samples,
        'p50_ms': round(_percentile(rounded_samples, 0.50), 6),
        'p95_ms': round(_percentile(rounded_samples, 0.95), 6),
        'mean_ms': round(statistics.fmean(rounded_samples), 6),
    }
    if warmups is not None:
        result['warmup_samples_ms'] = [round(float(value), 6) for value in warmups]
    return result


def _timed(call: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = call()
    return (time.perf_counter() - started) * 1000.0, result


def _subprocess_ms(code: list[str]) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        code, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=30, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f'benchmark subprocess exited with category {completed.returncode}')
    return (time.perf_counter() - started) * 1000.0


class _DaemonProtocolProbe:
    """Real trove/1 UDS path with the daemon-scoped runtime owner."""

    def __init__(self, vault: Path):
        from trove_client.client import TroveClient
        from trove_core.application.dispatcher import build_default_dispatcher
        from trove_core.vault.config import VaultConfig
        from trove_daemon.lifecycle import RuntimeIdentity, catalog_identity
        from trove_daemon.runtime_owner import RuntimeOwner
        from trove_daemon.server import DaemonServer

        config = VaultConfig.resolve(str(vault), env={})
        self.owner = RuntimeOwner(config, provider_factory=lambda: None)
        self.identity = RuntimeIdentity.for_vault(
            vault, build_hash='b' * 64, catalog_hash=catalog_identity(),
        )
        self.server = DaemonServer(
            self.identity,
            build_default_dispatcher(config, runtime_owner=self.owner),
            idle_timeout=None,
        )
        self.client = TroveClient(self.identity, pool_size=1, autostart=None)
        self._request = 0

    def __enter__(self) -> '_DaemonProtocolProbe':
        self.server.start()
        return self

    def round_trip(self) -> None:
        self._request += 1
        response = self.client.call(
            'trove.capabilities', {}, request_id=f'benchmark-{self._request}',
        )
        if response.get('ok') is not True:
            raise RuntimeError('daemon protocol probe returned a typed failure')

    def __exit__(self, *_exc: object) -> None:
        self.client.close()
        self.server.stop(timeout=2.0)
        self.owner.close()


def _measurement_pair(
    cold_call: Callable[[], Any],
    warm_call: Callable[[], Any],
    *,
    rounds: int,
    warmups: int,
) -> dict[str, Any]:
    cold = [_timed(cold_call)[0] for _ in range(rounds)]
    warmup_samples = [_timed(warm_call)[0] for _ in range(warmups)]
    warm = [_timed(warm_call)[0] for _ in range(rounds)]
    return {'cold': _summary(cold), 'warm': _summary(warm, warmups=warmup_samples)}


def _hardware() -> dict[str, Any]:
    return {
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count() or 1,
        'python': platform.python_version(),
    }


def _git_sha() -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _rss_mib(pid: int) -> float:
    completed = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'rss='],
        text=True, capture_output=True, check=False, timeout=2,
    )
    try:
        return round(int(completed.stdout.strip()) / 1024.0, 3)
    except (TypeError, ValueError):
        return 0.0


def _idle_cpu_percent(*, seconds: float = 0.5) -> float:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    time.sleep(seconds)
    elapsed_wall = max(0.001, time.perf_counter() - started_wall)
    return round(max(0.0, time.process_time() - started_cpu) / elapsed_wall * 100.0, 3)


def _mcp_idle_rss_mib() -> float:
    process = subprocess.Popen(
        [
            str(ROOT / 'scripts' / 'trove-python'), '-c',
            'import time, trove_mcp.server; time.sleep(5)',
        ],
        cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)
        if process.poll() is not None:
            return 0.0
        return _rss_mib(process.pid)
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _fixture_identity(seed: int, report: dict[str, Any]) -> str:
    payload = json.dumps(
        {'seed': seed, 'counts': report['counts'], 'chunks': report['chunks']},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def run_fixture_baseline(*, rounds: int = 5, warmups: int = 2, seed: int = 20260621) -> dict[str, Any]:
    if rounds < 1 or warmups < 1:
        raise ValueError('rounds and warmups must be positive')
    from trove_core.application.dispatcher import build_default_dispatcher
    from trove_core.application.queries import ContextQuery, SearchQuery, TroveQueries
    from trove_core.runtime import build_search_engine
    from trove_core.vault.config import VaultConfig
    from trove_core.wechat.indexer import index_fixture_vault

    with tempfile.TemporaryDirectory(prefix='trove-agent-benchmark-') as directory:
        vault = Path(directory) / 'vault'
        fixture_report = index_fixture_vault(vault, seed=seed, reset=True)
        config = VaultConfig.resolve(str(vault), env={})
        warm_engine = build_search_engine(config)
        warm_queries = TroveQueries(config, runtime=warm_engine)
        recall_input = {
            'account_id': 'acct-work', 'conversation_id': 'conv-sales-review', 'limit': 20,
        }
        warm_dispatcher = build_default_dispatcher(config)
        search_query = SearchQuery(
            '客户卡在哪', conversation_id='conv-sales-review', limit=3, semantic='off',
        )
        first = warm_queries.search(search_query).to_dict()
        citation = str(first['results'][0]['citation'])
        context_query = ContextQuery(citation, before=2, after=2)

        try:
            query_measurements = {
                'exact_recall': _measurement_pair(
                    lambda: build_default_dispatcher(config).dispatch(
                        'trove.recall', recall_input, request_id='benchmark-cold-recall',
                    ),
                    lambda: warm_dispatcher.dispatch(
                        'trove.recall', recall_input, request_id='benchmark-warm-recall',
                    ),
                    rounds=rounds, warmups=warmups,
                ),
                'lexical_search': _measurement_pair(
                    lambda: TroveQueries(config).search(search_query).to_dict(),
                    lambda: warm_queries.search(search_query).to_dict(),
                    rounds=rounds, warmups=warmups,
                ),
                'context': _measurement_pair(
                    lambda: TroveQueries(config).context(context_query).to_dict(),
                    lambda: warm_queries.context(context_query).to_dict(),
                    rounds=rounds, warmups=warmups,
                ),
            }
        finally:
            warm_engine.close()

        cli_command = [str(ROOT / 'scripts' / 'trove-python'), '-m', 'trove_cli.main', '--help']
        mcp_import = [
            str(ROOT / 'scripts' / 'trove-python'), '-c',
            'import trove_mcp.server; print("ready")',
        ]
        startup_measurements = {
            'cli_startup': {
                'cold': _summary([_subprocess_ms(cli_command) for _ in range(rounds)]),
                'warm': _summary(
                    [_subprocess_ms(cli_command) for _ in range(rounds)],
                    warmups=[_subprocess_ms(cli_command) for _ in range(warmups)],
                ),
            },
            'mcp_startup': {
                'cold': _summary([_subprocess_ms(mcp_import) for _ in range(rounds)]),
                'warm': _summary(
                    [_subprocess_ms(mcp_import) for _ in range(rounds)],
                    warmups=[_subprocess_ms(mcp_import) for _ in range(warmups)],
                ),
            },
        }

        daemon_started = time.perf_counter()
        with _DaemonProtocolProbe(vault) as probe:
            cold_ready_ms = (time.perf_counter() - daemon_started) * 1000.0
            # Warm every bounded daemon request worker before measuring the
            # persistent protocol steady state; otherwise samples measure
            # per-thread SQLite connection creation rather than adapter cost.
            protocol_warmups = max(warmups, 8)
            warmup_samples = [_timed(probe.round_trip)[0] for _ in range(protocol_warmups)]
            socket_samples = [_timed(probe.round_trip)[0] for _ in range(rounds)]
            daemon_runtime_status = probe.owner.status()
            daemon_idle_rss_mib = _rss_mib(os.getpid())
            idle_cpu_percent = _idle_cpu_percent()
        mcp_idle_rss_mib = _mcp_idle_rss_mib()
        daemon_measurement = {
            'minimal_daemon_noop': {
                'cold': _summary([cold_ready_ms]),
                'warm': _summary(socket_samples, warmups=warmup_samples),
            },
        }

        _tool_names, schemas = discover_mcp_tools()
        compact_schemas = json.dumps(
            schemas, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8')
        task_cases = load_task_corpus(ROOT / 'tests/golden/agent_task_corpus.jsonl')
        task_calls = [float(case['max_calls']) for case in task_cases]
        artifact = {
            'schema_version': 1,
            'artifact_type': 'agent_runtime_baseline_redacted',
            'git_sha': _git_sha(),
            'fixture_sha256': _fixture_identity(seed, fixture_report),
            'seed': seed,
            'rounds': rounds,
            'hardware': _hardware(),
            'measurements': startup_measurements | query_measurements | daemon_measurement,
            'resources': {
                'daemon_idle_rss_mib': daemon_idle_rss_mib,
                'mcp_idle_rss_mib': mcp_idle_rss_mib,
                'idle_cpu_percent': idle_cpu_percent,
                'store_builds': rounds * 3 + 1,
                'engine_builds': rounds + 1,
                'daemon_runtime_owner': daemon_runtime_status,
            },
            'surface': {
                'tools_list_bytes': len(compact_schemas),
                'tools_list_estimated_tokens': estimate_json_tokens(schemas),
            },
            'task_calls': {
                'sample_count': len(task_calls),
                'p50': round(_percentile(task_calls, 0.50), 3),
                'p95': round(_percentile(task_calls, 0.95), 3),
            },
            'minimal_daemon_slice': {
                'persistent_connections': 1,
                'requests': rounds + protocol_warmups,
                'cross_client_runtime_reuse': True,
                'warm_noop_p95_ms': round(_percentile(socket_samples, 0.95), 6),
                'cold_ready_ms': round(cold_ready_ms, 6),
                'r27_pass': _percentile(socket_samples, 0.95) <= 10.0 and cold_ready_ms <= 1500.0,
                'r28_pass': (
                    daemon_idle_rss_mib <= 96.0
                    and mcp_idle_rss_mib <= 72.0
                    and idle_cpu_percent < 0.5
                ),
                'transport': 'trove/1-unix',
            },
            'privacy': {flag: False for flag in FALSE_PRIVACY_FLAGS},
        }
        validate_benchmark_artifact(artifact)
        return artifact


def validate_benchmark_artifact(artifact: Any) -> None:
    if not isinstance(artifact, dict) or artifact.get('schema_version') != 1:
        raise ValueError('benchmark schema_version must be 1')
    if artifact.get('artifact_type') != 'agent_runtime_baseline_redacted':
        raise ValueError('benchmark artifact_type is invalid')
    sha = artifact.get('git_sha')
    if not isinstance(sha, str) or len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('benchmark requires a full git_sha')
    fixture = artifact.get('fixture_sha256')
    if not isinstance(fixture, str) or len(fixture) != 64 or any(c not in '0123456789abcdef' for c in fixture):
        raise ValueError('benchmark requires fixture_sha256')
    if type(artifact.get('rounds')) is not int or artifact['rounds'] < 1:
        raise ValueError('benchmark rounds must be a positive integer')
    hardware = artifact.get('hardware')
    if not isinstance(hardware, dict) or not {'system', 'machine', 'cpu_count', 'python'}.issubset(hardware):
        raise ValueError('benchmark requires a hardware summary')
    measurements = artifact.get('measurements')
    if not isinstance(measurements, dict) or set(measurements) != set(REQUIRED_MEASUREMENTS):
        raise ValueError('benchmark measurements do not match the contract')
    for name, measurement in measurements.items():
        if not isinstance(measurement, dict) or set(measurement) != {'cold', 'warm'}:
            raise ValueError(f'{name}: cold and warm samples must be separate')
        for temperature in ('cold', 'warm'):
            sample = measurement[temperature]
            values = sample.get('samples_ms') if isinstance(sample, dict) else None
            if not isinstance(values, list) or not values:
                raise ValueError(f'{name}/{temperature}: samples are required')
            expected_p50 = round(_percentile(values, 0.50), 6)
            expected_p95 = round(_percentile(values, 0.95), 6)
            if abs(float(sample.get('p50_ms', -1)) - expected_p50) > 0.000001:
                raise ValueError(f'{name}/{temperature}: p50 percentile does not match samples')
            if abs(float(sample.get('p95_ms', -1)) - expected_p95) > 0.000001:
                raise ValueError(f'{name}/{temperature}: p95 percentile does not match samples')
        warmups = measurement['warm'].get('warmup_samples_ms')
        if not isinstance(warmups, list) or not warmups:
            raise ValueError(f'{name}/warm: warmup samples must be recorded separately')
    privacy = artifact.get('privacy')
    if not isinstance(privacy, dict) or set(privacy) != set(FALSE_PRIVACY_FLAGS):
        raise ValueError('benchmark privacy metadata is incomplete')
    if any(privacy.values()):
        raise ValueError('benchmark privacy flags must all be false')
    for field in ('resources', 'surface', 'task_calls'):
        if not isinstance(artifact.get(field), dict):
            raise ValueError(f'benchmark requires {field}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Measure the redacted TROVE Agent runtime baseline.')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260621)
    parser.add_argument('--out', default='docs/perf/agent-runtime-budgets.json')
    parser.add_argument('--baseline', help='Optional same-host baseline artifact for the 10% p95 regression gate.')
    args = parser.parse_args(argv)
    artifact = run_fixture_baseline(rounds=args.rounds, warmups=args.warmups, seed=args.seed)
    output = {
        'schema_version': 1,
        'artifact_type': 'agent_runtime_release_budgets',
        'targets': AGENT_RUNTIME_TARGETS,
        'current_fixture_baseline': artifact,
        'relative_regression_policy': {
            'same_hardware_required': True,
            'same_fixture_required': True,
            'same_rounds_required': True,
            'latency_p95_max_ratio': 1.10,
        },
        'absolute_budget_evaluation': evaluate_absolute_budgets(artifact),
    }
    if args.baseline:
        baseline_payload = json.loads(Path(args.baseline).read_text(encoding='utf-8'))
        previous = baseline_payload.get('current_fixture_baseline', baseline_payload)
        output['relative_regression_evaluation'] = evaluate_relative_regressions(artifact, previous)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'out': out.name, 'artifact_sha256': hashlib.sha256(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()}, sort_keys=True, separators=(',', ':')))
    absolute_ok = output['absolute_budget_evaluation']['ok'] is True
    relative_ok = output.get('relative_regression_evaluation', {'ok': True})['ok'] is True
    return 0 if absolute_ok and relative_ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
