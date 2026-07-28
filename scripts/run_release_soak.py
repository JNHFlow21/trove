#!/usr/bin/env python3
"""Run a bounded, synthetic-only release soak and emit redacted evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from functools import wraps
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime

ensure_project_runtime(__file__)

from release_gate_contracts import RELEASE_ASSURANCE  # noqa: E402

from trove_core.application.queries import ContextQuery, ListQuery, TroveQueries  # noqa: E402
from trove_core.runtime import RuntimeOverloaded, RuntimeTimedOut, SearchRuntimeCache  # noqa: E402
from trove_core.search.evidence_provenance import (  # noqa: E402
    collect_git_provenance,
    evidence_manifest_path,
    verify_evidence_manifest,
    write_evidence_artifact,
)
from trove_core.search.query import SearchRequest  # noqa: E402
from trove_core.vault.config import VaultConfig  # noqa: E402
from trove_core.vault import generation as generation_module  # noqa: E402
from trove_core.vault.locks import VaultOperationLocked  # noqa: E402
from trove_core.vault.mutations import coordinated_vault_mutation  # noqa: E402
from trove_core.wechat.indexer import index_fixture_vault  # noqa: E402


SOAK_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
COMMAND_ID = 'run-release-soak-v1'
DOGFOOD_SCHEMA_VERSION = 1
DOGFOOD_REQUIRED_SECONDS = 3 * 24 * 60 * 60
DOGFOOD_SKILLS = (
    'trove-file-recall',
    'trove-group-summary',
    'trove-media-enrichment',
    'trove-profile',
    'trove-recall',
    'trove-search',
)
DOGFOOD_PRIVACY = {
    'content_included': False,
    'queries_included': False,
    'citations_included': False,
    'client_names_included': False,
    'account_identifiers_included': False,
    'private_paths_included': False,
    'raw_errors_included': False,
    'secret_values_included': False,
}
RUNTIME_CONTRACT = {
    'protocol': 'trove/1',
    'product_surface': 'mcp_cli_skills',
    'legacy_surface_present': False,
    'network_listener_present': False,
}
_SOAK_RUN_LOCK = threading.Lock()
_SYNTHETIC_EXACT_TERM = '示例教育'
_SYNTHETIC_CITATION = 'trove://wechat/acct-work/conv-example_edu-private/message_0/1'
_ERROR_KEYS = ('runtime_overloaded', 'runtime_timeouts', 'lock_errors', 'unhandled_errors')
_HEARTBEAT_KEYS = (
    'searches_completed',
    'contacts_completed',
    'contexts_completed',
    'evidence_completed',
    'writes_completed',
)
_CONTENTION_BLOCK_OBSERVATION_SECONDS = 0.02


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError('dogfood timestamps must be timezone-aware')
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.isoformat().replace('+00:00', 'Z')


def _parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith('Z'):
        raise ValueError('dogfood timestamp is invalid')
    try:
        parsed = datetime.fromisoformat(value[:-1] + '+00:00')
    except ValueError as exc:
        raise ValueError('dogfood timestamp is invalid') from exc
    if parsed.tzinfo is None or _utc_timestamp(parsed) != value:
        raise ValueError('dogfood timestamp is invalid')
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in '0123456789abcdef' for character in value)
    )


def _dogfood_start_from_distribution(
    distribution_manifest: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    from verify_distribution import verify_distribution

    verify_distribution(distribution_manifest)
    manifest = json.loads(distribution_manifest.read_text(encoding='utf-8'))
    if manifest.get('source_dirty') is not False:
        raise ValueError('dogfood requires a clean distribution')
    report = {
        'schema_version': DOGFOOD_SCHEMA_VERSION,
        'artifact_type': 'trove_v1_dogfood_start_redacted',
        'ok': True,
        'subject_sha': manifest.get('source_git_sha'),
        'distribution_set_sha256': manifest.get('distribution_set_sha256'),
        'started_at_utc': _utc_timestamp(now or datetime.now(timezone.utc)),
        'required_duration_seconds': DOGFOOD_REQUIRED_SECONDS,
        'required_clients': 2,
        'required_skills': list(DOGFOOD_SKILLS),
        'privacy': dict(DOGFOOD_PRIVACY),
    }
    validate_dogfood_start(report)
    return report


def validate_dogfood_start(report: Mapping[str, Any]) -> None:
    if set(report) != {
        'schema_version', 'artifact_type', 'ok', 'subject_sha',
        'distribution_set_sha256', 'started_at_utc', 'required_duration_seconds',
        'required_clients', 'required_skills', 'privacy',
    }:
        raise ValueError('dogfood start shape is invalid')
    if (
        report.get('schema_version') != DOGFOOD_SCHEMA_VERSION
        or report.get('artifact_type') != 'trove_v1_dogfood_start_redacted'
        or report.get('ok') is not True
        or not _is_git_sha(report.get('subject_sha'))
        or not _is_sha256(report.get('distribution_set_sha256'))
        or report.get('required_duration_seconds') != DOGFOOD_REQUIRED_SECONDS
        or report.get('required_clients') != 2
        or report.get('required_skills') != list(DOGFOOD_SKILLS)
        or report.get('privacy') != DOGFOOD_PRIVACY
    ):
        raise ValueError('dogfood start contract is invalid')
    _parse_utc_timestamp(report.get('started_at_utc'))


def _validate_dogfood_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, Mapping) or set(metrics) != {
        'sessions', 'active_days', 'source_free_install', 'mcp_connected',
        'recall_succeeded', 'provider_status_succeeded', 'skills',
    }:
        raise ValueError('dogfood client metrics shape is invalid')
    if (
        type(metrics.get('sessions')) is not int or metrics['sessions'] < 3
        or type(metrics.get('active_days')) is not int or metrics['active_days'] < 3
        or any(
            metrics.get(key) is not True
            for key in (
                'source_free_install', 'mcp_connected', 'recall_succeeded',
                'provider_status_succeeded',
            )
        )
    ):
        raise ValueError('dogfood client environment gate failed')
    skills = metrics.get('skills')
    if not isinstance(skills, Mapping) or set(skills) != set(DOGFOOD_SKILLS):
        raise ValueError('dogfood skill coverage is incomplete')
    normalized_skills: dict[str, dict[str, int]] = {}
    for skill in DOGFOOD_SKILLS:
        counters = skills[skill]
        keys = {'tasks', 'tasks_succeeded', 'cited_outcomes', 'calls', 'operator_repairs'}
        if (
            not isinstance(counters, Mapping)
            or set(counters) != keys
            or any(type(counters[key]) is not int or counters[key] < 0 for key in keys)
            or counters['tasks'] < 1
            or counters['tasks_succeeded'] != counters['tasks']
            or counters['cited_outcomes'] != counters['tasks']
            or counters['operator_repairs'] != 0
            or counters['calls'] < counters['tasks']
        ):
            raise ValueError('dogfood skill outcome gate failed')
        if skill == 'trove-recall' and counters['calls'] > counters['tasks']:
            raise ValueError('dogfood recall call budget failed')
        if skill == 'trove-search' and counters['calls'] > counters['tasks'] * 2:
            raise ValueError('dogfood search call budget failed')
        normalized_skills[skill] = {key: int(counters[key]) for key in sorted(keys)}
    return {
        'sessions': int(metrics['sessions']),
        'active_days': int(metrics['active_days']),
        'source_free_install': True,
        'mcp_connected': True,
        'recall_succeeded': True,
        'provider_status_succeeded': True,
        'skills': normalized_skills,
    }


def record_dogfood_client(
    start: Mapping[str, Any],
    client_id: str,
    metrics: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_dogfood_start(start)
    if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
        raise ValueError('dogfood client id is invalid')
    normalized = _validate_dogfood_metrics(metrics)
    recorded = now or datetime.now(timezone.utc)
    if recorded.astimezone(timezone.utc) < _parse_utc_timestamp(start['started_at_utc']):
        raise ValueError('dogfood client receipt predates the candidate')
    client_hash = hashlib.sha256(client_id.encode('utf-8')).hexdigest()
    report = {
        'schema_version': DOGFOOD_SCHEMA_VERSION,
        'artifact_type': 'trove_v1_dogfood_client_redacted',
        'ok': True,
        'subject_sha': start['subject_sha'],
        'distribution_set_sha256': start['distribution_set_sha256'],
        'client_id_sha256': client_hash,
        'recorded_at_utc': _utc_timestamp(recorded),
        **normalized,
        'privacy': dict(DOGFOOD_PRIVACY),
    }
    validate_dogfood_client(report)
    return report


def validate_dogfood_client(report: Mapping[str, Any]) -> None:
    required = {
        'schema_version', 'artifact_type', 'ok', 'subject_sha',
        'distribution_set_sha256', 'client_id_sha256', 'recorded_at_utc',
        'sessions', 'active_days', 'source_free_install', 'mcp_connected',
        'recall_succeeded', 'provider_status_succeeded', 'skills', 'privacy',
    }
    if set(report) != required:
        raise ValueError('dogfood client receipt shape is invalid')
    if (
        report.get('schema_version') != DOGFOOD_SCHEMA_VERSION
        or report.get('artifact_type') != 'trove_v1_dogfood_client_redacted'
        or report.get('ok') is not True
        or not _is_git_sha(report.get('subject_sha'))
        or not _is_sha256(report.get('distribution_set_sha256'))
        or not _is_sha256(report.get('client_id_sha256'))
        or report.get('privacy') != DOGFOOD_PRIVACY
    ):
        raise ValueError('dogfood client receipt contract is invalid')
    _parse_utc_timestamp(report.get('recorded_at_utc'))
    _validate_dogfood_metrics({
        key: report[key]
        for key in (
            'sessions', 'active_days', 'source_free_install', 'mcp_connected',
            'recall_succeeded', 'provider_status_succeeded', 'skills',
        )
    })


def finalize_dogfood_record(
    start: Mapping[str, Any],
    clients: list[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_dogfood_start(start)
    completed = now or datetime.now(timezone.utc)
    started = _parse_utc_timestamp(start['started_at_utc'])
    duration_seconds = int((completed.astimezone(timezone.utc) - started).total_seconds())
    if duration_seconds < DOGFOOD_REQUIRED_SECONDS:
        raise ValueError('three-day dogfood duration is incomplete')
    if len(clients) < int(start['required_clients']):
        raise ValueError('dogfood requires at least two clients')
    normalized_clients: list[dict[str, Any]] = []
    seen_clients: set[str] = set()
    for client in clients:
        validate_dogfood_client(client)
        if (
            client['subject_sha'] != start['subject_sha']
            or client['distribution_set_sha256'] != start['distribution_set_sha256']
            or _parse_utc_timestamp(client['recorded_at_utc']) < started
            or _parse_utc_timestamp(client['recorded_at_utc']) > completed.astimezone(timezone.utc)
            or client['client_id_sha256'] in seen_clients
        ):
            raise ValueError('dogfood client receipt is not bound to the candidate')
        seen_clients.add(client['client_id_sha256'])
        normalized_clients.append(dict(client))
    normalized_clients.sort(key=lambda item: item['client_id_sha256'])
    tasks = sum(
        counters['tasks']
        for client in normalized_clients
        for counters in client['skills'].values()
    )
    calls = sum(
        counters['calls']
        for client in normalized_clients
        for counters in client['skills'].values()
    )
    record = {
        'schema_version': DOGFOOD_SCHEMA_VERSION,
        'artifact_type': 'trove_v1_dogfood_redacted',
        'ok': True,
        'subject_sha': start['subject_sha'],
        'distribution_set_sha256': start['distribution_set_sha256'],
        'started_at_utc': start['started_at_utc'],
        'completed_at_utc': _utc_timestamp(completed),
        'duration_seconds': duration_seconds,
        'required_duration_seconds': DOGFOOD_REQUIRED_SECONDS,
        'clients': normalized_clients,
        'summary': {
            'clients': len(normalized_clients),
            'skills_per_client': len(DOGFOOD_SKILLS),
            'tasks': tasks,
            'tasks_succeeded': tasks,
            'cited_outcomes': tasks,
            'calls': calls,
            'operator_repairs': 0,
            'ordinary_call_budget_passed': True,
        },
        'privacy': dict(DOGFOOD_PRIVACY),
    }
    validate_dogfood_record(record)
    return record


def validate_dogfood_record(report: Mapping[str, Any]) -> None:
    required = {
        'schema_version', 'artifact_type', 'ok', 'subject_sha',
        'distribution_set_sha256', 'started_at_utc', 'completed_at_utc',
        'duration_seconds', 'required_duration_seconds', 'clients', 'summary',
        'privacy',
    }
    if set(report) != required:
        raise ValueError('dogfood record shape is invalid')
    if (
        report.get('schema_version') != DOGFOOD_SCHEMA_VERSION
        or report.get('artifact_type') != 'trove_v1_dogfood_redacted'
        or report.get('ok') is not True
        or not _is_git_sha(report.get('subject_sha'))
        or not _is_sha256(report.get('distribution_set_sha256'))
        or report.get('required_duration_seconds') != DOGFOOD_REQUIRED_SECONDS
        or type(report.get('duration_seconds')) is not int
        or report['duration_seconds'] < DOGFOOD_REQUIRED_SECONDS
        or report.get('privacy') != DOGFOOD_PRIVACY
    ):
        raise ValueError('dogfood record contract is invalid')
    started = _parse_utc_timestamp(report.get('started_at_utc'))
    completed = _parse_utc_timestamp(report.get('completed_at_utc'))
    if int((completed - started).total_seconds()) != report['duration_seconds']:
        raise ValueError('dogfood duration does not match timestamps')
    clients = report.get('clients')
    if not isinstance(clients, list) or len(clients) < 2:
        raise ValueError('dogfood record requires two clients')
    if [item.get('client_id_sha256') for item in clients if isinstance(item, Mapping)] != sorted(
        item.get('client_id_sha256') for item in clients if isinstance(item, Mapping)
    ):
        raise ValueError('dogfood clients are not canonical')
    seen: set[str] = set()
    tasks = calls = 0
    for client in clients:
        validate_dogfood_client(client)
        if (
            client['subject_sha'] != report['subject_sha']
            or client['distribution_set_sha256'] != report['distribution_set_sha256']
            or client['client_id_sha256'] in seen
            or _parse_utc_timestamp(client['recorded_at_utc']) < started
            or _parse_utc_timestamp(client['recorded_at_utc']) > completed
        ):
            raise ValueError('dogfood record client binding is invalid')
        seen.add(client['client_id_sha256'])
        tasks += sum(item['tasks'] for item in client['skills'].values())
        calls += sum(item['calls'] for item in client['skills'].values())
    expected_summary = {
        'clients': len(clients),
        'skills_per_client': len(DOGFOOD_SKILLS),
        'tasks': tasks,
        'tasks_succeeded': tasks,
        'cited_outcomes': tasks,
        'calls': calls,
        'operator_repairs': 0,
        'ordinary_call_budget_passed': True,
    }
    if report.get('summary') != expected_summary:
        raise ValueError('dogfood summary is invalid')


def _read_bounded_json_fd(fd: int, *, limit: int = 64 * 1024) -> dict[str, Any]:
    payload = bytearray()
    while len(payload) <= limit:
        chunk = os.read(fd, min(8192, limit + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > limit:
        raise ValueError('dogfood metrics exceed the input bound')
    value = json.loads(payload.decode('utf-8')) if payload else None
    if not isinstance(value, dict):
        raise ValueError('dogfood metrics must be an object')
    return value


def _load_verified_dogfood_artifact(path: Path, artifact_type: str) -> dict[str, Any]:
    verify_evidence_manifest(path, required=True)
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('artifact_type') != artifact_type:
        raise ValueError('dogfood evidence artifact type is invalid')
    return value


class _Counters:
    def __init__(self) -> None:
        self._values = {
            'searches_completed': 0,
            'contacts_completed': 0,
            'contexts_completed': 0,
            'evidence_completed': 0,
            'writes_completed': 0,
            'writer_generation_contentions': 0,
            'runtime_overloaded': 0,
            'runtime_timeouts': 0,
            'lock_errors': 0,
            'unhandled_errors': 0,
            'resource_samples': 0,
        }
        self._lock = threading.Lock()

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._values[key] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)


def _serialized_soak(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _SOAK_RUN_LOCK:
            return function(*args, **kwargs)
    return wrapped


class _DarwinTaskInfo(ctypes.Structure):
    _fields_ = [
        ('virtual_size', ctypes.c_uint64),
        ('resident_size', ctypes.c_uint64),
        ('total_user', ctypes.c_uint64),
        ('total_system', ctypes.c_uint64),
        ('threads_user', ctypes.c_uint64),
        ('threads_system', ctypes.c_uint64),
        ('policy', ctypes.c_int32),
        ('faults', ctypes.c_int32),
        ('pageins', ctypes.c_int32),
        ('cow_faults', ctypes.c_int32),
        ('messages_sent', ctypes.c_int32),
        ('messages_received', ctypes.c_int32),
        ('syscalls_mach', ctypes.c_int32),
        ('syscalls_unix', ctypes.c_int32),
        ('context_switches', ctypes.c_int32),
        ('thread_count', ctypes.c_int32),
        ('running_thread_count', ctypes.c_int32),
        ('priority', ctypes.c_int32),
    ]


def _darwin_task_info() -> tuple[int, int] | None:
    if platform.system() != 'Darwin':
        return None
    try:
        library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        proc_pidinfo = library.proc_pidinfo
        proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        proc_pidinfo.restype = ctypes.c_int
        info = _DarwinTaskInfo()
        written = proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if written != ctypes.sizeof(info):
            return None
        return int(info.resident_size), int(info.thread_count)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _linux_rss_and_threads() -> tuple[int | None, int | None]:
    if platform.system() != 'Linux':
        return None, None
    rss: int | None = None
    threads: int | None = None
    try:
        fields = Path('/proc/self/statm').read_text(encoding='ascii').split()
        rss = int(fields[1]) * int(os.sysconf('SC_PAGE_SIZE'))
    except (IndexError, OSError, TypeError, ValueError):
        pass
    try:
        for line in Path('/proc/self/status').read_text(encoding='ascii').splitlines():
            if line.startswith('Threads:'):
                threads = int(line.split(':', 1)[1].strip())
                break
    except (OSError, TypeError, ValueError):
        pass
    return rss, threads


def _ps_current_rss() -> int | None:
    """Portable POSIX fallback; ``ps rss`` is current, not high-water RSS."""

    try:
        completed = subprocess.run(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        kib = int(completed.stdout.strip()) if completed.returncode == 0 else 0
        return kib * 1024 if kib > 0 else None
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return None


def _windows_current_rss() -> int | None:
    if os.name != 'nt':
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ('cb', ctypes.c_ulong),
            ('page_fault_count', ctypes.c_ulong),
            ('peak_working_set_size', ctypes.c_size_t),
            ('working_set_size', ctypes.c_size_t),
            ('quota_peak_paged_pool_usage', ctypes.c_size_t),
            ('quota_paged_pool_usage', ctypes.c_size_t),
            ('quota_peak_non_paged_pool_usage', ctypes.c_size_t),
            ('quota_non_paged_pool_usage', ctypes.c_size_t),
            ('pagefile_usage', ctypes.c_size_t),
            ('peak_pagefile_usage', ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.working_set_size) if ok else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _open_fd_count() -> int | None:
    for directory in ('/proc/self/fd', '/dev/fd'):
        try:
            return len(os.listdir(directory))
        except OSError:
            continue
    if os.name == 'nt':
        try:
            count = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetProcessHandleCount(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(count),
            )
            return int(count.value) if ok else None
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return None


class CurrentProcessSampler:
    """Sample current process resources without retaining paths or content."""

    def __call__(self) -> dict[str, int | None]:
        darwin = _darwin_task_info()
        linux_rss, linux_threads = _linux_rss_and_threads()
        rss = (
            darwin[0]
            if darwin is not None
            else linux_rss
            if linux_rss is not None
            else _windows_current_rss()
            if os.name == 'nt'
            else _ps_current_rss()
        )
        process_threads = darwin[1] if darwin is not None else linux_threads
        return {
            'rss_bytes': rss,
            'fd_count': _open_fd_count(),
            'thread_count': process_threads if process_threads is not None else threading.active_count(),
        }


def classify_soak_error(exc: Exception) -> str:
    if isinstance(exc, RuntimeOverloaded):
        return 'runtime_overloaded'
    if isinstance(exc, RuntimeTimedOut):
        return 'runtime_timeouts'
    if isinstance(exc, VaultOperationLocked):
        return 'lock_errors'
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        if 'locked' in message or 'busy' in message:
            return 'lock_errors'
    return 'unhandled_errors'


def _validated_sample(sample: Mapping[str, Any]) -> dict[str, int | None]:
    validated: dict[str, int | None] = {}
    for key in ('rss_bytes', 'fd_count', 'thread_count'):
        value = sample.get(key)
        if value is None:
            validated[key] = None
        elif type(value) is int and value >= 0:
            validated[key] = value
        else:
            raise ValueError('resource sampler returned an invalid metric')
    return validated


def _resource_summary(
    samples: list[dict[str, int | None]],
    *,
    window_samples: int,
    sustained_windows: int,
    growth_thresholds: Mapping[str, int],
    sample_offsets: list[float] | None = None,
    heartbeat_samples: list[Mapping[str, int]] | None = None,
    heartbeat_baseline: Mapping[str, int] | None = None,
    sample_interval_seconds: float | None = None,
    duration_seconds: float | None = None,
    missed_deadlines: int = 0,
    require_writer_heartbeat_per_window: bool = False,
) -> dict[str, Any]:
    if window_samples < 1 or sustained_windows < 1:
        raise ValueError('resource window sizes must be positive')
    metrics: dict[str, Any] = {}
    any_sustained = False
    required_sample_count = window_samples * (sustained_windows + 1)
    full_window_count = len(samples) // window_samples
    complete = len(samples) >= required_sample_count
    for key in ('rss_bytes', 'fd_count', 'thread_count'):
        values = [int(sample[key]) for sample in samples if sample.get(key) is not None]
        complete = complete and len(values) == len(samples)
        windows = [
            values[offset:offset + window_samples]
            for offset in range(0, len(values), window_samples)
            if len(values[offset:offset + window_samples]) == window_samples
        ]
        medians = [float(median(window)) for window in windows]
        first = medians[0] if medians else None
        last = medians[-1] if medians else None
        threshold = int(growth_thresholds[key])
        sustained = bool(
            first is not None
            and len(medians) >= sustained_windows + 1
            and all(value - first > threshold for value in medians[-sustained_windows:])
        )
        any_sustained = any_sustained or sustained
        metrics[key] = {
            'first_window_median': first,
            'last_window_median': last,
            'last_minus_first': None if first is None or last is None else last - first,
            'peak': max(values) if values else None,
            'sustained_growth_threshold': threshold,
            'sustained_growth': sustained,
        }
    summary: dict[str, Any] = {
        'sample_count': len(samples),
        'required_sample_count': required_sample_count,
        'full_window_count': full_window_count,
        'window_samples': window_samples,
        'sustained_windows': sustained_windows,
        'complete': complete,
        'any_sustained_growth': any_sustained,
        'metrics': metrics,
    }
    if sample_offsets is None:
        return summary

    if (
        heartbeat_samples is None
        or heartbeat_baseline is None
        or sample_interval_seconds is None
        or duration_seconds is None
        or sample_interval_seconds <= 0
        or duration_seconds <= 0
        or len(sample_offsets) != len(samples)
        or len(heartbeat_samples) != len(samples)
    ):
        summary.update({
            'expected_sample_count': 0,
            'minimum_sample_count': 0,
            'missed_deadlines': max(0, int(missed_deadlines)),
            'max_sample_gap_seconds': None,
            'max_allowed_sample_gap_seconds': 0.0,
            'heartbeat_window_count': 0,
            'writer_heartbeat_required': bool(require_writer_heartbeat_per_window),
            'window_heartbeats': [],
        })
        summary['complete'] = False
        return summary

    interval = float(sample_interval_seconds)
    duration = float(duration_seconds)
    expected_sample_count = max(1, int(math.ceil(duration / interval)))
    # One edge sample may be lost to an explicitly bounded scheduling jitter;
    # a pause can never be hidden by rapidly collecting overdue samples.
    minimum_sample_count = max(1, expected_sample_count - 1)
    max_allowed_gap = interval * 1.5
    gaps = [max(0.0, float(sample_offsets[0]))] if sample_offsets else [duration]
    gaps.extend(
        max(0.0, float(current) - float(previous))
        for previous, current in zip(sample_offsets, sample_offsets[1:])
    )
    if sample_offsets:
        gaps.append(max(0.0, duration - float(sample_offsets[-1])))
    max_sample_gap = max(gaps)

    previous = {key: int(heartbeat_baseline.get(key, 0)) for key in _HEARTBEAT_KEYS}
    window_heartbeats: list[dict[str, Any]] = []
    for end in range(window_samples - 1, len(heartbeat_samples), window_samples):
        snapshot = heartbeat_samples[end]
        deltas = {
            key: max(0, int(snapshot.get(key, 0)) - previous[key])
            for key in _HEARTBEAT_KEYS
        }
        required_keys = _HEARTBEAT_KEYS if require_writer_heartbeat_per_window else _HEARTBEAT_KEYS[:-1]
        window_heartbeats.append({
            **deltas,
            'complete': all(deltas[key] > 0 for key in required_keys),
        })
        previous = {key: int(snapshot.get(key, 0)) for key in _HEARTBEAT_KEYS}

    cadence_complete = bool(
        minimum_sample_count <= len(samples) <= expected_sample_count
        and int(missed_deadlines) == 0
        and max_sample_gap <= max_allowed_gap
    )
    checked_heartbeats = (
        window_heartbeats
        if require_writer_heartbeat_per_window
        else window_heartbeats[1:] or window_heartbeats
    )
    heartbeats_complete = bool(
        len(window_heartbeats) == full_window_count
        and len(window_heartbeats) > 0
        and all(window['complete'] is True for window in checked_heartbeats)
    )
    summary.update({
        'expected_sample_count': expected_sample_count,
        'minimum_sample_count': minimum_sample_count,
        'missed_deadlines': max(0, int(missed_deadlines)),
        'max_sample_gap_seconds': round(max_sample_gap, 6),
        'max_allowed_sample_gap_seconds': round(max_allowed_gap, 6),
        'heartbeat_window_count': len(window_heartbeats),
        'writer_heartbeat_required': bool(require_writer_heartbeat_per_window),
        'window_heartbeats': window_heartbeats,
    })
    summary['complete'] = bool(summary['complete'] and cadence_complete and heartbeats_complete)
    return summary


def _version() -> str:
    try:
        return importlib.metadata.version('trove-runtime')
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


def _platform_receipt() -> dict[str, str]:
    return {
        'system': platform.system() or 'unknown',
        'release': platform.release() or 'unknown',
        'machine': platform.machine() or 'unknown',
        'python_version': platform.python_version(),
    }


def _write_synthetic_marker(
    cfg: VaultConfig,
    sequence: int,
    *,
    invalidate: Callable[[], None] | None = None,
) -> None:
    # A generation-exclusive coordinated mutation is the production boundary:
    # the writer may arrive while readers are active, then publication waits
    # for their leases rather than relying on a harness-only pause.
    with coordinated_vault_mutation(cfg, operation='scope_rebuild'):
        connection = sqlite3.connect(cfg.paths.sqlite_path, timeout=5.0)
        try:
            connection.execute(
                "INSERT INTO schema_meta(key,value) VALUES('release_soak_sequence',?) "
                'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                (str(sequence),),
            )
            connection.commit()
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key='release_soak_sequence'"
            ).fetchone()
            if row is None or str(row[0]) != str(sequence):
                raise RuntimeError('The coordinated generation mutation was not durably observed.')
            if invalidate is not None:
                # Close the old runtime while publication is still exclusive;
                # no new-generation reader can race an old cached SQLite/WAL
                # handle before invalidation completes.
                invalidate()
        finally:
            connection.close()


def _generation_lock_conflicts(cfg: VaultConfig, operation: int) -> bool:
    """Prove an independent descriptor cannot obtain the requested kernel lock."""

    flags = os.O_RDONLY | int(getattr(os, 'O_DIRECTORY', 0)) | int(getattr(os, 'O_NOFOLLOW', 0))
    fd = os.open(cfg.root, flags)
    try:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _default_runtime_factory(cfg: VaultConfig, readers: int) -> SearchRuntimeCache:
    return SearchRuntimeCache(
        cfg,
        provider_factory=lambda: None,
        max_workers=min(8, readers),
        max_queue=max(32, readers),
        timeout_seconds=15.0,
        submit_timeout_seconds=0.05,
    )


def _gate_identity_for_controls(
    *,
    production_seams: bool,
    duration_seconds: float,
    readers: int,
    warmup_searches: int,
    writer_interval_seconds: float,
    sample_interval_seconds: float,
    resource_window_samples: int,
    sustained_windows: int,
    rss_growth_threshold_bytes: int,
    fd_growth_threshold: int,
    thread_growth_threshold: int,
) -> tuple[str, str]:
    locked = (
        readers == 32
        and warmup_searches == 8
        and float(writer_interval_seconds) == 5.0
        and float(sample_interval_seconds) == 5.0
        and resource_window_samples == 5
        and sustained_windows == 3
        and rss_growth_threshold_bytes == 64 * 1024 * 1024
        and fd_growth_threshold == 8
        and thread_growth_threshold == 4
    )
    if production_seams and locked and float(duration_seconds) == 600.0:
        return 'concurrency-10m', 'release'
    if production_seams and locked and float(duration_seconds) == 3600.0:
        return 'soak-1h', 'release'
    return 'test-only', 'test-only'


@_serialized_soak
def run_release_soak(
    *,
    duration_seconds: float = 3600.0,
    readers: int = 32,
    warmup_searches: int = 8,
    writer_interval_seconds: float = 5.0,
    sample_interval_seconds: float = 5.0,
    resource_window_samples: int = 5,
    sustained_windows: int = 3,
    rss_growth_threshold_bytes: int = 64 * 1024 * 1024,
    fd_growth_threshold: int = 8,
    thread_growth_threshold: int = 4,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    sampler: Callable[[], Mapping[str, Any]] | None = None,
    subject: Mapping[str, Any] | None = None,
    runtime_factory: Callable[[VaultConfig, int], SearchRuntimeCache] = _default_runtime_factory,
    _isolated_worker: bool = False,
) -> dict[str, Any]:
    """Exercise the real bounded runtime against a disposable synthetic Vault.

    ``clock``, ``sleeper``, and ``sampler`` are explicit seams for short,
    deterministic tests. Production invocations use monotonic wall time and
    current-process resource sampling.
    """

    if not isinstance(duration_seconds, (int, float)) or not 0 < duration_seconds <= 86_400:
        raise ValueError('duration_seconds must be greater than zero and at most 86400')
    if type(readers) is not int or not 1 <= readers <= 256:
        raise ValueError('readers must be 1..256')
    if type(warmup_searches) is not int or not 0 <= warmup_searches <= 10_000:
        raise ValueError('warmup_searches must be 0..10000')
    if type(resource_window_samples) is not int or resource_window_samples < 1:
        raise ValueError('resource_window_samples must be a positive integer')
    if type(sustained_windows) is not int or sustained_windows < 1:
        raise ValueError('sustained_windows must be a positive integer')
    for name, value in (
        ('writer_interval_seconds', writer_interval_seconds),
        ('sample_interval_seconds', sample_interval_seconds),
    ):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'{name} must be greater than zero')
    if any(type(value) is not int or value < 0 for value in (
        rss_growth_threshold_bytes,
        fd_growth_threshold,
        thread_growth_threshold,
    )):
        raise ValueError('growth thresholds must be non-negative integers')

    release_subject = dict(subject) if subject is not None else collect_git_provenance(_ROOT)
    commit_sha = str(release_subject.get('commit_sha') or '').lower()
    dirty = release_subject.get('dirty')
    sampler_fn = sampler or CurrentProcessSampler()
    counters = _Counters()
    samples: list[dict[str, int | None]] = []
    sample_offsets: list[float] = []
    heartbeat_samples: list[dict[str, int]] = []
    heartbeat_baseline: dict[str, int] = {}
    missed_deadlines = 0
    cleanup = {
        'readers_joined': False,
        'writer_joined': False,
        'runtime_closed': False,
        'temp_vault_removed': False,
    }
    runtime: SearchRuntimeCache | None = None
    application_queries: TroveQueries | None = None
    reader_pool: ThreadPoolExecutor | None = None
    reader_futures: list[Future[Any]] = []
    writer_thread: threading.Thread | None = None
    stop = threading.Event()
    start = threading.Event()
    readers_ready = threading.Event()
    ready_lock = threading.Lock()
    ready_count = 0
    lease_activity = threading.Condition()
    leased_readers = 0
    observed_read_leases: set[int] = set()
    pause_reader_operations = threading.Event()
    publish_acquire_started = threading.Event()
    publish_acquire_completed = threading.Event()
    publish_acquire_active = threading.Event()
    publish_kernel_lock_verified = threading.Event()
    original_generation_acquire = generation_module.VaultGenerationLease.acquire
    original_generation_release = generation_module.VaultGenerationLease.release
    lease_observer_installed = False
    elapsed_seconds = 0.0
    temp = tempfile.TemporaryDirectory(prefix='trove-release-soak-synthetic-')
    temp_root = Path(temp.name)

    request = SearchRequest(
        _SYNTHETIC_EXACT_TERM,
        limit=3,
        include_vector=False,
        semantic='off',
        reranker_mode='off',
        expand_query=False,
        include_media_hints=False,
    )

    def record_error(exc: Exception) -> None:
        counters.increment(classify_soak_error(exc))

    def reader_loop(worker_index: int) -> None:
        nonlocal ready_count
        with ready_lock:
            ready_count += 1
            if ready_count == readers:
                readers_ready.set()
        if not start.wait(timeout=10.0):
            counters.increment('unhandled_errors')
            return
        operation_index = worker_index
        while not stop.is_set():
            try:
                while pause_reader_operations.is_set() and not stop.is_set():
                    stop.wait(0.001)
                if stop.is_set():
                    return
                if runtime is None or application_queries is None:
                    raise RuntimeError('read services unavailable')
                operation = operation_index % 4
                if operation == 0:
                    runtime.search(request)
                    counters.increment('searches_completed')
                elif operation == 1:
                    with generation_module.vault_generation_read(cfg):
                        result = application_queries.list_contacts(ListQuery(limit=3))
                    if not result.ok:
                        raise RuntimeError('synthetic list read failed')
                    counters.increment('contacts_completed')
                elif operation == 2:
                    with generation_module.vault_generation_read(cfg):
                        result = application_queries.context(ContextQuery(_SYNTHETIC_CITATION, before=1, after=1))
                    if not result.ok:
                        raise RuntimeError('synthetic context read failed')
                    counters.increment('contexts_completed')
                else:
                    with generation_module.vault_generation_read(cfg):
                        result = application_queries.evidence(_SYNTHETIC_CITATION)
                    if not result.ok:
                        raise RuntimeError('synthetic evidence read failed')
                    counters.increment('evidence_completed')
                operation_index += 1
                # Fixed workload pacing is independent of writer state. It
                # keeps 32 callers active without an unrealistic zero-think
                # hot loop that can starve an OS advisory-lock writer.
                stop.wait(0.005)
            except Exception as exc:  # Only redacted typed counts leave this worker.
                record_error(exc)
                stop.wait(0.001)

    def observed_generation_acquire(lease):
        nonlocal leased_readers
        matches_soak = (
            isinstance(lease.cfg, VaultConfig)
            and lease.cfg.root.resolve() == cfg.root.resolve()
        )
        lease_identity = id(lease)
        if matches_soak and lease.mode == 'publish':
            # This is only an attempt marker. Completion is published strictly
            # after the production acquire returns with an active lease.
            publish_acquire_started.set()
        result = original_generation_acquire(lease)
        if not matches_soak:
            return result
        if lease.mode == 'publish':
            if lease.active:
                publish_acquire_active.set()
                if _generation_lock_conflicts(cfg, fcntl.LOCK_SH):
                    publish_kernel_lock_verified.set()
            publish_acquire_completed.set()
            if not lease.active or not publish_kernel_lock_verified.is_set():
                if lease.active:
                    original_generation_release(lease)
                raise RuntimeError('Generation publisher returned without the production kernel lease.')
            return result
        if not lease.active:
            raise RuntimeError('Generation reader returned without an active production lease.')
        with lease_activity:
            observed_read_leases.add(lease_identity)
            leased_readers += 1
            lease_activity.notify_all()
        return result

    def observed_generation_release(lease) -> None:
        nonlocal leased_readers
        lease_identity = id(lease)
        observed = False
        with lease_activity:
            observed = lease_identity in observed_read_leases
        try:
            original_generation_release(lease)
        finally:
            if observed:
                with lease_activity:
                    observed_read_leases.discard(lease_identity)
                    leased_readers = max(0, leased_readers - 1)
                    lease_activity.notify_all()

    def writer_loop() -> None:
        if not start.wait(timeout=10.0):
            counters.increment('unhandled_errors')
            return
        sequence = 0
        next_write = clock() + float(writer_interval_seconds)
        while not stop.is_set():
            remaining = next_write - clock()
            if remaining > 0:
                if stop.wait(min(remaining, 0.05)):
                    return
                continue
            sequence += 1
            mutation_thread: threading.Thread | None = None
            probe_lease = None
            mutation_errors: list[Exception] = []
            try:
                if runtime is None:
                    raise RuntimeError('runtime unavailable')
                pause_reader_operations.set()
                drain_deadline = time.monotonic() + 2.0
                with lease_activity:
                    while leased_readers > 0 and not stop.is_set() and time.monotonic() < drain_deadline:
                        lease_activity.wait(timeout=0.005)
                    if leased_readers > 0:
                        raise RuntimeTimedOut('Synthetic readers did not drain before the contention probe.')
                if stop.is_set():
                    return
                probe_lease = generation_module.VaultGenerationLease(cfg, mode='read')
                original_generation_acquire(probe_lease)
                if not probe_lease.active or not _generation_lock_conflicts(cfg, fcntl.LOCK_EX):
                    raise RuntimeError('The dedicated generation reader did not own a shared kernel lease.')

                def mutate() -> None:
                    try:
                        _write_synthetic_marker(
                            cfg,
                            sequence,
                            invalidate=runtime.release_resources,
                        )
                    except Exception as exc:
                        mutation_errors.append(exc)

                mutation_thread = threading.Thread(
                    target=mutate,
                    name='trove-soak-coordinated-writer',
                    daemon=False,
                )
                mutation_thread.start()
                if not publish_acquire_started.wait(timeout=2.0):
                    raise RuntimeTimedOut('The coordinated writer did not reach generation publication.')
                # A slow thread is not contention proof. The production
                # ``acquire`` must remain incomplete while the observed read
                # lease is held, and must return active only after that exact
                # reader releases.
                if publish_acquire_completed.wait(timeout=_CONTENTION_BLOCK_OBSERVATION_SECONDS):
                    raise RuntimeError('Generation publication completed before reader release.')
                with lease_activity:
                    contention_proven = (
                        probe_lease.active
                        and mutation_thread.is_alive()
                    )
                if not contention_proven:
                    raise RuntimeError('Generation writer contention was not proven.')
                original_generation_release(probe_lease)
                probe_lease = None
                if not publish_acquire_completed.wait(timeout=2.0):
                    raise RuntimeTimedOut('Generation publication did not acquire after reader release.')
                if not publish_acquire_active.is_set():
                    raise RuntimeError('Generation publication did not return an active production lease.')
                if not publish_kernel_lock_verified.is_set():
                    raise RuntimeError('Generation publication did not hold the exclusive kernel lease.')
                mutation_thread.join(timeout=10.0)
                if mutation_thread.is_alive():
                    raise RuntimeTimedOut('The coordinated writer did not complete after reader release.')
                if mutation_errors:
                    raise mutation_errors[0]
                runtime.invalidate('release_soak_writer')
                counters.increment('writes_completed')
                counters.increment('writer_generation_contentions')
            except Exception as exc:
                # Teardown can interrupt an in-flight contention handshake.
                # That is an expected cancellation, not a soak failure.
                if not stop.is_set():
                    record_error(exc)
            finally:
                if probe_lease is not None:
                    original_generation_release(probe_lease)
                pause_reader_operations.clear()
                if mutation_thread is not None and mutation_thread.is_alive():
                    mutation_thread.join(timeout=1.0)
                publish_acquire_started.clear()
                publish_acquire_completed.clear()
                publish_acquire_active.clear()
                publish_kernel_lock_verified.clear()
            next_write += float(writer_interval_seconds)

    try:
        index_fixture_vault(temp_root, reset=True)
        cfg = VaultConfig.resolve(str(temp_root), env={})
        generation_module.VaultGenerationLease.acquire = observed_generation_acquire
        generation_module.VaultGenerationLease.release = observed_generation_release
        lease_observer_installed = True
        runtime = runtime_factory(cfg, readers)
        application_queries = TroveQueries(cfg, runtime=runtime)
        for _ in range(warmup_searches):
            try:
                runtime.search(request)
            except Exception as exc:
                record_error(exc)

        reader_pool = ThreadPoolExecutor(max_workers=readers, thread_name_prefix='trove-soak-reader')
        reader_futures = [reader_pool.submit(reader_loop, worker_index) for worker_index in range(readers)]
        if not readers_ready.wait(timeout=10.0):
            counters.increment('unhandled_errors')
        writer_thread = threading.Thread(target=writer_loop, name='trove-soak-writer', daemon=False)
        writer_thread.start()

        measured_start = clock()
        measured_end = measured_start + float(duration_seconds)
        heartbeat_baseline = counters.snapshot()
        start.set()
        next_sample = measured_start
        while not stop.is_set():
            now = clock()
            if now >= measured_end:
                break
            if now >= next_sample:
                overdue_slots = int(max(0.0, now - next_sample) // float(sample_interval_seconds))
                if overdue_slots:
                    missed_deadlines += overdue_slots
                    next_sample += overdue_slots * float(sample_interval_seconds)
                try:
                    samples.append(_validated_sample(sampler_fn()))
                    counters.increment('resource_samples')
                    sample_offsets.append(max(0.0, clock() - measured_start))
                    heartbeat_samples.append(counters.snapshot())
                except Exception as exc:
                    record_error(exc)
                next_sample += float(sample_interval_seconds)
            remaining = min(measured_end, next_sample) - clock()
            if remaining > 0:
                sleeper(min(remaining, 0.05))
        elapsed_seconds = max(0.0, min(float(duration_seconds), clock() - measured_start))
    except Exception as exc:
        record_error(exc)
    finally:
        stop.set()
        start.set()
        pause_reader_operations.clear()
        with lease_activity:
            lease_activity.notify_all()
        if writer_thread is not None:
            try:
                writer_thread.join(timeout=20.0)
                cleanup['writer_joined'] = not writer_thread.is_alive()
            except Exception as exc:
                record_error(exc)
        else:
            cleanup['writer_joined'] = True
        if reader_pool is not None:
            try:
                reader_pool.shutdown(wait=True, cancel_futures=True)
                cleanup['readers_joined'] = all(future.done() for future in reader_futures)
            except Exception as exc:
                record_error(exc)
        else:
            cleanup['readers_joined'] = True
        if lease_observer_installed:
            generation_module.VaultGenerationLease.acquire = original_generation_acquire
            generation_module.VaultGenerationLease.release = original_generation_release
            lease_observer_installed = False
        if runtime is not None:
            try:
                runtime.close()
                cleanup['runtime_closed'] = True
            except Exception as exc:
                record_error(exc)
        else:
            cleanup['runtime_closed'] = True
        try:
            temp.cleanup()
            cleanup['temp_vault_removed'] = not temp_root.exists()
        except Exception as exc:
            record_error(exc)

    resource_summary = _resource_summary(
        samples,
        window_samples=resource_window_samples,
        sustained_windows=sustained_windows,
        growth_thresholds={
            'rss_bytes': rss_growth_threshold_bytes,
            'fd_count': fd_growth_threshold,
            'thread_count': thread_growth_threshold,
        },
        sample_offsets=sample_offsets,
        heartbeat_samples=heartbeat_samples,
        heartbeat_baseline=heartbeat_baseline,
        sample_interval_seconds=float(sample_interval_seconds),
        duration_seconds=float(duration_seconds),
        missed_deadlines=missed_deadlines,
        require_writer_heartbeat_per_window=bool(
            float(duration_seconds) in {600.0, 3600.0}
            and readers == 32
            and float(writer_interval_seconds) == 5.0
            and float(sample_interval_seconds) == 5.0
            and resource_window_samples == 5
        ),
    )
    counts = counters.snapshot()
    subject_valid = (
        len(commit_sha) in {40, 64}
        and all(character in '0123456789abcdef' for character in commit_sha)
        and type(dirty) is bool
        and not dirty
    )
    error_free = all(counts[key] == 0 for key in _ERROR_KEYS)
    cleanup_complete = all(cleanup.values())
    completed_duration = elapsed_seconds >= float(duration_seconds) * 0.99
    ok = bool(
        subject_valid
        and error_free
        and cleanup_complete
        and completed_duration
        and counts['searches_completed'] > 0
        and counts['contacts_completed'] > 0
        and counts['contexts_completed'] > 0
        and counts['evidence_completed'] > 0
        and counts['writes_completed'] > 0
        and counts['writer_generation_contentions'] > 0
        and resource_summary['complete']
        and not resource_summary['any_sustained_growth']
    )
    production_seams = (
        _isolated_worker
        and
        subject is None
        and sampler is None
        and runtime_factory is _default_runtime_factory
        and clock is time.monotonic
        and sleeper is time.sleep
    )
    gate_id, receipt_profile = _gate_identity_for_controls(
        production_seams=production_seams,
        duration_seconds=duration_seconds,
        readers=readers,
        warmup_searches=warmup_searches,
        writer_interval_seconds=writer_interval_seconds,
        sample_interval_seconds=sample_interval_seconds,
        resource_window_samples=resource_window_samples,
        sustained_windows=sustained_windows,
        rss_growth_threshold_bytes=rss_growth_threshold_bytes,
        fd_growth_threshold=fd_growth_threshold,
        thread_growth_threshold=thread_growth_threshold,
    )
    release_receipt = {
        'schema_version': RECEIPT_SCHEMA_VERSION,
        'subject': {'commit_sha': commit_sha, 'dirty': dirty},
        'fixture': {'kind': 'synthetic_or_redacted'},
        'privacy': {
            'synthetic_or_redacted': True,
            'redacted': True,
            'sensitive_content_included': False,
            'private_paths_included': False,
            'raw_command_output_included': False,
            'credential_values_included': False,
        },
        'counts': counts,
        'platform': _platform_receipt(),
        'version': _version(),
        'command_id': COMMAND_ID,
        'gate_id': gate_id,
        'profile': receipt_profile,
    }
    return {
        'schema_version': SOAK_SCHEMA_VERSION,
        'artifact_type': 'release_soak_redacted',
        'ok': ok,
        'release_receipt': release_receipt,
        'configuration': {
            'readers': readers,
            'writers': 1,
            'warmup_searches': warmup_searches,
            'duration_seconds': float(duration_seconds),
            'sample_interval_seconds': float(sample_interval_seconds),
            'writer_interval_seconds': float(writer_interval_seconds),
            'search_mode': 'exact_off',
            'read_workload': 'exact+contacts+context+evidence',
            'process_isolated': bool(_isolated_worker),
        },
        'elapsed_seconds': round(elapsed_seconds, 6),
        'counts': counts,
        'resources': resource_summary,
        'cleanup': cleanup,
        'assurance': {
            **RELEASE_ASSURANCE,
            'clean_sha_reproducible': bool(receipt_profile == 'release' and subject_valid),
        },
        'privacy': dict(release_receipt['privacy']),
        'runtime_contract': dict(RUNTIME_CONTRACT),
    }


def _isolated_soak_worker(send_connection, options: dict[str, Any]) -> None:
    try:
        report = run_release_soak(**options, _isolated_worker=True)
        send_connection.send({'kind': 'report', 'report': report})
    except BaseException:
        try:
            send_connection.send({'kind': 'worker_failure'})
        except BaseException:
            pass
    finally:
        send_connection.close()


def _terminate_and_reap(process: multiprocessing.Process) -> bool:
    if process.is_alive():
        process.terminate()
        process.join(timeout=3.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=3.0)
    return not process.is_alive()


def _isolated_failure_report(options: Mapping[str, Any], *, reason_code: str, reaped: bool) -> dict[str, Any]:
    subject = collect_git_provenance(_ROOT)
    counts = {
        'searches_completed': 0,
        'contacts_completed': 0,
        'contexts_completed': 0,
        'evidence_completed': 0,
        'writes_completed': 0,
        'writer_generation_contentions': 0,
        'runtime_overloaded': 0,
        'runtime_timeouts': int(reason_code == 'worker_hard_timeout'),
        'lock_errors': 0,
        'unhandled_errors': int(reason_code != 'worker_hard_timeout'),
        'resource_samples': 0,
    }
    duration = float(options['duration_seconds'])
    gate_id, profile_name = _gate_identity_for_controls(
        production_seams=True,
        duration_seconds=duration,
        readers=int(options['readers']),
        warmup_searches=int(options['warmup_searches']),
        writer_interval_seconds=float(options['writer_interval_seconds']),
        sample_interval_seconds=float(options['sample_interval_seconds']),
        resource_window_samples=int(options['resource_window_samples']),
        sustained_windows=int(options['sustained_windows']),
        rss_growth_threshold_bytes=int(options['rss_growth_threshold_bytes']),
        fd_growth_threshold=int(options['fd_growth_threshold']),
        thread_growth_threshold=int(options['thread_growth_threshold']),
    )
    privacy = {
        'synthetic_or_redacted': True,
        'redacted': True,
        'sensitive_content_included': False,
        'private_paths_included': False,
        'raw_command_output_included': False,
        'credential_values_included': False,
    }
    platform_receipt = _platform_receipt()
    receipt = {
        'schema_version': RECEIPT_SCHEMA_VERSION,
        'subject': subject,
        'fixture': {'kind': 'synthetic_or_redacted'},
        'privacy': privacy,
        'counts': counts,
        'platform': platform_receipt,
        'version': _version(),
        'command_id': COMMAND_ID,
        'gate_id': gate_id,
        'profile': profile_name,
    }
    return {
        'schema_version': SOAK_SCHEMA_VERSION,
        'artifact_type': 'release_soak_redacted',
        'ok': False,
        'failure_code': reason_code,
        'release_receipt': receipt,
        'configuration': {
            'readers': int(options['readers']),
            'writers': 1,
            'warmup_searches': int(options['warmup_searches']),
            'duration_seconds': duration,
            'sample_interval_seconds': float(options['sample_interval_seconds']),
            'writer_interval_seconds': float(options['writer_interval_seconds']),
            'search_mode': 'exact_off',
            'read_workload': 'exact+contacts+context+evidence',
            'process_isolated': True,
        },
        'elapsed_seconds': 0.0,
        'counts': counts,
        'resources': _resource_summary(
            [],
            window_samples=int(options['resource_window_samples']),
            sustained_windows=int(options['sustained_windows']),
            growth_thresholds={
                'rss_bytes': int(options['rss_growth_threshold_bytes']),
                'fd_count': int(options['fd_growth_threshold']),
                'thread_count': int(options['thread_growth_threshold']),
            },
        ),
        'cleanup': {
            'readers_joined': False,
            'writer_joined': False,
            'runtime_closed': False,
            'temp_vault_removed': False,
            'worker_process_reaped': bool(reaped),
        },
        'assurance': {
            **RELEASE_ASSURANCE,
            'clean_sha_reproducible': bool(
                profile_name == 'release'
                and isinstance(subject, dict)
                and subject.get('dirty') is False
                and isinstance(subject.get('commit_sha'), str)
                and len(subject['commit_sha']) in {40, 64}
            ),
        },
        'privacy': privacy,
        'runtime_contract': dict(RUNTIME_CONTRACT),
    }


def run_release_soak_isolated(
    options: dict[str, Any],
    *,
    hard_timeout_seconds: float | None = None,
    worker_target: Callable[..., None] = _isolated_soak_worker,
) -> dict[str, Any]:
    duration = float(options['duration_seconds'])
    timeout = float(hard_timeout_seconds if hard_timeout_seconds is not None else duration + 120.0)
    if timeout <= 0:
        raise ValueError('hard_timeout_seconds must be positive')
    context = multiprocessing.get_context('spawn')
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(send_connection, dict(options)),
        name='trove-release-soak-worker',
        daemon=False,
    )
    process.start()
    send_connection.close()
    reason_code: str | None = None
    try:
        process.join(timeout=timeout)
        if process.is_alive():
            reason_code = 'worker_hard_timeout'
        elif process.exitcode != 0:
            reason_code = 'worker_failed'
        elif not receive_connection.poll(2.0):
            reason_code = 'worker_missing_report'
        else:
            try:
                payload = receive_connection.recv()
            except (EOFError, OSError):
                payload = None
            if isinstance(payload, dict) and payload.get('kind') == 'report' and isinstance(payload.get('report'), dict):
                return payload['report']
            reason_code = 'worker_failed'
    except (KeyboardInterrupt, SystemExit):
        reason_code = 'worker_interrupted'
    finally:
        reaped = _terminate_and_reap(process)
        receive_connection.close()
    return _isolated_failure_report(options, reason_code=reason_code or 'worker_failed', reaped=reaped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run a synthetic bounded release soak.')
    dogfood_mode = parser.add_mutually_exclusive_group()
    dogfood_mode.add_argument(
        '--dogfood-start-manifest', type=Path,
        help='Start the three-day gate for an already verified clean distribution manifest.',
    )
    dogfood_mode.add_argument(
        '--dogfood-client-start', type=Path,
        help='Record one content-free Agent client receipt against a dogfood start artifact.',
    )
    dogfood_mode.add_argument(
        '--dogfood-finalize-start', type=Path,
        help='Finalize a three-day dogfood record from its start artifact and client receipts.',
    )
    parser.add_argument('--dogfood-client-id')
    parser.add_argument('--dogfood-metrics-fd', type=int)
    parser.add_argument('--dogfood-client-receipt', type=Path, action='append', default=[])
    parser.add_argument('--duration-seconds', type=float, default=3600.0)
    parser.add_argument('--readers', type=int, default=32)
    parser.add_argument('--warmup-searches', type=int, default=8)
    parser.add_argument('--writer-interval-seconds', type=float, default=5.0)
    parser.add_argument('--sample-interval-seconds', type=float, default=5.0)
    parser.add_argument('--resource-window-samples', type=int, default=5)
    parser.add_argument('--sustained-windows', type=int, default=3)
    parser.add_argument('--rss-growth-threshold-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--fd-growth-threshold', type=int, default=8)
    parser.add_argument('--thread-growth-threshold', type=int, default=4)
    parser.add_argument(
        '--four-hour-fault-idle', action='store_true',
        help='Lock duration to four hours and run the bounded fault suite before the idle/resource soak.',
    )
    parser.add_argument('--out', help='Redacted artifact path; writes an independent .manifest.json sidecar.')
    args = parser.parse_args(argv)
    if args.dogfood_start_manifest or args.dogfood_client_start or args.dogfood_finalize_start:
        try:
            if not args.out:
                raise ValueError('dogfood evidence requires an output path')
            if args.dogfood_start_manifest:
                if args.dogfood_client_id or args.dogfood_metrics_fd is not None or args.dogfood_client_receipt:
                    raise ValueError('dogfood start received incompatible arguments')
                report = _dogfood_start_from_distribution(args.dogfood_start_manifest.expanduser())
            elif args.dogfood_client_start:
                if not args.dogfood_client_id or args.dogfood_metrics_fd is None or args.dogfood_client_receipt:
                    raise ValueError('dogfood client receipt requires id and metrics fd')
                start = _load_verified_dogfood_artifact(
                    args.dogfood_client_start.expanduser(),
                    'trove_v1_dogfood_start_redacted',
                )
                report = record_dogfood_client(
                    start,
                    args.dogfood_client_id,
                    _read_bounded_json_fd(args.dogfood_metrics_fd),
                )
            else:
                if args.dogfood_client_id or args.dogfood_metrics_fd is not None or len(args.dogfood_client_receipt) < 2:
                    raise ValueError('dogfood finalization requires at least two client receipts')
                start = _load_verified_dogfood_artifact(
                    args.dogfood_finalize_start.expanduser(),
                    'trove_v1_dogfood_start_redacted',
                )
                clients = [
                    _load_verified_dogfood_artifact(
                        path.expanduser(), 'trove_v1_dogfood_client_redacted',
                    )
                    for path in args.dogfood_client_receipt
                ]
                report = finalize_dogfood_record(start, clients)
            out = Path(args.out).expanduser()
            write_evidence_artifact(report, out)
            output = {
                'ok': True,
                'artifact_file': out.name,
                'manifest_file': evidence_manifest_path(out).name,
                'private_paths_printed': False,
                'raw_content_printed': False,
            }
        except Exception:
            print(json.dumps({'ok': False, 'error_code': 'dogfood_gate_failed'}, sort_keys=True))
            return 2
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    try:
        fault_suite_passed = True
        if args.four_hour_fault_idle:
            completed = subprocess.run(
                [
                    str(_ROOT / 'scripts' / 'trove-python'), '-m', 'unittest',
                    'tests.e2e.test_provider_fault_recovery',
                    'tests.e2e.test_operation_crash_recovery',
                    'tests.e2e.test_agent_concurrency',
                ],
                cwd=_ROOT, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=1800, check=False,
            )
            fault_suite_passed = completed.returncode == 0
        options = {
            'duration_seconds': 14_400.0 if args.four_hour_fault_idle else args.duration_seconds,
            'readers': args.readers,
            'warmup_searches': args.warmup_searches,
            'writer_interval_seconds': args.writer_interval_seconds,
            'sample_interval_seconds': args.sample_interval_seconds,
            'resource_window_samples': args.resource_window_samples,
            'sustained_windows': args.sustained_windows,
            'rss_growth_threshold_bytes': args.rss_growth_threshold_bytes,
            'fd_growth_threshold': args.fd_growth_threshold,
            'thread_growth_threshold': args.thread_growth_threshold,
        }
        previous_handlers: dict[int, Any] = {}

        def interrupt_handler(_signum, _frame):
            raise KeyboardInterrupt

        if threading.current_thread() is threading.main_thread():
            for signal_number in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, interrupt_handler)
        try:
            report = run_release_soak_isolated(options)
        finally:
            for signal_number, previous in previous_handlers.items():
                signal.signal(signal_number, previous)
        if args.four_hour_fault_idle:
            report['fault_idle_validation'] = {
                'profile': 'four_hour',
                'fault_suite_passed': fault_suite_passed,
                'idle_resource_sampling': True,
                'writer_conflict_injection': True,
            }
            if not fault_suite_passed:
                report['ok'] = False
                report['failure_code'] = 'fault_suite_failed'
        if args.out:
            out = Path(args.out).expanduser()
            write_evidence_artifact(report, out)
            output = {
                'ok': report['ok'],
                'artifact_file': out.name,
                'manifest_file': evidence_manifest_path(out).name,
                'counts': report['counts'],
                'private_paths_printed': False,
                'raw_content_printed': False,
            }
        else:
            output = report
    except ValueError:
        print(json.dumps({'ok': False, 'error_code': 'invalid_soak_configuration'}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({'ok': False, 'error_code': 'soak_artifact_write_failed'}, sort_keys=True))
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
