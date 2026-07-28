#!/usr/bin/env python3
"""Gate redacted retrieval-eval reports for private release checks.

Inputs are already-redacted matrix reports.  The gate first proves both
reports come from the same frozen case pack, same k, and same case-hash set.
It then compares the full matching set, optionally excluding known drift hashes
from regression checks only. Positive-expected cases drive hit/rank and recall
fallback checks; negative-only cases are gated through a separate pass-rate
floor so precision/recall semantics do not mix with exclusion checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT / 'scripts'))
from project_runtime_guard import ensure_project_runtime
ensure_project_runtime(__file__)

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Any

from trove_core.search.eval_schema import validate_redacted_artifact
from trove_core.search.evidence_provenance import (
    EvidenceProvenanceError,
    validate_artifact_provenance,
    verify_evidence_manifest,
    write_evidence_artifact,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().read_text(encoding='utf-8'))
    validate_redacted_artifact(data)
    if data.get('artifact_type') != 'retrieval_eval_matrix_redacted':
        raise SystemExit(f'not a retrieval eval matrix report: {path.name}')
    return data


def _load_benchmark(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().read_text(encoding='utf-8'))
    validate_redacted_artifact(data)
    if data.get('artifact_type') != 'search_benchmark_redacted':
        raise SystemExit(f'not a redacted search benchmark: {path.name}')
    return data


def _release_evidence_failures(path: Path, report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    try:
        validate_artifact_provenance(report.get('provenance'), release=True)
    except EvidenceProvenanceError:
        failures.append('invalid_or_missing_provenance')
    try:
        verify_evidence_manifest(path, required=True)
    except EvidenceProvenanceError:
        failures.append('invalid_or_missing_manifest')
    return failures


def _provenance_pair_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = baseline.get('provenance') if isinstance(baseline.get('provenance'), dict) else {}
    cand = candidate.get('provenance') if isinstance(candidate.get('provenance'), dict) else {}
    checks = {
        'platform_match': bool(base.get('platform')) and base.get('platform') == cand.get('platform'),
        'fixture_match': bool((base.get('fixture') or {}).get('sha256')) and (base.get('fixture') or {}).get('sha256') == (cand.get('fixture') or {}).get('sha256'),
        'seed_match': isinstance(base.get('seed'), int) and base.get('seed') == cand.get('seed'),
        'case_pack_match': bool(base.get('case_pack_sha256')) and base.get('case_pack_sha256') == cand.get('case_pack_sha256'),
        'execution_match': bool(base.get('execution')) and base.get('execution') == cand.get('execution'),
        'provider_match': bool(base.get('provider')) and base.get('provider') == cand.get('provider'),
        'schema_version_match': isinstance((base.get('store') or {}).get('schema_version'), int) and (base.get('store') or {}).get('schema_version') == (cand.get('store') or {}).get('schema_version'),
        'index_generation_match': bool((base.get('store') or {}).get('index_generation_sha256')) and (base.get('store') or {}).get('index_generation_sha256') == (cand.get('store') or {}).get('index_generation_sha256'),
        'document_count_match': isinstance((base.get('store') or {}).get('document_count'), int) and (base.get('store') or {}).get('document_count') == (cand.get('store') or {}).get('document_count'),
    }
    return {
        'ok': all(checks.values()),
        'checks': checks,
        'baseline_commit': (base.get('git') or {}).get('commit_sha'),
        'candidate_commit': (cand.get('git') or {}).get('commit_sha'),
    }


def _load_release_config(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SystemExit(f'release config contains non-finite number: {value}')

    data = json.loads(path.expanduser().read_text(encoding='utf-8'), parse_constant=reject_constant)
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise SystemExit('release config schema_version must be 1')
    quality = data.get('quality')
    latency = data.get('latency')
    if not isinstance(quality, dict) or not isinstance(latency, dict):
        raise SystemExit('release config requires quality and latency objects')
    required_quality = {
        'min_recall_at_k', 'min_negative_pass_rate', 'max_recall_drop',
        'max_mrr_drop', 'category_floors',
    }
    missing_quality = sorted(required_quality - set(quality))
    if missing_quality:
        raise SystemExit('release config missing quality fields: ' + ','.join(missing_quality))
    if not isinstance(quality.get('category_floors'), dict) or not quality['category_floors']:
        raise SystemExit('release config category_floors must be a non-empty object')
    for key in ('min_recall_at_k', 'min_negative_pass_rate', 'max_recall_drop', 'max_mrr_drop'):
        try:
            value = float(quality[key])
        except (TypeError, ValueError) as exc:
            raise SystemExit(f'release config quality.{key} must be numeric') from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f'release config quality.{key} must be between 0 and 1')
    for category, floor in quality['category_floors'].items():
        if not str(category).strip() or not isinstance(floor, dict) or not floor:
            raise SystemExit('release config category floors require named non-empty objects')
        unknown = set(floor) - {'min_recall_at_k', 'min_negative_pass_rate'}
        if unknown:
            raise SystemExit(f'release config category {category} has unknown fields: ' + ','.join(sorted(unknown)))
        for key, raw in floor.items():
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f'release config category {category}.{key} must be numeric') from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemExit(f'release config category {category}.{key} must be between 0 and 1')
    required_latency = {'exact_cold', 'exact_warm', 'rewrite_cold', 'rewrite_warm'}
    missing_latency = sorted(required_latency - set(latency))
    if missing_latency:
        raise SystemExit('release config missing latency profiles: ' + ','.join(missing_latency))
    unexpected_latency = sorted(set(latency) - required_latency)
    if unexpected_latency:
        raise SystemExit('release config has unexpected latency profiles: ' + ','.join(unexpected_latency))
    required_latency_fields = {
        'baseline_artifact', 'candidate_artifact', 'semantic_mode', 'temperature',
        'max_p50_ms', 'max_p95_ms', 'max_p50_regression_ratio', 'max_p95_regression_ratio',
    }
    for profile in sorted(required_latency):
        cfg = latency.get(profile)
        if not isinstance(cfg, dict):
            raise SystemExit(f'release config latency.{profile} must be an object')
        missing = sorted(required_latency_fields - set(cfg))
        if missing:
            raise SystemExit(f'release config latency.{profile} missing fields: ' + ','.join(missing))
        if not str(cfg['baseline_artifact']).strip() or not str(cfg['candidate_artifact']).strip():
            raise SystemExit(f'release config latency.{profile} artifact paths must be non-empty')
        expected_semantic = 'off' if profile.startswith('exact_') else 'on'
        expected_temperature = 'cold' if profile.endswith('_cold') else 'warm'
        if cfg['semantic_mode'] != expected_semantic or cfg['temperature'] != expected_temperature:
            raise SystemExit(f'release config latency.{profile} route semantics do not match its name')
        for key in ('max_p50_ms', 'max_p95_ms', 'max_p50_regression_ratio', 'max_p95_regression_ratio'):
            try:
                value = float(cfg[key])
            except (TypeError, ValueError) as exc:
                raise SystemExit(f'release config latency.{profile}.{key} must be numeric') from exc
            if not math.isfinite(value) or value <= 0.0:
                raise SystemExit(f'release config latency.{profile}.{key} must be finite and > 0')
    data['_config_root'] = str(path.expanduser().resolve().parent)
    return data


def _resolve_config_artifact(config: dict[str, Any], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(str(config.get('_config_root') or '.')) / path
    return path.resolve()


def _latency_profile_gate(profile: str, cfg: dict[str, Any], release_config: dict[str, Any]) -> dict[str, Any]:
    baseline_path = _resolve_config_artifact(release_config, cfg.get('baseline_artifact'))
    candidate_path = _resolve_config_artifact(release_config, cfg.get('candidate_artifact'))
    baseline = _load_benchmark(baseline_path)
    candidate = _load_benchmark(candidate_path)
    failures = [
        *['baseline_' + failure for failure in _release_evidence_failures(baseline_path, baseline)],
        *['candidate_' + failure for failure in _release_evidence_failures(candidate_path, candidate)],
    ]
    pair = _provenance_pair_gate(baseline, candidate)
    if not pair['ok']:
        failures.append('provenance_mismatch')
    expected_temperature = str(cfg.get('temperature'))
    expected_semantic = str(cfg.get('semantic_mode'))
    if ((baseline.get('provenance') or {}).get('execution') or {}).get('temperature') != expected_temperature:
        failures.append('baseline_temperature_mismatch')
    if ((candidate.get('provenance') or {}).get('execution') or {}).get('temperature') != expected_temperature:
        failures.append('candidate_temperature_mismatch')
    if str(baseline.get('semantic_mode')) != expected_semantic:
        failures.append('baseline_semantic_mode_mismatch')
    if str(candidate.get('semantic_mode')) != expected_semantic:
        failures.append('candidate_semantic_mode_mismatch')
    base_latency = baseline.get('latency_ms') or {}
    cand_latency = candidate.get('latency_ms') or {}
    base_p50 = float(base_latency.get('p50') or 0.0)
    base_p95 = float(base_latency.get('p95') or 0.0)
    cand_p50 = float(cand_latency.get('p50') or 0.0)
    cand_p95 = float(cand_latency.get('p95') or 0.0)
    thresholds = {
        'max_p50_ms': float(cfg['max_p50_ms']),
        'max_p95_ms': float(cfg['max_p95_ms']),
        'max_p50_regression_ratio': float(cfg['max_p50_regression_ratio']),
        'max_p95_regression_ratio': float(cfg['max_p95_regression_ratio']),
    }
    if cand_p50 > thresholds['max_p50_ms'] + 1e-12:
        failures.append('p50_absolute_limit_exceeded')
    if cand_p95 > thresholds['max_p95_ms'] + 1e-12:
        failures.append('p95_absolute_limit_exceeded')
    if base_p50 <= 0.0 or cand_p50 > base_p50 * thresholds['max_p50_regression_ratio'] + 1e-12:
        failures.append('p50_relative_limit_exceeded')
    if base_p95 <= 0.0 or cand_p95 > base_p95 * thresholds['max_p95_regression_ratio'] + 1e-12:
        failures.append('p95_relative_limit_exceeded')
    return {
        'ok': not failures,
        'profile': profile,
        'route': 'exact' if profile.startswith('exact_') else 'rewrite',
        'temperature': expected_temperature,
        'semantic_mode': expected_semantic,
        'baseline_file': baseline_path.name,
        'candidate_file': candidate_path.name,
        'baseline_latency_ms': {'p50': base_p50, 'p95': base_p95},
        'candidate_latency_ms': {'p50': cand_p50, 'p95': cand_p95},
        'thresholds': thresholds,
        'provenance_pair': pair,
        'failures': sorted(set(failures)),
    }


def _mode_cases(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    payload = (report.get('modes') or {}).get(mode)
    if not payload:
        raise SystemExit(f'mode not found in report: {mode}')
    return {str(case.get('case_hash')): case for case in (payload.get('cases') or []) if case.get('case_hash')}


def _mode_case_hashes(report: dict[str, Any], mode: str) -> list[str]:
    payload = (report.get('modes') or {}).get(mode)
    if not payload:
        raise SystemExit(f'mode not found in report: {mode}')
    return [str(case.get('case_hash')) for case in (payload.get('cases') or []) if case.get('case_hash')]


def _case_is_positive(case: dict[str, Any]) -> bool:
    if 'positive_expected' in case:
        return bool(case.get('positive_expected'))
    return bool(case.get('expected_citation_hashes'))


def _case_is_negative_only(case: dict[str, Any]) -> bool:
    if 'negative_only' in case:
        return bool(case.get('negative_only'))
    return not bool(case.get('expected_citation_hashes')) and str(case.get('category') or '') == 'negative_scope'


def _ignore_hashes(values: list[str] | None, files: list[str] | None) -> set[str]:
    out = {str(v).strip() for v in (values or []) if str(v).strip()}
    for file_name in files or []:
        text = Path(file_name).expanduser().read_text(encoding='utf-8').strip()
        if not text:
            continue
        if text.startswith('['):
            loaded = json.loads(text)
            out.update(str(v).strip() for v in loaded if str(v).strip())
        else:
            out.update(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#'))
    return out


def build_gate_report(
    baseline_path: Path,
    candidate_path: Path,
    *,
    mode: str,
    ignore_case_hashes: set[str] | None = None,
    min_recall_at_k: float | None = None,
    min_negative_pass_rate: float | None = None,
    require_case_quality_v2: bool = False,
    min_case_count: int | None = None,
    max_literal_substring_rate: float = 0.0,
    max_avg_word_overlap_ratio: float = 0.50,
    release_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    release_mode = release_config is not None
    release_quality = (release_config or {}).get('quality') or {}
    if release_mode:
        min_recall_at_k = float(release_quality['min_recall_at_k'])
        min_negative_pass_rate = float(release_quality['min_negative_pass_rate'])
    evidence_failures: list[str] = []
    provenance_pair = {'ok': True, 'checks': {}}
    if release_mode:
        evidence_failures.extend('baseline_' + failure for failure in _release_evidence_failures(baseline_path, baseline))
        evidence_failures.extend('candidate_' + failure for failure in _release_evidence_failures(candidate_path, candidate))
        provenance_pair = _provenance_pair_gate(baseline, candidate)
        if not provenance_pair['ok']:
            evidence_failures.append('quality_provenance_mismatch')
    ignored = ignore_case_hashes or set()
    base_hashes_raw = _mode_case_hashes(baseline, mode)
    cand_hashes_raw = _mode_case_hashes(candidate, mode)
    base_cases = _mode_cases(baseline, mode)
    cand_cases = _mode_cases(candidate, mode)
    base_hashes = set(base_hashes_raw)
    cand_hashes = set(cand_hashes_raw)
    baseline_only_hashes = sorted(base_hashes - cand_hashes)
    candidate_only_hashes = sorted(cand_hashes - base_hashes)
    duplicate_baseline = len(base_hashes_raw) - len(base_hashes)
    duplicate_candidate = len(cand_hashes_raw) - len(cand_hashes)
    baseline_anchor = str(((baseline.get('case_pack_anchor') or {}).get('sha256_prefix')) or '')
    candidate_anchor = str(((candidate.get('case_pack_anchor') or {}).get('sha256_prefix')) or '')
    baseline_k = baseline.get('k')
    candidate_k = candidate.get('k')
    frozen_pack_failures: list[str] = []
    if not baseline_anchor or not candidate_anchor or baseline_anchor != candidate_anchor:
        frozen_pack_failures.append('case_pack_anchor_mismatch')
    if baseline_k != candidate_k:
        frozen_pack_failures.append('k_mismatch')
    if baseline_only_hashes or candidate_only_hashes or duplicate_baseline or duplicate_candidate:
        frozen_pack_failures.append('case_set_mismatch')
    frozen_pack_ok = not frozen_pack_failures
    # Do not fall back to a common-subset comparison when the freeze anchor,
    # k, or case set diverges.  Regression checks are meaningful only after
    # the frozen-pack gate proves both reports are the same evaluation universe.
    common_hashes = sorted((base_hashes & cand_hashes) - ignored) if frozen_pack_ok else []
    positive_hashes = [
        case_hash for case_hash in common_hashes
        if _case_is_positive(base_cases[case_hash]) and _case_is_positive(cand_cases[case_hash])
    ]
    negative_hashes = [
        case_hash for case_hash in common_hashes
        if _case_is_negative_only(base_cases[case_hash]) and _case_is_negative_only(cand_cases[case_hash])
    ]
    hit_regressions = [
        case_hash for case_hash in positive_hashes
        if bool(base_cases[case_hash].get('hit')) and not bool(cand_cases[case_hash].get('hit'))
    ]
    rank_regressions = [
        case_hash for case_hash in positive_hashes
        if bool(base_cases[case_hash].get('hit'))
        and bool(cand_cases[case_hash].get('hit'))
        and float(cand_cases[case_hash].get('reciprocal_rank') or 0.0) + 1e-12 < float(base_cases[case_hash].get('reciprocal_rank') or 0.0)
    ]
    negative_regressions = [
        case_hash for case_hash in negative_hashes
        if bool(base_cases[case_hash].get('hit')) and not bool(cand_cases[case_hash].get('hit'))
    ]
    baseline_metrics = ((baseline.get('modes') or {}).get(mode) or {}).get('metrics') or {}
    candidate_metrics = ((candidate.get('modes') or {}).get(mode) or {}).get('metrics') or {}
    recall_key = f"recall_at_{candidate.get('k', 3)}"
    candidate_recall = float(candidate_metrics.get(recall_key) or 0.0)
    baseline_recall = float(baseline_metrics.get(recall_key) or 0.0)
    baseline_mrr = float(baseline_metrics.get('mrr') or 0.0)
    candidate_mrr = float(candidate_metrics.get('mrr') or 0.0)
    recall_floor_miss = min_recall_at_k is not None and candidate_recall + 1e-12 < float(min_recall_at_k)
    baseline_negative_pass_rate = baseline_metrics.get('negative_pass_rate')
    candidate_negative_pass_rate = candidate_metrics.get('negative_pass_rate')
    negative_pass_rate_floor = (
        min_negative_pass_rate
        if min_negative_pass_rate is not None
        else baseline_negative_pass_rate
    )
    negative_pass_rate_floor_miss = (
        negative_pass_rate_floor is not None
        and (candidate_negative_pass_rate is None or float(candidate_negative_pass_rate) + 1e-12 < float(negative_pass_rate_floor))
    )
    max_recall_drop = float(release_quality['max_recall_drop']) if release_mode else None
    max_mrr_drop = float(release_quality['max_mrr_drop']) if release_mode else None
    recall_relative_miss = bool(release_mode and baseline_recall - candidate_recall > max_recall_drop + 1e-12)
    mrr_relative_miss = bool(release_mode and baseline_mrr - candidate_mrr > max_mrr_drop + 1e-12)
    category_failures: list[str] = []
    category_results: dict[str, Any] = {}
    if release_mode:
        category_floors = release_quality.get('category_floors') or {}
        candidate_categories = candidate_metrics.get('per_category') or {}
        for category in sorted(set(category_floors) | set(candidate_categories)):
            metrics = candidate_categories.get(category)
            floor = category_floors.get(category)
            failures: list[str] = []
            if not isinstance(metrics, dict):
                failures.append('missing_category_metrics')
            if not isinstance(floor, dict):
                failures.append('missing_category_floor')
            elif isinstance(metrics, dict):
                if int(metrics.get('positive_queries') or 0) > 0:
                    if 'min_recall_at_k' not in floor:
                        failures.append('missing_recall_floor')
                    elif float(metrics.get(recall_key) or 0.0) + 1e-12 < float(floor['min_recall_at_k']):
                        failures.append('recall_floor_miss')
                if int(metrics.get('negative_only_queries') or 0) > 0:
                    if 'min_negative_pass_rate' not in floor:
                        failures.append('missing_negative_floor')
                    elif metrics.get('negative_pass_rate') is None or float(metrics['negative_pass_rate']) + 1e-12 < float(floor['min_negative_pass_rate']):
                        failures.append('negative_floor_miss')
            if failures:
                category_failures.append(category)
            category_results[category] = {'ok': not failures, 'failures': failures, 'thresholds': floor}
    latency_profiles: dict[str, Any] = {}
    if release_mode:
        for profile, cfg in sorted((release_config.get('latency') or {}).items()):
            if profile.startswith('_'):
                continue
            latency_profiles[profile] = _latency_profile_gate(profile, cfg, release_config)
    latency_ok = all(item.get('ok') for item in latency_profiles.values()) if release_mode else True
    quality = candidate.get('case_quality') or {}
    quality_present = bool(quality)
    quality_failures: list[str] = []
    if require_case_quality_v2 and not quality_present:
        quality_failures.append('missing_case_quality')
    if quality_present:
        if int(quality.get('cases') or 0) < int(min_case_count or 0):
            quality_failures.append('case_count_below_min')
        if float(quality.get('literal_substring_rate') or 0.0) > max_literal_substring_rate + 1e-12:
            quality_failures.append('literal_substring_rate_above_max')
        if float(quality.get('avg_word_overlap_ratio') or 0.0) > max_avg_word_overlap_ratio + 1e-12:
            quality_failures.append('avg_word_overlap_ratio_above_max')
    ok = (
        not hit_regressions
        and not rank_regressions
        and not negative_regressions
        and not recall_floor_miss
        and not negative_pass_rate_floor_miss
        and not quality_failures
        and frozen_pack_ok
        and bool(candidate.get('complete'))
        and not evidence_failures
        and provenance_pair.get('ok', False)
        and not recall_relative_miss
        and not mrr_relative_miss
        and not category_failures
        and latency_ok
    )
    report = {
        'schema_version': 1,
        'artifact_type': 'retrieval_eval_gate_redacted',
        'created_at': now_iso(),
        'ok': ok,
        'release_mode': release_mode,
        'mode': mode,
        'baseline': baseline_path.name,
        'candidate': candidate_path.name,
        'candidate_complete': bool(candidate.get('complete')),
        'case_counts': {
            'baseline': len(base_cases),
            'candidate': len(cand_cases),
            'common_compared': len(common_hashes),
            'positive_compared': len(positive_hashes),
            'negative_only_compared': len(negative_hashes),
            'ignored': len(ignored),
            'baseline_only': len(baseline_only_hashes),
            'candidate_only': len(candidate_only_hashes),
            'baseline_duplicate_hashes': duplicate_baseline,
            'candidate_duplicate_hashes': duplicate_candidate,
        },
        'frozen_pack_gate': {
            'ok': frozen_pack_ok,
            'failures': frozen_pack_failures,
            'case_pack_anchor_sha256_prefix': {
                'baseline': baseline_anchor or None,
                'candidate': candidate_anchor or None,
                'match': bool(baseline_anchor and candidate_anchor and baseline_anchor == candidate_anchor),
            },
            'k': {
                'baseline': baseline_k,
                'candidate': candidate_k,
                'match': baseline_k == candidate_k,
            },
            'case_set': {
                'match': not baseline_only_hashes and not candidate_only_hashes and not duplicate_baseline and not duplicate_candidate,
                'baseline_count': len(base_cases),
                'candidate_count': len(cand_cases),
                'baseline_only_count': len(baseline_only_hashes),
                'candidate_only_count': len(candidate_only_hashes),
                'baseline_duplicate_hash_count': duplicate_baseline,
                'candidate_duplicate_hash_count': duplicate_candidate,
                'baseline_only_case_hashes': baseline_only_hashes[:100],
                'candidate_only_case_hashes': candidate_only_hashes[:100],
            },
        },
        'candidate_metrics': {
            recall_key: candidate_recall,
            'mrr': candidate_mrr,
            'negative_pass_rate': candidate_negative_pass_rate,
            'case_success_rate': candidate_metrics.get('case_success_rate'),
            'positive_queries': candidate_metrics.get('positive_queries'),
            'negative_only_queries': candidate_metrics.get('negative_only_queries'),
            'failure_classes': candidate_metrics.get('failure_classes'),
            'per_category': candidate_metrics.get('per_category'),
        },
        'evidence_gate': {
            'ok': not evidence_failures and provenance_pair.get('ok', False),
            'failures': sorted(set(evidence_failures)),
            'provenance_pair': provenance_pair,
            'manifests_required': release_mode,
        },
        'relative_quality_gate': {
            'ok': not recall_relative_miss and not mrr_relative_miss,
            'baseline_recall_at_k': baseline_recall,
            'candidate_recall_at_k': candidate_recall,
            'baseline_mrr': baseline_mrr,
            'candidate_mrr': candidate_mrr,
            'max_recall_drop': max_recall_drop,
            'max_mrr_drop': max_mrr_drop,
            'recall_floor_miss': recall_relative_miss,
            'mrr_floor_miss': mrr_relative_miss,
        },
        'category_quality_gate': {
            'ok': not category_failures,
            'failed_categories': category_failures,
            'categories': category_results,
        },
        'latency_gate': {
            'ok': latency_ok,
            'required_profiles': ['exact_cold', 'exact_warm', 'rewrite_cold', 'rewrite_warm'] if release_mode else [],
            'profiles': latency_profiles,
        },
        'case_quality_gate': {
            'required': bool(require_case_quality_v2),
            'present': quality_present,
            'ok': not quality_failures,
            'failures': quality_failures,
            'thresholds': {
                'min_case_count': min_case_count,
                'max_literal_substring_rate': max_literal_substring_rate,
                'max_avg_word_overlap_ratio': max_avg_word_overlap_ratio,
            },
            'candidate': {
                'cases': quality.get('cases'),
                'literal_substring_rate': quality.get('literal_substring_rate'),
                'avg_word_overlap_ratio': quality.get('avg_word_overlap_ratio'),
                'max_word_overlap_ratio': quality.get('max_word_overlap_ratio'),
                'query_types': quality.get('query_types'),
            },
        },
        'thresholds': {
            'min_recall_at_k': min_recall_at_k,
            'min_negative_pass_rate': min_negative_pass_rate,
            'effective_negative_pass_rate_floor': negative_pass_rate_floor,
            'zero_hit_regressions': True,
            'zero_rank_regressions': True,
            'zero_negative_regressions': True,
            'same_frozen_case_pack_required': True,
        },
        'hit_regressions': len(hit_regressions),
        'rank_regressions': len(rank_regressions),
        'negative_regressions': len(negative_regressions),
        'hit_regression_case_hashes': hit_regressions[:100],
        'rank_regression_case_hashes': rank_regressions[:100],
        'negative_regression_case_hashes': negative_regressions[:100],
        'recall_floor_miss': bool(recall_floor_miss),
        'negative_pass_rate_floor_miss': bool(negative_pass_rate_floor_miss),
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
            'token_values_included': False,
        },
    }
    validate_redacted_artifact(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='Baseline redacted matrix report.')
    parser.add_argument('--candidate', required=True, help='Candidate redacted matrix report.')
    parser.add_argument('--mode', default='hybrid-weighted')
    parser.add_argument('--ignore-case-hash', action='append', default=[])
    parser.add_argument('--ignore-case-hashes-file', action='append', default=[])
    parser.add_argument('--min-recall-at-k', type=float)
    parser.add_argument('--min-negative-pass-rate', type=float)
    parser.add_argument('--require-case-quality-v2', action='store_true')
    parser.add_argument('--min-case-count', type=int)
    parser.add_argument('--max-literal-substring-rate', type=float, default=0.0)
    parser.add_argument('--max-avg-word-overlap-ratio', type=float, default=0.50)
    parser.add_argument('--release-config', help='Explicit release thresholds plus exact/rewrite cold/warm benchmark artifact pairs.')
    parser.add_argument('--out', help='Optional redacted gate report path.')
    args = parser.parse_args(argv)
    release_config = _load_release_config(Path(args.release_config)) if args.release_config else None
    report = build_gate_report(
        Path(args.baseline),
        Path(args.candidate),
        mode=args.mode,
        ignore_case_hashes=_ignore_hashes(args.ignore_case_hash, args.ignore_case_hashes_file),
        min_recall_at_k=args.min_recall_at_k,
        min_negative_pass_rate=args.min_negative_pass_rate,
        require_case_quality_v2=args.require_case_quality_v2,
        min_case_count=args.min_case_count,
        max_literal_substring_rate=args.max_literal_substring_rate,
        max_avg_word_overlap_ratio=args.max_avg_word_overlap_ratio,
        release_config=release_config,
    )
    if args.out:
        out = Path(args.out).expanduser()
        write_evidence_artifact(report, out)
    print(json.dumps({
        'ok': report['ok'],
        'mode': report['mode'],
        'common_compared': report['case_counts']['common_compared'],
        'positive_compared': report['case_counts']['positive_compared'],
        'negative_only_compared': report['case_counts']['negative_only_compared'],
        'ignored': report['case_counts']['ignored'],
        'hit_regressions': report['hit_regressions'],
        'rank_regressions': report['rank_regressions'],
        'negative_regressions': report['negative_regressions'],
        'recall_floor_miss': report['recall_floor_miss'],
        'negative_pass_rate_floor_miss': report['negative_pass_rate_floor_miss'],
        'frozen_pack_ok': report['frozen_pack_gate']['ok'],
        'baseline_only_cases': report['case_counts']['baseline_only'],
        'candidate_only_cases': report['case_counts']['candidate_only'],
        'anchor_match': report['frozen_pack_gate']['case_pack_anchor_sha256_prefix']['match'],
        'k_match': report['frozen_pack_gate']['k']['match'],
        'case_quality_ok': report['case_quality_gate']['ok'],
        'release_mode': report['release_mode'],
        'evidence_ok': report['evidence_gate']['ok'],
        'category_quality_ok': report['category_quality_gate']['ok'],
        'latency_ok': report['latency_gate']['ok'],
        'redacted_file': Path(args.out).name if args.out else None,
        'raw_queries_printed': False,
        'raw_snippets_printed': False,
        'private_paths_printed': False,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
