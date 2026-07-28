from __future__ import annotations

import math
from typing import Any

from .eval_schema import validate_redacted_artifact

WRITE_COST_METRICS = (
    'elapsed_ms',
    'bytes_written',
    'fts_rows_written',
    'vector_rows_written',
)

DEFAULT_WRITE_COST_RATIOS = {
    'elapsed_ms': 1.10,
    'bytes_written': 1.00,
    'fts_rows_written': 1.00,
    'vector_rows_written': 1.00,
}


def _mode_payload(report: dict[str, Any], mode: str) -> dict[str, Any]:
    value = (report.get('modes') or {}).get(mode)
    return value if isinstance(value, dict) else {}


def _case_map(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    return {
        str(case['case_hash']): case
        for case in (_mode_payload(report, mode).get('cases') or [])
        if isinstance(case, dict) and case.get('case_hash')
    }


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _ratio(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else math.inf
    return candidate / baseline


def evaluate_retrieval_document_migration(
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    baseline_write_cost: dict[str, Any],
    candidate_write_cost: dict[str, Any],
    *,
    mode: str = 'hybrid-weighted',
    max_write_cost_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Gate any canonical-document or double-FTS migration.

    Passing this gate only makes a candidate eligible for a reviewed migration;
    it never mutates an index or enables a strategy automatically.
    """

    failures: list[str] = []
    expected_eval_type = 'retrieval_eval_matrix_redacted'
    for label, report in (('baseline', baseline_eval), ('candidate', candidate_eval)):
        if report.get('artifact_type') != expected_eval_type or not report.get('complete'):
            failures.append(f'{label}_eval_missing_or_incomplete')
    for label, report in (('baseline', baseline_write_cost), ('candidate', candidate_write_cost)):
        if report.get('artifact_type') != 'retrieval_write_cost_redacted' or not report.get('complete'):
            failures.append(f'{label}_write_cost_missing_or_incomplete')

    base_anchor = str((baseline_eval.get('case_pack_anchor') or {}).get('sha256_prefix') or '')
    cand_anchor = str((candidate_eval.get('case_pack_anchor') or {}).get('sha256_prefix') or '')
    base_cases = _case_map(baseline_eval, mode)
    cand_cases = _case_map(candidate_eval, mode)
    frozen_checks = {
        'case_pack_match': bool(base_anchor) and base_anchor == cand_anchor,
        'k_match': baseline_eval.get('k') == candidate_eval.get('k'),
        'case_set_match': bool(base_cases) and set(base_cases) == set(cand_cases),
    }
    if not all(frozen_checks.values()):
        failures.append('frozen_ab_evidence_mismatch')

    hit_regressions = 0
    rank_regressions = 0
    negative_regressions = 0
    if all(frozen_checks.values()):
        for case_hash, baseline_case in base_cases.items():
            candidate_case = cand_cases[case_hash]
            negative_only = bool(baseline_case.get('negative_only'))
            if baseline_case.get('hit') and not candidate_case.get('hit'):
                if negative_only:
                    negative_regressions += 1
                else:
                    hit_regressions += 1
            if (
                not negative_only
                and baseline_case.get('hit')
                and candidate_case.get('hit')
                and float(candidate_case.get('reciprocal_rank') or 0.0) + 1e-12
                < float(baseline_case.get('reciprocal_rank') or 0.0)
            ):
                rank_regressions += 1
    if hit_regressions:
        failures.append('positive_hit_regression')
    if rank_regressions:
        failures.append('rank_regression')
    if negative_regressions:
        failures.append('negative_case_regression')

    baseline_metrics = _mode_payload(baseline_eval, mode).get('metrics') or {}
    candidate_metrics = _mode_payload(candidate_eval, mode).get('metrics') or {}
    recall_key = f"recall_at_{candidate_eval.get('k', 3)}"
    metric_checks: dict[str, bool] = {}
    for key in (recall_key, 'mrr'):
        base_value = _finite_nonnegative(baseline_metrics.get(key))
        cand_value = _finite_nonnegative(candidate_metrics.get(key))
        metric_checks[key] = base_value is not None and cand_value is not None and cand_value + 1e-12 >= base_value
    base_negative = baseline_metrics.get('negative_pass_rate')
    cand_negative = candidate_metrics.get('negative_pass_rate')
    metric_checks['negative_pass_rate'] = (
        base_negative is None and cand_negative is None
    ) or (
        _finite_nonnegative(base_negative) is not None
        and _finite_nonnegative(cand_negative) is not None
        and float(cand_negative) + 1e-12 >= float(base_negative)
    )
    category_checks: dict[str, bool] = {}
    base_categories = baseline_metrics.get('per_category') or {}
    cand_categories = candidate_metrics.get('per_category') or {}
    for category, base_category in base_categories.items():
        candidate_category = cand_categories.get(category) or {}
        checks: list[bool] = []
        if int(base_category.get('positive_queries') or 0):
            checks.append(
                _finite_nonnegative(candidate_category.get(recall_key)) is not None
                and float(candidate_category.get(recall_key)) + 1e-12 >= float(base_category.get(recall_key) or 0.0)
            )
        if int(base_category.get('negative_only_queries') or 0):
            checks.append(
                _finite_nonnegative(candidate_category.get('negative_pass_rate')) is not None
                and float(candidate_category.get('negative_pass_rate')) + 1e-12 >= float(base_category.get('negative_pass_rate') or 0.0)
            )
        category_checks[str(category)] = bool(checks) and all(checks)
    if not all(metric_checks.values()) or not all(category_checks.values()):
        failures.append('quality_or_category_regression')

    ratio_limits = dict(DEFAULT_WRITE_COST_RATIOS)
    ratio_limits.update(max_write_cost_ratios or {})
    write_ratios: dict[str, float | None] = {}
    write_checks: dict[str, bool] = {}
    base_write_metrics = baseline_write_cost.get('metrics') or {}
    cand_write_metrics = candidate_write_cost.get('metrics') or {}
    for key in WRITE_COST_METRICS:
        base_value = _finite_nonnegative(base_write_metrics.get(key))
        cand_value = _finite_nonnegative(cand_write_metrics.get(key))
        limit = _finite_nonnegative(ratio_limits.get(key))
        if base_value is None or cand_value is None or limit is None:
            write_ratios[key] = None
            write_checks[key] = False
            continue
        ratio = _ratio(base_value, cand_value)
        write_ratios[key] = round(ratio, 6) if math.isfinite(ratio) else None
        write_checks[key] = math.isfinite(ratio) and ratio <= limit + 1e-12
    if not all(write_checks.values()):
        failures.append('write_cost_regression')

    ok = not failures
    report = {
        'schema_version': 1,
        'artifact_type': 'retrieval_document_migration_gate_redacted',
        'ok': ok,
        'decision': 'eligible_for_manual_migration' if ok else 'retain_current_retrieval_documents',
        'automatic_migration': False,
        'mode': mode,
        'frozen_ab_gate': {'ok': all(frozen_checks.values()), 'checks': frozen_checks},
        'quality_gate': {
            'ok': not any(failure in failures for failure in (
                'positive_hit_regression', 'rank_regression', 'negative_case_regression',
                'quality_or_category_regression',
            )),
            'hit_regressions': hit_regressions,
            'rank_regressions': rank_regressions,
            'negative_regressions': negative_regressions,
            'metric_checks': metric_checks,
            'category_checks': category_checks,
        },
        'write_cost_gate': {
            'ok': all(write_checks.values()),
            'ratios': write_ratios,
            'maximum_ratios': ratio_limits,
            'checks': write_checks,
        },
        'failures': sorted(set(failures)),
        'privacy': {
            'raw_queries_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
        },
    }
    validate_redacted_artifact(report)
    return report
