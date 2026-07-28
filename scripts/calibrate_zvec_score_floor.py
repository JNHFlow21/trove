#!/usr/bin/env python3
"""Generate and optionally install a manifested ZVEC score-floor artifact.

The calibration set must be an explicit fixed ``dev`` split. Positive cases
anchor the lowest acceptable relevant score; true ``negative_no_results``
cases anchor the highest known irrelevant score. Holdout cases are rejected.
Only aggregate scores, counts, hashes, and provenance leave the process.
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
from typing import Any

from trove_core.runtime import configured_embedding_provider, zvec_collection_path_for_provider, zvec_ledger_backend
from trove_core.search.eval_schema import (
    expected_citations,
    load_case_pack,
    stable_hash,
    validate_redacted_artifact,
)
from trove_core.search.evidence_provenance import (
    build_artifact_provenance,
    evidence_manifest_path,
    sha256_file,
    verify_evidence_manifest,
    write_evidence_artifact,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.generation import coordinated_vault_generation_publish, vault_generation_read
from trove_core.vector.score_calibration import build_score_calibration_artifact, index_identity
from trove_core.vector.zvec_store import ZVEC_ADAPTIVE_OVERFETCH_MAX, ZVecStore


ROOT = Path(__file__).resolve().parents[1]


class CalibrationInputError(ValueError):
    pass


def _hashes(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if value}


def load_fixed_dev_split(path: Path) -> tuple[set[str], set[str]]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationInputError('invalid calibration split manifest') from exc
    splits = payload.get('splits') if isinstance(payload, dict) else None
    dev = splits.get('dev') if isinstance(splits, dict) else None
    if not isinstance(dev, dict):
        raise CalibrationInputError('calibration split manifest requires splits.dev')
    positive = _hashes(dev.get('positive_case_hashes'))
    negative = _hashes(dev.get('negative_case_hashes'))
    if not positive or not negative:
        raise CalibrationInputError('dev calibration requires positive and negative case hashes')
    holdout: set[str] = set()
    for name, split in splits.items():
        if name == 'dev' or not isinstance(split, dict):
            continue
        holdout.update(_hashes(split.get('positive_case_hashes')))
        holdout.update(_hashes(split.get('case_hashes')))
    if positive.intersection(holdout) or negative.intersection(holdout):
        raise CalibrationInputError('dev calibration hashes overlap a holdout split')
    if positive.intersection(negative):
        raise CalibrationInputError('positive and negative dev calibration hashes overlap')
    return positive, negative


def _row_value(row: Any, key: str) -> str | None:
    try:
        if hasattr(row, 'keys') and key in row.keys():
            value = row[key]
            return str(value) if value else None
    except Exception:
        return None
    if isinstance(row, dict):
        value = row.get(key)
        return str(value) if value else None
    return None


def _matches_expected(row: Any, expected: set[str]) -> bool:
    return any(
        value in expected
        for value in (
            _row_value(row, 'citation'),
            _row_value(row, 'parent_citation'),
        )
        if value
    )


def collect_dev_score_bounds(
    zvec: ZVecStore,
    provider: Any,
    cases: list[dict[str, Any]],
    *,
    positive_hashes: set[str],
    negative_hashes: set[str],
    candidate_limit: int,
    allow_missing_positive_targets: bool = False,
    allow_unseparable_positive_targets: bool = False,
) -> tuple[float, float, int, int, list[str], list[str]]:
    by_hash = {str(stable_hash(case.get('case_id'))): case for case in cases}
    missing = sorted((positive_hashes | negative_hashes).difference(by_hash))
    if missing:
        raise CalibrationInputError(f'calibration split references {len(missing)} missing cases')
    positive_scores: list[tuple[str, float]] = []
    negative_scores: list[float] = []
    missing_positive_hashes: list[str] = []
    for case_hash in sorted(positive_hashes):
        case = by_hash[case_hash]
        expected = set(expected_citations(case))
        if not expected:
            raise CalibrationInputError('positive calibration case requires expected citations')
        candidates = zvec.calibration_candidates(
            str(case['query']),
            filters=dict(case.get('filters') or {}),
            limit=candidate_limit,
            provider=provider,
        )
        relevant = [score for row, score in candidates if _matches_expected(row, expected)]
        if not relevant:
            if allow_missing_positive_targets:
                missing_positive_hashes.append(case_hash)
                continue
            raise CalibrationInputError('positive calibration target is absent from the bounded vector window')
        positive_scores.append((case_hash, max(relevant)))
    for case_hash in sorted(negative_hashes):
        case = by_hash[case_hash]
        oracle = case.get('oracle') or {}
        if oracle.get('negative_no_results') is not True:
            raise CalibrationInputError('score-floor negatives must use the negative_no_results oracle')
        candidates = zvec.calibration_candidates(
            str(case['query']),
            filters=dict(case.get('filters') or {}),
            limit=candidate_limit,
            provider=provider,
        )
        if not candidates:
            raise CalibrationInputError('negative calibration case produced no scored candidates')
        negative_scores.append(max(score for _row, score in candidates))
    if not positive_scores:
        raise CalibrationInputError('dev calibration has no positive targets inside the bounded vector window')
    max_negative = max(negative_scores)
    unseparable_positive_hashes: list[str] = []
    if max_negative >= min(score for _hash, score in positive_scores):
        if not allow_unseparable_positive_targets:
            return max_negative, min(score for _hash, score in positive_scores), len(positive_scores), len(negative_scores)
        kept = [(case_hash, score) for case_hash, score in positive_scores if score > max_negative]
        unseparable_positive_hashes = [case_hash for case_hash, score in positive_scores if score <= max_negative]
        if not kept:
            raise CalibrationInputError('dev calibration has no positive targets separable from negatives')
        positive_scores = kept
    result = (
        max_negative,
        min(score for _hash, score in positive_scores),
        len(positive_scores),
        len(negative_scores),
    )
    if allow_missing_positive_targets or allow_unseparable_positive_targets:
        return (*result, missing_positive_hashes, unseparable_positive_hashes)
    return result


def collect_revision_bound_dev_score_bounds(
    zvec: ZVecStore,
    provider: Any,
    cases: list[dict[str, Any]],
    *,
    positive_hashes: set[str],
    negative_hashes: set[str],
    candidate_limit: int,
    allow_missing_positive_targets: bool = False,
    allow_unseparable_positive_targets: bool = False,
) -> tuple[tuple[float, float, int, int, list[str], list[str]], dict[str, Any]]:
    """Collect one fixed-dev sample from exactly one score-domain revision."""

    before = zvec._authoritative_score_metadata()
    before_identity = index_identity(before)
    bounds = collect_dev_score_bounds(
        zvec,
        provider,
        cases,
        positive_hashes=positive_hashes,
        negative_hashes=negative_hashes,
        candidate_limit=candidate_limit,
        allow_missing_positive_targets=allow_missing_positive_targets,
        allow_unseparable_positive_targets=allow_unseparable_positive_targets,
    )
    after = zvec._authoritative_score_metadata()
    if index_identity(after) != before_identity:
        raise CalibrationInputError('vector generation changed during score calibration')
    return bounds, after


def _safe_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise CalibrationInputError('calibration artifacts must be written outside the source repository')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--dev-split-manifest', required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--cloud', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--candidate-limit', type=int, default=ZVEC_ADAPTIVE_OVERFETCH_MAX)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixture-id', default='synthetic_or_redacted')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument(
        '--allow-missing-positive-targets',
        action='store_true',
        help='Exclude dev positives absent from the bounded dense vector window; record redacted hashes in the artifact.',
    )
    parser.add_argument(
        '--allow-unseparable-positive-targets',
        action='store_true',
        help='Exclude dev positives whose dense score is not separable from dev negatives; record redacted hashes in the artifact.',
    )
    args = parser.parse_args(argv)

    cfg = VaultConfig.resolve(args.vault)
    case_path = Path(args.cases).expanduser().resolve()
    split_path = Path(args.dev_split_manifest).expanduser().resolve()
    out = _safe_out(Path(args.out))
    if not args.cloud and not args.model_path:
        raise CalibrationInputError('--model-path is required unless --cloud is selected')
    provider = configured_embedding_provider(
        args.model_path,
        strict=True,
        vault_root=cfg.root,
        prefer_cloud=bool(args.cloud),
    )
    if provider is None:
        raise CalibrationInputError('a local embedding provider is required')
    store = SQLiteStore(cfg.paths.sqlite_path)
    zvec = ZVecStore(
        zvec_collection_path_for_provider(cfg, provider),
        store=store,
        ledger_backend=zvec_ledger_backend(provider),
    )
    positive_hashes, negative_hashes = load_fixed_dev_split(split_path)
    cases = load_case_pack(case_path)
    candidate_limit = max(1, min(int(args.candidate_limit), ZVEC_ADAPTIVE_OVERFETCH_MAX))
    with vault_generation_read(cfg):
        status = zvec.status(provider=provider)
        if not status.get('complete') or status.get('rebuild_required'):
            raise CalibrationInputError('a complete model-matched ZVEC generation is required')
        bounds, metadata = collect_revision_bound_dev_score_bounds(
            zvec,
            provider,
            cases,
            positive_hashes=positive_hashes,
            negative_hashes=negative_hashes,
            candidate_limit=candidate_limit,
            allow_missing_positive_targets=bool(args.allow_missing_positive_targets),
            allow_unseparable_positive_targets=bool(args.allow_unseparable_positive_targets),
        )
        if len(bounds) == 4:
            max_negative, min_positive, positive_count, negative_count = bounds
            missing_positive_hashes = []
            unseparable_positive_hashes = []
        else:
            (
                max_negative,
                min_positive,
                positive_count,
                negative_count,
                missing_positive_hashes,
                unseparable_positive_hashes,
            ) = bounds
        provenance = build_artifact_provenance(
            repo_root=ROOT,
            sqlite_path=cfg.paths.sqlite_path,
            case_pack_path=case_path,
            seed=int(args.seed),
            fixture_id=str(args.fixture_id),
            provider=provider,
            temperature='warm',
            warmups=0,
            rounds=positive_count + negative_count,
        )
        report = build_score_calibration_artifact(
            metadata=metadata,
            provider=provider,
            max_negative_top_score=max_negative,
            min_positive_target_score=min_positive,
            positive_case_count=positive_count,
            negative_case_count=negative_count,
            case_pack_sha256=sha256_file(case_path),
            split_manifest_sha256=sha256_file(split_path),
            provenance=provenance,
        )
        if missing_positive_hashes:
            report['dev_evidence']['excluded_missing_positive_count'] = len(missing_positive_hashes)
            report['dev_evidence']['excluded_missing_positive_hashes'] = list(missing_positive_hashes)
        if unseparable_positive_hashes:
            report['dev_evidence']['excluded_unseparable_positive_count'] = len(unseparable_positive_hashes)
            report['dev_evidence']['excluded_unseparable_positive_hashes'] = list(unseparable_positive_hashes)
    validate_redacted_artifact(report)
    write_evidence_artifact(report, out)
    verify_evidence_manifest(out, required=True)
    installed = None
    if args.apply:
        with coordinated_vault_generation_publish(cfg, operation='vector-calibrate') as publication:
            # Installation is one atomic metadata-sidecar replacement.  The
            # old or new generation is complete even if validation or replace
            # raises, so calibration failures must not strand a global
            # generation-recovery marker that blocks every subsequent read.
            publication.mark_consistent()
            installed = zvec.apply_score_calibration_file(out, provider=provider)
    summary = {
        'ok': True,
        'applied': bool(installed),
        'artifact_file': out.name,
        'manifest_file': evidence_manifest_path(out).name,
        'positive_case_count': positive_count,
        'negative_no_result_case_count': negative_count,
        'calibration_state': (installed or {}).get('state') if installed else 'generated',
        'raw_queries_printed': False,
        'raw_citations_printed': False,
        'private_paths_printed': False,
        'holdout_observations_used': False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
