#!/usr/bin/env python3
"""Gate canonical retrieval-document or double-FTS A/B migrations."""
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

from trove_core.search.eval_schema import validate_redacted_artifact
from trove_core.search.evidence_provenance import (
    EvidenceProvenanceError,
    validate_artifact_provenance,
    verify_evidence_manifest,
    write_evidence_artifact,
)
from trove_core.search.retrieval_document_gate import evaluate_retrieval_document_migration


def _load(path: Path, artifact_type: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding='utf-8'))
    validate_redacted_artifact(payload)
    if payload.get('artifact_type') != artifact_type:
        raise SystemExit(f'invalid artifact type: expected {artifact_type}')
    return payload


def _verify_release_evidence(path: Path, payload: dict[str, Any], label: str) -> list[str]:
    failures: list[str] = []
    try:
        validate_artifact_provenance(payload.get('provenance'), release=True)
    except EvidenceProvenanceError:
        failures.append(f'{label}_invalid_or_dirty_provenance')
    try:
        verify_evidence_manifest(path, required=True)
    except EvidenceProvenanceError:
        failures.append(f'{label}_invalid_or_missing_sidecar')
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-eval', required=True)
    parser.add_argument('--candidate-eval', required=True)
    parser.add_argument('--baseline-write-cost', required=True)
    parser.add_argument('--candidate-write-cost', required=True)
    parser.add_argument('--mode', default='hybrid-weighted')
    parser.add_argument('--max-elapsed-ratio', type=float, default=1.10)
    parser.add_argument('--max-bytes-ratio', type=float, default=1.00)
    parser.add_argument('--max-fts-rows-ratio', type=float, default=1.00)
    parser.add_argument('--max-vector-rows-ratio', type=float, default=1.00)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    paths = {
        'baseline_eval': Path(args.baseline_eval),
        'candidate_eval': Path(args.candidate_eval),
        'baseline_write_cost': Path(args.baseline_write_cost),
        'candidate_write_cost': Path(args.candidate_write_cost),
    }
    payloads = {
        'baseline_eval': _load(paths['baseline_eval'], 'retrieval_eval_matrix_redacted'),
        'candidate_eval': _load(paths['candidate_eval'], 'retrieval_eval_matrix_redacted'),
        'baseline_write_cost': _load(paths['baseline_write_cost'], 'retrieval_write_cost_redacted'),
        'candidate_write_cost': _load(paths['candidate_write_cost'], 'retrieval_write_cost_redacted'),
    }
    evidence_failures = [
        failure
        for label in paths
        for failure in _verify_release_evidence(paths[label], payloads[label], label)
    ]
    report = evaluate_retrieval_document_migration(
        payloads['baseline_eval'],
        payloads['candidate_eval'],
        payloads['baseline_write_cost'],
        payloads['candidate_write_cost'],
        mode=args.mode,
        max_write_cost_ratios={
            'elapsed_ms': args.max_elapsed_ratio,
            'bytes_written': args.max_bytes_ratio,
            'fts_rows_written': args.max_fts_rows_ratio,
            'vector_rows_written': args.max_vector_rows_ratio,
        },
    )
    report['evidence_gate'] = {'ok': not evidence_failures, 'failures': sorted(evidence_failures)}
    if evidence_failures:
        report['ok'] = False
        report['decision'] = 'retain_current_retrieval_documents'
        report['failures'] = sorted(set([*report['failures'], 'release_evidence_invalid']))
    report['provenance'] = payloads['candidate_eval'].get('provenance')
    validate_redacted_artifact(report)
    write_evidence_artifact(report, Path(args.out).expanduser())
    print(json.dumps({
        'ok': report['ok'],
        'decision': report['decision'],
        'automatic_migration': False,
        'failures': report['failures'],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
