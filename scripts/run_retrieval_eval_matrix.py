#!/usr/bin/env python3
"""Run a redacted retrieval evaluation matrix over fixture or private case packs."""
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

from trove_core.search.eval_matrix import EvalCasePackCompatibilityError, run_eval_matrix
from trove_core.search.evidence_provenance import evidence_manifest_path, write_evidence_artifact
from trove_core.search.eval_schema import validate_redacted_artifact
from trove_core.vault.config import VaultConfig

ROOT = Path(__file__).resolve().parents[1]


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_redacted_out(path: Path, vault_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    repo_root = ROOT.resolve()
    # Fixture CI reports may go to temp dirs or the repo only when explicitly requested.
    if resolved == repo_root or is_relative_to(resolved, repo_root):
        if 'tests' not in resolved.parts and resolved.suffix == '.json':
            raise SystemExit('redacted matrix output for real/private packs must not be written inside the source repo')
    proof_root = (vault_root / 'proof' / 'retrieval-eval' / 'redacted').resolve()
    if is_relative_to(resolved, (vault_root / 'proof').resolve()) and not (resolved == proof_root or is_relative_to(resolved, proof_root)):
        raise SystemExit('redacted matrix output under Vault proof must stay in retrieval-eval/redacted')
    return resolved


def parse_modes(values: list[str] | None) -> list[str]:
    if not values:
        return ['hybrid-weighted']
    modes: list[str] = []
    for value in values:
        modes.extend([part.strip() for part in value.split(',') if part.strip()])
    return modes


def load_case_hash_filter(path: Path | None, *, group: str | None = None, include_negatives: bool = False) -> set[str] | None:
    if path is None:
        return None
    text = path.expanduser().read_text(encoding='utf-8').strip()
    if not text:
        return set()
    if text.startswith('{'):
        data = json.loads(text)
        if group:
            split = ((data.get('splits') or {}).get(group) or {})
            values = set(str(v) for v in split.get('positive_case_hashes') or split.get('case_hashes') or [] if v)
            if include_negatives:
                values.update(str(v) for v in (data.get('negative_case_hashes') or []) if v)
                values.update(str(v) for v in (split.get('negative_case_hashes') or []) if v)
            return values
        values = data.get('case_hashes') or data.get('positive_case_hashes') or []
        return {str(v) for v in values if v}
    if text.startswith('['):
        return {str(v) for v in json.loads(text) if v}
    return {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', help='Selected TROVE runtime Vault root.')
    parser.add_argument('--cases', required=True, help='Fixture or private case pack (.json/.jsonl).')
    parser.add_argument('--modes', nargs='*', help='Modes or comma-separated modes.')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--model-path')
    parser.add_argument('--cloud', action='store_true', help='Use the selected Vault cloud vector/episode/rerank production path.')
    parser.add_argument('--reranker-model-path')
    parser.add_argument('--retrieval-candidate-limit', type=int, default=200)
    parser.add_argument('--fusion-candidate-limit', type=int, default=200)
    parser.add_argument('--reranker-candidate-limit', type=int, default=50)
    parser.add_argument('--category', action='append', default=[], help='Run only matching case categories. Repeatable.')
    parser.add_argument('--max-cases', type=int, help='Deterministically sample at most this many cases after filters.')
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--fixture-id', default='synthetic_or_redacted', help='Synthetic/redacted fixture label; only its hash is recorded.')
    parser.add_argument('--case-hash-file', help='Redacted hash allow-list, or a retrieval-quality split manifest.')
    parser.add_argument('--case-hash-group', help='Split name inside --case-hash-file, for example dev or holdout.')
    parser.add_argument('--case-hash-include-negatives', action='store_true', help='When using a split manifest, add its negative cases for pass-rate checks.')
    parser.add_argument('--resume', help='Resume from an existing redacted matrix report.')
    parser.add_argument('--partial-out', help='Atomic partial redacted report path for long local runs.')
    parser.add_argument('--out', help='Redacted output json. Defaults to Vault proof/retrieval-eval/redacted/eval-matrix.redacted.json')
    args = parser.parse_args(argv)
    cfg = VaultConfig.resolve(args.vault)
    partial_out = validate_redacted_out(Path(args.partial_out), cfg.root) if args.partial_out else None
    case_hash_filters = load_case_hash_filter(
        Path(args.case_hash_file) if args.case_hash_file else None,
        group=args.case_hash_group,
        include_negatives=args.case_hash_include_negatives,
    )
    try:
        report = run_eval_matrix(
            cfg.root,
            Path(args.cases),
            modes=parse_modes(args.modes),
            k=args.k,
            model_path=args.model_path,
            reranker_model_path=args.reranker_model_path,
            category_filters=args.category,
            case_hash_filters=case_hash_filters,
            max_cases=args.max_cases,
            sample_seed=args.sample_seed,
            fixture_id=args.fixture_id,
            resume_path=Path(args.resume) if args.resume else None,
            partial_out=partial_out,
            retrieval_candidate_limit=args.retrieval_candidate_limit,
            fusion_candidate_limit=args.fusion_candidate_limit,
            reranker_candidate_limit=args.reranker_candidate_limit,
            cloud=bool(args.cloud),
        )
    except EvalCasePackCompatibilityError as exc:
        failure = exc.to_redacted_dict()
        validate_redacted_artifact(failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    validate_redacted_artifact(report)
    if args.out:
        out = validate_redacted_out(Path(args.out), cfg.root)
    else:
        out = cfg.root / 'proof' / 'retrieval-eval' / 'redacted' / 'eval-matrix.redacted.json'
    write_evidence_artifact(report, out)
    summary = {
        'ok': True,
        'complete': report.get('complete'),
        'case_count': report['case_count'],
        'modes': sorted(report['modes'].keys()),
        'redacted_file': out.name,
        'manifest_file': evidence_manifest_path(out).name,
        'raw_queries_printed': False,
        'raw_snippets_printed': False,
        'private_paths_printed': False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
