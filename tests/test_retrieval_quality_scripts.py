from __future__ import annotations

import json
import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from trove_core.search.eval_matrix import run_eval_matrix
from trove_core.search.eval_schema import load_case_pack, stable_hash, validate_redacted_artifact
from trove_core.wechat.indexer import index_fixture_vault
from scripts.run_retrieval_eval_matrix import main as run_eval_matrix_main
from scripts.calibrate_zvec_score_floor import load_fixed_dev_split
from scripts.split_retrieval_eval_pack import build_split_manifest


ROOT = Path(__file__).resolve().parents[1]


class RetrievalQualityScriptTests(unittest.TestCase):
    def test_eval_matrix_cli_returns_redacted_typed_failure_for_stale_pack(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vault = root / 'vault'
            index_fixture_vault(vault, reset=True)
            cases = root / 'stale.json'
            private_query = 'private query must stay private'
            stale_citation = 'trove://wechat/stale-account/stale-conversation/message_0/999'
            cases.write_text(json.dumps({'cases': [{
                'case_id': 'stale-case',
                'query': private_query,
                'category': 'exact_sparse',
                'oracle': {'expected_any_citation': [stale_citation]},
            }]}), encoding='utf-8')
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = run_eval_matrix_main([
                    '--vault', str(vault),
                    '--cases', str(cases),
                    '--modes', 'exact',
                ])

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), '')
            failure = json.loads(stderr.getvalue())
            self.assertEqual(failure['error_code'], 'case_pack_incompatible_with_index')
            self.assertEqual(failure['compatibility']['state'], 'incompatible')
            rendered = stderr.getvalue()
            self.assertNotIn(private_query, rendered)
            self.assertNotIn(stale_citation, rendered)
            self.assertNotIn(str(root), rendered)

    def test_eval_matrix_filters_by_case_hash(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            index_fixture_vault(vault, reset=True)
            cases_path = ROOT / 'tests/golden/retrieval_core.jsonl'
            first = load_case_pack(cases_path)[0]
            report = run_eval_matrix(
                vault,
                cases_path,
                modes=['hybrid-weighted'],
                k=3,
                case_hash_filters={str(stable_hash(first.get('case_id')))},
            )
            validate_redacted_artifact(report)
            self.assertEqual(report['case_count'], 1)
            self.assertTrue(report['controls']['case_hash_filter_enabled'])
            self.assertEqual(report['controls']['case_hash_filter_count'], 1)

    def test_split_manifest_is_redacted_and_stratified(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'split.redacted.json'
            subprocess.run(
                [
                    str(ROOT / 'scripts/trove-python'),
                    str(ROOT / 'scripts/split_retrieval_eval_pack.py'),
                    '--cases',
                    str(ROOT / 'tests/golden/retrieval_core.jsonl'),
                    '--out',
                    str(out),
                    '--holdout-positive-target',
                    '3',
                ],
                check=True,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                text=True,
            )
            data = json.loads(out.read_text(encoding='utf-8'))
            validate_redacted_artifact(data)
            self.assertEqual(data['artifact_type'], 'retrieval_quality_split_redacted')
            self.assertEqual(data['counts']['holdout_positive_cases'], 3)
            self.assertEqual(
                data['counts']['positive_cases'],
                data['counts']['dev_positive_cases'] + data['counts']['holdout_positive_cases'],
            )

    def test_split_keeps_calibration_negatives_out_of_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cases_path = root / 'cases.jsonl'
            cases = [
                *load_case_pack(ROOT / 'tests/golden/retrieval_core.jsonl'),
                load_case_pack(ROOT / 'tests/golden/retrieval_scope.jsonl')[0],
            ]
            cases_path.write_text(
                ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases),
                encoding='utf-8',
            )
            manifest = build_split_manifest(cases_path, holdout_positive_target=3)
            manifest_path = root / 'split.redacted.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

            positive_hashes, negative_hashes = load_fixed_dev_split(manifest_path)

            self.assertTrue(positive_hashes)
            self.assertTrue(negative_hashes)
            self.assertEqual(manifest['splits']['holdout']['negative_case_hashes'], [])
            self.assertTrue(negative_hashes.isdisjoint(manifest['splits']['holdout']['case_hashes']))

    def test_failure_diagnosis_writes_only_redacted_fields(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            index_fixture_vault(vault, reset=True)
            out = Path(d) / 'diagnosis.redacted.json'
            subprocess.run(
                [
                    str(ROOT / 'scripts/trove-python'),
                    str(ROOT / 'scripts/diagnose_retrieval_failures.py'),
                    '--vault',
                    str(vault),
                    '--cases',
                    str(ROOT / 'tests/golden/retrieval_core.jsonl'),
                    '--out',
                    str(out),
                ],
                check=True,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                text=True,
            )
            data = json.loads(out.read_text(encoding='utf-8'))
            validate_redacted_artifact(data)
            self.assertEqual(data['artifact_type'], 'retrieval_failure_diagnosis_redacted')
            self.assertIn('leak_counts', data['summary'])
            self.assertFalse(data['privacy']['raw_queries_included'])


if __name__ == '__main__':
    unittest.main()
