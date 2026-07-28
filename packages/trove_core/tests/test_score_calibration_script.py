from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.calibrate_zvec_score_floor import (
    CalibrationInputError,
    collect_dev_score_bounds,
    collect_revision_bound_dev_score_bounds,
    load_fixed_dev_split,
)
from trove_core.search.eval_schema import stable_hash


class _CalibrationZvec:
    def __init__(self, results):
        self.results = results

    def calibration_candidates(self, query, **_kwargs):
        return list(self.results[query])


class _RevisionChangingCalibrationZvec(_CalibrationZvec):
    def __init__(self, results):
        super().__init__(results)
        self.revision = 1
        self.calls = 0

    def _authoritative_score_metadata(self):
        return {
            'schema_version': 4,
            'backend': 'zvec',
            'generation_id': 'a' * 32,
            'generation_revision': self.revision,
            'vector_text_version': 3,
            'dimensions': 16,
            'embedding_identity_sha256': 'b' * 64,
        }

    def calibration_candidates(self, query, **kwargs):
        rows = super().calibration_candidates(query, **kwargs)
        self.calls += 1
        if self.calls == 1:
            self.revision += 1
        return rows


class ScoreCalibrationScriptTests(unittest.TestCase):
    def test_fixed_dev_split_rejects_holdout_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.json'
            path.write_text(json.dumps({
                'splits': {
                    'dev': {'positive_case_hashes': ['a'], 'negative_case_hashes': ['b']},
                    'holdout': {'positive_case_hashes': ['a']},
                },
            }), encoding='utf-8')
            with self.assertRaisesRegex(CalibrationInputError, 'overlap'):
                load_fixed_dev_split(path)

    def test_score_bounds_use_relevant_positive_and_true_no_result_negative(self):
        positive = {
            'case_id': 'positive',
            'query': 'positive private query',
            'oracle': {'expected_any_citation': ['parent-target']},
            'filters': {},
        }
        negative = {
            'case_id': 'negative',
            'query': 'negative private query',
            'oracle': {'negative_no_results': True},
            'filters': {'account_id': 'private-account'},
        }
        zvec = _CalibrationZvec({
            positive['query']: [
                ({'citation': 'distractor'}, 0.90),
                ({'citation': 'chunk-target', 'parent_citation': 'parent-target'}, 0.60),
            ],
            negative['query']: [({'citation': 'irrelevant'}, 0.40)],
        })

        bounds = collect_dev_score_bounds(
            zvec,  # type: ignore[arg-type]
            object(),
            [positive, negative],
            positive_hashes={str(stable_hash('positive'))},
            negative_hashes={str(stable_hash('negative'))},
            candidate_limit=200,
        )

        self.assertEqual(bounds, (0.40, 0.60, 1, 1))

    def test_negative_exclusion_case_cannot_tune_no_result_floor(self):
        negative = {
            'case_id': 'negative',
            'query': 'private query',
            'oracle': {'negative_excluded_citations': ['private-citation']},
            'filters': {},
        }
        zvec = _CalibrationZvec({negative['query']: [({'citation': 'irrelevant'}, 0.40)]})
        with self.assertRaisesRegex(CalibrationInputError, 'negative_no_results'):
            collect_dev_score_bounds(
                zvec,  # type: ignore[arg-type]
                object(),
                [negative],
                positive_hashes=set(),
                negative_hashes={str(stable_hash('negative'))},
                candidate_limit=200,
            )

    def test_revision_change_between_cases_rejects_mixed_score_evidence(self):
        positive = {
            'case_id': 'positive',
            'query': 'positive private query',
            'oracle': {'expected_any_citation': ['target']},
            'filters': {},
        }
        negative = {
            'case_id': 'negative',
            'query': 'negative private query',
            'oracle': {'negative_no_results': True},
            'filters': {},
        }
        zvec = _RevisionChangingCalibrationZvec({
            positive['query']: [({'citation': 'target'}, 0.60)],
            negative['query']: [({'citation': 'irrelevant'}, 0.40)],
        })

        with self.assertRaisesRegex(CalibrationInputError, 'generation changed'):
            collect_revision_bound_dev_score_bounds(
                zvec,  # type: ignore[arg-type]
                object(),
                [positive, negative],
                positive_hashes={str(stable_hash('positive'))},
                negative_hashes={str(stable_hash('negative'))},
                candidate_limit=200,
            )


if __name__ == '__main__':
    unittest.main()
