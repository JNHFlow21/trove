from __future__ import annotations

import copy
import math
import unittest

from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.vector.score_calibration import (
    VectorScoreCalibrationError,
    build_score_calibration_artifact,
    embedding_identity,
    score_domain_identity,
    score_calibration_status,
    validate_score_calibration_artifact,
)


def _provenance() -> dict:
    return {
        'schema_version': 1,
        'git': {'commit_sha': 'a' * 40, 'dirty': False},
        'platform': {
            'system': 'test',
            'release': 'test',
            'machine': 'test',
            'python_implementation': 'CPython',
            'python_version': '3.13.0',
            'cpu_count': 1,
            'processor_sha256': 'b' * 64,
            'memory_bytes': None,
        },
        'fixture': {'kind': 'synthetic_or_redacted', 'sha256': 'c' * 64},
        'seed': 42,
        'case_pack_sha256': 'd' * 64,
        'store': {
            'schema_version': 1,
            'schema_manifest_sha256': 'e' * 64,
            'content_identity_sha256': 'f' * 64,
            'index_generation_sha256': '1' * 64,
            'document_count': 2,
        },
        'provider': {
            'provider_sha256': '2' * 64,
            'model_sha256': '3' * 64,
            'dimensions': 16,
        },
        'execution': {
            'temperature': 'warm',
            'warmups': 0,
            'rounds': 2,
            'includes_engine_build': False,
        },
        'privacy': {
            'raw_fixture_identity_included': False,
            'raw_case_pack_included': False,
            'private_paths_included': False,
            'provider_names_included': False,
            'model_names_included': False,
        },
    }


def _metadata(provider: FakeEmbeddingProvider) -> dict:
    return {
        'schema_version': 4,
        'backend': 'zvec',
        'generation_id': 'generation-a',
        'generation_revision': 1,
        'collection_contract_version': 1,
        'vector_text_version': 3,
        'dimensions': 16,
        'embedding_identity_sha256': embedding_identity(provider)['sha256'],
        'indexed_count': 2,
        'expected_document_count': 2,
        'complete': True,
    }


class ScoreCalibrationTests(unittest.TestCase):
    def test_dirty_provenance_is_valid_for_local_runtime_but_not_release_evidence(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        metadata = _metadata(provider)
        provenance = _provenance()
        provenance['git']['dirty'] = True
        artifact = build_score_calibration_artifact(
            metadata=metadata,
            provider=provider,
            max_negative_top_score=0.30,
            min_positive_target_score=0.50,
            positive_case_count=5,
            negative_case_count=3,
            case_pack_sha256='d' * 64,
            split_manifest_sha256='7' * 64,
            provenance=provenance,
        )

        self.assertEqual(
            validate_score_calibration_artifact(
                artifact,
                metadata=metadata,
                provider=provider,
                release=False,
            )['state'],
            'available',
        )
        with self.assertRaisesRegex(VectorScoreCalibrationError, 'provenance_invalid'):
            validate_score_calibration_artifact(
                artifact,
                metadata=metadata,
                provider=provider,
                release=True,
            )

    def test_query_cache_wrapper_keeps_underlying_embedding_identity(self):
        provider = FakeEmbeddingProvider(dimensions=16)

        class Wrapper:
            def __init__(self, wrapped):
                self._provider = wrapped

            def __getattr__(self, name):
                return getattr(self._provider, name)

        self.assertEqual(embedding_identity(provider), embedding_identity(Wrapper(provider)))

    def test_dev_artifact_binds_model_index_and_separating_floor(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        metadata = _metadata(provider)

        artifact = build_score_calibration_artifact(
            metadata=metadata,
            provider=provider,
            max_negative_top_score=0.30,
            min_positive_target_score=0.50,
            positive_case_count=5,
            negative_case_count=3,
            case_pack_sha256='d' * 64,
            split_manifest_sha256='7' * 64,
            provenance=_provenance(),
        )
        calibration = validate_score_calibration_artifact(
            artifact,
            metadata=metadata,
            provider=provider,
        )

        self.assertEqual(calibration['inclusive_min_score'], math.nextafter(0.30, math.inf))
        self.assertEqual(calibration['embedding_identity_sha256'], embedding_identity(provider)['sha256'])
        self.assertEqual(calibration['index_identity_sha256'], score_domain_identity(metadata))
        self.assertEqual(calibration['generation_id'], 'generation-a')
        self.assertEqual(calibration['generation_revision'], 1)
        self.assertEqual(artifact['binding']['backend'], 'zvec')
        self.assertFalse(calibration['holdout_observations_used'])
        self.assertNotIn('citation-a', str(artifact))

    def test_overlap_or_holdout_tuning_fails_closed(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        metadata = _metadata(provider)
        with self.assertRaisesRegex(VectorScoreCalibrationError, 'not_separable'):
            build_score_calibration_artifact(
                metadata=metadata,
                provider=provider,
                max_negative_top_score=0.51,
                min_positive_target_score=0.50,
                positive_case_count=5,
                negative_case_count=3,
                case_pack_sha256='d' * 64,
                split_manifest_sha256='7' * 64,
                provenance=_provenance(),
            )

        artifact = build_score_calibration_artifact(
            metadata=metadata,
            provider=provider,
            max_negative_top_score=0.30,
            min_positive_target_score=0.50,
            positive_case_count=5,
            negative_case_count=3,
            case_pack_sha256='d' * 64,
            split_manifest_sha256='7' * 64,
            provenance=_provenance(),
        )
        artifact['dev_evidence']['holdout_observations_used'] = True
        with self.assertRaisesRegex(VectorScoreCalibrationError, 'evidence_insufficient'):
            validate_score_calibration_artifact(artifact, metadata=metadata, provider=provider)

    def test_case_pack_provenance_must_equal_dev_evidence(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        metadata = _metadata(provider)
        with self.assertRaisesRegex(VectorScoreCalibrationError, 'provenance_mismatch'):
            build_score_calibration_artifact(
                metadata=metadata,
                provider=provider,
                max_negative_top_score=0.30,
                min_positive_target_score=0.50,
                positive_case_count=5,
                negative_case_count=3,
                case_pack_sha256='6' * 64,
                split_manifest_sha256='7' * 64,
                provenance=_provenance(),
            )

    def test_incremental_revision_preserves_calibration_but_generation_or_model_change_invalidates(self):
        provider = FakeEmbeddingProvider(dimensions=16)
        metadata = _metadata(provider)
        artifact = build_score_calibration_artifact(
            metadata=metadata,
            provider=provider,
            max_negative_top_score=0.30,
            min_positive_target_score=0.50,
            positive_case_count=5,
            negative_case_count=3,
            case_pack_sha256='d' * 64,
            split_manifest_sha256='7' * 64,
            provenance=_provenance(),
        )
        calibration = validate_score_calibration_artifact(artifact, metadata=metadata, provider=provider)
        calibration.update({
            'artifact_sha256': '8' * 64,
            'artifact_manifest_sha256': '9' * 64,
        })
        calibrated = {**metadata, 'score_calibration': calibration}
        self.assertEqual(score_calibration_status(calibrated, provider)['state'], 'available')

        changed_index = copy.deepcopy(calibrated)
        changed_index['generation_revision'] += 1
        self.assertEqual(score_calibration_status(changed_index, provider)['state'], 'available')
        changed_generation = copy.deepcopy(calibrated)
        changed_generation['generation_id'] = 'generation-b'
        self.assertEqual(
            score_calibration_status(changed_generation, provider)['reason_code'],
            'vector_score_calibration_index_mismatch',
        )
        other_model = FakeEmbeddingProvider(dimensions=32)
        self.assertEqual(
            score_calibration_status(calibrated, other_model)['reason_code'],
            'vector_score_calibration_model_mismatch',
        )


if __name__ == '__main__':
    unittest.main()
