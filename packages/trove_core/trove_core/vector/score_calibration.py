from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from trove_core.search.evidence_provenance import (
    stable_payload_sha256,
    validate_artifact_provenance,
)


SCORE_CALIBRATION_SCHEMA_VERSION = 1
SCORE_CALIBRATION_ARTIFACT_TYPE = 'zvec_score_calibration_redacted'
SCORE_CALIBRATION_METHOD = 'bounded_dev_recall_floor_v2'
SCORE_CALIBRATION_METRIC = 'inner_product'


class VectorScoreCalibrationError(RuntimeError):
    """Typed fail-closed error for an untrusted vector score domain."""

    vector_state = 'degraded'

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        ).encode('utf-8')
    ).hexdigest()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def embedding_identity(provider: Any | None) -> dict[str, Any]:
    """Return a stable, path-free identity for one embedding score domain.

    Local providers already calculate a content-bound daemon identity.  Other
    providers use their explicit provider/model contract.  The public binding
    is a digest so model paths and provider labels never enter redacted score
    calibration artifacts.
    """

    while provider is not None:
        wrapped_provider = getattr(provider, '_provider', None)
        if wrapped_provider is None or not callable(getattr(wrapped_provider, 'embed', None)):
            break
        provider = wrapped_provider
    if provider is None:
        payload = {
            'provider': 'none',
            'model_id': 'none',
            'model_hash': '0' * 64,
            'dimensions': 0,
            'request_format': '',
            'normalize_embeddings': False,
        }
    else:
        daemon_identity = getattr(provider, '_daemon_identity', None)
        provider_name = str(
            getattr(daemon_identity, 'provider', None)
            or getattr(provider, 'provider_name', None)
            or getattr(provider, 'name', None)
            or provider.__class__.__qualname__
        )
        model_id = str(
            getattr(daemon_identity, 'model_id', None)
            or getattr(provider, 'model_id', None)
            or getattr(provider, 'model', None)
            or 'unspecified'
        )
        exact_model_hash = str(
            getattr(daemon_identity, 'model_hash', None)
            or getattr(provider, 'model_hash', None)
            or ''
        )
        if len(exact_model_hash) != 64 or any(ch not in '0123456789abcdef' for ch in exact_model_hash.lower()):
            exact_model_hash = _canonical_sha256({
                'provider': provider_name,
                'model_id': model_id,
                'class': f'{provider.__class__.__module__}.{provider.__class__.__qualname__}',
            })
        payload = {
            'provider': provider_name,
            'model_id': model_id,
            'model_hash': exact_model_hash.lower(),
            'dimensions': int(getattr(provider, 'dimensions', 0) or getattr(daemon_identity, 'dimensions', 0) or 0),
            'request_format': str(getattr(provider, 'request_format', '') or ''),
            'normalize_embeddings': bool(getattr(provider, 'normalize_embeddings', False)),
        }
        if bool(getattr(provider, 'supports_sparse', False)):
            payload['supports_sparse'] = True
            payload['query_instruct_sha256'] = hashlib.sha256(
                str(getattr(provider, 'query_instruct', '') or '').encode('utf-8')
            ).hexdigest()
    return {
        'sha256': _canonical_sha256(payload),
        'dimensions': int(payload['dimensions']),
    }


def index_identity(metadata: dict[str, Any]) -> str:
    """Identify one exact published index snapshot.

    This revision-bound identity is used to reject mixed calibration samples
    and a vector generation changing during one search.
    """

    basis = {
        'backend': str(metadata.get('backend') or ''),
        'generation_id': str(metadata.get('generation_id') or ''),
        'generation_revision': _safe_int(metadata.get('generation_revision')),
        'schema_version': _safe_int(metadata.get('schema_version')),
        'collection_contract_version': _safe_int(metadata.get('collection_contract_version')),
        'vector_text_version': _safe_int(metadata.get('vector_text_version')),
        'dimensions': _safe_int(metadata.get('dimensions')),
        'embedding_identity_sha256': str(metadata.get('embedding_identity_sha256') or ''),
    }
    return _canonical_sha256(basis)


def score_domain_identity(metadata: dict[str, Any]) -> str:
    """Identify the stable similarity score domain.

    The absolute similarity scale is defined by the backend, embedding model,
    dimensions, and vector-text contract. Adding or removing documents inside
    the same published generation does not change that scale, so routine
    incremental indexing must not disable semantic search.
    """

    basis = {
        'backend': str(metadata.get('backend') or ''),
        'generation_id': str(metadata.get('generation_id') or ''),
        'schema_version': _safe_int(metadata.get('schema_version')),
        'collection_contract_version': _safe_int(metadata.get('collection_contract_version')),
        'vector_text_version': _safe_int(metadata.get('vector_text_version')),
        'dimensions': _safe_int(metadata.get('dimensions')),
        'embedding_identity_sha256': str(metadata.get('embedding_identity_sha256') or ''),
    }
    return _canonical_sha256(basis)


def build_score_calibration_artifact(
    *,
    metadata: dict[str, Any],
    provider: Any,
    max_negative_top_score: float,
    min_positive_target_score: float,
    positive_case_count: int,
    negative_case_count: int,
    case_pack_sha256: str,
    split_manifest_sha256: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build a redacted dev-only threshold artifact.

    A usable threshold exists only when the fixed dev positives and true
    no-result negatives are separable.  Holdout observations are deliberately
    absent from the schema so they cannot be used to tune the floor.
    """

    max_negative = float(max_negative_top_score)
    min_positive = float(min_positive_target_score)
    if not math.isfinite(max_negative) or not math.isfinite(min_positive):
        raise VectorScoreCalibrationError('vector_score_calibration_non_finite')
    if min(max_negative, min_positive) < -1.000001 or max(max_negative, min_positive) > 1.000001:
        raise VectorScoreCalibrationError('vector_score_calibration_metric_invalid')
    if max_negative >= min_positive:
        raise VectorScoreCalibrationError('vector_score_calibration_not_separable')
    if _safe_int(positive_case_count) < 1 or _safe_int(negative_case_count) < 1:
        raise VectorScoreCalibrationError('vector_score_calibration_evidence_insufficient')
    if len(str(case_pack_sha256)) != 64 or len(str(split_manifest_sha256)) != 64:
        raise VectorScoreCalibrationError('vector_score_calibration_provenance_invalid')
    validate_artifact_provenance(provenance, release=False)
    if str(provenance.get('case_pack_sha256') or '') != str(case_pack_sha256):
        raise VectorScoreCalibrationError('vector_score_calibration_provenance_mismatch')
    provider_identity = embedding_identity(provider)
    metadata_identity = str(metadata.get('embedding_identity_sha256') or '')
    if metadata_identity != provider_identity['sha256']:
        raise VectorScoreCalibrationError('vector_score_calibration_model_mismatch')
    backend = str(metadata.get('backend') or '')
    generation_id = str(metadata.get('generation_id') or '')
    generation_revision = _safe_int(metadata.get('generation_revision'))
    if backend != 'zvec' or not generation_id or generation_revision < 1:
        raise VectorScoreCalibrationError('vector_score_calibration_index_mismatch')
    # Every value in (max_negative, min_positive] separates the fixed dev
    # evidence.  Choose the smallest representable safe floor rather than the
    # midpoint so relevance filtering preserves the maximum possible recall.
    floor = math.nextafter(max_negative, math.inf)
    return {
        'schema_version': SCORE_CALIBRATION_SCHEMA_VERSION,
        'artifact_type': SCORE_CALIBRATION_ARTIFACT_TYPE,
        'calibration_split': 'dev',
        'method': SCORE_CALIBRATION_METHOD,
        'binding': {
            'backend': backend,
            'generation_id': generation_id,
            'generation_revision': generation_revision,
            'embedding_identity_sha256': provider_identity['sha256'],
            'index_identity_sha256': score_domain_identity(metadata),
            'vector_text_version': _safe_int(metadata.get('vector_text_version')),
            'dimensions': _safe_int(metadata.get('dimensions')),
        },
        'threshold': {
            'metric': SCORE_CALIBRATION_METRIC,
            'inclusive_min_score': floor,
        },
        'dev_evidence': {
            'case_pack_sha256': str(case_pack_sha256),
            'split_manifest_sha256': str(split_manifest_sha256),
            'positive_case_count': _safe_int(positive_case_count),
            'negative_no_result_case_count': _safe_int(negative_case_count),
            'max_negative_top_score': max_negative,
            'min_positive_target_score': min_positive,
            'separation_margin': min_positive - max_negative,
            'holdout_observations_used': False,
        },
        'provenance': provenance,
        'privacy': {
            'raw_queries_included': False,
            'raw_citations_included': False,
            'raw_snippets_included': False,
            'private_paths_included': False,
            'provider_names_included': False,
            'model_names_included': False,
            'secret_values_included': False,
        },
    }


def validate_score_calibration_artifact(
    artifact: Any,
    *,
    metadata: dict[str, Any],
    provider: Any,
    release: bool = True,
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise VectorScoreCalibrationError('vector_score_calibration_artifact_invalid')
    if artifact.get('schema_version') != SCORE_CALIBRATION_SCHEMA_VERSION:
        raise VectorScoreCalibrationError('vector_score_calibration_schema_invalid')
    if artifact.get('artifact_type') != SCORE_CALIBRATION_ARTIFACT_TYPE:
        raise VectorScoreCalibrationError('vector_score_calibration_artifact_invalid')
    if artifact.get('calibration_split') != 'dev' or artifact.get('method') != SCORE_CALIBRATION_METHOD:
        raise VectorScoreCalibrationError('vector_score_calibration_split_invalid')
    try:
        validate_artifact_provenance(artifact.get('provenance'), release=release)
    except Exception as exc:
        raise VectorScoreCalibrationError('vector_score_calibration_provenance_invalid') from exc
    binding = artifact.get('binding')
    threshold = artifact.get('threshold')
    evidence = artifact.get('dev_evidence')
    if not all(isinstance(value, dict) for value in (binding, threshold, evidence)):
        raise VectorScoreCalibrationError('vector_score_calibration_artifact_invalid')
    current_embedding = embedding_identity(provider)
    if (
        binding.get('embedding_identity_sha256') != current_embedding['sha256']
        or binding.get('embedding_identity_sha256') != metadata.get('embedding_identity_sha256')
    ):
        raise VectorScoreCalibrationError('vector_score_calibration_model_mismatch')
    if binding.get('index_identity_sha256') != score_domain_identity(metadata):
        raise VectorScoreCalibrationError('vector_score_calibration_index_mismatch')
    if (
        binding.get('backend') != metadata.get('backend')
        or binding.get('generation_id') != metadata.get('generation_id')
    ):
        raise VectorScoreCalibrationError('vector_score_calibration_index_mismatch')
    if _safe_int(binding.get('vector_text_version')) != _safe_int(metadata.get('vector_text_version')):
        raise VectorScoreCalibrationError('vector_score_calibration_index_mismatch')
    if _safe_int(binding.get('dimensions')) != _safe_int(metadata.get('dimensions')):
        raise VectorScoreCalibrationError('vector_score_calibration_model_mismatch')
    if threshold.get('metric') != SCORE_CALIBRATION_METRIC:
        raise VectorScoreCalibrationError('vector_score_calibration_metric_invalid')
    try:
        floor = float(threshold['inclusive_min_score'])
        max_negative = float(evidence['max_negative_top_score'])
        min_positive = float(evidence['min_positive_target_score'])
        positive_count = int(evidence['positive_case_count'])
        negative_count = int(evidence['negative_no_result_case_count'])
    except (KeyError, TypeError, ValueError) as exc:
        raise VectorScoreCalibrationError('vector_score_calibration_artifact_invalid') from exc
    if not all(math.isfinite(value) for value in (floor, max_negative, min_positive)):
        raise VectorScoreCalibrationError('vector_score_calibration_non_finite')
    if min(floor, max_negative, min_positive) < -1.000001 or max(floor, max_negative, min_positive) > 1.000001:
        raise VectorScoreCalibrationError('vector_score_calibration_metric_invalid')
    if not (max_negative < floor <= min_positive):
        raise VectorScoreCalibrationError('vector_score_calibration_threshold_invalid')
    if positive_count < 1 or negative_count < 1 or evidence.get('holdout_observations_used') is not False:
        raise VectorScoreCalibrationError('vector_score_calibration_evidence_insufficient')
    for key in ('case_pack_sha256', 'split_manifest_sha256'):
        value = str(evidence.get(key) or '')
        if len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value.lower()):
            raise VectorScoreCalibrationError('vector_score_calibration_provenance_invalid')
    if str((artifact.get('provenance') or {}).get('case_pack_sha256') or '') != str(evidence['case_pack_sha256']):
        raise VectorScoreCalibrationError('vector_score_calibration_provenance_mismatch')
    return {
        'schema_version': SCORE_CALIBRATION_SCHEMA_VERSION,
        'state': 'available',
        'artifact_payload_sha256': stable_payload_sha256(artifact),
        'provenance_sha256': stable_payload_sha256(artifact.get('provenance')),
        'embedding_identity_sha256': current_embedding['sha256'],
        'index_identity_sha256': score_domain_identity(metadata),
        'backend': str(metadata.get('backend') or ''),
        'generation_id': str(metadata.get('generation_id') or ''),
        'generation_revision': _safe_int(metadata.get('generation_revision')),
        'case_pack_sha256': str(evidence['case_pack_sha256']),
        'split_manifest_sha256': str(evidence['split_manifest_sha256']),
        'metric': SCORE_CALIBRATION_METRIC,
        'inclusive_min_score': floor,
        'method': SCORE_CALIBRATION_METHOD,
        'positive_case_count': positive_count,
        'negative_no_result_case_count': negative_count,
        'holdout_observations_used': False,
    }


def score_calibration_status(metadata: dict[str, Any], provider: Any | None) -> dict[str, Any]:
    calibration = metadata.get('score_calibration')
    if not isinstance(calibration, dict):
        return {
            'state': 'unavailable',
            'reason_code': 'vector_score_calibration_missing',
            'bounded': True,
        }
    current_embedding = embedding_identity(provider)
    current_embedding_sha = (
        current_embedding['sha256']
        if provider is not None
        else str(metadata.get('embedding_identity_sha256') or '')
    )
    reason_code = None
    try:
        floor = float(calibration['inclusive_min_score'])
    except (KeyError, TypeError, ValueError):
        floor = math.nan
    if calibration.get('schema_version') != SCORE_CALIBRATION_SCHEMA_VERSION:
        reason_code = 'vector_score_calibration_schema_invalid'
    elif calibration.get('method') != SCORE_CALIBRATION_METHOD:
        reason_code = 'vector_score_calibration_schema_invalid'
    elif calibration.get('embedding_identity_sha256') != current_embedding_sha:
        reason_code = 'vector_score_calibration_model_mismatch'
    elif calibration.get('embedding_identity_sha256') != metadata.get('embedding_identity_sha256'):
        reason_code = 'vector_score_calibration_model_mismatch'
    elif calibration.get('index_identity_sha256') != score_domain_identity(metadata):
        reason_code = 'vector_score_calibration_index_mismatch'
    elif (
        calibration.get('backend') != metadata.get('backend')
        or calibration.get('generation_id') != metadata.get('generation_id')
    ):
        reason_code = 'vector_score_calibration_index_mismatch'
    elif calibration.get('metric') != SCORE_CALIBRATION_METRIC or not math.isfinite(floor):
        reason_code = 'vector_score_calibration_threshold_invalid'
    elif any(
        len(str(calibration.get(key) or '')) != 64
        or any(ch not in '0123456789abcdef' for ch in str(calibration.get(key) or '').lower())
        for key in ('artifact_sha256', 'artifact_manifest_sha256', 'artifact_payload_sha256', 'provenance_sha256')
    ):
        reason_code = 'vector_score_calibration_provenance_invalid'
    elif _safe_int(calibration.get('positive_case_count')) < 1 or _safe_int(calibration.get('negative_no_result_case_count')) < 1:
        reason_code = 'vector_score_calibration_evidence_insufficient'
    elif calibration.get('holdout_observations_used') is not False:
        reason_code = 'vector_score_calibration_split_invalid'
    if reason_code:
        return {
            'state': 'unavailable',
            'reason_code': reason_code,
            'bounded': True,
        }
    return {
        'state': 'available',
        'reason_code': None,
        'artifact_sha256': calibration.get('artifact_sha256'),
        'artifact_manifest_sha256': calibration.get('artifact_manifest_sha256'),
        'artifact_payload_sha256': calibration.get('artifact_payload_sha256'),
        'provenance_sha256': calibration.get('provenance_sha256'),
        'embedding_identity_sha256': calibration.get('embedding_identity_sha256'),
        'index_identity_sha256': calibration.get('index_identity_sha256'),
        'backend': calibration.get('backend'),
        'generation_id': calibration.get('generation_id'),
        'generation_revision': calibration.get('generation_revision'),
        'metric': SCORE_CALIBRATION_METRIC,
        'inclusive_min_score': floor,
        'positive_case_count': _safe_int(calibration.get('positive_case_count')),
        'negative_no_result_case_count': _safe_int(calibration.get('negative_no_result_case_count')),
        'holdout_observations_used': False,
        'bounded': True,
    }
