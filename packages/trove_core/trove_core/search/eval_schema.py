from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2

EVAL_CATEGORIES = {
    'customer_profile',
    'blocker_diagnosis',
    'follow_up_action',
    'decision_history',
    'technical_project_memory',
    'exact_sparse',
    'semantic_paraphrase',
    'time_scoped',
    'source_family',
    'bounded_context',
    'profile_or_wiki_navigation',
    'negative_scope',
    'vector_lifecycle',
    'voice_transcript',
    'image_observation',
    # Backward-compatible/generated buckets used by older smoke packs.
    'high_frequency',
    'sparse_precision',
    'real_scenario',
    'semantic',
    'exact_phrase',
    'keyword_combo',
    'short_query',
    'sender_filter',
    'cross_source_family',
    'semantic_rewrite',
    'multi_hop',
    'hard_distractor',
}

EVAL_MODES = {
    'exact',
    'fts',
    'metadata',
    'parent_child',
    'vector',
    'hybrid-weighted',
    'hybrid-rrf',
    'feature-rerank',
    'local-reranker',
    'cloud-reranker',
    'vector-unavailable',
    'vector-degraded',
}

ORACLE_KEYS = {
    'expected_any_citation',
    'expected_all_citations',
    'expected_any_conversation',
    'expected_any_conversation_id',
    'expected_source_family',
    'expected_retrieval_paths_any',
    'context_anchor',
    'profile_page',
    'negative_no_results',
    'negative_excluded_citations',
    'hard_distractor_citation_hash',
    'negative_no_excluded_scope',
    'semantic_min_results',
    'min_results',
    'type',
}

WORD_RE = re.compile(r'[a-z0-9][a-z0-9_.-]{1,}|[\u4e00-\u9fff]+', re.I)


def compact_for_match(text: Any) -> str:
    """Normalize text for literal-substring leakage checks."""

    value = str(text or '').lower()
    return re.sub(r'[\s\W_]+', '', value, flags=re.UNICODE)


def eval_text_tokens(text: Any) -> set[str]:
    """Small dependency-free tokenizer for redacted eval-quality metrics.

    For CJK runs, use 2-character shingles.  This catches copied phrasing
    without requiring a local dictionary and keeps the metric deterministic.
    """

    tokens: set[str] = set()
    for match in WORD_RE.finditer(str(text or '').lower()):
        token = match.group(0)
        if re.fullmatch(r'[\u4e00-\u9fff]+', token):
            if len(token) == 1:
                tokens.add(token)
            else:
                tokens.update(token[i:i + 2] for i in range(len(token) - 1))
        elif len(token) >= 2:
            tokens.add(token)
    return tokens


def query_expected_quality(query: Any, expected_text: Any) -> dict[str, Any]:
    query_s = str(query or '')
    expected_s = str(expected_text or '')
    q_compact = compact_for_match(query_s)
    e_compact = compact_for_match(expected_s)
    q_tokens = eval_text_tokens(query_s)
    e_tokens = eval_text_tokens(expected_s)
    overlap = sorted(q_tokens & e_tokens)
    return {
        'literal_substring': bool(q_compact and e_compact and q_compact in e_compact),
        'word_overlap_ratio': (len(overlap) / len(q_tokens)) if q_tokens else 0.0,
        'query_token_count': len(q_tokens),
        'expected_token_count': len(e_tokens),
        'overlap_token_count': len(overlap),
    }


def case_quality(case: dict[str, Any]) -> dict[str, Any]:
    private = case.get('private') or {}
    expected_text = (
        private.get('bounded_context_note')
        or private.get('expected_text')
        or case.get('expected_text')
        or ''
    )
    quality = dict(case.get('quality') or {})
    measured = query_expected_quality(case.get('query'), expected_text)
    # Trust but verify: generator-stored metrics are overwritten by deterministic
    # measurement when private expected text is present.
    if expected_text:
        quality.update(measured)
    else:
        quality.setdefault('literal_substring', False)
        quality.setdefault('word_overlap_ratio', 0.0)
        quality.setdefault('query_token_count', 0)
        quality.setdefault('expected_token_count', 0)
        quality.setdefault('overlap_token_count', 0)
    quality['rewrite_strategy'] = str(quality.get('rewrite_strategy') or case.get('query_type') or 'unknown')
    return quality


def case_pack_quality_stats(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    case_list = list(cases)
    qualities = [case_quality(case) for case in case_list]
    total = len(qualities)
    literal_hits = sum(1 for q in qualities if q.get('literal_substring'))
    ratios = [float(q.get('word_overlap_ratio') or 0.0) for q in qualities]
    by_strategy: dict[str, Counter[str]] = {}
    for q in qualities:
        strategy = str(q.get('rewrite_strategy') or 'unknown')
        bucket = by_strategy.setdefault(strategy, Counter())
        bucket['cases'] += 1
        if q.get('literal_substring'):
            bucket['literal_substring_hits'] += 1
        bucket['overlap_ratio_sum_x10000'] += int(round(float(q.get('word_overlap_ratio') or 0.0) * 10000))
    return {
        'schema_version': 1,
        'cases': total,
        'literal_substring_hits': literal_hits,
        'literal_substring_rate': (literal_hits / total) if total else 0.0,
        'avg_word_overlap_ratio': (sum(ratios) / total) if total else 0.0,
        'max_word_overlap_ratio': max(ratios) if ratios else 0.0,
        'query_types': dict(sorted(Counter(str(c.get('query_type') or c.get('category') or 'unknown') for c in case_list).items())),
        'rewrite_strategies': {
            strategy: {
                'cases': int(counter['cases']),
                'literal_substring_hits': int(counter['literal_substring_hits']),
                'avg_word_overlap_ratio': (
                    int(counter['overlap_ratio_sum_x10000']) / max(int(counter['cases']), 1) / 10000
                ),
            }
            for strategy, counter in sorted(by_strategy.items())
        },
    }

FORBIDDEN_REDACTED_KEYS = {
    'query',
    'content',
    'snippet',
    'snippets',
    'body',
    'text',
    'raw',
    'raw_text',
    'private',
    'private_note',
    'private_notes',
    'bounded_context_note',
    'context_notes',
    'sender',
    'sender_name',
    'conversation_title',
    'account_label',
    'display_name',
    'citation',
    'citations',
    'expected_citation',
    'expected_citations',
    'expected_any_citation',
    'expected_all_citations',
    'conversation_id',
    'expected_any_conversation_id',
    'path',
    'paths',
    'model_path',
    'vault_path',
    'token',
    'api_key',
    'secret',
}

PRIVATE_PATH_RE = re.compile(r'(/Users/[^\s\"\'<>)]*|/Volumes/[^\s\"\'<>)]*|[A-Za-z]:\\\\[^\s\"\'<>)]*)')
TOKEN_RE = re.compile(r'(?i)(bearer\s+[a-z0-9._\-]{12,}|trove-local-[a-z0-9._\-]{8,}|(api[_-]?key|secret|token)\s*[:=]\s*[\"\']?[A-Za-z0-9_\-.]{16,})')
MEDIA_PATH_RE = re.compile(r'(?i)\.(jpg|jpeg|png|gif|webp|heic|mp3|m4a|wav|amr|silk|mp4|mov|dat)(\b|$)')


class EvalSchemaError(ValueError):
    pass


class RedactionError(EvalSchemaError):
    pass


@dataclass(frozen=True)
class CasePackStats:
    cases: int
    categories: dict[str, int]
    source_families: dict[str, int]
    query_types: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            'cases': self.cases,
            'categories': self.categories,
            'source_families': self.source_families,
            'query_types': self.query_types,
        }


def stable_hash(value: Any, *, length: int = 16) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()[:length]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def expected_citations(case: dict[str, Any]) -> list[str]:
    oracle = case.get('oracle') or {}
    out: list[str] = []
    for key in ('expected_citation', 'expected_citations', 'expected_any_citation', 'expected_all_citations'):
        out.extend(str(v) for v in _as_list(case.get(key)) if v)
    for key in ('expected_any_citation', 'expected_all_citations'):
        out.extend(str(v) for v in _as_list(oracle.get(key)) if v)
    # Preserve first-seen order while de-duping.
    return list(dict.fromkeys(out))


def expected_conversations(case: dict[str, Any]) -> list[str]:
    oracle = case.get('oracle') or {}
    vals = _as_list(case.get('expected_any_conversation') or case.get('expected_any_conversation_id'))
    vals.extend(_as_list(oracle.get('expected_any_conversation') or oracle.get('expected_any_conversation_id')))
    return list(dict.fromkeys(str(v) for v in vals if v))


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise EvalSchemaError('case must be an object')
    if 'query' not in case or not str(case.get('query') or '').strip():
        raise EvalSchemaError('case.query is required')
    category = str(case.get('category') or 'exact_sparse')
    if category not in EVAL_CATEGORIES:
        raise EvalSchemaError(f'unsupported category: {category}')
    oracle = dict(case.get('oracle') or {})
    if case.get('expected_citation'):
        oracle.setdefault('expected_any_citation', [case['expected_citation']])
    if case.get('expected_citations'):
        oracle.setdefault('expected_any_citation', case['expected_citations'])
    if case.get('expected_any_citation'):
        oracle.setdefault('expected_any_citation', case['expected_any_citation'])
    normalized = dict(case)
    normalized['schema_version'] = int(case.get('schema_version') or SCHEMA_VERSION)
    normalized['category'] = category
    normalized['case_id'] = str(case.get('case_id') or stable_hash(case.get('query')))
    normalized['query'] = str(case.get('query'))
    normalized['limit'] = int(case.get('limit') or 10)
    normalized['filters'] = dict(case.get('filters') or {})
    normalized['tags'] = [str(t) for t in _as_list(case.get('tags'))]
    normalized['query_type'] = str(case.get('query_type') or category)
    normalized['oracle'] = oracle
    return normalized


def validate_case(case: dict[str, Any], *, allow_private: bool = True) -> dict[str, Any]:
    normalized = normalize_case(case)
    oracle = normalized.get('oracle') or {}
    if not allow_private:
        validate_redacted_artifact(normalized)
    for key in oracle:
        if key not in ORACLE_KEYS:
            # Keep schema forward-compatible but reject likely raw debug blobs.
            if key in FORBIDDEN_REDACTED_KEYS:
                raise EvalSchemaError(f'forbidden oracle key: {key}')
    return normalized


def load_jsonl_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSchemaError(f'{path.name}:{idx}: invalid json') from exc
        if isinstance(obj, dict) and obj.get('record_type') in {'manifest', 'metadata'}:
            continue
        cases.append(validate_case(obj, allow_private=True))
    return cases


def load_case_pack(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == '.jsonl':
        return load_jsonl_cases(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
        raw_cases = data
    elif isinstance(data, dict):
        raw_cases = data.get('cases') or []
    else:
        raise EvalSchemaError('case pack must be a json list, json object with cases, or jsonl')
    return [validate_case(c, allow_private=True) for c in raw_cases]


def case_pack_stats(cases: Iterable[dict[str, Any]]) -> CasePackStats:
    case_list = list(cases)
    categories = Counter(str(c.get('category') or 'unknown') for c in case_list)
    source_families = Counter(str(c.get('source_family') or (c.get('filters') or {}).get('source_type') or 'message') for c in case_list)
    query_types = Counter(str(c.get('query_type') or c.get('category') or 'unknown') for c in case_list)
    return CasePackStats(
        cases=len(case_list),
        categories=dict(sorted(categories.items())),
        source_families=dict(sorted(source_families.items())),
        query_types=dict(sorted(query_types.items())),
    )


def redact_case(case: dict[str, Any]) -> dict[str, Any]:
    c = validate_case(case, allow_private=True)
    oracle = c.get('oracle') or {}
    redacted = {
        'schema_version': SCHEMA_VERSION,
        'case_ref': stable_hash(c.get('case_id')),
        'case_hash': stable_hash(c.get('case_id')),
        'category': c.get('category'),
        'query_type': c.get('query_type'),
        'query_hash': stable_hash(c.get('query')),
        'query_length': len(c.get('query') or ''),
        'tags': c.get('tags') or [],
        'source_family': c.get('source_family') or (c.get('filters') or {}).get('source_type') or 'message',
        'filter_keys': sorted((c.get('filters') or {}).keys()),
        'oracle_types': sorted([k for k, v in oracle.items() if v not in (None, [], {}, False)]),
        'expected_citation_hashes': [stable_hash(v) for v in expected_citations(c)],
        'expected_conversation_hashes': [stable_hash(v) for v in expected_conversations(c)],
        'context_required': bool(c.get('context_oracle') or oracle.get('context_anchor') or c.get('context')),
        'quality': {
            'rewrite_strategy': case_quality(c).get('rewrite_strategy'),
            'literal_substring': bool(case_quality(c).get('literal_substring')),
            'word_overlap_ratio': round(float(case_quality(c).get('word_overlap_ratio') or 0.0), 6),
            'query_token_count': int(case_quality(c).get('query_token_count') or 0),
            'expected_token_count': int(case_quality(c).get('expected_token_count') or 0),
            'overlap_token_count': int(case_quality(c).get('overlap_token_count') or 0),
        },
    }
    validate_redacted_artifact(redacted)
    return redacted


def redact_case_pack(cases: Iterable[dict[str, Any]], *, created_at: str | None = None, inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    case_list = [validate_case(c, allow_private=True) for c in cases]
    payload = {
        'schema_version': SCHEMA_VERSION,
        'artifact_type': 'retrieval_eval_cases_redacted',
        'created_at': created_at,
        'privacy': {
            'raw_queries_included': False,
            'raw_snippets_included': False,
            'raw_citations_included': False,
            'private_paths_included': False,
            'token_values_included': False,
        },
        'stats': case_pack_stats(case_list).to_dict(),
        'case_quality': case_pack_quality_stats(case_list),
        'inventory': inventory or {},
        'cases': [redact_case(c) for c in case_list],
    }
    validate_redacted_artifact(payload)
    return payload


def _is_hash_key(key: str) -> bool:
    return key.endswith('_hash') or key.endswith('_hashes') or key in {'case_hash', 'query_hash'}


def _walk_redacted(obj: Any, path: str = '$') -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            leaf = key_s.split('.')[-1]
            if leaf in FORBIDDEN_REDACTED_KEYS and not _is_hash_key(leaf):
                raise RedactionError(f'redacted artifact contains forbidden key at {path}.{key_s}')
            if (leaf.endswith('_path') or leaf.endswith('_paths')) and leaf not in {'retrieval_paths'}:
                raise RedactionError(f'redacted artifact contains path-like key at {path}.{key_s}')
            _walk_redacted(value, f'{path}.{key_s}')
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            _walk_redacted(value, f'{path}[{idx}]')
    elif isinstance(obj, str):
        if PRIVATE_PATH_RE.search(obj):
            raise RedactionError(f'redacted artifact contains private path at {path}')
        if TOKEN_RE.search(obj):
            raise RedactionError(f'redacted artifact contains token-like value at {path}')
        if MEDIA_PATH_RE.search(obj) and ('/' in obj or '\\' in obj):
            raise RedactionError(f'redacted artifact contains media path at {path}')


def validate_redacted_artifact(data: Any) -> None:
    _walk_redacted(data)
