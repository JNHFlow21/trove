from __future__ import annotations

from sqlite3 import Row

from trove_core.bounds import BoundedLimit, RERANK_CANDIDATES
from .fusion import RankedRow
from .query_understanding import QueryUnderstanding, analyze_query


def _row_text(row: Row) -> str:
    parts: list[str] = []
    for key in ('content', 'conversation_title', 'sender_name', 'source_type'):
        try:
            parts.append(str(row[key] or ''))
        except Exception:
            pass
    return '\n'.join(parts).lower()


def _row_value(row: Row, key: str) -> str:
    try:
        return str(row[key] or '')
    except Exception:
        return ''


def _timestamp(row: Row) -> str:
    try:
        return str(row['timestamp'] or '')
    except Exception:
        return ''


def feature_score(row: Row, paths: list[str], base_score: float, understanding: QueryUnderstanding, filters: dict[str, str] | None = None) -> tuple[float, list[str]]:
    text = _row_text(row)
    filters = filters or {}
    reasons: list[str] = []
    score = float(base_score)
    if understanding.original and understanding.original.lower() in text:
        score += 6.0
        reasons.append('exact_phrase')
    original = understanding.original.lower()
    if original:
        title = _row_value(row, 'conversation_title').lower()
        content = _row_value(row, 'content').lower()
        if title and original in title:
            score += 2.5
            reasons.append('title_phrase')
        elif content and original in content:
            score += 0.75
            reasons.append('content_phrase')
    overlap = sum(1 for term in understanding.terms if term.lower() in text)
    if overlap:
        score += overlap * 1.25
        reasons.append('query_terms')
    expansion_overlap = sum(1 for term in understanding.expansions if term.lower() in text)
    if expansion_overlap:
        score += expansion_overlap * 0.45
        reasons.append('expanded_terms')
    if 'exact' in paths:
        score += 1.0
        reasons.append('exact_route')
    if 'evidence' in paths:
        score += 0.8
        reasons.append('context_chunk_route')
    if 'vector' in paths:
        score += 0.4
        reasons.append('semantic_route')
    for key, value in filters.items():
        if not value:
            continue
        try:
            if key == 'sender' and (value == row['sender_id'] or value in row['sender_name']):
                score += 0.5
                reasons.append('filter_sender_match')
            elif key in {'source_family', 'scope_type'} and value in {'all', row['source_type']}:
                score += 0.7
                reasons.append('filter_source_match')
            elif key == 'source_type' and row['source_type'] == value:
                score += 0.7
                reasons.append('filter_source_match')
            elif key in row.keys() and row[key] == value:
                score += 0.4
                reasons.append(f'filter_{key}_match')
        except Exception:
            continue
    if not reasons:
        reasons.append('base_fusion')
    return score, sorted(set(reasons))


def rerank_with_features(
    ranked: list[RankedRow],
    query: str,
    *,
    filters: dict[str, str] | None = None,
    limit: int,
    understanding: QueryUnderstanding | None = None,
) -> tuple[list[RankedRow], dict]:
    limit = BoundedLimit(limit, field='reranker_candidate_limit', spec=RERANK_CANDIDATES)
    understanding = understanding or analyze_query(query)
    explanations: dict[str, list[str]] = {}
    rescored: list[RankedRow] = []
    for row, paths, base_score in ranked:
        score, reasons = feature_score(row, paths, base_score, understanding, filters)
        try:
            explanations[row['citation']] = reasons
        except Exception:
            pass
        rescored.append((row, paths, score))
    rescored.sort(key=lambda item: (-item[2], _timestamp(item[0])))
    return rescored[:limit], {
        'state': 'available',
        'mode': 'features',
        'feature_schema_version': 1,
        'candidate_count': len(ranked),
        'reason_codes': sorted({reason for reasons in explanations.values() for reason in reasons}),
    }
