from __future__ import annotations
from sqlite3 import Row

RankedRow = tuple[Row, list[str], float]


def _timestamp(row: Row) -> str:
    try:
        return str(row['timestamp'] or '')
    except Exception:
        return ''


def _fusion_key(row: Row) -> str:
    """Canonical evidence key for route fusion.

    Message routes often return a parent citation while chunk/vector routes
    return child citations with ``parent_citation``.  Ranking them as unrelated
    candidates makes exact, evidence, and semantic routes compete against the
    same evidence instead of reinforcing it.
    """
    try:
        parent = str(row['parent_citation'] or '')
        if parent:
            return parent
    except Exception:
        pass
    return str(row['citation'])


def fuse_ranked_rows(groups: list[tuple[str, list[Row], float]], limit: int) -> list[RankedRow]:
    """Weighted route fusion used as the backward-compatible baseline.

    The path score intentionally stays simple and deterministic: high-confidence
    lexical/parent-child routes get larger route weights, and repeated evidence
    across routes adds up.  The function returns route participation alongside
    the row so downstream explainability and eval reports can attribute wins.
    """
    by_citation: dict[str, RankedRow] = {}
    for path, rows, base_score in groups:
        for idx, row in enumerate(rows):
            citation = _fusion_key(row)
            score = base_score - (idx * 0.01)
            if citation in by_citation:
                old_row, paths, old_score = by_citation[citation]
                if path not in paths:
                    paths.append(path)
                by_citation[citation] = (old_row, paths, old_score + score)
            else:
                by_citation[citation] = (row, [path], score)
    fused = sorted(by_citation.values(), key=lambda item: (-item[2], _timestamp(item[0])))
    return fused[:limit]


def fuse_ranked_rows_rrf(groups: list[tuple[str, list[Row], float]], limit: int, *, rrf_k: int = 60) -> list[RankedRow]:
    """Reciprocal-rank fusion across retrieval routes.

    RRF makes the rank contract explicit for evaluation: route order matters
    more than arbitrary score magnitudes, while the third item in each group
    remains a route-confidence multiplier for exact/parent-child/vector balance.
    """
    by_citation: dict[str, RankedRow] = {}
    for path, rows, weight in groups:
        route_weight = max(float(weight), 0.01)
        for idx, row in enumerate(rows, start=1):
            citation = _fusion_key(row)
            score = route_weight / float(rrf_k + idx)
            if citation in by_citation:
                old_row, paths, old_score = by_citation[citation]
                if path not in paths:
                    paths.append(path)
                by_citation[citation] = (old_row, paths, old_score + score)
            else:
                by_citation[citation] = (row, [path], score)
    fused = sorted(by_citation.values(), key=lambda item: (-item[2], _timestamp(item[0])))
    return fused[:limit]


def route_counts(groups: list[tuple[str, list[Row], float]]) -> dict[str, int]:
    return {path: len(rows) for path, rows, _ in groups}
