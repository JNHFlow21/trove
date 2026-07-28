#!/usr/bin/env python3
"""Generate real local-private retrieval evaluation cases from a TROVE Vault.

Private executable cases may contain real queries, citations, and bounded notes,
but they are written only under the selected Vault proof/private directory.
Stdout contains a redacted aggregate summary only.
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
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from trove_core.search.eval_schema import SCHEMA_VERSION, case_pack_quality_stats, query_expected_quality, redact_case_pack, stable_hash
from trove_core.vault.config import VaultConfig

ROOT = Path(__file__).resolve().parents[1]

KEYWORDS: dict[str, list[str]] = {
    'customer_profile': ['客户', '画像', '需求', '负责人', '联系人', '公司', '身份', '老师', '学员', '校长'],
    'blocker_diagnosis': ['卡', '卡住', '阻碍', '问题', '原因', '失败', '不行', '不能', '没法', '价格', '预算', '审批', '担心'],
    'follow_up_action': ['跟进', '回复', '明天', '下周', '下次', '安排', '确认', '约', '会议', '待办', '负责', '推动'],
    'decision_history': ['决定', '结论', '最终', '选择', '方案', '确认', '同意', '不做', '改成', '优先'],
    'technical_project_memory': ['CI', 'API', 'Vault', 'ZVEC', 'vector', '向量', '隐私', 'runtime', '.venv', '评测', 'evidence', 'context', 'token', 'desktop', 'console'],
    'exact_sparse': ['ZVEC', 'Vault', 'API', 'CI', 'token', 'citation', 'evidence', 'context', 'SQLite', 'privacy', '预算', '审批', '试点'],
}

SEMANTIC_TEMPLATES = [
    ('blocker_diagnosis', '为什么这件事推进不下去'),
    ('follow_up_action', '最近需要跟进哪些事情'),
    ('decision_history', '之前做过什么关键决定'),
    ('customer_profile', '这个客户的核心需求是什么'),
    ('technical_project_memory', '本地知识库运行和检索有哪些关键讨论'),
]

SOURCE_TABLES = [
    ('favorite', 'favorites', 'citation', 'favorite_id', 'title', 'text', 'timestamp', 'account_id'),
    ('moment', 'moment_items', 'citation', 'moment_id', "'Moment'", 'text', 'timestamp', 'account_id'),
    ('transcript', 'transcripts', 'citation', 'transcript_id', "'Voice transcript'", 'text', 'created_at', "''"),
    ('image_observation', 'image_observations', 'citation', 'observation_id', "'Image observation'", "caption || char(10) || visible_text", 'created_at', "''"),
]

CATEGORY_PLAN = [
    ('semantic_paraphrase', 12),
    ('multi_hop', 8),
    ('negative_scope', 8),
    ('time_scoped', 8),
    ('sender_filter', 8),
    ('hard_distractor', 8),
    ('customer_profile', 10),
    ('cross_source_family', 12),
    ('voice_transcript', 8),
    ('image_observation', 8),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_root(path: Path, vault_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    repo_root = ROOT.resolve()
    proof_root = (vault_root / 'proof' / 'retrieval-eval').resolve()
    if resolved == repo_root or is_relative_to(resolved, repo_root):
        raise SystemExit('refusing to write real eval artifacts inside the source repo')
    if not (resolved == proof_root or is_relative_to(resolved, proof_root)):
        raise SystemExit('real eval artifacts must stay under the selected Vault proof/retrieval-eval directory')
    return resolved


def connect(vault_root: Path) -> sqlite3.Connection:
    db = vault_root / 'index' / 'trove.sqlite'
    if not db.exists():
        raise SystemExit('missing selected Vault SQLite index')
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','virtual table') AND name=?", (table,)).fetchone() is not None


def count_table(conn: sqlite3.Connection, table: str, where: str = '1=1') -> int:
    if not table_exists(conn, table):
        return 0
    return int(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {where}').fetchone()[0])


def compact(text: str | None) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def excerpt(text: str | None, limit: int = 180) -> str:
    value = compact(text)
    return value[:limit]


def phrase_from_text(text: str, keywords: list[str] | None = None, *, max_chars: int = 28) -> str:
    value = compact(text)
    if not value:
        return ''
    keywords = keywords or []
    lower = value.lower()
    hit = -1
    for kw in keywords:
        idx = lower.find(kw.lower())
        if idx >= 0:
            hit = idx
            break
    if hit < 0:
        hit = 0
    start = max(0, hit - 6)
    phrase = value[start:start + max_chars].strip(' ，。！？、,.!?:;；')
    return phrase or value[:max_chars]


def phrase_candidates_from_text(text: str, *, max_chars: int = 28) -> list[str]:
    value = compact(text)
    if not value:
        return []
    offsets = [0, len(value) // 5, len(value) // 3, len(value) // 2, max(0, (len(value) * 2) // 3), max(0, len(value) - max_chars)]
    out: list[str] = []
    for offset in offsets:
        phrase = value[offset:offset + max_chars].strip(' ，。！？、,.!?:;；')
        if len(phrase) >= min(6, max_chars) and phrase not in out:
            out.append(phrase)
    for match in re.finditer(r'[A-Za-z][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{6,}', value):
        phrase = match.group(0)[:max_chars].strip(' ，。！？、,.!?:;；')
        if len(phrase) >= min(6, max_chars) and phrase not in out:
            out.append(phrase)
    return out or [value[:max_chars]]


def like_escape(value: str) -> str:
    return str(value or '').replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def source_phrase_count(conn: sqlite3.Connection, source_family: str, phrase: str) -> int:
    for family, table, _citation_col, _id_col, _title_expr, text_expr, _ts_col, _account_expr in SOURCE_TABLES:
        if family != source_family or not table_exists(conn, table):
            continue
        return int(conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE ({text_expr}) LIKE ? ESCAPE '\\'",
            (f"%{like_escape(phrase)}%",),
        ).fetchone()[0])
    return 0


def source_phrase_from_text(conn: sqlite3.Connection, item: dict[str, Any], used_queries: set[str], *, max_chars: int = 24) -> str:
    """Pick a source-family anchor that is specific enough to rank.

    Favorites and Moments often share boilerplate prefixes.  A private eval
    query built from such a prefix creates an impossible oracle.  Prefer a
    later phrase whose source-family occurrence count is smallest.
    """

    family = str(item.get('source_family') or '')
    candidates = [c for c in phrase_candidates_from_text(str(item.get('content') or item.get('title') or ''), max_chars=max_chars) if c not in used_queries]
    if not candidates:
        candidates = phrase_candidates_from_text(str(item.get('content') or item.get('title') or ''), max_chars=max_chars)
    if not candidates:
        return ''
    scored = [(source_phrase_count(conn, family, phrase), idx, phrase) for idx, phrase in enumerate(candidates)]
    scored.sort(key=lambda row: (row[0] if row[0] > 0 else 999999, row[1]))
    return scored[0][2]


def keyword_terms(text: str, *, limit: int = 3) -> list[str]:
    value = compact(text)
    terms: list[str] = []
    for token in re.findall(r'[A-Za-z][A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,4}', value):
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


REWRITE_HINTS = [
    (('预算', '价格', '报价', '审批', '成交', '客户'), '推进阻力'),
    (('试点', '排期', '下周', '确认', '校长'), '试运行安排'),
    (('API', 'token', '127.0.0.1', 'local', 'search', 'context'), '本机鉴权'),
    (('evidence', 'citation', '证据', '答案'), '溯源要求'),
    (('UI', 'Web Console', '控制台', '搜索框', 'Settings', 'panel'), '界面布局'),
    (('语音', 'transcript', 'Voice'), '音频纪要'),
    (('图片', 'OCR', 'Image'), '图像识别'),
    (('朋友圈', 'moment', 'Moment'), '动态记录'),
    (('收藏', 'favorite', 'Favorite'), '收藏资料'),
]


STOP_TERMS = {
    '我们', '这个', '那个', '可以', '如果', '需要', '一下', '已经', '今天', '明天',
    '下周', '之前', '现在', '时候', '没有', '不是', '还是', '以及', '进行',
    '确认', '负责', '收到', '先做', '记录',
}


def candidate_terms(text: str) -> list[str]:
    value = compact(text)
    terms: list[str] = []
    # Prefer short CJK terms: one anchor term plus one new hint keeps overlap
    # below the 50% gate while preserving retrievability.
    for token in re.findall(r'[\u4e00-\u9fff]{2}|[A-Za-z][A-Za-z0-9_.-]{2,}', value):
        if token in STOP_TERMS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def salient_terms(text: str, *, limit: int = 3) -> list[str]:
    return candidate_terms(text)[:limit]


def global_term_count(conn: sqlite3.Connection, term: str) -> int:
    escaped = f"%{like_escape(term)}%"
    total = 0
    if table_exists(conn, 'messages'):
        total += int(conn.execute("SELECT COUNT(*) FROM messages WHERE content LIKE ? ESCAPE '\\'", (escaped,)).fetchone()[0])
    for _family, table, _citation_col, _id_col, title_expr, text_expr, _ts_col, _account_expr in SOURCE_TABLES:
        if not table_exists(conn, table):
            continue
        total += int(conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE ({title_expr}) LIKE ? ESCAPE '\\' OR ({text_expr}) LIKE ? ESCAPE '\\'",
            (escaped, escaped),
        ).fetchone()[0])
    return total


def rare_terms_for_text(conn: sqlite3.Connection, text: str, *, limit: int = 2) -> list[str]:
    terms = candidate_terms(text)
    if not terms:
        return []
    scored = []
    for idx, term in enumerate(terms[:40]):
        count = global_term_count(conn, term)
        scored.append((count if count > 0 else 999999, -len(term), idx, term))
    scored.sort()
    return [term for *_rest, term in scored[:limit]]


def negative_scope_filters(conn: sqlite3.Connection, source_account_id: str, anchor_terms: list[str]) -> dict[str, str] | None:
    """Return a populated alternate conversation where the anchor is absent.

    A score-floor negative must be a true no-result case, not merely a case
    with one excluded citation.  The query remains grounded in real evidence
    from ``source_account_id`` while the account filter selects a populated,
    different account whose messages contain no row matching all anchor terms.
    """

    source = str(source_account_id or '')
    terms = [str(term).strip() for term in anchor_terms if str(term).strip()]
    if not source or not terms:
        return None
    candidates = conn.execute(
        "SELECT account_id, conversation_id, COUNT(*) AS n FROM messages "
        "WHERE account_id <> ? AND account_id <> '' AND conversation_id <> '' "
        "GROUP BY account_id, conversation_id ORDER BY n ASC, account_id, conversation_id",
        (source,),
    ).fetchall()
    clauses = ' AND '.join("content LIKE ? ESCAPE '\\'" for _term in terms)
    params = tuple(f"%{like_escape(term)}%" for term in terms)
    for candidate in candidates:
        account_id = str(candidate['account_id'] or '')
        conversation_id = str(candidate['conversation_id'] or '')
        matches = int(conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE account_id = ? AND conversation_id = ? AND {_message_text_where()} AND {clauses}",
            (account_id, conversation_id, *params),
        ).fetchone()[0])
        if matches == 0:
            return {'account_id': account_id, 'conversation_id': conversation_id}
    return None


def semantic_hint(text: str, category: str = '') -> str:
    haystack = f'{category} {text}'.lower()
    for needles, hint in REWRITE_HINTS:
        if any(str(n).lower() in haystack for n in needles):
            return hint
    return {
        'customer_profile': '客户画像',
        'multi_hop': '关联脉络',
        'negative_scope': '排除干扰',
        'time_scoped': '时间线索',
        'sender_filter': '发言人线索',
        'hard_distractor': '同题辨别',
        'voice_transcript': '音频纪要',
        'image_observation': '图像识别',
    }.get(category, '相关线索')


def _quality_accepted(query: str, expected_text: str) -> bool:
    quality = query_expected_quality(query, expected_text)
    return not quality['literal_substring'] and float(quality['word_overlap_ratio']) <= 0.50


def rewrite_query(expected_text: str, *, category: str, strategy: str, second_text: str | None = None, preferred_terms: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """Deterministically rewrite an anchor into an independent eval query.

    First principle: a retrieval eval query may share topic vocabulary, but it
    must not be a copied contiguous substring.  We keep at most one short anchor
    term and add non-anchor task/fuzzy words until overlap is ≤50%.
    """

    text = compact(expected_text)
    combined = compact(f'{expected_text} {second_text or ""}')
    terms = list(dict.fromkeys([*(preferred_terms or []), *salient_terms(combined, limit=3)]))
    anchor_terms = terms[:3]
    anchor_text = ' '.join(anchor_terms)
    anchor = terms[0] if terms else ''
    other = terms[1] if len(terms) > 1 else ''
    third = terms[2] if len(terms) > 2 else ''
    hint = semantic_hint(combined, category)
    if strategy == 'task_question':
        candidates = [
            f'{anchor_text} 谁提过 {hint}',
            f'{anchor_text} 什么时候讨论过 {hint}',
            f'找一下 {hint} 的来龙去脉',
        ]
    elif strategy == 'fuzzy_memory':
        candidates = [
            f'{anchor_text} {hint} 线索',
            f'{hint} 线索',
            f'{anchor} 相关记忆',
        ]
    elif strategy == 'multi_hop':
        candidates = [
            f'{anchor_text} 关联脉络',
            f'{anchor} {other} {third} 哪两条记录串起 {hint}',
            f'{hint} 前后因果',
        ]
    elif strategy == 'negative':
        candidates = [
            f'{anchor_text} 之外的无关线索',
            f'{anchor} {other} 不要命中 {hint} 干扰项',
            '完全不存在的检索占位线索',
        ]
    elif strategy == 'time_sender_filter':
        candidates = [
            f'{anchor_text} {hint} 时间线索',
            f'{anchor_text} 指定范围内谁说过',
            f'{hint} 发言人线索',
        ]
    elif strategy == 'hard_distractor':
        candidates = [
            f'{anchor_text} {hint} 同题辨别',
            f'{anchor_text} 同主题里哪条才是正确线索',
            f'{hint} 正确会话',
        ]
    else:
        candidates = [
            f'{anchor_text} {hint} 线索',
            f'{hint} 相关记忆',
            f'找一下 {anchor} 背后的原因',
        ]
    for candidate in candidates:
        candidate = compact(candidate)
        if candidate and _quality_accepted(candidate, text):
            quality = query_expected_quality(candidate, text)
            quality['rewrite_strategy'] = strategy
            return candidate, quality
    base = compact(candidates[0] if candidates else f'{hint} 线索')
    for suffix in ('复盘', '记录', '背景', '线索'):
        candidate = compact(f'{base} {suffix}')
        if _quality_accepted(candidate, text):
            quality = query_expected_quality(candidate, text)
            quality['rewrite_strategy'] = strategy
            return candidate, quality
    candidate = compact(f'{hint} 外部记忆 复盘')
    quality = query_expected_quality(candidate, text)
    quality['rewrite_strategy'] = strategy
    return candidate, quality


def day_bounds(timestamp: str | None) -> tuple[str, str] | None:
    value = str(timestamp or '')
    if len(value) < 10:
        return None
    day = value[:10]
    return f'{day}T00:00:00', f'{day}T23:59:59'


def _message_text_where(alias: str = '') -> str:
    """Only use human-authored text messages as private eval anchors.

    Media placeholders (especially legacy unknown_binary rows) are retrieval
    implementation details, not user-query ground truth.  Keeping them out of
    generated private packs prevents drift when media classification improves.
    """

    prefix = f'{alias}.' if alias else ''
    return f"{prefix}content <> '' AND COALESCE(NULLIF({prefix}content_kind, ''), 'text') = 'text'"


def short_cjk_query(text: str) -> str:
    value = compact(text)
    for token in re.findall(r'[\u4e00-\u9fff]{2}', value):
        if token.strip():
            return token
    return value[:2]


def scaled_plan(max_cases: int) -> list[tuple[str, int]]:
    if max_cases <= 0:
        return [(cat, 0) for cat, _ in CATEGORY_PLAN]
    total = sum(count for _, count in CATEGORY_PLAN)
    if max_cases >= total:
        extra = max_cases - total
        plan = list(CATEGORY_PLAN)
        idx = 0
        while extra > 0:
            cat, count = plan[idx % len(plan)]
            plan[idx % len(plan)] = (cat, count + 1)
            extra -= 1
            idx += 1
        return plan
    allocated: list[tuple[str, int, float]] = []
    used = 0
    for cat, count in CATEGORY_PLAN:
        exact = (max_cases * count) / total
        base = int(exact)
        allocated.append((cat, base, exact - base))
        used += base
    remaining = max_cases - used
    by_remainder = sorted(range(len(allocated)), key=lambda i: (-allocated[i][2], i))
    counts = {cat: count for cat, count, _ in allocated}
    for idx in by_remainder[:remaining]:
        cat = allocated[idx][0]
        counts[cat] += 1
    return [(cat, counts[cat]) for cat, _ in CATEGORY_PLAN]


def message_rows_for_category(conn: sqlite3.Connection, category: str, limit: int) -> list[sqlite3.Row]:
    keywords = KEYWORDS.get(category, [])
    if not keywords:
        return []
    clauses = []
    params: list[str] = []
    for kw in keywords:
        clauses.append('(content LIKE ? OR conversation_title LIKE ? OR sender_name LIKE ?)')
        params.extend([f'%{kw}%', f'%{kw}%', f'%{kw}%'])
    sql = f"""
        SELECT * FROM messages
        WHERE {_message_text_where()} AND ({' OR '.join(clauses)})
        ORDER BY timestamp DESC
        LIMIT ?
    """
    return list(conn.execute(sql, (*params, limit)))


def recent_message_rows(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return list(conn.execute(f"SELECT * FROM messages WHERE {_message_text_where()} ORDER BY timestamp DESC LIMIT ?", (limit,)))


def source_rows(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = []
    for source_family, table, citation_col, id_col, title_expr, text_expr, ts_col, account_expr in SOURCE_TABLES:
        if not table_exists(conn, table):
            continue
        bucket: list[dict[str, Any]] = []
        sql = f"""
            SELECT {citation_col} AS citation, {id_col} AS source_id, {title_expr} AS title,
                   {text_expr} AS content, {ts_col} AS timestamp, {account_expr} AS account_id
            FROM {table}
            WHERE ({text_expr}) IS NOT NULL AND ({text_expr}) <> ''
            ORDER BY {ts_col} DESC
            LIMIT ?
        """
        for row in conn.execute(sql, (limit,)):
            item = dict(row)
            item['source_family'] = source_family
            bucket.append(item)
        if bucket:
            buckets.append(bucket)
    rows: list[dict[str, Any]] = []
    for idx in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if idx < len(bucket):
                rows.append(bucket[idx])
                if len(rows) >= limit:
                    return rows
    return rows


def inventory(conn: sqlite3.Connection) -> dict[str, Any]:
    date_row = conn.execute('SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts FROM messages').fetchone() if table_exists(conn, 'messages') else None
    families = {
        'private_chat': count_table(conn, 'messages', "conversation_type='private'"),
        'group_chat': count_table(conn, 'messages', "conversation_type='group'"),
        'contact': count_table(conn, 'observations', "source_type='contact'"),
        'moment': count_table(conn, 'moment_items') + count_table(conn, 'moment_interactions'),
        'favorite': count_table(conn, 'favorites'),
        'transcript': count_table(conn, 'transcripts'),
        'image_observation': count_table(conn, 'image_observations'),
    }
    return {
        'counts': {
            'accounts': count_table(conn, 'accounts'),
            'conversations': count_table(conn, 'conversations'),
            'messages': count_table(conn, 'messages'),
            'chunks': count_table(conn, 'evidence_chunks'),
        },
        'source_families': families,
        'date_range_present': bool(date_row and date_row['min_ts'] and date_row['max_ts']),
    }


def make_case(
    category: str,
    row: sqlite3.Row | dict[str, Any],
    query: str,
    *,
    query_type: str,
    tags: list[str] | None = None,
    expected_paths: list[str] | None = None,
    source_family: str = 'message',
    filters: dict[str, str] | None = None,
    negative: bool = False,
    quality: dict[str, Any] | None = None,
    expected_all_citations: list[str] | None = None,
    negative_excluded_citations: list[str] | None = None,
    hard_distractor_citation: str | None = None,
    second_row: sqlite3.Row | dict[str, Any] | None = None,
    expect_citation: bool = True,
) -> dict[str, Any]:
    row_d = dict(row)
    citation = row_d.get('citation')
    second_d = dict(second_row) if second_row is not None else {}
    conversation_id = row_d.get('conversation_id') or row_d.get('source_id') or ''
    source_type = row_d.get('source_type') or source_family
    oracle: dict[str, Any] = {
        'type': 'negative_no_results' if negative else ('expected_all_citations' if expected_all_citations else 'expected_any_citation'),
        'min_results': 0 if negative else 1,
        'expected_retrieval_paths_any': expected_paths or ['exact', 'fts'],
    }
    if negative:
        oracle['negative_no_results'] = True
    elif expect_citation:
        if citation:
            oracle['expected_any_citation'] = [citation]
        if expected_all_citations:
            oracle['expected_all_citations'] = expected_all_citations
            oracle['expected_any_citation'] = list(dict.fromkeys([*(oracle.get('expected_any_citation') or []), *expected_all_citations]))
        if conversation_id:
            oracle['expected_any_conversation_id'] = [conversation_id]
        oracle['expected_source_family'] = source_type
    if negative_excluded_citations:
        oracle['negative_excluded_citations'] = negative_excluded_citations
        oracle['type'] = 'negative_excluded_citations'
        oracle['min_results'] = 0
    if hard_distractor_citation:
        oracle['hard_distractor_citation_hash'] = stable_hash(hard_distractor_citation)
    case_id = f"rv-{category}-{stable_hash(str(citation) + '|' + query, length=12)}"
    content = row_d.get('content') or row_d.get('text') or ''
    if second_d:
        content = compact(f"{content}\n{second_d.get('content') or second_d.get('text') or ''}")
    if quality is None:
        quality = query_expected_quality(query, content)
        quality['rewrite_strategy'] = query_type
    return {
        'schema_version': SCHEMA_VERSION,
        'case_id': case_id,
        'category': category,
        'query_type': query_type,
        'tags': tags or [],
        'query': query,
        'limit': 10,
        'filters': filters or {},
        'source_family': source_type,
        'oracle': oracle,
        'quality': quality,
        'context': expect_citation and not negative and source_family == 'message',
        'context_oracle': ({'anchor_citation': citation, 'before': 5, 'after': 5} if citation and expect_citation and not negative and source_family == 'message' else {}),
        'private': {
            'anchor_citation': citation,
            'account_label': row_d.get('account_label') or row_d.get('account_id') or '',
            'conversation_title': row_d.get('conversation_title') or row_d.get('title') or '',
            'conversation_id': conversation_id,
            'sender_name': row_d.get('sender_name') or row_d.get('actor') or '',
            'timestamp': row_d.get('timestamp') or '',
            'bounded_context_note': excerpt(content),
        },
    }


def add_balanced(cases: list[dict[str, Any]], seen: set[str], case: dict[str, Any], *, max_cases: int, per_category_limit: int) -> None:
    if len(cases) >= max_cases:
        return
    # A case says “this query should find this citation”.  If the same query
    # and filters point at multiple citations in one category, the oracle is
    # under-specified: no ranker can put every duplicate at top-k.  De-dupe at
    # query+filter grain and let later rows/topoff add distinct anchors.
    key = f"{case['category']}|{case['query']}|{json.dumps(case.get('filters') or {}, ensure_ascii=False, sort_keys=True)}"
    if key in seen:
        return
    counts = Counter(c['category'] for c in cases)
    if counts[case['category']] >= per_category_limit:
        return
    seen.add(key)
    cases.append(case)


def generate_cases(vault_root: Path, *, max_cases: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conn = connect(vault_root)
    try:
        inv = inventory(conn)
        if inv['counts']['messages'] <= 0:
            raise SystemExit('selected Vault index has no messages')
        cases: list[dict[str, Any]] = []
        seen: set[str] = set()
        source_query_seen: set[str] = set()
        rows = recent_message_rows(conn, limit=max(max_cases * 4, 240))
        by_category_quota = dict(scaled_plan(max_cases))
        for category, quota in scaled_plan(max_cases):
            before = len(cases)
            if category == 'semantic_paraphrase':
                for idx, row in enumerate(rows):
                    strategy = ['paraphrase', 'task_question', 'fuzzy_memory'][idx % 3]
                    query, quality = rewrite_query(row['content'], category=category, strategy=strategy, preferred_terms=rare_terms_for_text(conn, row['content']))
                    add_balanced(
                        cases,
                        seen,
                        make_case(
                            category,
                            row,
                            query,
                            query_type=strategy,
                            tags=['real_anchor', strategy],
                            expected_paths=['exact', 'fts', 'vector'],
                            quality=quality,
                        ),
                        max_cases=max_cases,
                        per_category_limit=quota,
                    )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category == 'multi_hop':
                paired = 0
                for first, second in zip(rows, rows[1:]):
                    if first['conversation_id'] != second['conversation_id']:
                        continue
                    query, quality = rewrite_query(first['content'], category=category, strategy='multi_hop', second_text=second['content'], preferred_terms=rare_terms_for_text(conn, compact(str(first['content']) + ' ' + str(second['content']))))
                    add_balanced(
                        cases,
                        seen,
                        make_case(
                            category,
                            first,
                            query,
                            query_type='multi_hop',
                            tags=['real_anchor', 'multi_hop'],
                            expected_paths=['exact', 'fts', 'evidence'],
                            expected_all_citations=[first['citation'], second['citation']],
                            second_row=second,
                            quality=quality,
                        ),
                        max_cases=max_cases,
                        per_category_limit=quota,
                    )
                    paired += 1
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
                if paired == 0:
                    for first, second in zip(rows, rows[1:]):
                        query, quality = rewrite_query(first['content'], category=category, strategy='multi_hop', second_text=second['content'], preferred_terms=rare_terms_for_text(conn, compact(str(first['content']) + ' ' + str(second['content']))))
                        add_balanced(cases, seen, make_case(category, first, query, query_type='multi_hop', tags=['real_anchor', 'multi_hop', 'cross_conversation_fallback'], expected_paths=['exact', 'fts'], expected_all_citations=[first['citation'], second['citation']], second_row=second, quality=quality), max_cases=max_cases, per_category_limit=quota)
                        if Counter(c['category'] for c in cases)[category] >= quota:
                            break
            elif category == 'negative_scope':
                for idx, row in enumerate(rows):
                    positive = rows[(idx + 3) % len(rows)] if rows else row
                    if positive['citation'] == row['citation']:
                        continue
                    anchor_terms = rare_terms_for_text(conn, positive['content'])
                    target_filters = negative_scope_filters(conn, positive['account_id'], anchor_terms)
                    if not target_filters:
                        continue
                    query, quality = rewrite_query(positive['content'], category=category, strategy='negative', preferred_terms=anchor_terms)
                    add_balanced(
                        cases,
                        seen,
                        make_case(
                            category,
                            positive,
                            query,
                            query_type='negative_no_results',
                            tags=['negative', 'scope_proven_no_result'],
                            expected_paths=['exact', 'fts'],
                            filters=target_filters,
                            negative=True,
                            quality=quality,
                            expect_citation=False,
                        ),
                        max_cases=max_cases,
                        per_category_limit=quota,
                    )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category == 'time_scoped':
                for row in rows:
                    bounds = day_bounds(row['timestamp'])
                    if not bounds:
                        continue
                    query, quality = rewrite_query(row['content'], category=category, strategy='time_sender_filter', preferred_terms=rare_terms_for_text(conn, row['content']))
                    add_balanced(
                        cases,
                        seen,
                        make_case(
                            category,
                            row,
                            query,
                            query_type='time_filter_rewrite',
                            tags=['real_anchor', 'time_filter'],
                            expected_paths=['exact', 'fts', 'evidence'],
                            filters={'since': bounds[0], 'until': bounds[1]},
                            quality=quality,
                        ),
                        max_cases=max_cases,
                        per_category_limit=quota,
                    )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category == 'sender_filter':
                for row in rows:
                    if not compact(row['sender_name']):
                        continue
                    query, quality = rewrite_query(row['content'], category=category, strategy='time_sender_filter', preferred_terms=rare_terms_for_text(conn, row['content']))
                    add_balanced(
                        cases,
                        seen,
                        make_case(
                            category,
                            row,
                            query,
                            query_type='sender_filter_rewrite',
                            tags=['real_anchor', 'sender_filter'],
                            expected_paths=['exact', 'fts', 'evidence'],
                            filters={'sender': row['sender_name']},
                            quality=quality,
                        ),
                        max_cases=max_cases,
                        per_category_limit=quota,
                    )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category == 'hard_distractor':
                for row in rows:
                    distractor = next((other for other in rows if other['citation'] != row['citation'] and other['conversation_id'] != row['conversation_id'] and set(salient_terms(row['content'], limit=4)) & set(salient_terms(other['content'], limit=4))), None)
                    if distractor is None:
                        distractor = next((other for other in rows if other['citation'] != row['citation'] and other['conversation_id'] != row['conversation_id']), None)
                    if distractor is None:
                        continue
                    query, quality = rewrite_query(row['content'], category=category, strategy='hard_distractor', preferred_terms=rare_terms_for_text(conn, row['content']))
                    add_balanced(
                        cases,
                        seen,
                        make_case(
                            category,
                            row,
                            query,
                            query_type='hard_distractor_rewrite',
                            tags=['real_anchor', 'hard_distractor'],
                            expected_paths=['exact', 'fts', 'evidence'],
                            negative_excluded_citations=[distractor['citation']],
                            hard_distractor_citation=distractor['citation'],
                            quality=quality,
                        ),
                        max_cases=max_cases,
                        per_category_limit=quota,
                    )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category == 'customer_profile':
                profile_rows = message_rows_for_category(conn, 'customer_profile', limit=max(max_cases, quota * 4))
                if not profile_rows:
                    profile_rows = rows
                for row in profile_rows:
                    query, quality = rewrite_query(row['content'], category=category, strategy='task_question', preferred_terms=rare_terms_for_text(conn, row['content']))
                    if query:
                        add_balanced(
                            cases,
                            seen,
                            make_case(
                                category,
                                row,
                                query,
                                query_type='task_question',
                                tags=['real_anchor', 'customer_profile', 'task_question'],
                                expected_paths=['exact', 'fts'],
                                quality=quality,
                            ),
                            max_cases=max_cases,
                            per_category_limit=quota,
                        )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category == 'cross_source_family':
                for item in source_rows(conn, limit=max(max_cases, quota * 4)):
                    query, quality = rewrite_query(str(item.get('content') or item.get('title') or ''), category=category, strategy='fuzzy_memory', preferred_terms=rare_terms_for_text(conn, str(item.get('content') or item.get('title') or '')))
                    if query:
                        source_query_seen.add(query)
                        add_balanced(cases, seen, make_case(category, item, query, query_type='fuzzy_memory', tags=['real_anchor', item['source_family']], expected_paths=['evidence', 'parent_child', 'fts'], source_family=item['source_family'], filters={'source_type': item['source_family']}, quality=quality), max_cases=max_cases, per_category_limit=quota)
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            elif category in {'voice_transcript', 'image_observation'}:
                wanted_family = 'transcript' if category == 'voice_transcript' else 'image_observation'
                for item in source_rows(conn, limit=max(max_cases, quota * 6)):
                    if item.get('source_family') != wanted_family:
                        continue
                    query, quality = rewrite_query(str(item.get('content') or item.get('title') or ''), category=category, strategy='fuzzy_memory', preferred_terms=rare_terms_for_text(conn, str(item.get('content') or item.get('title') or '')))
                    if query:
                        source_query_seen.add(query)
                        add_balanced(
                            cases,
                            seen,
                            make_case(
                                category,
                                item,
                                query,
                                query_type='fuzzy_memory',
                                tags=['real_anchor', wanted_family, 'fuzzy_memory'],
                                expected_paths=['evidence', 'parent_child', 'fts'],
                                source_family=wanted_family,
                                filters={'source_type': wanted_family},
                                quality=quality,
                            ),
                            max_cases=max_cases,
                            per_category_limit=quota,
                        )
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
            if len(cases) == before and category == 'cross_source_family':
                # If a fixture or sparse Vault lacks extra source families, keep the
                # generator useful by falling back to message anchors while preserving
                # the category contract in the redacted stats.
                for row in rows:
                    query, quality = rewrite_query(row['content'], category=category, strategy='fuzzy_memory', preferred_terms=rare_terms_for_text(conn, row['content']))
                    add_balanced(cases, seen, make_case(category, row, query, query_type='fuzzy_memory', tags=['real_anchor', 'fallback_message'], expected_paths=['exact', 'fts'], quality=quality), max_cases=max_cases, per_category_limit=quota)
                    if Counter(c['category'] for c in cases)[category] >= quota:
                        break
        if len(cases) < max_cases:
            topoff_limit = max_cases - len(cases)
            for row in rows:
                query, quality = rewrite_query(row['content'], category='semantic_paraphrase', strategy='paraphrase', preferred_terms=rare_terms_for_text(conn, row['content']))
                if query:
                    add_balanced(cases, seen, make_case('semantic_paraphrase', row, query, query_type='paraphrase', tags=['real_anchor', 'topoff'], expected_paths=['exact', 'fts', 'evidence'], quality=quality), max_cases=max_cases, per_category_limit=by_category_quota.get('semantic_paraphrase', 12) + topoff_limit)
                if len(cases) >= max_cases:
                    break
        return cases[:max_cases], inv
    finally:
        conn.close()


def write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + '\n')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vault', help='Selected TROVE runtime Vault root. Defaults to TROVE_VAULT_ROOT or platform default.')
    parser.add_argument('--out-root', help='Output root under <vault>/proof/retrieval-eval. Defaults there.')
    parser.add_argument('--max-cases', type=int, default=200)
    parser.add_argument('--min-cases', type=int, default=1)
    parser.add_argument('--reuse-existing', action='store_true', help='Reuse the frozen private pack if it already exists instead of sampling the current index.')
    parser.add_argument('--force-overwrite', action='store_true', help='Overwrite an existing private pack. Use only when intentionally creating a new frozen pack.')
    parser.add_argument('--pack-stem', default='cases', help='Base file name for the frozen pack under private/redacted, without .local.jsonl/.redacted.json suffix.')
    parser.add_argument('--max-literal-substring-rate', type=float, default=0.0)
    parser.add_argument('--max-avg-word-overlap-ratio', type=float, default=0.50)
    args = parser.parse_args(argv)
    cfg = VaultConfig.resolve(args.vault)
    vault_root = cfg.root.expanduser().resolve()
    out_root = validate_output_root(Path(args.out_root) if args.out_root else vault_root / 'proof' / 'retrieval-eval', vault_root)
    pack_stem = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(args.pack_stem or 'cases')).strip('.-') or 'cases'
    private_path = out_root / 'private' / f'{pack_stem}.local.jsonl'
    redacted_path = out_root / 'redacted' / f'{pack_stem}.redacted.json'
    if private_path.exists() and args.reuse_existing:
        cases = read_jsonl(private_path)
        conn = connect(vault_root)
        try:
            inv = inventory(conn)
        finally:
            conn.close()
    else:
        if private_path.exists() and not args.force_overwrite:
            raise SystemExit('frozen private eval pack already exists; pass --reuse-existing or --force-overwrite')
        cases, inv = generate_cases(vault_root, max_cases=args.max_cases)
        if len(cases) < args.min_cases:
            raise SystemExit('not enough local evidence to generate requested eval case count')
        write_jsonl(private_path, cases)
    if len(cases) < args.min_cases:
        raise SystemExit('not enough local evidence to satisfy requested eval case count')
    quality = case_pack_quality_stats(cases)
    if float(quality['literal_substring_rate']) > args.max_literal_substring_rate + 1e-12:
        raise SystemExit('eval case quality gate failed: literal substring rate above max')
    if float(quality['avg_word_overlap_ratio']) > args.max_avg_word_overlap_ratio + 1e-12:
        raise SystemExit('eval case quality gate failed: average word overlap above max')
    created_at = now_iso()
    redacted_path.parent.mkdir(parents=True, exist_ok=True)
    redacted = redact_case_pack(cases, created_at=created_at, inventory=inv)
    redacted['private_artifact'] = {
        'stored_under_vault_proof_private': True,
        'file_name': private_path.name,
        'sha256_prefix': sha256_file(private_path),
        'case_count': len(cases),
    }
    from trove_core.search.eval_schema import validate_redacted_artifact
    validate_redacted_artifact(redacted)
    redacted_path.write_text(json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary = {
        'ok': True,
        'schema_version': SCHEMA_VERSION,
        'case_count': len(cases),
        'categories': dict(sorted(Counter(c['category'] for c in cases).items())),
        'source_families': dict(sorted(Counter(c.get('source_family') or 'message' for c in cases).items())),
        'case_quality': quality,
        'private_file_sha256_prefix': sha256_file(private_path),
        'private_file': private_path.name,
        'redacted_file': redacted_path.name,
        'raw_queries_printed': False,
        'raw_snippets_printed': False,
        'private_paths_printed': False,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
