from __future__ import annotations

import re
from typing import Any

from trove_core.store.sqlite_store import vector_document_text


VECTOR_TEXT_V4_EXPERIMENT_VERSION = 4


def row_value(row: Any, key: str) -> str:
    try:
        if hasattr(row, 'keys') and key not in row.keys():
            return ''
        return str(row[key] or '')
    except Exception:
        return ''


def bounded_text(value: str, *, max_chars: int = 240) -> str:
    compact = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + '…'


INTENT_TAG_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('follow_up', ('下次', '明天', '后天', '今天', '周', '约', '跟进', '回访', '安排', '同步', '提醒', '推进')),
    ('decision', ('决定', '决策', '确认', '审批', '通过', '同意', '拍板', '上线', '试点', '选择', '定下来')),
    ('issue_risk', ('风险', '担心', '问题', '卡点', '阻碍', '异议', '反对', '不同意', '故障', '延迟', '不稳定')),
    ('customer_profile', ('客户', '老板', '负责人', '联系人', '团队', '需求', '痛点', '行业', '公司', '岗位')),
    ('commercial_blocker', ('价格', '报价', '预算', '太贵', '费用', '成本', '付款', '合同', '采购', '账期', '折扣')),
)


def infer_intent_tags(*texts: str) -> list[str]:
    haystack = '\n'.join(str(text or '') for text in texts)
    return [tag for tag, terms in INTENT_TAG_RULES if any(term in haystack for term in terms)]


def vector_document_text_v3(row: Any) -> str:
    """Current production vector text contract.

    Kept as a named wrapper so experiments can compare against v4 without
    touching production call sites or bumping the production vector version.
    """

    return vector_document_text(row)


def vector_document_text_v4(
    row: Any,
    *,
    previous_text: str = '',
    next_text: str = '',
    previous_actor: str = '',
    next_actor: str = '',
    max_neighbor_chars: int = 240,
) -> str:
    """Experimental local-only vector text.

    v4 intentionally lives beside the production v3 path.  It appends bounded
    same-thread neighbor context and structured intent tags, but callers must
    opt in explicitly; no production vector collection/version is changed.
    """

    base = vector_document_text_v3(row)
    content = row_value(row, 'content')
    prev = bounded_text(previous_text, max_chars=max_neighbor_chars)
    nxt = bounded_text(next_text, max_chars=max_neighbor_chars)
    tags = infer_intent_tags(content, prev, nxt)
    extra_parts = [
        f"结构化意图标签: {'; '.join(tags)}" if tags else '结构化意图标签: none',
        f"相邻上文说话人: {bounded_text(previous_actor, max_chars=80)}" if previous_actor else '',
        f"相邻上文: {prev}" if prev else '',
        f"相邻下文说话人: {bounded_text(next_actor, max_chars=80)}" if next_actor else '',
        f"相邻下文: {nxt}" if nxt else '',
    ]
    return '\n'.join([base, *(part for part in extra_parts if part.strip())])
