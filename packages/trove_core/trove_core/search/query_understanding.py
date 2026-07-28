from __future__ import annotations

from dataclasses import dataclass
import re


DOMAIN_TERMS = [
    '客户',
    '画像',
    '价格',
    '预算',
    '报价',
    '审批',
    '试点',
    '团队',
    '决定',
    '上线',
    'token',
    'evidence',
    'citation',
    'context',
    'vault',
    'zvec',
    'vector',
    '向量',
]

LOW_SIGNAL_TERMS = {'嗯', '啊', '哦', '好', '对', '呀', '哈'}
MULTI_HOP_MARKERS = (
    '关联脉络', '哪两条记录', '前后因果', '来龙去脉',
    '前因后果', '先后发生', '从什么到什么', 'how did', 'what led to',
)

SYNONYMS: dict[str, list[str]] = {
    '价格': ['报价', '预算', '太贵'],
    '报价': ['价格', '预算'],
    '预算': ['价格', '报价'],
    '决定': ['决策', '确认'],
    '审批': ['决策', '确认'],
    '上线': ['发布', '交付'],
    '客户': ['客户画像', '联系人'],
    '画像': ['客户画像', 'profile'],
    'token': ['local token', '令牌'],
    'zvec': ['vector', '向量'],
    'vector': ['zvec', '向量'],
    '向量': ['zvec', 'vector'],
    'context': ['上下文'],
    'evidence': ['证据', 'citation'],
    'citation': ['证据', 'evidence'],
}


@dataclass(frozen=True)
class QueryUnderstanding:
    original: str
    normalized: str
    terms: list[str]
    expansions: list[str]
    intents: list[str]

    @property
    def expanded_queries(self) -> list[str]:
        queries: list[str] = [self.original]
        if self.terms:
            joined = ' '.join(self.terms)
            if joined and joined not in queries:
                queries.append(joined)
        for term in self.terms + self.expansions:
            if term and term not in queries:
                queries.append(term)
        return queries[:8]

    def to_status(self) -> dict:
        return {
            'enabled': True,
            'term_count': len(self.terms),
            'expansion_count': len(self.expansions),
            'expanded_query_count': len(self.expanded_queries),
            'intents': self.intents,
        }


def _ascii_terms(query: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r'[A-Za-z0-9][A-Za-z0-9_\-]{1,}', query)]


def analyze_query(query: str, *, enabled: bool = True) -> QueryUnderstanding:
    normalized = ' '.join(str(query or '').strip().split())
    lowered = normalized.lower()
    if not enabled or not normalized:
        return QueryUnderstanding(normalized, lowered, [], [], [])

    multi_hop = any(marker in lowered for marker in MULTI_HOP_MARKERS)
    terms: list[str] = []
    for term in DOMAIN_TERMS:
        if term.lower() in lowered and term not in terms:
            terms.append(term)
    for term in _ascii_terms(normalized):
        if term not in terms:
            terms.append(term)
    if multi_hop and ' ' in normalized:
        for part in normalized.split():
            lowered_part = part.lower()
            if (
                len(part) >= 2
                and not any(marker in lowered_part for marker in MULTI_HOP_MARKERS)
                and part not in terms
            ):
                terms.append(part)
    if not terms and ' ' in normalized:
        for part in normalized.split():
            if len(part) >= 2 and part not in terms:
                terms.append(part)
    if not terms and len(normalized) <= 12:
        terms.append(normalized)

    terms = [term for term in terms if len(term) > 1 or term in DOMAIN_TERMS and term not in LOW_SIGNAL_TERMS]

    expansions: list[str] = []
    for term in terms:
        for exp in SYNONYMS.get(term.lower(), SYNONYMS.get(term, [])):
            if exp not in expansions and exp not in terms:
                expansions.append(exp)

    intents: list[str] = []
    if any(t in lowered for t in ('客户', '画像', 'profile', '联系人')):
        intents.append('customer_profile')
    if any(t in lowered for t in ('价格', '预算', '报价', '太贵')):
        intents.append('commercial_blocker')
    if any(t in lowered for t in ('决定', '决策', '审批', '上线')):
        intents.append('decision_history')
    if any(t in lowered for t in ('zvec', 'vector', '向量', 'token', 'context', 'evidence', 'citation')):
        intents.append('technical_project_memory')
    if multi_hop:
        intents.append('multi_hop')
    if not intents:
        intents.append('general_evidence')

    return QueryUnderstanding(normalized, lowered, terms[:8], expansions[:8], intents[:4])
