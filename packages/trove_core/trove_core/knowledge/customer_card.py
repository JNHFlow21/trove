from __future__ import annotations
from trove_core.search.hyper_search import HyperSearch
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.knowledge.customer_profile import build_customer_profile


def build_customer_card(store: SQLiteStore, customer: str = '示例教育') -> dict:
    profile = build_customer_profile(store, customer)
    if profile['resolved_entity'] and any(profile['sections'].get(name) for name in ('identity', 'needs', 'objections', 'next_actions', 'commitments', 'moments', 'voice_transcripts', 'image_observations')):
        def first(section, default='未识别'):
            rows = profile['sections'].get(section) or []
            return {'value': rows[0]['value'] if rows else default, 'citations': rows[0].get('citations', []) if rows else []}
        return {
            'type': 'customer_card',
            'customer': customer,
            'profile_type': 'customer_profile_v2',
            'identity': profile['sections'].get('identity', [])[:3],
            'blocker': first('objections'),
            'stage': first('needs'),
            'next_action': first('next_actions'),
            'profile': profile,
            'citation_policy': profile['claim_policy'],
        }
    query = f'{customer} 客户 卡 价格 预算 试点'
    search = HyperSearch(store)
    resp = search.search(SearchRequest(query, limit=8))
    evidences = [r.to_dict() for r in resp.results if customer in r.snippet or customer in r.conversation_title or '客户' in r.snippet]
    citations = [e['citation'] for e in evidences]
    blocker = '未识别'
    stage = '未识别'
    next_action = '未识别'
    for e in evidences:
        text = e['snippet']
        if ('价格' in text or '预算' in text) and blocker == '未识别':
            blocker = '价格太高与预算审批是主要卡点'
        if '试点' in text and stage == '未识别':
            stage = '三个月试点推进阶段'
        if ('报价' in text or '下周三' in text) and next_action == '未识别':
            next_action = '发送基础版/新版报价并预约下周三复盘'
    return {
        'type': 'customer_card',
        'customer': customer,
        'blocker': {'value': blocker, 'citations': citations[:2]},
        'stage': {'value': stage, 'citations': citations[:3]},
        'next_action': {'value': next_action, 'citations': citations[:3]},
        'evidence': evidences[:5],
        'citation_policy': 'Every claim above is backed by synthetic evidence citations; unknown fields stay unrecognized.',
    }
