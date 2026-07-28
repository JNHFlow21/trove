from __future__ import annotations
from trove_core.search.hyper_search import HyperSearch
from trove_core.bounds import BoundedLimit, PROFILE_WIKI_REPORT
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore


def build_cited_report(store: SQLiteStore, query: str, limit: int = 5) -> dict:
    limit = BoundedLimit(limit, field='limit', spec=PROFILE_WIKI_REPORT)
    resp = HyperSearch(store).search(SearchRequest(query, limit=limit))
    findings = []
    for ev in resp.results:
        findings.append({
            'claim': ev.snippet,
            'citation': ev.citation,
            'speaker': ev.sender_name,
            'conversation': ev.conversation_title,
            'timestamp': ev.timestamp,
        })
    return {
        'type': 'cited_report',
        'query': query,
        'summary': 'Evidence-only report generated from retrieved synthetic WeChat messages.',
        'findings': findings,
        'citations': [f['citation'] for f in findings],
    }
