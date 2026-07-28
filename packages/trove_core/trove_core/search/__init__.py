from __future__ import annotations

__all__ = ['HyperSearch', 'SearchRequest', 'SearchResponse', 'ContextService']


def __getattr__(name):
    if name == 'HyperSearch':
        from .hyper_search import HyperSearch
        return HyperSearch
    if name in {'SearchRequest', 'SearchResponse'}:
        from .query import SearchRequest, SearchResponse
        return {'SearchRequest': SearchRequest, 'SearchResponse': SearchResponse}[name]
    if name == 'ContextService':
        from .context import ContextService
        return ContextService
    raise AttributeError(name)
