from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from trove_core.search.hyper_search import HyperSearch
from trove_core.bounds import BoundedLimit, PROFILE_WIKI_REPORT
from trove_core.search.query import SearchRequest
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint


def slugify(title: str) -> str:
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff_-]+', '-', title.strip()).strip('-')
    return s[:80] or 'wiki-page'


def build_wiki_page(store: SQLiteStore, title: str, *, limit: int = 8) -> dict[str, Any]:
    limit = BoundedLimit(limit, field='limit', spec=PROFILE_WIKI_REPORT)
    resp = HyperSearch(store).search(SearchRequest(title, limit=limit, include_vector=False))
    claims = []
    for item in resp.results[:limit]:
        claims.append({
            'claim': item.snippet,
            'citations': [item.citation],
            'context_anchor': item.context_anchor,
            'source_type': item.source_type,
        })
    return {
        'title': title,
        'slug': slugify(title),
        'claim_policy': 'cited_projection_only',
        'claims': claims,
        'citation_count': sum(len(c['citations']) for c in claims),
        'uncited_claims': 0,
    }


@mutation_entrypoint('wiki_write')
def write_wiki_page(
    vault_root: str | Path,
    title: str,
    page: dict[str, Any],
    *,
    write_session: VaultWriteSession | None = None,
) -> Path:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='wiki_write', write_session=write_session):
        cfg.ensure()
        out_dir = cfg.root / 'wiki'
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{page.get('slug') or slugify(title)}.redacted.json"
        path.write_text(json.dumps(page, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path
