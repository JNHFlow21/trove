from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import sqlite3
from typing import Any

from trove_core.store.repositories import MultimodalRepository


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip('\x00')
    return str(value).strip('\x00')


@dataclass(frozen=True)
class FavoriteRecord:
    favorite_id: str
    account_id: str
    citation: str
    timestamp: str = ''
    title: str = ''
    text: str = ''
    media_refs: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class FavoritesImporter:
    def __init__(self, favorite_db: Path, *, account_id: str):
        self.favorite_db = Path(favorite_db)
        self.account_id = account_id
        self.last_favorites: list[FavoriteRecord] = []

    def load(self, limit: int | None = None) -> list[FavoriteRecord]:
        self.last_favorites = []
        if not self.favorite_db.exists():
            return []
        out: list[FavoriteRecord] = []
        try:
            with sqlite3.connect(f'file:{self.favorite_db}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                candidates = [t for t in tables if any(token in t.lower() for token in ('fav', 'favorite'))]
                for table in candidates:
                    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                    query_cols = [c for c in cols if any(token in c.lower() for token in ('id', 'time', 'title', 'content', 'text', 'xml', 'url', 'media', 'image', 'path'))]
                    if not query_cols:
                        continue
                    query = 'SELECT rowid AS __rowid__, ' + ','.join(f'"{c}"' for c in query_cols) + f' FROM "{table}"'
                    if limit:
                        query += f' LIMIT {int(limit)}'
                    for row in conn.execute(query):
                        rowid = str(row['__rowid__'])
                        text_parts = []
                        title = ''
                        ts = ''
                        media_refs = []
                        for c in query_cols:
                            val = _text(row[c])
                            if not val:
                                continue
                            lc = c.lower()
                            if 'title' in lc:
                                title = title or val
                            elif 'time' in lc:
                                ts = ts or val
                            elif 'media' in lc or 'image' in lc or 'path' in lc or 'url' in lc:
                                media_refs.append({'field': c, 'state': 'metadata_only'})
                            elif 'content' in lc or 'text' in lc or 'xml' in lc:
                                text_parts.append(val)
                        if not title and not text_parts and not media_refs:
                            continue
                        fav_id = _stable('favorite', f'{self.account_id}:{table}:{rowid}')
                        out.append(FavoriteRecord(fav_id, self.account_id, f'trove://wechat/{self.account_id}/favorite/{fav_id}', ts, title, '\n'.join(text_parts)[:4000], media_refs, {'table': table, 'rowid': int(rowid)}))
        except sqlite3.DatabaseError:
            self.last_favorites = out
            return out
        self.last_favorites = out
        return out

    def persist_loaded_to_store(self, repo: MultimodalRepository) -> int:
        """Persist favorites already parsed outside the Vault writer."""
        report = repo.upsert_favorite_batch([favorite.to_dict() for favorite in self.last_favorites])
        return report['favorites']

    def import_to_store(self, repo: MultimodalRepository, *, limit: int | None = None) -> int:
        self.load(limit=limit)
        return self.persist_loaded_to_store(repo)
