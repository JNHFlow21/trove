from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import os
import sqlite3
from typing import Iterable

from trove_core.wechat.media.resources import discover_media_assets, summarize_media_references
from trove_core.wechat.scope import classify_wechat_identity

SENSITIVE_NAMES = {'key_store.json', 'wechat_keys.log'}
DENY_PARTS = {'auto_reply', 'LaunchAgents', 'codesign'}
LIVE_SEND_HINTS = ('wechatkos', 'auto_reply', 'send_wechat_message', 'wechat_sender')
DB_SUFFIXES = {'.db', '.sqlite', '.sqlite3'}
EXPORT_SUFFIXES = {'.jsonl', '.json', '.md'}


@dataclass(frozen=True)
class SourceCandidate:
    source_id: str
    redacted_path: str
    category: str
    size_bytes: int
    mtime: float
    file_count: int
    sqlite_count: int
    jsonl_count: int
    sensitive: bool = False
    importable: bool = False
    reason: str = ''
    media_counts: dict[str, int] = field(default_factory=dict)
    scope_counts: dict[str, int] = field(default_factory=dict)
    excluded_counts: dict[str, int] = field(default_factory=dict)
    coverage_gaps: list[dict[str, str]] = field(default_factory=list)
    moment_sources: int = 0
    moment_rows: int = 0
    favorite_sources: int = 0
    favorite_rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def stable_source_id(path: Path) -> str:
    return hashlib.sha256(str(path.expanduser().resolve()).encode('utf-8')).hexdigest()[:16]


def redact_path(path: Path) -> str:
    path = path.expanduser()
    home = Path.home()
    try:
        return '~/' + str(path.resolve().relative_to(home.resolve()))
    except ValueError:
        return f'<external>/{path.name}'


def classify_path(path: Path) -> tuple[str, bool, bool, str]:
    text = path.as_posix().lower()
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if name in SENSITIVE_NAMES or any(p.lower() in parts for p in DENY_PARTS):
        return 'key material denylist', True, False, 'denylisted sensitive runtime material'
    if any(hint in text for hint in LIVE_SEND_HINTS):
        return 'live-send/runtime denylist', True, False, 'live-send or auto-reply material is outside TROVE import scope'
    if 'decrypted' in parts and 'current' in parts:
        return 'runtime decrypted DB copies', False, True, 'preferred importable decrypted source tier'
    if 'decrypted' in parts and 'snapshots' in parts:
        return 'legacy decrypted snapshot', False, True, 'fallback only; do not combine with canonical DB without dedupe'
    if 'output' in parts or '_accounts' in parts:
        return 'historical exported outputs', False, True, 'fallback JSON/Markdown export tier'
    if any(suffix in DB_SUFFIXES for suffix in [path.suffix.lower()]):
        return 'sqlite archive', False, True, 'sqlite-like archive source'
    if path.suffix.lower() in EXPORT_SUFFIXES:
        return 'raw/export file', False, True, 'export-like source file'
    return 'unknown', False, False, 'not recognized as an importable WeChat source'


def _count_table_rows(db_path: Path, table_names: set[str] | None = None) -> int:
    try:
        with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            total = 0
            for table in tables:
                if table_names is not None and table not in table_names:
                    continue
                try:
                    total += int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                except sqlite3.DatabaseError:
                    continue
            return total
    except sqlite3.DatabaseError:
        return 0


def _auxiliary_source_rows(path: Path) -> tuple[int, int, int, int]:
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob('*.db')) if path.exists() else []
    moment_sources = 0
    moment_rows = 0
    favorite_sources = 0
    favorite_rows = 0
    for file in files:
        name = file.name.lower()
        if name in {'sns.db', 'moment.db', 'moments.db'}:
            moment_sources += 1
            moment_rows += _count_table_rows(file, {'SnsTimeLine', 'SnsMessage_tmp3'})
        if name in {'favorite.db', 'favorites.db', 'fav.db'}:
            favorite_sources += 1
            favorite_rows += _count_table_rows(file, None)
    return moment_sources, moment_rows, favorite_sources, favorite_rows


def summarize_path(path: Path) -> SourceCandidate:
    path = path.expanduser()
    if path.is_file():
        files = [path]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in {'.git', 'node_modules', '.venv', '__pycache__'}]
            for filename in filenames:
                files.append(Path(dirpath) / filename)
    size = 0
    mtime = 0.0
    sqlite_count = 0
    jsonl_count = 0
    sensitive = False
    categories: dict[str, int] = {}
    reasons: list[str] = []
    for file in files:
        try:
            st = file.stat()
        except OSError:
            continue
        size += st.st_size
        mtime = max(mtime, st.st_mtime)
        category, sens, importable, reason = classify_path(file)
        categories[category] = categories.get(category, 0) + 1
        if reason and reason not in reasons:
            reasons.append(reason)
        sensitive = sensitive or sens
        if file.suffix.lower() in DB_SUFFIXES:
            sqlite_count += 1
        if file.suffix.lower() == '.jsonl':
            jsonl_count += 1
    if not files and path.exists():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            pass
    if categories:
        category = max(categories.items(), key=lambda item: item[1])[0]
    else:
        category, sens, _, reason = classify_path(path)
        sensitive = sensitive or sens
        reasons.append(reason)
    importable = bool(sqlite_count or jsonl_count) and not sensitive and category not in {'live-send/runtime denylist', 'key material denylist'}
    media_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    coverage_gaps: list[dict[str, str]] = []
    moment_sources, moment_rows, favorite_sources, favorite_rows = _auxiliary_source_rows(path)
    if importable and path.exists() and path.is_dir() and sqlite_count:
        try:
            media_counts = summarize_media_references(discover_media_assets(path, limit_per_table=500))
        except Exception:
            media_counts = {}
        scope_counts, excluded_counts, coverage_gaps = summarize_scope(path)
    return SourceCandidate(
        source_id=stable_source_id(path),
        redacted_path=redact_path(path),
        category=category,
        size_bytes=size,
        mtime=mtime,
        file_count=len(files),
        sqlite_count=sqlite_count,
        jsonl_count=jsonl_count,
        sensitive=sensitive,
        importable=importable,
        reason='; '.join(reasons[:3]),
        media_counts=media_counts,
        scope_counts=scope_counts,
        excluded_counts=excluded_counts,
        coverage_gaps=coverage_gaps,
        moment_sources=moment_sources,
        moment_rows=moment_rows,
        favorite_sources=favorite_sources,
        favorite_rows=favorite_rows,
    )


def _add_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def summarize_scope(path: Path, *, limit_per_table: int = 5000) -> tuple[dict[str, int], dict[str, int], list[dict[str, str]]]:
    """Return redacted included/excluded/coverage-gap counts for a candidate account/source dir."""
    scope_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    gaps: list[dict[str, str]] = []
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in DB_SUFFIXES:
            gaps.append({'source': path.name, 'reason': 'standalone sqlite source needs adapter-specific scope inspection'})
        return scope_counts, excluded_counts, gaps
    # Contact identities.
    contact_db = path / 'contact.db'
    if contact_db.exists():
        try:
            with closing(sqlite3.connect(f'file:{contact_db}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'contact' in tables:
                    for row in conn.execute('SELECT username FROM contact LIMIT ?', (limit_per_table,)):
                        decision = classify_wechat_identity(row['username'], source_family='contact', is_contact=True)
                        _add_count(scope_counts, decision.scope_type)
                        if not decision.allowed:
                            _add_count(excluded_counts, decision.scope_type)
        except sqlite3.DatabaseError:
            gaps.append({'source': 'contact.db', 'reason': 'contact schema unreadable'})
    # Message conversations via Name2Id rows.
    for db_path in sorted(path.glob('message_*.db')):
        try:
            with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if 'Name2Id' not in tables:
                    gaps.append({'source': db_path.name, 'reason': 'missing Name2Id conversation map'})
                    continue
                for row in conn.execute('SELECT user_name FROM Name2Id LIMIT ?', (limit_per_table,)):
                    decision = classify_wechat_identity(row['user_name'], has_chat_history=True)
                    _add_count(scope_counts, decision.scope_type)
                    if not decision.allowed:
                        _add_count(excluded_counts, decision.scope_type)
        except sqlite3.DatabaseError:
            gaps.append({'source': db_path.name, 'reason': 'message schema unreadable'})
    for name in ('sns.db', 'moment.db', 'moments.db'):
        sns_path = path / name
        if sns_path.exists():
            _add_count(scope_counts, 'moment_sources')
            scope_counts['moment_rows'] = scope_counts.get('moment_rows', 0) + _count_table_rows(sns_path, {'SnsTimeLine', 'SnsMessage_tmp3'})
            break
    for name in ('favorite.db', 'favorites.db', 'fav.db'):
        fav_path = path / name
        if fav_path.exists():
            _add_count(scope_counts, 'favorite_sources')
            scope_counts['favorite_rows'] = scope_counts.get('favorite_rows', 0) + _count_table_rows(fav_path, None)
            break
    return scope_counts, excluded_counts, gaps


def inventory(paths: Iterable[str | Path]) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.exists():
            candidates.append(summarize_path(path))
    candidates.sort(key=lambda c: (c.importable, c.mtime, c.size_bytes), reverse=True)
    return candidates


def sqlite_row_count(path: Path) -> int | None:
    try:
        with closing(sqlite3.connect(f'file:{path}?mode=ro', uri=True)) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            total = 0
            for table in tables:
                if 'message' in table.lower() or table.lower() in {'msg', 'messages'}:
                    try:
                        total += int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                    except sqlite3.DatabaseError:
                        pass
            return total
    except sqlite3.DatabaseError:
        return None
