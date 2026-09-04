from __future__ import annotations

from contextlib import closing

from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import re
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any

from trove_core.store.repositories import MultimodalRepository, SnsCacheMappingRecord
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.resources import MediaReference
from trove_core.wechat.media.source_registry import account_dir_hash, bind_account_assets, register_source_snapshot
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.wechat.media_mapping_assessment import d0_mapping_conclusion


HEX32_RE = re.compile(r'(?i)(?<![0-9a-f])([0-9a-f]{32})(?![0-9a-f])')
SPLIT_CACHE_KEY_RE = re.compile(r'(?i)(?:^|[/\\\\])([0-9a-f]{2})[/\\\\]([0-9a-f]{30})(?![0-9a-f])')
CACHE_TABLE_HINTS = ('cache', 'media', 'image', 'img', 'photo', 'thumb')
CACHE_TABLE_EXACT_DENY = {'snstimeline', 'snsmessage_tmp3', 'snsadtimeline'}
FEED_COLUMN_HINTS = ('feed', 'sns', 'tid', 'moment', 'timeline', 'object')
MEDIA_ID_COLUMN_HINTS = ('media', 'thumb', 'image', 'img', 'photo', 'id')
MEDIA_INDEX_COLUMN_HINTS = ('idx', 'index', 'seq', 'pos', 'order')
SNS_VIDEO_TYPES = {'4', '6', '43', '62', 'video', 'mp4', 'mov', 'm4v'}
SNS_CACHE_MAPPING_SCAN_LIMIT = 20_000


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _stable_scoped(prefix: str, account_id: str, value: str) -> str:
    return _stable(prefix, f'{account_id}:{value}')


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip('\x00')
    return str(value).strip('\x00')


def _iso_from_unix(value: Any) -> str:
    try:
        num = int(value)
    except Exception:
        return ''
    if num > 10_000_000_000:
        num = num // 1000
    if num <= 0:
        return ''
    return datetime.fromtimestamp(num, tz=timezone.utc).isoformat().replace('+00:00', 'Z')


def _strip_to_xml(value: str) -> str:
    text = value or ''
    pos = text.find('<')
    return text[pos:] if pos >= 0 else text


def _local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def _child_text(node: ET.Element, name: str) -> str:
    for child in list(node):
        if _local_name(child.tag) == name:
            return (child.text or '').strip('\x00').strip()
    return ''



def _hash_marker(value: str, *, algo: str = 'sha256', length: int = 16) -> str:
    text = str(value or '')
    if not text:
        return ''
    h = hashlib.md5(text.encode('utf-8')).hexdigest() if algo == 'md5' else hashlib.sha256(text.encode('utf-8')).hexdigest()
    return h[:length]


def _cache_key_markers(value: str) -> list[str]:
    text = str(value or '')
    keys = [m.group(1).lower() for m in HEX32_RE.finditer(text)]
    keys.extend((m.group(1) + m.group(2)).lower() for m in SPLIT_CACHE_KEY_RE.finditer(text))
    return list(dict.fromkeys(keys))


def _safe_db_text(value: Any, *, limit: int = 4096) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value[:limit].decode('utf-8', errors='ignore').strip('\x00')
    return str(value)[:limit].strip('\x00')


@dataclass(frozen=True)
class SnsCacheMapping:
    cache_key: str
    path_ref: str
    moment_id: str
    source_citation: str
    media_idx: int | None
    mapping_source: str
    confidence: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _all_descendant_text(node: ET.Element, names: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for child in node.iter():
        name = _local_name(child.tag)
        if name in names and name not in out:
            value = (child.text or '').strip('\x00').strip()
            if value:
                out[name] = value
        for attr_name, attr_value in child.attrib.items():
            if attr_name in names and attr_name not in out and attr_value:
                out[attr_name] = str(attr_value).strip()
    return out


def _structured_media_ref(node: ET.Element, idx: int) -> dict[str, Any]:
    values = _all_descendant_text(node, {'id', 'type', 'url', 'thumb', 'width', 'height', 'size'})
    media_type_raw = values.get('type') or 'image'
    media_type = 'video' if str(media_type_raw).lower() in SNS_VIDEO_TYPES else 'image' if media_type_raw in {'2', '3', 'image', 'img', 'pic'} else str(media_type_raw)
    url = values.get('url') or ''
    thumb = values.get('thumb') or ''
    ref: dict[str, Any] = {
        'idx': idx,
        'media_type': media_type,
        'state': 'metadata_only',
    }
    if url:
        ref['url_hash'] = _hash_marker(url, algo='sha256', length=16)
        ref['url_md5'] = _hash_marker(url, algo='md5', length=32)
        tail = url.rstrip('/').rsplit('/', 1)[-1]
        if tail and tail != url:
            ref['url_tail_hash'] = _hash_marker(tail, algo='sha256', length=16)
    if thumb:
        ref['thumb_hash'] = _hash_marker(thumb, algo='sha256', length=16)
        ref['thumb_md5'] = _hash_marker(thumb, algo='md5', length=32)
    for key in ('width', 'height'):
        if values.get(key):
            try:
                ref[key] = int(values[key])
            except ValueError:
                pass
    if values.get('size'):
        ref['size_hash'] = _hash_marker(values['size'], algo='sha256', length=16)
    if values.get('id'):
        ref['media_id_hash'] = _hash_marker(values['id'], algo='sha256', length=16)
    return ref


def _media_idx(media: dict[str, Any], fallback: int) -> int:
    value = media.get('idx')
    if value is None or value == '':
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _media_modality(media: dict[str, Any]) -> str:
    return 'video' if str(media.get('media_type') or '').lower() in SNS_VIDEO_TYPES else 'image'


def _media_citation(moment: MomentRecord, media: dict[str, Any], fallback: int) -> tuple[int, str, str]:
    idx = _media_idx(media, fallback)
    modality = _media_modality(media)
    prefix = 'video' if modality == 'video' else 'image'
    return idx, modality, f'{moment.citation}#{prefix}-{idx}'

def _first_text(root: ET.Element, name: str) -> str:
    for node in root.iter():
        if _local_name(node.tag) == name:
            return (node.text or '').strip('\x00').strip()
    return ''


def _parse_timeline_xml(raw: str) -> dict[str, Any] | None:
    text = _strip_to_xml(raw)
    if not text:
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    native_id = _first_text(root, 'id')
    author = _first_text(root, 'username')
    timestamp = _iso_from_unix(_first_text(root, 'createTime'))
    body = _first_text(root, 'contentDesc')
    media_refs: list[dict[str, Any]] = []
    for node in root.iter():
        if _local_name(node.tag) != 'media':
            continue
        media_refs.append(_structured_media_ref(node, len(media_refs)))
    return {
        'native_id': native_id,
        'author_id': author,
        'timestamp': timestamp,
        'text': body,
        'media_refs': media_refs,
        'parse_status': 'parsed',
    }


@dataclass(frozen=True)
class MomentRecord:
    moment_id: str
    account_id: str
    citation: str
    author_id: str = ''
    timestamp: str = ''
    text: str = ''
    link: dict[str, Any] | None = None
    media_refs: list[dict[str, Any]] | None = None
    comments: list[dict[str, Any]] | None = None
    likes: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MomentInteractionRecord:
    interaction_id: str
    moment_id: str
    account_id: str
    citation: str
    interaction_type: str
    actor_id: str = ''
    actor_name: str = ''
    text: str = ''
    timestamp: str = ''
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MomentsImporter:
    """Import WeChat SNS data with a strict table whitelist.

    Only SnsTimeLine creates moment_items.  SnsMessage_tmp3 is reserved for
    interactions, and SnsAdTimeLine is excluded by default.  Other SNS tables are
    counted in last_report['skipped_tables'] instead of being heuristically read.
    """

    TIMELINE_TABLE = 'SnsTimeLine'
    INTERACTION_TABLE = 'SnsMessage_tmp3'
    AD_TABLE = 'SnsAdTimeLine'

    def __init__(self, sns_db: Path, *, account_id: str, include_ads: bool = False):
        self.sns_db = Path(sns_db)
        self.account_id = account_id
        self.include_ads = bool(include_ads)
        self.last_report: dict[str, Any] = {}
        self.last_moments: list[MomentRecord] = []
        self.last_interactions: list[MomentInteractionRecord] = []
        self._sns_cache_mappings: list[SnsCacheMapping] = []

    def _empty_report(self) -> dict[str, Any]:
        return {
            'source_rows': 0,
            'timeline_rows': 0,
            'interaction_source_rows': 0,
            'imported_moments': 0,
            'imported_interactions': 0,
            'excluded_counts': {},
            'skipped_tables': {},
            'table_counts': {},
            'parse_success': 0,
            'parse_failed': 0,
            'orphan_interactions': 0,
            'interaction_type_counts': {},
            'media_refs_count': 0,
            'media_refs_nonempty_count': 0,
            'media_refs_nonempty_rate': 0.0,
            'sns_cache_inventory_files': 0,
            'sns_cache_mapping_records': 0,
            'sns_cache_mapping_tables': {},
            'sns_cache_media_cached': 0,
            'sns_cache_media_inventory_only': 0,
            'sns_cache_cached_conversion_rate': 0.0,
            'sns_cache_mapping_status': 'not_scanned',
            'raw_content_included': False,
        }

    def _table_names(self, conn: sqlite3.Connection) -> list[str]:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    def _table_count(self, conn: sqlite3.Connection, table: str) -> int:
        try:
            return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        except sqlite3.DatabaseError:
            return 0

    def _load_timeline_raw(self, conn: sqlite3.Connection, table: str, *, limit: int | None) -> list[MomentRecord]:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
        if not cols:
            return []
        wanted = [c for c in ['tid', 'id', 'sns_id', 'user_name', 'username', 'author', 'create_time', 'createTime', 'content', 'text', 'xml'] if c in cols]
        if not wanted:
            return []
        query = 'SELECT rowid AS __rowid__, ' + ','.join(f'"{c}"' for c in wanted) + f' FROM "{table}" ORDER BY rowid'
        if limit:
            query += f' LIMIT {int(limit)}'
        out: list[MomentRecord] = []
        for row in conn.execute(query):
            rowid = str(row['__rowid__'])
            native_id = _text(row['tid'] if 'tid' in row.keys() else row['id'] if 'id' in row.keys() else row['sns_id'] if 'sns_id' in row.keys() else rowid)
            author = _text(row['user_name'] if 'user_name' in row.keys() else row['username'] if 'username' in row.keys() else row['author'] if 'author' in row.keys() else '')
            timestamp = ''
            for c in ('create_time', 'createTime'):
                if c in row.keys():
                    timestamp = _iso_from_unix(row[c]) or _text(row[c])
                    if timestamp:
                        break
            raw_text = ''
            for c in ('content', 'text', 'xml'):
                if c in row.keys():
                    raw_text = _text(row[c])
                    if raw_text:
                        break
            parsed = _parse_timeline_xml(raw_text) if raw_text else None
            if parsed and parsed.get('native_id'):
                native_id = str(parsed.get('native_id') or native_id)
                author = str(parsed.get('author_id') or author)
                timestamp = str(parsed.get('timestamp') or timestamp)
                text = str(parsed.get('text') or '')
                media_refs = list(parsed.get('media_refs') or [])
                parse_status = 'parsed'
            else:
                text = raw_text
                media_refs = []
                parse_status = 'raw'
            if not text and not media_refs and not native_id:
                continue
            moment_id = _stable_scoped('moment', self.account_id, native_id or f'{table}:{rowid}')
            out.append(MomentRecord(
                moment_id=moment_id,
                account_id=self.account_id,
                citation=f'trove://wechat/{self.account_id}/moment/{moment_id}',
                author_id=author,
                timestamp=timestamp,
                text=text[:4000],
                media_refs=media_refs,
                metadata={'table': table, 'rowid': int(rowid), 'native_id_hash': hashlib.sha256((native_id or rowid).encode('utf-8')).hexdigest()[:16], 'parse_status': parse_status, '_feed_keys': [key for key in [native_id, _text(row['tid']) if 'tid' in row.keys() else ''] if key]},
            ))
        return out

    def _dedupe_moments(self, moments: list[MomentRecord]) -> list[MomentRecord]:
        chosen: dict[str, MomentRecord] = {}
        all_feed_keys: dict[str, list[str]] = {}
        for moment in moments:
            key = moment.moment_id
            all_feed_keys.setdefault(key, [])
            all_feed_keys[key].extend(str(v) for v in ((moment.metadata or {}).get('_feed_keys') or []) if v)
            prev = chosen.get(key)
            if prev is None:
                chosen[key] = moment
                continue
            prev_ts = str(prev.timestamp or '')
            cur_ts = str(moment.timestamp or '')
            if (cur_ts, len(moment.text or '')) >= (prev_ts, len(prev.text or '')):
                chosen[key] = moment
        out: list[MomentRecord] = []
        for moment_id, moment in chosen.items():
            metadata = dict(moment.metadata or {})
            metadata['_feed_keys'] = sorted(dict.fromkeys(all_feed_keys.get(moment_id, [])))
            out.append(MomentRecord(
                moment_id=moment.moment_id,
                account_id=moment.account_id,
                citation=moment.citation,
                author_id=moment.author_id,
                timestamp=moment.timestamp,
                text=moment.text,
                link=moment.link,
                media_refs=[
                    {key: value for key, value in item.items() if key != 'path_ref'}
                    for item in (moment.media_refs or [])
                ],
                comments=moment.comments,
                likes=moment.likes,
                metadata=metadata,
            ))
        return sorted(out, key=lambda m: (m.timestamp or '', m.moment_id))

    def _load_interactions(
        self,
        conn: sqlite3.Connection,
        table: str,
        moment_by_feed: dict[str, MomentRecord],
        *,
        limit: int | None,
        report: dict[str, Any],
    ) -> list[MomentInteractionRecord]:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
        required = {'feed_id', 'from_username', 'type', 'create_time'}
        if not required.issubset(set(cols)):
            report['skipped_tables'][table] = report['table_counts'].get(table, 0)
            return []
        select_cols = [c for c in ['local_id', 'create_time', 'type', 'feed_id', 'from_username', 'from_nickname', 'content'] if c in cols]
        query = 'SELECT rowid AS __rowid__, ' + ','.join(f'"{c}"' for c in select_cols) + f' FROM "{table}" ORDER BY create_time,rowid'
        if limit:
            query += f' LIMIT {int(limit)}'
        out: list[MomentInteractionRecord] = []
        for row in conn.execute(query):
            feed_id = _text(row['feed_id'])
            raw_type = _text(row['type'])
            if raw_type == '1':
                interaction_type = 'like'
            elif raw_type == '2':
                interaction_type = 'comment'
            else:
                interaction_type = raw_type or 'unknown'
            report['interaction_type_counts'][interaction_type] = report['interaction_type_counts'].get(interaction_type, 0) + 1
            actor_id = _text(row['from_username'])
            actor_name = _text(row['from_nickname']) if 'from_nickname' in row.keys() else ''
            create_time = _text(row['create_time'])
            timestamp = _iso_from_unix(create_time) or create_time
            moment = moment_by_feed.get(feed_id)
            orphan = moment is None
            if orphan:
                report['orphan_interactions'] += 1
            moment_id = moment.moment_id if moment else _stable('moment-orphan', f'{self.account_id}:{feed_id}')
            base_citation = moment.citation if moment else f'trove://wechat/{self.account_id}/moment/{moment_id}'
            interaction_id = _stable_scoped('moment-interaction', self.account_id, f'{feed_id}:{actor_id}:{raw_type}:{create_time}')
            text = _text(row['content']) if interaction_type == 'comment' and 'content' in row.keys() else ''
            out.append(MomentInteractionRecord(
                interaction_id=interaction_id,
                moment_id=moment_id,
                account_id=self.account_id,
                citation=f'{base_citation}/interaction/{interaction_id}',
                interaction_type=interaction_type,
                actor_id=actor_id,
                actor_name=actor_name,
                text=text[:2000],
                timestamp=timestamp,
                metadata={
                    'table': table,
                    'rowid': int(row['__rowid__']),
                    'feed_id_hash': hashlib.sha256(feed_id.encode('utf-8')).hexdigest()[:16],
                    'orphan': orphan,
                    'raw_type': raw_type,
                },
            ))
        return out

    def _load_all(self, limit: int | None = None) -> tuple[list[MomentRecord], list[MomentInteractionRecord]]:
        report = self._empty_report()
        moments: list[MomentRecord] = []
        interactions: list[MomentInteractionRecord] = []
        if not self.sns_db.exists():
            self.last_report = report
            self.last_moments = []
            self.last_interactions = []
            return [], []
        try:
            with closing(sqlite3.connect(f'file:{self.sns_db}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                tables = self._table_names(conn)
                interaction_tables: list[str] = []
                for table in tables:
                    count = self._table_count(conn, table)
                    report['table_counts'][table] = count
                    if table == self.TIMELINE_TABLE:
                        report['timeline_rows'] = count
                        report['source_rows'] += count
                        moments.extend(self._load_timeline_raw(conn, table, limit=limit))
                    elif table == self.INTERACTION_TABLE:
                        report['interaction_source_rows'] = count
                        report['source_rows'] += count
                        interaction_tables.append(table)
                    elif table == self.AD_TABLE:
                        if self.include_ads:
                            report['source_rows'] += count
                            # Ads stay out of default product surfaces; D1 only records the opt-in source kind.
                            report['skipped_tables'][table] = count
                        else:
                            report['excluded_counts']['moment_ad'] = report['excluded_counts'].get('moment_ad', 0) + count
                    elif 'sns' in table.lower() or 'moment' in table.lower():
                        report['skipped_tables'][table] = count
                moments = self._attach_local_cache(self._dedupe_moments(moments), scan_limit=limit)
                for key, value in self.last_report.items():
                    if key.startswith('sns_cache'):
                        report[key] = value
                moment_by_feed: dict[str, MomentRecord] = {}
                for moment in moments:
                    for key in ((moment.metadata or {}).get('_feed_keys') or []):
                        moment_by_feed[str(key)] = moment
                for table in interaction_tables:
                    interactions.extend(self._load_interactions(conn, table, moment_by_feed, limit=limit, report=report))
                report['imported_moments'] = len(moments)
                report['imported_interactions'] = len(interactions)
                report['parse_success'] = sum(1 for m in moments if (m.metadata or {}).get('parse_status') == 'parsed')
                report['parse_failed'] = sum(1 for m in moments if (m.metadata or {}).get('parse_status') == 'raw')
                report['media_refs_count'] = sum(len(m.media_refs or []) for m in moments)
                report['media_refs_nonempty_count'] = sum(1 for m in moments if m.media_refs)
                report['media_refs_nonempty_rate'] = round(report['media_refs_nonempty_count'] / len(moments), 6) if moments else 0.0
                report['media_cache_states'] = {}
                for m in moments:
                    for ref in m.media_refs or []:
                        state = str(ref.get('state') or 'metadata_only')
                        report['media_cache_states'][state] = report['media_cache_states'].get(state, 0) + 1
        except sqlite3.DatabaseError:
            pass
        self.last_report = report
        self.last_moments = moments
        self.last_interactions = interactions
        return moments, interactions

    def load(self, limit: int | None = None) -> list[MomentRecord]:
        moments, interactions = self._load_all(limit=limit)
        self.last_interactions = interactions
        return moments

    def _cache_index(self) -> dict[str, Path]:
        account_dir = self.sns_db.parent
        index: dict[str, Path] = {}
        if not account_dir.exists() or not account_dir.is_dir():
            return index
        roots: list[Path] = []
        for candidate in account_dir.rglob('*'):
            if not candidate.is_dir():
                continue
            parts = [part.lower() for part in candidate.parts]
            name = candidate.name.lower()
            if name == 'img' and len(parts) >= 2 and parts[-2] == 'sns':
                roots.append(candidate)
            elif name in {'bkg', 'publish'} and len(parts) >= 2 and parts[-2] == 'sns':
                roots.append(candidate)
        for root in roots:
            for file in root.rglob('*'):
                if not file.is_file():
                    continue
                filename = file.name.lower()
                parent = file.parent.name.lower()
                key = ''
                if len(filename) == 32 and all(c in '0123456789abcdef' for c in filename):
                    key = filename
                elif len(parent) == 2 and len(filename) == 30 and all(c in '0123456789abcdef' for c in parent + filename):
                    key = parent + filename
                if key and key not in index:
                    index[key] = file.resolve()
        return index

    def _known_moment_keys(self, moments: list[MomentRecord]) -> dict[str, MomentRecord]:
        out: dict[str, MomentRecord] = {}
        for moment in moments:
            for value in ((moment.metadata or {}).get('_feed_keys') or []):
                key = str(value or '').strip()
                if key:
                    out[key] = moment
        return out

    def _row_moment(self, row: sqlite3.Row, columns: list[str], moments_by_key: dict[str, MomentRecord]) -> tuple[MomentRecord | None, str]:
        feed_columns = [c for c in columns if any(hint in c.lower() for hint in FEED_COLUMN_HINTS)]
        for col in feed_columns + columns:
            value = _safe_db_text(row[col])
            if value in moments_by_key:
                return moments_by_key[value], value
        return None, ''

    def _row_cache_keys(self, row: sqlite3.Row, columns: list[str], cache_keys: set[str]) -> list[str]:
        found: list[str] = []
        for col in columns:
            for key in _cache_key_markers(_safe_db_text(row[col])):
                if key in cache_keys:
                    found.append(key)
        return list(dict.fromkeys(found))

    def _row_media_id_hashes(self, row: sqlite3.Row, columns: list[str], cache_keys: set[str]) -> list[str]:
        hashes: list[str] = []
        for col in columns:
            name = col.lower()
            if not any(hint in name for hint in MEDIA_ID_COLUMN_HINTS):
                continue
            value = _safe_db_text(row[col], limit=512).strip()
            if not value or value.lower() in cache_keys:
                continue
            if value in {'0', '-1'}:
                continue
            hashes.append(hashlib.sha256(value.encode('utf-8')).hexdigest()[:16])
        return list(dict.fromkeys(hashes))

    def _row_media_idx(self, row: sqlite3.Row, columns: list[str]) -> int | None:
        for col in columns:
            name = col.lower()
            if not any(hint in name for hint in MEDIA_INDEX_COLUMN_HINTS):
                continue
            try:
                value = int(row[col])
            except Exception:
                continue
            if value >= 0:
                return value
        return None

    def _sns_cache_key_mappings(self, moments: list[MomentRecord], cache: dict[str, Path], *, scan_limit: int | None = None) -> tuple[list[SnsCacheMapping], dict[str, Any]]:
        cache_keys = set(cache)
        report: dict[str, Any] = {
            'inventory_files': len(cache),
            'records': 0,
            'tables': {},
            'status': 'missing_cache_inventory' if not cache else 'inventory_only',
            'd0_mapping_conclusion': d0_mapping_conclusion(),
        }
        if not cache or not moments or not self.sns_db.exists():
            return [], report
        moments_by_key = self._known_moment_keys(moments)
        if not moments_by_key:
            return [], report
        raw_records: list[dict[str, Any]] = []
        remaining_budget = SNS_CACHE_MAPPING_SCAN_LIMIT if scan_limit is None else max(0, min(int(scan_limit), SNS_CACHE_MAPPING_SCAN_LIMIT))
        try:
            with closing(sqlite3.connect(f'file:{self.sns_db}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                for table in tables:
                    table_lower = table.lower()
                    if table == self.TIMELINE_TABLE or table_lower in CACHE_TABLE_EXACT_DENY:
                        continue
                    if not any(hint in table_lower for hint in CACHE_TABLE_HINTS):
                        continue
                    if remaining_budget <= 0:
                        report['status'] = 'scan_budget_exhausted'
                        break
                    info = list(conn.execute(f'PRAGMA table_info("{table}")'))
                    columns = [
                        r[1] for r in info
                        if 'blob' not in str(r[2] or '').lower()
                        and (
                            any(hint in str(r[1]).lower() for hint in FEED_COLUMN_HINTS)
                            or any(hint in str(r[1]).lower() for hint in MEDIA_ID_COLUMN_HINTS)
                            or any(hint in str(r[1]).lower() for hint in MEDIA_INDEX_COLUMN_HINTS)
                            or 'key' in str(r[1]).lower()
                            or 'path' in str(r[1]).lower()
                        )
                    ]
                    if not columns:
                        continue
                    try:
                        query_limit = remaining_budget
                        rows = conn.execute('SELECT rowid AS __rowid__, ' + ','.join(f'"{c}"' for c in columns) + f' FROM "{table}" ORDER BY rowid LIMIT {int(query_limit)}')
                    except sqlite3.DatabaseError:
                        continue
                    for row in rows:
                        remaining_budget -= 1
                        keys = self._row_cache_keys(row, columns, cache_keys)
                        if not keys:
                            continue
                        moment, feed_key = self._row_moment(row, columns, moments_by_key)
                        if moment is None:
                            continue
                        media_hashes = self._row_media_id_hashes(row, columns, cache_keys)
                        media_idx = self._row_media_idx(row, columns)
                        for key in keys:
                            raw_records.append({
                                'cache_key': key,
                                'table': table,
                                'rowid': int(row['__rowid__']),
                                'moment': moment,
                                'feed_key_hash': hashlib.sha256(feed_key.encode('utf-8')).hexdigest()[:16] if feed_key else '',
                                'media_id_hashes': media_hashes,
                                'media_idx': media_idx,
                            })
        except sqlite3.DatabaseError:
            return [], report
        if not raw_records:
            return [], report

        by_moment: dict[str, list[dict[str, Any]]] = {}
        for record in raw_records:
            by_moment.setdefault(record['moment'].moment_id, []).append(record)
        mappings: list[SnsCacheMapping] = []
        seen: set[tuple[str, str]] = set()
        for moment in moments:
            media_refs = list(moment.media_refs or [])
            if not media_refs:
                continue
            records = by_moment.get(moment.moment_id, [])
            if not records:
                continue
            records = sorted(records, key=lambda r: (str(r['table']), int(r['rowid']), str(r['cache_key'])))
            used_cache: set[str] = set()
            for media in media_refs:
                idx = _media_idx(media, 0)
                _, _, source_citation = _media_citation(moment, media, idx)
                media_hash = str(media.get('media_id_hash') or '')
                candidates = [
                    r for r in records
                    if r['cache_key'] not in used_cache and (
                        (media_hash and media_hash in set(r.get('media_id_hashes') or []))
                        or (r.get('media_idx') is not None and int(r['media_idx']) == idx)
                    )
                ]
                if not candidates:
                    continue
                chosen = candidates[0]
                used_cache.add(chosen['cache_key'])
                key = (moment.moment_id, chosen['cache_key'])
                if key in seen:
                    continue
                seen.add(key)
                mappings.append(SnsCacheMapping(
                    cache_key=chosen['cache_key'],
                    path_ref=str(cache[chosen['cache_key']]),
                    moment_id=moment.moment_id,
                    source_citation=source_citation,
                    media_idx=idx,
                    mapping_source=f"sns-db:{chosen['table']}",
                    confidence=0.95,
                    metadata={
                        'rowid': chosen['rowid'],
                        'feed_key_hash': chosen['feed_key_hash'],
                        'media_id_matched': bool(media_hash and media_hash in set(chosen.get('media_id_hashes') or [])),
                        'media_idx_matched': bool(chosen.get('media_idx') is not None and int(chosen['media_idx']) == idx),
                    },
                ))
            mapped_idx = {(mp.moment_id, str(mp.media_idx or 0)) for mp in mappings}
            unmapped_refs = [m for fallback, m in enumerate(media_refs) if (moment.moment_id, str(_media_idx(m, fallback))) not in mapped_idx]
            remaining = [r for r in records if r['cache_key'] not in used_cache]
            if len(unmapped_refs) == 1 and len(remaining) == 1:
                media = unmapped_refs[0]
                idx, _, source_citation = _media_citation(moment, media, 0)
                chosen = remaining[0]
                key = (moment.moment_id, chosen['cache_key'])
                if key not in seen:
                    seen.add(key)
                    mappings.append(SnsCacheMapping(
                        cache_key=chosen['cache_key'],
                        path_ref=str(cache[chosen['cache_key']]),
                        moment_id=moment.moment_id,
                        source_citation=source_citation,
                        media_idx=idx,
                        mapping_source=f"sns-db:{chosen['table']}",
                        confidence=0.8,
                        metadata={
                            'rowid': chosen['rowid'],
                            'feed_key_hash': chosen['feed_key_hash'],
                            'singleton_media': True,
                        },
                    ))
        report['records'] = len(mappings)
        report['tables'] = {}
        for mapping in mappings:
            table = mapping.mapping_source.split(':', 1)[-1]
            report['tables'][table] = report['tables'].get(table, 0) + 1
        report['status'] = 'mapped' if mappings else 'inventory_only'
        return mappings, report

    def _attach_local_cache(self, moments: list[MomentRecord], *, scan_limit: int | None = None) -> list[MomentRecord]:
        cache = self._cache_index()
        mappings, mapping_report = self._sns_cache_key_mappings(moments, cache, scan_limit=scan_limit)
        self._sns_cache_mappings = mappings
        self.last_report = dict(self.last_report) | {
            'sns_cache_inventory_files': mapping_report['inventory_files'],
            'sns_cache_mapping_records': mapping_report['records'],
            'sns_cache_mapping_tables': mapping_report['tables'],
            'sns_cache_mapping_status': mapping_report['status'],
            'sns_cache_d0_mapping_conclusion': mapping_report['d0_mapping_conclusion'],
        }
        if not cache:
            return [self._mark_missing_cache(moment) for moment in moments]
        mapping_by_citation = {m.source_citation: m for m in mappings}
        out: list[MomentRecord] = []
        for moment in moments:
            media_refs: list[dict[str, Any]] = []
            for fallback, media in enumerate(moment.media_refs or []):
                item = dict(media)
                idx, _, citation = _media_citation(moment, item, fallback)
                mapped = mapping_by_citation.get(citation)
                if mapped is not None:
                    item['state'] = 'cached'
                    item['path_ref'] = mapped.path_ref
                    item['cache_key'] = mapped.cache_key
                    item['cache_mapping_source'] = mapped.mapping_source
                elif item.get('url_md5') or item.get('thumb_md5') or item.get('media_id_hash'):
                    item['state'] = 'inventory_only'
                media_refs.append(item)
            out.append(MomentRecord(
                moment_id=moment.moment_id,
                account_id=moment.account_id,
                citation=moment.citation,
                author_id=moment.author_id,
                timestamp=moment.timestamp,
                text=moment.text,
                link=moment.link,
                media_refs=media_refs,
                comments=moment.comments,
                likes=moment.likes,
                metadata=moment.metadata,
            ))
        cached_count = sum(1 for moment in out for ref in (moment.media_refs or []) if ref.get('state') == 'cached')
        inventory_only_count = sum(1 for moment in out for ref in (moment.media_refs or []) if ref.get('state') == 'inventory_only')
        media_count = sum(len(moment.media_refs or []) for moment in out)
        self.last_report = dict(self.last_report) | {
            'sns_cache_media_cached': cached_count,
            'sns_cache_media_inventory_only': inventory_only_count,
            'sns_cache_cached_conversion_rate': round(cached_count / media_count, 6) if media_count else 0.0,
        }
        return out

    def _mark_missing_cache(self, moment: MomentRecord) -> MomentRecord:
        if not moment.media_refs:
            return moment
        refs = []
        for media in moment.media_refs or []:
            item = dict(media)
            if item.get('url_md5') or item.get('thumb_md5'):
                item['state'] = 'missing_local_cache'
            refs.append(item)
        return MomentRecord(moment.moment_id, moment.account_id, moment.citation, moment.author_id, moment.timestamp, moment.text, moment.link, refs, moment.comments, moment.likes, moment.metadata)

    def _media_asset_refs(self, moments: list[MomentRecord]) -> list[MediaReference]:
        refs: list[MediaReference] = []
        for moment in moments:
            for fallback, media in enumerate(moment.media_refs or []):
                idx, modality, citation = _media_citation(moment, media, fallback)
                basis = f"{self.account_id}:{moment.moment_id}:{idx}:{media.get('url_hash') or media.get('thumb_hash') or media.get('media_id_hash') or ''}"
                asset_id = _stable('asset', basis)
                prefix = 'video' if modality == 'video' else 'image'
                refs.append(MediaReference(
                    asset_id=asset_id,
                    account_id=self.account_id,
                    source_type='moment',
                    source_id=f'{moment.moment_id}#{prefix}-{idx}',
                    modality=modality,
                    media_type=str(media.get('media_type') or 'image'),
                    citation=citation,
                    content_hash=str(media.get('url_hash') or media.get('thumb_hash') or _stable('mediaref', basis)),
                    path_ref=None,
                    cache_state='source_available' if media.get('path_ref') else str(media.get('state') or 'metadata_only'),
                    metadata={
                        'moment_citation': moment.citation,
                        'media_idx': idx,
                        'url_hash': media.get('url_hash'),
                        'url_md5': media.get('url_md5'),
                        'thumb_hash': media.get('thumb_hash'),
                        'thumb_md5': media.get('thumb_md5'),
                        'cache_key': media.get('cache_key'),
                        'cache_mapping_source': media.get('cache_mapping_source'),
                        'width': media.get('width'),
                        'height': media.get('height'),
                    },
                ))
        return refs

    def persist_loaded_to_store(
        self,
        repo: MultimodalRepository,
        *,
        bind_source: bool = True,
    ) -> int:
        """Persist moments already parsed outside the Vault writer.

        ``bind_source`` stays enabled for the direct importer API.  Sync
        prepares the source snapshot separately and disables it here so the
        recursive manifest scan never runs while the writer is held.
        """
        moments = self.last_moments
        interactions = self.last_interactions
        persistence = repo.upsert_moment_batch(
            [
                {
                    'moment_id': moment.moment_id,
                    'account_id': moment.account_id,
                    'citation': moment.citation,
                    'author_id': moment.author_id,
                    'timestamp': moment.timestamp,
                    'text': moment.text,
                    'link': moment.link,
                    'media_refs': moment.media_refs,
                    'comments': moment.comments,
                    'metadata': {
                        key: value
                        for key, value in ((moment.metadata or {}) | {'likes': moment.likes or []}).items()
                        if not str(key).startswith('_')
                    },
                }
                for moment in moments
            ],
            [
                {
                    'interaction_id': interaction.interaction_id,
                    'moment_id': interaction.moment_id,
                    'account_id': interaction.account_id,
                    'citation': interaction.citation,
                    'interaction_type': interaction.interaction_type,
                    'actor_id': interaction.actor_id,
                    'actor_name': interaction.actor_name,
                    'text': interaction.text,
                    'timestamp': interaction.timestamp,
                    'metadata': interaction.metadata,
                }
                for interaction in interactions
            ],
        )
        media_link_report = MediaLinker(repo).link_references(self._media_asset_refs(moments))
        repo.upsert_sns_cache_mappings([
            SnsCacheMappingRecord(
                mapping_id=_stable('snsmap', f'{self.account_id}:{mapping.cache_key}:{mapping.source_citation}'),
                account_id=self.account_id,
                cache_key=mapping.cache_key,
                moment_id=mapping.moment_id,
                source_citation=mapping.source_citation,
                media_idx=mapping.media_idx,
                path_ref=None,
                mapping_source=mapping.mapping_source,
                confidence=mapping.confidence,
                metadata=mapping.metadata,
            )
            for mapping in self._sns_cache_mappings
        ])
        # Direct importer use (tests/operators) gets the same immutable source
        # binding as the full import orchestrator when the SNS source lives in
        # the Vault. Full import may later replace this with its run-level
        # snapshot revision.
        if bind_source and repo.store.path.parent.name == 'index':
            vault_root = repo.store.path.parent.parent
            cfg = VaultConfig.resolve(str(vault_root), env={})
            account_dir = self.sns_db.parent.resolve()
            snapshot_root = account_dir.parent
            if path_is_under(account_dir, cfg.root):
                snapshot = register_source_snapshot(cfg, repo.store, snapshot_root)
                bind_account_assets(
                    repo.store,
                    account_id=self.account_id,
                    snapshot=snapshot,
                    account_hash=account_dir_hash(account_dir),
                )
        self.last_report = dict(self.last_report) | {
            'imported_moments': len(moments),
            'imported_interactions': len(interactions),
            'media_assets_seen': media_link_report.assets_seen,
            'media_links_accepted': media_link_report.accepted_links,
            'media_links_excluded': media_link_report.excluded_links,
            'persistence_commits': persistence['commits'],
        }
        return len(moments)

    def import_to_store(self, repo: MultimodalRepository, *, limit: int | None = None) -> int:
        self._load_all(limit=limit)
        return self.persist_loaded_to_store(repo)
