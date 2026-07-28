from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import unicodedata
from typing import Any, Iterable

from .sqlite_store import SQLiteStore
from trove_core.domain.messages import Account, Conversation, Message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _load_json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _load_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


_ENTITY_USER_ID_KEYS = {'user_id', 'wechat_username', 'wechat_id', 'username', 'wxid', 'openim_id', 'primary_user_id'}
_ENTITY_ALIAS_KEYS = {'alias', 'aliases', 'remark', 'nickname', 'display_name', 'group_alias', 'group_display_name', 'group_name', 'name'}


def _normalize_identifier(value: Any) -> str:
    text = unicodedata.normalize('NFKC', str(value or '')).strip().casefold()
    return ' '.join(text.split())


def _flatten_identifier_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_identifier_values(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_identifier_values(item))
        return out
    normalized = _normalize_identifier(value)
    return [normalized] if normalized else []


def _identifier_records(display_name: str, identifiers: dict[str, Any] | None) -> set[tuple[str, str]]:
    records: set[tuple[str, str]] = set()
    display = _normalize_identifier(display_name)
    if display:
        records.add(('display_name', display))
    for key, value in (identifiers or {}).items():
        if key not in _ENTITY_USER_ID_KEYS and key not in _ENTITY_ALIAS_KEYS:
            continue
        kind = 'user_id' if key in _ENTITY_USER_ID_KEYS else key
        records.update((kind, normalized) for normalized in _flatten_identifier_values(value))
    return records


@dataclass
class WeChatRepository:
    store: SQLiteStore

    def replace_fixture(self, accounts: Iterable[Account], conversations: Iterable[Conversation], messages: Iterable[Message]) -> int:
        self.store.upsert_accounts(accounts)
        self.store.upsert_conversations(conversations)
        return self.store.upsert_messages(messages)

    def apply_delta(
        self,
        accounts: Iterable[Account],
        conversations: Iterable[Conversation],
        messages: Iterable[Message],
        *,
        deleted_citations: Iterable[str] = (),
        source_key: str | None = None,
        source_snapshot_complete: bool = False,
        max_chars: int = 900,
        overlap_chars: int = 120,
    ) -> dict[str, Any]:
        return self.store.apply_message_delta(
            accounts,
            conversations,
            messages,
            deleted_citations=deleted_citations,
            source_key=source_key,
            source_snapshot_complete=source_snapshot_complete,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )


@dataclass(frozen=True)
class MediaAssetRecord:
    asset_id: str
    account_id: str
    source_type: str
    source_id: str
    modality: str
    media_type: str
    citation: str
    local_type: str | None = None
    content_hash: str | None = None
    path_ref: str | None = None
    cache_state: str = 'unknown'
    processing_state: str = 'pending'
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MediaAssetLinkRecord:
    link_id: str
    asset_id: str
    account_id: str
    source_type: str
    source_citation: str
    scope_type: str
    accepted: bool
    reason: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderJobRecord:
    job_id: str
    provider: str
    model: str
    job_type: str
    status: str
    asset_id: str | None = None
    retry_count: int = 0
    usage: dict[str, Any] | None = None
    cost_rmb: float = 0.0
    request_hash: str | None = None
    error_code: str | None = None
    citation: str | None = None


@dataclass(frozen=True)
class TranscriptRecord:
    transcript_id: str
    asset_id: str
    citation: str
    text: str
    job_id: str | None = None
    language: str | None = None
    confidence: float = 0.0
    duration_seconds: float = 0.0
    status: str = 'active'


@dataclass(frozen=True)
class ImageObservationRecord:
    observation_id: str
    asset_id: str
    citation: str
    caption: str
    job_id: str | None = None
    visible_text: str = ''
    objects: list[dict[str, Any]] | None = None
    business_signals: list[dict[str, Any]] | None = None
    confidence: float = 0.0
    status: str = 'proposed'
    content_sha256: str = ''
    model_id: str = ''
    prompt_version: str = ''


@dataclass(frozen=True)
class MediaUnderstandingRecord:
    content_sha256: str
    modality: str
    model_id: str
    prompt_version: str
    caption: str | None = None
    visible_text: str | None = None
    objects: list[dict[str, Any]] | None = None
    business_signals: list[dict[str, Any]] | None = None
    keyframes: list[dict[str, Any]] | None = None
    audio_transcript: str | None = None
    confidence: float | None = None
    origin: str = 'lazy_agent'
    status: str = 'active'
    source_citations: list[str] | None = None
    metadata: dict[str, Any] | None = None
    replace: bool = False


@dataclass(frozen=True)
class SnsCacheMappingRecord:
    mapping_id: str
    account_id: str
    cache_key: str
    moment_id: str
    source_citation: str
    media_idx: int | None
    path_ref: str | None
    mapping_source: str
    confidence: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    display_name: str
    identifiers: dict[str, Any] | None = None
    status: str = 'active'
    confidence: float = 1.0


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    entity_id: str
    observation_type: str
    value: dict[str, Any]
    status: str
    confidence: float
    citation: str
    source_type: str
    valid_from: str | None = None
    supersedes_observation_id: str | None = None


@dataclass(frozen=True)
class RelationshipRecord:
    relationship_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str | None = None
    object_ref: str | None = None
    citation: str | None = None
    confidence: float = 0.0
    status: str = 'active'
    metadata: dict[str, Any] | None = None


class MultimodalRepository:
    OBSERVATION_STATUSES = {'proposed', 'active', 'superseded', 'rejected', 'merge_candidate', 'merged', 'needs_review'}
    JOB_STATUSES = {'pending', 'running', 'completed', 'retryable_failure', 'terminal_failure', 'needs_review', 'skipped'}

    @staticmethod
    def _upsert_identifier_rows(
        conn: Any,
        *,
        entity_id: str,
        records: Iterable[tuple[str, str]],
        source: str,
        confidence: float,
        citation: str | None,
        timestamp: str,
    ) -> None:
        for kind, normalized in records:
            if not normalized:
                continue
            conn.execute(
                """INSERT INTO entity_identifiers(
                       entity_id,identifier_type,normalized_value,source,confidence,citation,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(entity_id,identifier_type,normalized_value,source) DO UPDATE SET
                       confidence=MAX(entity_identifiers.confidence,excluded.confidence),
                       citation=COALESCE(excluded.citation,entity_identifiers.citation),
                       updated_at=excluded.updated_at""",
                (entity_id, kind, normalized, source, min(max(float(confidence or 0), 0.0), 1.0), citation, timestamp, timestamp),
            )

    def _refresh_observation_identifiers(self, conn: Any, entity_id: str, *, timestamp: str) -> None:
        conn.execute("DELETE FROM entity_identifiers WHERE entity_id=? AND source='observation'", (entity_id,))
        recognized = tuple(sorted(_ENTITY_USER_ID_KEYS | _ENTITY_ALIAS_KEYS))
        placeholders = ','.join('?' for _ in recognized)
        for row in conn.execute(
            f"""SELECT observation_type,value_json,confidence,citation
                  FROM observations
                 WHERE entity_id=?
                   AND status IN ('active','needs_review','merge_candidate')
                   AND lower(observation_type) IN ({placeholders})""",
            (entity_id, *tuple(value.lower() for value in recognized)),
        ):
            kind = 'user_id' if str(row['observation_type']).lower() in _ENTITY_USER_ID_KEYS else str(row['observation_type']).lower()
            records = [(kind, value) for value in _flatten_identifier_values(_load_json_dict(row['value_json']))]
            self._upsert_identifier_rows(
                conn,
                entity_id=entity_id,
                records=records,
                source='observation',
                confidence=float(row['confidence'] or 0),
                citation=row['citation'],
                timestamp=timestamp,
            )

    def __init__(self, store: SQLiteStore):
        self.store = store
        self.store.initialize()

    def upsert_media_asset(self, asset: MediaAssetRecord) -> dict[str, Any]:
        ts = _now()
        with self.store.connect() as conn:
            existing = None
            if asset.content_hash:
                # Content-identity dedup is intentionally gated on a real
                # content hash.  Distinct image/message citations can share a
                # source_id while the bytes are still unavailable; null hashes
                # must not collapse those citation-level assets.
                existing = conn.execute(
                    """SELECT asset_id FROM media_assets
                       WHERE account_id=?
                         AND source_type=?
                         AND source_id=?
                         AND modality=?
                         AND media_type=?
                         AND content_hash=?""",
                    (asset.account_id, asset.source_type, asset.source_id, asset.modality, asset.media_type, asset.content_hash),
                ).fetchone()
            asset_id = existing['asset_id'] if existing else asset.asset_id
            cursor = conn.execute(
                """INSERT INTO media_assets(asset_id,account_id,source_type,source_id,modality,media_type,local_type,citation,content_hash,path_ref,cache_state,processing_state,metadata_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(asset_id) DO UPDATE SET
                   account_id=excluded.account_id,source_type=excluded.source_type,source_id=excluded.source_id,modality=excluded.modality,
                   media_type=excluded.media_type,local_type=excluded.local_type,citation=excluded.citation,content_hash=excluded.content_hash,
                   path_ref=excluded.path_ref,cache_state=excluded.cache_state,processing_state=excluded.processing_state,
                   metadata_json=excluded.metadata_json,updated_at=excluded.updated_at
                   WHERE media_assets.account_id IS NOT excluded.account_id
                      OR media_assets.source_type IS NOT excluded.source_type
                      OR media_assets.source_id IS NOT excluded.source_id
                      OR media_assets.modality IS NOT excluded.modality
                      OR media_assets.media_type IS NOT excluded.media_type
                      OR media_assets.local_type IS NOT excluded.local_type
                      OR media_assets.citation IS NOT excluded.citation
                      OR media_assets.content_hash IS NOT excluded.content_hash
                      OR media_assets.path_ref IS NOT excluded.path_ref
                      OR media_assets.cache_state IS NOT excluded.cache_state
                      OR media_assets.processing_state IS NOT excluded.processing_state
                      OR media_assets.metadata_json IS NOT excluded.metadata_json""",
                (asset_id, asset.account_id, asset.source_type, asset.source_id, asset.modality, asset.media_type, asset.local_type,
                 asset.citation, asset.content_hash, asset.path_ref, asset.cache_state, asset.processing_state, _json(asset.metadata or {}), ts, ts),
            )
            if max(cursor.rowcount, 0):
                conn.commit()
            else:
                conn.rollback()
            row = conn.execute('SELECT * FROM media_assets WHERE asset_id=?', (asset_id,)).fetchone()
            return dict(row)

    def upsert_media_asset_link(self, link: MediaAssetLinkRecord) -> dict[str, Any]:
        with self.store.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO media_asset_links(link_id,asset_id,account_id,source_type,source_citation,scope_type,accepted,reason,metadata_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(link_id) DO UPDATE SET
                   asset_id=excluded.asset_id,source_type=excluded.source_type,source_citation=excluded.source_citation,
                   scope_type=excluded.scope_type,accepted=excluded.accepted,reason=excluded.reason,metadata_json=excluded.metadata_json
                   WHERE media_asset_links.asset_id IS NOT excluded.asset_id
                      OR media_asset_links.source_type IS NOT excluded.source_type
                      OR media_asset_links.source_citation IS NOT excluded.source_citation
                      OR media_asset_links.scope_type IS NOT excluded.scope_type
                      OR media_asset_links.accepted IS NOT excluded.accepted
                      OR media_asset_links.reason IS NOT excluded.reason
                      OR media_asset_links.metadata_json IS NOT excluded.metadata_json""",
                (link.link_id, link.asset_id, link.account_id, link.source_type, link.source_citation, link.scope_type,
                 1 if link.accepted else 0, link.reason, _json(link.metadata or {}), _now()),
            )
            if max(cursor.rowcount, 0):
                conn.commit()
            else:
                conn.rollback()
            return dict(conn.execute('SELECT * FROM media_asset_links WHERE link_id=?', (link.link_id,)).fetchone())

    def upsert_media_graph(
        self,
        assets: Iterable[MediaAssetRecord],
        links: Iterable[MediaAssetLinkRecord],
        *,
        source_states: Iterable[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Bulk-upsert assets, links and discovery watermarks in one commit.

        Every conflict update has an explicit difference predicate, so an
        identical discovery pass performs no durable write and publishes no
        SQLite generation.  ``changed_asset_ids`` is the set-based queue input.
        """
        asset_rows = list({item.asset_id: item for item in assets}.values())
        link_rows = list({item.link_id: item for item in links}.values())
        state_rows = list({str(item.get('source_key') or ''): dict(item) for item in source_states if item.get('source_key')}.values())
        wal_path = self.store.path.with_name(self.store.path.name + '-wal')
        wal_before = wal_path.stat().st_size if wal_path.exists() else 0
        changed_asset_ids: set[str] = set()
        asset_id_map: dict[str, str] = {}
        assets_changed = links_changed = states_changed = media_jobs_changed = 0
        source_rows_changed = source_rows_deleted = 0
        stale_assets_changed = stale_links_changed = 0
        stale_asset_ids: set[str] = set()
        sql_statements = 0
        ts = _now()
        with self.store.connect() as conn:
            existing_assets: dict[str, dict[str, Any]] = {}
            requested_asset_ids = [item.asset_id for item in asset_rows]
            for start in range(0, len(requested_asset_ids), 500):
                batch = requested_asset_ids[start:start + 500]
                if not batch:
                    continue
                placeholders = ','.join('?' for _ in batch)
                existing_assets.update({
                    str(row['asset_id']): dict(row)
                    for row in conn.execute(
                        f'SELECT * FROM media_assets WHERE asset_id IN ({placeholders})',
                        batch,
                    )
                })
            asset_values: list[tuple[Any, ...]] = []
            persisted_asset_ids: list[str] = []
            for asset in asset_rows:
                persisted_id = asset.asset_id
                if asset.content_hash:
                    sql_statements += 1
                    existing = conn.execute(
                        """SELECT asset_id FROM media_assets
                           WHERE account_id=? AND source_type=? AND source_id=?
                             AND modality=? AND media_type=? AND content_hash=?""",
                        (asset.account_id, asset.source_type, asset.source_id, asset.modality, asset.media_type, asset.content_hash),
                    ).fetchone()
                    if existing is not None:
                        persisted_id = str(existing['asset_id'])
                existing_asset = existing_assets.get(persisted_id)
                asset_path_ref = asset.path_ref
                asset_content_hash = asset.content_hash
                asset_cache_state = asset.cache_state
                asset_processing_state = asset.processing_state
                asset_metadata = dict(asset.metadata or {})
                if existing_asset is not None and asset_metadata.get('registration') == 'message_media':
                    # A deterministic metadata-only placeholder must never
                    # downgrade bytes or processing already discovered for the
                    # same message asset.
                    asset_path_ref = asset_path_ref or existing_asset.get('path_ref')
                    asset_content_hash = asset_content_hash or existing_asset.get('content_hash')
                    cache_rank = {
                        'unknown': 0,
                        'inventory_only': 0,
                        'metadata_only': 1,
                        'missing_local_cache': 2,
                        'source_available': 3,
                        'cached': 3,
                        'copied': 4,
                        'normalized': 5,
                    }
                    existing_cache = str(existing_asset.get('cache_state') or 'unknown')
                    if cache_rank.get(existing_cache, 0) > cache_rank.get(asset_cache_state, 0):
                        asset_cache_state = existing_cache
                    existing_processing = str(existing_asset.get('processing_state') or 'pending')
                    if existing_processing not in {'pending', 'metadata_only'}:
                        asset_processing_state = existing_processing
                    try:
                        previous_metadata = _load_json_dict(existing_asset.get('metadata_json'))
                    except Exception:
                        previous_metadata = {}
                    asset_metadata = previous_metadata | asset_metadata
                asset_id_map[asset.asset_id] = persisted_id
                persisted_asset_ids.append(persisted_id)
                asset_values.append((
                    persisted_id, asset.account_id, asset.source_type, asset.source_id, asset.modality,
                    asset.media_type, asset.local_type, asset.citation, asset_content_hash, asset_path_ref,
                    asset_cache_state, asset_processing_state, _json(asset_metadata), ts, ts,
                ))
            if asset_values:
                sql_statements += 1
                cursor = conn.executemany(
                    """INSERT INTO media_assets(asset_id,account_id,source_type,source_id,modality,media_type,local_type,citation,content_hash,path_ref,cache_state,processing_state,metadata_json,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(asset_id) DO UPDATE SET
                         account_id=excluded.account_id,source_type=excluded.source_type,source_id=excluded.source_id,
                         modality=excluded.modality,media_type=excluded.media_type,local_type=excluded.local_type,
                         citation=excluded.citation,content_hash=excluded.content_hash,path_ref=excluded.path_ref,
                         cache_state=excluded.cache_state,processing_state=excluded.processing_state,
                         metadata_json=excluded.metadata_json,updated_at=excluded.updated_at
                       WHERE media_assets.account_id IS NOT excluded.account_id
                          OR media_assets.source_type IS NOT excluded.source_type
                          OR media_assets.source_id IS NOT excluded.source_id
                          OR media_assets.modality IS NOT excluded.modality
                          OR media_assets.media_type IS NOT excluded.media_type
                          OR media_assets.local_type IS NOT excluded.local_type
                          OR media_assets.citation IS NOT excluded.citation
                          OR media_assets.content_hash IS NOT excluded.content_hash
                          OR media_assets.path_ref IS NOT excluded.path_ref
                          OR media_assets.cache_state IS NOT excluded.cache_state
                          OR media_assets.processing_state IS NOT excluded.processing_state
                          OR media_assets.metadata_json IS NOT excluded.metadata_json""",
                    asset_values,
                )
                assets_changed = max(cursor.rowcount, 0)
                if assets_changed:
                    changed_asset_ids.update(persisted_asset_ids)

            link_values: list[tuple[Any, ...]] = []
            linked_asset_ids: list[str] = []
            for link in link_rows:
                persisted_id = asset_id_map.get(link.asset_id, link.asset_id)
                linked_asset_ids.append(persisted_id)
                link_values.append((
                    link.link_id, persisted_id, link.account_id, link.source_type, link.source_citation,
                    link.scope_type, 1 if link.accepted else 0, link.reason, _json(link.metadata or {}), ts,
                ))
            if link_values:
                sql_statements += 1
                cursor = conn.executemany(
                    """INSERT INTO media_asset_links(link_id,asset_id,account_id,source_type,source_citation,scope_type,accepted,reason,metadata_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(link_id) DO UPDATE SET
                         asset_id=excluded.asset_id,account_id=excluded.account_id,source_type=excluded.source_type,
                         source_citation=excluded.source_citation,scope_type=excluded.scope_type,
                         accepted=excluded.accepted,reason=excluded.reason,metadata_json=excluded.metadata_json
                       WHERE media_asset_links.asset_id IS NOT excluded.asset_id
                          OR media_asset_links.account_id IS NOT excluded.account_id
                          OR media_asset_links.source_type IS NOT excluded.source_type
                          OR media_asset_links.source_citation IS NOT excluded.source_citation
                          OR media_asset_links.scope_type IS NOT excluded.scope_type
                          OR media_asset_links.accepted IS NOT excluded.accepted
                          OR media_asset_links.reason IS NOT excluded.reason
                          OR media_asset_links.metadata_json IS NOT excluded.metadata_json""",
                    link_values,
                )
                links_changed = max(cursor.rowcount, 0)
                if links_changed:
                    changed_asset_ids.update(linked_asset_ids)

            source_row_values: list[tuple[Any, ...]] = []
            state_values: list[tuple[Any, ...]] = []
            for state in state_rows:
                for row in state.get('row_updates') or ():
                    source_row_values.append((
                        state['source_key'], int(row.get('row_id') or 0),
                        str(row.get('row_fingerprint') or ''), str(row.get('asset_id') or ''),
                        str(row.get('citation') or ''), ts,
                    ))
                deleted_row_ids = [int(value) for value in (state.get('deleted_row_ids') or ())]
                stale_asset_ids.update(str(value) for value in (state.get('stale_asset_ids') or ()) if value)
                for start in range(0, len(deleted_row_ids), 500):
                    batch = deleted_row_ids[start:start + 500]
                    placeholders = ','.join('?' for _ in batch)
                    sql_statements += 1
                    cursor = conn.execute(
                        f'DELETE FROM media_source_rows WHERE source_key=? AND row_id IN ({placeholders})',
                        [state['source_key'], *batch],
                    )
                    source_rows_deleted += max(cursor.rowcount, 0)
                state_values.append((
                    state['source_key'], str(state.get('file_fingerprint') or ''),
                    str(state.get('table_fingerprint') or ''), int(state.get('row_watermark') or 0),
                    int(state.get('row_count') or 0), ts,
                ))
            if source_row_values:
                sql_statements += 1
                cursor = conn.executemany(
                    """INSERT INTO media_source_rows(source_key,row_id,row_fingerprint,asset_id,citation,updated_at)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(source_key,row_id) DO UPDATE SET
                         row_fingerprint=excluded.row_fingerprint,
                         asset_id=excluded.asset_id,
                         citation=excluded.citation,
                         updated_at=excluded.updated_at
                       WHERE media_source_rows.row_fingerprint IS NOT excluded.row_fingerprint
                          OR media_source_rows.asset_id IS NOT excluded.asset_id
                          OR media_source_rows.citation IS NOT excluded.citation""",
                    source_row_values,
                )
                source_rows_changed = max(cursor.rowcount, 0)
            if state_values:
                sql_statements += 1
                cursor = conn.executemany(
                    """INSERT INTO media_source_state(source_key,file_fingerprint,table_fingerprint,row_watermark,row_count,updated_at)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(source_key) DO UPDATE SET
                         file_fingerprint=excluded.file_fingerprint,
                         table_fingerprint=excluded.table_fingerprint,
                         row_watermark=excluded.row_watermark,
                         row_count=excluded.row_count,
                         updated_at=excluded.updated_at
                       WHERE media_source_state.file_fingerprint IS NOT excluded.file_fingerprint
                          OR media_source_state.table_fingerprint IS NOT excluded.table_fingerprint
                          OR media_source_state.row_watermark IS NOT excluded.row_watermark
                          OR media_source_state.row_count IS NOT excluded.row_count""",
                    state_values,
                )
                states_changed = max(cursor.rowcount, 0)

            for asset_id in sorted(stale_asset_ids):
                # Keep evidence identity for audit, but make a deleted source
                # ineligible for future work once no discovery row references it.
                sql_statements += 2
                cursor = conn.execute(
                    """UPDATE media_asset_links
                       SET accepted=0,reason='source_deleted'
                       WHERE asset_id=?
                         AND NOT EXISTS (SELECT 1 FROM media_source_rows r WHERE r.asset_id=?)
                         AND (accepted<>0 OR reason<>'source_deleted')""",
                    (asset_id, asset_id),
                )
                stale_links_changed += max(cursor.rowcount, 0)
                cursor = conn.execute(
                    """UPDATE media_assets SET processing_state='stale',updated_at=?
                       WHERE asset_id=?
                         AND NOT EXISTS (SELECT 1 FROM media_source_rows r WHERE r.asset_id=?)
                         AND processing_state<>'stale'""",
                    (ts, asset_id, asset_id),
                )
                count = max(cursor.rowcount, 0)
                stale_assets_changed += count
                if count:
                    changed_asset_ids.add(asset_id)

            # Publish durable voice work in the same transaction as the media
            # graph.  A crash can therefore never leave a changed private-chat
            # asset committed without its queue record.  Exact accepted links
            # are authoritative here; legacy fuzzy/JSON recovery stays in the
            # explicit media repair path.
            changed_ids = sorted(changed_asset_ids)
            for start in range(0, len(changed_ids), 500):
                batch = changed_ids[start:start + 500]
                placeholders = ','.join('?' for _ in batch)
                eligible = f"""asset_id IN ({placeholders})
                    AND modality='voice'
                    AND processing_state<>'stale'
                    AND EXISTS (
                        SELECT 1 FROM media_asset_links l
                        WHERE l.asset_id=media_assets.asset_id
                          AND l.accepted=1 AND l.scope_type='private_chat'
                    )"""
                sql_statements += 3
                cursor = conn.execute(
                    f"""UPDATE media_jobs
                        SET status='pending',error_code=NULL,updated_at=?
                        WHERE job_type='voice_transcribe'
                          AND status='skipped' AND error_code='out_of_scope'
                          AND asset_id IN (SELECT asset_id FROM media_assets WHERE {eligible})""",
                    [ts, *batch],
                )
                media_jobs_changed += max(cursor.rowcount, 0)
                cursor = conn.execute(
                    f"""UPDATE media_jobs
                        SET status='skipped',error_code='out_of_scope',updated_at=?
                        WHERE job_type='voice_transcribe'
                          AND asset_id IN ({placeholders})
                          AND (status<>'skipped' OR COALESCE(error_code,'')<>'out_of_scope')
                          AND NOT EXISTS (
                              SELECT 1 FROM media_assets ma
                              WHERE ma.asset_id=media_jobs.asset_id
                                AND ma.modality='voice' AND ma.processing_state<>'stale'
                                AND EXISTS (
                                    SELECT 1 FROM media_asset_links l
                                    WHERE l.asset_id=ma.asset_id
                                      AND l.accepted=1 AND l.scope_type='private_chat'
                                )
                          )""",
                    [ts, *batch],
                )
                media_jobs_changed += max(cursor.rowcount, 0)
                cursor = conn.execute(
                    f"""INSERT OR IGNORE INTO media_jobs(
                            job_id,asset_id,job_type,status,retry_count,error_code,
                            last_duration_ms,created_at,updated_at
                        )
                        SELECT 'mediajob:voice_transcribe:' || asset_id,
                               asset_id,'voice_transcribe','pending',0,NULL,0.0,?,?
                        FROM media_assets
                        WHERE {eligible}""",
                    [ts, ts, *batch],
                )
                media_jobs_changed += max(cursor.rowcount, 0)

            durable = (
                assets_changed + links_changed + states_changed
                + source_rows_changed + source_rows_deleted
                + stale_assets_changed + stale_links_changed + media_jobs_changed
            )
            if durable:
                conn.commit()
                commits = 1
            else:
                conn.rollback()
                commits = 0
        wal_after = wal_path.stat().st_size if wal_path.exists() else 0
        return {
            'assets_seen': len(asset_rows),
            'assets_upserted': assets_changed,
            'links_seen': len(link_rows),
            'links_upserted': links_changed,
            'states_updated': states_changed,
            'source_rows_updated': source_rows_changed,
            'source_rows_deleted': source_rows_deleted,
            'stale_assets_updated': stale_assets_changed,
            'stale_links_updated': stale_links_changed,
            'media_jobs_updated': media_jobs_changed,
            'changed_asset_ids': sorted(changed_asset_ids),
            'asset_id_map': asset_id_map,
            'metrics': {
                'sql_statements': sql_statements,
                'commits': commits,
                'rows_scanned': len(asset_rows) + len(link_rows) + len(state_rows),
                'candidate_rows': len(asset_rows) + len(link_rows),
                'rows_written': durable,
                'wal_bytes': max(0, wal_after - wal_before),
            },
        }

    def record_decode_result(self, *, decode_id: str, asset_id: str, status: str, wrapper_type: str | None = None, input_hash: str | None = None, output_hash: str | None = None, derivative_ref: str | None = None, error_code: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO media_decode_results(decode_id,asset_id,status,wrapper_type,input_hash,output_hash,derivative_ref,error_code,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (decode_id, asset_id, status, wrapper_type, input_hash, output_hash, derivative_ref, error_code, _json(metadata or {}), _now()),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM media_decode_results WHERE decode_id=?', (decode_id,)).fetchone())

    def record_provider_job(self, job: ProviderJobRecord) -> dict[str, Any]:
        if job.status not in self.JOB_STATUSES:
            raise ValueError(f'invalid provider job status: {job.status}')
        ts = _now()
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO provider_jobs(job_id,asset_id,provider,model,job_type,status,retry_count,usage_json,cost_rmb,request_hash,error_code,citation,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,retry_count=excluded.retry_count,usage_json=excluded.usage_json,
                   cost_rmb=excluded.cost_rmb,error_code=excluded.error_code,citation=excluded.citation,updated_at=excluded.updated_at""",
                (job.job_id, job.asset_id, job.provider, job.model, job.job_type, job.status, job.retry_count, _json(job.usage or {}),
                 float(job.cost_rmb), job.request_hash, job.error_code, job.citation, ts, ts),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM provider_jobs WHERE job_id=?', (job.job_id,)).fetchone())

    def insert_transcript(self, transcript: TranscriptRecord) -> dict[str, Any]:
        if not transcript.citation:
            raise ValueError('transcript citation is required')
        with self.store.connect() as conn:
            previous_citations = [
                str(row['citation']) for row in conn.execute(
                    "SELECT citation FROM transcripts WHERE asset_id=? AND status='active' AND transcript_id<>?",
                    (transcript.asset_id, transcript.transcript_id),
                )
            ]
            conn.execute(
                "UPDATE transcripts SET status='superseded' WHERE asset_id=? AND status='active' AND transcript_id<>?",
                (transcript.asset_id, transcript.transcript_id),
            )
            conn.execute(
                'INSERT OR REPLACE INTO transcripts(transcript_id,asset_id,job_id,citation,text,language,confidence,duration_seconds,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (transcript.transcript_id, transcript.asset_id, transcript.job_id, transcript.citation, transcript.text, transcript.language,
                 transcript.confidence, transcript.duration_seconds, transcript.status, _now()),
            )
            conn.commit()
            row = dict(conn.execute('SELECT * FROM transcripts WHERE transcript_id=?', (transcript.transcript_id,)).fetchone())
        self.store.upsert_evidence_chunks_for_source_citations(
            'transcript', list(dict.fromkeys([*previous_citations, transcript.citation])),
        )
        return row

    def insert_image_observation(self, observation: ImageObservationRecord) -> dict[str, Any]:
        now = _now()
        with self.store.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO image_observations(observation_id,asset_id,job_id,citation,caption,visible_text,objects_json,business_signals_json,content_sha256,model_id,prompt_version,confidence,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (observation.observation_id, observation.asset_id, observation.job_id, observation.citation, observation.caption,
                 observation.visible_text, _json(observation.objects or []), _json(observation.business_signals or []),
                 observation.content_sha256, observation.model_id, observation.prompt_version, observation.confidence,
                 observation.status, now, now),
            )
            conn.commit()
            row = dict(conn.execute('SELECT * FROM image_observations WHERE observation_id=?', (observation.observation_id,)).fetchone())
        self.store.upsert_evidence_chunks_for_source_citations('image_observation', [observation.citation])
        return row

    def upsert_sns_cache_mapping(
        self,
        *,
        mapping_id: str,
        account_id: str,
        cache_key: str,
        moment_id: str,
        source_citation: str,
        media_idx: int | None,
        path_ref: str | None,
        mapping_source: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO sns_cache_mappings(mapping_id,account_id,cache_key,moment_id,source_citation,media_idx,path_ref,mapping_source,confidence,metadata_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(account_id, cache_key, source_citation) DO UPDATE SET
                   moment_id=excluded.moment_id,media_idx=excluded.media_idx,path_ref=excluded.path_ref,
                   mapping_source=excluded.mapping_source,confidence=excluded.confidence,metadata_json=excluded.metadata_json""",
                (mapping_id, account_id, cache_key, moment_id, source_citation, media_idx, path_ref,
                 mapping_source, float(confidence), _json(metadata or {}), _now()),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM sns_cache_mappings WHERE account_id=? AND cache_key=? AND source_citation=?', (account_id, cache_key, source_citation)).fetchone())

    def upsert_sns_cache_mappings(self, mappings: Iterable[SnsCacheMappingRecord]) -> int:
        records = list(mappings)
        if not records:
            return 0
        now = _now()
        with self.store.connect() as conn:
            conn.executemany(
                """INSERT INTO sns_cache_mappings(mapping_id,account_id,cache_key,moment_id,source_citation,media_idx,path_ref,mapping_source,confidence,metadata_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(account_id, cache_key, source_citation) DO UPDATE SET
                   moment_id=excluded.moment_id,media_idx=excluded.media_idx,path_ref=excluded.path_ref,
                   mapping_source=excluded.mapping_source,confidence=excluded.confidence,metadata_json=excluded.metadata_json""",
                [
                    (
                        m.mapping_id, m.account_id, m.cache_key, m.moment_id, m.source_citation, m.media_idx,
                        m.path_ref, m.mapping_source, float(m.confidence), _json(m.metadata or {}), now,
                    )
                    for m in records
                ],
            )
            conn.commit()
        return len(records)

    @staticmethod
    def _merge_understanding_text(value: str | None, existing: Any, *, replace: bool) -> str:
        if value is None:
            return str(existing or '') if existing is not None else ''
        text = str(value)
        if text or replace:
            return text
        return str(existing or '') if existing is not None else ''

    @staticmethod
    def _merge_understanding_list(value: list[dict[str, Any]] | None, existing_json: Any, *, replace: bool) -> list[Any]:
        if value is None:
            return _load_json_list(existing_json) if existing_json is not None else []
        if value or replace:
            return value
        return _load_json_list(existing_json) if existing_json is not None else []

    @staticmethod
    def _merge_understanding_confidence(value: float | None, existing: Any, *, replace: bool) -> float:
        if value is None:
            return float(existing or 0) if existing is not None else 0.0
        numeric = float(value or 0)
        if numeric > 0 or replace:
            return numeric
        return float(existing or 0) if existing is not None else 0.0

    def upsert_media_understanding(self, record: MediaUnderstandingRecord) -> dict[str, Any]:
        content_sha256 = str(record.content_sha256 or '').lower()
        if len(content_sha256) != 64 or any(c not in '0123456789abcdef' for c in content_sha256):
            raise ValueError('content_sha256 must be a 64-character hex sha256')
        if record.modality not in {'image', 'video'}:
            raise ValueError('media understanding modality must be image or video')
        if not str(record.model_id or '').strip():
            raise ValueError('model_id is required')
        if not str(record.prompt_version or '').strip():
            raise ValueError('prompt_version is required')
        ts = _now()
        new_citations = [str(c) for c in (record.source_citations or []) if str(c or '').strip()]
        with self.store.connect() as conn:
            existing = conn.execute('SELECT * FROM media_understanding WHERE content_sha256=?', (content_sha256,)).fetchone()
            metadata = _load_json_dict(existing['metadata_json']) if existing is not None else {}
            merged_metadata = dict(metadata)
            for key, value in (record.metadata or {}).items():
                if key != 'history':
                    merged_metadata[key] = value
            existing_citations = _load_json_list(existing['source_citations_json']) if existing is not None else []
            source_citations = list(dict.fromkeys([str(c) for c in existing_citations if str(c or '').strip()] + new_citations))
            replace = bool(record.replace)
            caption = self._merge_understanding_text(record.caption, existing['caption'] if existing is not None else None, replace=replace)
            visible_text = self._merge_understanding_text(record.visible_text, existing['visible_text'] if existing is not None else None, replace=replace)
            objects = self._merge_understanding_list(record.objects, existing['objects_json'] if existing is not None else None, replace=replace)
            business_signals = self._merge_understanding_list(record.business_signals, existing['business_signals_json'] if existing is not None else None, replace=replace)
            keyframes = self._merge_understanding_list(record.keyframes, existing['keyframes_json'] if existing is not None and 'keyframes_json' in existing.keys() else None, replace=replace)
            audio_transcript = self._merge_understanding_text(record.audio_transcript, existing['audio_transcript'] if existing is not None and 'audio_transcript' in existing.keys() else None, replace=replace)
            confidence = self._merge_understanding_confidence(record.confidence, existing['confidence'] if existing is not None else None, replace=replace)
            if existing is not None:
                old_snapshot = {
                    'model_id': existing['model_id'],
                    'prompt_version': existing['prompt_version'],
                    'caption': existing['caption'],
                    'visible_text': existing['visible_text'],
                    'objects': _load_json_list(existing['objects_json']),
                    'business_signals': _load_json_list(existing['business_signals_json']),
                    'keyframes': _load_json_list(existing['keyframes_json']) if 'keyframes_json' in existing.keys() else [],
                    'audio_transcript': existing['audio_transcript'] if 'audio_transcript' in existing.keys() else '',
                    'confidence': float(existing['confidence'] or 0),
                    'origin': existing['origin'],
                    'status': existing['status'] if 'status' in existing.keys() else 'active',
                    'updated_at': existing['updated_at'],
                }
                new_snapshot = {
                    'model_id': record.model_id,
                    'prompt_version': record.prompt_version,
                    'caption': caption,
                    'visible_text': visible_text,
                    'objects': objects,
                    'business_signals': business_signals,
                    'keyframes': keyframes,
                    'audio_transcript': audio_transcript,
                    'confidence': confidence,
                    'origin': record.origin,
                    'status': record.status,
                }
                if {k: old_snapshot[k] for k in new_snapshot} != new_snapshot:
                    history = list(merged_metadata.get('history') or [])
                    history.append(old_snapshot)
                    merged_metadata['history'] = history[-20:]
            conn.execute(
                """INSERT INTO media_understanding(content_sha256,modality,caption,visible_text,objects_json,business_signals_json,
                   keyframes_json,audio_transcript,model_id,prompt_version,confidence,origin,status,source_citations_json,metadata_json,fetch_hit_count,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(content_sha256) DO UPDATE SET
                   modality=excluded.modality,caption=excluded.caption,visible_text=excluded.visible_text,
                   objects_json=excluded.objects_json,business_signals_json=excluded.business_signals_json,
                   keyframes_json=excluded.keyframes_json,audio_transcript=excluded.audio_transcript,
                   model_id=excluded.model_id,prompt_version=excluded.prompt_version,confidence=excluded.confidence,
                   origin=excluded.origin,status=excluded.status,source_citations_json=excluded.source_citations_json,
                   metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""",
                (
                    content_sha256, record.modality, caption, visible_text,
                    _json(objects), _json(business_signals),
                    _json(keyframes), audio_transcript, record.model_id, record.prompt_version,
                    confidence, record.origin, record.status, _json(source_citations),
                    _json(merged_metadata), int(existing['fetch_hit_count']) if existing is not None and 'fetch_hit_count' in existing.keys() else 0,
                    ts, ts,
                ),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM media_understanding WHERE content_sha256=?', (content_sha256,)).fetchone())

    def media_understanding_for_hash(self, content_sha256: str) -> dict[str, Any] | None:
        content_sha256 = str(content_sha256 or '').lower()
        if not content_sha256:
            return None
        with self.store.connect() as conn:
            row = conn.execute('SELECT * FROM media_understanding WHERE content_sha256=? AND status=?', (content_sha256, 'active')).fetchone()
            return dict(row) if row is not None else None

    def merge_image_observation(self, observation: ImageObservationRecord) -> dict[str, Any]:
        """Merge OCR/caption observations for one asset into a single searchable row."""
        with self.store.connect() as conn:
            existing = conn.execute('SELECT * FROM image_observations WHERE asset_id=? ORDER BY created_at, observation_id LIMIT 1', (observation.asset_id,)).fetchone()
        if existing is None:
            return self.insert_image_observation(observation)
        with self.store.connect() as conn:
            row = dict(existing)
            caption = observation.caption or row.get('caption') or ''
            visible_text = observation.visible_text or row.get('visible_text') or ''
            objects = _load_json_list(row.get('objects_json')) + (observation.objects or [])
            business_signals = _load_json_list(row.get('business_signals_json')) + (observation.business_signals or [])
            confidence = max(float(row.get('confidence') or 0.0), float(observation.confidence or 0.0))
            status = observation.status if observation.status == 'needs_review' or row.get('status') != 'active' else 'active'
            conn.execute(
                """UPDATE image_observations
                   SET job_id=COALESCE(?, job_id), caption=?, visible_text=?, objects_json=?, business_signals_json=?,
                       confidence=?, status=?
                   WHERE observation_id=?""",
                (observation.job_id, caption, visible_text, _json(objects), _json(business_signals), confidence, status, row['observation_id']),
            )
            conn.commit()
            merged = dict(conn.execute('SELECT * FROM image_observations WHERE observation_id=?', (row['observation_id'],)).fetchone())
        self.store.upsert_evidence_chunks_for_source_citations('image_observation', [merged['citation']])
        return merged

    def project_media_understanding(
        self,
        *,
        asset_id: str,
        citation: str,
        content_sha256: str,
        model_id: str,
        prompt_version: str,
        caption: str,
        visible_text: str,
        objects: list[dict[str, Any]],
        business_signals: list[dict[str, Any]],
        confidence: float,
    ) -> dict[str, Any]:
        """Project one versioned understanding into cited searchable evidence."""
        sha = str(content_sha256 or '').lower()
        if len(sha) != 64 or any(char not in '0123456789abcdef' for char in sha):
            raise ValueError('content_sha256 must be a 64-character hex sha256')
        if not str(asset_id or '').strip() or not str(citation or '').strip():
            raise ValueError('asset_id and citation are required')
        if not str(model_id or '').strip() or not str(prompt_version or '').strip():
            raise ValueError('model_id and prompt_version are required')
        identity = f'{asset_id}:{citation}:{sha}:{model_id}:{prompt_version}'
        observation_id = _stable('imageobs', identity)
        now = _now()
        affected: list[str] = [citation]
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            affected.extend(str(row['citation']) for row in conn.execute(
                """SELECT citation FROM image_observations
                     WHERE asset_id=? AND citation=? AND status='active' AND observation_id<>?""",
                (asset_id, citation, observation_id),
            ))
            conn.execute(
                """UPDATE image_observations SET status='superseded',updated_at=?
                     WHERE asset_id=? AND citation=? AND status='active' AND observation_id<>?""",
                (now, asset_id, citation, observation_id),
            )
            conn.execute(
                """INSERT INTO image_observations(
                       observation_id,asset_id,job_id,citation,caption,visible_text,objects_json,business_signals_json,
                       content_sha256,model_id,prompt_version,confidence,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(asset_id,citation,content_sha256,model_id,prompt_version) DO UPDATE SET
                       caption=excluded.caption,visible_text=excluded.visible_text,objects_json=excluded.objects_json,
                       business_signals_json=excluded.business_signals_json,confidence=excluded.confidence,
                       status='active',updated_at=excluded.updated_at""",
                (observation_id, asset_id, None, citation, caption, visible_text, _json(objects),
                 _json(business_signals), sha, model_id, prompt_version, float(confidence), 'active', now, now),
            )
            conn.commit()
            row = dict(conn.execute(
                """SELECT * FROM image_observations
                     WHERE asset_id=? AND citation=? AND content_sha256=? AND model_id=? AND prompt_version=?""",
                (asset_id, citation, sha, model_id, prompt_version),
            ).fetchone())
        self.store.upsert_evidence_chunks_for_source_citations(
            'image_observation', list(dict.fromkeys(affected)),
        )
        return row

    def update_image_caption(
        self,
        *,
        asset_id: str,
        citation: str,
        caption: str,
        labels: list[str] | None = None,
        confidence: float = 0.0,
        status: str = 'active',
        job_id: str | None = None,
    ) -> dict[str, Any]:
        labels = labels or []
        with self.store.connect() as conn:
            existing = conn.execute('SELECT * FROM image_observations WHERE asset_id=? ORDER BY created_at, observation_id LIMIT 1', (asset_id,)).fetchone()
            if existing is None:
                observation_id = _stable('imageobs', f'{asset_id}:local-vlm-caption')
                objects = [{'label': label} for label in labels]
                conn.execute(
                    'INSERT OR REPLACE INTO image_observations(observation_id,asset_id,job_id,citation,caption,visible_text,objects_json,business_signals_json,confidence,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (observation_id, asset_id, job_id, citation, caption, '', _json(objects), _json([]), confidence, status, _now()),
                )
                conn.commit()
                row = dict(conn.execute('SELECT * FROM image_observations WHERE observation_id=?', (observation_id,)).fetchone())
            else:
                row = dict(existing)
                objects = _load_json_list(row.get('objects_json'))
                seen = {str(item.get('label')) for item in objects if isinstance(item, dict) and item.get('label') is not None}
                for label in labels:
                    if label not in seen:
                        objects.append({'label': label})
                        seen.add(label)
                merged_status = status if status == 'needs_review' else (row.get('status') or status)
                if merged_status == 'proposed':
                    merged_status = status
                conn.execute(
                    """UPDATE image_observations
                       SET job_id=COALESCE(?, job_id), caption=?, objects_json=?, confidence=?, status=?
                       WHERE observation_id=?""",
                    (job_id, caption, _json(objects), max(float(row.get('confidence') or 0.0), float(confidence or 0.0)), merged_status, row['observation_id']),
                )
                conn.commit()
                row = dict(conn.execute('SELECT * FROM image_observations WHERE observation_id=?', (row['observation_id'],)).fetchone())
        self.store.upsert_evidence_chunks_for_source_citations('image_observation', [row['citation']])
        return row

    def upsert_entity(self, entity: EntityRecord) -> dict[str, Any]:
        ts = _now()
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO entities(entity_id,entity_type,display_name,identifiers_json,status,confidence,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(entity_id) DO UPDATE SET entity_type=excluded.entity_type,display_name=excluded.display_name,
                   identifiers_json=excluded.identifiers_json,status=excluded.status,confidence=excluded.confidence,updated_at=excluded.updated_at""",
                (entity.entity_id, entity.entity_type, entity.display_name, _json(entity.identifiers or {}), entity.status, entity.confidence, ts, ts),
            )
            conn.execute("DELETE FROM entity_identifiers WHERE entity_id=? AND source='entity'", (entity.entity_id,))
            self._upsert_identifier_rows(
                conn,
                entity_id=entity.entity_id,
                records=_identifier_records(entity.display_name, entity.identifiers),
                source='entity',
                confidence=entity.confidence,
                citation=None,
                timestamp=ts,
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM entities WHERE entity_id=?', (entity.entity_id,)).fetchone())

    def add_observation(self, observation: ObservationRecord) -> dict[str, Any]:
        if observation.status not in self.OBSERVATION_STATUSES:
            raise ValueError(f'invalid observation status: {observation.status}')
        if not observation.citation:
            raise ValueError('observation citation is required')
        ts = _now()
        with self.store.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO observations(observation_id,entity_id,observation_type,value_json,status,confidence,citation,source_type,valid_from,supersedes_observation_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (observation.observation_id, observation.entity_id, observation.observation_type, _json(observation.value), observation.status,
                 observation.confidence, observation.citation, observation.source_type, observation.valid_from, observation.supersedes_observation_id, ts, ts),
            )
            self._refresh_observation_identifiers(conn, observation.entity_id, timestamp=ts)
            conn.commit()
            return dict(conn.execute('SELECT * FROM observations WHERE observation_id=?', (observation.observation_id,)).fetchone())

    def upsert_contact_batch(
        self,
        entities: Iterable[EntityRecord],
        observations: Iterable[ObservationRecord],
    ) -> dict[str, int]:
        """Persist one contact snapshot without per-row commits or refreshes."""

        entity_rows = list(entities)
        observation_rows = list(observations)
        for observation in observation_rows:
            if observation.status not in self.OBSERVATION_STATUSES:
                raise ValueError(f'invalid observation status: {observation.status}')
            if not observation.citation:
                raise ValueError('observation citation is required')
        timestamp = _now()
        entity_ids = sorted({row.entity_id for row in entity_rows})
        with self.store.connect() as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                if entity_rows:
                    conn.executemany(
                        """INSERT INTO entities(entity_id,entity_type,display_name,identifiers_json,status,confidence,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?)
                           ON CONFLICT(entity_id) DO UPDATE SET entity_type=excluded.entity_type,display_name=excluded.display_name,
                           identifiers_json=excluded.identifiers_json,status=excluded.status,confidence=excluded.confidence,updated_at=excluded.updated_at""",
                        [
                            (
                                row.entity_id, row.entity_type, row.display_name, _json(row.identifiers or {}),
                                row.status, row.confidence, timestamp, timestamp,
                            )
                            for row in entity_rows
                        ],
                    )
                if entity_ids:
                    conn.executemany(
                        "DELETE FROM entity_identifiers WHERE entity_id=? AND source IN ('entity','observation')",
                        [(entity_id,) for entity_id in entity_ids],
                    )
                identifier_values: list[tuple[Any, ...]] = []
                for row in entity_rows:
                    identifier_values.extend((
                        row.entity_id, kind, normalized, 'entity',
                        min(max(float(row.confidence or 0), 0.0), 1.0), None, timestamp, timestamp,
                    ) for kind, normalized in _identifier_records(row.display_name, row.identifiers))
                if observation_rows:
                    conn.executemany(
                        'INSERT OR REPLACE INTO observations(observation_id,entity_id,observation_type,value_json,status,confidence,citation,source_type,valid_from,supersedes_observation_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                        [
                            (
                                row.observation_id, row.entity_id, row.observation_type, _json(row.value), row.status,
                                row.confidence, row.citation, row.source_type, row.valid_from,
                                row.supersedes_observation_id, timestamp, timestamp,
                            )
                            for row in observation_rows
                        ],
                    )
                recognized = tuple(sorted(value.lower() for value in (_ENTITY_USER_ID_KEYS | _ENTITY_ALIAS_KEYS)))
                recognized_placeholders = ','.join('?' for _ in recognized)
                for start in range(0, len(entity_ids), 300):
                    batch = entity_ids[start:start + 300]
                    entity_placeholders = ','.join('?' for _ in batch)
                    for row in conn.execute(
                        f"""SELECT entity_id,observation_type,value_json,confidence,citation
                              FROM observations
                             WHERE entity_id IN ({entity_placeholders})
                               AND status IN ('active','needs_review','merge_candidate')
                               AND lower(observation_type) IN ({recognized_placeholders})""",
                        [*batch, *recognized],
                    ):
                        observation_type = str(row['observation_type']).lower()
                        kind = 'user_id' if observation_type in _ENTITY_USER_ID_KEYS else observation_type
                        identifier_values.extend((
                            str(row['entity_id']), kind, normalized, 'observation',
                            min(max(float(row['confidence'] or 0), 0.0), 1.0), row['citation'], timestamp, timestamp,
                        ) for normalized in _flatten_identifier_values(_load_json_dict(row['value_json'])))
                if identifier_values:
                    conn.executemany(
                        """INSERT INTO entity_identifiers(
                               entity_id,identifier_type,normalized_value,source,confidence,citation,created_at,updated_at
                           ) VALUES(?,?,?,?,?,?,?,?)
                           ON CONFLICT(entity_id,identifier_type,normalized_value,source) DO UPDATE SET
                               confidence=MAX(entity_identifiers.confidence,excluded.confidence),
                               citation=COALESCE(excluded.citation,entity_identifiers.citation),
                               updated_at=excluded.updated_at""",
                        identifier_values,
                    )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {'entities': len(entity_rows), 'observations': len(observation_rows), 'commits': 1}

    def add_relationship(self, relationship: RelationshipRecord) -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO relationships(relationship_id,subject_entity_id,predicate,object_entity_id,object_ref,citation,confidence,status,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (relationship.relationship_id, relationship.subject_entity_id, relationship.predicate, relationship.object_entity_id,
                 relationship.object_ref, relationship.citation, relationship.confidence, relationship.status, _json(relationship.metadata or {}), _now()),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM relationships WHERE relationship_id=?', (relationship.relationship_id,)).fetchone())


    def insert_moment_item(self, *, moment_id: str, account_id: str, citation: str, author_id: str | None = None, timestamp: str | None = None, text: str = '', link: dict[str, Any] | None = None, media_refs: list[dict[str, Any]] | None = None, comments: list[dict[str, Any]] | None = None, status: str = 'active', metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO moment_items(moment_id,account_id,author_id,citation,timestamp,text,link_json,media_refs_json,comments_json,status,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (moment_id, account_id, author_id, citation, timestamp, text, _json(link or {}), _json(media_refs or []), _json(comments or []), status, _json(metadata or {})),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM moment_items WHERE moment_id=?', (moment_id,)).fetchone())

    def insert_moment_interaction(self, *, interaction_id: str, moment_id: str, account_id: str, citation: str, interaction_type: str, actor_id: str = '', actor_name: str = '', text: str = '', timestamp: str | None = None, status: str = 'active', metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.store.connect() as conn:
            parent = conn.execute('SELECT account_id FROM moment_items WHERE moment_id=?', (moment_id,)).fetchone()
            if parent is not None and str(parent['account_id']) != str(account_id):
                raise ValueError('moment interaction account_id must match parent moment')
            conn.execute(
                'INSERT OR REPLACE INTO moment_interactions(interaction_id,moment_id,account_id,citation,interaction_type,actor_id,actor_name,text,timestamp,status,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (interaction_id, moment_id, account_id, citation, interaction_type, actor_id, actor_name, text, timestamp, status, _json(metadata or {})),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM moment_interactions WHERE interaction_id=?', (interaction_id,)).fetchone())

    def upsert_moment_batch(
        self,
        moments: Iterable[dict[str, Any]],
        interactions: Iterable[dict[str, Any]],
    ) -> dict[str, int]:
        """Persist one bounded SNS snapshot with a single durable commit."""

        moment_rows = list(moments)
        interaction_rows = list(interactions)
        with self.store.connect() as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                if moment_rows:
                    conn.executemany(
                        'INSERT OR REPLACE INTO moment_items(moment_id,account_id,author_id,citation,timestamp,text,link_json,media_refs_json,comments_json,status,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                        [
                            (
                                row['moment_id'], row['account_id'], row.get('author_id'), row['citation'],
                                row.get('timestamp'), row.get('text') or '', _json(row.get('link') or {}),
                                _json(row.get('media_refs') or []), _json(row.get('comments') or []),
                                row.get('status') or 'active', _json(row.get('metadata') or {}),
                            )
                            for row in moment_rows
                        ],
                    )
                parent_accounts: dict[str, str] = {}
                requested_parents = sorted({str(row['moment_id']) for row in interaction_rows})
                for start in range(0, len(requested_parents), 500):
                    batch = requested_parents[start:start + 500]
                    placeholders = ','.join('?' for _ in batch)
                    parent_accounts.update({
                        str(row['moment_id']): str(row['account_id'])
                        for row in conn.execute(
                            f'SELECT moment_id,account_id FROM moment_items WHERE moment_id IN ({placeholders})',
                            batch,
                        )
                    })
                for row in interaction_rows:
                    parent_account = parent_accounts.get(str(row['moment_id']))
                    if parent_account is not None and parent_account != str(row['account_id']):
                        raise ValueError('moment interaction account_id must match parent moment')
                if interaction_rows:
                    conn.executemany(
                        'INSERT OR REPLACE INTO moment_interactions(interaction_id,moment_id,account_id,citation,interaction_type,actor_id,actor_name,text,timestamp,status,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                        [
                            (
                                row['interaction_id'], row['moment_id'], row['account_id'], row['citation'],
                                row['interaction_type'], row.get('actor_id') or '', row.get('actor_name') or '',
                                row.get('text') or '', row.get('timestamp'), row.get('status') or 'active',
                                _json(row.get('metadata') or {}),
                            )
                            for row in interaction_rows
                        ],
                    )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {
            'moments': len(moment_rows),
            'interactions': len(interaction_rows),
            'commits': 1,
        }

    def insert_favorite(self, *, favorite_id: str, account_id: str, citation: str, timestamp: str | None = None, title: str = '', text: str = '', media_refs: list[dict[str, Any]] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO favorites(favorite_id,account_id,citation,timestamp,title,text,media_refs_json,metadata_json) VALUES(?,?,?,?,?,?,?,?)',
                (favorite_id, account_id, citation, timestamp, title, text, _json(media_refs or []), _json(metadata or {})),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM favorites WHERE favorite_id=?', (favorite_id,)).fetchone())

    def upsert_favorite_batch(self, favorites: Iterable[dict[str, Any]]) -> dict[str, int]:
        rows = list(favorites)
        with self.store.connect() as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                if rows:
                    conn.executemany(
                        'INSERT OR REPLACE INTO favorites(favorite_id,account_id,citation,timestamp,title,text,media_refs_json,metadata_json) VALUES(?,?,?,?,?,?,?,?)',
                        [
                            (
                                row['favorite_id'], row['account_id'], row['citation'], row.get('timestamp'),
                                row.get('title') or '', row.get('text') or '', _json(row.get('media_refs') or []),
                                _json(row.get('metadata') or {}),
                            )
                            for row in rows
                        ],
                    )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {'favorites': len(rows), 'commits': 1}

    def upsert_evidence_item(self, *, evidence_id: str, citation: str, account_id: str, source_type: str, source_id: str, title: str = '', actor: str = '', timestamp: str | None = None, content: str = '', metadata: dict[str, Any] | None = None, status: str = 'active') -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO evidence_items(evidence_id,citation,account_id,source_type,source_id,title,actor,timestamp,content,metadata_json,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(citation) DO UPDATE SET
                   title=excluded.title,actor=excluded.actor,timestamp=excluded.timestamp,content=excluded.content,
                   metadata_json=excluded.metadata_json,status=excluded.status""",
                (evidence_id, citation, account_id, source_type, source_id, title, actor, timestamp, content, _json(metadata or {}), status, _now()),
            )
            conn.commit()
            return dict(conn.execute('SELECT * FROM evidence_items WHERE citation=?', (citation,)).fetchone())


    def provider_job_status(self, *, limit: int = 50) -> list[dict[str, Any]]:
        from trove_core.bounds import BoundedLimit, TRACE_EVENTS_APPROVALS

        limit = BoundedLimit(limit, field='limit', spec=TRACE_EVENTS_APPROVALS)
        if not self.store.path.is_file():
            return []
        with self.store.connect() as conn:
            return [dict(row) for row in conn.execute('SELECT job_id, asset_id, provider, model, job_type, status, retry_count, cost_rmb, citation, created_at, updated_at FROM provider_jobs ORDER BY updated_at DESC LIMIT ?', (limit,))]

    def active_observations(self, entity_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        from trove_core.bounds import BoundedLimit, PRIVATE_LIST

        limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
        with self.store.connect() as conn:
            return [dict(row) for row in conn.execute(
                'SELECT * FROM observations WHERE entity_id=? AND status=? ORDER BY created_at, observation_id LIMIT ?',
                (entity_id, 'active', limit),
            )]
