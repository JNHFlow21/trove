from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re
import sqlite3
import time
import os
import urllib.error
import wave
from typing import Any, Mapping
from uuid import uuid4

from trove_protocol.provider import Provider

from trove_core.approvals import ApprovalGrant, ApprovalValidationError, require_claimed_approval_grant
from trove_core.asr.base import ASRProvider, ASRRequest, ASRResult, ASRUsage
# Compatibility inspection seam for callers/tests that assert ordinary reads
# never construct cloud ASR. Runtime construction remains ProviderFactory-only.
from trove_core.asr.volcengine_flash import VolcengineASRFlashProvider  # noqa: F401
from trove_core.local_vlm.base import LocalVLMCaptionProvider
from trove_core.local_vlm.jobs import run_image_caption_job
from trove_core.local_vlm.mlx_vlm_provider import MlxVLMCaptionProvider
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.providers.config import ProviderConfig
from trove_core.providers.factory import ProviderFactory, ProviderUnavailable
from trove_core.providers.pricing import estimate_asr_flash_rmb
from trove_core.security.egress import cloud_asr_payload
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint, record_vault_mutation_noop
from trove_core.vision.base import VisionProvider
from trove_core.vision.jobs import run_image_observation_job
from trove_core.vision.macos_vision_provider import MacOSVisionOCRProvider
from trove_core.wechat.media.audio_resolver import normalize_audio_file
from trove_core.wechat.media.image_resolver import resolve_image_file
from trove_core.wechat.media.materializer import materialize_media_asset, publish_materialization_result

VOICE_JOB_TYPE = 'voice_transcribe'
IMAGE_JOB_TYPE = 'image_observe'
JOB_TYPES = {VOICE_JOB_TYPE, IMAGE_JOB_TYPE}
RETRYABLE_STATUSES = {'pending', 'failed'}
OUT_OF_SCOPE_REASON = 'out_of_scope'
LOCAL_ASR_PROVIDER_NAME = 'local-faster-whisper'
CLOUD_ASR_PROVIDER_NAME = 'volcengine-asr-flash'
CLOUD_ASR_MODEL_ID = 'bigmodel:volc.bigasr.auc_turbo'
IMAGE_PRECOMPUTE_DISABLED_REASON = 'image_precompute_disabled'
MESSAGE_SHARD_LOCAL_RE = re.compile(r'(?P<shard>message_\d+)[/:][^:/#]+:(?P<local_id>\d+)(?:[#?].*)?$')


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}'


def provider_media_ready(provider: Provider) -> bool:
    """Probe only the stable Provider contract; never import source internals."""
    try:
        capabilities = provider.capabilities()
        health = provider.health()
    except Exception:
        return False
    return bool(
        isinstance(capabilities, Mapping)
        and 'media' in capabilities.get('capabilities', ())
        and isinstance(health, Mapping)
        and health.get('ok') is True
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _path_from_ref(cfg: VaultConfig, path_ref: str) -> Path:
    path = Path(path_ref).expanduser()
    if path.is_absolute():
        return path
    return cfg.root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_cloud_asr_cost(path: Path) -> tuple[float, float] | None:
    """Return the fixed-rate cloud ASR estimate from the exact WAV to upload."""

    try:
        with wave.open(str(path), 'rb') as audio:
            rate = int(audio.getframerate())
            frames = int(audio.getnframes())
    except (EOFError, OSError, wave.Error):
        return None
    if rate <= 0 or frames <= 0:
        return None
    duration = frames / rate
    return duration, estimate_asr_flash_rmb(duration)


def _load_json_obj(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _citation_variants(value: str | None) -> list[str]:
    text = str(value or '').strip()
    if not text:
        return []
    variants = [text]
    if '#chunk-' in text:
        variants.append(text.split('#chunk-', 1)[0])
    if '#image' in text:
        variants.append(text.split('#image', 1)[0])
    if '#voice' in text:
        variants.append(text.split('#voice', 1)[0])
    if '#' in text:
        variants.append(text.split('#', 1)[0])
    return list(dict.fromkeys(v for v in variants if v))


def _message_private_citation(conn: sqlite3.Connection, citation: str | None) -> str | None:
    for candidate in _citation_variants(citation):
        row = conn.execute(
            "SELECT citation FROM messages WHERE citation=? AND conversation_type='private' LIMIT 1",
            (candidate,),
        ).fetchone()
        if row is not None:
            return str(row['citation'])
    return None


def _message_group_citation(conn: sqlite3.Connection, citation: str | None, *, account_id: str) -> str | None:
    """Resolve one exact requested citation to a group message.

    This is deliberately not used by the ordinary media queue. It exists only
    for an explicitly scoped person-profile task that already selected the
    person's own group utterance.
    """
    for candidate in _citation_variants(citation):
        row = conn.execute(
            """SELECT citation FROM messages
                WHERE citation=? AND account_id=? AND conversation_type='group'
                LIMIT 1""",
            (candidate, account_id),
        ).fetchone()
        if row is not None:
            return str(row['citation'])
    return None


def _asset_references_citation(
    conn: sqlite3.Connection,
    asset: sqlite3.Row | dict[str, Any],
    citation: str,
) -> bool:
    variants = set(_citation_variants(citation))
    if not variants:
        return False
    metadata = _load_json_obj(asset['metadata_json'] if 'metadata_json' in asset.keys() else None)  # type: ignore[union-attr]
    candidates: list[str] = _citation_variants(str(asset['citation']))
    for key in ('message_citation', 'source_citation', 'citation'):
        candidates.extend(_citation_variants(metadata.get(key)))
    if variants.intersection(candidates):
        return True
    asset_id = str(asset['asset_id'])
    for row in conn.execute(
        "SELECT source_citation FROM media_asset_links WHERE asset_id=? AND accepted=1",
        (asset_id,),
    ):
        if variants.intersection(_citation_variants(row['source_citation'])):
            return True
    return False


def _private_message_citation_for_asset(conn: sqlite3.Connection, asset: sqlite3.Row | dict[str, Any]) -> str | None:
    """Resolve an asset to one exact private-message citation, or None.

    Voice jobs are allowed only when this resolver can prove the media belongs
    to a private chat message. Group/channel/cache-only media intentionally
    stays out of the ASR queue.
    """
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone():
        return None
    asset_id = str(asset['asset_id'])
    metadata = _load_json_obj(asset['metadata_json'] if 'metadata_json' in asset.keys() else None)  # type: ignore[union-attr]
    candidates: list[str] = []
    for key in ('message_citation', 'source_citation', 'citation'):
        if metadata.get(key):
            candidates.extend(_citation_variants(str(metadata[key])))
    candidates.extend(_citation_variants(str(asset['citation'])))
    for row in conn.execute(
        "SELECT source_citation FROM media_asset_links WHERE asset_id=? AND accepted=1 ORDER BY created_at, link_id",
        (asset_id,),
    ):
        candidates.extend(_citation_variants(row['source_citation']))
    for citation in list(dict.fromkeys(candidates)):
        resolved = _message_private_citation(conn, citation)
        if resolved:
            return resolved

    account_id = str(asset['account_id'])
    conversation_id = str(metadata.get('conversation_id') or '')
    local_id = metadata.get('message_local_id') or metadata.get('local_id')
    shard = metadata.get('message_shard_id') or metadata.get('shard_id')
    # A local id is only unique inside one conversation shard.  Treating
    # missing coordinates as wildcards can attach a group/orphan asset to an
    # unrelated private message that happens to reuse the same local id.
    if local_id is not None and conversation_id and shard:
        clauses = ["account_id=?", "local_id=?", "conversation_type='private'"]
        params: list[Any] = [account_id, int(local_id)]
        clauses.append('conversation_id=?')
        params.append(conversation_id)
        clauses.append('shard_id=?')
        params.append(str(shard))
        rows = list(conn.execute(f"SELECT citation FROM messages WHERE {' AND '.join(clauses)} LIMIT 2", params))
        if len(rows) == 1:
            return str(rows[0]['citation'])

    match = MESSAGE_SHARD_LOCAL_RE.search(str(asset['citation'] or '')) or MESSAGE_SHARD_LOCAL_RE.search(str(asset['source_id'] or ''))
    if match:
        rows = list(conn.execute(
            """SELECT citation,conversation_type FROM messages
               WHERE account_id=? AND shard_id=? AND local_id=?
               LIMIT 2""",
            (account_id, match.group('shard'), int(match.group('local_id'))),
        ))
        if len(rows) == 1 and str(rows[0]['conversation_type']) == 'private':
            return str(rows[0]['citation'])
    return None


def _scoped_voice_citation_for_asset(
    conn: sqlite3.Connection,
    asset: sqlite3.Row | dict[str, Any],
    *,
    requested_citation: str,
    allow_group_voice: bool,
) -> str | None:
    if allow_group_voice:
        group_citation = _message_group_citation(
            conn,
            requested_citation,
            account_id=str(asset['account_id']),
        )
        if group_citation and _asset_references_citation(conn, asset, group_citation):
            return group_citation
    private_citation = _message_private_citation(conn, requested_citation)
    if private_citation and _asset_references_citation(conn, asset, private_citation):
        return private_citation
    return None


@dataclass(frozen=True)
class MediaBudgetReport:
    ok: bool
    action: str
    requested_budget: int
    candidates: int
    processed: int
    completed: int
    idempotent: int
    skipped: int
    failed: int
    avg_item_ms: float
    elapsed_ms: float
    errors: dict[str, int]
    provider: dict[str, Any]
    raw_content_included: bool = False
    raw_paths_included: bool = False
    cloud_calls_made: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_asset_processing_state(store: SQLiteStore, asset_id: str, state: str) -> None:
    with store.connect() as conn:
        conn.execute(
            'UPDATE media_assets SET processing_state=?, updated_at=datetime("now") WHERE asset_id=?',
            (state, asset_id),
        )
        conn.commit()


def ensure_media_jobs(store: SQLiteStore) -> None:
    store.initialize()


def _mark_media_job(store: SQLiteStore, job_id: str, status: str, *, error_code: str | None = None, duration_ms: float = 0.0, increment_retry: bool = False) -> None:
    with store.connect() as conn:
        if increment_retry:
            conn.execute(
                """UPDATE media_jobs
                   SET status=?, error_code=?, retry_count=retry_count+1, last_duration_ms=?, updated_at=?
                   WHERE job_id=?""",
                (status, error_code, float(duration_ms), _now(), job_id),
            )
        else:
            conn.execute(
                """UPDATE media_jobs
                   SET status=?, error_code=?, last_duration_ms=?, updated_at=?
                   WHERE job_id=?""",
                (status, error_code, float(duration_ms), _now(), job_id),
            )
        conn.commit()


def _prepare_media_delta_ids(conn: sqlite3.Connection, asset_ids: list[str] | None) -> str:
    if asset_ids is None:
        return ''
    conn.execute('CREATE TEMP TABLE IF NOT EXISTS _trove_media_delta_ids(asset_id TEXT PRIMARY KEY) WITHOUT ROWID')
    conn.execute('DELETE FROM _trove_media_delta_ids')
    if asset_ids:
        conn.executemany(
            'INSERT OR IGNORE INTO _trove_media_delta_ids(asset_id) VALUES(?)',
            [(value,) for value in asset_ids],
        )
    return ' AND ma.asset_id IN (SELECT asset_id FROM _trove_media_delta_ids)'


def _private_media_exists_sql(asset_alias: str = 'ma') -> str:
    return f"""EXISTS (
        SELECT 1 FROM messages m
        WHERE m.conversation_type='private'
          AND m.account_id={asset_alias}.account_id
          AND (
            m.citation={asset_alias}.citation
            OR {asset_alias}.citation LIKE m.citation || '#%'
            OR EXISTS (
                SELECT 1 FROM media_asset_links l
                WHERE l.asset_id={asset_alias}.asset_id AND l.accepted=1
                  AND (l.source_citation=m.citation OR l.source_citation LIKE m.citation || '#%')
            )
            OR (
                json_valid({asset_alias}.metadata_json)
                AND (
                    json_extract({asset_alias}.metadata_json,'$.message_citation')=m.citation
                    OR (
                        json_extract({asset_alias}.metadata_json,'$.message_local_id') IS NOT NULL
                        AND NULLIF(json_extract({asset_alias}.metadata_json,'$.conversation_id'),'') IS NOT NULL
                        AND NULLIF(json_extract({asset_alias}.metadata_json,'$.message_shard_id'),'') IS NOT NULL
                        AND CAST(json_extract({asset_alias}.metadata_json,'$.message_local_id') AS INTEGER)=m.local_id
                        AND json_extract({asset_alias}.metadata_json,'$.conversation_id')=m.conversation_id
                        AND json_extract({asset_alias}.metadata_json,'$.message_shard_id')=m.shard_id
                    )
                )
            )
          )
    )"""


def _mark_out_of_scope_media_jobs(
    store: SQLiteStore,
    *,
    include_images: bool = False,
    asset_ids: list[str] | None = None,
) -> dict[str, int]:
    if not store.path.exists():
        return {'skipped': 0, 'commits': 0, 'sql_statements': 0, 'rows_written': 0}
    unique_ids = None if asset_ids is None else list(dict.fromkeys(str(value) for value in asset_ids if value))
    if unique_ids == []:
        return {'skipped': 0, 'commits': 0, 'sql_statements': 0, 'rows_written': 0}
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_jobs') or not store._table_exists(conn, 'media_assets'):
            return {'skipped': 0, 'commits': 0, 'sql_statements': 0, 'rows_written': 0}
        id_filter = _prepare_media_delta_ids(conn, unique_ids)
        private_exists = _private_media_exists_sql('ma')
        now = _now()
        # Re-enable a previously rejected job only when the current set-based
        # scope proof now succeeds.
        reset_cursor = conn.execute(
            f"""UPDATE media_jobs
                SET status='pending',error_code=NULL,updated_at=?
                WHERE job_id IN (
                    SELECT mj.job_id FROM media_jobs mj JOIN media_assets ma ON ma.asset_id=mj.asset_id
                    WHERE mj.status='skipped' AND mj.error_code=? {id_filter}
                      AND ((mj.job_type=? AND ma.modality='voice' AND {private_exists})
                           OR (mj.job_type=? AND ?=1 AND ma.modality='image'))
                )""",
            (now, OUT_OF_SCOPE_REASON, VOICE_JOB_TYPE, IMAGE_JOB_TYPE, 1 if include_images else 0),
        )
        skipped_cursor = conn.execute(
            f"""UPDATE media_jobs
                SET status='skipped',error_code=?,updated_at=?
                WHERE job_id IN (
                    SELECT mj.job_id FROM media_jobs mj JOIN media_assets ma ON ma.asset_id=mj.asset_id
                    WHERE (mj.status<>'skipped' OR COALESCE(mj.error_code,'')<>?) {id_filter}
                      AND (
                        (mj.job_type=? AND (ma.modality<>'voice' OR NOT {private_exists}))
                        OR (mj.job_type=? AND (?=0 OR ma.modality<>'image'))
                        OR mj.job_type NOT IN (?,?)
                      )
                )""",
            (
                OUT_OF_SCOPE_REASON, now, OUT_OF_SCOPE_REASON,
                VOICE_JOB_TYPE, IMAGE_JOB_TYPE, 1 if include_images else 0,
                VOICE_JOB_TYPE, IMAGE_JOB_TYPE,
            ),
        )
        skipped = max(skipped_cursor.rowcount, 0)
        asset_cursor = conn.execute(
            f"""UPDATE media_assets SET processing_state='skipped',updated_at=?
                WHERE asset_id IN (
                    SELECT ma.asset_id FROM media_assets ma
                    JOIN media_jobs mj ON mj.asset_id=ma.asset_id
                    WHERE mj.status='skipped' AND mj.error_code=? {id_filter}
                ) AND processing_state<>'skipped'""",
            (now, OUT_OF_SCOPE_REASON),
        )
        rows_written = max(reset_cursor.rowcount, 0) + skipped + max(asset_cursor.rowcount, 0)
        if rows_written:
            conn.commit()
            commits = 1
        else:
            conn.rollback()
            commits = 0
    return {'skipped': skipped, 'commits': commits, 'sql_statements': 3, 'rows_written': rows_written}


def enqueue_media_jobs(
    store: SQLiteStore,
    *,
    modalities: set[str] | None = None,
    include_images: bool = False,
    asset_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    ensure_media_jobs(store)
    modalities = set(modalities or {'voice'})
    if not include_images:
        modalities.discard('image')
    unique_ids = None if asset_ids is None else list(dict.fromkeys(str(value) for value in asset_ids if value))
    if unique_ids == []:
        return {
            'seen': 0, 'queued': 0, 'by_type': {}, 'skipped_out_of_scope': 0,
            'include_images': bool(include_images), 'raw_content_included': False,
            'metrics': {'sql_statements': 0, 'commits': 0, 'rows_scanned': 0, 'candidate_rows': 0, 'rows_written': 0, 'wal_bytes': 0},
        }
    wal_path = store.path.with_name(store.path.name + '-wal')
    wal_before = wal_path.stat().st_size if wal_path.exists() else 0
    scope_report = _mark_out_of_scope_media_jobs(store, include_images=include_images, asset_ids=unique_ids)
    skipped_out_of_scope = int(scope_report['skipped'])
    now = _now()
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_assets'):
            return {'seen': 0, 'queued': 0, 'by_type': {}, 'skipped_out_of_scope': skipped_out_of_scope, 'include_images': bool(include_images), 'raw_content_included': False}
        if not modalities:
            return {'seen': 0, 'queued': 0, 'by_type': {}, 'skipped_out_of_scope': skipped_out_of_scope, 'include_images': bool(include_images), 'raw_content_included': False}
        id_filter = _prepare_media_delta_ids(conn, unique_ids)
        clauses = ["ma.modality IN (%s)" % ','.join('?' for _ in modalities)]
        params: list[Any] = sorted(modalities)
        eligible = f"""{' AND '.join(clauses)} {id_filter}
          AND (NOT EXISTS (SELECT 1 FROM media_asset_links l WHERE l.asset_id=ma.asset_id)
               OR EXISTS (SELECT 1 FROM media_asset_links l WHERE l.asset_id=ma.asset_id AND l.accepted=1))
          AND (ma.modality<>'voice' OR {_private_media_exists_sql('ma')})"""
        seen = int(conn.execute(f'SELECT COUNT(*) FROM media_assets ma WHERE {eligible}', params).fetchone()[0])
        cursor = conn.execute(
            f"""INSERT OR IGNORE INTO media_jobs(job_id,asset_id,job_type,status,retry_count,error_code,last_duration_ms,created_at,updated_at)
                SELECT 'mediajob:' ||
                       CASE ma.modality WHEN 'voice' THEN ? ELSE ? END || ':' || ma.asset_id,
                       ma.asset_id,
                       CASE ma.modality WHEN 'voice' THEN ? ELSE ? END,
                       'pending',0,NULL,0.0,?,?
                FROM media_assets ma
                WHERE {eligible}
                RETURNING job_type""",
            [VOICE_JOB_TYPE, IMAGE_JOB_TYPE, VOICE_JOB_TYPE, IMAGE_JOB_TYPE, now, now, *params],
        )
        by_type: dict[str, int] = {}
        inserted = 0
        for row in cursor:
            job_type = str(row['job_type'])
            inserted += 1
            by_type[job_type] = by_type.get(job_type, 0) + 1
        if inserted:
            conn.commit()
            queue_commits = 1
        else:
            conn.rollback()
            queue_commits = 0
    wal_after = wal_path.stat().st_size if wal_path.exists() else 0
    return {
        'seen': seen,
        'queued': inserted,
        'by_type': dict(sorted(by_type.items())),
        'skipped_out_of_scope': skipped_out_of_scope,
        'include_images': bool(include_images),
        'raw_content_included': False,
        'metrics': {
            'sql_statements': int(scope_report['sql_statements']) + 2,
            'commits': int(scope_report['commits']) + queue_commits,
            'rows_scanned': seen,
            'candidate_rows': 0 if unique_ids is None else len(unique_ids),
            'rows_written': int(scope_report['rows_written']) + inserted,
            'wal_bytes': max(0, wal_after - wal_before),
        },
    }


def _job_candidates(store: SQLiteStore, *, job_type: str, limit: int, max_retries: int = 2, conversation_id: str | None = None) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if not store.path.exists():
        return []
    if not store.readonly:
        ensure_media_jobs(store)
    with store.connect_once() as conn:
        if not store._table_exists(conn, 'media_jobs'):
            return []
        conversation_filter = ''
        params: list[Any] = [job_type, int(max_retries)]
        if conversation_id:
            conversation_filter = """
                 AND EXISTS (
                   SELECT 1 FROM messages m
                   WHERE m.conversation_id=?
                     AND (
                       m.citation=ma.citation
                       OR EXISTS (
                         SELECT 1 FROM media_asset_links l
                         WHERE l.asset_id=ma.asset_id AND l.accepted=1 AND l.source_citation=m.citation
                       )
                     )
                 )"""
            params.append(conversation_id)
        params.append(int(limit))
        rows = conn.execute(
            f"""SELECT mj.job_id, mj.retry_count, mj.status AS job_status,
                      ma.asset_id, ma.citation, ma.path_ref, ma.cache_state
               FROM media_jobs mj
               JOIN media_assets ma ON ma.asset_id=mj.asset_id
               WHERE mj.job_type=?
                 AND (
                   mj.status IN ('pending', 'failed')
                   OR (mj.status='skipped' AND mj.error_code='missing_local_cache' AND COALESCE(ma.path_ref, '') <> '')
                   OR (mj.status='running' AND julianday(mj.updated_at) < julianday('now','-30 minutes'))
                 )
                 AND mj.retry_count <= ?
                 {conversation_filter}
               ORDER BY CASE
                          WHEN COALESCE(ma.path_ref, '') <> '' AND ma.cache_state NOT IN ('missing_local_cache', 'metadata_only') THEN 0
                          WHEN COALESCE(ma.path_ref, '') <> '' THEN 1
                          ELSE 2
                        END,
                        mj.updated_at, mj.job_id
               LIMIT ?""",
            params,
        )
        return [dict(row) for row in rows]


def _conversation_voice_asset_ids(store: SQLiteStore, conversation_id: str) -> list[str]:
    """Return voice assets linked to one conversation without scanning all media JSON."""
    with store.connect_once() as conn:
        if not all(store._table_exists(conn, table) for table in ('messages', 'media_assets')):
            return []
        link_union = ''
        if store._table_exists(conn, 'media_asset_links'):
            link_union = """
                UNION
                SELECT ma.asset_id
                  FROM messages m
                  JOIN media_asset_links l
                    ON l.source_citation=m.citation AND l.accepted=1
                  JOIN media_assets ma
                    ON ma.asset_id=l.asset_id AND ma.modality='voice'
                 WHERE m.conversation_id=? AND m.conversation_type='private'
            """
        params = [conversation_id, *([conversation_id] if link_union else [])]
        rows = conn.execute(
            f"""
                SELECT ma.asset_id
                  FROM messages m
                  JOIN media_assets ma
                    ON ma.citation=m.citation AND ma.modality='voice'
                 WHERE m.conversation_id=? AND m.conversation_type='private'
                {link_union}
                ORDER BY asset_id
            """,
            params,
        )
        return [str(row['asset_id']) for row in rows]


def _voice_asset_for_citation(
    store: SQLiteStore,
    citation: str,
    *,
    allow_group_voice: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    candidates = _citation_variants(citation)
    if not candidates:
        return None, None
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_assets'):
            return None, None
        placeholders = ','.join('?' for _ in candidates)
        rows = list(conn.execute(
            f"""SELECT *
                FROM media_assets
                WHERE modality='voice'
                  AND citation IN ({placeholders})
                ORDER BY updated_at DESC, asset_id
                LIMIT 5""",
            candidates,
        ))
        if not rows and store._table_exists(conn, 'media_asset_links'):
            rows = list(conn.execute(
                f"""SELECT ma.*
                    FROM media_asset_links l
                    JOIN media_assets ma ON ma.asset_id=l.asset_id
                    WHERE ma.modality='voice'
                      AND l.accepted=1
                      AND l.source_citation IN ({placeholders})
                    ORDER BY l.created_at DESC, ma.asset_id
                    LIMIT 5""",
                candidates,
            ))
        for row in rows:
            scoped_citation = _scoped_voice_citation_for_asset(
                conn,
                row,
                requested_citation=citation,
                allow_group_voice=allow_group_voice,
            )
            if scoped_citation:
                return dict(row), scoped_citation
        if rows:
            return dict(rows[0]), None
    return None, None


def _existing_transcript_for_asset(store: SQLiteStore, asset_id: str) -> dict[str, Any] | None:
    with store.connect() as conn:
        if not store._table_exists(conn, 'transcripts'):
            return None
        row = conn.execute(
            """SELECT t.transcript_id,t.citation,t.language,t.confidence,
                      t.duration_seconds,t.status
                 FROM transcripts t
                 JOIN provider_jobs pj ON pj.job_id=t.job_id
                 JOIN media_assets ma ON ma.asset_id=t.asset_id
                WHERE t.asset_id=? AND t.status='active'
                  AND pj.provider=? AND pj.model=? AND pj.status='completed'
                  AND pj.request_hash=ma.content_hash
             ORDER BY t.created_at DESC LIMIT 1""",
            (asset_id, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
        ).fetchone()
        return dict(row) if row is not None else None


def _cloud_asr_provider_from_runtime(env: dict[str, str] | None = None) -> tuple[ASRProvider | None, str | None]:
    env = env if env is not None else os.environ
    pcfg = ProviderConfig.resolve(env)
    if not pcfg.cloud_asr_enabled:
        return None, 'cloud_asr_disabled'
    try:
        return ProviderFactory.resolve(env).create_asr(), None
    except ProviderUnavailable as exc:
        return None, exc.code


@dataclass(frozen=True)
class _VoiceInferenceClaim:
    asset_id: str
    citation: str
    path_ref: str
    source_path: Path
    audio_path: Path
    content_sha256: str
    file_fingerprint: tuple[int, int, int, int]
    job_id: str
    transcript_id: str
    attempt_token: str
    allow_group_voice: bool
    update_media_job: bool


def _file_fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _asr_stable(prefix: str, value: str) -> str:
    # Preserve the IDs emitted by trove_core.asr.jobs.
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _cached_voice_result(asset_id: str, transcript: dict[str, Any]) -> dict[str, Any]:
    return {
        'ok': True,
        'status': 'cached',
        'asset_id': asset_id,
        'transcript': transcript,
        'cloud_calls_made': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def _claim_voice_inference(
    store: SQLiteStore,
    *,
    asset_id: str,
    expected_path_ref: str,
    message_citation: str,
    allow_group_voice: bool,
    source_path: Path,
    audio_path: Path,
    content_sha256: str,
    file_fingerprint: tuple[int, int, int, int],
    provider: ASRProvider,
    decode: Any,
) -> _VoiceInferenceClaim | dict[str, Any]:
    job_id = _asr_stable('job', f'asr:{provider.name}:{asset_id}')
    transcript_id = _asr_stable('transcript', f'{asset_id}:{provider.name}')
    attempt_token = f'asr-attempt:{uuid4().hex}'
    now = _now()
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        asset = conn.execute('SELECT * FROM media_assets WHERE asset_id=?', (asset_id,)).fetchone()
        if asset is None or str(asset['path_ref'] or '') != expected_path_ref:
            conn.rollback()
            return {'ok': False, 'status': 'retryable_failure', 'reason': 'voice_source_changed'}
        if _scoped_voice_citation_for_asset(
            conn,
            asset,
            requested_citation=message_citation,
            allow_group_voice=allow_group_voice,
        ) != message_citation:
            conn.rollback()
            return {'ok': False, 'status': 'retryable_failure', 'reason': 'voice_scope_changed'}
        update_media_job = _private_message_citation_for_asset(conn, asset) == message_citation
        existing = conn.execute(
            """SELECT t.* FROM transcripts t
                 JOIN provider_jobs pj ON pj.job_id=t.job_id
                 JOIN media_assets ma ON ma.asset_id=t.asset_id
                WHERE t.asset_id=? AND t.status='active'
                  AND pj.provider=? AND pj.model=? AND pj.status='completed'
                  AND pj.request_hash=ma.content_hash
                ORDER BY t.created_at DESC LIMIT 1""",
            (asset_id, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
        ).fetchone()
        if existing is not None:
            conn.rollback()
            return _cached_voice_result(asset_id, dict(existing))
        prior = conn.execute(
            'SELECT status,request_hash,updated_at FROM provider_jobs WHERE job_id=?',
            (job_id,),
        ).fetchone()
        if prior is not None and str(prior['status']) == 'running':
            recent = conn.execute(
                "SELECT julianday(?) >= julianday('now','-30 minutes')",
                (prior['updated_at'],),
            ).fetchone()
            if recent is not None and bool(recent[0]):
                conn.rollback()
                return {
                    'ok': True,
                    'status': 'in_progress',
                    'asset_id': asset_id,
                    'job_id': job_id,
                    'cloud_calls_made': False,
                    'raw_content_included': False,
                    'raw_paths_included': False,
                }
        conn.execute(
            'UPDATE media_assets SET content_hash=?,processing_state=?,updated_at=? WHERE asset_id=?',
            (content_sha256, 'processing', now, asset_id),
        )
        conn.execute(
            """INSERT OR REPLACE INTO media_decode_results(
                   decode_id,asset_id,status,wrapper_type,input_hash,output_hash,
                   derivative_ref,error_code,metadata_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                _stable('decode', f'lazy-audio:{asset_id}:{decode.input_hash or expected_path_ref}'),
                asset_id, decode.status, decode.codec or 'audio', decode.input_hash,
                decode.output_hash, decode.derivative_ref, decode.error_code,
                json.dumps({'pipeline': 'lazy_voice_asr'}, ensure_ascii=False, sort_keys=True), now,
            ),
        )
        conn.execute(
            """INSERT INTO provider_jobs(
                   job_id,asset_id,provider,model,job_type,status,retry_count,usage_json,
                   cost_rmb,request_hash,error_code,citation,created_at,updated_at)
               VALUES(?,?,?,?,?,'running',0,'{}',0,?,NULL,?,?,?)
               ON CONFLICT(job_id) DO UPDATE SET
                   asset_id=excluded.asset_id,provider=excluded.provider,model=excluded.model,
                   job_type=excluded.job_type,status='running',usage_json='{}',cost_rmb=0,
                   request_hash=excluded.request_hash,error_code=NULL,citation=excluded.citation,
                   updated_at=excluded.updated_at""",
            (
                job_id, asset_id, provider.name,
                f'{getattr(provider, "model_name", "")}:{getattr(provider, "resource_id", "")}',
                'asr', attempt_token, message_citation, now, now,
            ),
        )
        if update_media_job:
            conn.execute(
                """UPDATE media_jobs SET status='running',error_code=NULL,updated_at=?
                     WHERE asset_id=? AND job_type=? AND status IN ('pending','failed','running')""",
                (now, asset_id, VOICE_JOB_TYPE),
            )
        conn.commit()
    return _VoiceInferenceClaim(
        asset_id=asset_id,
        citation=message_citation,
        path_ref=expected_path_ref,
        source_path=source_path,
        audio_path=audio_path,
        content_sha256=content_sha256,
        file_fingerprint=file_fingerprint,
        job_id=job_id,
        transcript_id=transcript_id,
        attempt_token=attempt_token,
        allow_group_voice=allow_group_voice,
        update_media_job=update_media_job,
    )


def _commit_voice_inference(
    store: SQLiteStore,
    claim: _VoiceInferenceClaim,
    provider: ASRProvider,
    *,
    result: ASRResult | None,
    failure_status: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    now = _now()
    affected_citations: list[str] = []
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        job = conn.execute(
            'SELECT status,request_hash FROM provider_jobs WHERE job_id=?',
            (claim.job_id,),
        ).fetchone()
        if job is None or str(job['status']) != 'running' or str(job['request_hash'] or '') != claim.attempt_token:
            conn.rollback()
            return {
                'ok': True,
                'status': 'superseded',
                'asset_id': claim.asset_id,
                'job_id': claim.job_id,
                'cloud_calls_made': getattr(provider, 'egress_kind', None) == 'cloud_asr_upload',
            }
        asset = conn.execute('SELECT * FROM media_assets WHERE asset_id=?', (claim.asset_id,)).fetchone()
        source_matches = (
            asset is not None
            and str(asset['path_ref'] or '') == claim.path_ref
            and str(asset['content_hash'] or '').lower() == claim.content_sha256
            and _scoped_voice_citation_for_asset(
                conn,
                asset,
                requested_citation=claim.citation,
                allow_group_voice=claim.allow_group_voice,
            ) == claim.citation
        )
        try:
            source_matches = bool(source_matches and _file_fingerprint(claim.source_path) == claim.file_fingerprint)
        except OSError:
            source_matches = False
        if not source_matches:
            conn.execute(
                "UPDATE provider_jobs SET status='terminal_failure',error_code='voice_source_changed',updated_at=? WHERE job_id=? AND request_hash=?",
                (now, claim.job_id, claim.attempt_token),
            )
            if claim.update_media_job:
                conn.execute(
                    "UPDATE media_jobs SET status='failed',error_code='voice_source_changed',retry_count=retry_count+1,updated_at=? WHERE asset_id=? AND job_type=?",
                    (now, claim.asset_id, VOICE_JOB_TYPE),
                )
            conn.commit()
            return {'ok': False, 'status': 'retryable_failure', 'reason': 'voice_source_changed', 'job_id': claim.job_id}
        if result is None:
            status = failure_status or 'terminal_failure'
            code = error_code or 'provider_rejected_or_failed'
            conn.execute(
                'UPDATE provider_jobs SET status=?,error_code=?,updated_at=? WHERE job_id=? AND request_hash=?',
                (status, code, now, claim.job_id, claim.attempt_token),
            )
            conn.execute(
                "UPDATE media_assets SET processing_state='failed',updated_at=? WHERE asset_id=?",
                (now, claim.asset_id),
            )
            if claim.update_media_job:
                conn.execute(
                    """UPDATE media_jobs SET status='failed',error_code=?,retry_count=retry_count+1,updated_at=?
                         WHERE asset_id=? AND job_type=?""",
                    (code, now, claim.asset_id, VOICE_JOB_TYPE),
                )
            conn.commit()
            return {'ok': False, 'status': status, 'error_code': code, 'job_id': claim.job_id}

        active = conn.execute(
            """SELECT t.* FROM transcripts t
                 JOIN provider_jobs pj ON pj.job_id=t.job_id
                 JOIN media_assets ma ON ma.asset_id=t.asset_id
                WHERE t.asset_id=? AND t.status='active'
                  AND pj.provider=? AND pj.model=? AND pj.status='completed'
                  AND pj.request_hash=ma.content_hash
                ORDER BY t.created_at DESC LIMIT 1""",
            (claim.asset_id, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
        ).fetchone()
        conn.execute(
            """UPDATE provider_jobs SET status='completed',usage_json=?,cost_rmb=?,request_hash=?,error_code=NULL,updated_at=?
                 WHERE job_id=? AND request_hash=?""",
            (
                json.dumps(result.usage.to_dict(), ensure_ascii=False, sort_keys=True),
                float(result.usage.estimated_cost_rmb), claim.content_sha256, now,
                claim.job_id, claim.attempt_token,
            ),
        )
        idempotent = active is not None
        if active is None:
            voice_citation = f'{claim.citation}#voice'
            affected_citations.extend(
                str(row['citation'])
                for row in conn.execute(
                    "SELECT citation FROM transcripts WHERE asset_id=? AND status='active'",
                    (claim.asset_id,),
                )
            )
            affected_citations.append(voice_citation)
            # A known local-faster-whisper projection is never a valid cache
            # under the cloud-only voice policy. Keep it until cloud succeeds,
            # then atomically supersede it before publishing the cloud result.
            conn.execute(
                "UPDATE transcripts SET status='superseded' WHERE asset_id=? AND status='active'",
                (claim.asset_id,),
            )
            conn.execute(
                """INSERT OR REPLACE INTO transcripts(
                       transcript_id,asset_id,job_id,citation,text,language,confidence,
                       duration_seconds,status,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    claim.transcript_id, claim.asset_id, claim.job_id, voice_citation,
                    result.text, result.language, float(result.confidence or 0.0),
                    float(result.usage.duration_seconds), 'active', now,
                ),
            )
        conn.execute(
            "UPDATE media_assets SET processing_state='done',updated_at=? WHERE asset_id=?",
            (now, claim.asset_id),
        )
        if claim.update_media_job:
            conn.execute(
                """UPDATE media_jobs SET status='done',error_code=NULL,updated_at=?
                     WHERE asset_id=? AND job_type=?""",
                (now, claim.asset_id, VOICE_JOB_TYPE),
            )
        conn.commit()
        transcript = dict(active) if active is not None else dict(conn.execute(
            'SELECT * FROM transcripts WHERE transcript_id=?', (claim.transcript_id,),
        ).fetchone())
    if affected_citations:
        store.upsert_evidence_chunks_for_source_citations('transcript', affected_citations)
    return {
        'ok': True,
        'status': 'completed',
        'asset_id': claim.asset_id,
        'job_id': claim.job_id,
        'transcript_id': transcript['transcript_id'],
        'transcript': transcript,
        'idempotent': idempotent,
        'estimated_cost_rmb': float(result.usage.estimated_cost_rmb),
    }


@mutation_entrypoint('media_transcribe')
def ensure_voice_transcript(
    vault_root: str | Path | None = None,
    *,
    citation: str,
    allow_cloud_asr: bool = False,
    allow_local_asr: bool = False,
    provider: ASRProvider | None = None,
    env: dict[str, str] | None = None,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict[str, Any] | None = None,
    estimate_cloud_asr_cost: bool = False,
    cloud_cost_ceiling_rmb: float | None = None,
    allow_group_voice: bool = False,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    """Ensure one voice citation has a cloud transcript, without implicit upload.

    Default behavior is intentionally lazy-safe: if no cached transcript exists,
    return a pending state and do not call any provider. Local ASR is forbidden;
    the compatibility ``allow_local_asr`` argument is ignored. Cloud ASR requires
    ``allow_cloud_asr=True``, provider readiness, and an exact approval grant.
    """
    if type(allow_local_asr) is not bool:
        raise TypeError('allow_local_asr must be an exact boolean')
    if type(estimate_cloud_asr_cost) is not bool:
        raise TypeError('estimate_cloud_asr_cost must be an exact boolean')
    if type(allow_group_voice) is not bool:
        raise TypeError('allow_group_voice must be an exact boolean')
    if cloud_cost_ceiling_rmb is not None and (
        type(cloud_cost_ceiling_rmb) not in {int, float}
        or not math.isfinite(float(cloud_cost_ceiling_rmb))
        or float(cloud_cost_ceiling_rmb) < 0
    ):
        raise ValueError('cloud_cost_ceiling_rmb is out of range')
    started = time.perf_counter()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('voice inference cannot run inside an outer writer session')
    cfg.ensure()
    store = SQLiteStore(cfg.paths.sqlite_path)
    asset, scoped_citation = _voice_asset_for_citation(
        store,
        citation,
        allow_group_voice=allow_group_voice,
    )
    if asset is None:
        record_vault_mutation_noop(operation='media_transcribe')
        return {'ok': False, 'status': 'media_unavailable', 'reason': 'voice_asset_not_found', 'citation': citation, 'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False}
    asset_id = str(asset['asset_id'])
    if scoped_citation is None:
        record_vault_mutation_noop(operation='media_transcribe')
        return {'ok': False, 'status': 'skipped', 'reason': OUT_OF_SCOPE_REASON, 'asset_id': asset_id, 'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False}
    existing = _existing_transcript_for_asset(store, asset_id)
    if existing:
        record_vault_mutation_noop(operation='media_transcribe')
        return _cached_voice_result(asset_id, existing)

    path_ref = str(asset.get('path_ref') or '')
    media_path = _path_from_ref(cfg, path_ref) if path_ref else Path()
    if not path_ref or not media_path.exists() or not path_is_under(media_path, cfg.root):
        # Locate/copy/decode/hash up to 64MB without the writer. Only the
        # conditional SQLite publication below is coordinated.
        materialized = materialize_media_asset(
            cfg,
            store,
            asset,
            citation=scoped_citation,
            publish=False,
        )
        if not materialized.ok:
            return {
                'ok': False, 'status': materialized.status, 'reason': materialized.reason,
                'asset_id': asset_id, 'cloud_calls_made': False,
                'raw_content_included': False, 'raw_paths_included': False,
            }
        with coordinated_vault_mutation(cfg, operation='media_transcribe'):
            current, current_citation = _voice_asset_for_citation(
                store,
                citation,
                allow_group_voice=allow_group_voice,
            )
            if (
                current is None
                or current_citation is None
                or str(current['asset_id']) != asset_id
                or str(current.get('path_ref') or '') != path_ref
                or not publish_materialization_result(
                    store,
                    materialized,
                    expected_path_ref=path_ref,
                )
            ):
                return {'ok': False, 'status': 'retryable_failure', 'reason': 'voice_source_changed', 'asset_id': asset_id, 'cloud_calls_made': False}
        asset, scoped_citation = _voice_asset_for_citation(
            store,
            citation,
            allow_group_voice=allow_group_voice,
        )
        if asset is None or scoped_citation is None:
            return {'ok': False, 'status': 'retryable_failure', 'reason': 'voice_source_changed', 'asset_id': asset_id, 'cloud_calls_made': False}
        path_ref = str(asset.get('path_ref') or '')
        media_path = _path_from_ref(cfg, path_ref)

    # File hashing, provider construction, model loading, normalization, and
    # inference intentionally happen without the Vault writer lease.
    content_sha256 = _sha256_file(media_path)
    approved_profile_scope = (
        approval_payload.get('profile_scope')
        if isinstance(approval_payload, dict) else None
    )
    if isinstance(approved_profile_scope, dict) and approved_profile_scope.get('content_hash') != content_sha256:
        record_vault_mutation_noop(operation='media_transcribe')
        return {
            'ok': False,
            'status': 'retryable_failure',
            'reason': 'voice_source_changed',
            'asset_id': asset_id,
            'cloud_calls_made': False,
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    resolved_provider = provider
    provider_reason = 'cloud_asr_not_requested'
    if getattr(resolved_provider, 'name', None) == LOCAL_ASR_PROVIDER_NAME:
        resolved_provider = None
        provider_reason = 'local_asr_forbidden'
    if resolved_provider is None and allow_cloud_asr:
        resolved_provider, reason = _cloud_asr_provider_from_runtime(env)
        if resolved_provider is None:
            record_vault_mutation_noop(operation='media_transcribe')
            return {'ok': False, 'status': 'needs_provider', 'reason': reason or provider_reason or 'provider_unavailable', 'asset_id': asset_id, 'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False}
    cost_estimate_only = resolved_provider is None and estimate_cloud_asr_cost
    if resolved_provider is None:
        with coordinated_vault_mutation(cfg, operation='media_transcribe'):
            with store.connect() as conn:
                conn.execute(
                    'UPDATE media_assets SET content_hash=?,updated_at=? WHERE asset_id=? AND path_ref=?',
                    (content_sha256, _now(), asset_id, path_ref),
                )
                conn.commit()
        if not cost_estimate_only:
            return {
                'ok': True, 'status': 'pending_transcript', 'reason': provider_reason,
                'asset_id': asset_id, 'content_sha256': content_sha256,
                'citation': scoped_citation, 'media_hint': {'modality': 'voice', 'asset_id': asset_id},
                'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False,
            }

    egress_kind = getattr(resolved_provider, 'egress_kind', None)
    if resolved_provider is not None:
        provider_config = ProviderConfig.resolve(env)
        if (
            egress_kind != 'cloud_asr_upload'
            or getattr(resolved_provider, 'name', None) != CLOUD_ASR_PROVIDER_NAME
            or getattr(resolved_provider, 'model_name', None) != provider_config.asr_model_name
            or getattr(resolved_provider, 'resource_id', None) != provider_config.asr_resource_id
            or getattr(resolved_provider, 'endpoint', None) != provider_config.asr_endpoint
        ):
            raise RuntimeError('cloud_asr_provider_identity_mismatch')
    if egress_kind == 'cloud_asr_upload':
        if not allow_cloud_asr:
            raise ApprovalValidationError('cloud ASR requires explicit allow_cloud_asr', code='cloud_asr_not_allowed')
        expected_payload = cloud_asr_payload(
            citation=citation,
            provider=resolved_provider.name,
            model=resolved_provider.model_name,
            resource_id=resolved_provider.resource_id,
            endpoint=provider_config.asr_endpoint,
        )
        if type(approval_payload) is not dict or any(approval_payload.get(key) != value for key, value in expected_payload.items()):
            raise ApprovalValidationError('cloud ASR approval payload does not match the outbound request', code='grant_payload_mismatch')
        require_claimed_approval_grant(
            approval_grant,  # type: ignore[arg-type]
            cfg.root,
            action='voice_cloud_asr',
            danger_class='cloud_asr_upload',
            payload=approval_payload,
        )

    resolved = normalize_audio_file(media_path, cfg.root, asset_id=asset_id)
    if resolved.derivative_ref is None or resolved.status not in {'copied', 'normalized'}:
        with coordinated_vault_mutation(cfg, operation='media_transcribe'):
            MultimodalRepository(store).record_decode_result(
                decode_id=_stable('decode', f'lazy-audio:{asset_id}:{resolved.input_hash or path_ref}'),
                asset_id=asset_id, status=resolved.status, wrapper_type=resolved.codec or 'audio',
                input_hash=resolved.input_hash, output_hash=resolved.output_hash,
                derivative_ref=resolved.derivative_ref, error_code=resolved.error_code,
                metadata={'pipeline': 'lazy_voice_asr'},
            )
        return {'ok': False, 'status': 'failed', 'reason': resolved.error_code or resolved.status, 'asset_id': asset_id, 'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False}

    audio_path = cfg.root / resolved.derivative_ref
    budget_estimate = _wav_cloud_asr_cost(audio_path)
    if cost_estimate_only:
        if budget_estimate is None:
            return {
                'ok': False, 'status': 'paused_budget', 'reason': 'cloud_asr_cost_estimate_unavailable',
                'asset_id': asset_id, 'content_sha256': content_sha256,
                'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False,
            }
        duration_seconds, estimated_cost_rmb = budget_estimate
        return {
            'ok': True, 'status': 'pending_transcript', 'reason': provider_reason,
            'asset_id': asset_id, 'content_sha256': content_sha256,
            'duration_seconds': duration_seconds, 'estimated_cost_rmb': estimated_cost_rmb,
            'citation': scoped_citation, 'media_hint': {'modality': 'voice', 'asset_id': asset_id},
            'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False,
        }

    if resolved_provider is None:  # narrowed by the cost-estimate-only return above
        raise RuntimeError('ASR provider resolution invariant failed')
    if egress_kind == 'cloud_asr_upload' and cloud_cost_ceiling_rmb is not None:
        if budget_estimate is None:
            return {
                'ok': False, 'status': 'paused_budget', 'reason': 'cloud_asr_cost_estimate_unavailable',
                'asset_id': asset_id, 'content_sha256': content_sha256,
                'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False,
            }
        _duration_seconds, estimated_cost_rmb = budget_estimate
        if estimated_cost_rmb > float(cloud_cost_ceiling_rmb):
            return {
                'ok': False, 'status': 'paused_budget', 'reason': 'cloud_asr_cost_ceiling_exceeded',
                'asset_id': asset_id, 'content_sha256': content_sha256,
                'estimated_cost_rmb': estimated_cost_rmb,
                'cost_ceiling_rmb': round(float(cloud_cost_ceiling_rmb), 6),
                'cloud_calls_made': False, 'raw_content_included': False, 'raw_paths_included': False,
            }
    fingerprint = _file_fingerprint(media_path)
    with coordinated_vault_mutation(cfg, operation='media_transcribe'):
        claim = _claim_voice_inference(
            store,
            asset_id=asset_id,
            expected_path_ref=path_ref,
            message_citation=scoped_citation,
            allow_group_voice=allow_group_voice,
            source_path=media_path,
            audio_path=audio_path,
            content_sha256=content_sha256,
            file_fingerprint=fingerprint,
            provider=resolved_provider,
            decode=resolved,
        )
        if isinstance(claim, _VoiceInferenceClaim):
            # Remove any legacy/local transcript chunk before cloud inference.
            # The historical transcript row stays recoverable until cloud
            # succeeds, but it is no longer eligible search/profile evidence.
            store.upsert_evidence_chunks_for_source_citations(
                'transcript', [f'{claim.citation}#voice'],
            )
    if isinstance(claim, dict):
        if estimate_cloud_asr_cost and budget_estimate is not None:
            claim.setdefault('estimated_cost_rmb', budget_estimate[1])
        claim.setdefault('content_sha256', content_sha256)
        claim.setdefault('elapsed_ms', round((time.perf_counter() - started) * 1000, 3))
        claim.setdefault('raw_content_included', False)
        claim.setdefault('raw_paths_included', False)
        return claim

    inference: ASRResult | None = None
    failure_status = error_code = None
    try:
        inference = resolved_provider.transcribe(ASRRequest(
            asset_id=asset_id,
            audio_path=audio_path,
            citation=scoped_citation,
        ))
        provider_status = str(inference.provider_status or '').lower() if inference is not None else ''
        if (
            inference is None
            or not str(inference.text or '').strip()
            or any(token in provider_status for token in ('error', 'fail', 'reject'))
        ):
            inference = None
            failure_status, error_code = 'terminal_failure', 'cloud_asr_invalid_response'
        if inference is not None and egress_kind == 'cloud_asr_upload' and budget_estimate is not None:
            # The submitted WAV duration and the fixed provider rate are known
            # before upload. Keep accounting on that same estimate so a noisy
            # or missing provider duration cannot undercount an already-paid task.
            duration_seconds, estimated_cost_rmb = budget_estimate
            inference = ASRResult(
                text=inference.text,
                language=inference.language,
                confidence=inference.confidence,
                usage=ASRUsage(
                    duration_seconds=duration_seconds,
                    estimated_cost_rmb=estimated_cost_rmb,
                ),
                citations=inference.citations,
                provider_status=inference.provider_status,
            )
    except TimeoutError:
        failure_status, error_code = 'retryable_failure', 'provider_timeout'
    except urllib.error.URLError:
        failure_status, error_code = 'retryable_failure', 'provider_transport_error'
    except Exception:
        failure_status, error_code = 'terminal_failure', 'provider_rejected_or_failed'

    with coordinated_vault_mutation(cfg, operation='media_transcribe'):
        committed = _commit_voice_inference(
            store,
            claim,
            resolved_provider,
            result=inference,
            failure_status=failure_status,
            error_code=error_code,
        )
    committed['content_sha256'] = content_sha256
    if estimate_cloud_asr_cost and budget_estimate is not None:
        committed.setdefault('estimated_cost_rmb', budget_estimate[1])
    committed['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
    committed['cloud_calls_made'] = egress_kind == 'cloud_asr_upload'
    committed['raw_content_included'] = False
    committed['raw_paths_included'] = False
    return committed


def voice_transcription_plan(
    vault_root: str | Path | None = None,
    *,
    conversation_id: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    candidates: list[dict[str, Any]] = []
    pending = 0
    if store.path.exists():
        with store.connect() as conn:
            if store._table_exists(conn, 'media_assets'):
                params: list[Any] = []
                conversation_filter = ''
                if conversation_id:
                    conversation_filter = """
                        AND EXISTS (
                          SELECT 1 FROM messages m
                          WHERE m.conversation_id=?
                            AND (
                              m.citation=ma.citation
                              OR EXISTS (
                                SELECT 1 FROM media_asset_links l
                                WHERE l.asset_id=ma.asset_id
                                  AND l.accepted=1
                                  AND l.source_citation=m.citation
                              )
                            )
                        )"""
                    params.append(conversation_id)
                params.append(max(0, int(limit)))
                rows = conn.execute(
                    f"""SELECT ma.*
                        FROM media_assets ma
                        WHERE ma.modality='voice'
                          {conversation_filter}
                        ORDER BY ma.updated_at, ma.asset_id
                        LIMIT ?""",
                    params,
                )
                for row in rows:
                    if _private_message_citation_for_asset(conn, row) is None:
                        continue
                    asset = dict(row)
                    candidates.append(asset)
                    existing = None
                    if store._table_exists(conn, 'transcripts'):
                        existing = conn.execute(
                            """SELECT 1
                                 FROM transcripts t
                                 JOIN provider_jobs pj ON pj.job_id=t.job_id
                                 JOIN media_assets current ON current.asset_id=t.asset_id
                                WHERE t.asset_id=? AND t.status='active'
                                  AND pj.provider=? AND pj.model=? AND pj.status='completed'
                                  AND pj.request_hash=current.content_hash
                                LIMIT 1""",
                            (asset['asset_id'], CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
                        ).fetchone()
                    if existing is None:
                        pending += 1
    return {
        'ok': True,
        'conversation_id': conversation_id,
        'pending': pending,
        'candidates': len(candidates),
        'estimated_seconds': pending * 10,
        'provider': CLOUD_ASR_PROVIDER_NAME,
        'model': CLOUD_ASR_MODEL_ID,
        'cloud_only': True,
        'cloud_calls_made': False,
        'raw_content_included': False,
    }


def _empty_media_status_payload() -> dict[str, Any]:
    return {
        'ok': True,
        'media_assets': {'voice': 0, 'image': 0},
        'coverage': {
            'voice_transcripts': {'done': 0, 'total': 0, 'ratio': 1.0},
            'image_observations': {'done': 0, 'total': 0, 'ratio': 1.0},
            'image_captions': {'done': 0, 'total': 0, 'ratio': 1.0},
        },
        'queue': {},
        'backlog': 0,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def media_status_payload(vault_root: str | Path | None = None) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    store = SQLiteStore(cfg.paths.sqlite_path)
    if not store.path.exists():
        return _empty_media_status_payload()
    db_uri = store.path.resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(db_uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row

        def count(table: str, where: str = '1=1', params: tuple[Any, ...] = ()) -> int:
            if not store._table_exists(conn, table):
                return 0
            return int(conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {where}', params).fetchone()[0])

        queue: dict[str, dict[str, int]] = {}
        backlog = 0
        if store._table_exists(conn, 'media_jobs'):
            for row in conn.execute('SELECT job_type,status,COUNT(*) AS n FROM media_jobs GROUP BY job_type,status'):
                n = int(row['n'])
                queue.setdefault(row['job_type'], {})[row['status']] = n
                if row['status'] in RETRYABLE_STATUSES:
                    backlog += n
            backlog += int(conn.execute(
                "SELECT COUNT(*) FROM media_jobs WHERE status='running' AND julianday(updated_at) < julianday('now','-30 minutes')"
            ).fetchone()[0])
        total_voice = count('media_assets', "modality='voice'")
        total_images = count('media_assets', "modality='image'")
        transcribed = 0
        if all(store._table_exists(conn, table) for table in ('transcripts', 'provider_jobs', 'media_assets')):
            transcribed = int(conn.execute(
                """SELECT COUNT(DISTINCT t.asset_id)
                     FROM transcripts t
                     JOIN provider_jobs pj ON pj.job_id=t.job_id
                     JOIN media_assets ma ON ma.asset_id=t.asset_id
                    WHERE ma.modality='voice' AND t.status='active'
                      AND pj.provider=? AND pj.model=? AND pj.status='completed'
                      AND pj.request_hash=ma.content_hash""",
                (CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
            ).fetchone()[0])
        observed = int(conn.execute('SELECT COUNT(DISTINCT asset_id) FROM image_observations').fetchone()[0]) if store._table_exists(conn, 'image_observations') else 0
        captioned = int(conn.execute("SELECT COUNT(DISTINCT asset_id) FROM image_observations WHERE TRIM(COALESCE(caption, '')) <> ''").fetchone()[0]) if store._table_exists(conn, 'image_observations') else 0
    return {
        'ok': True,
        'media_assets': {'voice': total_voice, 'image': total_images},
        'coverage': {
            'voice_transcripts': {'done': transcribed, 'total': total_voice, 'ratio': round(transcribed / total_voice, 4) if total_voice else 1.0},
            'image_observations': {'done': observed, 'total': total_images, 'ratio': round(observed / total_images, 4) if total_images else 1.0},
            'image_captions': {'done': captioned, 'total': total_images, 'ratio': round(captioned / total_images, 4) if total_images else 1.0},
        },
        'queue': queue,
        'backlog': backlog,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def _empty_voice_budget_report(
    provider: ASRProvider | None,
    *,
    budget: int,
    started: float,
    conversation_id: str | None,
) -> dict[str, Any]:
    report = MediaBudgetReport(
        ok=True,
        action='media_transcribe',
        requested_budget=max(0, int(budget)),
        candidates=0,
        processed=0,
        completed=0,
        idempotent=0,
        skipped=0,
        failed=0,
        avg_item_ms=0.0,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        errors={},
        provider={
            'name': getattr(provider, 'name', 'cloud-asr-approval-required'),
            'model': getattr(provider, 'model_name', ''),
            'resource_id': getattr(provider, 'resource_id', ''),
            'local_only': False,
            'cloud_only': True,
        },
    ).to_dict()
    if conversation_id:
        report['conversation_id'] = conversation_id
    return report


def default_macos_vision_provider(*, languages: list[str] | None = None) -> MacOSVisionOCRProvider:
    return MacOSVisionOCRProvider(languages=languages)


def default_local_vlm_caption_provider(
    cfg: VaultConfig,
    *,
    model_id: str | None = None,
    model_cache: str | Path | None = None,
) -> MlxVLMCaptionProvider:
    cache = Path(model_cache).expanduser() if model_cache is not None else cfg.root / 'models' / 'local-vlm'
    return MlxVLMCaptionProvider(
        model_id=model_id or 'mlx-community/Qwen2.5-VL-3B-Instruct-4bit',
        cache_dir=cache,
    )


@mutation_entrypoint('media_transcribe')
def run_voice_transcription_budget(
    vault_root: str | Path | None = None,
    *,
    budget: int,
    provider: ASRProvider | None = None,
    model_size: str = 'small',
    model_cache: str | Path | None = None,
    device: str = 'auto',
    compute_type: str = 'auto',
    language: str = 'zh',
    conversation_id: str | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('voice inference cannot run inside an outer writer session')
    cfg.ensure()
    # Retain historical tuning arguments for CLI compatibility, but never use
    # them to construct a local model. Voice execution is cloud-only.
    _ = (model_size, model_cache, device, compute_type, language)
    resolved_provider = provider
    if int(budget) <= 0:
        record_vault_mutation_noop(operation='media_transcribe')
        return _empty_voice_budget_report(
            resolved_provider,
            budget=budget,
            started=started,
            conversation_id=conversation_id,
        )
    store = SQLiteStore(cfg.paths.sqlite_path)

    # Normal runs consume the durable queue co-committed by import. A scoped
    # conversation request may enqueue only that bounded conversation; the
    # old unscoped all-assets repair scan is intentionally not on this path.
    if conversation_id:
        asset_ids = _conversation_voice_asset_ids(
            SQLiteStore(cfg.paths.sqlite_path, readonly=True),
            conversation_id,
        )
        with coordinated_vault_mutation(cfg, operation='media_transcribe'):
            enqueue_media_jobs(store, modalities={'voice'}, asset_ids=asset_ids)
    candidates = _job_candidates(
        SQLiteStore(cfg.paths.sqlite_path, readonly=True),
        job_type=VOICE_JOB_TYPE,
        limit=max(0, int(budget)),
        conversation_id=conversation_id,
    )

    if not candidates:
        record_vault_mutation_noop(operation='media_transcribe')
        return _empty_voice_budget_report(
            resolved_provider,
            budget=budget,
            started=started,
            conversation_id=conversation_id,
        )
    if resolved_provider is None:
        record_vault_mutation_noop(operation='media_transcribe')
        report = MediaBudgetReport(
            ok=False,
            action='media_transcribe',
            requested_budget=max(0, int(budget)),
            candidates=len(candidates),
            processed=0,
            completed=0,
            idempotent=0,
            skipped=len(candidates),
            failed=0,
            avg_item_ms=0.0,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            errors={'cloud_asr_approval_required': len(candidates)},
            provider={
                'name': 'cloud-asr-approval-required',
                'model': '',
                'resource_id': '',
                'local_only': False,
                'cloud_only': True,
            },
        ).to_dict()
        if conversation_id:
            report['conversation_id'] = conversation_id
        return report
    if getattr(resolved_provider, 'name', None) == LOCAL_ASR_PROVIDER_NAME:
        raise RuntimeError('local_asr_forbidden')
    prepare = getattr(resolved_provider, 'prepare', None)
    if callable(prepare):
        prepare()

    errors: dict[str, int] = {}
    durations: list[float] = []
    completed = idempotent = skipped = failed = 0
    for asset in candidates:
        item_started = time.perf_counter()
        result = ensure_voice_transcript(
            cfg.root,
            citation=str(asset['citation']),
            allow_local_asr=False,
            allow_cloud_asr=getattr(resolved_provider, 'egress_kind', None) == 'cloud_asr_upload',
            provider=resolved_provider,
        )
        status = str(result.get('status') or 'failed')
        if status == 'completed':
            if result.get('idempotent'):
                idempotent += 1
            else:
                completed += 1
        elif status == 'cached':
            idempotent += 1
        elif status in {'in_progress', 'skipped', 'media_unavailable', 'unavailable', 'awaiting_approval'}:
            skipped += 1
            code = str(result.get('reason') or status)
            errors[code] = errors.get(code, 0) + 1
        else:
            failed += 1
            code = str(result.get('error_code') or result.get('reason') or status)
            errors[code] = errors.get(code, 0) + 1
        durations.append((time.perf_counter() - item_started) * 1000)

    report = MediaBudgetReport(
        ok=failed == 0,
        action='media_transcribe',
        requested_budget=max(0, int(budget)),
        candidates=len(candidates),
        processed=len(candidates),
        completed=completed,
        idempotent=idempotent,
        skipped=skipped,
        failed=failed,
        avg_item_ms=round(sum(durations) / len(durations), 3) if durations else 0.0,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        errors=dict(sorted(errors.items())),
        provider={
            'name': resolved_provider.name,
            'model': getattr(resolved_provider, 'model_name', ''),
            'resource_id': getattr(resolved_provider, 'resource_id', ''),
            'local_only': False,
            'cloud_only': True,
        },
        cloud_calls_made=getattr(resolved_provider, 'egress_kind', None) == 'cloud_asr_upload' and completed > 0,
    ).to_dict()
    if conversation_id:
        report['conversation_id'] = conversation_id
    return report

def _existing_image_derivative(store: SQLiteStore, cfg: VaultConfig, asset_id: str) -> Path | None:
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_decode_results'):
            return None
        row = conn.execute(
            """SELECT derivative_ref
               FROM media_decode_results
               WHERE asset_id=?
                 AND status IN ('copied', 'decoded')
                 AND COALESCE(derivative_ref, '') <> ''
               ORDER BY created_at DESC
               LIMIT 1""",
            (asset_id,),
        ).fetchone()
    if not row:
        return None
    path = cfg.root / str(row['derivative_ref'])
    if path.exists() and path_is_under(path, cfg.root):
        return path
    return None


def _prepare_image_path(
    cfg: VaultConfig,
    store: SQLiteStore,
    repo: MultimodalRepository,
    asset: dict[str, Any],
    *,
    pipeline_name: str,
) -> tuple[Path | None, str | None]:
    asset_id = str(asset['asset_id'])
    path_ref = str(asset.get('path_ref') or '')
    existing = _existing_image_derivative(store, cfg, asset_id)
    if existing is not None:
        return existing, None
    media_path = _path_from_ref(cfg, path_ref) if path_ref else Path()
    if not path_ref or not media_path.exists() or not path_is_under(media_path, cfg.root):
        with store.connect() as conn:
            full_asset = conn.execute('SELECT * FROM media_assets WHERE asset_id=?', (asset_id,)).fetchone()
        materialized = materialize_media_asset(
            cfg,
            store,
            full_asset,
            citation=str(asset.get('citation') or ''),
            publish=False,
        ) if full_asset is not None else None
        if materialized is None or not materialized.ok:
            return None, materialized.reason if materialized is not None else 'media_asset_not_found'
        with coordinated_vault_mutation(cfg, operation='media_observe'):
            published = publish_materialization_result(
                store,
                materialized,
                expected_path_ref=path_ref,
            )
        if not published:
            with store.connect() as conn:
                current = conn.execute('SELECT path_ref FROM media_assets WHERE asset_id=?', (asset_id,)).fetchone()
            if current is None or not current['path_ref']:
                return None, 'materialization_publish_conflict'
            path_ref = str(current['path_ref'])
        else:
            path_ref = str(materialized.path_ref or '')
        media_path = _path_from_ref(cfg, path_ref)
    resolved = resolve_image_file(media_path, cfg.root, asset_id=asset_id)
    with coordinated_vault_mutation(cfg, operation='media_observe'):
        repo.record_decode_result(
            decode_id=_stable('decode', f'image:{asset_id}:{resolved.input_hash or path_ref}'),
            asset_id=asset_id,
            status=resolved.status,
            wrapper_type=resolved.wrapper_type,
            input_hash=resolved.input_hash,
            output_hash=resolved.output_hash,
            derivative_ref=resolved.derivative_ref,
            error_code=resolved.error_code,
            metadata={'local_pipeline': pipeline_name, 'image_type': resolved.image_type},
        )
    if resolved.derivative_ref is None or resolved.status not in {'copied', 'decoded'}:
        return None, resolved.error_code or resolved.status
    return cfg.root / resolved.derivative_ref, None


def _image_caption_candidates(store: SQLiteStore, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    ensure_media_jobs(store)
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_assets'):
            return []
        rows = conn.execute(
            """SELECT ma.asset_id, ma.citation, ma.path_ref, ma.cache_state
               FROM media_assets ma
               WHERE ma.modality='image'
                 AND (
                   NOT EXISTS (SELECT 1 FROM media_asset_links l WHERE l.asset_id=ma.asset_id)
                   OR EXISTS (SELECT 1 FROM media_asset_links l WHERE l.asset_id=ma.asset_id AND l.accepted=1)
                 )
                 AND NOT EXISTS (
                   SELECT 1 FROM image_observations io
                   WHERE io.asset_id=ma.asset_id
                     AND TRIM(COALESCE(io.caption, '')) <> ''
                 )
               ORDER BY CASE
                          WHEN COALESCE(ma.path_ref, '') <> '' AND ma.cache_state NOT IN ('missing_local_cache', 'metadata_only') THEN 0
                          WHEN COALESCE(ma.path_ref, '') <> '' THEN 1
                          ELSE 2
                        END,
                        ma.updated_at, ma.asset_id
               LIMIT ?""",
            (int(limit),),
        )
        return [dict(row) for row in rows]


def _image_observation_candidate_ids(store: SQLiteStore, *, limit: int) -> list[str]:
    """Discover a bounded image batch without holding the Vault writer."""

    if limit <= 0:
        return []
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_assets'):
            return []
        return [
            str(row['asset_id'])
            for row in conn.execute(
                """SELECT ma.asset_id
                     FROM media_assets ma
                    WHERE ma.modality='image'
                      AND (NOT EXISTS (SELECT 1 FROM media_asset_links l WHERE l.asset_id=ma.asset_id)
                           OR EXISTS (SELECT 1 FROM media_asset_links l WHERE l.asset_id=ma.asset_id AND l.accepted=1))
                      AND NOT EXISTS (
                          SELECT 1 FROM image_observations io
                           WHERE io.asset_id=ma.asset_id
                             AND TRIM(COALESCE(io.visible_text,''))<>''
                      )
                    ORDER BY CASE
                               WHEN COALESCE(ma.path_ref,'')<>'' AND ma.cache_state NOT IN ('missing_local_cache','metadata_only') THEN 0
                               WHEN COALESCE(ma.path_ref,'')<>'' THEN 1 ELSE 2 END,
                             ma.updated_at,ma.asset_id
                    LIMIT ?""",
                (int(limit),),
            )
        ]


@mutation_entrypoint('media_observe')
def run_image_caption_budget(
    vault_root: str | Path | None = None,
    *,
    budget: int = 100,
    provider: LocalVLMCaptionProvider | None = None,
    model_id: str | None = None,
    model_cache: str | Path | None = None,
    include_images: bool = False,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('image provider work cannot run inside an outer writer session')
    cfg.ensure()
    store = SQLiteStore(cfg.paths.sqlite_path)
    repo = MultimodalRepository(store)
    with coordinated_vault_mutation(cfg, operation='media_observe'):
        store.initialize()
    provider = provider or default_local_vlm_caption_provider(cfg, model_id=model_id, model_cache=model_cache)
    if not include_images:
        elapsed_ms = (time.perf_counter() - started) * 1000
        report = MediaBudgetReport(
            ok=True,
            action='media_caption_disabled',
            requested_budget=max(0, int(budget)),
            candidates=0,
            processed=0,
            completed=0,
            idempotent=0,
            skipped=0,
            failed=0,
            avg_item_ms=0.0,
            elapsed_ms=round(elapsed_ms, 3),
            errors={},
            provider={
                'name': provider.name,
                'model': getattr(provider, 'model', ''),
                'resource_id': getattr(provider, 'resource_id', ''),
                'local_only': True,
            },
        ).to_dict()
        report['disabled_reason'] = IMAGE_PRECOMPUTE_DISABLED_REASON
        return report
    candidates = _image_caption_candidates(store, limit=max(0, int(budget)))
    errors: dict[str, int] = {}
    durations: list[float] = []
    completed = idempotent = skipped = failed = 0

    for asset in candidates:
        item_start = time.perf_counter()
        asset_id = str(asset['asset_id'])
        citation = str(asset['citation'])
        with store.connect() as conn:
            existing = conn.execute("SELECT 1 FROM image_observations WHERE asset_id=? AND TRIM(COALESCE(caption, '')) <> '' LIMIT 1", (asset_id,)).fetchone()
        if existing:
            idempotent += 1
            durations.append((time.perf_counter() - item_start) * 1000)
            continue
        image_path, error_code = _prepare_image_path(cfg, store, repo, asset, pipeline_name='local_vlm_caption')
        if image_path is None:
            skipped += 1
            code = error_code or 'missing_local_cache'
            errors[code] = errors.get(code, 0) + 1
            durations.append((time.perf_counter() - item_start) * 1000)
            continue
        result = run_image_caption_job(
            repo,
            asset_id=asset_id,
            image_path=image_path,
            provider=provider,
            citation=citation,
            mutation_context=lambda: coordinated_vault_mutation(cfg, operation='media_observe'),
        )
        if result.get('status') == 'completed':
            if result.get('idempotent'):
                idempotent += 1
            else:
                completed += 1
        else:
            failed += 1
            code = str(result.get('error_code') or result.get('status') or 'failed')
            errors[code] = errors.get(code, 0) + 1
        durations.append((time.perf_counter() - item_start) * 1000)

    elapsed_ms = (time.perf_counter() - started) * 1000
    report = MediaBudgetReport(
        ok=failed == 0,
        action='media_caption',
        requested_budget=max(0, int(budget)),
        candidates=len(candidates),
        processed=len(candidates),
        completed=completed,
        idempotent=idempotent,
        skipped=skipped,
        failed=failed,
        avg_item_ms=round(sum(durations) / len(durations), 3) if durations else 0.0,
        elapsed_ms=round(elapsed_ms, 3),
        errors=dict(sorted(errors.items())),
        provider={
            'name': provider.name,
            'model': getattr(provider, 'model', ''),
            'resource_id': getattr(provider, 'resource_id', ''),
            'local_only': True,
        },
    )
    return report.to_dict()


@mutation_entrypoint('media_observe')
def run_image_observation_budget(
    vault_root: str | Path | None = None,
    *,
    budget: int,
    provider: VisionProvider | None = None,
    languages: list[str] | None = None,
    caption: bool = False,
    caption_budget: int = 100,
    caption_provider: LocalVLMCaptionProvider | None = None,
    caption_model_id: str | None = None,
    caption_model_cache: str | Path | None = None,
    include_images: bool = False,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('image provider work cannot run inside an outer writer session')
    cfg.ensure()
    store = SQLiteStore(cfg.paths.sqlite_path)
    repo = MultimodalRepository(store)
    with coordinated_vault_mutation(cfg, operation='media_observe'):
        store.initialize()
    image_candidate_ids = (
        _image_observation_candidate_ids(store, limit=max(0, int(budget)))
        if include_images else []
    )
    with coordinated_vault_mutation(cfg, operation='media_observe'):
        queue_report = enqueue_media_jobs(
            store,
            modalities={'image'},
            include_images=include_images,
            asset_ids=image_candidate_ids,
        )
    provider = provider or default_macos_vision_provider(languages=languages)
    if not include_images:
        elapsed_ms = (time.perf_counter() - started) * 1000
        report = MediaBudgetReport(
            ok=True,
            action='media_observe_disabled',
            requested_budget=max(0, int(budget)),
            candidates=0,
            processed=0,
            completed=0,
            idempotent=0,
            skipped=0,
            failed=0,
            avg_item_ms=0.0,
            elapsed_ms=round(elapsed_ms, 3),
            errors={},
            provider={
                'name': provider.name,
                'model': getattr(provider, 'model', ''),
                'local_only': True,
            },
        ).to_dict()
        report['disabled_reason'] = IMAGE_PRECOMPUTE_DISABLED_REASON
        report['queue'] = queue_report
        report['caption_enabled'] = False
        report['caption_budget'] = 0
        report['caption'] = None
        return report
    candidates = _job_candidates(store, job_type=IMAGE_JOB_TYPE, limit=max(0, int(budget)))
    errors: dict[str, int] = {}
    durations: list[float] = []
    completed = idempotent = skipped = failed = 0

    for asset in candidates:
        item_start = time.perf_counter()
        asset_id = str(asset['asset_id'])
        job_id = str(asset['job_id'])
        citation = str(asset['citation'])
        with store.connect() as conn:
            existing = conn.execute("SELECT 1 FROM image_observations WHERE asset_id=? AND TRIM(COALESCE(visible_text, '')) <> '' LIMIT 1", (asset_id,)).fetchone()
        if existing:
            idempotent += 1
            with coordinated_vault_mutation(cfg, operation='media_observe'):
                _set_asset_processing_state(store, asset_id, 'done')
                _mark_media_job(store, job_id, 'done', duration_ms=(time.perf_counter() - item_start) * 1000)
            durations.append((time.perf_counter() - item_start) * 1000)
            continue
        image_path, error_code = _prepare_image_path(cfg, store, repo, asset, pipeline_name='macos_vision')
        if image_path is None:
            code = error_code or 'missing_local_cache'
            errors[code] = errors.get(code, 0) + 1
            if code in {'missing_local_cache', 'outside_vault_media_path'}:
                skipped += 1
                with coordinated_vault_mutation(cfg, operation='media_observe'):
                    _set_asset_processing_state(store, asset_id, 'skipped')
                    _mark_media_job(store, job_id, 'skipped', error_code=code)
            else:
                failed += 1
                with coordinated_vault_mutation(cfg, operation='media_observe'):
                    _set_asset_processing_state(store, asset_id, 'failed')
                    _mark_media_job(store, job_id, 'failed', error_code=code, increment_retry=True, duration_ms=(time.perf_counter() - item_start) * 1000)
            continue
        result = run_image_observation_job(
            repo,
            asset_id=asset_id,
            image_path=image_path,
            provider=provider,
            citation=citation,
            mutation_context=lambda: coordinated_vault_mutation(cfg, operation='media_observe'),
        )
        if result.get('status') == 'completed':
            if result.get('idempotent'):
                idempotent += 1
            else:
                completed += 1
            with coordinated_vault_mutation(cfg, operation='media_observe'):
                _set_asset_processing_state(store, asset_id, 'done')
                _mark_media_job(store, job_id, 'done', duration_ms=(time.perf_counter() - item_start) * 1000)
        else:
            failed += 1
            code = str(result.get('error_code') or result.get('status') or 'failed')
            errors[code] = errors.get(code, 0) + 1
            with coordinated_vault_mutation(cfg, operation='media_observe'):
                _set_asset_processing_state(store, asset_id, 'failed')
                _mark_media_job(store, job_id, 'failed', error_code=code, increment_retry=True, duration_ms=(time.perf_counter() - item_start) * 1000)
        durations.append((time.perf_counter() - item_start) * 1000)

    elapsed_ms = (time.perf_counter() - started) * 1000
    report = MediaBudgetReport(
        ok=failed == 0,
        action='media_observe',
        requested_budget=max(0, int(budget)),
        candidates=len(candidates),
        processed=len(candidates),
        completed=completed,
        idempotent=idempotent,
        skipped=skipped,
        failed=failed,
        avg_item_ms=round(sum(durations) / len(durations), 3) if durations else 0.0,
        elapsed_ms=round(elapsed_ms, 3),
        errors=dict(sorted(errors.items())),
        provider={
            'name': provider.name,
            'model': getattr(provider, 'model', ''),
            'local_only': True,
        },
    )
    data = report.to_dict()
    data['caption_enabled'] = bool(caption)
    data['caption_budget'] = max(0, int(caption_budget))
    data['caption'] = None
    if caption:
        caption_report = run_image_caption_budget(
            cfg.root,
            budget=max(0, int(caption_budget)),
            provider=caption_provider,
            model_id=caption_model_id,
            model_cache=caption_model_cache,
            include_images=True,
        )
        data['caption'] = caption_report
        data['ok'] = bool(data['ok'] and caption_report.get('ok', False))
    return data


@mutation_entrypoint('maintain')
def run_media_maintenance(
    vault_root: str | Path | None = None,
    *,
    voice_budget: int = 50,
    image_budget: int = 0,
    caption: bool = False,
    caption_budget: int = 0,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('media provider work cannot run inside an outer writer session')
    # This orchestrator has no direct database mutation.  Child pipelines own
    # their short claims/commits; do not wrap model/provider work in maintain.
    record_vault_mutation_noop(operation='maintain')
    voice = run_voice_transcription_budget(
        vault_root,
        budget=max(0, int(voice_budget)),
    ) if voice_budget else None
    image = run_image_observation_budget(
        vault_root,
        budget=max(0, int(image_budget)),
        caption=bool(caption),
        caption_budget=max(0, int(caption_budget)),
        include_images=True,
    ) if image_budget or (caption and caption_budget) else None
    return {
        'status': 'completed',
        'voice': voice,
        'image': image,
        'backlog': media_status_payload(vault_root).get('queue', {}),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
