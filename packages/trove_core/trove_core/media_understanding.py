from __future__ import annotations

from contextlib import closing

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from trove_core.media_fetch import _mime, _path_from_ref
from trove_core.store.repositories import MediaUnderstandingRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint, record_vault_mutation_noop
from trove_core.wechat.media.dat_decoder import decode_wechat_dat_file

CAPTION_MAX_CHARS = 4000
VISIBLE_TEXT_MAX_CHARS = 20000
STRUCTURED_LIST_MAX_ITEMS = 100
STRUCTURED_JSON_MAX_BYTES = 100_000
STRUCTURED_JSON_MAX_DEPTH = 3
_MEDIA_FRAGMENT_RE = re.compile(r'#(?:image|video)-\d+\Z')
_MEDIA_CHILD_LIKE = ('#image-%', '#video-%')

_SCALAR_TYPES = (str, int, float, bool, type(None))
_STRUCTURED_SCHEMAS: dict[str, dict[str, Any]] = {
    'objects': {
        'label': str,
        'type': str,
        'category': str,
        'description': str,
        'text': str,
        'count': int,
        'confidence': float,
        'attributes': dict,
    },
    'business_signals': {
        'type': str,
        'text': str,
        'label': str,
        'description': str,
        'value': str,
        'severity': str,
        'currency': str,
        'amount': float,
        'confidence': float,
    },
    'keyframes': {
        'time_seconds': float,
        'timestamp': str,
        'description': str,
        'caption': str,
        'visible_text': str,
        'confidence': float,
    },
}


def _json_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + (max((_json_depth(v) for v in value.values()), default=0))
    if isinstance(value, list):
        return 1 + (max((_json_depth(v) for v in value), default=0))
    return 0


def _validate_structured_item(item: dict[str, Any], *, field: str) -> dict[str, Any]:
    schema = _STRUCTURED_SCHEMAS[field]
    if _json_depth(item) > STRUCTURED_JSON_MAX_DEPTH:
        raise ValueError(f'{field} item nesting is too deep')
    out: dict[str, Any] = {}
    for key, value in item.items():
        if key not in schema:
            raise ValueError(f'{field} contains unsupported field: {key}')
        expected = schema[key]
        if expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{field}.{key} must be a number')
            out[key] = float(value)
            continue
        if expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{field}.{key} must be an integer')
            out[key] = value
            continue
        if expected is str:
            if not isinstance(value, str):
                raise ValueError(f'{field}.{key} must be a string')
            out[key] = value
            continue
        if expected is dict:
            if not isinstance(value, dict):
                raise ValueError(f'{field}.{key} must be an object')
            if any(not isinstance(k, str) or not isinstance(v, _SCALAR_TYPES) for k, v in value.items()):
                raise ValueError(f'{field}.{key} must contain only scalar values')
            out[key] = dict(value)
            continue
        raise ValueError(f'{field}.{key} has unsupported schema')
    return out


def _parse_json_list(value: Any, *, field: str) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if value == '':
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{field} must be valid JSON') from exc
    if not isinstance(parsed, list):
        raise ValueError(f'{field} must be a JSON array')
    if len(parsed) > STRUCTURED_LIST_MAX_ITEMS:
        raise ValueError(f'{field} has too many items')
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f'{field} items must be JSON objects')
        out.append(_validate_structured_item(item, field=field))
    encoded = json.dumps(out, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode('utf-8')) > STRUCTURED_JSON_MAX_BYTES:
        raise ValueError(f'{field} JSON is too large')
    return out


def _validate_text(value: Any, *, field: str, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value or '')
    if len(text) > limit:
        raise ValueError(f'{field} exceeds {limit} characters')
    return text


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _escape_like(value: str) -> str:
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def _media_citation_mode(citation: str) -> str:
    text = str(citation or '').strip()
    if not text:
        return 'invalid'
    if _MEDIA_FRAGMENT_RE.search(text):
        return 'media'
    if '#' in text:
        return 'invalid'
    return 'parent'


def _unique_asset_rows(rows: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for row in rows:
        if row is None:
            continue
        asset_id = str(row['asset_id'])
        if asset_id in seen:
            continue
        seen.add(asset_id)
        out.append(row)
    return out


def _asset_rows_for_exact_citation(conn: Any, store: SQLiteStore, citation: str) -> list[Any]:
    if not store._table_exists(conn, 'media_assets'):
        return []
    rows: list[Any] = []
    if store._table_exists(conn, 'image_observations'):
        rows.extend(conn.execute(
            """SELECT ma.*
                 FROM image_observations io
                 JOIN media_assets ma ON ma.asset_id=io.asset_id
                 WHERE ma.modality IN ('image','video') AND io.citation=?
                 ORDER BY ma.updated_at DESC""",
            (citation,),
        ).fetchall())
    if store._table_exists(conn, 'media_asset_links'):
        rows.extend(conn.execute(
            """SELECT ma.*
                 FROM media_asset_links l
                 JOIN media_assets ma ON ma.asset_id=l.asset_id
                 WHERE ma.modality IN ('image','video') AND l.accepted=1 AND l.source_citation=?
                 ORDER BY ma.updated_at DESC""",
            (citation,),
        ).fetchall())
    rows.extend(conn.execute(
        """SELECT *
             FROM media_assets
             WHERE modality IN ('image','video') AND citation=?
             ORDER BY updated_at DESC""",
        (citation,),
    ).fetchall())
    try:
        rows.extend(conn.execute(
            """SELECT *
                 FROM media_assets
                 WHERE modality IN ('image','video')
                   AND json_extract(metadata_json, '$.message_citation')=?
                 ORDER BY updated_at DESC""",
            (citation,),
        ).fetchall())
    except Exception:
        pass
    return _unique_asset_rows(rows)


def _asset_rows_for_parent_citation(conn: Any, store: SQLiteStore, citation: str) -> list[Any]:
    if not store._table_exists(conn, 'media_assets'):
        return []
    escaped = _escape_like(citation)
    child_patterns = tuple(escaped + suffix for suffix in _MEDIA_CHILD_LIKE)
    rows: list[Any] = []
    if store._table_exists(conn, 'image_observations'):
        rows.extend(conn.execute(
            """SELECT ma.*
                 FROM image_observations io
                 JOIN media_assets ma ON ma.asset_id=io.asset_id
                 WHERE ma.modality IN ('image','video')
                   AND (io.citation=? OR io.citation LIKE ? ESCAPE '\\' OR io.citation LIKE ? ESCAPE '\\')
                 ORDER BY ma.updated_at DESC""",
            (citation, *child_patterns),
        ).fetchall())
    if store._table_exists(conn, 'media_asset_links'):
        rows.extend(conn.execute(
            """SELECT ma.*
                 FROM media_asset_links l
                 JOIN media_assets ma ON ma.asset_id=l.asset_id
                 WHERE ma.modality IN ('image','video') AND l.accepted=1
                   AND (l.source_citation=? OR l.source_citation LIKE ? ESCAPE '\\' OR l.source_citation LIKE ? ESCAPE '\\')
                 ORDER BY ma.updated_at DESC""",
            (citation, *child_patterns),
        ).fetchall())
    rows.extend(conn.execute(
        """SELECT *
             FROM media_assets
             WHERE modality IN ('image','video')
               AND (citation=? OR citation LIKE ? ESCAPE '\\' OR citation LIKE ? ESCAPE '\\')
             ORDER BY updated_at DESC""",
        (citation, *child_patterns),
    ).fetchall())
    try:
        rows.extend(conn.execute(
            """SELECT *
                 FROM media_assets
                 WHERE modality IN ('image','video')
                   AND (json_extract(metadata_json, '$.message_citation')=?
                        OR json_extract(metadata_json, '$.message_citation') LIKE ? ESCAPE '\\'
                        OR json_extract(metadata_json, '$.message_citation') LIKE ? ESCAPE '\\')
                 ORDER BY updated_at DESC""",
            (citation, *child_patterns),
        ).fetchall())
    except Exception:
        pass
    return _unique_asset_rows(rows)


def _resolve_annotation_media_hash(vault_root: str | Path | None, citation: str) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    store = SQLiteStore(cfg.paths.sqlite_path)
    unavailable_base = {'raw_paths_included': False, 'raw_content_included': False}
    if not store.path.exists():
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'vault_index_missing'}
    mode = _media_citation_mode(citation)
    if mode == 'invalid':
        return unavailable_base | {'ok': False, 'code': 'invalid_media_citation', 'status': 'invalid_media_citation', 'reason': 'citation_must_be_media_or_unique_parent'}
    store.initialize()
    with store.connect() as conn:
        candidates = _asset_rows_for_exact_citation(conn, store, citation) if mode == 'media' else _asset_rows_for_parent_citation(conn, store, citation)
    if not candidates:
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'image_asset_not_found'}
    if len(candidates) > 1:
        return unavailable_base | {'ok': False, 'code': 'ambiguous_media_citation', 'status': 'ambiguous_media_citation', 'reason': 'parent_citation_matches_multiple_assets'}
    asset = candidates[0]
    path_ref = str(asset['path_ref'] or '')
    if not path_ref:
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'missing_local_cache', 'asset_id': asset['asset_id']}
    source = _path_from_ref(cfg, path_ref)
    if not source.exists():
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'missing_local_cache', 'asset_id': asset['asset_id']}
    if not path_is_under(source, cfg.root):
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'outside_vault_media_path', 'asset_id': asset['asset_id']}
    modality = str(asset['modality'] or 'image')
    if modality == 'video':
        content_sha256 = _sha256_file(source)
    elif source.suffix.lower() == '.dat':
        decoded = decode_wechat_dat_file(source)
        if decoded.output_bytes is None:
            return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': decoded.error_code or decoded.status, 'asset_id': asset['asset_id']}
        content_sha256 = _sha256_bytes(decoded.output_bytes)
    else:
        content_sha256 = _sha256_file(source)
    return {
        'ok': True,
        'code': 'ok',
        'status': 'available',
        'citation': citation,
        'asset_id': asset['asset_id'],
        'content_sha256': content_sha256,
        'mime': _mime(source),
        'modality': 'video' if modality == 'video' else 'image',
        'raw_paths_included': False,
        'raw_content_included': False,
    }


@mutation_entrypoint('media_annotate')
def annotate_media_understanding(
    vault_root: str | Path | None,
    *,
    citation: str,
    caption: str | None = None,
    visible_text: str | None = None,
    objects: Any = None,
    business_signals: Any = None,
    keyframes: Any = None,
    audio_transcript: str | None = None,
    confidence: float | None = None,
    model_id: str,
    prompt_version: str,
    replace: bool = False,
    expected_content_sha256: str | None = None,
    expected_asset_id: str | None = None,
    execution_location: str = 'local',
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    model_id = str(model_id or '').strip()
    prompt_version = str(prompt_version or '').strip()
    if not model_id:
        raise ValueError('model_id is required')
    if not prompt_version:
        raise ValueError('prompt_version is required')
    if execution_location != 'local':
        raise ValueError('agent annotation requires attested local execution')
    caption_text = _validate_text(caption, field='caption', limit=CAPTION_MAX_CHARS)
    visible_text_value = _validate_text(visible_text, field='visible_text', limit=VISIBLE_TEXT_MAX_CHARS)
    objects_list = _parse_json_list(objects, field='objects')
    business_signals_list = _parse_json_list(business_signals, field='business_signals')
    keyframes_list = _parse_json_list(keyframes, field='keyframes')
    audio_transcript_value = _validate_text(audio_transcript, field='audio_transcript', limit=VISIBLE_TEXT_MAX_CHARS)
    confidence_value = None if confidence is None else float(confidence or 0.0)
    if confidence_value is not None and (confidence_value < 0 or confidence_value > 1):
        raise ValueError('confidence must be between 0 and 1')

    fetched = _resolve_annotation_media_hash(vault_root, citation)
    if not fetched.get('ok'):
        record_vault_mutation_noop(operation='media_annotate')
        return {
            'ok': False,
            'code': fetched.get('code') or 'media_unavailable',
            'status': fetched.get('status') or 'media_unavailable',
            'reason': fetched.get('reason') or 'media_unavailable',
            'citation': citation,
            'raw_content_included': False,
        }
    content_sha256 = str(fetched.get('content_sha256') or '')
    expected_sha = str(expected_content_sha256 or '').lower().strip()
    if expected_sha and expected_sha != content_sha256:
        record_vault_mutation_noop(operation='media_annotate')
        return {
            'ok': False,
            'code': 'annotation_content_hash_mismatch',
            'status': 'rejected',
            'citation': citation,
            'raw_content_included': False,
        }
    if expected_asset_id is not None and str(expected_asset_id) != str(fetched.get('asset_id') or ''):
        record_vault_mutation_noop(operation='media_annotate')
        return {
            'ok': False,
            'code': 'annotation_asset_mismatch',
            'status': 'rejected',
            'citation': citation,
            'raw_content_included': False,
        }
    modality = 'video' if str(fetched.get('mime') or '').startswith('video/') else 'image'
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    with coordinated_vault_mutation(
        cfg,
        operation='media_annotate',
        write_session=write_session,
    ):
        repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
        row = repo.upsert_media_understanding(MediaUnderstandingRecord(
            content_sha256=content_sha256,
            modality=modality,
            caption=caption_text,
            visible_text=visible_text_value,
            objects=objects_list,
            business_signals=business_signals_list,
            keyframes=keyframes_list,
            audio_transcript=audio_transcript_value,
            model_id=model_id,
            prompt_version=prompt_version,
            confidence=confidence_value,
            origin='lazy_agent',
            source_citations=[citation],
            replace=bool(replace),
        ))
        projection = repo.project_media_understanding(
            asset_id=str(fetched.get('asset_id') or ''),
            citation=citation,
            content_sha256=content_sha256,
            model_id=str(row['model_id']),
            prompt_version=str(row['prompt_version']),
            caption=str(row['caption'] or ''),
            visible_text=str(row['visible_text'] or ''),
            objects=json.loads(row['objects_json']),
            business_signals=json.loads(row['business_signals_json']),
            confidence=float(row['confidence'] or 0),
        )
    source_citations = json.loads(row['source_citations_json'])
    history = json.loads(row['metadata_json']).get('history') or []
    return {
        'ok': True,
        'code': 'ok',
        'status': 'cached',
        'citation': citation,
        'asset_id': fetched.get('asset_id'),
        'content_sha256': content_sha256,
        'modality': row['modality'],
        'model_id': row['model_id'],
        'prompt_version': row['prompt_version'],
        'confidence': float(row['confidence'] or 0),
        'source_citations_count': len(source_citations),
        'history_count': len(history),
        'evidence_projection': {
            'observation_id': projection['observation_id'],
            'citation': projection['citation'],
            'status': projection['status'],
        },
        'propose_channel': {
            'tool': 'trove_observe_propose',
            'auto_proposed': False,
            'reason': 'annotation becomes cited derived evidence; personal facts still require the observation proposal and review lifecycle',
        },
        'raw_content_included': False,
    }


def _empty_understanding_status() -> dict[str, Any]:
    return {
        'ok': True,
        'total': 0,
        'active': 0,
        'stale': 0,
        'modality_distribution': {},
        'model_distribution': {},
        'total_fetch_hits': 0,
        'reused_content_hashes': 0,
        'raw_content_included': False,
    }


def _sqlite_readonly_uri(path: Path) -> str:
    return 'file:' + quote(str(path.resolve())) + '?mode=ro'


def media_understanding_status(vault_root: str | Path | None) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    path = cfg.paths.sqlite_path
    if not path.exists():
        return _empty_understanding_status()
    with closing(sqlite3.connect(_sqlite_readonly_uri(path), uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name='media_understanding' LIMIT 1"
        ).fetchone()
        if table is None:
            return _empty_understanding_status()
        total = int(conn.execute('SELECT COUNT(*) FROM media_understanding').fetchone()[0])
        active = int(conn.execute("SELECT COUNT(*) FROM media_understanding WHERE status='active'").fetchone()[0])
        stale = int(conn.execute("SELECT COUNT(*) FROM media_understanding WHERE status='stale'").fetchone()[0])
        modality_distribution = {
            str(row['modality']): int(row['count'])
            for row in conn.execute("SELECT modality, COUNT(*) AS count FROM media_understanding WHERE status='active' GROUP BY modality ORDER BY modality")
        }
        model_distribution = {
            f"{row['model_id']}@{row['prompt_version']}": int(row['count'])
            for row in conn.execute("SELECT model_id, prompt_version, COUNT(*) AS count FROM media_understanding WHERE status='active' GROUP BY model_id,prompt_version ORDER BY model_id,prompt_version")
        }
        total_fetch_hits = int(conn.execute("SELECT COALESCE(SUM(fetch_hit_count), 0) FROM media_understanding WHERE status='active'").fetchone()[0])
        reused_content_hashes = 0
        for row in conn.execute("SELECT source_citations_json FROM media_understanding WHERE status='active'"):
            try:
                citations = json.loads(row['source_citations_json'])
            except json.JSONDecodeError:
                citations = []
            if isinstance(citations, list) and len(set(str(c) for c in citations)) > 1:
                reused_content_hashes += 1
    return {
        'ok': True,
        'total': total,
        'active': active,
        'stale': stale,
        'modality_distribution': modality_distribution,
        'model_distribution': model_distribution,
        'total_fetch_hits': total_fetch_hits,
        'reused_content_hashes': reused_content_hashes,
        'raw_content_included': False,
    }


@mutation_entrypoint('media_invalidate')
def invalidate_media_understanding(
    vault_root: str | Path | None,
    *,
    content_sha256: str | None = None,
    model_id: str | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    sha = str(content_sha256 or '').lower().strip()
    model = str(model_id or '').strip()
    if sha and (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
        raise ValueError('content_sha256 must be a 64-character hex sha256')
    if not sha and not model:
        raise ValueError('invalidate requires content_sha256 or model_id')
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    with coordinated_vault_mutation(
        cfg,
        operation='media_invalidate',
        write_session=write_session,
    ):
        store = SQLiteStore(cfg.paths.sqlite_path)
        store.initialize()
        with store.connect() as conn:
            if sha:
                cursor = conn.execute("UPDATE media_understanding SET status='stale', updated_at=datetime('now') WHERE content_sha256=? AND status='active'", (sha,))
                mode = 'content_sha256'
            else:
                cursor = conn.execute("UPDATE media_understanding SET status='stale', updated_at=datetime('now') WHERE model_id=? AND status='active'", (model,))
                mode = 'model_id'
            conn.commit()
    return {
        'ok': True,
        'code': 'ok',
        'status': 'invalidated',
        'mode': mode,
        'invalidated': max(int(cursor.rowcount or 0), 0),
        'auto_rerun': False,
        'raw_content_included': False,
    }
