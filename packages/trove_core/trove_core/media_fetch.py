from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import shutil
import sqlite3
import struct
from typing import Any, Mapping

from trove_protocol.provider import Provider

from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.wechat.media.image_resolver import resolve_image_file
from trove_core.wechat.media.materializer import materialize_media_asset, publish_materialization_result
from trove_core.approvals import ApprovalGrant


MIME_BY_SUFFIX = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
    '.mp4': 'video/mp4',
    '.mov': 'video/quicktime',
    '.m4v': 'video/x-m4v',
}

VIDEO_SUFFIXES = {'.mp4', '.mov', '.m4v'}
PREVIEW_CACHE_MAX_BYTES = 512 * 1024 * 1024
PREVIEW_CACHE_MAX_FILES = 1000


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}'


def stage_media_via_provider(
    provider: Provider,
    *,
    asset_id: str,
    staging_path: str,
) -> Mapping[str, Any]:
    """Request bytes through staging metadata; JSON/base64 blobs are rejected."""
    result = provider.invoke('media', {'asset_id': asset_id, 'staging_path': staging_path})
    if (
        not isinstance(result, Mapping)
        or result.get('encoding') != 'staging_path'
        or result.get('blob_in_json') is not False
        or 'data' in result
        or type(result.get('size')) is not int
        or not isinstance(result.get('sha256'), str)
    ):
        raise RuntimeError('provider_media_staging_invalid')
    return result


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _is_sha256(value: str | None) -> bool:
    text = str(value or '').lower()
    return len(text) == 64 and all(c in '0123456789abcdef' for c in text)


def _json_list(value: str | None) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _understanding_payload(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    source_citations = _json_list(row['source_citations_json'])
    return {
        'content_sha256': row['content_sha256'],
        'modality': row['modality'],
        'caption': row['caption'],
        'visible_text': row['visible_text'],
        'objects': _json_list(row['objects_json']),
        'business_signals': _json_list(row['business_signals_json']),
        'keyframes': _json_list(row['keyframes_json']) if 'keyframes_json' in row.keys() else [],
        'audio_transcript': row['audio_transcript'] if 'audio_transcript' in row.keys() else '',
        'model_id': row['model_id'],
        'prompt_version': row['prompt_version'],
        'confidence': float(row['confidence'] or 0),
        'origin': row['origin'],
        'status': row['status'] if 'status' in row.keys() else 'active',
        'source_citations_count': len(source_citations),
        'fetch_hit_count': int(row['fetch_hit_count']) if 'fetch_hit_count' in row.keys() else 0,
    }


def _understanding_hit(conn: sqlite3.Connection, content_sha256: str, citation: str) -> sqlite3.Row | None:
    row = conn.execute('SELECT * FROM media_understanding WHERE content_sha256=? AND status=?', (content_sha256, 'active')).fetchone()
    if row is None:
        return None
    citations = [str(c) for c in _json_list(row['source_citations_json']) if str(c or '').strip()]
    if citation not in citations:
        citations.append(citation)
    conn.execute(
        'UPDATE media_understanding SET fetch_hit_count=fetch_hit_count+1, source_citations_json=?, updated_at=datetime("now") WHERE content_sha256=?',
        (json.dumps(citations, ensure_ascii=False, sort_keys=True), content_sha256),
    )
    conn.commit()
    return conn.execute('SELECT * FROM media_understanding WHERE content_sha256=? AND status=?', (content_sha256, 'active')).fetchone()


def _project_cached_understanding(
    cfg: VaultConfig,
    store: SQLiteStore,
    *,
    asset_id: str,
    citation: str,
    understanding: dict[str, Any] | None,
) -> bool:
    """Link one global content-hash understanding to this exact citation."""

    if understanding is None:
        return False
    with coordinated_vault_mutation(cfg, operation='media_fetch'):
        MultimodalRepository(store).project_media_understanding(
            asset_id=asset_id,
            citation=citation,
            content_sha256=str(understanding['content_sha256']),
            model_id=str(understanding['model_id']),
            prompt_version=str(understanding['prompt_version']),
            caption=str(understanding.get('caption') or ''),
            visible_text=str(understanding.get('visible_text') or ''),
            objects=list(understanding.get('objects') or []),
            business_signals=list(understanding.get('business_signals') or []),
            confidence=float(understanding.get('confidence') or 0.0),
        )
    return True


def _citation_variants(value: str | None) -> list[str]:
    text = str(value or '').strip()
    if not text:
        return []
    variants = [text]
    if '#chunk-' in text:
        variants.append(text.split('#chunk-', 1)[0])
    if '#image' in text:
        variants.append(text.split('#image', 1)[0])
    if '#' in text:
        variants.append(text.split('#', 1)[0])
    return list(dict.fromkeys(v for v in variants if v))


def _path_from_ref(cfg: VaultConfig, path_ref: str) -> Path:
    path = Path(path_ref).expanduser()
    return path if path.is_absolute() else cfg.root / path


def _mime(path: Path, image_type: str | None = None) -> str:
    if image_type:
        return MIME_BY_SUFFIX.get('.' + image_type.lower().lstrip('.'), f'image/{image_type.lower()}')
    return MIME_BY_SUFFIX.get(path.suffix.lower(), 'application/octet-stream')


def _preview_cache_lru(preview_dir: Path, *, keep: set[Path]) -> dict[str, Any]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in preview_dir.iterdir() if p.is_file()]
    total = 0
    entries: list[tuple[float, int, Path]] = []
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        total += st.st_size
        entries.append((st.st_mtime, st.st_size, path))
    removed = 0
    current_count = len(entries)
    keep_resolved = {p.resolve() for p in keep if p.exists()}
    entries.sort(key=lambda item: item[0])
    while (current_count > PREVIEW_CACHE_MAX_FILES or total > PREVIEW_CACHE_MAX_BYTES) and entries:
        _, size, path = entries.pop(0)
        try:
            if path.resolve() in keep_resolved:
                continue
            path.unlink()
            total -= size
            removed += 1
            current_count -= 1
        except OSError:
            continue
    return {
        'policy': 'lru',
        'max_files': PREVIEW_CACHE_MAX_FILES,
        'max_bytes': PREVIEW_CACHE_MAX_BYTES,
        'removed': removed,
    }


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) >= 24:
        return struct.unpack('>II', data[16:24])
    return None


def _gif_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:6] in {b'GIF87a', b'GIF89a'} and len(data) >= 10:
        return struct.unpack('<HH', data[6:10])
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b'\xff\xd8'):
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            return None
        seg_len = int.from_bytes(data[i:i + 2], 'big')
        if seg_len < 2 or i + seg_len > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF} and seg_len >= 7:
            height = int.from_bytes(data[i + 3:i + 5], 'big')
            width = int.from_bytes(data[i + 5:i + 7], 'big')
            return width, height
        i += seg_len
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        return None
    chunk = data[12:16]
    if chunk == b'VP8X' and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], 'little')
        height = 1 + int.from_bytes(data[27:30], 'little')
        return width, height
    if chunk == b'VP8L' and len(data) >= 25:
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return width, height
    return None


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        data = path.read_bytes()[:1024 * 1024]
    except OSError:
        return None, None
    for parser in (_png_dimensions, _gif_dimensions, _jpeg_dimensions, _webp_dimensions):
        dims = parser(data)
        if dims:
            return dims
    return None, None


def _asset_row_by_citation(conn: sqlite3.Connection, store: SQLiteStore, citation: str) -> sqlite3.Row | None:
    variants = _citation_variants(citation)
    if not variants or not store._table_exists(conn, 'media_assets'):
        return None
    if store._table_exists(conn, 'evidence_chunks'):
        for value in list(variants):
            row = conn.execute('SELECT parent_citation FROM evidence_chunks WHERE chunk_citation=? LIMIT 1', (value,)).fetchone()
            if row is not None:
                variants.extend(_citation_variants(row['parent_citation']))
    variants = list(dict.fromkeys(variants))
    placeholders = ','.join('?' for _ in variants)
    if store._table_exists(conn, 'image_observations'):
        row = conn.execute(
            f"""SELECT ma.*
                FROM image_observations io
                JOIN media_assets ma ON ma.asset_id=io.asset_id
                WHERE ma.modality IN ('image','video','file','attachment','document') AND io.citation IN ({placeholders})
                ORDER BY ma.updated_at DESC
                LIMIT 1""",
            variants,
        ).fetchone()
        if row is not None:
            return row
    if store._table_exists(conn, 'media_asset_links'):
        row = conn.execute(
            f"""SELECT ma.*
                FROM media_asset_links l
                JOIN media_assets ma ON ma.asset_id=l.asset_id
                WHERE ma.modality IN ('image','video','file','attachment','document') AND l.accepted=1 AND l.source_citation IN ({placeholders})
                ORDER BY ma.updated_at DESC
                LIMIT 1""",
            variants,
        ).fetchone()
        if row is not None:
            return row
    row = conn.execute(
        f"""SELECT *
            FROM media_assets
            WHERE modality IN ('image','video','file','attachment','document') AND citation IN ({placeholders})
            ORDER BY updated_at DESC
            LIMIT 1""",
        variants,
    ).fetchone()
    if row is not None:
        return row
    try:
        row = conn.execute(
            f"""SELECT *
                FROM media_assets
                WHERE modality IN ('image','video','file','attachment','document')
                  AND json_extract(metadata_json, '$.message_citation') IN ({placeholders})
                ORDER BY updated_at DESC
                LIMIT 1""",
            variants,
        ).fetchone()
    except sqlite3.DatabaseError:
        row = None
    return row


def _copy_video_preview(cfg: VaultConfig, asset: sqlite3.Row, source: Path, citation: str) -> dict[str, Any]:
    suffix = source.suffix.lower() if source.suffix.lower() in VIDEO_SUFFIXES else '.mp4'
    preview = cfg.root / 'media' / 'previews' / f"{_stable('preview', str(asset['asset_id']) + ':' + citation)}{suffix}"
    preview.parent.mkdir(parents=True, exist_ok=True)
    copy_needed = True
    if preview.exists():
        try:
            src_stat = source.stat()
            dst_stat = preview.stat()
            copy_needed = dst_stat.st_size != src_stat.st_size or dst_stat.st_mtime < src_stat.st_mtime
        except OSError:
            copy_needed = True
    if copy_needed and source.resolve() != preview.resolve():
        shutil.copyfile(source, preview)
    keyframe_paths: list[str] = []
    ffmpeg = shutil.which('ffmpeg')
    strategy = 'original_video'
    if ffmpeg:
        frame = preview.with_suffix('.keyframe-0.jpg')
        if frame.exists() and not copy_needed:
            keyframe_paths.append(str(frame))
            strategy = 'ffmpeg_keyframe_cached'
        else:
            try:
                subprocess.run(
                    [ffmpeg, '-y', '-nostdin', '-hide_banner', '-loglevel', 'error', '-ss', '0', '-i', str(preview), '-frames:v', '1', str(frame)],
                    check=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                if frame.exists():
                    keyframe_paths.append(str(frame))
                    strategy = 'ffmpeg_keyframe'
            except Exception:
                strategy = 'original_video_ffmpeg_failed'
    preview_cache = _preview_cache_lru(preview.parent, keep={preview, *(Path(p) for p in keyframe_paths)})
    return {
        'path': str(preview),
        'mime': _mime(preview),
        'width': None,
        'height': None,
        'preview': {
            'strategy': strategy,
            'keyframe_paths': keyframe_paths,
            'ffmpeg_available': bool(ffmpeg),
            'raw_paths_included': True,
        },
        'preview_cache': preview_cache,
    }


@mutation_entrypoint('media_fetch')
def fetch_media(
    vault_root: str | Path | None,
    citation: str,
    *,
    materialize_preview: bool = True,
    allow_remote: bool = False,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict[str, Any] | None = None,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('media materialization cannot run inside an outer writer session')
    store = SQLiteStore(cfg.paths.sqlite_path)
    unavailable_base = {'understanding': None}
    # Schema admission is the only pre-work writer window. Source location,
    # local/remote copy, decode, hashing, ffmpeg and preview I/O stay outside.
    with coordinated_vault_mutation(cfg, operation='media_fetch'):
        if store.path.exists():
            store.initialize()
    if not store.path.exists():
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'vault_index_missing', 'raw_paths_included': False, 'raw_content_included': False}
    with store.connect() as conn:
        asset = _asset_row_by_citation(conn, store, citation)
    if asset is None:
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'image_asset_not_found', 'raw_paths_included': False, 'raw_content_included': False}
    path_ref = str(asset['path_ref'] or '')
    source = _path_from_ref(cfg, path_ref) if path_ref else Path()
    if not path_ref or not source.exists() or not path_is_under(source, cfg.root):
        materialized = materialize_media_asset(
            cfg,
            store,
            asset,
            citation=citation,
            allow_remote=allow_remote,
            approval_grant=approval_grant,
            approval_payload=approval_payload,
            publish=False,
        )
        if not materialized.ok:
            code = 'approval_required' if materialized.status == 'awaiting_approval' else 'media_unavailable'
            return unavailable_base | materialized.to_redacted_dict() | {
                'ok': False,
                'code': code,
                'status': materialized.status,
            }
        with coordinated_vault_mutation(cfg, operation='media_fetch'):
            published = publish_materialization_result(
                store,
                materialized,
                expected_path_ref=path_ref,
            )
        with store.connect() as conn:
            refreshed = conn.execute('SELECT * FROM media_assets WHERE asset_id=?', (asset['asset_id'],)).fetchone()
        if refreshed is None or not refreshed['path_ref'] or (not published and str(refreshed['path_ref']) == path_ref):
            return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'materialization_failed', 'asset_id': asset['asset_id'], 'raw_paths_included': False, 'raw_content_included': False}
        asset = refreshed
        source = _path_from_ref(cfg, str(asset['path_ref']))
    if not source.exists() or not path_is_under(source, cfg.root):
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'materialized_cache_unavailable', 'asset_id': asset['asset_id'], 'raw_paths_included': False, 'raw_content_included': False}
    modality = str(asset['modality'] or 'image')
    if modality == 'video':
        content_sha256 = _sha256_file(source)
        understanding: dict[str, Any] | None = None
        with coordinated_vault_mutation(cfg, operation='media_fetch'):
            with store.connect() as conn:
                if not _is_sha256(str(asset['content_hash'] or '')) or str(asset['content_hash']).lower() != content_sha256:
                    conn.execute('UPDATE media_assets SET content_hash=?, updated_at=datetime("now") WHERE asset_id=?', (content_sha256, asset['asset_id']))
                    conn.commit()
                if store._table_exists(conn, 'media_understanding'):
                    row = _understanding_hit(conn, content_sha256, citation) if materialize_preview else conn.execute('SELECT * FROM media_understanding WHERE content_sha256=? AND status=?', (content_sha256, 'active')).fetchone()
                    understanding = _understanding_payload(row)
        if materialize_preview:
            _project_cached_understanding(
                cfg, store, asset_id=str(asset['asset_id']), citation=citation,
                understanding=understanding,
            )
        video_preview = _copy_video_preview(cfg, asset, source, citation) if materialize_preview else {
            'path': str(source),
            'mime': _mime(source),
            'width': None,
            'height': None,
            'preview': {'strategy': 'metadata_only', 'keyframe_paths': [], 'ffmpeg_available': bool(shutil.which('ffmpeg')), 'raw_paths_included': True},
            'preview_cache': {'policy': 'lru', 'max_files': PREVIEW_CACHE_MAX_FILES, 'max_bytes': PREVIEW_CACHE_MAX_BYTES, 'removed': 0},
        }
        return {
            'ok': True,
            'code': 'ok',
            'status': 'available',
            'citation': citation,
            'asset_id': asset['asset_id'],
            'content_sha256': content_sha256,
            **video_preview,
            'understanding': understanding,
            'raw_paths_included': True,
            'raw_content_included': False,
        }
    if modality in {'file', 'attachment', 'document'}:
        content_sha256 = _sha256_file(source)
        with coordinated_vault_mutation(cfg, operation='media_fetch'):
            with store.connect() as conn:
                if not _is_sha256(str(asset['content_hash'] or '')) or str(asset['content_hash']).lower() != content_sha256:
                    conn.execute(
                        'UPDATE media_assets SET content_hash=?, updated_at=datetime("now") WHERE asset_id=?',
                        (content_sha256, asset['asset_id']),
                    )
                    conn.commit()
        return {
            'ok': True,
            'code': 'ok',
            'status': 'available',
            'citation': citation,
            'asset_id': asset['asset_id'],
            'content_sha256': content_sha256,
            'path': str(source),
            'mime': _mime(source),
            'understanding': None,
            'raw_paths_included': True,
            'raw_content_included': False,
        }
    resolved = resolve_image_file(source, cfg.root, asset_id=str(asset['asset_id']))
    if resolved.derivative_ref is None or resolved.status not in {'copied', 'decoded'}:
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': resolved.error_code or resolved.status, 'asset_id': asset['asset_id'], 'raw_paths_included': False, 'raw_content_included': False}
    decoded = cfg.root / resolved.derivative_ref
    if not path_is_under(decoded, cfg.root):
        return unavailable_base | {'ok': False, 'code': 'media_unavailable', 'status': 'media_unavailable', 'reason': 'outside_vault_media_path', 'asset_id': asset['asset_id'], 'raw_paths_included': False, 'raw_content_included': False}
    content_sha256 = _sha256_file(decoded)
    understanding: dict[str, Any] | None = None
    with coordinated_vault_mutation(cfg, operation='media_fetch'):
        with store.connect() as conn:
            if not _is_sha256(str(asset['content_hash'] or '')) or str(asset['content_hash']).lower() != content_sha256:
                conn.execute('UPDATE media_assets SET content_hash=?, updated_at=datetime("now") WHERE asset_id=?', (content_sha256, asset['asset_id']))
                conn.commit()
            if store._table_exists(conn, 'media_understanding'):
                row = _understanding_hit(conn, content_sha256, citation) if materialize_preview else conn.execute('SELECT * FROM media_understanding WHERE content_sha256=? AND status=?', (content_sha256, 'active')).fetchone()
                understanding = _understanding_payload(row)
    if materialize_preview:
        _project_cached_understanding(
            cfg, store, asset_id=str(asset['asset_id']), citation=citation,
            understanding=understanding,
        )
    preview = cfg.root / 'media' / 'previews' / f"{_stable('preview', str(asset['asset_id']) + ':' + citation)}{decoded.suffix.lower() or '.img'}"
    if materialize_preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        if decoded.resolve() != preview.resolve():
            shutil.copyfile(decoded, preview)
        preview_cache = _preview_cache_lru(preview.parent, keep={preview})
        width, height = image_dimensions(preview)
        output_path = preview
    else:
        preview_cache = {'policy': 'lru', 'max_files': PREVIEW_CACHE_MAX_FILES, 'max_bytes': PREVIEW_CACHE_MAX_BYTES, 'removed': 0}
        width, height = image_dimensions(decoded)
        output_path = decoded
    return {
        'ok': True,
        'code': 'ok',
        'status': 'available',
        'citation': citation,
        'asset_id': asset['asset_id'],
        'content_sha256': content_sha256,
        'path': str(output_path),
        'mime': _mime(output_path, resolved.image_type),
        'width': width,
        'height': height,
        'preview_cache': preview_cache,
        'understanding': understanding,
        'raw_paths_included': True,
        'raw_content_included': False,
    }
