from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import shutil
from typing import Any, Iterable

from trove_core.approvals import ApprovalGrant, ApprovalValidationError, require_claimed_approval_grant
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import mutation_entrypoint, record_vault_mutation_noop
from trove_core.wechat.media.hash_store import sha256_file

FILE_SUFFIX_RE = re.compile(r'(?i)([^\s/\\]+\.(pdf|docx?|xlsx?|pptx?|zip|rar|7z|txt|csv|md|jpg|jpeg|png|gif|webp|heic|mp3|m4a|wav|amr|silk|mp4|mov))')
SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9._\-\u4e00-\u9fff]+')
WECHAT_DIR_MARKERS = {'MicroMsg', 'WeChat Files', 'xwechat_files', 'com.tencent.xinWeChat'}
SYSTEM_DIRS = {
    Path('/'),
    Path('/System'),
    Path('/bin'),
    Path('/sbin'),
    Path('/usr'),
    Path('/etc'),
    Path('/var'),
    Path('/private/var'),
    Path('/Library'),
}


@dataclass(frozen=True)
class _FileCandidate:
    asset_id: str
    file_name: str
    media_type: str
    size_bytes: int | None
    timestamp: str | None
    conversation: dict[str, Any]
    citation: str
    cache_state: str
    source_path: Path | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            'asset_id': self.asset_id,
            'file_name': self.file_name,
            'media_type': self.media_type,
            'size_bytes': self.size_bytes,
            'timestamp': self.timestamp,
            'conversation': self.conversation,
            'citation': self.citation,
            'cache_state': self.cache_state,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _load_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_media_types(media_types: Iterable[str] | str | None) -> set[str]:
    if media_types is None:
        return set()
    if isinstance(media_types, str):
        values = [media_types]
    else:
        values = list(media_types)
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _safe_file_name(name: str, fallback: str) -> str:
    raw = Path(name or fallback).name or fallback
    clean = SAFE_NAME_RE.sub('_', raw).strip('._ ')
    return clean or fallback


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}'


def _media_type_from_name(name: str, default: str = 'file') -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'}:
        return 'image'
    if suffix in {'.mp3', '.m4a', '.wav', '.amr', '.silk'}:
        return 'voice'
    if suffix in {'.mp4', '.mov'}:
        return 'video'
    if suffix:
        return 'document'
    return default or 'file'


def _resolve_source_path(vault_root: Path | None, path_ref: str | None, metadata: dict[str, Any]) -> Path | None:
    raw = path_ref or metadata.get('path_ref') or metadata.get('path_hint')
    if not raw:
        return None
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute() and vault_root is not None:
        candidate = vault_root / candidate
    return candidate


def _conversation_from_row(row: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        'account_id': row['message_account_id'] or row['account_id'] or metadata.get('account_id') or '',
        'conversation_id': row['conversation_id'] or metadata.get('conversation_id') or '',
        'title': row['conversation_title'] or metadata.get('conversation_title') or '',
        'type': row['conversation_type'] or metadata.get('conversation_type') or '',
    }


def _row_candidate(row: Any, *, vault_root: Path | None) -> _FileCandidate:
    metadata = _load_json(row['metadata_json'])
    source_path = _resolve_source_path(vault_root, row['path_ref'], metadata)
    file_name = metadata.get('file_name') or (source_path.name if source_path is not None else '') or Path(str(row['source_id'] or row['asset_id'])).name
    file_name = _safe_file_name(str(file_name), f'{row["asset_id"]}.bin')
    stat_size = None
    if source_path is not None:
        try:
            stat_size = source_path.stat().st_size if source_path.is_file() else None
        except OSError:
            stat_size = None
    size = stat_size if stat_size is not None else metadata.get('size_bytes')
    try:
        size = int(size) if size is not None else None
    except (TypeError, ValueError):
        size = None
    timestamp = row['message_timestamp'] or metadata.get('timestamp') or row['updated_at'] or row['created_at']
    return _FileCandidate(
        asset_id=row['asset_id'],
        file_name=file_name,
        media_type=str(row['media_type'] or _media_type_from_name(file_name)),
        size_bytes=size,
        timestamp=timestamp,
        conversation=_conversation_from_row(row, metadata),
        citation=row['source_citation'] or row['citation'],
        cache_state=row['cache_state'] or ('cached' if source_path and source_path.exists() else 'metadata_only'),
        source_path=source_path,
    )


def _message_file_candidates(rows: Iterable[Any]) -> list[_FileCandidate]:
    out: list[_FileCandidate] = []
    for row in rows:
        match = FILE_SUFFIX_RE.search(row['content'] or '')
        if not match:
            continue
        file_name = _safe_file_name(match.group(1), 'wechat-file')
        out.append(_FileCandidate(
            asset_id=_stable('msgfile', row['citation'] + ':' + file_name),
            file_name=file_name,
            media_type=_media_type_from_name(file_name),
            size_bytes=None,
            timestamp=row['timestamp'],
            conversation={
                'account_id': row['account_id'],
                'conversation_id': row['conversation_id'],
                'title': row['conversation_title'],
                'type': row['conversation_type'],
            },
            citation=row['citation'],
            cache_state='metadata_only',
            source_path=None,
        ))
    return out


def _matches(candidate: _FileCandidate, *, contact: str | None, conversation_id: str | None, file_name: str | None, media_types: set[str], since: str | None, until: str | None) -> bool:
    if conversation_id and candidate.conversation.get('conversation_id') != conversation_id:
        return False
    if contact:
        needle = contact.lower()
        haystack = ' '.join(str(v or '') for v in [candidate.conversation.get('title'), candidate.conversation.get('conversation_id'), candidate.file_name]).lower()
        if needle not in haystack:
            return False
    if file_name:
        needle = file_name.lower()
        if needle not in candidate.file_name.lower():
            return False
    if media_types and candidate.media_type.lower() not in media_types:
        return False
    if since and (candidate.timestamp or '') < since:
        return False
    if until and (candidate.timestamp or '') > until:
        return False
    return True


def _collect_candidates(
    store: SQLiteStore,
    *,
    contact: str | None = None,
    conversation_id: str | None = None,
    file_name: str | None = None,
    media_types: Iterable[str] | str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 500,
) -> list[_FileCandidate]:
    from trove_core.bounds import BoundedLimit, PRIVATE_LIST

    limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
    if not store.path.is_file():
        return []
    store.initialize()
    type_filter = _as_media_types(media_types)
    vault_root = store.path.parent.parent if store.path.name == 'trove.sqlite' else None
    candidates: list[_FileCandidate] = []
    with store.connect() as conn:
        if store._table_exists(conn, 'media_assets'):
            media_where = ["ma.modality IN ('file','attachment','document','image','voice','video')"]
            media_params: list[Any] = []
            if conversation_id:
                escaped = str(conversation_id).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                media_where.append("(m.conversation_id=? OR ma.metadata_json LIKE ? ESCAPE '\\')")
                media_params.extend((conversation_id, f'%{escaped}%'))
            if contact:
                escaped = str(contact).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                media_where.append("(m.conversation_title LIKE ? ESCAPE '\\' OR m.sender_name LIKE ? ESCAPE '\\' OR ma.metadata_json LIKE ? ESCAPE '\\')")
                media_params.extend((f'%{escaped}%', f'%{escaped}%', f'%{escaped}%'))
            if file_name:
                escaped = str(file_name).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
                media_where.append("(ma.path_ref LIKE ? ESCAPE '\\' OR ma.metadata_json LIKE ? ESCAPE '\\')")
                media_params.extend((f'%{escaped}%', f'%{escaped}%'))
            if type_filter:
                placeholders = ','.join('?' for _ in type_filter)
                media_where.append(f'ma.media_type IN ({placeholders})')
                media_params.extend(sorted(type_filter))
            if since:
                media_where.append('COALESCE(m.timestamp,ma.updated_at,ma.created_at)>=?')
                media_params.append(since)
            if until:
                media_where.append('COALESCE(m.timestamp,ma.updated_at,ma.created_at)<=?')
                media_params.append(until)
            rows = conn.execute(
                f"""SELECT ma.*, mal.source_citation,
                          m.account_id AS message_account_id,
                          m.conversation_id AS conversation_id,
                          m.conversation_title AS conversation_title,
                          m.conversation_type AS conversation_type,
                          m.timestamp AS message_timestamp
                   FROM media_assets ma
                   LEFT JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                   LEFT JOIN messages m ON m.citation=mal.source_citation OR m.citation=ma.citation
                   WHERE {' AND '.join(media_where)}
                   ORDER BY COALESCE(m.timestamp, ma.updated_at, ma.created_at) DESC, ma.asset_id
                   LIMIT ?""",
                (*media_params, limit),
            )
            for row in rows:
                candidate = _row_candidate(row, vault_root=vault_root)
                if _matches(candidate, contact=contact, conversation_id=conversation_id, file_name=file_name, media_types=type_filter, since=since, until=until):
                    candidates.append(candidate)
        message_where = []
        params: list[Any] = []
        if conversation_id:
            message_where.append('conversation_id=?')
            params.append(conversation_id)
        if contact:
            message_where.append('(conversation_title LIKE ? OR sender_name LIKE ?)')
            params.extend([f'%{contact}%', f'%{contact}%'])
        if file_name:
            message_where.append('content LIKE ?')
            params.append(f'%{file_name}%')
        where_sql = 'WHERE ' + ' AND '.join(message_where) if message_where else ''
        msg_rows = conn.execute(
            f'SELECT * FROM messages {where_sql} ORDER BY timestamp DESC LIMIT ?',
            (*params, limit),
        )
        for candidate in _message_file_candidates(msg_rows):
            if _matches(candidate, contact=contact, conversation_id=conversation_id, file_name=file_name, media_types=type_filter, since=since, until=until):
                candidates.append(candidate)
    seen: set[tuple[str, str]] = set()
    deduped: list[_FileCandidate] = []
    for candidate in candidates:
        key = (candidate.asset_id, candidate.citation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def list_conversation_files(store: SQLiteStore, *, contact: str | None = None, conversation_id: str | None = None, file_name: str | None = None, media_types: Iterable[str] | str | None = None, since: str | None = None, until: str | None = None, limit: int = 100) -> dict[str, Any]:
    from trove_core.bounds import BoundedLimit, PRIVATE_LIST

    limit = BoundedLimit(limit, field='limit', spec=PRIVATE_LIST)
    candidates = _collect_candidates(
        store,
        contact=contact,
        conversation_id=conversation_id,
        file_name=file_name,
        media_types=media_types,
        since=since,
        until=until,
        limit=limit,
    )
    return {
        'files': [c.public_dict() for c in candidates[:limit]],
        'count': min(len(candidates), limit),
        'total_candidates': len(candidates),
        'raw_paths_included': False,
    }


def _selection_filters(selection: Any) -> dict[str, Any]:
    if selection is None:
        return {}
    if isinstance(selection, list):
        return {'asset_ids': [str(v) for v in selection if str(v).strip()]}
    if isinstance(selection, dict):
        data = dict(selection)
        if data.get('conversation') and not data.get('conversation_id'):
            data['conversation_id'] = data.get('conversation')
        if data.get('filename') and not data.get('file_name'):
            data['file_name'] = data.get('filename')
        if 'asset_id' in data and 'asset_ids' not in data:
            data['asset_ids'] = [data['asset_id']]
        if 'asset_ids' in data:
            data['asset_ids'] = [str(v) for v in (data.get('asset_ids') or []) if str(v).strip()]
        return data
    return {'asset_ids': [str(selection)]}


def _has_export_scope(filters: dict[str, Any]) -> bool:
    return bool(
        str(filters.get('contact') or '').strip()
        or str(filters.get('conversation_id') or '').strip()
        or [v for v in (filters.get('asset_ids') or []) if str(v).strip()]
    )


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_dest_dir(vault_root: Path, dest_dir: str | Path) -> Path:
    raw = Path(dest_dir).expanduser()
    if not raw.is_absolute():
        raise ValueError('dest_dir must be an absolute path')
    dest = raw.resolve()
    vault = vault_root.expanduser().resolve()
    if _is_relative_to(dest, vault):
        raise ValueError('dest_dir must be outside the runtime Vault')
    if dest in SYSTEM_DIRS:
        raise ValueError('dest_dir must not be a system directory')
    if any(part in WECHAT_DIR_MARKERS for part in dest.parts):
        raise ValueError('dest_dir must not be inside a WeChat data directory')
    return dest


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'version': 1, 'created_at': _now_iso(), 'updated_at': None, 'entries': [], 'raw_paths_included': False}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data.get('entries'), list):
        data['entries'] = []
    data.setdefault('version', 1)
    data.setdefault('created_at', _now_iso())
    data['raw_paths_included'] = False
    return data


def _unique_dest(dest: Path, file_name: str, content_hash: str) -> Path:
    base = _safe_file_name(file_name, f'{content_hash[:12]}.bin')
    candidate = dest / base
    if not candidate.exists():
        return candidate
    try:
        if candidate.is_file() and sha256_file(candidate) == content_hash:
            return candidate
    except OSError:
        pass
    stem = Path(base).stem or content_hash[:12]
    suffix = Path(base).suffix
    for idx in range(1, 1000):
        candidate = dest / f'{stem}-{idx}{suffix}'
        if not candidate.exists():
            return candidate
    raise RuntimeError('unable to allocate archive filename')


def archive_approval_payload(
    cfg: VaultConfig,
    *,
    selection: Any,
    dest_dir: str | Path,
    mode: str = 'copy',
) -> dict[str, Any]:
    from trove_core.bounds import BoundSpec, BoundedLimit

    filters = _selection_filters(selection)
    if not _has_export_scope(filters):
        raise ValueError('files archive requires contact, conversation_id, or asset_id selection')
    dest = _validate_dest_dir(cfg.root, dest_dir)
    archive_limit = BoundedLimit(
        filters.get('limit', 500),
        field='limit',
        spec=BoundSpec(1, 500, 500),
    )
    return {
        'selection': {
            'account_id': filters.get('account_id'),
            'contact': filters.get('contact'),
            'conversation_id': filters.get('conversation_id'),
            'file_name': filters.get('file_name'),
            'asset_ids': sorted(str(value) for value in (filters.get('asset_ids') or [])),
            'media_types': sorted(str(value) for value in (filters.get('media_types') or filters.get('media_type') or [])),
            'since': filters.get('since'),
            'until': filters.get('until'),
            'limit': int(archive_limit),
        },
        'mode': mode,
        'dest_dir': str(dest),
    }


@mutation_entrypoint('files_archive')
def archive_files(
    cfg: VaultConfig,
    *,
    selection: Any,
    dest_dir: str | Path,
    mode: str = 'copy',
    approval_grant: ApprovalGrant,
    approval_payload: dict[str, Any],
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    from trove_core.bounds import BoundSpec, BoundedLimit

    if mode != 'copy':
        raise ValueError('files archive supports copy mode only')
    filters = _selection_filters(selection)
    if not _has_export_scope(filters):
        raise ValueError('files archive requires contact, conversation_id, or asset_id selection')
    dest = _validate_dest_dir(cfg.root, dest_dir)
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('file export cannot run inside a Vault writer session')
    payload = archive_approval_payload(cfg, selection=selection, dest_dir=dest, mode=mode)
    if type(approval_payload) is not dict or approval_payload != payload:
        raise ApprovalValidationError(
            'file archive approval payload does not match the export request',
            code='grant_payload_mismatch',
        )
    require_claimed_approval_grant(
        approval_grant,
        cfg.root,
        action='files_archive',
        danger_class='local-file-export',
        payload=payload,
    )
    store = SQLiteStore(cfg.paths.sqlite_path)
    limit = BoundedLimit(
        filters.get('limit', 500),
        field='limit',
        spec=BoundSpec(1, 500, 500),
    )
    candidates = _collect_candidates(
        store,
        contact=filters.get('contact'),
        conversation_id=filters.get('conversation_id'),
        file_name=filters.get('file_name'),
        media_types=filters.get('media_types') or filters.get('media_type'),
        since=filters.get('since'),
        until=filters.get('until'),
        limit=limit,
    )
    account_id = str(filters.get('account_id') or '').strip()
    if account_id:
        candidates = [
            candidate for candidate in candidates
            if candidate.conversation.get('account_id') == account_id
        ]
    asset_ids = {str(v) for v in (filters.get('asset_ids') or [])}
    if asset_ids:
        candidates = [c for c in candidates if c.asset_id in asset_ids]
    candidates = candidates[:limit]

    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / 'trove-archive-manifest.json'
    manifest = _load_manifest(manifest_path)
    existing_hashes = {entry.get('hash') for entry in manifest.get('entries', []) if entry.get('hash')}
    copied = 0
    skipped_duplicate = 0
    skipped_missing = 0
    archived_entries: list[dict[str, Any]] = []
    batch_hashes: set[str] = set()
    for candidate in candidates:
        source = candidate.source_path
        if source is None or not source.exists() or not source.is_file():
            skipped_missing += 1
            continue
        content_hash = sha256_file(source)
        if content_hash in existing_hashes or content_hash in batch_hashes:
            skipped_duplicate += 1
            continue
        target = _unique_dest(dest, candidate.file_name, content_hash)
        shutil.copy2(source, target)
        entry = {
            'file_name': target.name,
            'hash': content_hash,
            'content_hash': content_hash,
            'source_citation': candidate.citation,
            'timestamp': candidate.timestamp,
            'archived_at': _now_iso(),
        }
        manifest['entries'].append(entry)
        archived_entries.append(entry)
        batch_hashes.add(content_hash)
        copied += 1
    manifest['updated_at'] = _now_iso()
    manifest['raw_paths_included'] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    # This operation reads the Vault and writes only to the approved external
    # destination. Approval consumption is its audit record; no Vault writer is
    # needed for hashing, copying, or the external manifest.
    record_vault_mutation_noop(operation='files_archive')
    return {
        'ok': True,
        'mode': 'copy',
        'selected': len(candidates),
        'copied': copied,
        'skipped_duplicate': skipped_duplicate,
        'skipped_missing': skipped_missing,
        'manifest': manifest_path.name,
        'entries': archived_entries,
        'raw_paths_included': False,
    }
