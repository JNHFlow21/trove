from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from trove_core.maintain import rotate_sqlite_backups
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore, open_store
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.message_refs import message_media_references_for_messages
from trove_core.domain.messages import Message


def _timestamp(value: str) -> datetime:
    text = str(value or '').strip().replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.fromtimestamp(0, tz=timezone.utc)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def message_media_references_from_store(store: SQLiteStore, *, limit: int | None = None):
    store.initialize()
    query = """SELECT m.*,p.appmsg_type,p.normalized_type,p.parse_status,p.normalized_json,
                      p.display_text,p.source_hash,p.parser_version,p.unsupported_reason
                 FROM messages m
                 LEFT JOIN message_payloads p ON p.citation=m.citation
                WHERE m.content_kind IN ('image','voice','video','sticker')
                   OR (m.content_kind='appmsg' AND p.normalized_type='file')
                ORDER BY m.id"""
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += ' LIMIT ?'
        params = (int(limit),)
    messages: list[Message] = []
    with store.connect() as conn:
        for row in conn.execute(query, params):
            payload = None
            if row['normalized_type']:
                try:
                    fields = json.loads(row['normalized_json'] or '{}')
                except json.JSONDecodeError:
                    fields = {}
                payload = {
                    'source_hash': row['source_hash'],
                    'appmsg_type': row['appmsg_type'],
                    'normalized_type': row['normalized_type'],
                    'parse_status': row['parse_status'],
                    'fields': fields,
                    'display_text': row['display_text'],
                    'unsupported_reason': row['unsupported_reason'],
                    'parser_version': row['parser_version'],
                }
            messages.append(Message(
                account_id=str(row['account_id']),
                account_label=str(row['account_label']),
                conversation_id=str(row['conversation_id']),
                conversation_title=str(row['conversation_title']),
                conversation_type=str(row['conversation_type']),  # type: ignore[arg-type]
                sender_id=str(row['sender_id']),
                sender_name=str(row['sender_name']),
                timestamp=_timestamp(str(row['timestamp'])),
                content=str(row['content']),
                shard_id=str(row['shard_id']),
                local_id=int(row['local_id']),
                sent_by_me=bool(row['sent_by_me']),
                source_type=str(row['source_type']),  # type: ignore[arg-type]
                content_kind=str(row['content_kind']),
                direction_hint=str(row['direction']),  # type: ignore[arg-type]
                normalized_payload=payload,
            ))
    return message_media_references_for_messages(messages)


def message_media_backfill_plan(vault_root: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    store = open_store(cfg.paths.sqlite_path, readonly=True)
    try:
        refs = message_media_references_from_store(store, limit=limit)
        missing_assets = 0
        missing_links = 0
        modality_counts: dict[str, int] = {}
        with store.connect() as conn:
            for ref in refs:
                modality_counts[ref.modality] = modality_counts.get(ref.modality, 0) + 1
                if conn.execute('SELECT 1 FROM media_assets WHERE asset_id=?', (ref.asset_id,)).fetchone() is None:
                    missing_assets += 1
                if conn.execute(
                    'SELECT 1 FROM media_asset_links WHERE asset_id=? AND source_citation=?',
                    (ref.asset_id, ref.citation),
                ).fetchone() is None:
                    missing_links += 1
    finally:
        store.close()
    return {
        'ok': True,
        'eligible_messages': len(refs),
        'missing_assets': missing_assets,
        'missing_links': missing_links,
        'would_change': max(missing_assets, missing_links),
        'modality_counts': dict(sorted(modality_counts.items())),
        'raw_content_included': False,
        'raw_paths_included': False,
    }


@mutation_entrypoint('message_media_backfill')
def backfill_message_media_references(
    vault_root: str | Path,
    *,
    limit: int | None = None,
    backup_retention: int = 5,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(cfg, operation='message_media_backfill', write_session=write_session):
        backup = rotate_sqlite_backups(cfg.paths.sqlite_path, retention=backup_retention, create=True)
        store = SQLiteStore(cfg.paths.sqlite_path)
        refs = message_media_references_from_store(store, limit=limit)
        linked = MediaLinker(MultimodalRepository(store)).link_references(refs)
    return {
        'ok': True,
        'backup_created': bool(backup),
        'eligible_messages': len(refs),
        'link_result': linked.to_dict(),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
