from __future__ import annotations

from contextlib import closing

from datetime import datetime
import hashlib
from pathlib import Path
import sqlite3

from trove_core.maintain import rotate_sqlite_backups
from trove_core.store.sqlite_store import SQLiteStore, open_store
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.wechat.decrypt.manifest import load_snapshot_guard
from trove_core.domain.content import classify_content_kind, display_content_for_kind
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter, decode_content, msg_table_for, stable_id
from trove_core.wechat.media.source_registry import resolve_snapshot_root
from trove_core.domain.messages import Message
from trove_core.wechat.parsers.appmsg import parse_appmsg
from trove_core.wechat.source_discovery import is_wechat_decrypted_account_dir, iter_importable_files


def appmsg_source_identity(source: str | Path) -> str:
    resolved = Path(source).expanduser().resolve()
    guard = load_snapshot_guard(resolved) if resolved.is_dir() else None
    revision = guard.run_id if guard is not None and guard.run_id else ''
    try:
        stat = resolved.stat()
        coordinates = f'{resolved}:{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:{revision}'
    except OSError:
        coordinates = f'{resolved}:missing:{revision}'
    return hashlib.sha256(coordinates.encode('utf-8')).hexdigest()


def discover_appmsg_messages(source: str | Path, *, limit_per_sqlite: int | None = None) -> list[Message]:
    root = Path(source).expanduser().resolve()
    guard = load_snapshot_guard(root) if root.is_dir() else None
    messages: dict[str, Message] = {}
    for unit in iter_importable_files(root):
        if not is_wechat_decrypted_account_dir(unit):
            continue
        if guard is not None and not guard.allows(unit):
            continue
        importer = WeChatDecryptedAccountImporter(unit)
        _, _, loaded = importer.load(
            limit_per_shard=limit_per_sqlite,
            content_kinds={'appmsg'},
        )
        for message in loaded:
            if message.content_kind == 'appmsg' and message.normalized_payload is not None:
                messages[message.citation] = message
    return list(messages.values())


def _exact_appmsg_from_snapshot(store: SQLiteStore, cfg: VaultConfig, row: sqlite3.Row) -> Message | None:
    with store.connect() as conn:
        snapshots = list(conn.execute(
            "SELECT snapshot_revision FROM source_snapshots WHERE state='available' ORDER BY updated_at DESC",
        ))
    for snapshot in snapshots:
        root, _ = resolve_snapshot_root(cfg, store, str(snapshot['snapshot_revision']))
        if root is None:
            continue
        candidates = [root]
        try:
            candidates.extend(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
        except OSError:
            continue
        account_dir = next(
            (path for path in candidates if stable_id('acct', path.name) == str(row['account_id'])),
            None,
        )
        if account_dir is None:
            continue
        db_path = account_dir / f"{Path(str(row['shard_id'])).name}.db"
        if not db_path.is_file():
            continue
        try:
            with closing(sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)) as source:
                source.row_factory = sqlite3.Row
                try:
                    names = [str(item['user_name']) for item in source.execute('SELECT user_name FROM Name2Id') if item['user_name']]
                except sqlite3.DatabaseError:
                    names = []
                if not names:
                    for names_db in sorted(account_dir.glob('message_*.db')):
                        try:
                            with closing(sqlite3.connect(f'file:{names_db}?mode=ro', uri=True)) as names_conn:
                                names_conn.row_factory = sqlite3.Row
                                names.extend(
                                    str(item['user_name']) for item in names_conn.execute('SELECT user_name FROM Name2Id')
                                    if item['user_name']
                                )
                        except sqlite3.DatabaseError:
                            continue
                username = next(
                    (name for name in names if stable_id('conv', f'{account_dir.name}:{name}') == str(row['conversation_id'])),
                    None,
                )
                if username is None:
                    continue
                table = msg_table_for(username)
                columns = [str(item[1]) for item in source.execute(f'PRAGMA table_info("{table}")')]
                content_columns = [
                    name for name in ('message_content', 'compress_content', 'WCDB_CT_message_content')
                    if name in columns
                ]
                if not content_columns:
                    continue
                selected = [*content_columns, *(['local_type'] if 'local_type' in columns else [])]
                source_row = source.execute(
                    'SELECT ' + ','.join(f'"{name}"' for name in selected) + f' FROM "{table}" WHERE local_id=? LIMIT 1',
                    (int(row['local_id']),),
                ).fetchone()
                if source_row is None:
                    continue
                raw_content = ''
                for column in content_columns:
                    raw_content = raw_content or decode_content(source_row[column])
                local_type = source_row['local_type'] if 'local_type' in source_row.keys() else None
                content_kind = classify_content_kind(raw_content, local_type=local_type)
                payload = parse_appmsg(raw_content) if content_kind == 'appmsg' else None
                timestamp = datetime.fromisoformat(str(row['timestamp']).replace('Z', '+00:00'))
                return Message(
                    account_id=str(row['account_id']), account_label=str(row['account_label']),
                    conversation_id=str(row['conversation_id']), conversation_title=str(row['conversation_title']),
                    conversation_type=str(row['conversation_type']), sender_id=str(row['sender_id']),
                    sender_name=str(row['sender_name']), timestamp=timestamp,
                    content=payload.display_text if payload is not None else display_content_for_kind(raw_content, content_kind),
                    shard_id=str(row['shard_id']), local_id=int(row['local_id']),
                    sent_by_me=bool(row['sent_by_me']), source_type='message', content_kind=content_kind,
                    direction_hint=str(row['direction']), normalized_payload=payload.to_dict() if payload is not None else None,
                )
        except (sqlite3.DatabaseError, TypeError, ValueError):
            continue
    return None


@mutation_entrypoint('appmsg_backfill')
def recover_appmsg_payload(
    vault_root: str | Path,
    citation: str,
    *,
    write_session: VaultWriteSession | None = None,
) -> dict:
    """Recover and reprocess one exact AppMsg citation from a bound source snapshot."""
    cfg = VaultConfig.resolve(str(vault_root), env={})
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('AppMsg source recovery cannot run inside an outer writer session')
    store = SQLiteStore(cfg.paths.sqlite_path)
    with coordinated_vault_mutation(cfg, operation='appmsg_backfill'):
        store.initialize()
    with store.connect() as conn:
        row = conn.execute('SELECT * FROM messages WHERE citation=?', (str(citation),)).fetchone()
    if row is None or str(row['content_kind']) != 'appmsg':
        return {
            'ok': False, 'status': 'unavailable', 'reason': 'appmsg_message_not_found',
            'raw_content_included': False, 'raw_paths_included': False,
        }
    # Bound snapshot traversal and source SQLite reads can take seconds and do
    # not mutate the Vault.
    message = _exact_appmsg_from_snapshot(store, cfg, row)
    if message is None:
        return {
            'ok': False, 'status': 'unavailable', 'reason': 'source_appmsg_not_found',
            'raw_content_included': False, 'raw_paths_included': False,
        }
    with coordinated_vault_mutation(cfg, operation='appmsg_backfill'):
        if message.content_kind != 'appmsg':
            report = store.apply_message_delta([], [], [message])
            return {
                'ok': True,
                'status': 'reclassified',
                'parse_status': 'not_appmsg',
                'normalized_type': message.content_kind,
                'source_hash': None,
                'parser_version': None,
                'reason': f'source_reclassified_{message.content_kind}',
                'changed': int(report.get('rows_written') or 0),
                'raw_content_included': False,
                'raw_paths_included': False,
            }
        report = store.reprocess_appmsg_payloads([message])
        with store.connect() as conn:
            payload = conn.execute(
                'SELECT parse_status,normalized_type,source_hash,parser_version,unsupported_reason FROM message_payloads WHERE citation=?',
                (str(citation),),
            ).fetchone()
        return {
            'ok': payload is not None and str(payload['parse_status']) == 'parsed',
            'status': 'completed' if payload is not None and str(payload['parse_status']) == 'parsed' else 'unsupported',
            'parse_status': str(payload['parse_status']) if payload is not None else 'missing',
            'normalized_type': str(payload['normalized_type']) if payload is not None else 'unsupported',
            'source_hash': str(payload['source_hash']) if payload is not None else None,
            'parser_version': str(payload['parser_version']) if payload is not None else None,
            'reason': str(payload['unsupported_reason'] or '') if payload is not None else 'appmsg_payload_missing',
            'changed': int(report.get('changed') or 0),
            'raw_content_included': False,
            'raw_paths_included': False,
        }


def appmsg_backfill_plan(
    vault_root: str | Path,
    source: str | Path,
    *,
    limit_per_sqlite: int | None = None,
) -> dict:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    messages = discover_appmsg_messages(source, limit_per_sqlite=limit_per_sqlite)
    store = open_store(cfg.paths.sqlite_path, readonly=True)
    try:
        report = store.appmsg_payload_reprocess_plan(messages)
    finally:
        store.close()
    return report | {'source_identity': appmsg_source_identity(source)}


@mutation_entrypoint('appmsg_backfill')
def backfill_appmsg_payloads(
    vault_root: str | Path,
    source: str | Path,
    *,
    limit_per_sqlite: int | None = None,
    backup_retention: int = 5,
    write_session: VaultWriteSession | None = None,
) -> dict:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    if write_session is not None:
        write_session.validate_for(cfg)
        raise RuntimeError('AppMsg source discovery cannot run inside an outer writer session')
    with coordinated_vault_mutation(cfg, operation='appmsg_backfill'):
        SQLiteStore(cfg.paths.sqlite_path).initialize()
    messages = discover_appmsg_messages(source, limit_per_sqlite=limit_per_sqlite)
    backup = rotate_sqlite_backups(cfg.paths.sqlite_path, retention=backup_retention, create=True)
    source_identity = appmsg_source_identity(source)
    with coordinated_vault_mutation(cfg, operation='appmsg_backfill'):
        report = SQLiteStore(cfg.paths.sqlite_path).reprocess_appmsg_payloads(messages)
    return {
        'ok': True,
        'backup_created': bool(backup),
        'backfill': report,
        'source_identity': source_identity,
        'raw_content_included': False,
        'raw_paths_included': False,
    }
