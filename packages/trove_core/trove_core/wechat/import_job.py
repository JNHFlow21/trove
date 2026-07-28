from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import tempfile
import time
from typing import Any

from trove_core.media_pipeline import enqueue_media_jobs
from trove_core.store.repositories import WeChatRepository
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store import change_journal
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.locks import VaultOperationLocked
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.vault.operations import reset_index_cache
from trove_core.vault.tracing import TraceTimeline
from trove_core.domain.messages import Message
from trove_core.wechat.process_config import (
    ImportProcessConfig,
    default_process_config,
    process_config_from_payload,
    read_latest_process_config,
    write_process_config,
)
from trove_core.wechat.source_inventory import summarize_path
from trove_core.wechat.source_discovery import is_wechat_decrypted_account_dir, iter_importable_files
from .importers import JsonlExportImporter, SQLiteArchiveImporter, WeChatDecryptedAccountImporter
from .import_receipts import (
    SourceFingerprint,
    SourceFingerprintUnavailable,
    completed_receipt,
    load_import_receipts,
    receipt_matches,
    source_stat_token,
    stable_import_source_key,
    strong_source_fingerprint,
    write_import_receipts,
)
from .auxiliary_import import (
    PreparedAuxiliaryImport,
    changed_citations,
    commit_prepared_auxiliary_sources,
    family_signature,
    prepare_auxiliary_sources,
    removed_citation_count,
)
from .media.linker import MediaLinker
from .media.message_refs import (
    message_media_references_for_messages as _message_media_references_for_messages,
    voice_media_references_for_messages as _voice_media_references_for_messages,
)
from .media.resources import MediaReference, discover_media_assets_delta, message_media_asset_id
from .decrypt.manifest import load_snapshot_guard
from .media.source_registry import (
    SourceSnapshot,
    account_dir_hash,
    bind_account_assets,
    inspect_source_snapshot,
    persist_source_snapshot,
    resolve_snapshot_root,
)


@dataclass(frozen=True)
class ImportJobResult:
    status: str
    vault: str
    sources_seen: int
    sources_imported: int
    messages_imported: int
    contacts_imported: int
    moments_imported: int
    favorites_imported: int
    media_links_accepted: int
    media_links_excluded: int
    scope_counts: dict[str, int]
    excluded_counts: dict[str, int]
    coverage_gaps: list[dict[str, str]]
    changed: int
    errors: list[str]
    started_at: float
    completed_at: float
    process_config_id: str = ""
    process_config_hash: str = ""
    trace_id: str = ""
    sources_skipped_unchanged: int = 0
    waterlines_updated: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def importer_for(path: Path):
    if is_wechat_decrypted_account_dir(path):
        return WeChatDecryptedAccountImporter(path)
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        return JsonlExportImporter(path)
    if suffix in {'.db', '.sqlite', '.sqlite3'}:
        return SQLiteArchiveImporter(path)
    return None



def _signature_hash(values: list[object]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _chunk_family_signature(store: SQLiteStore, source_type: str) -> dict[str, str]:
    if source_type in {'contact', 'moment', 'favorite'}:
        return family_signature(store, source_type)
    store.initialize()
    if not store.path.exists():
        return {}
    with store.connect() as conn:
        if source_type == 'transcript':
            if not store._table_exists(conn, 'transcripts'):
                return {}
            return {
                row['citation']: _signature_hash([row['transcript_id'], row['text'], row['language'], row['status']])
                for row in conn.execute('SELECT citation,transcript_id,text,language,status FROM transcripts ORDER BY citation')
            }
        if source_type == 'image_observation':
            if not store._table_exists(conn, 'image_observations'):
                return {}
            return {
                row['citation']: _signature_hash([row['observation_id'], row['caption'], row['visible_text'], row['status']])
                for row in conn.execute('SELECT citation,observation_id,caption,visible_text,status FROM image_observations ORDER BY citation')
            }
    return {}


def _signature_change_count(before: dict[str, str], after: dict[str, str]) -> int:
    return len(changed_citations(before, after)) + removed_citation_count(before, after)


def _voice_asset_id_for_citation(citation: str) -> str:
    return message_media_asset_id(citation, 'voice', 'voice')


def _resolve_import_process_config(vault_root: Path, process_config: ImportProcessConfig | None) -> ImportProcessConfig:
    if process_config is not None:
        return process_config
    latest = read_latest_process_config(vault_root)
    if latest.get('status') == 'ok':
        return process_config_from_payload(latest.get('config') or {})
    return default_process_config()


@dataclass
class _PreparedImportSource:
    file: Path
    source_key: str
    accounts: list[Any]
    conversations: list[Any]
    messages: list[Message]
    source_fingerprint: SourceFingerprint | None
    source_stat_before: str | None
    source_snapshot: SourceSnapshot | None = None
    auxiliary: PreparedAuxiliaryImport | None = None
    message_media_refs: tuple[MediaReference, ...] = ()
    discovered_media_refs: tuple[MediaReference, ...] = ()
    media_source_states: tuple[dict[str, Any], ...] = ()
    account_id: str | None = None
    account_hash: str | None = None
    receipt_stable: bool = True


@dataclass
class _PreparedImportJob:
    started_at: float
    effective_config: ImportProcessConfig
    expected_generation: int
    reset_index: bool
    limit_per_sqlite: int | None
    trace: TraceTimeline
    trace_id: str
    import_receipts: dict[str, dict[str, Any]]
    sources: list[_PreparedImportSource]
    sources_seen: int
    sources_skipped_unchanged: int
    scope_counts: dict[str, int]
    excluded_counts: dict[str, int]
    coverage_gaps: list[dict[str, str]]
    errors: list[str]
    import_waterline_updates: dict[tuple[str, str, str], dict[str, object]]


def _prepare_import_job(
    cfg: VaultConfig,
    sources: list[Path],
    *,
    started_at: float,
    effective_config: ImportProcessConfig,
    expected_generation: int,
    reset_index: bool,
    limit_per_sqlite: int | None,
    force_rescan: bool,
    trace: TraceTimeline,
    trace_id: str,
) -> _PreparedImportJob:
    """Traverse, hash, parse, and discover sources without the Vault writer."""

    process_config_hash = effective_config.redacted_hash()
    import_receipts = {} if reset_index else load_import_receipts(cfg.root)
    receipt_skip_enabled = limit_per_sqlite is None and not reset_index and not force_rescan
    prepared_sources: list[_PreparedImportSource] = []
    sources_seen = 0
    sources_skipped_unchanged = 0
    scope_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    coverage_gaps: list[dict[str, str]] = []
    errors: list[str] = []
    import_waterline_updates: dict[tuple[str, str, str], dict[str, object]] = {}
    snapshot_cache: dict[Path, SourceSnapshot] = {}

    state_store = SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    scan_store = state_store
    temporary_scan_dir: tempfile.TemporaryDirectory[str] | None = None
    if reset_index:
        # A reset publication cannot use old media watermarks or it would skip
        # rows needed by the replacement index.  The temporary DB is local
        # preparation state and never touches the Vault publication.
        temporary_scan_dir = tempfile.TemporaryDirectory(prefix='trove-import-scan-')
        scan_store = SQLiteStore(Path(temporary_scan_dir.name) / 'scan.sqlite')
        scan_store.initialize()

    try:
        for source in sources:
            expanded_source = Path(source).expanduser()
            trace.progress(trace_id, 'import_source', {'source_name': expanded_source.name})
            snapshot_guard = load_snapshot_guard(expanded_source) if expanded_source.is_dir() else None
            for file in iter_importable_files(expanded_source):
                sources_seen += 1
                if snapshot_guard is not None and is_wechat_decrypted_account_dir(file) and not snapshot_guard.allows(file):
                    errors.append(f'{file.name}: not_selected_account')
                    excluded_counts['not_selected_account'] = excluded_counts.get('not_selected_account', 0) + 1
                    continue
                summary = summarize_path(file)
                summary_importable = True if is_wechat_decrypted_account_dir(file) else summary.importable
                if summary.sensitive or not summary_importable:
                    errors.append(f'{file.name}: source is not approved for import ({summary.category})')
                    continue
                importer = importer_for(file)
                if importer is None:
                    continue
                source_key = stable_import_source_key(file)
                source_fingerprint: SourceFingerprint | None = None
                if limit_per_sqlite is None:
                    try:
                        source_fingerprint = strong_source_fingerprint(file)
                    except SourceFingerprintUnavailable:
                        # Import remains allowed, but this source cannot create
                        # or match a completed skip receipt.
                        source_fingerprint = None

                prior_receipt = import_receipts.get(source_key)
                prior_snapshot_available = True
                if receipt_skip_enabled and isinstance(importer, WeChatDecryptedAccountImporter):
                    prior_snapshot_revision = str((prior_receipt or {}).get('snapshot_revision') or '')
                    if prior_snapshot_revision:
                        _, snapshot_error = resolve_snapshot_root(cfg, state_store, prior_snapshot_revision)
                        prior_snapshot_available = snapshot_error is None
                    else:
                        prior_snapshot_available = False
                if (
                    receipt_skip_enabled
                    and source_fingerprint is not None
                    and prior_snapshot_available
                    and receipt_matches(
                        prior_receipt,
                        source_fingerprint,
                        process_config_hash=process_config_hash,
                    )
                ):
                    sources_skipped_unchanged += 1
                    trace.progress(trace_id, 'skip_unchanged_source', {
                        'source_key_hash': hashlib.sha256(source_key.encode('utf-8')).hexdigest()[:16],
                        'manifest_sha256': source_fingerprint.manifest_sha256,
                    })
                    continue

                stat_before: str | None = None
                receipt_stable = source_fingerprint is not None
                if source_fingerprint is not None:
                    try:
                        stat_before = source_stat_token(file)
                    except SourceFingerprintUnavailable:
                        receipt_stable = False

                try:
                    if isinstance(importer, SQLiteArchiveImporter):
                        accounts, conversations, messages = importer.load(limit=limit_per_sqlite)
                    elif isinstance(importer, WeChatDecryptedAccountImporter):
                        accounts, conversations, messages = importer.load(limit_per_shard=limit_per_sqlite)
                        if limit_per_sqlite is None:
                            import_waterline_updates.update(importer.last_waterline_updates)
                        for key, value in importer.last_scope_counts.items():
                            scope_counts[key] = scope_counts.get(key, 0) + value
                        for key, value in importer.last_excluded_counts.items():
                            excluded_counts[key] = excluded_counts.get(key, 0) + value
                    else:
                        accounts, conversations, messages = importer.load()

                    source_snapshot: SourceSnapshot | None = None
                    prepared_auxiliary: PreparedAuxiliaryImport | None = None
                    discovered_media_refs: tuple[MediaReference, ...] = ()
                    media_source_states: tuple[dict[str, Any], ...] = ()
                    account_id: str | None = None
                    account_hash_value: str | None = None
                    if isinstance(importer, WeChatDecryptedAccountImporter):
                        account_id = importer.account_id
                        account_hash_value = account_dir_hash(file)
                        snapshot_root = file.resolve().parent
                        source_snapshot = snapshot_cache.get(snapshot_root)
                        if source_snapshot is None:
                            source_snapshot = inspect_source_snapshot(cfg, snapshot_root)
                            snapshot_cache[snapshot_root] = source_snapshot
                        try:
                            prepared_auxiliary = prepare_auxiliary_sources(
                                file,
                                account_id=account_id,
                                limit=limit_per_sqlite,
                            )
                        except Exception as aux_exc:
                            errors.append(f'{file.name}: {aux_exc.__class__.__name__}: {aux_exc}')
                        try:
                            media_discovery = discover_media_assets_delta(
                                file,
                                store=scan_store,
                                account_id=account_id,
                                limit_per_table=limit_per_sqlite,
                            )
                            discovered_media_refs = tuple(media_discovery.refs)
                            media_source_states = tuple(media_discovery.source_states)
                        except Exception as link_exc:
                            coverage_gaps.append({
                                'source': file.name,
                                'reason': f'media link inventory skipped: {link_exc.__class__.__name__}',
                            })

                    prepared_sources.append(_PreparedImportSource(
                        file=Path(file),
                        source_key=source_key,
                        accounts=list(accounts),
                        conversations=list(conversations),
                        messages=list(messages),
                        source_fingerprint=source_fingerprint,
                        source_stat_before=stat_before,
                        source_snapshot=source_snapshot,
                        auxiliary=prepared_auxiliary,
                        message_media_refs=tuple(_message_media_references_for_messages(list(messages))),
                        discovered_media_refs=discovered_media_refs,
                        media_source_states=media_source_states,
                        account_id=account_id,
                        account_hash=account_hash_value,
                        receipt_stable=receipt_stable,
                    ))
                except Exception as exc:
                    errors.append(f'{file.name}: {exc.__class__.__name__}: {exc}')

        # Replace the former second byte-for-byte hash with a metadata-only
        # tree comparison.  It runs outside the writer and only controls
        # whether a completed receipt may be published.
        for item in prepared_sources:
            if item.source_fingerprint is None:
                continue
            try:
                stat_after = source_stat_token(item.file)
            except SourceFingerprintUnavailable:
                stat_after = None
            if not item.receipt_stable or stat_after != item.source_stat_before:
                item.receipt_stable = False
                errors.append(
                    'source_changed_during_import:'
                    + hashlib.sha256(item.source_key.encode('utf-8')).hexdigest()[:16]
                )
    finally:
        if scan_store is not state_store:
            scan_store.close_all()
        state_store.close_all()
        if temporary_scan_dir is not None:
            temporary_scan_dir.cleanup()

    return _PreparedImportJob(
        started_at=started_at,
        effective_config=effective_config,
        expected_generation=expected_generation,
        reset_index=reset_index,
        limit_per_sqlite=limit_per_sqlite,
        trace=trace,
        trace_id=trace_id,
        import_receipts=import_receipts,
        sources=prepared_sources,
        sources_seen=sources_seen,
        sources_skipped_unchanged=sources_skipped_unchanged,
        scope_counts=scope_counts,
        excluded_counts=excluded_counts,
        coverage_gaps=coverage_gaps,
        errors=errors,
        import_waterline_updates=import_waterline_updates,
    )


@mutation_entrypoint('full_import')
def run_import_job(
    vault_root: Path,
    sources: list[Path],
    *,
    reset_index: bool = False,
    limit_per_sqlite: int | None = None,
    process_config: ImportProcessConfig | None = None,
    force_rescan: bool = False,
    write_session: VaultWriteSession | None = None,
) -> ImportJobResult:
    started = time.time()
    if type(force_rescan) is not bool:
        raise TypeError('force_rescan must be an exact boolean')
    if write_session is not None:
        raise ValueError('full import preparation cannot run inside an active parent writer')
    cfg = VaultConfig.resolve(str(vault_root), env={})
    effective_config = _resolve_import_process_config(cfg.root, process_config)
    try:
        # Keep admission/schema recovery short.  No source traversal,
        # hashing, parsing, or media discovery runs in this writer window.
        with coordinated_vault_mutation(
            cfg,
            operation='full_import',
            write_session=write_session,
        ):
            cfg.ensure()
            admission_store = SQLiteStore(cfg.paths.sqlite_path)
            try:
                expected_generation = change_journal.recover_sync_commit_generation(admission_store)
            finally:
                admission_store.close_all()

        trace = TraceTimeline(cfg.root)
        trace_id = trace.start('import', {
            'sources_count': len(sources),
            'process_config_id': effective_config.config_id,
        })
        config_errors = effective_config.validate()
        if config_errors:
            trace.fail(trace_id, {'errors': config_errors})
            now = time.time()
            return ImportJobResult(
                'failed', str(cfg.root), 0, 0, 0, 0, 0, 0, 0, 0,
                {}, {}, [], 0, config_errors[:50], started, now,
                effective_config.config_id, effective_config.redacted_hash(), trace_id,
            )

        prepared = _prepare_import_job(
            cfg,
            [Path(source) for source in sources],
            started_at=started,
            effective_config=effective_config,
            expected_generation=expected_generation,
            reset_index=reset_index,
            limit_per_sqlite=limit_per_sqlite,
            force_rescan=force_rescan,
            trace=trace,
            trace_id=trace_id,
        )

        # CAS is the first database mutation in the final publication window.
        with coordinated_vault_mutation(
            cfg,
            operation='full_import',
            write_session=write_session,
        ) as active_session:
            return _publish_import_job(
                cfg,
                prepared,
                reset_index=reset_index,
                write_session=active_session,
            )
    except VaultOperationLocked:
        return ImportJobResult(
            status='locked',
            vault=str(cfg.root),
            sources_seen=0,
            sources_imported=0,
            messages_imported=0,
            contacts_imported=0,
            moments_imported=0,
            favorites_imported=0,
            media_links_accepted=0,
            media_links_excluded=0,
            scope_counts={},
            excluded_counts={},
            coverage_gaps=[],
            changed=0,
            errors=['VaultOperationLocked'],
            started_at=started,
            completed_at=time.time(),
            process_config_id=effective_config.config_id,
            process_config_hash=effective_config.redacted_hash(),
        )


def _publish_import_job(
    cfg: VaultConfig,
    prepared: _PreparedImportJob,
    *,
    reset_index: bool,
    write_session: VaultWriteSession,
) -> ImportJobResult:
    """Publish prepared rows after a shared-generation CAS."""

    write_session.validate_for(cfg)
    effective_config = prepared.effective_config
    process_config_hash = effective_config.redacted_hash()
    trace = prepared.trace
    trace_id = prepared.trace_id
    started = prepared.started_at
    limit_per_sqlite = prepared.limit_per_sqlite
    store = SQLiteStore(cfg.paths.sqlite_path)

    # This must remain the first target-database mutation.  If another sync or
    # import published while sources were being prepared, none of these rows
    # are allowed to reach the Vault.
    claimed_generation = change_journal.claim_sync_commit_generation(
        store,
        prepared.expected_generation,
    )
    if claimed_generation is None:
        result = ImportJobResult(
            status='retry_required',
            vault=str(cfg.root),
            sources_seen=prepared.sources_seen,
            sources_imported=0,
            messages_imported=0,
            contacts_imported=0,
            moments_imported=0,
            favorites_imported=0,
            media_links_accepted=0,
            media_links_excluded=0,
            scope_counts=prepared.scope_counts,
            excluded_counts=prepared.excluded_counts,
            coverage_gaps=[],
            changed=0,
            errors=['sync_commit_generation_changed'],
            started_at=started,
            completed_at=time.time(),
            process_config_id=effective_config.config_id,
            process_config_hash=process_config_hash,
            trace_id=trace_id,
            sources_skipped_unchanged=prepared.sources_skipped_unchanged,
        )
        trace.fail(trace_id, {'status': result.status, 'errors': result.errors})
        return result

    if reset_index:
        # The old-database CAS happens before destructive replacement. Restore
        # the even base generation in the new DB, then recreate the same odd
        # claim so completion stays monotonic and stale preparers cannot ABA.
        store.close_all()
        reset_index_cache(cfg.root, write_session=write_session)
        store = SQLiteStore(cfg.paths.sqlite_path)
        change_journal.restore_sync_commit_generation_after_reset(
            store,
            prepared.expected_generation,
        )
        restored_claim = change_journal.claim_sync_commit_generation(
            store,
            prepared.expected_generation,
        )
        if restored_claim != claimed_generation:
            raise RuntimeError('failed to restore full import publication claim')
        # A replacement index must not inherit completed receipts from the DB
        # generation that was just removed.
        write_import_receipts(cfg.root, {}, write_session=write_session)

    # Publish the process config only after the CAS, so a stale prepared import
    # cannot overwrite configuration while its database rows are rejected.
    write_process_config(cfg.root, effective_config, write_session=write_session)

    repo = WeChatRepository(store)
    multimodal_repo = MultimodalRepository(store)
    import_receipts = dict(prepared.import_receipts)
    receipt_updates: dict[str, dict[str, object]] = {}
    sources_seen = prepared.sources_seen
    sources_imported = 0
    sources_skipped_unchanged = prepared.sources_skipped_unchanged
    messages_imported = 0
    contacts_imported = 0
    moments_imported = 0
    favorites_imported = 0
    media_links_accepted = 0
    media_links_excluded = 0
    scope_counts = dict(prepared.scope_counts)
    excluded_counts = dict(prepared.excluded_counts)
    coverage_gaps = list(prepared.coverage_gaps)
    errors = list(prepared.errors)
    changed = 0
    changed_conversations: set[tuple[str, str]] = set()
    message_chunk_reports: list[dict[str, object]] = []
    message_dirty_count = 0
    changed_media_asset_ids: set[str] = set()
    changed_evidence_families: dict[str, int] = {
        'contact': 0,
        'moment': 0,
        'favorite': 0,
        'transcript': 0,
        'image_observation': 0,
    }
    aux_dirty_refs: list[dict[str, str]] = []
    aux_projection_citations: dict[str, set[str]] = {}
    aux_removed_refs: list[dict[str, str]] = []

    for item in prepared.sources:
        source_contacts = 0
        source_moments = 0
        source_favorites = 0
        try:
            if item.source_snapshot is not None:
                persist_source_snapshot(store, item.source_snapshot)

            changed_citation_set: set[str] = set()
            if item.accounts or item.conversations or item.messages or limit_per_sqlite is None:
                delta = repo.apply_delta(
                    item.accounts,
                    item.conversations,
                    item.messages,
                    source_key=item.source_key,
                    source_snapshot_complete=limit_per_sqlite is None,
                    max_chars=effective_config.chunk_max_chars,
                    overlap_chars=effective_config.chunk_overlap_chars,
                )
                changed_refs = list(delta.get('changed_refs') or ())
                changed_now = int(delta.get('messages_changed') or 0)
                changed += changed_now
                messages_imported += changed_now
                message_dirty_count += int(delta.get('dirty_recorded') or 0)
                if int(delta.get('citations_changed') or 0):
                    message_chunk_reports.append(dict(delta.get('chunks') or {}))
                tombstones = set(delta.get('tombstone_citations') or ())
                changed_citation_set = {
                    str(ref.get('citation'))
                    for ref in changed_refs
                    if ref.get('citation') and ref.get('citation') not in tombstones
                }
                try:
                    media_report = MediaLinker(multimodal_repo).link_references([
                        ref for ref in item.message_media_refs
                        if ref.citation in changed_citation_set
                    ])
                    media_links_accepted += media_report.accepted_links
                    media_links_excluded += media_report.excluded_links
                    changed_media_asset_ids.update(media_report.changed_asset_ids)
                    for key, value in media_report.excluded_counts.items():
                        excluded_counts[key] = excluded_counts.get(key, 0) + value
                except Exception as media_link_exc:
                    coverage_gaps.append({
                        'source': item.file.name,
                        'reason': f'message media registration skipped: {media_link_exc.__class__.__name__}',
                    })
                for ref in changed_refs:
                    if ref.get('account_id') and ref.get('conversation_id'):
                        changed_conversations.add((str(ref['account_id']), str(ref['conversation_id'])))

            if item.auxiliary is not None:
                aux_report = commit_prepared_auxiliary_sources(
                    item.auxiliary,
                    store=store,
                    repo=multimodal_repo,
                    bind_source=False,
                )
                source_contacts = aux_report.contacts_imported
                source_moments = aux_report.moments_imported
                source_favorites = aux_report.favorites_imported
                contacts_imported += source_contacts
                moments_imported += source_moments
                favorites_imported += source_favorites
                aux_dirty_refs.extend(aux_report.dirty_refs())
                for family, citations in aux_report.changed_citations.items():
                    aux_projection_citations.setdefault(family, set()).update(citations)
                for family, citations in aux_report.removed_citations.items():
                    aux_projection_citations.setdefault(family, set()).update(citations)
                    aux_removed_refs.extend({
                        'citation': citation,
                        'account_id': item.account_id or '',
                        'conversation_id': '',
                        'source_type': family,
                    } for citation in citations)
                for family, count in aux_report.changed_families().items():
                    changed_evidence_families[family] += count
                for key, value in aux_report.scope_counts.items():
                    scope_counts[key] = scope_counts.get(key, 0) + value
                for key, value in aux_report.excluded_counts.items():
                    excluded_counts[key] = excluded_counts.get(key, 0) + value

            if item.discovered_media_refs or item.media_source_states:
                try:
                    link_report = MediaLinker(multimodal_repo).link_references(
                        item.discovered_media_refs,
                        source_states=item.media_source_states,
                    )
                    media_links_accepted += link_report.accepted_links
                    media_links_excluded += link_report.excluded_links
                    changed_media_asset_ids.update(link_report.changed_asset_ids)
                    for key, value in link_report.excluded_counts.items():
                        excluded_counts[key] = excluded_counts.get(key, 0) + value
                except Exception as link_exc:
                    coverage_gaps.append({
                        'source': item.file.name,
                        'reason': f'media link inventory skipped: {link_exc.__class__.__name__}',
                    })

            if (
                item.source_snapshot is not None
                and item.account_id is not None
                and item.account_hash is not None
            ):
                try:
                    bind_account_assets(
                        store,
                        account_id=item.account_id,
                        snapshot=item.source_snapshot,
                        account_hash=item.account_hash,
                    )
                except Exception as binding_exc:
                    coverage_gaps.append({
                        'source': item.file.name,
                        'reason': f'media source binding skipped: {binding_exc.__class__.__name__}',
                    })

            if item.messages or source_contacts or source_moments or source_favorites:
                sources_imported += 1
            if (
                item.source_fingerprint is not None
                and item.receipt_stable
                and limit_per_sqlite is None
            ):
                receipt_updates[item.source_key] = completed_receipt(
                    item.source_fingerprint,
                    process_config_hash=process_config_hash,
                    snapshot_revision=(
                        item.source_snapshot.snapshot_revision
                        if item.source_snapshot is not None else None
                    ),
                )
        except Exception as exc:
            errors.append(f'{item.file.name}: {exc.__class__.__name__}: {exc}')

    # Message chunks were published atomically with each exact citation delta.
    # Auxiliary projections likewise refresh only exact changed/tombstoned
    # citations; family-wide rebuilds are reserved for explicit metadata repair.
    projection_errors: list[str] = []
    try:
        media_queue_report = enqueue_media_jobs(
            store,
            modalities={'voice'},
            asset_ids=tuple(sorted(changed_media_asset_ids)),
        )
        trace.append(
            'media_jobs',
            'complete',
            {
                'candidates': len(changed_media_asset_ids),
                'queued': int(media_queue_report.get('queued') or 0),
                'skipped_out_of_scope': int(media_queue_report.get('skipped_out_of_scope') or 0),
            },
            trace_id=trace_id,
        )
    except Exception as media_queue_exc:
        error_code = media_queue_exc.__class__.__name__
        projection_errors.append(f'media_jobs_failed:{error_code}')
        trace.append('media_jobs', 'fail', {'error_code': error_code}, trace_id=trace_id)
    try:
        zero_report = {
            'parents': 0,
            'chunks': 0,
            'conversations': 0,
            'source_types': [],
            'max_chars': effective_config.chunk_max_chars,
            'overlap_chars': effective_config.chunk_overlap_chars,
        }
        message_chunk_report = {
            **zero_report,
            'parents': sum(int(item.get('parents') or 0) for item in message_chunk_reports),
            'chunks': sum(int(item.get('chunks') or 0) for item in message_chunk_reports),
            'citations': sum(int(item.get('citations') or 0) for item in message_chunk_reports),
            'conversations': len(changed_conversations),
        }
        family_reports: list[dict[str, object]] = []
        for family, citations in sorted(aux_projection_citations.items()):
            if citations:
                family_reports.append(store.rebuild_evidence_chunks_for_source_citations(
                    family,
                    sorted(citations),
                    max_chars=effective_config.chunk_max_chars,
                    overlap_chars=effective_config.chunk_overlap_chars,
                ))
        family_chunk_report = {
            **zero_report,
            'parents': sum(int(item.get('parents') or 0) for item in family_reports),
            'chunks': sum(int(item.get('chunks') or 0) for item in family_reports),
            'citations': sum(int(item.get('citations') or 0) for item in family_reports),
        }
        changed_families = {
            family: count
            for family, count in changed_evidence_families.items()
            if count > 0
        }
        if aux_removed_refs:
            store.record_citation_tombstones(aux_removed_refs)
        removed_citations = {ref['citation'] for ref in aux_removed_refs}
        live_aux_citations = {
            citation
            for citations in aux_projection_citations.values()
            for citation in citations
            if citation not in removed_citations
        }
        if live_aux_citations:
            store.clear_citation_tombstones(live_aux_citations)
        chunk_report = {
            'parents': int(message_chunk_report.get('parents', 0)) + int(family_chunk_report.get('parents', 0)),
            'chunks': int(message_chunk_report.get('chunks', 0)) + int(family_chunk_report.get('chunks', 0)),
            'conversations': int(message_chunk_report.get('conversations', 0)),
            'source_types': sorted(changed_families),
            'family_changed_counts': changed_families,
            'message': message_chunk_report,
            'non_message': family_chunk_report,
            'max_chars': effective_config.chunk_max_chars,
            'overlap_chars': effective_config.chunk_overlap_chars,
        }
        trace.append('chunking', 'complete', chunk_report, trace_id=trace_id)
    except Exception as chunk_exc:
        error_code = chunk_exc.__class__.__name__
        projection_errors.append(f'chunking_failed:{error_code}')
        trace.append('chunking', 'fail', {'error_code': error_code}, trace_id=trace_id)
    try:
        if aux_dirty_refs:
            dirty_count = message_dirty_count + change_journal.record_dirty_citations(store, aux_dirty_refs)
        else:
            dirty_count = message_dirty_count
        trace.append('dirty_citations', 'complete', {'count': dirty_count}, trace_id=trace_id)
    except Exception as dirty_exc:
        error_code = dirty_exc.__class__.__name__
        projection_errors.append(f'dirty_citations_failed:{error_code}')
        trace.append('dirty_citations', 'fail', {'error_code': error_code}, trace_id=trace_id)

    waterlines_written = 0
    if not errors and not projection_errors and not coverage_gaps and limit_per_sqlite is None:
        try:
            waterlines_written = change_journal.write_waterlines(
                store,
                prepared.import_waterline_updates,
            )
            trace.append('sync_waterlines', 'complete', {'count': waterlines_written}, trace_id=trace_id)
        except Exception as waterline_exc:
            error_code = waterline_exc.__class__.__name__
            projection_errors.append(f'sync_waterlines_failed:{error_code}')
            trace.append('sync_waterlines', 'fail', {'error_code': error_code}, trace_id=trace_id)

    all_errors = (errors + projection_errors)[:50]
    imported_anything = bool(
        sources_imported or changed or contacts_imported or moments_imported or favorites_imported
    )
    if projection_errors:
        status = 'degraded' if imported_anything else 'failed'
    elif errors or coverage_gaps:
        status = 'partial' if imported_anything else 'failed'
    else:
        status = 'completed'
    result = ImportJobResult(
        status=status,
        vault=str(cfg.root),
        sources_seen=sources_seen,
        sources_imported=sources_imported,
        messages_imported=messages_imported,
        contacts_imported=contacts_imported,
        moments_imported=moments_imported,
        favorites_imported=favorites_imported,
        media_links_accepted=media_links_accepted,
        media_links_excluded=media_links_excluded,
        scope_counts=scope_counts,
        excluded_counts=excluded_counts,
        coverage_gaps=coverage_gaps[:50],
        changed=changed,
        errors=all_errors,
        started_at=started,
        completed_at=time.time(),
        process_config_id=effective_config.config_id,
        process_config_hash=process_config_hash,
        trace_id=trace_id,
        sources_skipped_unchanged=sources_skipped_unchanged,
        waterlines_updated=waterlines_written,
    )
    change_journal.complete_sync_commit_generation(store, claimed_generation)
    if status == 'completed' and limit_per_sqlite is None:
        import_receipts.update(receipt_updates)
        write_import_receipts(cfg.root, import_receipts, write_session=write_session)
    write_import_result(cfg.root, result, write_session=write_session)
    terminal_payload = {
        'status': status,
        'sources_imported': sources_imported,
        'sources_skipped_unchanged': sources_skipped_unchanged,
        'messages_imported': messages_imported,
        'errors': all_errors[:10],
    }
    if status in {'completed', 'partial'}:
        trace.complete(trace_id, terminal_payload)
    else:
        trace.fail(trace_id, terminal_payload)
    return result


def write_import_result(
    vault_root: Path,
    result: ImportJobResult,
    *,
    write_session: VaultWriteSession,
) -> None:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    write_session.validate_for(cfg)
    cfg.ensure()
    path = cfg.paths.jobs_dir / 'last_import.json'
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
