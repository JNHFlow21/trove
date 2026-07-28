from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from typing import Any

from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.importers.contacts import ContactIdentityImporter
from trove_core.wechat.importers.favorites import FavoritesImporter
from trove_core.wechat.importers.moments import MomentsImporter


AUXILIARY_SOURCE_TYPES = {'contact', 'moment', 'favorite'}


def _signature_hash(values: list[object]) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    return _signature_hash([path.name, int(stat.st_size), int(stat.st_mtime_ns)])


def auxiliary_source_fingerprints(account_dir: Path, *, account_id: str) -> dict[str, str]:
    """Return stable fingerprints for auxiliary DB files present in an account dir."""
    account_dir = Path(account_dir)
    out: dict[str, str] = {}
    for name in ('contact.db',):
        path = account_dir / name
        if path.exists():
            out[f'{account_id}:{name}'] = _source_fingerprint(path)
    for name in ('sns.db', 'moment.db', 'moments.db'):
        path = account_dir / name
        if path.exists():
            out[f'{account_id}:{name}'] = _source_fingerprint(path)
            break
    for name in ('favorite.db', 'favorites.db', 'fav.db'):
        path = account_dir / name
        if path.exists():
            out[f'{account_id}:{name}'] = _source_fingerprint(path)
            break
    return out


def family_for_auxiliary_source_key(source_key: str) -> str | None:
    name = str(source_key).rsplit(':', 1)[-1].lower()
    if name == 'contact.db':
        return 'contact'
    if name in {'sns.db', 'moment.db', 'moments.db'}:
        return 'moment'
    if name in {'favorite.db', 'favorites.db', 'fav.db'}:
        return 'favorite'
    return None


def family_signature(store: SQLiteStore, source_type: str, *, account_id: str | None = None) -> dict[str, str]:
    """Return a stable per-parent-citation signature for non-message evidence.

    The key is the parent citation that vector indexing can accept as a dirty
    citation.  Contacts can have several observation rows for the same contact
    citation, so aggregate all rows per citation before hashing.
    """
    store.initialize()
    if not store.path.exists():
        return {}
    source_type = str(source_type)
    with store.connect() as conn:
        if source_type == 'favorite':
            if not store._table_exists(conn, 'favorites'):
                return {}
            where = ' WHERE account_id=?' if account_id else ''
            params = [account_id] if account_id else []
            return {
                row['citation']: _signature_hash([row['account_id'], row['favorite_id'], row['title'], row['timestamp'], row['text'], row['media_refs_json'], row['metadata_json']])
                for row in conn.execute(f'SELECT citation,account_id,favorite_id,title,timestamp,text,media_refs_json,metadata_json FROM favorites{where} ORDER BY citation', params)
            }
        if source_type == 'moment':
            rows: dict[str, str] = {}
            where = ' WHERE account_id=?' if account_id else ''
            params = [account_id] if account_id else []
            if store._table_exists(conn, 'moment_items'):
                rows.update({
                    row['citation']: _signature_hash([row['account_id'], row['moment_id'], row['author_id'], row['timestamp'], row['text'], row['link_json'], row['media_refs_json'], row['comments_json'], row['metadata_json']])
                    for row in conn.execute(f'SELECT citation,account_id,moment_id,author_id,timestamp,text,link_json,media_refs_json,comments_json,metadata_json FROM moment_items{where} ORDER BY citation', params)
                })
            if store._table_exists(conn, 'moment_interactions'):
                rows.update({
                    row['citation']: _signature_hash([row['account_id'], row['interaction_id'], row['interaction_type'], row['actor_id'], row['actor_name'] if 'actor_name' in row.keys() else '', row['timestamp'], row['text'], row['metadata_json']])
                    for row in conn.execute(f'SELECT citation,account_id,interaction_id,interaction_type,actor_id,actor_name,timestamp,text,metadata_json FROM moment_interactions{where} ORDER BY citation', params)
                })
            return rows
        if source_type == 'contact':
            if not store._table_exists(conn, 'observations'):
                return {}
            grouped: dict[str, list[list[Any]]] = {}
            where = 'source_type="contact"'
            params: list[Any] = []
            if account_id:
                where += ' AND citation LIKE ?'
                params.append(f'trove://wechat/{account_id}/contact/%')
            for row in conn.execute(f'SELECT citation,entity_id,observation_id,observation_type,value_json,status,confidence FROM observations WHERE {where} ORDER BY citation,observation_id', params):
                grouped.setdefault(row['citation'], []).append([
                    row['entity_id'], row['observation_id'], row['observation_type'], row['value_json'], row['status'], row['confidence'],
                ])
            return {citation: _signature_hash(values) for citation, values in grouped.items()}
    return {}


def changed_citations(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(citation for citation, digest in after.items() if before.get(citation) != digest)


def removed_citation_count(before: dict[str, str], after: dict[str, str]) -> int:
    return sum(1 for citation in before if citation not in after)


@dataclass
class AuxiliaryImportReport:
    contacts_imported: int = 0
    moments_imported: int = 0
    favorites_imported: int = 0
    changed_citations: dict[str, list[str]] = field(default_factory=dict)
    removed_citations: dict[str, list[str]] = field(default_factory=dict)
    removed_counts: dict[str, int] = field(default_factory=dict)
    source_fingerprints: dict[str, str] = field(default_factory=dict)
    scope_counts: dict[str, int] = field(default_factory=dict)
    excluded_counts: dict[str, int] = field(default_factory=dict)

    def changed_count(self, source_type: str) -> int:
        return len(self.changed_citations.get(source_type, [])) + int(self.removed_counts.get(source_type, 0))

    def changed_families(self) -> dict[str, int]:
        return {
            source_type: self.changed_count(source_type)
            for source_type in sorted(AUXILIARY_SOURCE_TYPES)
            if self.changed_count(source_type) > 0
        }

    def dirty_refs(self) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []
        for source_type, citations in self.changed_citations.items():
            for citation in citations:
                refs.append({'citation': citation, 'account_id': '', 'conversation_id': '', 'source_type': source_type})
        for source_type, citations in self.removed_citations.items():
            for citation in citations:
                refs.append({'citation': citation, 'account_id': '', 'conversation_id': '', 'source_type': source_type})
        return refs

    def to_dict(self) -> dict[str, Any]:
        return {
            'contacts_imported': self.contacts_imported,
            'moments_imported': self.moments_imported,
            'favorites_imported': self.favorites_imported,
            'changed_families': self.changed_families(),
            'dirty_citations': {k: len(v) for k, v in sorted(self.changed_citations.items())},
            'removed_counts': dict(sorted(self.removed_counts.items())),
            'sources_fingerprinted': len(self.source_fingerprints),
            'scope_counts': dict(sorted(self.scope_counts.items())),
            'excluded_counts': dict(sorted(self.excluded_counts.items())),
            'raw_content_included': False,
            'raw_paths_included': False,
        }


@dataclass
class PreparedAuxiliaryImport:
    """Auxiliary rows parsed from source SQLite files without Vault writes."""

    account_dir: Path
    account_id: str
    limit: int | None
    contact_importer: ContactIdentityImporter | None = None
    moment_importer: MomentsImporter | None = None
    favorite_importer: FavoritesImporter | None = None
    source_fingerprints: dict[str, str] = field(default_factory=dict)


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _record_family_change(report: AuxiliaryImportReport, source_type: str, before: dict[str, str], after: dict[str, str]) -> None:
    changed = changed_citations(before, after)
    removed_citations = sorted(citation for citation in before if citation not in after)
    removed = len(removed_citations)
    if changed:
        report.changed_citations.setdefault(source_type, []).extend(changed)
    if removed:
        report.removed_citations.setdefault(source_type, []).extend(removed_citations)
        report.removed_counts[source_type] = report.removed_counts.get(source_type, 0) + removed


def _delete_not_in(conn, table: str, where: str, params: list[Any]) -> None:
    conn.execute(f'DELETE FROM {table} WHERE {where}', params)


def _prepare_keep_table(conn, citations: set[str]) -> None:
    conn.execute('CREATE TEMP TABLE IF NOT EXISTS _trove_aux_keep_citations(citation TEXT PRIMARY KEY)')
    conn.execute('DELETE FROM _trove_aux_keep_citations')
    if citations:
        conn.executemany(
            'INSERT OR IGNORE INTO _trove_aux_keep_citations(citation) VALUES(?)',
            [(citation,) for citation in sorted(citations)],
        )


def _moment_parent_citation_sql(column: str) -> str:
    """Resolve a Moment media citation to its indexed parent citation."""
    if column not in {
        'media_asset_links.source_citation',
        'sns_cache_mappings.source_citation',
    }:
        raise ValueError('unsupported Moment citation column')
    return f"""CASE
        WHEN instr({column}, '#image-') > 0
          THEN substr({column}, 1, instr({column}, '#image-') - 1)
        WHEN instr({column}, '#video-') > 0
          THEN substr({column}, 1, instr({column}, '#video-') - 1)
        ELSE {column}
    END"""


def _prune_contact_rows(store: SQLiteStore, account_id: str, keep_citations: set[str]) -> None:
    prefix = f'trove://wechat/{account_id}/contact/%'
    with store.connect() as conn:
        if not store._table_exists(conn, 'observations'):
            return
        _prepare_keep_table(conn, keep_citations)
        if keep_citations:
            _delete_not_in(
                conn,
                'observations',
                "source_type='contact' AND citation LIKE ? AND citation NOT IN (SELECT citation FROM _trove_aux_keep_citations)",
                [prefix],
            )
        else:
            _delete_not_in(conn, 'observations', "source_type='contact' AND citation LIKE ?", [prefix])
        conn.commit()


def _prune_favorite_rows(store: SQLiteStore, account_id: str, keep_citations: set[str]) -> None:
    with store.connect() as conn:
        if not store._table_exists(conn, 'favorites'):
            return
        _prepare_keep_table(conn, keep_citations)
        if keep_citations:
            _delete_not_in(
                conn,
                'favorites',
                'account_id=? AND citation NOT IN (SELECT citation FROM _trove_aux_keep_citations)',
                [account_id],
            )
        else:
            _delete_not_in(conn, 'favorites', 'account_id=?', [account_id])
        conn.commit()


def _prune_moment_rows(
    store: SQLiteStore,
    account_id: str,
    keep_moment_citations: set[str],
    keep_interaction_citations: set[str],
    keep_media_citations: set[str] | None = None,
) -> None:
    with store.connect() as conn:
        if store._table_exists(conn, 'moment_items'):
            _prepare_keep_table(conn, keep_moment_citations)
            if keep_moment_citations:
                _delete_not_in(
                    conn,
                    'moment_items',
                    'account_id=? AND citation NOT IN (SELECT citation FROM _trove_aux_keep_citations)',
                    [account_id],
                )
            else:
                _delete_not_in(conn, 'moment_items', 'account_id=?', [account_id])
        if store._table_exists(conn, 'moment_interactions'):
            _prepare_keep_table(conn, keep_interaction_citations)
            if keep_interaction_citations:
                _delete_not_in(
                    conn,
                    'moment_interactions',
                    'account_id=? AND citation NOT IN (SELECT citation FROM _trove_aux_keep_citations)',
                    [account_id],
                )
            else:
                _delete_not_in(conn, 'moment_interactions', 'account_id=?', [account_id])
        if store._table_exists(conn, 'media_asset_links'):
            _prepare_keep_table(conn, keep_media_citations or set())
            if keep_media_citations:
                _delete_not_in(
                    conn,
                    'media_asset_links',
                    "account_id=? AND source_type='moment' AND source_citation NOT IN (SELECT citation FROM _trove_aux_keep_citations)",
                    [account_id],
                )
            else:
                _delete_not_in(conn, 'media_asset_links', "account_id=? AND source_type='moment'", [account_id])
            parent_citation = _moment_parent_citation_sql('media_asset_links.source_citation')
            conn.execute(
                f"""DELETE FROM media_asset_links
                    WHERE source_type='moment'
                      AND NOT EXISTS (
                        SELECT 1 FROM moment_items mi
                        WHERE mi.citation=({parent_citation})
                      )"""
            )
        if store._table_exists(conn, 'sns_cache_mappings'):
            _prepare_keep_table(conn, keep_media_citations or set())
            if keep_media_citations:
                _delete_not_in(
                    conn,
                    'sns_cache_mappings',
                    'account_id=? AND source_citation NOT IN (SELECT citation FROM _trove_aux_keep_citations)',
                    [account_id],
                )
            else:
                _delete_not_in(conn, 'sns_cache_mappings', 'account_id=?', [account_id])
            parent_citation = _moment_parent_citation_sql('sns_cache_mappings.source_citation')
            conn.execute(
                f"""DELETE FROM sns_cache_mappings
                    WHERE NOT EXISTS (
                      SELECT 1 FROM moment_items mi
                      WHERE mi.citation=({parent_citation})
                    )"""
            )
        if store._table_exists(conn, 'media_assets'):
            if store._table_exists(conn, 'media_asset_links'):
                _delete_not_in(
                    conn,
                    'media_assets',
                    "account_id=? AND source_type='moment' AND asset_id NOT IN (SELECT asset_id FROM media_asset_links WHERE account_id=? AND source_type='moment')",
                    [account_id, account_id],
                )
                conn.execute(
                    """DELETE FROM media_assets
                       WHERE source_type='moment'
                         AND asset_id NOT IN (SELECT asset_id FROM media_asset_links WHERE source_type='moment')"""
                )
            else:
                _delete_not_in(conn, 'media_assets', "account_id=? AND source_type='moment'", [account_id])
        conn.commit()


def prepare_auxiliary_sources(
    account_dir: Path,
    *,
    account_id: str,
    limit: int | None = None,
    only: set[str] | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> PreparedAuxiliaryImport:
    """Parse changed auxiliary source families without taking the Vault writer."""

    account_dir = Path(account_dir)
    prepared = PreparedAuxiliaryImport(account_dir=account_dir, account_id=account_id, limit=limit)
    selected = None if only is None else {str(family) for family in only if str(family) in AUXILIARY_SOURCE_TYPES}
    overrides = {
        str(family): Path(path)
        for family, path in (source_overrides or {}).items()
        if str(family) in AUXILIARY_SOURCE_TYPES
    }

    def should_import(family: str) -> bool:
        return selected is None or family in selected

    contact_db = overrides.get('contact', account_dir / 'contact.db')
    if contact_db.exists() and should_import('contact'):
        cimp = ContactIdentityImporter(contact_db, account_id=account_id)
        cimp.load(limit=limit)
        prepared.contact_importer = cimp
        prepared.source_fingerprints[f'{account_id}:{contact_db.name}'] = _source_fingerprint(contact_db)

    moment_sources = (
        [(overrides['moment'].name, overrides['moment'])]
        if 'moment' in overrides
        else [(name, account_dir / name) for name in ('sns.db', 'moment.db', 'moments.db')]
    )
    for sns_name, sns_db in moment_sources:
        if sns_db.exists() and should_import('moment'):
            mimp = MomentsImporter(sns_db, account_id=account_id)
            mimp.load(limit=limit)
            prepared.moment_importer = mimp
            prepared.source_fingerprints[f'{account_id}:{sns_name}'] = _source_fingerprint(sns_db)
            break

    favorite_sources = (
        [(overrides['favorite'].name, overrides['favorite'])]
        if 'favorite' in overrides
        else [(name, account_dir / name) for name in ('favorite.db', 'favorites.db', 'fav.db')]
    )
    for fav_name, fav_db in favorite_sources:
        if fav_db.exists() and should_import('favorite'):
            fimp = FavoritesImporter(fav_db, account_id=account_id)
            fimp.load(limit=limit)
            prepared.favorite_importer = fimp
            prepared.source_fingerprints[f'{account_id}:{fav_name}'] = _source_fingerprint(fav_db)
            break

    return prepared


def commit_prepared_auxiliary_sources(
    prepared: PreparedAuxiliaryImport,
    *,
    store: SQLiteStore,
    repo: MultimodalRepository,
    bind_source: bool = True,
) -> AuxiliaryImportReport:
    """Commit prepared auxiliary rows; source files are never scanned here."""

    report = AuxiliaryImportReport(source_fingerprints=dict(prepared.source_fingerprints))
    account_id = prepared.account_id
    limit = prepared.limit

    cimp = prepared.contact_importer
    if cimp is not None:
        before = family_signature(store, 'contact', account_id=account_id)
        report.contacts_imported += cimp.persist_loaded_to_ontology(repo)
        if limit is None:
            _prune_contact_rows(store, account_id, {contact.citation for contact in cimp.last_contacts})
        after = family_signature(store, 'contact', account_id=account_id)
        _record_family_change(report, 'contact', before, after)
        _merge_counts(report.scope_counts, cimp.last_scope_counts)
        _merge_counts(report.excluded_counts, cimp.last_excluded_counts)

    mimp = prepared.moment_importer
    if mimp is not None:
        before = family_signature(store, 'moment', account_id=account_id)
        imported = mimp.persist_loaded_to_store(repo, bind_source=bind_source)
        report.moments_imported += imported
        if limit is None:
            moments = mimp.last_moments
            _prune_moment_rows(
                store,
                account_id,
                {moment.citation for moment in moments},
                {interaction.citation for interaction in (mimp.last_interactions or [])},
                {ref.citation for ref in mimp._media_asset_refs(moments)},
            )
        after = family_signature(store, 'moment', account_id=account_id)
        _record_family_change(report, 'moment', before, after)
        report.scope_counts['moment'] = report.scope_counts.get('moment', 0) + imported

    fimp = prepared.favorite_importer
    if fimp is not None:
        before = family_signature(store, 'favorite', account_id=account_id)
        imported = fimp.persist_loaded_to_store(repo)
        report.favorites_imported += imported
        if limit is None:
            _prune_favorite_rows(store, account_id, {favorite.citation for favorite in fimp.last_favorites})
        after = family_signature(store, 'favorite', account_id=account_id)
        _record_family_change(report, 'favorite', before, after)
        report.scope_counts['favorite'] = report.scope_counts.get('favorite', 0) + imported

    for key, values in list(report.changed_citations.items()):
        report.changed_citations[key] = sorted(dict.fromkeys(values))
    for key, values in list(report.removed_citations.items()):
        report.removed_citations[key] = sorted(dict.fromkeys(values))
    return report


def import_auxiliary_sources(
    account_dir: Path,
    *,
    account_id: str,
    store: SQLiteStore,
    repo: MultimodalRepository,
    limit: int | None = None,
    only: set[str] | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> AuxiliaryImportReport:
    prepared = prepare_auxiliary_sources(
        account_dir,
        account_id=account_id,
        limit=limit,
        only=only,
        source_overrides=source_overrides,
    )
    return commit_prepared_auxiliary_sources(prepared, store=store, repo=repo)
