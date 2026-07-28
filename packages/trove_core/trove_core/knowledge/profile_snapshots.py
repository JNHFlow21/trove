from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from trove_core.knowledge.customer_profile import build_customer_profile
from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.knowledge.profile_enrichment import (
    CLOUD_ASR_PROVIDER_NAME,
    CLOUD_ASR_MODEL_ID,
    ProfileEnrichmentError,
    ProfileEnrichmentService,
)
from trove_core.store.sqlite_store import SQLiteStore


PROFILE_SNAPSHOT_SCHEMA = 'customer-profile/v2'
AUTOMATIC_PROFILE_SNAPSHOT_SCHEMA = 'customer-profile/auto-v1'
FINAL_STATES = {'complete', 'complete_with_terminal_gaps'}
_HTTP_URL_RE = re.compile(r'https?://[^\s)\]}]+', re.IGNORECASE)
_FILE_URL_RE = re.compile(r'file://[^\s)\]}]+', re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(r'/(?:Users|home|Volumes|private|tmp|var|opt)/[^\s)\]},;]+')
_WINDOWS_PATH_RE = re.compile(r'\b[A-Za-z]:\\[^\s)\]},;]+')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _sanitize_text(value: str) -> str:
    text = _HTTP_URL_RE.sub('[redacted-url]', str(value))
    text = _FILE_URL_RE.sub('[redacted-path]', text)
    text = _ABSOLUTE_PATH_RE.sub('[redacted-path]', text)
    return _WINDOWS_PATH_RE.sub('[redacted-path]', text)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or type(value) in {bool, int, float}:
        return value
    return _sanitize_text(str(value))


def _safe_profile_projection(profile: dict[str, Any]) -> dict[str, Any]:
    sections: dict[str, list[dict[str, Any]]] = {}
    for name, rows in (profile.get('sections') or {}).items():
        if name in {'pending_voice', 'ambiguities'} or not isinstance(rows, list):
            continue
        safe_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Automatic/formal snapshots are durable product output. Pending
            # hypotheses remain queryable in the live projection, but must not
            # silently become accepted facts in an immutable saved profile.
            status = row.get('status')
            if status is not None and str(status) != 'active':
                continue
            citations = [str(citation) for citation in (row.get('citations') or []) if str(citation).startswith('trove://')]
            if not citations:
                continue
            item = dict(row)
            item['citations'] = citations
            safe_rows.append(_sanitize(item))
        sections[str(name)] = safe_rows
    return {
        'type': 'customer_profile',
        'customer': _sanitize_text(str(profile.get('customer') or '')),
        'sections': sections,
        'claim_policy': str(profile.get('claim_policy') or ''),
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def _collect_citations(profile: dict[str, Any]) -> list[str]:
    citations: set[str] = set()
    for rows in (profile.get('sections') or {}).values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            citations.update(str(citation) for citation in (row.get('citations') or []) if str(citation).startswith('trove://'))
    return sorted(citations)


def _run_rows(store: SQLiteStore, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with store.connect() as conn:
        run = conn.execute('SELECT * FROM profile_enrichment_runs WHERE run_id=?', (run_id,)).fetchone()
        if run is None:
            raise ProfileEnrichmentError('enrichment run not found', code='enrichment_run_not_found')
        tasks = [dict(row) for row in conn.execute(
            'SELECT * FROM profile_enrichment_tasks WHERE run_id=? ORDER BY task_id', (run_id,),
        )]
    return dict(run), tasks


def _evidence_digests(store: SQLiteStore, tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    voice_asset_ids = sorted({
        str(task['asset_id']) for task in tasks
        if str(task.get('modality') or '') == 'voice' and task.get('asset_id')
    })
    visual_asset_ids = sorted({
        str(task['asset_id']) for task in tasks
        if str(task.get('modality') or '') in {'image', 'video'} and task.get('asset_id')
    })
    appmsg_citations = sorted({
        str(task['citation']) for task in tasks if str(task.get('modality') or '') == 'appmsg'
    })
    voice_rows: dict[str, dict[str, Any]] = {}
    visual_rows: dict[str, dict[str, Any]] = {}
    appmsg_rows: dict[str, dict[str, Any]] = {}
    with store.connect() as conn:
        if voice_asset_ids:
            rows = conn.execute(
                """WITH scoped(asset_id) AS (SELECT CAST(value AS TEXT) FROM json_each(?)),
                          ranked AS (
                            SELECT t.*,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY t.asset_id
                                       ORDER BY t.created_at DESC,t.transcript_id DESC
                                   ) AS rank
                              FROM transcripts t JOIN scoped s ON s.asset_id=t.asset_id
                              JOIN provider_jobs pj ON pj.job_id=t.job_id
                              JOIN media_assets ma ON ma.asset_id=t.asset_id
                             WHERE t.status='active' AND pj.provider=? AND pj.model=? AND pj.status='completed'
                               AND pj.request_hash=ma.content_hash
                          )
                    SELECT asset_id,citation,text,language,confidence,duration_seconds,status
                      FROM ranked WHERE rank=1""",
                (_json(voice_asset_ids), CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
            )
            for row in rows:
                payload = dict(row)
                voice_rows[str(payload.pop('asset_id'))] = payload
        if visual_asset_ids:
            rows = conn.execute(
                """WITH scoped(asset_id) AS (SELECT CAST(value AS TEXT) FROM json_each(?)),
                          ranked AS (
                            SELECT io.*,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY io.asset_id
                                       ORDER BY io.updated_at DESC,io.observation_id DESC
                                   ) AS rank
                              FROM image_observations io JOIN scoped s ON s.asset_id=io.asset_id
                             WHERE io.status='active'
                          )
                    SELECT asset_id,citation,caption,visible_text,objects_json,business_signals_json,
                           content_sha256,model_id,prompt_version,confidence,status
                      FROM ranked WHERE rank=1""",
                (_json(visual_asset_ids),),
            )
            for row in rows:
                payload = dict(row)
                visual_rows[str(payload.pop('asset_id'))] = payload
        if appmsg_citations:
            rows = conn.execute(
                """WITH scoped(citation) AS (SELECT CAST(value AS TEXT) FROM json_each(?))
                    SELECT mp.citation,mp.normalized_type,mp.parse_status,mp.normalized_json,
                           mp.display_text,mp.source_hash,mp.parser_version
                      FROM message_payloads mp JOIN scoped s ON s.citation=mp.citation""",
                (_json(appmsg_citations),),
            )
            appmsg_rows = {str(row['citation']): dict(row) for row in rows}
    values: list[dict[str, str]] = []
    for task in tasks:
        modality = str(task['modality'])
        row: dict[str, Any] | None = None
        if modality == 'voice' and task.get('asset_id'):
            row = voice_rows.get(str(task['asset_id']))
        elif modality in {'image', 'video'} and task.get('asset_id'):
            row = visual_rows.get(str(task['asset_id']))
        elif modality == 'appmsg':
            row = appmsg_rows.get(str(task['citation']))
        if row is not None:
            values.append({
                'citation': str(row['citation']),
                'modality': modality,
                'digest': _digest(row),
            })
    return sorted(values, key=lambda item: (item['citation'], item['modality'], item['digest']))


def _completeness(tasks: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family: dict[str, dict[str, int]] = {}
    gaps: list[dict[str, Any]] = []
    for task in tasks:
        modality = str(task['modality'])
        counters = family.setdefault(modality, {'eligible': 0, 'completed': 0, 'terminal_gaps': 0, 'pending': 0})
        counters['eligible'] += 1
        state = str(task['state'])
        if state == 'completed':
            counters['completed'] += 1
        elif state in {'unavailable', 'cancelled'}:
            counters['terminal_gaps'] += 1
            gaps.append({
                'citation': str(task['citation']),
                'modality': modality,
                'state': state,
                'reason': str(task.get('terminal_reason') or 'unavailable'),
                'source_revision': str(task['source_revision']),
            })
        else:
            counters['pending'] += 1
    for counters in family.values():
        eligible = counters['eligible']
        counters['coverage_ratio'] = round(counters['completed'] / eligible, 6) if eligible else 1.0
    return {'families': family, 'eligible': len(tasks), 'completed': sum(1 for task in tasks if task['state'] == 'completed')}, gaps


def _snapshot_content(store: SQLiteStore, run: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    with store.connect() as conn:
        entity = conn.execute('SELECT display_name FROM entities WHERE entity_id=?', (run['entity_id'],)).fetchone()
    customer = str(entity['display_name'] if entity is not None else run['entity_id'])
    profile = _safe_profile_projection(build_customer_profile(store, customer, limit=50))
    citations = _collect_citations(profile)
    completeness, gaps = _completeness(tasks)
    processors = sorted({
        f"{task.get('modality')}:{task.get('processor_identity') or ''}@{task.get('prompt_version') or ''}"
        for task in tasks
    })
    summary = {
        'state': str(run['state']),
        'completeness': completeness,
        'processor_versions': processors,
    }
    return {
        'schema_version': PROFILE_SNAPSHOT_SCHEMA,
        'canonical_entity_id': str(run['entity_id']),
        'source_revision': str(run['source_revision']),
        'profile': profile,
        'evidence_citations': citations,
        'evidence_digests': _evidence_digests(store, tasks),
        'enrichment_summary': summary,
        'unresolved_gaps': gaps,
        'raw_content_included': False,
        'raw_paths_included': False,
        'provider_payloads_included': False,
    }


def _canonical_entity_id(store: SQLiteStore, customer: str) -> tuple[str, dict[str, Any]]:
    resolution = resolve_customer(store, customer)
    resolved = resolution.get('resolved')
    entity_id = str((resolved or {}).get('entity_id') or '').strip()
    if not entity_id or entity_id.startswith('unresolved:'):
        code = 'customer_ambiguous' if resolution.get('ambiguous') else 'canonical_entity_required'
        raise ProfileEnrichmentError(
            'customer must resolve to a canonical entity', code=code,
        )
    return entity_id, resolved


def automatic_profile_snapshot_payload(
    store: SQLiteStore,
    *,
    entity_id: str,
    selector: str,
) -> dict[str, Any]:
    """Build one deterministic local-only profile projection for publication.

    Automatic maintenance never executes provider work. New media/appmsg work
    is represented as an explicit deferred gap while the latest cited text and
    already-cached structured evidence can still advance the formal snapshot.
    """

    profile = build_customer_profile(store, selector, limit=50)
    resolved_id = str((profile.get('resolved_entity') or {}).get('entity_id') or '')
    if resolved_id != entity_id:
        raise ProfileEnrichmentError(
            'automatic profile selector no longer resolves to its canonical entity',
            code='profile_subscription_identity_drift',
        )
    safe_profile = _safe_profile_projection(profile)
    citations = _collect_citations(safe_profile)
    _, deferred, media_revision = ProfileEnrichmentService(store).summarize_discovery(
        selector,
        purpose='person_relationship_profile_enrichment',
    )
    gaps = [
        {
            'modality': modality,
            'state': 'deferred',
            'reason': 'explicit_enrichment_required',
            'count': count,
        }
        for modality, count in sorted(deferred.items())
    ]
    completeness_state = 'current' if not gaps else 'current_with_deferred_enrichment'
    summary = {
        'state': completeness_state,
        'automatic_refresh': True,
        'processor_versions': ['local:profile-auto-refresh/v1'],
        'deferred_by_modality': deferred,
    }
    semantic_revision = _digest({
        'entity_id': entity_id,
        'profile': safe_profile,
        'evidence_citations': citations,
        'media_revision': media_revision,
        'unresolved_gaps': gaps,
    })
    content = {
        'schema_version': AUTOMATIC_PROFILE_SNAPSHOT_SCHEMA,
        'canonical_entity_id': entity_id,
        'source_revision': 'profile-src-' + semantic_revision[:24],
        'profile': safe_profile,
        'evidence_citations': citations,
        'evidence_digests': [],
        'enrichment_summary': summary,
        'unresolved_gaps': gaps,
        'raw_content_included': False,
        'raw_paths_included': False,
        'provider_payloads_included': False,
    }
    return {
        'entity_id': entity_id,
        'schema_version': AUTOMATIC_PROFILE_SNAPSHOT_SCHEMA,
        'source_revision': content['source_revision'],
        'completeness_state': completeness_state,
        'content': content,
        'content_hash': _digest(content),
    }


def persist_profile_snapshot_conn(
    conn: Any,
    *,
    entity_id: str,
    content: dict[str, Any],
    content_hash: str,
    source_revision: str,
    run_id: str | None,
    schema_version: str,
    completeness_state: str,
    created_at: str,
) -> tuple[dict[str, Any], bool]:
    """Deduplicate and insert one immutable snapshot in the caller transaction."""

    # A formal enrichment run is idempotent for its whole lifetime, not merely
    # while its snapshot happens to be the newest row. Automatic maintenance
    # may have published newer versions between an original finalize and a
    # retry of that same finalize call.
    if run_id is not None:
        existing = conn.execute(
            """SELECT * FROM profile_snapshots
                 WHERE entity_id=? AND run_id=? AND content_hash=?
                 ORDER BY version DESC,created_at DESC,profile_id DESC LIMIT 1""",
            (entity_id, run_id, content_hash),
        ).fetchone()
        if existing is not None:
            return dict(existing), False
    latest = conn.execute(
        """SELECT * FROM profile_snapshots WHERE entity_id=?
             ORDER BY version DESC,created_at DESC,profile_id DESC LIMIT 1""",
        (entity_id,),
    ).fetchone()
    if latest is not None and str(latest['content_hash']) == content_hash:
        return dict(latest), False
    version = int(latest['version']) + 1 if latest is not None else 1
    profile_id = 'profile-' + hashlib.sha256(
        f'{entity_id}:{version}:{content_hash}'.encode('utf-8')
    ).hexdigest()[:24]
    projection = content | {'content_hash': content_hash, 'created_at': created_at}
    conn.execute(
        """INSERT INTO profile_snapshots(
               profile_id,entity_id,version,projection_json,content_hash,source_revision,run_id,
               schema_version,completeness_state,evidence_citations_json,enrichment_summary_json,
               gaps_json,created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            profile_id,
            entity_id,
            version,
            _json(projection),
            content_hash,
            source_revision,
            run_id,
            schema_version,
            completeness_state,
            _json(content['evidence_citations']),
            _json(content['enrichment_summary']),
            _json(content['unresolved_gaps']),
            created_at,
        ),
    )
    row = conn.execute(
        'SELECT * FROM profile_snapshots WHERE profile_id=?', (profile_id,),
    ).fetchone()
    return dict(row), True


def finalize_profile_snapshot(
    store: SQLiteStore,
    run_id: str,
    *,
    actor: str,
    session: str,
) -> dict[str, Any]:
    manifest = ProfileEnrichmentService(store).manifest(run_id, actor=actor, session=session, limit=1)
    state = str(manifest['state'])
    if state not in FINAL_STATES:
        return {
            'ok': False,
            'finalized': False,
            'run_id': run_id,
            'completeness_state': state,
            'pause_reason': state,
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    run, tasks = _run_rows(store, run_id)
    content = _snapshot_content(store, run, tasks)
    content_hash = _digest(content)
    created_at = _now()
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row, created = persist_profile_snapshot_conn(
            conn,
            entity_id=str(run['entity_id']),
            content=content,
            content_hash=content_hash,
            source_revision=str(run['source_revision']),
            run_id=run_id,
            schema_version=PROFILE_SNAPSHOT_SCHEMA,
            completeness_state=state,
            created_at=created_at,
        )
        if not created:
            conn.rollback()
            return _snapshot_result(row, created=False)
        conn.commit()
    return _snapshot_result(row, created=True)


def _snapshot_result(row: dict[str, Any], *, created: bool, stale: bool = False) -> dict[str, Any]:
    summary = json.loads(row.get('enrichment_summary_json') or '{}')
    gaps = json.loads(row.get('gaps_json') or '[]')
    return {
        'ok': True,
        'type': 'profile_snapshot',
        'finalized': True,
        'created': created,
        'cache_hit': not created,
        'profile_id': row['profile_id'],
        'entity_id': row['entity_id'],
        'version': int(row['version']),
        'content_hash': row['content_hash'],
        'source_revision': row['source_revision'],
        'schema_version': row['schema_version'],
        'completeness_state': row['completeness_state'],
        'freshness_state': 'stale' if stale else 'current',
        'stale': stale,
        'evidence_citations_count': len(json.loads(row.get('evidence_citations_json') or '[]')),
        'enrichment_summary': summary,
        'unresolved_gaps': gaps,
        'created_at': row['created_at'],
        'raw_content_included': False,
        'raw_paths_included': False,
        'provider_payloads_included': False,
    }


def profile_snapshot_status(
    store: SQLiteStore,
    customer: str,
    *,
    resolved_entity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    service = ProfileEnrichmentService(store)
    if resolved_entity is None:
        _, resolved = service._resolved_scope(customer)
    else:
        resolved = resolved_entity
    entity_id = str(resolved.get('entity_id') or '').strip()
    if not entity_id or entity_id.startswith('unresolved:'):
        raise ProfileEnrichmentError(
            'customer must resolve to a canonical entity', code='canonical_entity_required',
        )
    with store.connect() as conn:
        row = conn.execute(
            'SELECT * FROM profile_snapshots WHERE entity_id=? ORDER BY version DESC,created_at DESC LIMIT 1',
            (entity_id,),
        ).fetchone()
    if row is None:
        return {
            'ok': True,
            'type': 'profile_snapshot_status',
            'entity_id': entity_id,
            'completeness_state': 'missing',
            'stale': True,
            'raw_content_included': False,
            'raw_paths_included': False,
        }
    snapshot = dict(row)
    if str(snapshot.get('schema_version') or '') == AUTOMATIC_PROFILE_SNAPSHOT_SCHEMA:
        with store.connect() as conn:
            subscription = conn.execute(
                """SELECT s.selector,s.enabled,s.last_profile_id,q.state AS queue_state
                     FROM profile_automation_subscriptions s
                LEFT JOIN profile_refresh_queue q ON q.entity_id=s.entity_id
                    WHERE s.entity_id=?""",
                (entity_id,),
            ).fetchone()
            entity = conn.execute(
                'SELECT display_name FROM entities WHERE entity_id=?', (entity_id,),
            ).fetchone()
        if subscription is not None and bool(subscription['enabled']):
            stale = (
                str(subscription['last_profile_id'] or '') != str(snapshot['profile_id'])
                or subscription['queue_state'] is not None
            )
        else:
            selector = str(
                (subscription['selector'] if subscription is not None else None)
                or (entity['display_name'] if entity is not None else customer)
            )
            current = automatic_profile_snapshot_payload(
                store, entity_id=entity_id, selector=selector,
            )
            stale = str(current['content_hash']) != str(snapshot['content_hash'])
    else:
        run: dict[str, Any] | None = None
        tasks: list[dict[str, Any]] | None = None
        purpose = 'customer_profile_enrichment'
        if snapshot.get('run_id'):
            run, tasks = _run_rows(store, str(snapshot['run_id']))
            purpose = str(run.get('purpose') or purpose)
        _, _, current_source_revision = service.discover(customer, purpose=purpose)
        stale = str(snapshot['source_revision']) != current_source_revision
        if not stale and run is not None and tasks is not None:
            stale = _digest(_snapshot_content(store, run, tasks)) != str(snapshot['content_hash'])
    return _snapshot_result(snapshot, created=False, stale=stale) | {'type': 'profile_snapshot_status'}


def list_profile_snapshots(
    store: SQLiteStore,
    customer: str,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ProfileEnrichmentError('limit must be from 1 to 100', code='invalid_snapshot_limit')
    entity_id, resolved = _canonical_entity_id(store, customer)
    with store.connect() as conn:
        total = int(conn.execute(
            'SELECT COUNT(*) FROM profile_snapshots WHERE entity_id=?', (entity_id,),
        ).fetchone()[0])
        rows = [dict(row) for row in conn.execute(
            """SELECT profile_id,entity_id,version,content_hash,source_revision,schema_version,
                      completeness_state,evidence_citations_json,enrichment_summary_json,gaps_json,created_at
                 FROM profile_snapshots
                WHERE entity_id=? ORDER BY version DESC,created_at DESC LIMIT ?""",
            (entity_id, limit),
        )]
    latest_status = (
        profile_snapshot_status(store, customer, resolved_entity=resolved)
        if rows else {'stale': True}
    )
    current_profile_id = str(latest_status.get('profile_id') or '')
    return {
        'ok': True,
        'type': 'profile_snapshot_list',
        'entity_id': entity_id,
        'count': total,
        'items': [
            _snapshot_result(
                row,
                created=False,
                stale=(
                    str(row['profile_id']) != current_profile_id
                    or bool(latest_status.get('stale'))
                ),
            )
            for row in rows
        ],
        'truncated': total > len(rows),
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def _profile_snapshot_row(
    store: SQLiteStore,
    customer: str,
    *,
    version: int | None = None,
) -> tuple[str, dict[str, Any]]:
    entity_id, _ = _canonical_entity_id(store, customer)
    if version is not None and (type(version) is not int or version < 1):
        raise ProfileEnrichmentError('version must be a positive integer', code='invalid_snapshot_version')
    with store.connect() as conn:
        if version is None:
            row = conn.execute(
                'SELECT * FROM profile_snapshots WHERE entity_id=? ORDER BY version DESC,created_at DESC LIMIT 1',
                (entity_id,),
            ).fetchone()
        else:
            row = conn.execute(
                'SELECT * FROM profile_snapshots WHERE entity_id=? AND version=? LIMIT 1',
                (entity_id, version),
            ).fetchone()
    if row is None:
        raise ProfileEnrichmentError('profile snapshot not found', code='profile_snapshot_not_found')
    return entity_id, dict(row)


def get_profile_snapshot(
    store: SQLiteStore,
    customer: str,
    *,
    version: int | None = None,
) -> dict[str, Any]:
    entity_id, row = _profile_snapshot_row(store, customer, version=version)
    projection = _sanitize(json.loads(row.get('projection_json') or '{}'))
    latest_status = profile_snapshot_status(store, customer)
    stale = (
        str(row['profile_id']) != str(latest_status.get('profile_id') or '')
        or bool(latest_status.get('stale'))
    )
    return _snapshot_result(row, created=False, stale=stale) | {
        'type': 'profile_snapshot',
        'profile': projection.get('profile') or {},
        'projection': projection,
    }


def _section_rows(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sections = ((snapshot.get('profile') or {}).get('sections') or {})
    return {
        str(name): [row for row in rows if isinstance(row, dict)]
        for name, rows in sections.items()
        if isinstance(rows, list)
    }


def diff_profile_snapshots(
    store: SQLiteStore,
    customer: str,
    *,
    from_version: int,
    to_version: int | None = None,
    max_changes: int = 200,
) -> dict[str, Any]:
    if type(max_changes) is not int or not 1 <= max_changes <= 500:
        raise ProfileEnrichmentError('max_changes must be from 1 to 500', code='invalid_snapshot_limit')
    before = get_profile_snapshot(store, customer, version=from_version)
    after = get_profile_snapshot(store, customer, version=to_version)
    before_sections = _section_rows(before)
    after_sections = _section_rows(after)
    changes: list[dict[str, Any]] = []
    for field in (
        'schema_version',
        'completeness_state',
        'evidence_citations_count',
        'unresolved_gaps',
    ):
        if before.get(field) != after.get(field):
            changes.append({
                'section': '_snapshot',
                'change': 'changed',
                'field': field,
                'before': _sanitize(before.get(field)),
                'after': _sanitize(after.get(field)),
            })
    for section in sorted(set(before_sections) | set(after_sections)):
        old = {_digest(row): row for row in before_sections.get(section, [])}
        new = {_digest(row): row for row in after_sections.get(section, [])}
        for digest in sorted(new.keys() - old.keys()):
            changes.append({'section': section, 'change': 'added', 'claim': new[digest]})
        for digest in sorted(old.keys() - new.keys()):
            changes.append({'section': section, 'change': 'removed', 'claim': old[digest]})
    total = len(changes)
    return {
        'ok': True,
        'type': 'profile_snapshot_diff',
        'entity_id': before['entity_id'],
        'from_version': int(before['version']),
        'to_version': int(after['version']),
        'changes_count': total,
        'changes': changes[:max_changes],
        'truncated': total > max_changes,
        'raw_content_included': False,
        'raw_paths_included': False,
        'provider_payloads_included': False,
    }
