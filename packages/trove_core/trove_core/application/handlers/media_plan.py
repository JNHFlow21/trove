"""Read-only bounded media understanding scope preview.

Answers "how much understanding work does one scope contain, what runs
locally for free, and what needs approved paid cloud calls" without
triggering any processing, provider construction, or network egress.

Scope anchors stay index-bounded by construction: a conversation scope is a
constant-prefix GLOB range over idx_media_assets_citation (chat asset
citations are exactly their message citations, so the conversation segment
of the citation selects precisely that conversation's assets), an account
scope aggregates the covering idx_media_assets_account_modality index, and
an author scope enumerates a capped sender window through
idx_messages_sender_time before probing assets by citation.  Understanding
state is classified only for the first ``limit`` candidates, so every probe
(transcripts, provider jobs, image observations, asset links) is a bounded
batch of index seeks.  Time filters apply to message timestamps and require
a conversation or author anchor; account-wide media carry only ingest-time
timestamps, so an account scope with a time filter fails typed instead of
silently scanning.

Name resolution is deliberately absent: callers pass identifiers obtained
from trove.resolve or trove.profile, and an unknown or ambiguous scope
fails typed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from trove_core.bounds import MEDIA_ENRICH_PLAN, BoundedInputError, bounded_limit
from trove_core.providers.pricing import estimate_asr_flash_rmb
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.generation import vault_generation_read

from .base import HandlerOutcome


_MEDIA_TYPES = ('image', 'voice', 'file')
_PIPELINED_MODALITIES = ('image', 'voice')
_KINDS = ('ocr', 'caption', 'transcribe')
_KIND_MODALITY = {'ocr': 'image', 'caption': 'image', 'transcribe': 'voice'}
_NO_SOURCE_STATES = frozenset({'missing_local_cache', 'metadata_only'})
_GLOB_METACHARS = frozenset('*?[]')
_CONVERSATION_SCAN_CAP = 50_000
_AUTHOR_SCAN_CAP = 5_000
_PROBE_CHUNK = 500
_TRANSCRIPT_DURATION_SAMPLE = 500
_DEFAULT_VOICE_AUDIO_SECONDS = 10.0
# Order-of-magnitude per-item wall-clock heuristics for one understood item.
_OCR_SECONDS_PER_ITEM = 0.3
_CAPTION_SECONDS_PER_ITEM = 12.0
_TRANSCRIBE_SECONDS_PER_ITEM = 8.0

_EXECUTION_PROFILES: dict[str, dict[str, Any]] = {
    'ocr': {
        'execution': 'local',
        'provider': 'local-macos-vision',
        'media_enrich_kind': 'annotate',
        'seconds_per_item': _OCR_SECONDS_PER_ITEM,
        'approval_required': False,
    },
    'caption': {
        'execution': 'local',
        'provider': 'local-vlm-qwen25-vl',
        'media_enrich_kind': 'annotate',
        'seconds_per_item': _CAPTION_SECONDS_PER_ITEM,
        'approval_required': False,
    },
    'transcribe': {
        'execution': 'cloud',
        'media_enrich_kind': 'transcribe',
        'seconds_per_item': _TRANSCRIBE_SECONDS_PER_ITEM,
        'approval_required': True,
        'approval': {
            'grant_action': 'voice_cloud_asr',
            'danger_class': 'cloud_asr_upload',
            'granularity': 'per_citation',
        },
    },
}


def _owner(config: Any) -> Any | None:
    return config if hasattr(config, 'read_store') and hasattr(config, 'config') else None


def _open(config: Any):
    owner = _owner(config)
    cfg = owner.config if owner is not None else config
    if not cfg.paths.sqlite_path.is_file():
        return None, None, cfg
    store = owner.read_store if owner is not None else SQLiteStore(cfg.paths.sqlite_path, readonly=True)
    store.initialize()
    return owner, store, cfg


def _close(owner: Any, store: Any) -> None:
    if owner is None and store is not None:
        store.close()


def _bounded(field: str, value: Any, spec: Any) -> int | HandlerOutcome:
    try:
        return bounded_limit(
            spec.default if value is None else value, field=field, spec=spec,
        )
    except BoundedInputError as exc:
        return HandlerOutcome.failure(exc.code, str(exc), details=exc.to_dict())


def _time_filter(field: str, value: Any) -> str | None | HandlerOutcome:
    """Normalize one optional time filter given as ISO 8601 or epoch seconds."""

    text = str(value or '').strip()
    if not text:
        return None
    try:
        if text.isdigit():
            parsed = datetime.fromtimestamp(int(text), tz=timezone.utc)
        else:
            parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return HandlerOutcome.failure(
            'invalid_request',
            f'{field} must be an ISO 8601 timestamp or epoch seconds.',
            details={'field': field},
        )
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _string_set(field: str, value: Any, allowed: Iterable[str], default: tuple[str, ...]) -> tuple[str, ...] | HandlerOutcome:
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        return HandlerOutcome.failure(
            'invalid_request',
            f'{field} must be a non-empty array.',
            details={'field': field},
        )
    allowed_set = frozenset(allowed)
    invalid = sorted({str(item) for item in value if item not in allowed_set})
    if invalid:
        return HandlerOutcome.failure(
            'invalid_request',
            f'{field} entries must be one of {", ".join(sorted(allowed_set))}.',
            details={'field': field, 'invalid': invalid},
        )
    return tuple(dict.fromkeys(str(item) for item in value))


def _resolve_conversation(
    conn: Any,
    conversation_id: str,
    account_id: str | None,
) -> str | HandlerOutcome:
    """Scope to exactly one stored conversation account or fail typed."""

    clauses = ['conversation_id=?']
    params: list[Any] = [conversation_id]
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    rows = conn.execute(
        f'SELECT account_id FROM conversations WHERE {" AND ".join(clauses)} ORDER BY account_id LIMIT 11',
        params,
    ).fetchall()
    candidates = [str(row[0]) for row in rows]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return HandlerOutcome.failure(
            'ambiguous_target',
            'Conversation id matches multiple accounts; pass account_id explicitly.',
            details={'candidates': candidates},
        )
    return HandlerOutcome.failure(
        'no_results',
        'No stored conversation matches the conversation scope.',
        details={'conversation_id': conversation_id},
    )


def _author_exists(conn: Any, author_id: str, account_id: str | None) -> bool:
    clauses = ['sender_id=?']
    params: list[Any] = [author_id]
    if account_id:
        clauses.append('account_id=?')
        params.append(account_id)
    return conn.execute(
        f'SELECT 1 FROM messages INDEXED BY idx_messages_sender_time WHERE {" AND ".join(clauses)} LIMIT 1',
        params,
    ).fetchone() is not None


def _glob_guard(*values: str) -> HandlerOutcome | None:
    if any(any(char in _GLOB_METACHARS for char in value) for value in values):
        return HandlerOutcome.failure(
            'invalid_request',
            'Resolved scope identifiers contain glob metacharacters.',
        )
    return None


def _chunks(values: list[str], size: int = _PROBE_CHUNK) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _modality_filter(modalities: tuple[str, ...]) -> tuple[str, list[Any]]:
    placeholders = ','.join('?' for _ in modalities)
    return f'ma.modality IN ({placeholders})', list(modalities)


def _source_ready(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get('path_ref') or '')) and str(row.get('cache_state') or '') not in _NO_SOURCE_STATES


def _accumulate_totals(grouped: Iterable[Any]) -> dict[str, Any]:
    by_modality: dict[str, int] = {}
    by_cache_state: dict[str, int] = {}
    total = 0
    for row in grouped:
        count = int(row['n'])
        total += count
        by_modality[str(row['modality'])] = by_modality.get(str(row['modality']), 0) + count
        by_cache_state[str(row['cache_state'])] = by_cache_state.get(str(row['cache_state']), 0) + count
    return {
        'state': 'exact',
        'media_assets': total,
        'by_modality': dict(sorted(by_modality.items())),
        'by_cache_state': dict(sorted(by_cache_state.items())),
    }


def _conversation_scope(
    conn: Any,
    *,
    account_id: str,
    conversation_id: str,
    totals_modalities: tuple[str, ...],
    candidate_modalities: tuple[str, ...],
    since: str | None,
    until: str | None,
    limit: int,
) -> dict[str, Any] | HandlerOutcome:
    failure = _glob_guard(account_id, conversation_id)
    if failure is not None:
        return failure
    prefix = f'trove://wechat/{account_id}/{conversation_id}/*'
    totals_sql, totals_params = _modality_filter(totals_modalities)
    covering_total = int(conn.execute(
        f'SELECT COUNT(*) FROM media_assets ma INDEXED BY idx_media_assets_citation'
        f' WHERE ma.citation GLOB ? AND {totals_sql}',
        [prefix, *totals_params],
    ).fetchone()[0])
    if covering_total > _CONVERSATION_SCAN_CAP:
        return {
            'totals': {
                'state': 'scan_capped',
                'media_assets': covering_total,
                'by_modality': {},
                'by_cache_state': {},
            },
            'candidates': [],
            'truncated': True,
            'candidate_universe': None,
        }
    join_sql = ''
    time_sql = ''
    time_params: list[Any] = []
    if since or until:
        join_sql = ' CROSS JOIN messages m INDEXED BY idx_messages_citation ON m.citation = ma.citation'
        if since:
            time_sql += ' AND m.timestamp>=?'
            time_params.append(since)
        if until:
            time_sql += ' AND m.timestamp<?'
            time_params.append(until)
    grouped = conn.execute(
        f'SELECT ma.modality, ma.cache_state, COUNT(*) AS n'
        f' FROM media_assets ma INDEXED BY idx_media_assets_citation{join_sql}'
        f' WHERE ma.citation GLOB ? AND {totals_sql}{time_sql}'
        f' GROUP BY ma.modality, ma.cache_state',
        [prefix, *totals_params, *time_params],
    ).fetchall()
    totals = _accumulate_totals(grouped)
    candidates: list[dict[str, Any]] = []
    truncated = False
    if candidate_modalities:
        candidate_sql, candidate_params = _modality_filter(candidate_modalities)
        rows = conn.execute(
            f'SELECT ma.asset_id, ma.modality, ma.cache_state, ma.path_ref, ma.content_hash, ma.citation'
            f' FROM media_assets ma INDEXED BY idx_media_assets_citation{join_sql}'
            f' WHERE ma.citation GLOB ? AND {candidate_sql}{time_sql}'
            f' ORDER BY ma.citation LIMIT ?',
            [prefix, *candidate_params, *time_params, limit + 1],
        ).fetchall()
        truncated = len(rows) > limit
        candidates = [dict(row) for row in rows[:limit]]
    candidate_universe = sum(totals['by_modality'].get(modality, 0) for modality in candidate_modalities)
    return {
        'totals': totals,
        'candidates': candidates,
        'truncated': truncated,
        'candidate_universe': candidate_universe,
    }


def _account_scope(
    conn: Any,
    *,
    account_id: str,
    totals_modalities: tuple[str, ...],
    candidate_modalities: tuple[str, ...],
    limit: int,
) -> dict[str, Any]:
    totals_sql, totals_params = _modality_filter(totals_modalities)
    grouped = conn.execute(
        f'SELECT ma.modality, ma.cache_state, COUNT(*) AS n'
        f' FROM media_assets ma INDEXED BY idx_media_assets_account_modality'
        f' WHERE ma.account_id=? AND {totals_sql}'
        f' GROUP BY ma.modality, ma.cache_state',
        [account_id, *totals_params],
    ).fetchall()
    totals = _accumulate_totals(grouped)
    candidate_universe = sum(totals['by_modality'].get(modality, 0) for modality in candidate_modalities)
    # Actionable-first enumeration without a sort: the covering index serves
    # each (modality, cache_state) tier directly, so the 'cached' tier (the
    # only tier whose rows carry a local source file in practice) is
    # evaluated before the metadata-only backlog.
    states = ['cached', *(state for state in sorted(totals['by_cache_state']) if state != 'cached')]
    candidate_ids: list[str] = []
    gather_truncated = False
    for state in states:
        for modality in candidate_modalities:
            remaining = limit + 1 - len(candidate_ids)
            if remaining <= 0:
                gather_truncated = True
                break
            rows = conn.execute(
                'SELECT ma.asset_id'
                ' FROM media_assets ma INDEXED BY idx_media_assets_account_modality'
                ' WHERE ma.account_id=? AND ma.modality=? AND ma.cache_state=?'
                ' LIMIT ?',
                (account_id, modality, state, remaining),
            ).fetchall()
            candidate_ids.extend(str(row['asset_id']) for row in rows)
        if gather_truncated:
            break
    candidate_ids = candidate_ids[:limit]
    candidates: list[dict[str, Any]] = []
    for chunk in _chunks(candidate_ids):
        placeholders = ','.join('?' for _ in chunk)
        candidates.extend(
            dict(row)
            for row in conn.execute(
                f'SELECT ma.asset_id, ma.modality, ma.cache_state, ma.path_ref, ma.content_hash, ma.citation'
                f' FROM media_assets ma WHERE ma.asset_id IN ({placeholders})',
                chunk,
            ).fetchall()
        )
    order = {asset_id: index for index, asset_id in enumerate(candidate_ids)}
    candidates.sort(key=lambda row: order[str(row['asset_id'])])
    return {
        'totals': totals,
        'candidates': candidates,
        'truncated': gather_truncated or candidate_universe > len(candidates),
        'candidate_universe': candidate_universe,
    }


def _author_scope(
    conn: Any,
    *,
    author_id: str,
    account_id: str | None,
    candidate_modalities: tuple[str, ...],
    since: str | None,
    until: str | None,
    limit: int,
) -> dict[str, Any]:
    clauses = ['m.sender_id=?']
    params: list[Any] = [author_id]
    if account_id:
        clauses.append('m.account_id=?')
        params.append(account_id)
    if since:
        clauses.append('m.timestamp>=?')
        params.append(since)
    if until:
        clauses.append('m.timestamp<?')
        params.append(until)
    where = ' AND '.join(clauses)
    messages_total: int | None = None
    if not account_id:
        messages_total = int(conn.execute(
            f'SELECT COUNT(*) FROM messages m INDEXED BY idx_messages_sender_time WHERE {where}',
            params,
        ).fetchone()[0])
    rows = conn.execute(
        f'SELECT m.citation, m.timestamp FROM messages m INDEXED BY idx_messages_sender_time'
        f' WHERE {where} ORDER BY m.timestamp DESC, m.citation DESC LIMIT ?',
        (*params, _AUTHOR_SCAN_CAP + 1),
    ).fetchall()
    scan_truncated = len(rows) > _AUTHOR_SCAN_CAP
    messages = [str(row['citation']) for row in rows[:_AUTHOR_SCAN_CAP]]
    candidates: list[dict[str, Any]] = []
    if candidate_modalities:
        modality_sql, modality_params = _modality_filter(candidate_modalities)
        assets_by_citation: dict[str, list[dict[str, Any]]] = {}
        for chunk in _chunks(messages):
            placeholders = ','.join('?' for _ in chunk)
            for row in conn.execute(
                f'SELECT ma.asset_id, ma.modality, ma.cache_state, ma.path_ref, ma.content_hash, ma.citation'
                f' FROM media_assets ma INDEXED BY idx_media_assets_citation'
                f' WHERE ma.citation IN ({placeholders}) AND {modality_sql}',
                [*chunk, *modality_params],
            ).fetchall():
                assets_by_citation.setdefault(str(row['citation']), []).append(dict(row))
        seen: set[str] = set()
        for citation in messages:
            for asset in assets_by_citation.get(citation, ()):
                asset_id = str(asset['asset_id'])
                if asset_id in seen:
                    continue
                seen.add(asset_id)
                candidates.append(asset)
    truncated = scan_truncated or len(candidates) > limit
    candidates = candidates[:limit]
    return {
        'totals': {
            'state': 'scan_capped',
            'media_assets': None,
            'by_modality': {},
            'by_cache_state': {},
            'messages_total': messages_total,
            'messages_scanned': len(messages),
        },
        'candidates': candidates,
        'truncated': truncated,
        'candidate_universe': None,
    }


def _probe_link_eligibility(conn: Any, asset_ids: list[str]) -> set[str]:
    """Assets excluded by the asset-link rule (e.g. rejected orphan cache media)."""

    excluded: set[str] = set()
    for chunk in _chunks(asset_ids):
        placeholders = ','.join('?' for _ in chunk)
        for row in conn.execute(
            f'SELECT l.asset_id, MAX(l.accepted) AS any_accepted'
            f' FROM media_asset_links l INDEXED BY idx_media_asset_links_asset'
            f' WHERE l.asset_id IN ({placeholders}) GROUP BY l.asset_id',
            chunk,
        ).fetchall():
            if not int(row['any_accepted'] or 0):
                excluded.add(str(row['asset_id']))
    return excluded


def _probe_private_citations(conn: Any, citations: list[str]) -> set[str]:
    """Citations provably attached to a private-chat message, by exact match."""

    private: set[str] = set()
    for chunk in _chunks(citations):
        placeholders = ','.join('?' for _ in chunk)
        for row in conn.execute(
            f"SELECT m.citation FROM messages m INDEXED BY idx_messages_citation"
            f" WHERE m.citation IN ({placeholders}) AND m.conversation_type='private'",
            chunk,
        ).fetchall():
            private.add(str(row['citation']))
    return private


def _probe_cloud_transcribed(
    conn: Any,
    candidates: list[Mapping[str, Any]],
    *,
    provider: str,
    model: str,
) -> set[str]:
    """Assets whose active transcript is a current cloud-ASR projection.

    Mirrors the execution-layer validity rule: the provider job must be the
    completed cloud model run whose request hash still matches the asset's
    current content hash (a stale or local-only transcript does not count).
    """

    by_id = {str(row['asset_id']): row for row in candidates}
    understood: set[str] = set()
    for chunk in _chunks(list(by_id)):
        placeholders = ','.join('?' for _ in chunk)
        for row in conn.execute(
            f'SELECT t.asset_id, pj.request_hash'
            f' FROM transcripts t JOIN provider_jobs pj ON pj.job_id=t.job_id'
            f" WHERE t.asset_id IN ({placeholders}) AND t.status='active'"
            f' AND pj.provider=? AND pj.model=? AND pj.status=?',
            [*chunk, provider, model, 'completed'],
        ).fetchall():
            asset = by_id.get(str(row['asset_id']))
            content_hash = str((asset or {}).get('content_hash') or '')
            request_hash = str(row['request_hash'] or '')
            if content_hash and request_hash and content_hash == request_hash:
                understood.add(str(row['asset_id']))
    return understood


def _probe_image_understood(conn: Any, asset_ids: list[str], kind: str) -> set[str]:
    """Assets with a non-empty OCR text / caption projection, matching the
    idempotency checks of the image observation and caption runners."""

    column = 'visible_text' if kind == 'ocr' else 'caption'
    understood: set[str] = set()
    for chunk in _chunks(asset_ids):
        placeholders = ','.join('?' for _ in chunk)
        for row in conn.execute(
            f'SELECT DISTINCT io.asset_id FROM image_observations io'
            f' WHERE io.asset_id IN ({placeholders})'
            f" AND TRIM(COALESCE(io.{column}, '')) <> ''",
            chunk,
        ).fetchall():
            understood.add(str(row['asset_id']))
    return understood


def _voice_audio_seconds(conn: Any) -> tuple[float, str, int]:
    rows = conn.execute(
        "SELECT duration_seconds FROM transcripts"
        " WHERE status='active' AND duration_seconds>0 LIMIT ?",
        (_TRANSCRIPT_DURATION_SAMPLE,),
    ).fetchall()
    durations = [float(row[0]) for row in rows]
    if not durations:
        return _DEFAULT_VOICE_AUDIO_SECONDS, 'default_seconds_per_item', 0
    return sum(durations) / len(durations), 'transcript_duration_sample_average', len(durations)


def _duration_tier(seconds: float) -> str:
    if seconds < 60:
        return 'under_a_minute'
    if seconds < 1800:
        return 'minutes'
    if seconds < 7200:
        return 'tens_of_minutes'
    return 'hours'


def _classify(
    conn: Any,
    *,
    candidates: list[dict[str, Any]],
    kinds: tuple[str, ...],
    execution: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asset_ids = [str(row['asset_id']) for row in candidates]
    link_excluded = _probe_link_eligibility(conn, asset_ids) if asset_ids else set()
    voice_rows = [row for row in candidates if str(row['modality']) == 'voice']
    private_citations = _probe_private_citations(
        conn, [str(row['citation']) for row in voice_rows],
    ) if voice_rows else set()

    eligibility: dict[str, str] = {}
    for row in candidates:
        asset_id = str(row['asset_id'])
        if asset_id in link_excluded:
            eligibility[asset_id] = 'excluded_by_asset_links'
        elif str(row['modality']) == 'voice' and str(row['citation']) not in private_citations:
            eligibility[asset_id] = 'voice_not_private_chat'
        else:
            eligibility[asset_id] = 'eligible'

    cloud_provider = cloud_model = ''
    if 'transcribe' in kinds:
        from trove_core.media_pipeline import CLOUD_ASR_MODEL_ID, CLOUD_ASR_PROVIDER_NAME

        cloud_provider, cloud_model = CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID
    understood_by_kind: dict[str, set[str]] = {}
    for kind in kinds:
        ids = {
            str(row['asset_id']) for row in candidates
            if str(row['modality']) == _KIND_MODALITY[kind] and eligibility[str(row['asset_id'])] == 'eligible'
        }
        if not ids:
            understood_by_kind[kind] = set()
        elif kind == 'transcribe':
            understood_by_kind[kind] = _probe_cloud_transcribed(
                conn,
                [row for row in candidates if str(row['asset_id']) in ids],
                provider=cloud_provider,
                model=cloud_model,
            )
        else:
            understood_by_kind[kind] = _probe_image_understood(conn, sorted(ids), kind)

    if 'transcribe' in kinds:
        audio_seconds, audio_basis, audio_sample = _voice_audio_seconds(conn)
    else:
        audio_seconds, audio_basis, audio_sample = _DEFAULT_VOICE_AUDIO_SECONDS, 'default_seconds_per_item', 0
    plans: list[dict[str, Any]] = []
    for kind in kinds:
        modality = _KIND_MODALITY[kind]
        profile = _EXECUTION_PROFILES[kind]
        rows = [row for row in candidates if str(row['modality']) == modality]
        eligible_rows = [row for row in rows if eligibility[str(row['asset_id'])] == 'eligible']
        understood = understood_by_kind[kind]
        pending_rows = [row for row in eligible_rows if str(row['asset_id']) not in understood]
        ready_rows = [row for row in pending_rows if _source_ready(row)]
        included = not (execution == 'local_only' and profile['execution'] == 'cloud')
        estimated_seconds = round(len(ready_rows) * float(profile['seconds_per_item']), 1)
        entry: dict[str, Any] = {
            'kind': kind,
            'media_type': modality,
            'execution': profile['execution'],
            'provider': cloud_provider if kind == 'transcribe' else str(profile.get('provider') or ''),
            'media_enrich_kind': profile['media_enrich_kind'],
            'included': included,
            'candidates': len(rows),
            'out_of_scope': len(rows) - len(eligible_rows),
            'understood': len([row for row in eligible_rows if str(row['asset_id']) in understood]),
            'pending': len(ready_rows),
            'pending_no_source': len(pending_rows) - len(ready_rows),
            'approval_required': bool(profile['approval_required']) and included,
            'estimated_seconds': estimated_seconds,
            'duration_tier': _duration_tier(estimated_seconds),
            'duration_estimate_basis': 'heuristic_seconds_per_item',
            'estimated_cost_rmb': 0.0,
            'estimated_tokens': None,
            'token_billing': False,
            'cloud_calls_made': False,
        }
        if not included:
            entry['exclusion_reason'] = 'cloud_only_kind_excluded_by_execution_filter'
        if kind == 'transcribe':
            per_item_cost = estimate_asr_flash_rmb(audio_seconds)
            entry.update({
                'estimated_audio_seconds': round(len(ready_rows) * audio_seconds, 1),
                'audio_seconds_basis': audio_basis,
                'audio_seconds_sample': audio_sample,
                'estimated_cost_rmb': round(len(ready_rows) * per_item_cost, 6),
                'cost_estimate_basis': 'duration_rate_rmb_per_audio_hour',
                'per_item': {
                    'audio_seconds': round(audio_seconds, 3),
                    'cost_rmb': per_item_cost,
                    'seconds': profile['seconds_per_item'],
                },
                'required_approval': dict(profile['approval']),
            })
        plans.append(entry)

    by_modality: dict[str, int] = {}
    for row in candidates:
        modality = str(row['modality'])
        by_modality[modality] = by_modality.get(modality, 0) + 1
    summary = {
        'evaluated': len(candidates),
        'by_modality': dict(sorted(by_modality.items())),
        'eligible': len([row for row in candidates if eligibility[str(row['asset_id'])] == 'eligible']),
        'excluded': len([row for row in candidates if eligibility[str(row['asset_id'])] != 'eligible']),
        'with_source': len([row for row in candidates if _source_ready(row)]),
        'no_source': len([row for row in candidates if not _source_ready(row)]),
    }
    return plans, summary


def media_enrich_plan(config: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    conversation_id = payload.get('conversation_id')
    conversation_id = str(conversation_id) if conversation_id else None
    author_id = payload.get('author_id')
    author_id = str(author_id) if author_id else None
    account_id = payload.get('account_id')
    account_id = str(account_id) if account_id else None
    if conversation_id and author_id:
        return HandlerOutcome.failure(
            'invalid_request',
            'Pass conversation_id or author_id, not both.',
        )
    if not (conversation_id or author_id or account_id):
        return HandlerOutcome.failure(
            'invalid_request',
            'A preview scope is required: pass account_id, conversation_id, or author_id.',
        )
    since = _time_filter('since', payload.get('since'))
    if isinstance(since, HandlerOutcome):
        return since
    until = _time_filter('until', payload.get('until'))
    if isinstance(until, HandlerOutcome):
        return until
    if since and until and since >= until:
        return HandlerOutcome.failure(
            'invalid_request',
            'since must be earlier than until.',
            details={'since': since, 'until': until},
        )
    if (since or until) and not (conversation_id or author_id):
        return HandlerOutcome.failure(
            'invalid_request',
            'Time filters require a conversation_id or author_id scope; account-wide media carry only ingest-time timestamps.',
        )
    media_types = _string_set('media_types', payload.get('media_types'), _MEDIA_TYPES, _MEDIA_TYPES)
    if isinstance(media_types, HandlerOutcome):
        return media_types
    kinds = _string_set('kinds', payload.get('kinds'), _KINDS, _KINDS)
    if isinstance(kinds, HandlerOutcome):
        return kinds
    execution = str(payload.get('execution') or 'auto')
    if execution not in {'auto', 'local_only'}:
        return HandlerOutcome.failure(
            'invalid_request',
            'execution must be one of auto, local_only.',
            details={'execution': execution},
        )
    limit = _bounded('limit', payload.get('limit'), MEDIA_ENRICH_PLAN)
    if isinstance(limit, HandlerOutcome):
        return limit

    owner, store, cfg = _open(config)
    if store is None:
        return HandlerOutcome.success(
            {
                'scope': {'account_id': account_id, 'conversation_id': conversation_id, 'author_id': author_id, 'since': since, 'until': until},
                'filters': {'media_types': list(media_types), 'kinds': list(kinds), 'execution': execution},
                'scope_totals': {'state': 'exact', 'media_assets': 0, 'by_modality': {}, 'by_cache_state': {}},
                'truncated': False,
                'candidates': {'evaluated': 0, 'by_modality': {}, 'eligible': 0, 'excluded': 0, 'with_source': 0, 'no_source': 0},
                'plan': [],
                'notes': [],
                'raw_content_included': False,
                'raw_paths_included': False,
            },
            page={'has_more': False},
            coverage={'state': 'complete', 'returned': 0, 'remaining': 0},
        )
    try:
        with vault_generation_read(cfg):
            with store.connect() as conn:
                if conversation_id:
                    resolved = _resolve_conversation(conn, conversation_id, account_id)
                    if isinstance(resolved, HandlerOutcome):
                        return resolved
                    account_id = resolved
                if author_id and not _author_exists(conn, author_id, account_id):
                    return HandlerOutcome.failure(
                        'no_results',
                        'No stored message matches the author scope.',
                        details={'author_id': author_id},
                    )
                totals_modalities = tuple(modality for modality in media_types if modality in _MEDIA_TYPES)
                candidate_modalities = tuple(
                    modality for modality in _PIPELINED_MODALITIES
                    if modality in media_types and any(_KIND_MODALITY[kind] == modality for kind in kinds)
                )
                if conversation_id:
                    scoped = _conversation_scope(
                        conn,
                        account_id=str(account_id),
                        conversation_id=conversation_id,
                        totals_modalities=totals_modalities,
                        candidate_modalities=candidate_modalities,
                        since=since,
                        until=until,
                        limit=limit,
                    )
                elif author_id:
                    scoped = _author_scope(
                        conn,
                        author_id=author_id,
                        account_id=account_id,
                        candidate_modalities=candidate_modalities,
                        since=since,
                        until=until,
                        limit=limit,
                    )
                else:
                    scoped = _account_scope(
                        conn,
                        account_id=str(account_id),
                        totals_modalities=totals_modalities,
                        candidate_modalities=candidate_modalities,
                        limit=limit,
                    )
                if isinstance(scoped, HandlerOutcome):
                    return scoped
                plan_kinds = tuple(kind for kind in kinds if _KIND_MODALITY[kind] in media_types)
                plans, summary = _classify(
                    conn,
                    candidates=scoped['candidates'],
                    kinds=plan_kinds,
                    execution=execution,
                )
    finally:
        _close(owner, store)

    notes: list[str] = []
    if 'file' in media_types:
        notes.append('file media has no understanding pipeline; it is counted in scope totals only.')
    if any(_KIND_MODALITY[kind] not in media_types for kind in kinds):
        notes.append('Some requested kinds target media types outside media_types; they are omitted from the plan.')
    if 'transcribe' in plan_kinds and execution == 'local_only':
        notes.append('transcribe is cloud-only today (local ASR is disabled by policy); execution=local_only excludes it from the plan.')
    elif 'transcribe' in plan_kinds:
        notes.append('transcribe runs on cloud ASR and requires an approved per-citation grant before any audio leaves the vault.')
    if summary['no_source']:
        notes.append('no_source items have no local media file; materialize them with trove.media_fetch first, then per-item estimates apply.')
    if scoped['truncated']:
        notes.append(f'The preview evaluated at most limit={limit} candidates; narrow the scope or raise limit for a fuller plan.')
    if author_id:
        notes.append(f'Author scope scans at most {_AUTHOR_SCAN_CAP} newest messages; use since/until to bound the window.')

    return HandlerOutcome.success(
        {
            'scope': {'account_id': account_id, 'conversation_id': conversation_id, 'author_id': author_id, 'since': since, 'until': until},
            'filters': {'media_types': list(media_types), 'kinds': list(kinds), 'execution': execution},
            'scope_totals': scoped['totals'],
            'truncated': bool(scoped['truncated']),
            'candidates': summary,
            'plan': plans,
            'notes': notes,
            'raw_content_included': False,
            'raw_paths_included': False,
        },
        page={'has_more': False},
        coverage={
            'state': 'partial' if scoped['truncated'] else 'complete',
            'returned': summary['evaluated'],
            'remaining': max((scoped['candidate_universe'] or summary['evaluated']) - summary['evaluated'], 0),
        },
    )


__all__ = ['media_enrich_plan']
