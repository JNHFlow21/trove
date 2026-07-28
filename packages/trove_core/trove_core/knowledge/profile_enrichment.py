from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
import secrets
from typing import Any, Iterable

from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.knowledge.logical_evidence import (
    deduplicate_logical_rows,
    logical_message_key,
    logical_moment_media_key,
)
from trove_core.store.sqlite_store import SQLiteStore


MANIFEST_VERSION = 'profile-enrichment/v2-cloud-asr-only'
DEFAULT_LEASE_SECONDS = 300
MAX_LEASE_SECONDS = 1800
MAX_MANIFEST_ITEMS = 5000
CLOUD_ASR_PROVIDER_NAME = 'volcengine-asr-flash'
CLOUD_ASR_MODEL_ID = 'bigmodel:volc.bigasr.auc_turbo'
TERMINAL_TASK_STATES = {'completed', 'unavailable', 'cancelled'}
ACTIVE_TASK_STATES = {'pending', 'materializing', 'awaiting_agent', 'awaiting_approval', 'processing', 'retryable_failure'}
COST_RESERVATION_STATES = ('materializing', 'awaiting_approval', 'processing')


class ProfileEnrichmentError(RuntimeError):
    code = 'profile_enrichment_error'

    def __init__(self, message: str, *, code: str | None = None):
        self.code = code or self.code
        super().__init__(message)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    except ValueError:
        return None


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{_hash(value)[:20]}'


def _bounded_identity(value: str, field: str) -> str:
    text = str(value or '').strip()
    if not text or len(text) > 200:
        raise ProfileEnrichmentError(f'{field} is required and must be at most 200 characters', code='invalid_task_identity')
    return text


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _valid_cost(value: object, *, positive: bool = False) -> bool:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        return False
    return float(value) > 0 if positive else float(value) >= 0


def _row_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


@dataclass(frozen=True)
class EligibleItem:
    asset_id: str | None
    citation: str
    modality: str
    relevance_reason: str
    source_revision: str
    content_hash: str | None
    next_tool: str
    complete: bool
    timestamp: str

    def identity(self) -> str:
        return '|'.join((self.asset_id or '', self.citation, self.modality, self.source_revision))


class ProfileEnrichmentService:
    """Persist and lease explicit profile enrichment work.

    Ordinary profile reads do not instantiate this service.  All mutating
    methods require an actor/session pair and re-check that ownership against
    hashes stored in the run before returning task metadata.
    """

    def __init__(self, store: SQLiteStore):
        self.store = store
        self.store.initialize()

    @staticmethod
    def _owner_hash(value: str, kind: str) -> str:
        return _hash(f'profile-enrichment:{kind}:{_bounded_identity(value, kind)}')

    @staticmethod
    def _cost(modality: str, execution_location: str) -> float:
        if execution_location == 'local' or modality == 'appmsg':
            return 0.0
        return {'voice': 0.05, 'image': 0.02, 'video': 0.10}.get(modality, 0.0)

    @staticmethod
    def _next_tool(modality: str) -> str:
        return {
            'voice': 'trove_voice_transcribe_lazy',
            'image': 'trove_media_fetch',
            'video': 'trove_media_fetch',
            'appmsg': 'trove_profile_enrichment_appmsg_execute',
        }.get(modality, 'trove_media_fetch')

    def _resolved_scope(self, customer: str) -> tuple[dict[str, Any], dict[str, Any]]:
        resolution = resolve_customer(self.store, customer)
        resolved = resolution.get('resolved')
        if not resolved:
            code = 'customer_ambiguous' if resolution.get('ambiguous') else 'customer_not_found'
            raise ProfileEnrichmentError('customer must resolve to one canonical entity', code=code)
        entity_id = str(resolved.get('entity_id') or '').strip()
        if not entity_id or entity_id.startswith('unresolved:'):
            raise ProfileEnrichmentError('customer must resolve to a canonical entity', code='canonical_entity_required')
        return resolution, resolved

    @staticmethod
    def _placeholders(values: Iterable[str]) -> tuple[list[str], str]:
        rows = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
        return rows, ','.join('?' for _ in rows)

    def _moment_media_items(self, resolved: dict[str, Any]) -> list[EligibleItem]:
        user_id = str(resolved.get('primary_user_id') or '').strip()
        if not user_id:
            return []
        sql = f"""SELECT ma.asset_id,ma.citation,ma.modality,ma.content_hash,ma.updated_at,
                         ma.source_id,mi.author_id,mi.text,
                         COALESCE(mi.timestamp,ma.updated_at) AS evidence_timestamp,
                         mal.source_type,mal.source_citation,
                         CASE WHEN mi.author_id=? THEN 'contact_authored_moment'
                              ELSE 'contact_linked_interaction' END AS relevance_reason,
                         COALESCE(msb.snapshot_revision,
                                  'asset:' || ma.asset_id) AS source_revision,
                         EXISTS(SELECT 1 FROM transcripts tr
                                  JOIN provider_jobs pj ON pj.job_id=tr.job_id
                                 WHERE tr.asset_id=ma.asset_id AND tr.status='active'
                                   AND pj.provider=? AND pj.model=? AND pj.status='completed'
                                   AND pj.request_hash=ma.content_hash) AS transcript_exists,
                         EXISTS(SELECT 1 FROM image_observations io WHERE io.asset_id=ma.asset_id AND io.status IN ('active','needs_review','proposed')) AS observation_exists,
                         EXISTS(SELECT 1 FROM media_understanding mu
                                 WHERE mu.content_sha256=ma.content_hash AND mu.status='active') AS understanding_exists
                    FROM media_assets ma
                    JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                    JOIN moment_items mi ON mi.citation=CASE
                    WHEN instr(mal.source_citation,'#media-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#media-')-1)
                    WHEN instr(mal.source_citation,'#image-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#image-')-1)
                    WHEN instr(mal.source_citation,'#video-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#video-')-1)
                    ELSE mal.source_citation END
               LEFT JOIN media_source_bindings msb ON msb.asset_id=ma.asset_id
                   WHERE ma.modality IN ('voice','image','video')
                     AND mal.source_type='moment'
                     AND (mi.author_id=? OR EXISTS (
                         SELECT 1 FROM moment_interactions mint
                          WHERE mint.moment_id=mi.moment_id AND mint.actor_id=? AND mint.status='active'
                     ))
                ORDER BY CASE WHEN mi.author_id=? THEN 0 ELSE 1 END,
                         COALESCE(mi.timestamp,ma.updated_at) DESC,ma.asset_id"""
        with self.store.connect() as conn:
            rows = list(conn.execute(
                sql, (user_id, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID, user_id, user_id, user_id)
            ))
        items: list[EligibleItem] = []
        seen: set[tuple[str, str]] = set()
        rows = deduplicate_logical_rows(rows, key=logical_moment_media_key)
        for row in rows:
            key = (str(row['citation']), str(row['modality']))
            if key in seen:
                continue
            seen.add(key)
            modality = str(row['modality'])
            complete = bool(row['transcript_exists']) if modality == 'voice' else bool(row['observation_exists'] and row['understanding_exists'])
            items.append(EligibleItem(
                asset_id=str(row['asset_id']),
                citation=str(row['citation']),
                modality=modality,
                relevance_reason=str(row['relevance_reason']),
                source_revision=str(row['source_revision']),
                content_hash=str(row['content_hash']) if row['content_hash'] else None,
                next_tool=self._next_tool(modality),
                complete=complete,
                timestamp=str(row['evidence_timestamp'] or row['updated_at'] or ''),
            ))
        return items

    def _message_media_items(
        self,
        resolved: dict[str, Any],
        *,
        include_person_group: bool = False,
    ) -> list[EligibleItem]:
        """Return every scoped media message, including unregistered assets.

        A message citation is itself enough to create a deterministic local
        recovery task. Requiring a pre-existing media_assets row silently drops
        the exact cases that need lazy discovery most.
        """

        conversation_ids, conv_marks = self._placeholders(resolved.get('conversation_ids') or [])
        sender_ids, sender_marks = self._placeholders([
            *(resolved.get('sender_ids') or []), resolved.get('primary_user_id'),
        ])
        clauses: list[str] = []
        params: list[str] = []
        if conversation_ids:
            clauses.append(f"(conversation_type='private' AND conversation_id IN ({conv_marks}))")
            params.extend(conversation_ids)
        if sender_ids:
            clauses.append(
                f"(conversation_type='private' AND (sender_id IN ({sender_marks}) OR conversation_id IN ({sender_marks})))"
            )
            params.extend(sender_ids)
            params.extend(sender_ids)
        if include_person_group and sender_ids:
            clauses.append(f"(conversation_type='group' AND sender_id IN ({sender_marks}))")
            params.extend(sender_ids)
        if not clauses:
            return []
        with self.store.connect() as conn:
            messages = list(conn.execute(
                f"""SELECT m.citation,m.content_kind,m.timestamp,m.conversation_type,
                           m.conversation_title,m.sender_name,m.shard_id,m.local_id
                       FROM messages m
                      WHERE content_kind IN ('voice','image','video')
                        AND ({' OR '.join(clauses)})
                      ORDER BY m.timestamp DESC,
                               CASE WHEN EXISTS(
                                   SELECT 1 FROM media_asset_links mal
                                    WHERE mal.source_citation=m.citation AND mal.accepted=1
                               ) OR EXISTS(
                                   SELECT 1 FROM media_assets ma WHERE ma.citation=m.citation
                               ) THEN 0 ELSE 1 END,
                               m.citation""",
                params,
            ))
            messages = deduplicate_logical_rows(messages, key=logical_message_key)
            citations_json = _json([str(row['citation']) for row in messages])
            asset_rows = list(conn.execute(
                """WITH scoped(citation) AS (SELECT CAST(value AS TEXT) FROM json_each(?)),
                          candidates AS (
                            SELECT mal.source_citation AS message_citation,ma.asset_id,ma.modality,
                                   ma.content_hash,ma.updated_at,msb.snapshot_revision,0 AS route_rank
                              FROM media_asset_links mal
                              JOIN scoped s ON s.citation=mal.source_citation
                              JOIN media_assets ma ON ma.asset_id=mal.asset_id
                         LEFT JOIN media_source_bindings msb ON msb.asset_id=ma.asset_id
                             WHERE mal.accepted=1
                            UNION ALL
                            SELECT ma.citation AS message_citation,ma.asset_id,ma.modality,
                                   ma.content_hash,ma.updated_at,msb.snapshot_revision,1 AS route_rank
                              FROM media_assets ma
                              JOIN scoped s ON s.citation=ma.citation
                         LEFT JOIN media_source_bindings msb ON msb.asset_id=ma.asset_id
                          )
                    SELECT c.*,
                           EXISTS(SELECT 1 FROM transcripts tr
                                   JOIN provider_jobs pj ON pj.job_id=tr.job_id
                                  WHERE tr.asset_id=c.asset_id AND tr.status='active'
                                    AND pj.provider=? AND pj.model=? AND pj.status='completed'
                                    AND pj.request_hash=c.content_hash) AS transcript_exists,
                           EXISTS(SELECT 1 FROM image_observations io
                                   WHERE io.asset_id=c.asset_id
                                     AND io.status IN ('active','needs_review','proposed')) AS observation_exists,
                           EXISTS(SELECT 1 FROM media_understanding mu
                                   WHERE mu.content_sha256=c.content_hash AND mu.status='active') AS understanding_exists
                      FROM candidates c
                  ORDER BY c.message_citation,c.modality,c.route_rank,c.asset_id""",
                (citations_json, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
            )) if messages else []

        assets_by_key: dict[tuple[str, str], Any] = {}
        for row in asset_rows:
            key = (str(row['message_citation']), str(row['modality']))
            current = assets_by_key.get(key)
            modality = str(row['modality'])
            complete = bool(row['transcript_exists']) if modality == 'voice' else bool(
                row['observation_exists'] and row['understanding_exists']
            )
            if current is None:
                assets_by_key[key] = row
                continue
            current_complete = bool(current['transcript_exists']) if modality == 'voice' else bool(
                current['observation_exists'] and current['understanding_exists']
            )
            if complete and not current_complete:
                assets_by_key[key] = row

        items: list[EligibleItem] = []
        for message in messages:
            citation = str(message['citation'])
            modality = str(message['content_kind'])
            asset = assets_by_key.get((citation, modality))
            complete = False
            if asset is not None:
                complete = bool(asset['transcript_exists']) if modality == 'voice' else bool(
                    asset['observation_exists'] and asset['understanding_exists']
                )
            asset_id = str(asset['asset_id']) if asset is not None else None
            items.append(EligibleItem(
                asset_id=asset_id,
                citation=citation,
                modality=modality,
                relevance_reason=(
                    'contact_group_speech'
                    if str(message['conversation_type']) == 'group'
                    else 'direct_private_chat'
                ),
                source_revision=(
                    str(asset['snapshot_revision'])
                    if asset is not None and asset['snapshot_revision']
                    else f"asset:{asset_id}" if asset_id else f"message:{_hash(citation)[:24]}"
                ),
                content_hash=str(asset['content_hash']) if asset is not None and asset['content_hash'] else None,
                next_tool=self._next_tool(modality),
                complete=complete,
                timestamp=str(message['timestamp'] or (asset['updated_at'] if asset is not None else '') or ''),
            ))
        return items

    def _appmsg_items(
        self,
        resolved: dict[str, Any],
        *,
        include_person_group: bool = False,
    ) -> list[EligibleItem]:
        conversation_ids, conv_marks = self._placeholders(resolved.get('conversation_ids') or [])
        sender_ids, sender_marks = self._placeholders([
            *(resolved.get('sender_ids') or []), resolved.get('primary_user_id'),
        ])
        private_clauses: list[str] = []
        scope_clauses: list[str] = []
        params: list[str] = []
        if conversation_ids:
            private_clauses.append(f'm.conversation_id IN ({conv_marks})')
            params.extend(conversation_ids)
        if sender_ids:
            private_clauses.append(f'(m.sender_id IN ({sender_marks}) OR m.conversation_id IN ({sender_marks}))')
            params.extend(sender_ids)
            params.extend(sender_ids)
        if private_clauses:
            scope_clauses.append(f"(m.conversation_type='private' AND ({' OR '.join(private_clauses)}))")
        if include_person_group and sender_ids:
            scope_clauses.append(f"(m.conversation_type='group' AND m.sender_id IN ({sender_marks}))")
            params.extend(sender_ids)
        if not scope_clauses:
            return []
        with self.store.connect() as conn:
            rows = list(conn.execute(
                f"""SELECT m.citation,mp.source_hash,mp.parse_status,m.timestamp,m.conversation_type,
                           m.conversation_title,m.sender_name,m.shard_id,m.local_id,m.content_kind
                       FROM messages m LEFT JOIN message_payloads mp ON mp.citation=m.citation
                      WHERE m.content_kind='appmsg'
                        AND ({' OR '.join(scope_clauses)})
                      ORDER BY m.timestamp DESC,
                               CASE WHEN mp.citation IS NOT NULL THEN 0 ELSE 1 END,
                               m.citation""",
                params,
            ))
        rows = deduplicate_logical_rows(rows, key=logical_message_key)
        return [EligibleItem(
            asset_id=None,
            citation=str(row['citation']),
            modality='appmsg',
            relevance_reason=(
                'contact_group_speech_appmsg'
                if str(row['conversation_type']) == 'group'
                else 'direct_private_chat_appmsg'
            ),
            source_revision=f"payload:{row['source_hash'] or 'missing'}",
            content_hash=str(row['source_hash'] or '') or None,
            next_tool=self._next_tool('appmsg'),
            complete=str(row['parse_status'] or '') in {'parsed', 'normalized'},
            timestamp=str(row['timestamp'] or ''),
        ) for row in rows]

    def discover(
        self,
        customer: str,
        *,
        purpose: str = 'customer_profile_enrichment',
    ) -> tuple[dict[str, Any], list[EligibleItem], str]:
        _, resolved = self._resolved_scope(customer)
        include_person_group = purpose == 'person_relationship_profile_enrichment'
        items = self._message_media_items(
            resolved,
            include_person_group=include_person_group,
        ) + self._moment_media_items(resolved) + self._appmsg_items(
            resolved,
            include_person_group=include_person_group,
        )
        deduplicated: dict[str, EligibleItem] = {}
        for item in items:
            current = deduplicated.get(item.identity())
            if current is None or (item.complete and not current.complete):
                deduplicated[item.identity()] = item
        items = list(deduplicated.values())
        rank = {
            'direct_private_chat': 0,
            'direct_private_chat_appmsg': 0,
            'contact_group_speech': 1,
            'contact_group_speech_appmsg': 1,
            'contact_authored_moment': 2,
            'contact_linked_interaction': 3,
        }
        # Stable sorts preserve relevance first and newest evidence inside each
        # relevance tier without parsing source-specific timestamp formats.
        items.sort(key=lambda item: item.citation)
        items.sort(key=lambda item: item.timestamp, reverse=True)
        items.sort(key=lambda item: rank.get(item.relevance_reason, 9))
        source_revision = 'src-' + _hash('\n'.join(sorted(item.identity() for item in items)))[:24]
        return resolved, items, source_revision

    def summarize_discovery(
        self,
        customer: str,
        *,
        purpose: str = 'customer_profile_enrichment',
    ) -> tuple[dict[str, Any], dict[str, int], str]:
        """Aggregate deferred media counts in SQL with constant Python memory."""

        _, resolved = self._resolved_scope(customer)
        include_person_group = purpose == 'person_relationship_profile_enrichment'
        conversation_ids, conv_marks = self._placeholders(
            resolved.get('conversation_ids') or [],
        )
        sender_ids, sender_marks = self._placeholders([
            *(resolved.get('sender_ids') or []), resolved.get('primary_user_id'),
        ])
        scope_clauses: list[str] = []
        scope_params: list[str] = []
        if conversation_ids:
            scope_clauses.append(
                f"(m.conversation_type='private' AND m.conversation_id IN ({conv_marks}))"
            )
            scope_params.extend(conversation_ids)
        if sender_ids:
            scope_clauses.append(
                f"(m.conversation_type='private' AND "
                f"(m.sender_id IN ({sender_marks}) OR m.conversation_id IN ({sender_marks})))"
            )
            scope_params.extend(sender_ids)
            scope_params.extend(sender_ids)
        if include_person_group and sender_ids:
            scope_clauses.append(
                f"(m.conversation_type='group' AND m.sender_id IN ({sender_marks}))"
            )
            scope_params.extend(sender_ids)

        summary: dict[str, dict[str, int]] = {}

        def add_rows(rows: Iterable[Any]) -> None:
            for row in rows:
                modality = str(row['modality'])
                counters = summary.setdefault(modality, {'total': 0, 'deferred': 0})
                counters['total'] += int(row['total'] or 0)
                counters['deferred'] += int(row['deferred'] or 0)

        with self.store.connect() as conn:
            if scope_clauses:
                add_rows(conn.execute(
                    f"""WITH scoped AS (
                            SELECT m.citation,m.content_kind
                              FROM messages m
                             WHERE m.content_kind IN ('voice','image','video')
                               AND ({' OR '.join(scope_clauses)})
                        ), candidates AS (
                            SELECT mal.source_citation AS message_citation,
                                   ma.asset_id,ma.modality,ma.content_hash
                              FROM media_asset_links mal
                              JOIN media_assets ma ON ma.asset_id=mal.asset_id
                              JOIN scoped sm ON sm.citation=mal.source_citation
                                            AND sm.content_kind=ma.modality
                             WHERE mal.accepted=1
                            UNION
                            SELECT ma.citation,ma.asset_id,ma.modality,ma.content_hash
                              FROM media_assets ma
                              JOIN scoped sm ON sm.citation=ma.citation
                                            AND sm.content_kind=ma.modality
                             WHERE ma.citation IS NOT NULL
                        ), completion AS (
                            SELECT s.citation,s.content_kind AS modality,
                                   MAX(CASE
                                       WHEN c.asset_id IS NULL THEN 0
                                       WHEN s.content_kind='voice' THEN EXISTS(
                                           SELECT 1 FROM transcripts tr
                                           JOIN provider_jobs pj ON pj.job_id=tr.job_id
                                           WHERE tr.asset_id=c.asset_id AND tr.status='active'
                                             AND pj.provider=? AND pj.model=?
                                             AND pj.status='completed'
                                             AND pj.request_hash=c.content_hash
                                       )
                                       ELSE EXISTS(
                                           SELECT 1 FROM image_observations io
                                            WHERE io.asset_id=c.asset_id
                                              AND io.status IN ('active','needs_review','proposed')
                                       ) AND EXISTS(
                                           SELECT 1 FROM media_understanding mu
                                            WHERE mu.content_sha256=c.content_hash
                                              AND mu.status='active'
                                       ) END) AS complete
                              FROM scoped s
                         LEFT JOIN candidates c
                                ON c.message_citation=s.citation
                               AND c.modality=s.content_kind
                          GROUP BY s.citation,s.content_kind
                        )
                        SELECT modality,COUNT(*) AS total,
                               SUM(CASE WHEN complete=0 THEN 1 ELSE 0 END) AS deferred
                          FROM completion GROUP BY modality""",
                    (
                        *scope_params,
                        CLOUD_ASR_PROVIDER_NAME,
                        CLOUD_ASR_MODEL_ID,
                    ),
                ))
                appmsg = conn.execute(
                    f"""SELECT 'appmsg' AS modality,COUNT(*) AS total,COUNT(*) AS deferred
                           FROM messages m LEFT JOIN message_payloads mp ON mp.citation=m.citation
                          WHERE m.content_kind='appmsg'
                            AND ({' OR '.join(scope_clauses)})
                            AND COALESCE(mp.parse_status,'missing') NOT IN ('parsed','normalized')""",
                    scope_params,
                ).fetchone()
                if appmsg is not None and int(appmsg['total'] or 0):
                    add_rows((appmsg,))

            user_id = str(resolved.get('primary_user_id') or '').strip()
            if user_id:
                add_rows(conn.execute(
                    """WITH scoped AS (
                            SELECT ma.citation,ma.modality,ma.asset_id,ma.content_hash
                              FROM media_assets ma
                              JOIN media_asset_links mal
                                ON mal.asset_id=ma.asset_id AND mal.accepted=1
                              JOIN moment_items mi ON mi.citation=CASE
                                   WHEN instr(mal.source_citation,'#media-')>0
                                     THEN substr(mal.source_citation,1,instr(mal.source_citation,'#media-')-1)
                                   WHEN instr(mal.source_citation,'#image-')>0
                                     THEN substr(mal.source_citation,1,instr(mal.source_citation,'#image-')-1)
                                   WHEN instr(mal.source_citation,'#video-')>0
                                     THEN substr(mal.source_citation,1,instr(mal.source_citation,'#video-')-1)
                                   ELSE mal.source_citation END
                             WHERE ma.modality IN ('voice','image','video')
                               AND mal.source_type='moment'
                               AND (mi.author_id=? OR EXISTS(
                                   SELECT 1 FROM moment_interactions mint
                                    WHERE mint.moment_id=mi.moment_id
                                      AND mint.actor_id=? AND mint.status='active'
                               ))
                        ), completion AS (
                            SELECT citation,modality,MAX(CASE
                                WHEN modality='voice' THEN EXISTS(
                                    SELECT 1 FROM transcripts tr
                                    JOIN provider_jobs pj ON pj.job_id=tr.job_id
                                    WHERE tr.asset_id=scoped.asset_id AND tr.status='active'
                                      AND pj.provider=? AND pj.model=?
                                      AND pj.status='completed'
                                      AND pj.request_hash=scoped.content_hash
                                )
                                ELSE EXISTS(
                                    SELECT 1 FROM image_observations io
                                     WHERE io.asset_id=scoped.asset_id
                                       AND io.status IN ('active','needs_review','proposed')
                                ) AND EXISTS(
                                    SELECT 1 FROM media_understanding mu
                                     WHERE mu.content_sha256=scoped.content_hash
                                       AND mu.status='active'
                                ) END) AS complete
                              FROM scoped GROUP BY citation,modality
                        )
                        SELECT modality,COUNT(*) AS total,
                               SUM(CASE WHEN complete=0 THEN 1 ELSE 0 END) AS deferred
                          FROM completion GROUP BY modality""",
                    (
                        user_id,
                        user_id,
                        CLOUD_ASR_PROVIDER_NAME,
                        CLOUD_ASR_MODEL_ID,
                    ),
                ))

        deferred = {
            modality: counters['deferred']
            for modality, counters in sorted(summary.items())
            if counters['deferred'] > 0
        }
        revision_payload = {
            'entity_id': str(resolved['entity_id']),
            'purpose': purpose,
            'modalities': dict(sorted(summary.items())),
        }
        source_revision = 'src-summary-' + _hash(_json(revision_payload))[:24]
        return resolved, deferred, source_revision

    def plan(
        self,
        customer: str,
        *,
        actor: str,
        session: str,
        mode: str = 'complete',
        execution_location: str = 'local',
        processor_identity: str = 'local-agent/default',
        prompt_version: str = 'profile-enrichment/v1',
        purpose: str = 'customer_profile_enrichment',
        item_budget: int = 500,
        cost_budget_rmb: float = 0.0,
        page_limit: int = 100,
        page_offset: int = 0,
        prepared_discovery: tuple[dict[str, Any], list[EligibleItem], str] | None = None,
    ) -> dict[str, Any]:
        if mode not in {'standard', 'complete'}:
            raise ProfileEnrichmentError('mode must be standard or complete', code='invalid_enrichment_mode')
        if execution_location not in {'local', 'remote'}:
            raise ProfileEnrichmentError('execution_location must be local or remote', code='invalid_execution_attestation')
        if type(item_budget) is not int or not 1 <= item_budget <= MAX_MANIFEST_ITEMS:
            raise ProfileEnrichmentError('item_budget is out of range', code='invalid_enrichment_budget')
        if not _valid_cost(cost_budget_rmb):
            raise ProfileEnrichmentError('cost_budget_rmb is out of range', code='invalid_enrichment_budget')
        actor_hash = self._owner_hash(actor, 'actor')
        session_hash = self._owner_hash(session, 'session')
        processor_identity = _bounded_identity(processor_identity, 'processor_identity')
        prompt_version = _bounded_identity(prompt_version, 'prompt_version')
        purpose = _bounded_identity(purpose, 'purpose')
        resolved, items, source_revision = (
            prepared_discovery
            if prepared_discovery is not None
            else self.discover(customer, purpose=purpose)
        )
        if mode == 'standard':
            items = [item for item in items if item.relevance_reason in {'direct_private_chat', 'direct_private_chat_appmsg'}]
            source_revision = 'src-' + _hash('\n'.join(sorted(item.identity() for item in items)))[:24]
        entity_id = str(resolved['entity_id'])
        consent = {
            'entity_id': entity_id,
            'mode': mode,
            'source_revision': source_revision,
            'actor_hash': actor_hash,
            'session_hash': session_hash,
            'execution_location': execution_location,
            'processor_identity': processor_identity,
            'prompt_version': prompt_version,
            'purpose': purpose,
            'item_budget': item_budget,
            'cost_budget_rmb': round(float(cost_budget_rmb), 6),
        }
        consent_hash = _hash(_json(consent))
        plan_key = _hash(_json(consent | {'manifest_version': MANIFEST_VERSION}))
        run_id = _stable('enrichrun', plan_key)
        now = _iso(_utcnow())
        estimated_total = round(sum(self._cost(item.modality, execution_location) for item in items if not item.complete), 6)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            existing = conn.execute('SELECT * FROM profile_enrichment_runs WHERE plan_key=?', (plan_key,)).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO profile_enrichment_runs(
                           run_id,plan_key,entity_id,mode,state,source_revision,actor_hash,session_hash,consent_hash,
                           execution_location,processor_identity,prompt_version,purpose,item_budget,cost_budget_rmb,estimated_cost_rmb,
                           actual_cost_rmb,deferred_count,manifest_version,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, plan_key, entity_id, mode, 'pending', source_revision, actor_hash, session_hash, consent_hash,
                     execution_location, processor_identity, prompt_version, purpose, item_budget, float(cost_budget_rmb), estimated_total,
                     0.0, max(len([item for item in items if not item.complete]) - item_budget, 0), MANIFEST_VERSION, now, now),
                )
                consumed = 0
                estimated_consumed = 0.0
                for item in items:
                    task_id = _stable('enrichtask', f'{run_id}|{item.identity()}')
                    estimate = self._cost(item.modality, execution_location)
                    approval_required = execution_location == 'remote' and item.modality in {'voice', 'image', 'video'}
                    if item.complete:
                        state, reason = 'completed', 'cache_hit'
                    elif consumed >= item_budget or (float(cost_budget_rmb) > 0 and estimated_consumed + estimate > float(cost_budget_rmb)):
                        state, reason = 'paused_budget', 'item_or_cost_budget_exhausted'
                    elif approval_required:
                        state, reason = 'awaiting_approval', None
                        consumed += 1
                        estimated_consumed += estimate
                    else:
                        state, reason = 'pending', None
                        consumed += 1
                        estimated_consumed += estimate
                    completed_at = now if state == 'completed' else None
                    conn.execute(
                        """INSERT INTO profile_enrichment_tasks(
                               task_id,run_id,asset_id,citation,modality,relevance_reason,source_revision,content_hash,
                               state,next_tool,approval_required,processor_identity,prompt_version,terminal_reason,estimated_cost_rmb,
                               created_at,updated_at,completed_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (task_id, run_id, item.asset_id, item.citation, item.modality, item.relevance_reason,
                         item.source_revision, item.content_hash, state, item.next_tool, int(approval_required),
                         processor_identity, prompt_version, reason, estimate, now, now, completed_at),
                    )
                self._refresh_run_state(conn, run_id, now=now)
                conn.commit()
            else:
                run_id = str(existing['run_id'])
                conn.rollback()
        return self.manifest(run_id, actor=actor, session=session, limit=page_limit, offset=page_offset)

    def _owned_run(self, conn: Any, run_id: str, *, actor: str, session: str) -> Any:
        row = conn.execute('SELECT * FROM profile_enrichment_runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise ProfileEnrichmentError('enrichment run not found', code='enrichment_run_not_found')
        if row['revoked_at']:
            raise ProfileEnrichmentError('enrichment run has been revoked', code='enrichment_run_revoked')
        actor_hash = self._owner_hash(actor, 'actor')
        session_hash = self._owner_hash(session, 'session')
        if not hmac.compare_digest(str(row['actor_hash']), actor_hash) or not hmac.compare_digest(str(row['session_hash']), session_hash):
            raise ProfileEnrichmentError('enrichment run is owned by another actor or session', code='enrichment_owner_mismatch')
        return row

    @staticmethod
    def _task_payload(row: Any) -> dict[str, Any]:
        return {
            'task_id': row['task_id'],
            'citation': row['citation'],
            'modality': row['modality'],
            'relevance_reason': row['relevance_reason'],
            'source_revision': row['source_revision'],
            'content_hash': row['content_hash'],
            'state': row['state'],
            'attempt_count': int(row['attempt_count'] or 0),
            'terminal_reason': row['terminal_reason'],
            'next_tool': row['next_tool'],
            'approval_required': bool(row['approval_required']),
            'approval_id': row['approval_id'],
            'processor_identity': row['processor_identity'],
            'prompt_version': row['prompt_version'],
            'estimated_cost_rmb': float(row['estimated_cost_rmb'] or 0),
            'actual_cost_rmb': float(row['actual_cost_rmb'] or 0),
            'worker_bound': bool(row['lease_owner_hash']),
            'lease_active': bool(row['lease_owner_hash'] and row['lease_expires_at']),
            'lease_expires_at': row['lease_expires_at'],
        }

    @staticmethod
    def _committed_or_reserved_cost(conn: Any, run_id: str, *, exclude_task_id: str) -> float:
        marks = ','.join('?' for _ in COST_RESERVATION_STATES)
        row = conn.execute(
            f"""SELECT COALESCE(SUM(CASE
                       WHEN state='completed' THEN actual_cost_rmb
                       WHEN state IN ({marks}) THEN estimated_cost_rmb
                       ELSE 0 END),0)
                  FROM profile_enrichment_tasks
                 WHERE run_id=? AND task_id<>?""",
            (*COST_RESERVATION_STATES, run_id, exclude_task_id),
        ).fetchone()
        return float(row[0] or 0)

    def manifest(self, run_id: str, *, actor: str, session: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ProfileEnrichmentError('limit must be from 1 to 500', code='invalid_manifest_page')
        if type(offset) is not int or offset < 0:
            raise ProfileEnrichmentError('offset must be non-negative', code='invalid_manifest_page')
        with self.store.connect() as conn:
            run = self._owned_run(conn, run_id, actor=actor, session=session)
            total = int(conn.execute('SELECT COUNT(*) FROM profile_enrichment_tasks WHERE run_id=?', (run_id,)).fetchone()[0])
            rows = list(conn.execute(
                """SELECT * FROM profile_enrichment_tasks WHERE run_id=?
                   ORDER BY CASE relevance_reason WHEN 'direct_private_chat' THEN 0 WHEN 'direct_private_chat_appmsg' THEN 0
                                WHEN 'contact_authored_moment' THEN 1 ELSE 2 END, created_at, task_id
                   LIMIT ? OFFSET ?""",
                (run_id, limit, offset),
            ))
            counts = {str(row['state']): int(row['n']) for row in conn.execute(
                'SELECT state,COUNT(*) AS n FROM profile_enrichment_tasks WHERE run_id=? GROUP BY state', (run_id,),
            )}
            execution = conn.execute(
                """SELECT
                       SUM(CASE WHEN attempt_count>0 THEN 1 ELSE 0 END) AS attempted,
                       SUM(CASE WHEN state='completed' THEN 1 ELSE 0 END) AS completed,
                       SUM(CASE WHEN state='completed' AND terminal_reason='cache_hit' THEN 1 ELSE 0 END) AS cache_hit,
                       SUM(CASE WHEN state='unavailable' THEN 1 ELSE 0 END) AS unavailable
                   FROM profile_enrichment_tasks WHERE run_id=?""",
                (run_id,),
            ).fetchone()
        return {
            'ok': True,
            'type': 'profile_enrichment_manifest',
            'run_id': run_id,
            'entity_id': run['entity_id'],
            'mode': run['mode'],
            'state': run['state'],
            'source_revision': run['source_revision'],
            'manifest_version': run['manifest_version'],
            'execution_location': run['execution_location'],
            'processor_identity': run['processor_identity'],
            'prompt_version': run['prompt_version'],
            'purpose': run['purpose'],
            'budget': {
                'items': int(run['item_budget']),
                'cost_rmb': float(run['cost_budget_rmb']),
                'estimated_cost_rmb': float(run['estimated_cost_rmb']),
                'actual_cost_rmb': float(run['actual_cost_rmb']),
            },
            'counts': counts,
            'execution_summary': {
                'attempted': int(execution['attempted'] or 0),
                'completed': int(execution['completed'] or 0),
                'cache_hit': int(execution['cache_hit'] or 0),
                'unavailable': int(execution['unavailable'] or 0),
            },
            'items': [self._task_payload(row) for row in rows],
            'page': {'offset': offset, 'limit': limit, 'total': total, 'next_offset': offset + len(rows) if offset + len(rows) < total else None},
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def reclaim_expired(self, run_id: str, *, actor: str, session: str, now: datetime | None = None) -> int:
        current = now or _utcnow()
        current_iso = _iso(current)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            run = self._owned_run(conn, run_id, actor=actor, session=session)
            cursor = conn.execute(
                """UPDATE profile_enrichment_tasks
                      SET state='pending',claim_token_hash=NULL,delivery_token_hash=NULL,delivery_consumed_at=NULL,
                          lease_owner_hash=NULL,lease_expires_at=NULL,heartbeat_at=NULL,
                          terminal_reason='lease_expired_reclaimed',estimated_cost_rmb=0,updated_at=?
                    WHERE run_id=? AND state IN ('materializing','processing','awaiting_agent','awaiting_approval')
                      AND lease_expires_at IS NOT NULL AND lease_expires_at<=?""",
                (current_iso, run_id, current_iso),
            )
            changed = max(cursor.rowcount, 0)
            if changed:
                self._refresh_run_state(conn, run_id, now=current_iso)
            conn.commit()
        return changed

    def claim(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        execution_location: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if type(lease_seconds) is not int or not 30 <= lease_seconds <= MAX_LEASE_SECONDS:
            raise ProfileEnrichmentError('lease_seconds must be from 30 to 1800', code='invalid_task_lease')
        lease_owner_hash = self._owner_hash(lease_owner, 'lease_owner')
        current = now or _utcnow()
        self.reclaim_expired(run_id, actor=actor, session=session, now=current)
        claim_token = secrets.token_urlsafe(32)
        delivery_token = secrets.token_urlsafe(32)
        now_iso = _iso(current)
        expiry = _iso(current + timedelta(seconds=lease_seconds))
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            run = self._owned_run(conn, run_id, actor=actor, session=session)
            if str(run['execution_location']) != execution_location:
                raise ProfileEnrichmentError('execution location does not match the run attestation', code='execution_attestation_mismatch')
            row = conn.execute('SELECT * FROM profile_enrichment_tasks WHERE run_id=? AND task_id=?', (run_id, task_id)).fetchone()
            if row is None:
                raise ProfileEnrichmentError('enrichment task not found in run', code='enrichment_task_not_found')
            if row['state'] == 'completed':
                conn.rollback()
                return {'ok': True, 'cache_hit': True, 'task': self._task_payload(row), 'raw_content_included': False}
            if row['state'] not in {'pending', 'retryable_failure'}:
                raise ProfileEnrichmentError(f"task cannot be claimed from state {row['state']}", code='enrichment_task_not_claimable')
            cursor = conn.execute(
                """UPDATE profile_enrichment_tasks
                      SET state='materializing',attempt_count=attempt_count+1,claim_token_hash=?,delivery_token_hash=?,
                          delivery_consumed_at=NULL,lease_owner_hash=?,lease_expires_at=?,heartbeat_at=?,
                          terminal_reason=NULL,updated_at=?
                    WHERE run_id=? AND task_id=? AND state IN ('pending','retryable_failure')""",
                (_hash(claim_token), _hash(delivery_token), lease_owner_hash, expiry, now_iso, now_iso, run_id, task_id),
            )
            if cursor.rowcount != 1:
                raise ProfileEnrichmentError('task claim lost a concurrent race', code='enrichment_task_claim_conflict')
            conn.execute("UPDATE profile_enrichment_runs SET state='running',updated_at=? WHERE run_id=?", (now_iso, run_id))
            claimed = conn.execute('SELECT * FROM profile_enrichment_tasks WHERE task_id=?', (task_id,)).fetchone()
            conn.commit()
        action = {
            'tool': claimed['next_tool'],
            'citation': claimed['citation'],
            'run_id': run_id,
            'task_id': task_id,
            'claim_token': claim_token,
        }
        if claimed['modality'] in {'image', 'video'} and execution_location == 'local':
            action['media_capability'] = delivery_token
            action['media_capability_one_time'] = True
        return {
            'ok': True,
            'task': self._task_payload(claimed),
            'agent_action': action,
            'lease': {'owner_bound': True, 'expires_at': expiry},
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def _claimed_task(self, conn: Any, run_id: str, task_id: str, *, actor: str, session: str, lease_owner: str, claim_token: str, now: datetime) -> tuple[Any, Any]:
        run = self._owned_run(conn, run_id, actor=actor, session=session)
        task = conn.execute('SELECT * FROM profile_enrichment_tasks WHERE run_id=? AND task_id=?', (run_id, task_id)).fetchone()
        if task is None:
            raise ProfileEnrichmentError('enrichment task not found in run', code='enrichment_task_not_found')
        token_hash = str(task['claim_token_hash'] or '')
        if not token_hash or not hmac.compare_digest(token_hash, _hash(str(claim_token or ''))):
            raise ProfileEnrichmentError('claim token is invalid or no longer active', code='invalid_claim_token')
        owner_hash = self._owner_hash(lease_owner, 'lease_owner')
        if not hmac.compare_digest(str(task['lease_owner_hash'] or ''), owner_hash):
            raise ProfileEnrichmentError('task lease is owned by another worker', code='task_lease_owner_mismatch')
        expiry = _parse_time(task['lease_expires_at'])
        if expiry is None or expiry <= now:
            raise ProfileEnrichmentError('task lease has expired', code='task_lease_expired')
        return run, task

    def redeem_local_media(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        media_capability: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _utcnow()
        now_iso = _iso(current)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            run, task = self._claimed_task(conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner, claim_token=claim_token, now=current)
            if run['execution_location'] != 'local' or task['modality'] not in {'image', 'video'}:
                raise ProfileEnrichmentError('local media capability is not valid for this task', code='invalid_media_capability_scope')
            delivery_hash = str(task['delivery_token_hash'] or '')
            if task['delivery_consumed_at'] or not delivery_hash or not hmac.compare_digest(delivery_hash, _hash(str(media_capability or ''))):
                raise ProfileEnrichmentError('media capability is invalid or already consumed', code='media_capability_replayed')
            cursor = conn.execute(
                """UPDATE profile_enrichment_tasks SET delivery_consumed_at=?,delivery_token_hash=NULL,
                          state='awaiting_agent',next_tool='trove_profile_enrichment_image_annotate',updated_at=?
                    WHERE task_id=? AND delivery_consumed_at IS NULL""",
                (now_iso, now_iso, task_id),
            )
            if cursor.rowcount != 1:
                raise ProfileEnrichmentError('media capability was consumed concurrently', code='media_capability_replayed')
            conn.commit()
        return {
            'ok': True,
            'citation': task['citation'],
            'task_id': task_id,
            'run_id': run_id,
            'next_tool': 'trove_media_fetch',
            'agent_instruction': 'Analyze the redeemed exact local preview, then call trove_profile_enrichment_image_annotate with the same claim; it persists cited evidence and completes the task.',
            'raw_content_included': False,
            'raw_paths_included': False,
        }

    def heartbeat(self, run_id: str, task_id: str, *, actor: str, session: str, lease_owner: str, claim_token: str, lease_seconds: int = DEFAULT_LEASE_SECONDS, now: datetime | None = None) -> dict[str, Any]:
        if type(lease_seconds) is not int or not 30 <= lease_seconds <= MAX_LEASE_SECONDS:
            raise ProfileEnrichmentError('lease_seconds must be from 30 to 1800', code='invalid_task_lease')
        current = now or _utcnow()
        expiry = _iso(current + timedelta(seconds=lease_seconds))
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            _, task = self._claimed_task(conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner, claim_token=claim_token, now=current)
            conn.execute('UPDATE profile_enrichment_tasks SET lease_expires_at=?,heartbeat_at=?,updated_at=? WHERE task_id=?', (expiry, _iso(current), _iso(current), task_id))
            conn.commit()
        return {'ok': True, 'task_id': task_id, 'lease_expires_at': expiry, 'raw_content_included': False}

    def voice_cloud_scope(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        estimated_cost_rmb: float | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if estimated_cost_rmb is not None and not _valid_cost(estimated_cost_rmb, positive=True):
            raise ProfileEnrichmentError('estimated_cost_rmb is out of range', code='invalid_enrichment_cost')
        current = now or _utcnow()
        with self.store.connect() as conn:
            # The immediate transaction makes the per-task reservation and the
            # run ceiling check one atomic operation across concurrent workers.
            conn.execute('BEGIN IMMEDIATE')
            run, task = self._claimed_task(
                conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner,
                claim_token=claim_token, now=current,
            )
            if task['modality'] != 'voice':
                raise ProfileEnrichmentError('task is not a voice task', code='enrichment_task_modality_mismatch')
            asset = conn.execute('SELECT content_hash FROM media_assets WHERE asset_id=?', (task['asset_id'],)).fetchone()
            content_hash = str(asset['content_hash'] or '').lower() if asset is not None else ''
            if len(content_hash) != 64 or any(char not in '0123456789abcdef' for char in content_hash):
                raise ProfileEnrichmentError(
                    'materialized voice content hash is required before cloud approval',
                    code='voice_content_hash_required',
                )
            if str(task['content_hash'] or '').lower() != content_hash:
                conn.execute('UPDATE profile_enrichment_tasks SET content_hash=?,updated_at=? WHERE task_id=?', (content_hash, _iso(current), task_id))
            run_budget = float(run['cost_budget_rmb'] or 0)
            existing_reservation = float(task['estimated_cost_rmb'] or 0)
            ceiling: float | None = None
            if run_budget > 0:
                other_cost = self._committed_or_reserved_cost(conn, run_id, exclude_task_id=task_id)
                available = max(run_budget - other_cost, 0.0)
                requested = available if estimated_cost_rmb is None else float(estimated_cost_rmb)
                reservation = max(existing_reservation, requested)
                if reservation <= 0 or other_cost + reservation > run_budget:
                    raise ProfileEnrichmentError(
                        'profile run cost budget cannot authorize cloud ASR',
                        code='enrichment_cost_budget_exhausted',
                    )
                ceiling = reservation
            else:
                reservation = max(existing_reservation, float(estimated_cost_rmb or 0))
            if reservation != existing_reservation:
                conn.execute(
                    'UPDATE profile_enrichment_tasks SET estimated_cost_rmb=?,updated_at=? WHERE task_id=?',
                    (reservation, _iso(current), task_id),
                )
            conn.commit()
            citation = str(task['citation'])
            return {
                'profile_run_hash': _hash(str(run['run_id'])),
                'task_set_hash': _hash(str(task['task_id'])),
                'citation_set_hash': _hash(citation),
                'source_revision_hash': _hash(str(task['source_revision'])),
                'content_hash': content_hash,
                'actor_hash': str(run['actor_hash']),
                'session_hash': str(run['session_hash']),
                'purpose': str(run['purpose']),
                # cost_budget_rmb=0 is the explicit unlimited policy. Actual
                # usage is still recorded; only the blocking ceiling is absent.
                'cost_ceiling_rmb': None if ceiling is None else round(ceiling, 6),
            }

    def image_annotation_scope(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _utcnow()
        with self.store.connect() as conn:
            run, task = self._claimed_task(
                conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner,
                claim_token=claim_token, now=current,
            )
            if run['execution_location'] != 'local':
                raise ProfileEnrichmentError('image annotation requires attested local execution', code='execution_attestation_mismatch')
            if task['modality'] not in {'image', 'video'}:
                raise ProfileEnrichmentError('task is not visual media', code='enrichment_task_modality_mismatch')
            if task['state'] != 'awaiting_agent' or not task['delivery_consumed_at']:
                raise ProfileEnrichmentError('one-time local media delivery must be redeemed first', code='media_capability_not_redeemed')
            asset = conn.execute('SELECT content_hash FROM media_assets WHERE asset_id=?', (task['asset_id'],)).fetchone()
            content_hash = str(asset['content_hash'] or '').lower() if asset is not None else ''
            if len(content_hash) != 64 or any(char not in '0123456789abcdef' for char in content_hash):
                raise ProfileEnrichmentError('materialized visual content hash is required', code='visual_content_hash_required')
            if str(task['content_hash'] or '').lower() != content_hash:
                conn.execute('UPDATE profile_enrichment_tasks SET content_hash=?,updated_at=? WHERE task_id=?', (content_hash, _iso(current), task_id))
                conn.commit()
            return {
                'run_id': run_id,
                'task_id': task_id,
                'asset_id': str(task['asset_id']),
                'citation': str(task['citation']),
                'content_sha256': content_hash,
                'model_id': str(task['processor_identity']),
                'prompt_version': str(task['prompt_version']),
            }

    def awaiting_approval(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        approval_id: str,
        approval_scope_hash: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _utcnow()
        now_iso = _iso(current)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            _, task = self._claimed_task(
                conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner,
                claim_token=claim_token, now=current,
            )
            conn.execute(
                """UPDATE profile_enrichment_tasks
                      SET state='awaiting_approval',approval_required=1,approval_id=?,approval_scope_hash=?,
                          next_tool='trove_profile_enrichment_voice_execute',updated_at=? WHERE task_id=?""",
                (_bounded_identity(approval_id, 'approval_id'), _bounded_identity(approval_scope_hash, 'approval_scope_hash'), now_iso, task_id),
            )
            self._refresh_run_state(conn, run_id, now=now_iso)
            conn.commit()
        return {'ok': True, 'task_id': task_id, 'state': 'awaiting_approval', 'raw_content_included': False}

    def pause_budget(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        reason: str = 'cloud_cost_budget_required',
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _utcnow()
        now_iso = _iso(current)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._claimed_task(
                conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner,
                claim_token=claim_token, now=current,
            )
            conn.execute(
                """UPDATE profile_enrichment_tasks
                      SET state='paused_budget',terminal_reason=?,claim_token_hash=NULL,delivery_token_hash=NULL,
                          lease_owner_hash=NULL,lease_expires_at=NULL,heartbeat_at=NULL,
                          estimated_cost_rmb=0,updated_at=? WHERE task_id=?""",
                (_bounded_identity(reason, 'terminal_reason'), now_iso, task_id),
            )
            self._refresh_run_state(conn, run_id, now=now_iso)
            conn.commit()
        return {'ok': True, 'task_id': task_id, 'state': 'paused_budget', 'reason': reason, 'raw_content_included': False}

    @staticmethod
    def _completion_prerequisite(conn: Any, task: Any) -> tuple[bool, str | None, str | None]:
        modality = str(task['modality'])
        if modality == 'voice':
            hit = conn.execute(
                """SELECT 1
                     FROM transcripts tr
                     JOIN provider_jobs pj ON pj.job_id=tr.job_id
                     JOIN media_assets ma ON ma.asset_id=tr.asset_id
                    WHERE tr.asset_id=? AND tr.status='active'
                      AND pj.provider=? AND pj.model=? AND pj.status='completed'
                      AND pj.request_hash=ma.content_hash
                      AND ma.content_hash=?
                    LIMIT 1""",
                (
                    task['asset_id'], CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID,
                    str(task['content_hash'] or ''),
                ),
            ).fetchone()
            return hit is not None, 'matching_transcript_required', None
        if modality in {'image', 'video'}:
            obs = conn.execute("SELECT 1 FROM image_observations WHERE asset_id=? AND status IN ('active','needs_review','proposed') LIMIT 1", (task['asset_id'],)).fetchone()
            asset = conn.execute('SELECT content_hash FROM media_assets WHERE asset_id=?', (task['asset_id'],)).fetchone()
            content_hash = str(asset['content_hash'] or '') if asset is not None else ''
            understanding = conn.execute(
                "SELECT 1 FROM media_understanding WHERE content_sha256=? AND status='active' LIMIT 1",
                (content_hash,),
            ).fetchone() if content_hash else None
            return obs is not None and understanding is not None, 'matching_media_annotation_required', content_hash or None
        if modality == 'appmsg':
            hit = conn.execute("SELECT 1 FROM message_payloads WHERE citation=? AND parse_status IN ('parsed','normalized') LIMIT 1", (task['citation'],)).fetchone()
            return hit is not None, 'normalized_appmsg_required', None
        return False, 'unsupported_enrichment_modality', None

    def complete(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        completion_key: str,
        actual_cost_rmb: float = 0.0,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        completion_key = _bounded_identity(completion_key, 'completion_key')
        if not _valid_cost(actual_cost_rmb):
            raise ProfileEnrichmentError('actual_cost_rmb is out of range', code='invalid_enrichment_cost')
        current = now or _utcnow()
        now_iso = _iso(current)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            run = self._owned_run(conn, run_id, actor=actor, session=session)
            task = conn.execute('SELECT * FROM profile_enrichment_tasks WHERE run_id=? AND task_id=?', (run_id, task_id)).fetchone()
            if task is None:
                raise ProfileEnrichmentError('enrichment task not found in run', code='enrichment_task_not_found')
            if task['state'] == 'completed':
                if hmac.compare_digest(str(task['completion_key'] or ''), completion_key):
                    conn.rollback()
                    return {'ok': True, 'idempotent': True, 'task': self._task_payload(task), 'raw_content_included': False}
                raise ProfileEnrichmentError('completed task cannot accept a different completion key', code='completion_replay_rejected')
            _, task = self._claimed_task(conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner, claim_token=claim_token, now=current)
            valid, reason, observed_content_hash = self._completion_prerequisite(conn, task)
            if not valid:
                raise ProfileEnrichmentError(reason or 'completion prerequisite missing', code=reason or 'completion_prerequisite_missing')
            used = conn.execute('SELECT task_id FROM profile_enrichment_tasks WHERE completion_key=? AND task_id<>?', (completion_key, task_id)).fetchone()
            if used is not None:
                raise ProfileEnrichmentError('completion key was already used by another task', code='completion_replay_rejected')
            run_budget = float(run['cost_budget_rmb'] or 0)
            if run_budget > 0:
                other_cost = self._committed_or_reserved_cost(conn, run_id, exclude_task_id=task_id)
                if other_cost + float(actual_cost_rmb) > run_budget:
                    raise ProfileEnrichmentError('actual task cost exceeds the immutable run ceiling', code='enrichment_cost_budget_exhausted')
            conn.execute(
                """UPDATE profile_enrichment_tasks
                      SET state='completed',completion_key=?,estimated_cost_rmb=0,actual_cost_rmb=?,content_hash=COALESCE(?,content_hash),terminal_reason=NULL,
                          claim_token_hash=NULL,delivery_token_hash=NULL,lease_owner_hash=NULL,lease_expires_at=NULL,
                          heartbeat_at=NULL,updated_at=?,completed_at=?
                    WHERE task_id=?""",
                (completion_key, float(actual_cost_rmb), observed_content_hash, now_iso, now_iso, task_id),
            )
            self._refresh_run_state(conn, run_id, now=now_iso)
            done = conn.execute('SELECT * FROM profile_enrichment_tasks WHERE task_id=?', (task_id,)).fetchone()
            run = conn.execute('SELECT * FROM profile_enrichment_runs WHERE run_id=?', (run_id,)).fetchone()
            conn.commit()
        return {'ok': True, 'idempotent': False, 'task': self._task_payload(done), 'run_state': run['state'], 'raw_content_included': False}

    def fail(
        self,
        run_id: str,
        task_id: str,
        *,
        actor: str,
        session: str,
        lease_owner: str,
        claim_token: str,
        reason: str,
        terminal: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        reason = _bounded_identity(reason, 'terminal_reason')
        current = now or _utcnow()
        now_iso = _iso(current)
        state = 'unavailable' if terminal else 'retryable_failure'
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            _, task = self._claimed_task(conn, run_id, task_id, actor=actor, session=session, lease_owner=lease_owner, claim_token=claim_token, now=current)
            conn.execute(
                """UPDATE profile_enrichment_tasks SET state=?,terminal_reason=?,claim_token_hash=NULL,
                          delivery_token_hash=NULL,lease_owner_hash=NULL,lease_expires_at=NULL,heartbeat_at=NULL,
                          estimated_cost_rmb=0,updated_at=?,completed_at=? WHERE task_id=?""",
                (state, reason, now_iso, now_iso if terminal else None, task_id),
            )
            self._refresh_run_state(conn, run_id, now=now_iso)
            conn.commit()
        return {'ok': True, 'task_id': task_id, 'state': state, 'raw_content_included': False}

    def resume_budget(
        self,
        run_id: str,
        *,
        actor: str,
        session: str,
        additional_items: int,
        additional_cost_rmb: float = 0.0,
    ) -> dict[str, Any]:
        if type(additional_items) is not int or not 1 <= additional_items <= MAX_MANIFEST_ITEMS:
            raise ProfileEnrichmentError('additional_items is out of range', code='invalid_enrichment_budget')
        if not _valid_cost(additional_cost_rmb):
            raise ProfileEnrichmentError('additional_cost_rmb is out of range', code='invalid_enrichment_budget')
        now = _iso(_utcnow())
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            run = self._owned_run(conn, run_id, actor=actor, session=session)
            rows = list(conn.execute("SELECT * FROM profile_enrichment_tasks WHERE run_id=? AND state='paused_budget' ORDER BY created_at,task_id LIMIT ?", (run_id, additional_items)))
            remaining_cost = float(additional_cost_rmb)
            released = 0
            for task in rows:
                estimate = float(task['estimated_cost_rmb'] or 0)
                if estimate > 0 and remaining_cost < estimate:
                    continue
                if estimate > 0:
                    remaining_cost -= estimate
                state = 'awaiting_approval' if task['approval_required'] else 'pending'
                conn.execute('UPDATE profile_enrichment_tasks SET state=?,terminal_reason=NULL,updated_at=? WHERE task_id=?', (state, now, task['task_id']))
                released += 1
            conn.execute(
                """UPDATE profile_enrichment_runs SET item_budget=item_budget+?,cost_budget_rmb=cost_budget_rmb+?,
                          deferred_count=MAX(deferred_count-?,0),updated_at=? WHERE run_id=?""",
                (additional_items, float(additional_cost_rmb), released, now, run_id),
            )
            self._refresh_run_state(conn, run_id, now=now)
            conn.commit()
        return self.manifest(run_id, actor=actor, session=session)

    def revoke(self, run_id: str, *, actor: str, session: str) -> dict[str, Any]:
        now = _iso(_utcnow())
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._owned_run(conn, run_id, actor=actor, session=session)
            conn.execute("UPDATE profile_enrichment_runs SET state='cancelled',revoked_at=?,updated_at=? WHERE run_id=?", (now, now, run_id))
            conn.execute(
                """UPDATE profile_enrichment_tasks SET state='cancelled',terminal_reason='run_revoked',
                          claim_token_hash=NULL,delivery_token_hash=NULL,lease_owner_hash=NULL,lease_expires_at=NULL,
                          estimated_cost_rmb=0,updated_at=?,completed_at=? WHERE run_id=? AND state NOT IN ('completed','unavailable','cancelled')""",
                (now, now, run_id),
            )
            conn.commit()
        return {'ok': True, 'run_id': run_id, 'state': 'cancelled', 'raw_content_included': False}

    @staticmethod
    def _refresh_run_state(conn: Any, run_id: str, *, now: str) -> str:
        counts = {str(row['state']): int(row['n']) for row in conn.execute(
            'SELECT state,COUNT(*) AS n FROM profile_enrichment_tasks WHERE run_id=? GROUP BY state', (run_id,),
        )}
        if counts.get('paused_budget'):
            state = 'paused_budget'
        elif counts.get('awaiting_approval'):
            state = 'awaiting_approval'
        elif counts.get('awaiting_agent'):
            state = 'awaiting_agent'
        elif any(counts.get(value) for value in ('pending', 'materializing', 'processing', 'retryable_failure')):
            state = 'running' if any(counts.get(value) for value in ('materializing', 'processing')) else 'pending'
        elif counts.get('unavailable'):
            state = 'complete_with_terminal_gaps'
        else:
            state = 'complete'
        completed_at = now if state in {'complete', 'complete_with_terminal_gaps'} else None
        actual = float(conn.execute('SELECT COALESCE(SUM(actual_cost_rmb),0) FROM profile_enrichment_tasks WHERE run_id=?', (run_id,)).fetchone()[0])
        conn.execute('UPDATE profile_enrichment_runs SET state=?,actual_cost_rmb=?,updated_at=?,completed_at=? WHERE run_id=?', (state, actual, now, completed_at, run_id))
        return state
