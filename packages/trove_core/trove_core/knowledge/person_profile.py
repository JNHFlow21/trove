from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable

from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.store.repositories import MultimodalRepository, ObservationRecord
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.domain.content import display_content_for_kind
from trove_core.wechat.decrypt.manifest import load_account_identity
from trove_core.knowledge.logical_evidence import (
    deduplicate_logical_rows,
    logical_message_key,
    logical_moment_interaction_key,
    logical_moment_key,
)


PERSON_PROFILE_SCHEMA = 'person-profile/v1'
MAX_SCOPED_MESSAGES = 250_000
SESSION_GAP_SECONDS = 8 * 60 * 60
MAX_CLAIMS_PER_WRITE = 50
MAX_CITATIONS_PER_CLAIM = 8
MAX_GROUP_CONTEXTS = 100
CLOUD_ASR_PROVIDER_NAME = 'volcengine-asr-flash'
CLOUD_ASR_MODEL_ID = 'bigmodel:volc.bigasr.auc_turbo'

PROFILE_DIMENSIONS = (
    'life_context',
    'narrative_identity',
    'personality_tendencies',
    'values_and_priorities',
    'goals_and_motivations',
    'psychological_needs',
    'situational_patterns',
    'emotion_and_stress',
    'interpersonal_style',
    'communication_style',
    'decision_style',
    'strengths_and_resources',
    'tensions_and_growth',
    'attachment_related_patterns',
    'relationship_stage',
    'trust_and_closeness',
    'reciprocity_and_influence',
    'conflict_and_repair',
    'boundaries',
    'relationship_actions',
)
EVIDENCE_CLASSES = {'fact', 'pattern', 'hypothesis', 'unknown', 'action'}

EVIDENCE_CATEGORIES: dict[str, tuple[str, ...]] = {
    'identity_and_life_context': (
        '生日', '家乡', '老家', '从小', '家里', '妈妈', '爸爸', '父母', '学校', '大学', '毕业', '工作', '公司', '住在',
    ),
    'narrative_identity': (
        '以前的我', '现在的我', '我一直', '我从小', '后来我', '经历过', '改变了我', '我觉得自己', '我想成为',
    ),
    'goals_and_needs': (
        '希望', '想要', '打算', '准备', '目标', '需要', '计划', '最近想', '下一步',
    ),
    'values_and_tradeoffs': (
        '重要', '更看重', '不能接受', '宁愿', '值得', '应该', '原则', '底线', '最在意',
    ),
    'emotion_and_stress': (
        '压力', '焦虑', '难受', '伤心', '生气', '害怕', '担心', '开心', '委屈', '失望', '崩溃',
    ),
    'interpersonal_and_relationships': (
        '朋友', '关系', '在乎', '喜欢', '信任', '陪伴', '理解', '见面', '联系', '介绍',
    ),
    'boundaries_and_conflict': (
        '不要', '不喜欢', '介意', '矛盾', '吵架', '冲突', '边界', '别再', '不想说', '不能这样',
    ),
    'gratitude_and_help': (
        '谢谢', '感谢', '帮我', '帮你', '麻烦你', '介绍给', '推荐给', '支持我', '照顾我',
    ),
    'commitments_and_open_loops': (
        '答应', '回头', '到时候', '下次', '明天', '周末', '发给你', '给你发', '记得', '别忘了', '约好',
    ),
    'interests_and_preferences': (
        '喜欢', '爱好', '想去', '想吃', '最近在看', '最近在听', '不喜欢', '最喜欢',
    ),
}


SCIENTIFIC_LENSES: dict[str, dict[str, Any]] = {
    'realistic_accuracy_model': {
        'purpose': 'Require relevant cues to be available, detected, and correctly used before forming a judgment.',
        'output': 'cited_claim_with_counterevidence_and_alternatives',
        'reference': {'title': 'On the accuracy of personality judgment: A realistic approach', 'doi': '10.1037/0033-295X.102.4.652'},
    },
    'big_five': {
        'purpose': 'Describe observable tendencies across extraversion, agreeableness, conscientiousness, negative emotionality, and open-mindedness.',
        'limit': 'Chat evidence is an observer inference, not a substitute for a validated self-report inventory.',
        'reference': {'title': 'The next Big Five Inventory (BFI-2)', 'doi': '10.1037/pspp0000096'},
    },
    'schwartz_values': {
        'purpose': 'Infer tentative value priorities from repeated choices, costs, and tradeoffs rather than slogans.',
        'reference': {'title': 'Refining the theory of basic individual values', 'doi': '10.1037/a0029393'},
    },
    'self_determination_theory': {
        'purpose': 'Track autonomy, competence, and relatedness needs and the contexts that support or thwart them.',
        'reference': {'title': 'Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being', 'doi': '10.1037/0003-066X.55.1.68'},
    },
    'caps_if_then_patterns': {
        'purpose': 'Represent context-bound if-then behavior patterns instead of global labels.',
        'reference': {'title': 'A cognitive-affective system theory of personality', 'doi': '10.1037/0033-295X.102.2.246'},
    },
    'interpersonal_circumplex': {
        'purpose': 'Describe interpersonal agency and communion across private, group, and public contexts.',
        'reference': {'title': 'Exploring Personality with the Interpersonal Circumplex', 'doi': '10.1111/j.1751-9004.2009.00172.x'},
    },
    'narrative_identity': {
        'purpose': 'Extract repeated self-stories, turning points, identity roles, and imagined futures.',
        'reference': {'title': 'Narrative Identity', 'doi': '10.1177/0963721413475622'},
    },
    'attachment_related_dimensions': {
        'purpose': 'Record closeness-seeking and distancing behavior only as relationship-specific hypotheses.',
        'limit': 'Never assign an attachment type without direct self-report or explicit confirmation.',
        'reference': {'title': 'An item response theory analysis of self-report measures of adult attachment', 'doi': '10.1037/0022-3514.78.2.350'},
    },
}


VIDEO_DERIVED_RELATIONSHIP_PRINCIPLES = (
    'remember_personal_details_and_key_dates',
    'follow_advice_with_action_result_feedback',
    'appear_at_key_moments_without_forcing_frequency',
    'notice_and_explicitly_acknowledge_small_kindnesses',
    'offer_relevant_value_before_requesting_help',
    'respect_introduction_chains_and_relationship_boundaries',
    'match_gifts_to_real_preferences_relationship_stage_and_boundaries',
    'alternate_outward_social_exposure_with_inward_reflection',
)


class PersonProfileClaimError(ValueError):
    code = 'invalid_person_profile_claim'


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _load_json(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or '{}')
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).astimezone(timezone.utc)
    except ValueError:
        return None


def _stable_id(prefix: str, value: Any) -> str:
    return f'{prefix}-{hashlib.sha256(_json(value).encode("utf-8")).hexdigest()[:20]}'


def _resolved_scope(store: SQLiteStore, person: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolution = resolve_customer(store, person)
    resolved = resolution.get('resolved') or {}
    if not resolved:
        return resolution, {}
    if str(resolved.get('entity_id') or '').startswith('unresolved:'):
        return resolution, {}
    return resolution, resolved


def _unique_strings(values: Iterable[Any], *, limit: int = 100) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value or '').strip()))[:limit]


def _scope_json(citations: Iterable[str]) -> str:
    """Encode a citation scope once for SQLite's in-memory ``json_each`` table.

    The previous implementation split the same scope into hundreds of batches
    and repeated several joins over the complete media inventory for every
    batch.  A one-parameter JSON table preserves the exact citation set while
    making each evidence family scan at most once.
    """

    return json.dumps(
        list(dict.fromkeys(str(citation) for citation in citations if str(citation or '').strip())),
        ensure_ascii=False,
        separators=(',', ':'),
    )


def _scope_parts(resolved: dict[str, Any]) -> tuple[list[str], list[str]]:
    conversation_ids = _unique_strings(resolved.get('conversation_ids') or [], limit=100)
    sender_ids = _unique_strings([
        *(resolved.get('sender_ids') or []), resolved.get('primary_user_id'),
    ], limit=100)
    return conversation_ids, sender_ids


def _operator_wechat_ids(store: SQLiteStore) -> set[str]:
    """Load private own-account ids only for evidence attribution.

    The ids never enter the profile payload.  If the store is not inside the
    standard Vault layout, attribution stays unknown rather than guessing.
    """

    sqlite_path = Path(store.path).expanduser()
    if sqlite_path.parent.name != 'index':
        return set()
    current = sqlite_path.parent.parent / 'sources' / 'wechat-integrated-decrypted' / 'current'
    try:
        account_dirs = [child for child in current.iterdir() if child.is_dir() and not child.is_symlink()]
    except OSError:
        return set()
    return {
        own_wxid
        for account_dir in account_dirs
        if (own_wxid := str(load_account_identity(account_dir).get('own_wxid') or ''))
    }


def _scoped_messages(store: SQLiteStore, resolved: dict[str, Any]) -> tuple[list[Any], bool]:
    conversation_ids, sender_ids = _scope_parts(resolved)
    selects: list[str] = []
    params: list[Any] = []
    columns = (
        'citation,conversation_id,conversation_title,conversation_type,sender_id,sender_name,'
        'timestamp,content,content_kind,sent_by_me,source_type,direction,shard_id,local_id'
    )
    if conversation_ids:
        marks = ','.join('?' for _ in conversation_ids)
        selects.append(
            f"SELECT {columns} FROM messages "
            f"WHERE conversation_type='private' AND conversation_id IN ({marks})"
        )
        params.extend(conversation_ids)
    if sender_ids:
        marks = ','.join('?' for _ in sender_ids)
        selects.append(
            f"SELECT {columns} FROM messages "
            f"WHERE conversation_type='group' AND sender_id IN ({marks})"
        )
        params.extend(sender_ids)
    if not selects:
        return [], False
    with store.connect() as conn:
        rows = list(conn.execute(
            f"SELECT * FROM ({' UNION ALL '.join(selects)}) ORDER BY timestamp,citation LIMIT ?",
            (*params, MAX_SCOPED_MESSAGES + 1),
        ))
    rows = deduplicate_logical_rows(rows, key=logical_message_key)
    capped = len(rows) > MAX_SCOPED_MESSAGES
    return rows[:MAX_SCOPED_MESSAGES], capped


def _message_direction(row: Any) -> str:
    value = str(row['direction'] or '')
    if value == 'outgoing' or bool(row['sent_by_me']):
        return 'self'
    if value == 'incoming':
        return 'peer'
    return 'unknown'


def _spread_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[-1]]
    indices = [round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)]
    return [rows[index] for index in dict.fromkeys(indices)]


def _message_evidence(rows: list[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows:
        content = display_content_for_kind(row['content'], row['content_kind'])
        compact = str(content or '').strip()
        if not compact or compact.startswith('[') and compact.endswith(']'):
            continue
        direction = _message_direction(row)
        evidence.append({
            '_content': compact,
            'summary': compact[:240],
            'citations': [str(row['citation'])],
            'timestamp': row['timestamp'],
            'direction': direction,
            'conversation_type': row['conversation_type'],
            'source_type': 'message',
            'subject_scope': 'person' if direction == 'peer' else 'relationship_or_self',
            'evidence_class': 'observed_message',
            'requires_human_interpretation': True,
        })
    return evidence


def _payload_display(value: str, normalized_json: str) -> str:
    text = str(value or '').strip()
    if text:
        return text
    payload = _load_json(normalized_json)
    parts = [payload.get(key) for key in ('title', 'description', 'label', 'address', 'url')]
    return ' | '.join(str(part).strip() for part in parts if str(part or '').strip())[:500]


def _derived_evidence(
    store: SQLiteStore,
    resolved: dict[str, Any],
    rows: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect bounded, cited representations of every currently understood source.

    The returned records never contain local paths or provider payloads.  Message
    metadata is used only to preserve direction and context so an outgoing
    statement cannot accidentally be attributed to the person being profiled.
    """

    message_by_citation = {str(row['citation']): row for row in rows}
    citations = list(message_by_citation)
    operator_ids = _operator_wechat_ids(store)
    evidence: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    def add(
        *,
        content: str,
        citation: str,
        source_type: str,
        timestamp: str | None = None,
        message_citation: str | None = None,
        direction: str | None = None,
        conversation_type: str | None = None,
        confidence: float | None = None,
        subject_scope: str | None = None,
    ) -> None:
        compact = str(content or '').strip()
        citation = str(citation or '').strip()
        if not compact or not citation:
            return
        key = (source_type, citation)
        if key in seen:
            return
        seen.add(key)
        base_row = message_by_citation.get(str(message_citation or ''))
        resolved_direction = direction or (_message_direction(base_row) if base_row is not None else 'unknown')
        resolved_context = conversation_type or (str(base_row['conversation_type']) if base_row is not None else 'unknown')
        evidence.append({
            '_content': compact,
            'summary': compact[:240],
            'citations': [citation],
            'timestamp': timestamp or (base_row['timestamp'] if base_row is not None else None),
            'direction': resolved_direction,
            'conversation_type': resolved_context,
            'source_type': source_type,
            'subject_scope': subject_scope or ('person' if resolved_direction == 'peer' else 'relationship_or_self'),
            'evidence_class': 'derived_observation' if source_type in {'image_observation', 'media_understanding'} else 'observed_source',
            'confidence': confidence,
            'requires_human_interpretation': True,
        })
        counts[source_type] += 1

    with store.connect() as conn:
        if citations:
            scope_json = _scope_json(citations)
            scoped_cte = "WITH scoped(citation) AS (SELECT CAST(value AS TEXT) FROM json_each(?))"
            transcript_rows = conn.execute(
                scoped_cte +
                """ SELECT DISTINCT t.transcript_id,t.citation,t.text,t.confidence,
                                      COALESCE(mal.source_citation,ma.citation) AS message_citation
                        FROM transcripts t
                        JOIN media_assets ma ON ma.asset_id=t.asset_id
                        JOIN provider_jobs pj ON pj.job_id=t.job_id
                   LEFT JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                       WHERE t.status='active'
                         AND pj.provider=? AND pj.model=? AND pj.status='completed'
                         AND pj.request_hash=ma.content_hash AND (
                             t.citation IN (SELECT citation FROM scoped)
                          OR ma.citation IN (SELECT citation FROM scoped)
                          OR mal.source_citation IN (SELECT citation FROM scoped)
                       )""",
                (scope_json, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
            )
            for item in transcript_rows:
                add(
                    content=item['text'], citation=item['citation'], source_type='transcript',
                    message_citation=item['message_citation'], confidence=float(item['confidence'] or 0),
                )

            observation_rows = conn.execute(
                scoped_cte +
                """ SELECT DISTINCT io.observation_id,io.citation,io.caption,io.visible_text,io.confidence,
                                      COALESCE(mal.source_citation,ma.citation) AS message_citation
                        FROM image_observations io
                        JOIN media_assets ma ON ma.asset_id=io.asset_id
                   LEFT JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                       WHERE io.status IN ('active','needs_review','proposed') AND (
                             io.citation IN (SELECT citation FROM scoped)
                          OR ma.citation IN (SELECT citation FROM scoped)
                          OR mal.source_citation IN (SELECT citation FROM scoped)
                       )""",
                (scope_json,),
            )
            for item in observation_rows:
                content = ' | '.join(part for part in (str(item['caption'] or '').strip(), str(item['visible_text'] or '').strip()) if part)
                add(
                    content=content, citation=item['citation'], source_type='image_observation',
                    message_citation=item['message_citation'], confidence=float(item['confidence'] or 0),
                )

            understanding_rows = conn.execute(
                scoped_cte +
                """ SELECT DISTINCT mu.content_sha256,mu.modality,mu.caption,mu.visible_text,mu.audio_transcript,
                                      mu.confidence,COALESCE(mal.source_citation,ma.citation) AS message_citation
                        FROM media_understanding mu
                        JOIN media_assets ma ON ma.content_hash=mu.content_sha256
                   LEFT JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                       WHERE mu.status='active' AND (
                             ma.citation IN (SELECT citation FROM scoped)
                          OR mal.source_citation IN (SELECT citation FROM scoped)
                       )""",
                (scope_json,),
            )
            for item in understanding_rows:
                content = ' | '.join(part for part in (
                    str(item['caption'] or '').strip(),
                    str(item['visible_text'] or '').strip(),
                    str(item['audio_transcript'] or '').strip(),
                ) if part)
                message_citation = str(item['message_citation'] or '')
                add(
                    content=content,
                    citation=f"{message_citation}#understanding-{str(item['content_sha256'])[:12]}",
                    source_type='media_understanding', message_citation=message_citation,
                    confidence=float(item['confidence'] or 0),
                )

            payload_rows = conn.execute(
                scoped_cte +
                """ SELECT mp.citation,mp.display_text,mp.normalized_json,m.timestamp,m.direction,
                             m.sent_by_me,m.conversation_type
                        FROM scoped s
                        JOIN message_payloads mp ON mp.citation=s.citation
                        JOIN messages m ON m.citation=mp.citation
                       WHERE mp.parse_status='parsed'""",
                (scope_json,),
            )
            for item in payload_rows:
                raw_direction = str(item['direction'] or '')
                if raw_direction == 'outgoing' or bool(item['sent_by_me']):
                    direction = 'self'
                elif raw_direction == 'incoming':
                    direction = 'peer'
                else:
                    direction = 'unknown'
                add(
                    content=_payload_display(item['display_text'], item['normalized_json']),
                    citation=item['citation'], source_type='appmsg_payload', timestamp=item['timestamp'],
                    message_citation=item['citation'], direction=direction,
                    conversation_type=item['conversation_type'],
                )

        entity_ids = _unique_strings([resolved.get('entity_id'), *(resolved.get('entity_ids') or [])])
        if entity_ids:
            marks = ','.join('?' for _ in entity_ids)
            observation_rows = conn.execute(
                f"""SELECT observation_type,value_json,citation,confidence,status,updated_at
                       FROM observations
                      WHERE entity_id IN ({marks}) AND observation_type<>'person_profile_claim'
                        AND status IN ('active','needs_review','merge_candidate')
                      ORDER BY confidence DESC,updated_at DESC""",
                entity_ids,
            )
            for item in observation_rows:
                value = _load_json(item['value_json'])
                content = value.get('text') or value.get('value') or value.get('display_name')
                if content is None:
                    content = '; '.join(
                        f'{key}: {val}' for key, val in value.items()
                        if key not in {'path', 'path_ref', 'raw', 'payload'} and isinstance(val, (str, int, float, bool))
                    )
                add(
                    content=f"{item['observation_type']}: {content}", citation=item['citation'],
                    source_type='entity_observation', timestamp=item['updated_at'], direction='unknown',
                    conversation_type='profile', confidence=float(item['confidence'] or 0), subject_scope='person',
                )

        user_id = str(resolved.get('primary_user_id') or '').strip()
        if user_id:
            moment_rows = conn.execute(
                """SELECT citation,author_id,text,timestamp FROM moment_items
                     WHERE author_id=? AND status='active' ORDER BY timestamp,citation""",
                (user_id,),
            )
            for item in deduplicate_logical_rows(moment_rows, key=logical_moment_key):
                add(
                    content=item['text'], citation=item['citation'], source_type='moment',
                    timestamp=item['timestamp'], direction='peer', conversation_type='public', subject_scope='person',
                )
            interaction_rows = conn.execute(
                """SELECT mi.citation,mi.interaction_type,mi.text,mi.timestamp,mi.actor_id,m.author_id
                       FROM moment_interactions mi JOIN moment_items m ON m.moment_id=mi.moment_id
                      WHERE mi.status='active' AND (mi.actor_id=? OR m.author_id=?)
                      ORDER BY mi.timestamp,mi.citation""",
                (user_id, user_id),
            )
            for item in deduplicate_logical_rows(
                interaction_rows,
                key=logical_moment_interaction_key,
            ):
                actor_id = str(item['actor_id'] or '')
                by_person = actor_id == user_id
                by_operator = actor_id in operator_ids
                if not by_person and not by_operator:
                    # A third party commenting on the person's Moment is public
                    # context, not evidence authored by either party in this
                    # relationship. Never relabel it as the operator.
                    continue
                content = f"{item['interaction_type']}: {item['text']}".strip(': ')
                add(
                    content=content, citation=item['citation'], source_type='moment_interaction',
                    timestamp=item['timestamp'], direction='peer' if by_person else 'self',
                    conversation_type='public_interaction',
                    subject_scope='person' if by_person else 'relationship_or_self',
                )
            moment_visual_rows = conn.execute(
                """SELECT DISTINCT io.citation,io.caption,io.visible_text,io.confidence,
                                   mal.source_citation,mi.timestamp,mi.author_id
                       FROM image_observations io
                       JOIN media_assets ma ON ma.asset_id=io.asset_id
                       JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                       JOIN moment_items mi ON mi.citation=CASE
                            WHEN instr(mal.source_citation,'#media-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#media-')-1)
                            WHEN instr(mal.source_citation,'#image-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#image-')-1)
                            WHEN instr(mal.source_citation,'#video-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#video-')-1)
                            ELSE mal.source_citation END
                      WHERE io.status IN ('active','needs_review','proposed') AND (
                            mi.author_id=? OR EXISTS (
                                SELECT 1 FROM moment_interactions mint
                                 WHERE mint.moment_id=mi.moment_id AND mint.actor_id=? AND mint.status='active'
                            )
                      )""",
                (user_id, user_id),
            )
            for item in moment_visual_rows:
                content = ' | '.join(part for part in (
                    str(item['caption'] or '').strip(), str(item['visible_text'] or '').strip(),
                ) if part)
                authored = str(item['author_id'] or '') == user_id
                add(
                    content=content, citation=item['citation'], source_type='image_observation',
                    timestamp=item['timestamp'], direction='peer' if authored else 'unknown',
                    conversation_type='public', confidence=float(item['confidence'] or 0),
                    subject_scope='person' if authored else 'relationship',
                )
            moment_understanding_rows = conn.execute(
                """SELECT DISTINCT mu.content_sha256,mu.caption,mu.visible_text,mu.audio_transcript,mu.confidence,
                                   mal.source_citation,mi.timestamp,mi.author_id
                       FROM media_understanding mu
                       JOIN media_assets ma ON ma.content_hash=mu.content_sha256
                       JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                       JOIN moment_items mi ON mi.citation=CASE
                            WHEN instr(mal.source_citation,'#media-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#media-')-1)
                            WHEN instr(mal.source_citation,'#image-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#image-')-1)
                            WHEN instr(mal.source_citation,'#video-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#video-')-1)
                            ELSE mal.source_citation END
                      WHERE mu.status='active' AND (
                            mi.author_id=? OR EXISTS (
                                SELECT 1 FROM moment_interactions mint
                                 WHERE mint.moment_id=mi.moment_id AND mint.actor_id=? AND mint.status='active'
                            )
                      )""",
                (user_id, user_id),
            )
            for item in moment_understanding_rows:
                content = ' | '.join(part for part in (
                    str(item['caption'] or '').strip(), str(item['visible_text'] or '').strip(),
                    str(item['audio_transcript'] or '').strip(),
                ) if part)
                authored = str(item['author_id'] or '') == user_id
                source_citation = str(item['source_citation'] or '')
                add(
                    content=content,
                    citation=f"{source_citation}#understanding-{str(item['content_sha256'])[:12]}",
                    source_type='media_understanding', timestamp=item['timestamp'],
                    direction='peer' if authored else 'unknown', conversation_type='public',
                    confidence=float(item['confidence'] or 0),
                    subject_scope='person' if authored else 'relationship',
                )
    return evidence, dict(sorted(counts.items()))


def _evidence_candidates(items: list[dict[str, Any]], *, limit: int) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in EVIDENCE_CATEGORIES}
    seen: dict[str, set[str]] = defaultdict(set)
    for item in items:
        compact = str(item.get('_content') or item.get('summary') or '').strip()
        for category, terms in EVIDENCE_CATEGORIES.items():
            if not any(term in compact for term in terms):
                continue
            citation = str((item.get('citations') or [''])[0])
            if citation in seen[category]:
                continue
            seen[category].add(citation)
            candidates[category].append({key: value for key, value in item.items() if not key.startswith('_')})
    return {name: _spread_sample(items, limit) for name, items in candidates.items()}


def _interaction_dynamics(rows: list[Any]) -> dict[str, Any]:
    private = [row for row in rows if str(row['conversation_type']) == 'private']
    sessions = 0
    self_started = 0
    peer_started = 0
    last_time: datetime | None = None
    last_direction: str | None = None
    self_response: list[float] = []
    peer_response: list[float] = []
    for row in private:
        timestamp = _parse_time(row['timestamp'])
        direction = _message_direction(row)
        if timestamp is None:
            continue
        new_session = last_time is None or (timestamp - last_time).total_seconds() >= SESSION_GAP_SECONDS
        if new_session:
            sessions += 1
            if direction == 'self':
                self_started += 1
            elif direction == 'peer':
                peer_started += 1
        elif direction in {'self', 'peer'} and last_direction in {'self', 'peer'} and last_direction != direction:
            delay_minutes = max((timestamp - last_time).total_seconds() / 60, 0.0)
            if direction == 'self':
                self_response.append(delay_minutes)
            elif direction == 'peer':
                peer_response.append(delay_minutes)
        last_time = timestamp
        # Unknown authorship breaks a response chain. It must not be used as
        # either side of a response-time estimate.
        last_direction = direction if direction in {'self', 'peer'} else None
    return {
        'sessions': sessions,
        'self_started_sessions': self_started,
        'peer_started_sessions': peer_started,
        'self_response_median_minutes': round(median(self_response), 2) if self_response else None,
        'peer_response_median_minutes': round(median(peer_response), 2) if peer_response else None,
        'interpretation_limit': 'Cadence is descriptive and must not be treated as a direct measure of care, attachment, or intent.',
    }


def _timeline(rows: list[Any]) -> list[dict[str, Any]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        timestamp = str(row['timestamp'] or '')
        month = timestamp[:7] if len(timestamp) >= 7 else 'unknown'
        buckets[month]['messages'] += 1
        buckets[month][_message_direction(row)] += 1
        buckets[month][str(row['conversation_type'] or 'unknown')] += 1
    return [
        {
            'month': month,
            'messages': counts['messages'],
            'self': counts['self'],
            'peer': counts['peer'],
            'unknown_direction': counts['unknown'],
            'private': counts['private'],
            'group': counts['group'],
        }
        for month, counts in sorted(buckets.items())
    ][-60:]


def _group_contexts(rows: list[Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row['conversation_type']) != 'group' or _message_direction(row) != 'peer':
            continue
        conversation_id = str(row['conversation_id'])
        item = grouped.setdefault(conversation_id, {
            'conversation_id': conversation_id,
            'conversation_title': str(row['conversation_title'] or ''),
            'messages': 0,
            'first_at': None,
            'last_at': None,
            'content_kinds': Counter(),
        })
        timestamp = str(row['timestamp'] or '') or None
        item['messages'] += 1
        item['first_at'] = min(value for value in (item['first_at'], timestamp) if value is not None)
        item['last_at'] = max(value for value in (item['last_at'], timestamp) if value is not None)
        item['content_kinds'][str(row['content_kind'] or 'text')] += 1
    ordered = sorted(grouped.values(), key=lambda item: (-int(item['messages']), str(item['conversation_id'])))
    return {
        'total_groups_spoken_in': len(ordered),
        'returned_groups': min(len(ordered), MAX_GROUP_CONTEXTS),
        'truncated': len(ordered) > MAX_GROUP_CONTEXTS,
        'groups': [
            {**item, 'content_kinds': dict(sorted(item['content_kinds'].items()))}
            for item in ordered[:MAX_GROUP_CONTEXTS]
        ],
        'interpretation_limit': 'Group membership is inferred from observed speech, not a complete membership roster.',
    }


def _context_style(rows: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for context, selected in {
        'private_peer': [row for row in rows if str(row['conversation_type']) == 'private' and _message_direction(row) == 'peer'],
        'group_peer': [row for row in rows if str(row['conversation_type']) == 'group' and _message_direction(row) == 'peer'],
        'private_self': [row for row in rows if str(row['conversation_type']) == 'private' and _message_direction(row) == 'self'],
    }.items():
        texts = [
            str(display_content_for_kind(row['content'], row['content_kind']) or '').strip()
            for row in selected
        ]
        texts = [text for text in texts if text and not (text.startswith('[') and text.endswith(']'))]
        lengths = [len(text) for text in texts]
        result[context] = {
            'messages': len(selected),
            'text_messages': len(texts),
            'median_text_characters': round(float(median(lengths)), 2) if lengths else None,
            'question_message_ratio': round(sum('?' in text or '？' in text for text in texts) / len(texts), 4) if texts else None,
        }
    result['interpretation_limit'] = 'These are descriptive context differences, not stable personality traits.'
    return result


def _activity_trend(rows: list[Any]) -> dict[str, Any]:
    timed = [(row, _parse_time(row['timestamp'])) for row in rows if _parse_time(row['timestamp']) is not None]
    if not timed:
        return {'latest_at': None, 'recent_90_days': 0, 'previous_90_days': 0, 'label': 'unknown'}
    latest = max(timestamp for _, timestamp in timed if timestamp is not None)
    recent_start = latest - timedelta(days=90)
    previous_start = latest - timedelta(days=180)
    recent = sum(timestamp >= recent_start for _, timestamp in timed if timestamp is not None)
    previous = sum(previous_start <= timestamp < recent_start for _, timestamp in timed if timestamp is not None)
    if recent == previous:
        label = 'stable_message_activity'
    elif recent > previous:
        label = 'higher_recent_message_activity'
    else:
        label = 'lower_recent_message_activity'
    return {
        'latest_at': latest.isoformat().replace('+00:00', 'Z'),
        'recent_90_days': recent,
        'previous_90_days': previous,
        'label': label,
        'interpretation_limit': 'Activity changes do not by themselves imply changes in closeness, care, or relationship quality.',
    }


def _key_events(evidence: dict[str, list[dict[str, Any]]], *, limit: int = 30) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in (
        'identity_and_life_context', 'goals_and_needs', 'values_and_tradeoffs', 'emotion_and_stress',
        'boundaries_and_conflict', 'gratitude_and_help', 'commitments_and_open_loops',
    ):
        for item in evidence.get(category) or []:
            citation = str((item.get('citations') or [''])[0])
            if not citation or citation in seen:
                continue
            seen.add(citation)
            events.append({
                'timestamp': item.get('timestamp'),
                'category': category,
                'summary': item.get('summary'),
                'citations': item.get('citations') or [],
                'direction': item.get('direction'),
                'source_type': item.get('source_type'),
            })
    events.sort(key=lambda item: (str(item.get('timestamp') or ''), str((item.get('citations') or [''])[0])))
    return _spread_sample(events, limit)


def _media_coverage(store: SQLiteStore, rows: list[Any]) -> dict[str, dict[str, int]]:
    citations = [str(row['citation']) for row in rows]
    kinds = Counter(str(row['content_kind'] or 'text') for row in rows)
    understood: dict[str, set[str]] = defaultdict(set)
    if citations:
        scope_json = _scope_json(citations)
        scoped_cte = "WITH scoped(citation) AS (SELECT CAST(value AS TEXT) FROM json_each(?))"
        with store.connect() as conn:
            voice_rows = conn.execute(
                scoped_cte +
                """, active_transcripts AS (
                         SELECT t.citation AS transcript_citation,ma.asset_id,ma.citation AS asset_citation
                           FROM transcripts t
                      LEFT JOIN media_assets ma ON ma.asset_id=t.asset_id
                           JOIN provider_jobs pj ON pj.job_id=t.job_id
                          WHERE t.status='active'
                            AND pj.provider=? AND pj.model=? AND pj.status='completed'
                            AND pj.request_hash=ma.content_hash
                     ), matched(citation) AS (
                         SELECT s.citation FROM active_transcripts t
                           JOIN scoped s ON s.citation=t.transcript_citation
                         UNION
                         SELECT s.citation FROM active_transcripts t
                           JOIN scoped s ON t.transcript_citation LIKE s.citation || '#%'
                         UNION
                         SELECT s.citation FROM active_transcripts t
                           JOIN scoped s ON s.citation=t.asset_citation
                         UNION
                         SELECT s.citation FROM active_transcripts t
                           JOIN media_asset_links mal ON mal.asset_id=t.asset_id AND mal.accepted=1
                           JOIN scoped s ON s.citation=mal.source_citation
                     ) SELECT citation FROM matched""",
                (scope_json, CLOUD_ASR_PROVIDER_NAME, CLOUD_ASR_MODEL_ID),
            )
            understood['voice'].update(str(item['citation']) for item in voice_rows)
            for modality in ('image', 'video'):
                visual_rows = conn.execute(
                    scoped_cte +
                    """, understood_assets AS (
                           SELECT DISTINCT ma.asset_id,ma.citation
                             FROM image_observations io
                             JOIN media_assets ma ON ma.asset_id=io.asset_id
                             JOIN media_understanding mu ON mu.content_sha256=ma.content_hash
                            WHERE ma.modality=?
                              AND io.status IN ('active','needs_review','proposed')
                              AND mu.status='active'
                       ), matched(citation) AS (
                           SELECT s.citation FROM understood_assets ua
                             JOIN scoped s ON s.citation=ua.citation
                           UNION
                           SELECT s.citation FROM understood_assets ua
                             JOIN media_asset_links mal ON mal.asset_id=ua.asset_id AND mal.accepted=1
                             JOIN scoped s ON s.citation=mal.source_citation
                       ) SELECT citation FROM matched""",
                    (scope_json, modality),
                )
                understood[modality].update(str(item['citation']) for item in visual_rows)
            appmsg_rows = conn.execute(
                scoped_cte +
                """ SELECT s.citation FROM scoped s
                       JOIN message_payloads mp ON mp.citation=s.citation
                      WHERE mp.parse_status='parsed'""",
                (scope_json,),
            )
            understood['appmsg'].update(str(item['citation']) for item in appmsg_rows)
    result: dict[str, dict[str, int]] = {}
    for modality in ('voice', 'image', 'video', 'appmsg'):
        total = int(kinds.get(modality, 0))
        done = min(len(understood.get(modality, set())), total)
        result[modality] = {'total': total, 'understood': done, 'gaps': max(total - done, 0)}
    return result


def _auxiliary_coverage(store: SQLiteStore, resolved: dict[str, Any]) -> dict[str, Any]:
    user_id = str(resolved.get('primary_user_id') or '')
    authored = interactions = identifiers = observations = relationships = 0
    person_authored_interactions = operator_person_interactions = excluded_public_interactions = 0
    operator_ids = _operator_wechat_ids(store)
    latest_snapshot: dict[str, Any] | None = None
    with store.connect() as conn:
        if user_id and store._table_exists(conn, 'moment_items'):
            authored_rows = conn.execute(
                """SELECT author_id,timestamp,text FROM moment_items
                     WHERE author_id=? AND status='active' ORDER BY timestamp,citation""",
                (user_id,),
            )
            authored = len(deduplicate_logical_rows(authored_rows, key=logical_moment_key))
        if user_id and store._table_exists(conn, 'moment_interactions'):
            interaction_rows = list(conn.execute(
                """SELECT mi.actor_id,m.author_id,mi.interaction_type,mi.timestamp FROM moment_interactions mi
                     LEFT JOIN moment_items m ON m.moment_id=mi.moment_id
                     WHERE mi.status='active' AND (mi.actor_id=? OR m.author_id=?)
                     ORDER BY mi.timestamp,mi.citation""",
                (user_id, user_id),
            ))
            interaction_rows = deduplicate_logical_rows(
                interaction_rows,
                key=logical_moment_interaction_key,
            )
            for item in interaction_rows:
                actor_id = str(item['actor_id'] or '')
                author_id = str(item['author_id'] or '')
                if actor_id == user_id:
                    person_authored_interactions += 1
                elif author_id == user_id and actor_id in operator_ids:
                    operator_person_interactions += 1
                else:
                    excluded_public_interactions += 1
            interactions = person_authored_interactions + operator_person_interactions
        entity_ids = _unique_strings([resolved.get('entity_id'), *(resolved.get('entity_ids') or [])])
        if entity_ids:
            marks = ','.join('?' for _ in entity_ids)
            identifiers = int(conn.execute(
                f'SELECT COUNT(*) FROM entity_identifiers WHERE entity_id IN ({marks})', entity_ids,
            ).fetchone()[0])
            observations = int(conn.execute(
                f"SELECT COUNT(*) FROM observations WHERE entity_id IN ({marks}) AND status IN ('active','needs_review','merge_candidate')",
                entity_ids,
            ).fetchone()[0])
            relationships = int(conn.execute(
                f"SELECT COUNT(*) FROM relationships WHERE subject_entity_id IN ({marks}) AND status='active'",
                entity_ids,
            ).fetchone()[0])
        if store._table_exists(conn, 'source_snapshots'):
            row = conn.execute(
                "SELECT snapshot_revision,state,created_at,updated_at FROM source_snapshots ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                latest_snapshot = dict(row)
    return {
        'moments_authored': authored,
        'moment_interactions': interactions,
        'person_authored_moment_interactions': person_authored_interactions,
        'operator_person_moment_interactions': operator_person_interactions,
        'unattributed_public_interactions_excluded': excluded_public_interactions,
        'entity_identifiers': identifiers,
        'entity_observations': observations,
        'entity_relationships': relationships,
        'latest_source_snapshot': latest_snapshot,
    }


def _profile_claims(store: SQLiteStore, resolved: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped = {dimension: [] for dimension in PROFILE_DIMENSIONS}
    entity_ids = _unique_strings([resolved.get('entity_id'), *(resolved.get('entity_ids') or [])])
    if not entity_ids:
        return grouped
    marks = ','.join('?' for _ in entity_ids)
    with store.connect() as conn:
        rows = list(conn.execute(
            f"""SELECT * FROM observations
                  WHERE entity_id IN ({marks}) AND observation_type='person_profile_claim'
                    AND status IN ('active','needs_review','merge_candidate')
                  ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END,confidence DESC,updated_at DESC""",
            entity_ids,
        ))
    for row in rows:
        value = _load_json(row['value_json'])
        dimension = str(value.get('dimension') or '')
        if dimension not in grouped:
            continue
        grouped[dimension].append({
            **value,
            'observation_id': row['observation_id'],
            'review_status': row['status'],
            'confidence': float(row['confidence'] or value.get('confidence') or 0),
            'source_type': row['source_type'],
        })
    return grouped


def _relationship_actions(
    evidence: dict[str, list[dict[str, Any]]],
    media: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    def add(action_type: str, reason: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        actions.append({
            'action_type': action_type,
            'reason': reason,
            'citations': _unique_strings(
                (citation for row in rows[:3] for citation in (row.get('citations') or [])), limit=6,
            ),
            'requires_human_confirmation': True,
            'auto_send_allowed': False,
        })

    person_rows = lambda values: [row for row in values if row.get('subject_scope') == 'person']
    self_rows = lambda values: [row for row in values if row.get('direction') == 'self']
    add('remember_key_detail', 'Review personal details and key dates before deciding whether a reminder is appropriate.', person_rows(evidence['identity_and_life_context']))
    add('express_gratitude', 'Check whether help, introductions, or kindness have received an explicit and proportionate acknowledgment.', evidence['gratitude_and_help'])
    commitments = self_rows(evidence['commitments_and_open_loops']) or evidence['commitments_and_open_loops']
    add('close_commitment_loop', 'Verify whether promises, planned follow-ups, or shared materials have been completed.', commitments)
    add('offer_help', 'Consider a specific, non-intrusive offer that matches an explicitly stated goal or need.', person_rows(evidence['goals_and_needs']))
    add('respect_boundary', 'Review stated dislikes, conflicts, and boundaries before initiating an action.', person_rows(evidence['boundaries_and_conflict']))
    gaps = {modality: values['gaps'] for modality, values in media.items() if values['gaps'] > 0}
    if gaps:
        actions.append({
            'action_type': 'complete_evidence_gap',
            'reason': 'Complete locally available media understanding before treating the profile as comprehensive.',
            'gaps': gaps,
            'next_tool': 'trove_profile_enrichment_plan',
            'requires_human_confirmation': False,
            'auto_send_allowed': False,
        })
    return actions


def _questions_for_user(
    evidence: dict[str, list[dict[str, Any]]],
    claims: dict[str, list[dict[str, Any]]],
    media: dict[str, dict[str, int]],
) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []

    def add(question_id: str, question: str, reason: str) -> None:
        if len(questions) < 8:
            questions.append({'question_id': question_id, 'question': question, 'reason': reason})

    if not evidence.get('identity_and_life_context'):
        add('confirm_basics', '你是否愿意补充或确认这个人的生日、常驻城市、职业和重要家庭背景？', '当前证据没有覆盖完整的基础信息。')
    if not evidence.get('goals_and_needs'):
        add('confirm_goals', '你知道对方当前最重要的目标、压力或需要的支持是什么吗？', '目标与需要缺少直接自述证据。')
    if not evidence.get('values_and_tradeoffs'):
        add('confirm_values', '你观察到对方做重要选择时最看重什么、会牺牲什么吗？', '价值判断应来自真实取舍，不能仅靠聊天语气推断。')
    if not evidence.get('boundaries_and_conflict'):
        add('confirm_boundaries', '你是否知道对方明确的禁区、不喜欢的沟通方式或发生过的冲突修复？', '边界信息不足时不应主动制定高介入行动。')
    if not any(claims.get(dimension) for dimension in ('relationship_stage', 'trust_and_closeness')):
        add('relationship_goal', '你希望与对方维持或发展成怎样的关系？目前你主观上如何描述关系阶段？', '关系目标属于你的意图，不能从对方数据中替你决定。')
    gaps = ', '.join(f"{name} {values['gaps']} 条" for name, values in media.items() if values['gaps'])
    if gaps:
        add('complete_media', f'是否继续补全这些尚未理解的多媒体证据：{gaps}？', '在补全前，画像只能标记为存在证据缺口。')
    if any(value.get('review_status') != 'active' for values in claims.values() for value in values):
        add('review_claims', '是否逐条确认、修正或拒绝当前待审核的人物判断？', '所有深层推断默认待人工确认。')
    return questions


def _analysis_protocol() -> dict[str, Any]:
    return {
        'ordered_steps': [
            'separate statements made by the person from statements made by the operator',
            'extract directly confirmed facts and unknowns',
            'find repeated context-bound if-then patterns',
            'search for counterevidence and plausible alternative explanations',
            'apply scientific lenses without clinical diagnosis',
            'propose cited claims as needs_review',
            'ask the operator to confirm, correct, or supplement',
            'derive only proportionate relationship actions and never auto-send',
        ],
        'attribution_rule': 'Evidence with subject_scope=relationship_or_self must not be attributed to the profiled person.',
        'minimum_pattern_evidence': 3,
        'hypothesis_requires_alternative_explanation': True,
        'pattern_and_hypothesis_require_counterevidence_review': True,
        'next_write_tool': 'trove_person_profile_claims_propose',
        'next_review_tool': 'trove_observe_approve',
    }


def _completion_workflow() -> dict[str, Any]:
    return {
        'ordinary_read_mutates_or_uploads': False,
        'complete_request_is_explicit': True,
        'steps': [
            {
                'step': 'refresh_local_source_if_needed',
                'tools': ['trove decrypt status', 'trove decrypt preflight', 'trove decrypt run --yes', 'trove import-real --yes'],
                'rule': 'Use only local decryption inputs; never place key values in arguments, output, logs, or the repository.',
            },
            {
                'step': 'create_or_reuse_enrichment_manifest',
                'tool': 'trove_profile_enrichment_plan',
                'mode': 'complete',
                'execution_location': 'local',
                'purpose': 'person_relationship_profile_enrichment',
            },
            {
                'step': 'consume_manifest',
                'rule': 'Continue until complete or a declared approval, agent, budget, or terminal-gap pause.',
            },
            {'step': 'rebuild_profile_and_propose_reviewable_claims', 'tool': 'trove_person_profile'},
        ],
        'cloud_upload_default': 'disabled',
    }


def _empty_profile(person: str, resolution: dict[str, Any]) -> dict[str, Any]:
    return {
        'type': 'person_profile',
        'schema_version': PERSON_PROFILE_SCHEMA,
        'person': person,
        'resolved_entity': None,
        'candidates': resolution.get('candidates') or [],
        'status': 'ambiguous' if resolution.get('ambiguous') else 'not_found',
        'data_coverage': {},
        'evidence_candidates': {name: [] for name in EVIDENCE_CATEGORIES},
        'evidence_projection': {
            'full_scope_analyzed': False,
            'per_category_limit': None,
            'selection': 'none',
            'expand_by_increasing_evidence_limit_or_opening_citation_context': False,
        },
        'person_model': {dimension: [] for dimension in PROFILE_DIMENSIONS},
        'relationship_model': {},
        'relationship_actions': [],
        'questions_for_user': [{
            'question_id': 'identify_person',
            'question': '请补充一个可唯一定位此人的微信备注、昵称、微信号或会话标识。',
            'reason': '当前无法唯一解析人物身份。',
        }],
        'analysis_protocol': _analysis_protocol(),
        'completion_workflow': _completion_workflow(),
        'scientific_framework': {'lenses': SCIENTIFIC_LENSES},
        'clinical_diagnosis_included': False,
        'raw_chat_dump_included': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def build_person_profile(store: SQLiteStore, person: str, *, evidence_limit: int = 12) -> dict[str, Any]:
    if type(evidence_limit) is not int or not 1 <= evidence_limit <= 50:
        raise ValueError('evidence_limit must be from 1 to 50')
    if not store.path.is_file():
        return _empty_profile(person, {'candidates': [], 'ambiguous': False})
    store.initialize()
    resolution, resolved = _resolved_scope(store, person)
    if not resolved:
        return _empty_profile(person, resolution)
    messages, capped = _scoped_messages(store, resolved)
    counts = Counter()
    content_kinds = Counter()
    conversation_ids: set[str] = set()
    group_ids: set[str] = set()
    for row in messages:
        counts['total'] += 1
        counts[str(row['conversation_type'] or 'unknown')] += 1
        counts[_message_direction(row)] += 1
        content_kinds[str(row['content_kind'] or 'text')] += 1
        conversation_ids.add(str(row['conversation_id']))
        if row['conversation_type'] == 'group':
            group_ids.add(str(row['conversation_id']))
    timestamps = [str(row['timestamp']) for row in messages if row['timestamp']]
    media = _media_coverage(store, messages)
    auxiliary = _auxiliary_coverage(store, resolved)
    derived_evidence, derived_counts = _derived_evidence(store, resolved, messages)
    unified_evidence = _message_evidence(messages) + derived_evidence
    evidence = _evidence_candidates(unified_evidence, limit=evidence_limit)
    claims = _profile_claims(store, resolved)
    data_gaps = [
        {'kind': f'{modality}_understanding', 'count': values['gaps'], 'next_tool': 'trove_profile_enrichment_plan'}
        for modality, values in media.items() if values['gaps'] > 0
    ]
    if capped:
        data_gaps.append({
            'kind': 'message_analysis_truncated',
            'analysis_cap': MAX_SCOPED_MESSAGES,
            'analyzed': len(messages),
            'more_messages_exist': True,
            'reason': 'The scoped message count exceeded the bounded in-memory analysis cap.',
        })
    unreviewed_claims = sum(
        1 for values in claims.values() for value in values if value.get('review_status') != 'active'
    )
    if unreviewed_claims:
        data_gaps.append({'kind': 'profile_claims_need_review', 'count': unreviewed_claims, 'next_tool': 'trove_observe_approve'})
    person_model = {
        dimension: values for dimension, values in claims.items()
    }
    return {
        'type': 'person_profile',
        'schema_version': PERSON_PROFILE_SCHEMA,
        'person': person,
        'resolved_entity': resolved,
        'candidates': resolution.get('candidates') or [],
        'status': 'ready_with_gaps' if data_gaps else 'ready',
        'data_coverage': {
            'messages': {
                'total': counts['total'],
                'analyzed': len(messages),
                'scope_complete': not capped,
                'private': counts['private'],
                'group': counts['group'],
                'self': counts['self'],
                'peer': counts['peer'],
                'unknown_direction': counts['unknown'],
                'first_at': min(timestamps) if timestamps else None,
                'last_at': max(timestamps) if timestamps else None,
                'content_kinds': dict(sorted(content_kinds.items())),
                'analysis_cap': MAX_SCOPED_MESSAGES,
                'analysis_cap_applied': capped,
            },
            'conversations': {
                'total': len(conversation_ids),
                'groups_spoken_in': len(group_ids),
            },
            'media': media,
            'understood_source_items': derived_counts,
            **auxiliary,
        },
        'evidence_candidates': evidence,
        'evidence_projection': {
            'full_scope_analyzed': not capped,
            'per_category_limit': evidence_limit,
            'selection': 'timeline_spread_sample',
            'expand_by_increasing_evidence_limit_or_opening_citation_context': True,
        },
        'person_model': person_model,
        'relationship_model': {
            'interaction_dynamics': _interaction_dynamics(messages),
            'timeline': _timeline(messages),
            'activity_trend': _activity_trend(messages),
            'context_style': _context_style(messages),
            'group_contexts': _group_contexts(messages),
            'key_events': _key_events(evidence),
            'relationship_principles': list(VIDEO_DERIVED_RELATIONSHIP_PRINCIPLES),
            'claim_status': {
                'active': sum(1 for values in claims.values() for value in values if value.get('review_status') == 'active'),
                'needs_review': unreviewed_claims,
            },
        },
        'relationship_actions': _relationship_actions(evidence, media),
        'questions_for_user': _questions_for_user(evidence, claims, media),
        'data_gaps': data_gaps,
        'analysis_protocol': _analysis_protocol(),
        'completion_workflow': _completion_workflow(),
        'scientific_framework': {
            'lenses': SCIENTIFIC_LENSES,
            'claim_classes': sorted(EVIDENCE_CLASSES),
            'minimum_pattern_evidence': 3,
            'requires_counterevidence_and_alternative_explanations': True,
            'trait_state_role_separated': True,
            'human_confirmation_required': True,
        },
        'clinical_diagnosis_included': False,
        'person_value_score_included': False,
        'auto_message_generation_enabled': False,
        'raw_chat_dump_included': False,
        'raw_content_included': False,
        'raw_paths_included': False,
    }


def _citation_exists(store: SQLiteStore, citation: str) -> bool:
    if not citation.startswith('trove://'):
        return False
    base = citation.split('#', 1)[0]
    with store.connect() as conn:
        for table in (
            'messages', 'moment_items', 'moment_interactions', 'transcripts', 'image_observations',
            'evidence_chunks', 'observations', 'relationships', 'entity_identifiers',
        ):
            if not store._table_exists(conn, table):
                continue
            columns = {str(row['name']) for row in conn.execute(f'PRAGMA table_info("{table}")')}
            if 'citation' not in columns:
                continue
            if conn.execute(
                f'SELECT 1 FROM "{table}" WHERE citation IN (?,?) LIMIT 1', (citation, base),
            ).fetchone() is not None:
                return True
    return False


def _validated_claim(store: SQLiteStore, claim: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(claim, dict):
        raise PersonProfileClaimError('each claim must be an object')
    dimension = str(claim.get('dimension') or '').strip()
    if dimension not in PROFILE_DIMENSIONS:
        raise PersonProfileClaimError(f'unsupported person profile dimension: {dimension}')
    evidence_class = str(claim.get('evidence_class') or '').strip()
    if evidence_class not in EVIDENCE_CLASSES:
        raise PersonProfileClaimError(f'unsupported evidence_class: {evidence_class}')
    statement = str(claim.get('statement') or '').strip()
    if not 1 <= len(statement) <= 1000:
        raise PersonProfileClaimError('statement must be from 1 to 1000 characters')
    citations = _unique_strings(claim.get('citations') or [], limit=MAX_CITATIONS_PER_CLAIM + 1)
    counterevidence = _unique_strings(claim.get('counterevidence_citations') or [], limit=MAX_CITATIONS_PER_CLAIM + 1)
    if evidence_class != 'unknown' and not citations:
        raise PersonProfileClaimError('cited evidence is required')
    if len(citations) > MAX_CITATIONS_PER_CLAIM or len(counterevidence) > MAX_CITATIONS_PER_CLAIM:
        raise PersonProfileClaimError(f'a claim may cite at most {MAX_CITATIONS_PER_CLAIM} evidence items')
    if evidence_class in {'pattern', 'hypothesis'} and len(citations) < 3:
        raise PersonProfileClaimError('patterns and hypotheses require at least three distinct citations')
    alternatives = _unique_strings(claim.get('alternative_explanations') or [], limit=3)
    if evidence_class == 'hypothesis' and not alternatives:
        raise PersonProfileClaimError('a hypothesis requires at least one alternative explanation')
    counterevidence_reviewed = bool(claim.get('counterevidence_reviewed', False))
    if evidence_class in {'pattern', 'hypothesis'} and not counterevidence_reviewed:
        raise PersonProfileClaimError('patterns and hypotheses require an explicit counterevidence review')
    for citation in [*citations, *counterevidence]:
        if not _citation_exists(store, citation):
            raise PersonProfileClaimError(f'evidence citation was not found: {citation}')
    try:
        confidence = float(claim.get('confidence') or 0)
    except (TypeError, ValueError):
        raise PersonProfileClaimError('confidence must be numeric') from None
    if not 0 <= confidence <= 1:
        raise PersonProfileClaimError('confidence must be from 0 to 1')
    confidence_ceiling = {'fact': 0.99, 'pattern': 0.85, 'hypothesis': 0.70, 'unknown': 0.0, 'action': 0.80}[evidence_class]
    confidence = min(confidence, confidence_ceiling)
    scope = str(claim.get('scope') or '').strip()
    if not scope or len(scope) > 200:
        raise PersonProfileClaimError('scope is required and must be at most 200 characters')
    return {
        'schema_version': PERSON_PROFILE_SCHEMA,
        'dimension': dimension,
        'evidence_class': evidence_class,
        'statement': statement,
        'citations': citations,
        'counterevidence_citations': counterevidence,
        'counterevidence_reviewed': counterevidence_reviewed,
        'alternative_explanations': alternatives,
        'scope': scope,
        'confidence': confidence,
        'cross_context': bool(claim.get('cross_context', False)),
        'clinical_diagnosis': False,
        'requires_human_confirmation': True,
    }


def propose_person_profile_claims(
    store: SQLiteStore,
    person: str,
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(claims, list) or not 1 <= len(claims) <= MAX_CLAIMS_PER_WRITE:
        raise PersonProfileClaimError(f'claims must contain from 1 to {MAX_CLAIMS_PER_WRITE} items')
    store.initialize()
    _, resolved = _resolved_scope(store, person)
    if not resolved:
        raise PersonProfileClaimError('person must resolve to one canonical entity')
    entity_id = str(resolved['entity_id'])
    validated = [_validated_claim(store, claim) for claim in claims]
    repo = MultimodalRepository(store)
    written = unchanged = 0
    observation_ids: list[str] = []
    for value in validated:
        observation_id = _stable_id('obs-profile', {
            'entity_id': entity_id,
            'dimension': value['dimension'],
            'evidence_class': value['evidence_class'],
            'statement': value['statement'],
            'citations': value['citations'],
        })
        observation_ids.append(observation_id)
        with store.connect() as conn:
            existing = conn.execute(
                'SELECT value_json,status,confidence FROM observations WHERE observation_id=?', (observation_id,),
            ).fetchone()
        if existing is not None and _load_json(existing['value_json']) == value and existing['status'] == 'needs_review' and float(existing['confidence']) == float(value['confidence']):
            unchanged += 1
            continue
        repo.add_observation(ObservationRecord(
            observation_id=observation_id,
            entity_id=entity_id,
            observation_type='person_profile_claim',
            value=value,
            status='needs_review',
            confidence=float(value['confidence']),
            citation=value['citations'][0] if value['citations'] else f'trove://profile/{observation_id}',
            source_type='agent',
        ))
        written += 1
    return {
        'ok': True,
        'type': 'person_profile_claim_proposal',
        'entity_id': entity_id,
        'written': written,
        'unchanged': unchanged,
        'observation_ids': observation_ids,
        'review_status': 'needs_review',
        'next_tool': 'trove_observe_approve',
        'raw_content_included': False,
        'raw_paths_included': False,
    }
