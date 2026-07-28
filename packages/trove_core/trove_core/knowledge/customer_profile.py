from __future__ import annotations

import json
from typing import Any

from trove_core.knowledge.entity_resolution import resolve_customer
from trove_core.knowledge.ontology import ProfileClaim
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.domain.content import display_content_for_kind


DEFAULT_STOP_PHRASES = {
    '嗯', '嗯嗯', '嗯呐', '啊', '好', '好的', '好滴', '好哒', '对', '对的', '早', '早呀', '收到', 'ok', 'OK',
}
BUSINESS_SECTIONS = {'needs', 'objections', 'commitments', 'next_actions', 'timeline_summary'}
BUSINESS_TERMS = {'客户', '预算', '审批', '价格', '需求', '试点', '报价', '合同', '校区', '方案', '老师'}
CLOUD_ASR_PROVIDER_NAME = 'volcengine-asr-flash'
CLOUD_ASR_MODEL_ID = 'bigmodel:volc.bigasr.auc_turbo'


def _json_loads(text: str) -> Any:
    try:
        return json.loads(text or '{}')
    except json.JSONDecodeError:
        return {}


def _claim_from_observation(row) -> ProfileClaim:
    value = _json_loads(row['value_json'])
    if isinstance(value, dict):
        text = value.get('text') or value.get('value') or json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return ProfileClaim(str(text), [row['citation']], float(row['confidence']), row['status'], row['source_type'])


def _section_for_observation_type(obs_type: str) -> str:
    lower = obs_type.lower()
    if lower in {'wechat_username', 'alias', 'remark', 'nickname', 'signature', 'avatar_ref'}:
        return 'identity'
    if 'need' in lower:
        return 'needs'
    if 'objection' in lower or 'pain' in lower or 'blocker' in lower:
        return 'objections'
    if 'next' in lower or 'action' in lower:
        return 'next_actions'
    if 'commit' in lower:
        return 'commitments'
    return 'observations'


def _claim_from_text(value: str, citation: str, *, confidence: float, status: str, source_type: str) -> dict[str, Any]:
    return ProfileClaim(str(value or '')[:500], [citation], confidence, status, source_type).to_dict()


def _claim_from_message(row: Any, *, confidence: float = 0.75) -> dict[str, Any]:
    content = display_content_for_kind(row['content'], row['content_kind'] if 'content_kind' in row.keys() else 'text')
    claim = _claim_from_text(content, row['citation'], confidence=confidence, status='active', source_type=row['source_type'])
    direction = str(row['direction'] or '')
    if direction == 'outgoing':
        claim['direction'] = 'self'
    elif direction == 'incoming':
        claim['direction'] = 'peer'
    else:
        claim['direction'] = 'unknown'
    claim['conversation_id'] = row['conversation_id']
    claim['conversation_type'] = row['conversation_type']
    return claim


def _candidate_aliases(resolution: dict[str, Any], customer: str) -> list[str]:
    aliases = [customer]
    for candidate in [resolution.get('resolved'), *(resolution.get('candidates') or [])]:
        if not candidate:
            continue
        aliases.append(str(candidate.get('display_name') or ''))
        aliases.append(str(candidate.get('primary_user_id') or ''))
        aliases.extend(str(v) for v in (candidate.get('aliases') or []))
    return [a for a in dict.fromkeys(a.strip() for a in aliases if a and a.strip()) if len(a.strip()) >= 2][:12]


def _resolved_user_id(resolution: dict[str, Any]) -> str | None:
    resolved = resolution.get('resolved') or {}
    value = str(resolved.get('primary_user_id') or '').strip()
    if value and not value.startswith('unresolved:'):
        return value
    return None


def _resolved_entity_ids(resolution: dict[str, Any]) -> list[str]:
    resolved = resolution.get('resolved') or {}
    values = [resolved.get('entity_id'), *(resolved.get('entity_ids') or [])]
    return [
        entity_id for entity_id in dict.fromkeys(str(v).strip() for v in values if v)
        if entity_id and not entity_id.startswith('unresolved:')
    ]


def _resolved_conversation_ids(resolution: dict[str, Any]) -> list[str]:
    resolved = resolution.get('resolved') or {}
    return list(dict.fromkeys(str(v).strip() for v in (resolved.get('conversation_ids') or []) if str(v).strip()))[:20]


def _resolved_sender_ids(resolution: dict[str, Any]) -> list[str]:
    resolved = resolution.get('resolved') or {}
    values = [*(resolved.get('sender_ids') or [])]
    user_id = _resolved_user_id(resolution)
    if user_id:
        values.append(user_id)
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))[:20]


def _evidence_claims(store: SQLiteStore, aliases: list[str], *, source_type: str, limit: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alias in aliases:
        for row in store.chunk_search(alias, filters={'source_type': source_type}, limit=max(limit, 5)):
            citation = str(row['citation'])
            if citation in seen:
                continue
            seen.add(citation)
            matches.append({'content': row['content'], 'citation': citation})
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break
    hints = (
        store.media_hints_for_citations([row['citation'] for row in matches])
        if matches and hasattr(store, 'media_hints_for_citations') else {}
    )
    claims: list[dict[str, Any]] = []
    for row in matches:
        claim = _claim_from_text(
            row['content'], row['citation'], confidence=0.7, status='active', source_type=source_type,
        )
        hint = hints.get(row['citation'])
        if hint:
            claim['media_hint'] = hint
        claims.append(claim)
    return claims


def _chat_claims(store: SQLiteStore, aliases: list[str], resolution: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows: list[Any] = []
    seen: set[str] = set()
    user_id = _resolved_user_id(resolution)
    conversation_ids = _resolved_conversation_ids(resolution)
    sender_ids = _resolved_sender_ids(resolution)
    with store.connect() as conn:
        if conversation_ids or sender_ids or user_id:
            private_conversations = [
                *conversation_ids,
            ]
            if not private_conversations and user_id:
                private_conversations.extend(
                    row['conversation_id'] for row in conn.execute(
                        """SELECT DISTINCT conversation_id
                           FROM messages
                           WHERE conversation_type='private' AND (sender_id=? OR conversation_id=?)
                           ORDER BY conversation_id
                           LIMIT 20""",
                        (user_id, user_id),
                    )
                )
            selects: list[str] = []
            params: list[Any] = []
            columns = (
                'citation,content,content_kind,source_type,direction,sent_by_me,'
                'conversation_id,conversation_type,timestamp'
            )
            if private_conversations:
                placeholders = ','.join('?' for _ in private_conversations)
                selects.append(
                    f"SELECT {columns} FROM messages "
                    f"WHERE conversation_type='private' AND conversation_id IN ({placeholders})"
                )
                params.extend(private_conversations)
            if sender_ids:
                placeholders = ','.join('?' for _ in sender_ids)
                selects.append(
                    f"SELECT {columns} FROM messages "
                    f"WHERE conversation_type='group' AND sender_id IN ({placeholders})"
                )
                params.extend(sender_ids)
            if selects:
                rows.extend(conn.execute(
                    f"SELECT * FROM ({' UNION ALL '.join(selects)}) "
                    "ORDER BY timestamp DESC,citation DESC LIMIT ?",
                    (*params, max(limit * 7, limit)),
                ))
    if not rows and not user_id and not conversation_ids:
        for alias in aliases[:3]:
            for row in store.exact_search(alias, limit=max(limit * 2, limit)):
                rows.append(row)
            if rows:
                break
    claims: list[dict[str, Any]] = []
    for row in rows:
        citation = str(row['citation'])
        if citation in seen:
            continue
        seen.add(citation)
        if user_id and 'sent_by_me' in row.keys():
            claims.append(_claim_from_message(row))
        else:
            claims.append(_claim_from_text(row['content'], citation, confidence=0.75, status='active', source_type=row['source_type']))
        if len(claims) >= limit:
            break
    return claims


def _moment_author_claims(store: SQLiteStore, user_id: str | None, *, limit: int) -> list[dict[str, Any]]:
    if not user_id or not store.path.exists():
        return []
    with store.connect() as conn:
        if not store._table_exists(conn, 'moment_items'):
            return []
        rows = list(conn.execute(
            """SELECT citation,text,timestamp
               FROM moment_items
               WHERE author_id=? AND status='active'
               ORDER BY timestamp DESC, citation DESC
               LIMIT ?""",
            (user_id, limit),
        ))
    hints = store.media_hints_for_citations([row['citation'] for row in rows]) if hasattr(store, 'media_hints_for_citations') else {}
    claims: list[dict[str, Any]] = []
    for row in rows:
        claim = _claim_from_text(row['text'], row['citation'], confidence=0.8, status='active', source_type='moment')
        claim['timestamp'] = row['timestamp']
        hint = hints.get(row['citation'])
        if hint:
            claim['media_hint'] = hint
        claims.append(claim)
    return claims


def _interaction_claims(store: SQLiteStore, user_id: str | None, *, limit: int) -> list[dict[str, Any]]:
    if not user_id or not store.path.exists():
        return []
    with store.connect() as conn:
        if not store._table_exists(conn, 'moment_interactions') or not store._table_exists(conn, 'moment_items'):
            return []
        rows = list(conn.execute(
            """SELECT mi.citation, mi.interaction_type, mi.text, mi.timestamp, mi.actor_id, mi.actor_name,
                      m.citation AS moment_citation, m.author_id
               FROM moment_interactions mi
               LEFT JOIN moment_items m ON m.moment_id=mi.moment_id
               WHERE mi.actor_id=? OR m.author_id=?
               ORDER BY mi.timestamp DESC, mi.citation DESC
               LIMIT ?""",
            (user_id, user_id, limit),
        ))
    claims: list[dict[str, Any]] = []
    for row in rows:
        text = row['text'] or ''
        label = row['interaction_type'] or 'interaction'
        value = f'{label}: {text}' if text else label
        claim = _claim_from_text(value, row['citation'] or row['moment_citation'], confidence=0.7, status='active', source_type='moment_interaction')
        claim['timestamp'] = row['timestamp']
        claim['interaction_type'] = row['interaction_type']
        claim['moment_citation'] = row['moment_citation']
        claim['direction'] = 'by_customer' if row['actor_id'] == user_id else 'on_customer_moment'
        claims.append(claim)
    return claims


def _scoped_image_claims(store: SQLiteStore, resolution: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    conversation_ids = _resolved_conversation_ids(resolution)
    sender_ids = _resolved_sender_ids(resolution)
    user_id = _resolved_user_id(resolution)
    clauses: list[str] = []
    params: list[Any] = []
    if conversation_ids:
        marks = ','.join('?' for _ in conversation_ids)
        clauses.append(f"(m.conversation_type='private' AND m.conversation_id IN ({marks}))")
        params.extend(conversation_ids)
    if sender_ids:
        marks = ','.join('?' for _ in sender_ids)
        clauses.append(f"(m.conversation_type='private' AND (m.sender_id IN ({marks}) OR m.conversation_id IN ({marks})))")
        params.extend(sender_ids)
        params.extend(sender_ids)
    if user_id:
        clauses.append("""(mal.source_type='moment' AND (
            mi.author_id=? OR EXISTS (
                SELECT 1 FROM moment_interactions mint
                 WHERE mint.moment_id=mi.moment_id AND mint.actor_id=? AND mint.status='active'
            )
        ))""")
        params.extend([user_id, user_id])
    if not clauses:
        return []
    with store.connect() as conn:
        rows = list(conn.execute(
            f"""SELECT io.*,ma.modality,mal.source_type,COALESCE(m.timestamp,mi.timestamp,io.updated_at,io.created_at) AS ts
                   FROM image_observations io
                   JOIN media_assets ma ON ma.asset_id=io.asset_id
                   JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
              LEFT JOIN messages m ON m.citation=mal.source_citation
              LEFT JOIN moment_items mi ON mi.citation=CASE
                   WHEN instr(mal.source_citation,'#media-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#media-')-1)
                   WHEN instr(mal.source_citation,'#image-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#image-')-1)
                   WHEN instr(mal.source_citation,'#video-')>0 THEN substr(mal.source_citation,1,instr(mal.source_citation,'#video-')-1)
                   ELSE mal.source_citation END
                  WHERE io.status='active' AND ({' OR '.join(clauses)})
               ORDER BY ts DESC,io.observation_id LIMIT ?""",
            (*params, limit),
        ))
    claims: list[dict[str, Any]] = []
    for row in rows:
        text = str(row['caption'] or '').strip()
        visible = str(row['visible_text'] or '').strip()
        if visible and visible not in text:
            text = f'{text}\n{visible}'.strip()
        claim = _claim_from_text(text or '[视觉证据]', row['citation'], confidence=float(row['confidence'] or 0), status='active', source_type='image_observation')
        claim.update({
            'derived_evidence': True,
            'auto_approved_personal_fact': False,
            'content_sha256': row['content_sha256'],
            'model_id': row['model_id'],
            'prompt_version': row['prompt_version'],
            'timestamp': row['ts'],
        })
        claims.append(claim)
    return claims


def _file_claims(store: SQLiteStore, resolution: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    conversation_ids = _resolved_conversation_ids(resolution)
    sender_ids = _resolved_sender_ids(resolution)
    if not conversation_ids and not sender_ids:
        return []
    with store.connect() as conn:
        if not store._table_exists(conn, 'media_assets') or not store._table_exists(conn, 'media_asset_links'):
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if conversation_ids:
            placeholders = ','.join('?' for _ in conversation_ids)
            clauses.append(f'm.conversation_id IN ({placeholders})')
            params.extend(conversation_ids)
        if sender_ids:
            placeholders = ','.join('?' for _ in sender_ids)
            clauses.append(f'm.sender_id IN ({placeholders})')
            params.extend(sender_ids)
        rows = list(conn.execute(
            f"""SELECT ma.media_type, ma.metadata_json, COALESCE(mal.source_citation, ma.citation) AS source_citation,
                       COALESCE(m.timestamp, ma.updated_at, ma.created_at) AS ts
                  FROM media_assets ma
                  LEFT JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                  LEFT JOIN messages m ON m.citation=mal.source_citation OR m.citation=ma.citation
                 WHERE ma.modality IN ('file','attachment','document','image','voice','video')
                   AND ({' OR '.join(clauses)})
                 ORDER BY ts DESC, ma.asset_id
                 LIMIT ?""",
            (*params, limit),
        ))
    claims: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = json.loads(row['metadata_json'] or '{}')
        except json.JSONDecodeError:
            metadata = {}
        file_name = metadata.get('file_name') or 'file'
        citation = row['source_citation']
        if citation:
            claims.append(ProfileClaim(f"{file_name} ({row['media_type'] or 'file'})", [citation], 0.75, 'active', 'file').to_dict())
    return claims


def _pending_voice_summary(store: SQLiteStore, resolution: dict[str, Any]) -> list[dict[str, Any]]:
    user_id = _resolved_user_id(resolution)
    conversation_ids = _resolved_conversation_ids(resolution)
    if (not user_id and not conversation_ids) or not store.path.exists():
        return []
    with store.connect() as conn:
        if not store._table_exists(conn, 'messages'):
            return []
        if conversation_ids:
            placeholders = ','.join('?' for _ in conversation_ids)
            scope_sql = f'conversation_id IN ({placeholders})'
            scope_params: tuple[Any, ...] = tuple(conversation_ids)
        else:
            scope_sql = '(sender_id=? OR conversation_id=?)'
            scope_params = (user_id, user_id)
        row = conn.execute(
            f"""WITH scoped_voice(citation) AS (
                    SELECT citation
                      FROM messages
                     WHERE conversation_type='private'
                       AND content_kind='voice'
                       AND {scope_sql}
               ), understood(citation) AS (
                    SELECT CASE WHEN instr(t.citation,'#')>0
                                THEN substr(t.citation,1,instr(t.citation,'#')-1)
                                ELSE t.citation END
                      FROM transcripts t
                      JOIN provider_jobs pj ON pj.job_id=t.job_id
                      JOIN media_assets ma0 ON ma0.asset_id=t.asset_id
                     WHERE t.status='active' AND pj.provider=? AND pj.model=? AND pj.status='completed'
                       AND pj.request_hash=ma0.content_hash
                    UNION
                    SELECT CASE WHEN instr(ma.citation,'#')>0
                                THEN substr(ma.citation,1,instr(ma.citation,'#')-1)
                                ELSE ma.citation END
                      FROM transcripts t
                      JOIN media_assets ma ON ma.asset_id=t.asset_id
                      JOIN provider_jobs pj ON pj.job_id=t.job_id
                     WHERE t.status='active' AND pj.provider=? AND pj.model=? AND pj.status='completed'
                       AND pj.request_hash=ma.content_hash AND ma.citation IS NOT NULL
                    UNION
                    SELECT CASE WHEN instr(mal.source_citation,'#')>0
                                THEN substr(mal.source_citation,1,instr(mal.source_citation,'#')-1)
                                ELSE mal.source_citation END
                      FROM transcripts t
                      JOIN media_assets ma ON ma.asset_id=t.asset_id
                      JOIN media_asset_links mal ON mal.asset_id=ma.asset_id AND mal.accepted=1
                      JOIN provider_jobs pj ON pj.job_id=t.job_id
                     WHERE t.status='active' AND pj.provider=? AND pj.model=? AND pj.status='completed'
                       AND pj.request_hash=ma.content_hash AND mal.source_citation IS NOT NULL
               )
               SELECT COUNT(*) AS pending_count
                 FROM scoped_voice sv
            LEFT JOIN understood u ON u.citation=sv.citation
                WHERE u.citation IS NULL""",
            (
                *scope_params,
                CLOUD_ASR_PROVIDER_NAME,
                CLOUD_ASR_MODEL_ID,
                CLOUD_ASR_PROVIDER_NAME,
                CLOUD_ASR_MODEL_ID,
                CLOUD_ASR_PROVIDER_NAME,
                CLOUD_ASR_MODEL_ID,
            ),
        ).fetchone()
    count = int(row['pending_count'] or 0) if row is not None else 0
    if count <= 0:
        return []
    return [{
        'kind': 'pending_voice',
        'count': count,
        'hint': f'{count} voice messages exist, not yet transcribed.',
        'scope': 'private_chat',
        'status': 'pending_transcript',
        'transcribe_tool': 'trove_voice_transcribe_lazy',
        'raw_content_included': False,
        'raw_paths_included': False,
    }]


def _chat_claim_section(text: str) -> str | None:
    value = str(text or '')
    if any(term in value for term in ['需求', '需要', '想要', '目标', '希望', '试点']):
        return 'needs'
    if any(term in value for term in ['卡住', '卡在', '阻碍', '顾虑', '担心', '价格', '预算', '审批', '问题']):
        return 'objections'
    if any(term in value for term in ['承诺', '确认', '同意', '可以推动', '会推动']):
        return 'commitments'
    if any(term in value for term in ['下一步', '下周', '明天', '跟进', '负责', '约', '复盘', '报价']):
        return 'next_actions'
    return None


def _is_low_signal_chatter(value: str) -> bool:
    text = str(value or '').strip()
    compact = ''.join(ch for ch in text if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')
    if not compact:
        return True
    lowered = compact.lower()
    if any(ch.isdigit() for ch in compact):
        return False
    if any(term in compact for term in BUSINESS_TERMS):
        return False
    return compact in DEFAULT_STOP_PHRASES or (len(compact) < 4 and lowered in {p.lower() for p in DEFAULT_STOP_PHRASES})


def _filter_low_signal_sections(sections: dict[str, list[dict[str, Any]]]) -> None:
    for name in BUSINESS_SECTIONS:
        sections[name] = [claim for claim in sections.get(name, []) if not _is_low_signal_chatter(claim.get('value') or '')]


def _append_unique(section: list[dict[str, Any]], claim: dict[str, Any]) -> None:
    key = (claim.get('value'), tuple(claim.get('citations') or []))
    for existing in section:
        if (existing.get('value'), tuple(existing.get('citations') or [])) == key:
            return
    section.append(claim)


def build_customer_profile(store: SQLiteStore, customer: str, *, limit: int = 5) -> dict[str, Any]:
    from trove_core.bounds import BoundedLimit, PROFILE_WIKI_REPORT

    limit = BoundedLimit(limit, field='limit', spec=PROFILE_WIKI_REPORT)
    sections: dict[str, list[dict[str, Any]]] = {
        'identity': [],
        'needs': [],
        'objections': [],
        'commitments': [],
        'next_actions': [],
        'file_exchanges': [],
        'timeline_summary': [],
        'moments': [],
        'moments_authored': [],
        'moments_mentioned': [],
        'interactions': [],
        'voice_transcripts': [],
        'pending_voice': [],
        'image_observations': [],
        'chat_evidence': [],
        'observations': [],
        'ambiguities': [],
    }
    if not store.path.is_file():
        bounded_sections = {key: [] for key in sections}
        bounded_sections['文件往来'] = []
        bounded_sections['时间线摘要'] = []
        return {
            'type': 'customer_profile',
            'customer': customer,
            'resolved_entity': None,
            'candidates': [],
            'sections': bounded_sections,
            'claim_policy': 'Every factual profile claim is projected from an observation or cited evidence handle; raw chat dumps are not included.',
            'raw_content_included': False,
        }
    store.initialize()
    resolution = resolve_customer(store, customer)
    entity_ids = _resolved_entity_ids(resolution)
    aliases = _candidate_aliases(resolution, customer)
    if resolution.get('resolved'):
        resolved = resolution['resolved']
        citations = [c for c in (resolved.get('citations') or []) if c]
        if citations:
            sections['identity'].append(ProfileClaim(
                resolved.get('display_name') or customer,
                citations[:3],
                float(resolved.get('confidence') or 0.8),
                'active',
                'entity',
            ).to_dict())
    with store.connect() as conn:
        if entity_ids:
            placeholders = ','.join('?' for _ in entity_ids)
            rows = list(conn.execute(
                f"SELECT * FROM observations WHERE entity_id IN ({placeholders}) AND status IN ('active','needs_review','merge_candidate') ORDER BY CASE WHEN source_type='operator' THEN 0 ELSE 1 END, confidence DESC, updated_at DESC LIMIT ?",
                (*entity_ids, limit * 4),
            ))
            for row in rows:
                claim = _claim_from_observation(row).to_dict()
                sections[_section_for_observation_type(row['observation_type'])].append(claim)
    user_id = _resolved_user_id(resolution)
    sections['moments_authored'].extend(_moment_author_claims(store, user_id, limit=limit))
    sections['moments_mentioned'].extend(_evidence_claims(store, aliases, source_type='moment', limit=limit))
    sections['interactions'].extend(_interaction_claims(store, user_id, limit=limit))
    sections['moments'].extend((sections['moments_authored'] + sections['moments_mentioned'])[:limit])
    sections['voice_transcripts'].extend(_evidence_claims(store, aliases, source_type='transcript', limit=limit))
    sections['pending_voice'].extend(_pending_voice_summary(store, resolution))
    scoped_image_claims = _scoped_image_claims(store, resolution, limit=limit)
    for claim in scoped_image_claims:
        _append_unique(sections['image_observations'], claim)
    if not scoped_image_claims:
        for claim in _evidence_claims(store, aliases, source_type='image_observation', limit=limit):
            _append_unique(sections['image_observations'], claim)
    for claim in _chat_claims(store, aliases, resolution, limit=limit):
        _append_unique(sections['chat_evidence'], claim)
        section = _chat_claim_section(claim.get('value') or '')
        if section:
            _append_unique(sections[section], claim)
        _append_unique(sections['timeline_summary'], ProfileClaim(
            claim.get('value') or '',
            claim.get('citations') or [],
            0.6,
            'active',
            'timeline',
        ).to_dict())
    sections['file_exchanges'].extend(_file_claims(store, resolution, limit=limit))
    if resolution['ambiguous']:
        sections['ambiguities'] = [{'value': 'multiple customer candidates require review', 'candidates': resolution['merge_candidates'], 'citations': []}]
    _filter_low_signal_sections(sections)
    bounded_sections = {k: v[:limit] for k, v in sections.items()}
    bounded_sections['文件往来'] = list(bounded_sections.get('file_exchanges') or [])
    bounded_sections['时间线摘要'] = list(bounded_sections.get('timeline_summary') or [])
    return {
        'type': 'customer_profile',
        'customer': customer,
        'resolved_entity': resolution.get('resolved'),
        'candidates': resolution.get('candidates', []),
        'sections': bounded_sections,
        'claim_policy': 'Every factual profile claim is projected from an observation or cited evidence handle; raw chat dumps are not included.',
        'raw_content_included': False,
    }
