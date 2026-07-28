from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
import unicodedata
from typing import Any

from trove_core.store.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    display_name: str
    entity_type: str
    confidence: float
    citations: list[str]
    match_reasons: list[str]
    primary_user_id: str | None = None
    entity_ids: list[str] | None = None
    aliases: list[str] | None = None
    conversation_ids: list[str] | None = None
    sender_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_loads(text: str) -> Any:
    try:
        return json.loads(text or '{}')
    except json.JSONDecodeError:
        return {}


USER_ID_KEYS = ('user_id', 'wechat_username', 'wechat_id', 'username', 'wxid', 'openim_id', 'primary_user_id')
ALIAS_KEYS = ('alias', 'aliases', 'remark', 'nickname', 'display_name', 'group_alias', 'group_name', 'group_display_name', 'name')


def normalize_identifier(value: Any) -> str:
    text = unicodedata.normalize('NFKC', str(value or '')).strip().casefold()
    return ' '.join(text.split())


_normalized_identifier = normalize_identifier


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_string_values(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_string_values(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _primary_user_id(entity_id: str, identifiers: dict[str, Any]) -> str:
    for key in USER_ID_KEYS:
        for value in _string_values(identifiers.get(key)):
            if value:
                return value
    return entity_id


def _aliases(display_name: str, identifiers: dict[str, Any]) -> list[str]:
    values = [display_name]
    for key, value in identifiers.items():
        if key in USER_ID_KEYS or key in ALIAS_KEYS:
            values.extend(_string_values(value))
    return list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))


def _wechat_account_ids(citations: list[str]) -> set[str]:
    account_ids: set[str] = set()
    for citation in citations:
        parts = str(citation or '').split('/')
        if len(parts) >= 4 and parts[:3] == ['trove:', '', 'wechat'] and parts[3]:
            account_ids.add(parts[3])
    return account_ids


def _query_tokens(query: str) -> list[str]:
    """Split concatenated multilingual names without weakening exact matching."""

    normalized = unicodedata.normalize('NFKC', str(query or '')).strip()
    raw = re.findall(r'[A-Za-z0-9_]+|[\u3400-\u9fff]+', normalized)
    return list(dict.fromkeys(_normalized_identifier(token) for token in raw if _normalized_identifier(token)))


def _candidate_matches_token(candidate: EntityCandidate, token: str) -> bool:
    values = [candidate.display_name, candidate.primary_user_id, *(candidate.aliases or [])]
    minimum_partial_length = 3 if token.isascii() else 2
    return any(
        normalized == token or (len(token) >= minimum_partial_length and token in normalized)
        for value in values
        if value and (normalized := _normalized_identifier(value))
    )


def _multi_token_candidates(store: SQLiteStore, query: str, *, limit: int) -> list[EntityCandidate]:
    tokens = _query_tokens(query)
    if len(tokens) < 2:
        return []
    token_candidates = [
        [
            candidate for candidate in find_customer_candidates(store, token, limit=max(limit * 4, 20))
            if _candidate_matches_token(candidate, token)
        ]
        for token in tokens
    ]
    if any(not candidates for candidates in token_candidates):
        return []

    def entity_keys(candidate: EntityCandidate) -> set[str]:
        return {str(value) for value in [candidate.entity_id, *(candidate.entity_ids or [])] if str(value or '')}

    common_entities = set.intersection(*(
        {entity_id for candidate in candidates for entity_id in entity_keys(candidate)}
        for candidates in token_candidates
    ))
    combined: list[EntityCandidate] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for entity_id in sorted(common_entities):
        matches = [
            next(candidate for candidate in candidates if entity_id in entity_keys(candidate))
            for candidates in token_candidates
        ]
        all_entity_ids = sorted({value for candidate in matches for value in entity_keys(candidate)})
        signature = tuple(all_entity_ids)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        representative = matches[0]
        primary_ids = _unique_nonempty(candidate.primary_user_id for candidate in matches)
        combined.append(EntityCandidate(
            entity_id=entity_id,
            display_name=representative.display_name,
            entity_type=representative.entity_type,
            confidence=min(candidate.confidence for candidate in matches),
            citations=_unique_nonempty(citation for candidate in matches for citation in candidate.citations)[:5],
            match_reasons=_unique_nonempty([
                'multi_token_identifier_match',
                *(reason for candidate in matches for reason in candidate.match_reasons),
            ])[:8],
            primary_user_id=primary_ids[0] if len(primary_ids) == 1 else representative.primary_user_id,
            entity_ids=all_entity_ids[:10],
            aliases=_unique_nonempty(alias for candidate in matches for alias in (candidate.aliases or []))[:12],
            conversation_ids=_unique_nonempty(
                conversation_id for candidate in matches for conversation_id in (candidate.conversation_ids or [])
            )[:20],
            sender_ids=_unique_nonempty(sender_id for candidate in matches for sender_id in (candidate.sender_ids or []))[:20],
        ))
    return sorted(combined, key=lambda candidate: candidate.confidence, reverse=True)[:limit]


def _unique_nonempty(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value or '').strip()))


def _score_aliases(needle: str, aliases: list[str], *, user_id: str | None = None) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    if user_id:
        user_lower = user_id.lower()
        if needle == user_lower:
            reasons.append('user_id_exact')
            score = max(score, 0.99)
        elif needle in user_lower:
            reasons.append('user_id_partial')
            score = max(score, 0.8)
    for alias in aliases:
        lower = alias.lower()
        if not lower:
            continue
        if needle == lower:
            reasons.append('identifier_exact')
            score = max(score, 0.98)
        elif needle in lower:
            reasons.append('identifier_partial')
            score = max(score, 0.75)
    return score, reasons


def _score_entity_identifiers(needle: str, display_name: str, identifiers: dict[str, Any], *, user_id: str | None = None) -> tuple[float, list[str]]:
    """Score identity fields in the frozen resolver priority order."""
    reasons: list[str] = []
    score = 0.0
    if user_id:
        user_lower = user_id.lower()
        if needle == user_lower:
            reasons.extend(['user_id_exact', 'identifier_exact'])
            score = max(score, 0.99)
        elif needle in user_lower:
            reasons.append('user_id_partial')
            score = max(score, 0.78)

    def exact_from(keys: tuple[str, ...], reason: str, value_score: float) -> None:
        nonlocal score
        for key in keys:
            for value in _string_values(identifiers.get(key)):
                if needle == value.lower():
                    reasons.extend([reason, 'identifier_exact'])
                    score = max(score, value_score)
                elif needle in value.lower():
                    reasons.append(f'{reason}_partial')
                    score = max(score, min(value_score - 0.2, 0.76))

    exact_from(('remark',), 'contact_remark_exact', 0.97)
    exact_from(('nickname',), 'contact_nickname_exact', 0.94)
    exact_from(('group_alias', 'group_display_name', 'group_name', 'name'), 'group_alias_exact', 0.90)
    exact_from(('alias', 'display_name'), 'identifier_exact', 0.88)
    display_lower = str(display_name or '').lower()
    if needle == display_lower:
        reasons.append('identifier_exact')
        score = max(score, 0.88)
    elif needle in display_lower:
        reasons.append('identifier_partial')
        score = max(score, 0.74)
    return score, list(dict.fromkeys(reasons))


def find_customer_candidates(store: SQLiteStore, query: str, *, limit: int = 5) -> list[EntityCandidate]:
    if not store.path.exists():
        return []
    needle = _normalized_identifier(query)
    if not needle:
        return []
    store.initialize()
    merged: dict[str, dict[str, Any]] = {}

    def update_candidate(
        key: str,
        *,
        entity_id: str,
        display_name: str,
        entity_type: str,
        confidence: float,
        citations: list[str] | None = None,
        reasons: list[str] | None = None,
        aliases: list[str] | None = None,
        primary_user_id: str | None = None,
        conversation_ids: list[str] | None = None,
        sender_ids: list[str] | None = None,
    ) -> None:
        if confidence <= 0:
            return
        item = merged.setdefault(key, {
            'entity_id': entity_id,
            'display_name': display_name,
            'entity_type': entity_type,
            'confidence': 0.0,
            'citations': [],
            'match_reasons': [],
            'primary_user_id': primary_user_id,
            'entity_ids': [],
            'aliases': [],
            'conversation_ids': [],
            'sender_ids': [],
        })
        if confidence > item['confidence']:
            item['entity_id'] = entity_id
            item['display_name'] = display_name
            item['entity_type'] = entity_type
            item['confidence'] = confidence
        if primary_user_id and not item.get('primary_user_id'):
            item['primary_user_id'] = primary_user_id
        for value in [entity_id]:
            if value and value not in item['entity_ids']:
                item['entity_ids'].append(value)
        for citation in citations or []:
            if citation and citation not in item['citations']:
                item['citations'].append(citation)
        for reason in reasons or []:
            if reason and reason not in item['match_reasons']:
                item['match_reasons'].append(reason)
        for alias in aliases or []:
            if alias and alias not in item['aliases']:
                item['aliases'].append(alias)
        for conversation_id in conversation_ids or []:
            if conversation_id and conversation_id not in item['conversation_ids']:
                item['conversation_ids'].append(conversation_id)
        for sender_id in sender_ids or []:
            if sender_id and sender_id not in item['sender_ids']:
                item['sender_ids'].append(sender_id)

    with store.connect() as conn:
        identifier_rows = list(conn.execute(
            """SELECT e.*,ei.identifier_type,ei.normalized_value,ei.confidence AS identifier_confidence,ei.citation AS identifier_citation
                 FROM entity_identifiers ei
                 JOIN entities e ON e.entity_id=ei.entity_id
                WHERE e.entity_type IN ('Customer','Person','Organization')
                  AND e.status<>'merged'
                  AND ei.normalized_value=?
                ORDER BY ei.confidence DESC,e.confidence DESC,e.entity_id
                LIMIT 100""",
            (needle,),
        ))
        if not identifier_rows:
            escaped = needle.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            identifier_rows = list(conn.execute(
                """SELECT e.*,ei.identifier_type,ei.normalized_value,ei.confidence AS identifier_confidence,ei.citation AS identifier_citation
                     FROM entity_identifiers ei
                     JOIN entities e ON e.entity_id=ei.entity_id
                    WHERE e.entity_type IN ('Customer','Person','Organization')
                      AND e.status<>'merged'
                      AND ei.normalized_value LIKE ? ESCAPE '\\'
                    ORDER BY ei.confidence DESC,e.confidence DESC,e.entity_id
                    LIMIT 100""",
                (f'%{escaped}%',),
            ))

        entity_rows: dict[str, Any] = {}
        match_rows: dict[str, list[Any]] = {}
        for row in identifier_rows:
            entity_rows[str(row['entity_id'])] = row
            match_rows.setdefault(str(row['entity_id']), []).append(row)

        for entity_id, row in entity_rows.items():
            identifiers = _json_loads(row['identifiers_json'])
            if not isinstance(identifiers, dict):
                identifiers = {}
            primary_user_id = _primary_user_id(row['entity_id'], identifiers)
            key = primary_user_id or row['entity_id']
            aliases = _aliases(row['display_name'], identifiers)
            score, reasons = _score_entity_identifiers(needle, row['display_name'], identifiers, user_id=primary_user_id)
            citations: list[str] = []
            for match in match_rows.get(entity_id, []):
                kind = str(match['identifier_type'] or '')
                reason = {
                    'user_id': 'user_id_exact',
                    'remark': 'contact_remark_exact',
                    'nickname': 'contact_nickname_exact',
                    'group_alias': 'group_alias_exact',
                    'group_display_name': 'group_alias_exact',
                    'group_name': 'group_alias_exact',
                }.get(kind, 'identifier_exact')
                reasons.extend([reason, 'identifier_exact'])
                score = max(score, min(float(match['identifier_confidence'] or 0), 1.0))
                if match['identifier_citation']:
                    citations.append(str(match['identifier_citation']))
            if score > 0:
                update_candidate(
                    key,
                    entity_id=row['entity_id'],
                    display_name=row['display_name'],
                    entity_type=row['entity_type'],
                    confidence=min(score, min(max(float(row['confidence'] or 0), 0.0), 1.0)),
                    citations=citations[:5],
                    reasons=reasons[:5],
                    aliases=aliases,
                    primary_user_id=primary_user_id,
                )

        # Expand only exact user-id siblings for already matched candidates.
        for key, item in list(merged.items()):
            user_id = str(item.get('primary_user_id') or '')
            if not user_id:
                continue
            user_norm = _normalized_identifier(user_id)
            for sibling in conn.execute(
                """SELECT DISTINCT e.*
                     FROM entity_identifiers ei
                     JOIN entities e ON e.entity_id=ei.entity_id
                    WHERE ei.identifier_type='user_id' AND ei.normalized_value=? AND e.status<>'merged'
                    ORDER BY e.entity_id""",
                (user_norm,),
            ):
                identifiers = _json_loads(sibling['identifiers_json'])
                if not isinstance(identifiers, dict):
                    identifiers = {}
                update_candidate(
                    key,
                    entity_id=sibling['entity_id'],
                    display_name=sibling['display_name'],
                    entity_type=sibling['entity_type'],
                    confidence=float(item['confidence']),
                    reasons=['user_id_sibling'],
                    aliases=_aliases(sibling['display_name'], identifiers),
                    primary_user_id=_primary_user_id(sibling['entity_id'], identifiers),
                )

        # Bind a contact-backed candidate to private conversations through any
        # exact alias, not only through the literal query string. This lets a
        # precise remark resolve a chat titled with the person's nickname.
        # Ambiguous aliases are ignored unless a contact citation binds the
        # candidate to the same WeChat account.
        all_aliases = {
            _normalized_identifier(alias)
            for item in merged.values()
            for alias in item.get('aliases') or []
            if _normalized_identifier(alias)
        }
        conversation_rows: list[Any] = []
        aliases_list = sorted(all_aliases)
        for start in range(0, len(aliases_list), 400):
            batch = aliases_list[start:start + 400]
            marks = ','.join('?' for _ in batch)
            conversation_rows.extend(conn.execute(
                f"""SELECT account_id,conversation_id,title FROM conversations
                      WHERE type='private' AND lower(trim(title)) IN ({marks})
                      ORDER BY account_id,conversation_id""",
                batch,
            ))
        rows_by_alias: dict[str, list[Any]] = {}
        for row in conversation_rows:
            rows_by_alias.setdefault(_normalized_identifier(row['title']), []).append(row)
        for key, item in list(merged.items()):
            candidate_aliases = {
                _normalized_identifier(alias) for alias in item.get('aliases') or []
                if _normalized_identifier(alias)
            }
            account_ids = _wechat_account_ids(list(item.get('citations') or []))
            accepted: list[Any] = []
            for alias in candidate_aliases:
                matches = rows_by_alias.get(alias, [])
                if account_ids:
                    for account_id in account_ids:
                        scoped = [row for row in matches if str(row['account_id']) == account_id]
                        if len(scoped) == 1:
                            accepted.extend(scoped)
                elif len(matches) == 1:
                    accepted.extend(matches)
            accepted_by_key = {
                (str(row['account_id']), str(row['conversation_id'])): row for row in accepted
            }
            if not accepted_by_key:
                continue
            sender_ids: list[str] = []
            for account_id, conversation_id in accepted_by_key:
                sender_ids.extend(
                    str(row['sender_id'])
                    for row in conn.execute(
                        """SELECT DISTINCT sender_id FROM messages
                            WHERE account_id=? AND conversation_id=?
                              AND conversation_type='private' AND direction='incoming'
                              AND sender_id<>''""",
                        (account_id, conversation_id),
                    )
                )
            update_candidate(
                key,
                entity_id=item['entity_id'],
                display_name=item['display_name'],
                entity_type=item['entity_type'],
                confidence=float(item['confidence']),
                aliases=[row['title'] for row in accepted_by_key.values()],
                primary_user_id=item.get('primary_user_id'),
                conversation_ids=[conversation_id for _account_id, conversation_id in accepted_by_key],
                sender_ids=sender_ids,
            )

        private_rows = list(conn.execute(
            """SELECT conversation_id,title
                 FROM conversations
                WHERE type='private' AND lower(title)=?
                ORDER BY conversation_id
                LIMIT ?""",
            (needle, max(limit * 4, 20)),
        )) if store._table_exists(conn, 'conversations') else []
        if merged and private_rows:
            conversation_ids = [str(row['conversation_id']) for row in private_rows]
            placeholders = ','.join('?' for _ in conversation_ids)
            sender_rows = list(conn.execute(
                f"""SELECT DISTINCT conversation_id,sender_id,sender_name
                       FROM messages
                      WHERE conversation_type='private'
                        AND conversation_id IN ({placeholders})
                        AND direction='incoming'""",
                tuple(conversation_ids),
            ))
            sender_ids = [str(row['sender_id']) for row in sender_rows if row['sender_id']]
            for key, item in merged.items():
                if any(needle == _normalized_identifier(alias) for alias in item['aliases']):
                    update_candidate(
                        key,
                        entity_id=item['entity_id'],
                        display_name=item['display_name'],
                        entity_type=item['entity_type'],
                        confidence=float(item['confidence']),
                        aliases=[row['title'] for row in private_rows],
                        primary_user_id=item.get('primary_user_id'),
                        conversation_ids=conversation_ids,
                        sender_ids=sender_ids,
                    )
        # Fallback: surface conversation/message matches as unresolved customer candidates.
        if not merged:
            for row in private_rows:
                entity_id = f'unresolved:{row["conversation_id"]}'
                update_candidate(
                    row['conversation_id'],
                    entity_id=entity_id,
                    display_name=row['title'],
                    entity_type='Customer',
                    confidence=1.0,
                    citations=[],
                    reasons=['private_conversation_title_exact'],
                    aliases=[row['title'], row['conversation_id']],
                    primary_user_id=row['conversation_id'],
                    conversation_ids=[row['conversation_id']],
                )
            rows = [] if private_rows else list(conn.execute("SELECT citation, conversation_id, conversation_title, sender_id, sender_name, conversation_type FROM messages WHERE conversation_title LIKE ? OR sender_name LIKE ? ORDER BY timestamp LIMIT ?", (f'%{query}%', f'%{query}%', limit)))
            for row in rows:
                is_private_exact = row['conversation_type'] == 'private' and needle == str(row['conversation_title'] or '').lower()
                is_group_exact = row['conversation_type'] == 'group' and needle == str(row['sender_name'] or '').lower()
                entity_key = row['conversation_id'] if is_private_exact else row['sender_id'] or row['conversation_title'] or row['sender_name']
                entity_id = f'unresolved:{entity_key}'
                update_candidate(
                    str(entity_key),
                    entity_id=entity_id,
                    display_name=row['conversation_title'] or row['sender_name'],
                    entity_type='Customer',
                    confidence=1.0 if is_private_exact else (0.9 if is_group_exact else 0.4),
                    citations=[row['citation']],
                    reasons=['private_conversation_title_exact' if is_private_exact else ('group_alias_exact' if is_group_exact else 'message_metadata')],
                    aliases=[row['conversation_title'], row['sender_name']],
                    primary_user_id=row['conversation_id'] if is_private_exact else row['sender_id'],
                    conversation_ids=[row['conversation_id']] if is_private_exact else [],
                    sender_ids=[row['sender_id']] if row['sender_id'] else [],
                )
    candidates = [
        EntityCandidate(
            item['entity_id'],
            item['display_name'],
            item['entity_type'],
            float(item['confidence']),
            item['citations'][:5],
            item['match_reasons'][:8],
            item.get('primary_user_id'),
            item.get('entity_ids')[:10],
            item.get('aliases')[:12],
            item.get('conversation_ids')[:20],
            item.get('sender_ids')[:20],
        )
        for item in merged.values()
    ]
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)[:limit]


def resolve_customer(store: SQLiteStore, query: str) -> dict[str, Any]:
    tokens = _query_tokens(query)
    exact_private_title = False
    if len(tokens) >= 2 and store.path.exists():
        store.initialize()
        with store.connect() as conn:
            exact_private_title = conn.execute(
                "SELECT 1 FROM conversations WHERE type='private' AND lower(title)=? LIMIT 1",
                (_normalized_identifier(query),),
            ).fetchone() is not None
    # A concatenated remark+nickname usually has no literal identifier row.
    # Intersect its component identifiers first and avoid an expensive,
    # guaranteed-empty whole-string scan. A literal private-chat title keeps
    # the established highest-priority path.
    candidates = (
        _multi_token_candidates(store, query, limit=5)
        if len(tokens) >= 2 and not exact_private_title
        else find_customer_candidates(store, query, limit=5)
    )
    if not candidates and len(tokens) >= 2 and not exact_private_title:
        candidates = find_customer_candidates(store, query, limit=5)
    priority = {
        'private_conversation_title_exact': 50,
        'user_id_exact': 45,
        'contact_remark_exact': 40,
        'contact_nickname_exact': 30,
        'group_alias_exact': 20,
        'identifier_exact': 10,
    }

    def candidate_priority(candidate: EntityCandidate) -> int:
        return max((priority.get(reason, 0) for reason in candidate.match_reasons), default=0)

    selected = None
    if candidates:
        # Resolver priority is an identity-strength ordering, not a secondary
        # tie-break after caller supplied confidence.  A temporary unresolved
        # entity may legitimately have confidence=1.0 while a contact-backed
        # remark has confidence<1.0; the stronger contact identifier must win.
        top_priority = max(candidate_priority(candidate) for candidate in candidates)
        priority_leaders = [candidate for candidate in candidates if candidate_priority(candidate) == top_priority]
        top_confidence = max(candidate.confidence for candidate in priority_leaders)
        leaders = [candidate for candidate in priority_leaders if abs(candidate.confidence - top_confidence) < 1e-9]
        if top_priority > 0 and len(leaders) == 1:
            selected = leaders[0]
        elif len(candidates) == 1:
            selected = candidates[0]
    return {
        'query': query,
        'resolved': selected.to_dict() if selected else None,
        'candidates': [c.to_dict() for c in candidates],
        'ambiguous': selected is None and len(candidates) > 1,
        'merge_candidates': [c.to_dict() for c in candidates] if len(candidates) > 1 else [],
    }
