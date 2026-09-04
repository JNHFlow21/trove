from __future__ import annotations

from contextlib import closing

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import sqlite3
from typing import Any

from trove_core.store.repositories import EntityRecord, MultimodalRepository, ObservationRecord
from trove_core.wechat.parsers.contact_extra import parse_contact_extra_buffer
from trove_core.wechat.scope import ScopeDecision, classify_wechat_identity


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}'


def _text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip('\x00')
    return str(value).strip('\x00')


@dataclass(frozen=True)
class ContactIdentity:
    account_id: str
    username: str
    entity_id: str
    display_name: str
    alias: str = ''
    remark: str = ''
    nickname: str = ''
    signature: str = ''
    region: str = ''
    gender: str = ''
    avatar_ref: str = ''
    citation: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


class ContactIdentityImporter:
    def __init__(self, contact_db: Path, *, account_id: str):
        self.contact_db = Path(contact_db)
        self.account_id = account_id
        self.last_scope_counts: dict[str, int] = {}
        self.last_excluded_counts: dict[str, int] = {}
        self.last_scope_decisions: dict[str, ScopeDecision] = {}
        self.last_contacts: list[ContactIdentity] = []

    def load(self, limit: int | None = None) -> list[ContactIdentity]:
        self.last_contacts = []
        if not self.contact_db.exists():
            return []
        out: list[ContactIdentity] = []
        scope_counts: dict[str, int] = {}
        excluded_counts: dict[str, int] = {}
        decisions: dict[str, ScopeDecision] = {}
        try:
            with closing(sqlite3.connect(f'file:{self.contact_db}?mode=ro', uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                table = 'contact' if 'contact' in tables else next((t for t in tables if 'contact' in t.lower()), None)
                if not table:
                    return []
                cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
                select_cols = [c for c in ['username', 'user_name', 'remark', 'nick_name', 'nickname', 'alias', 'signature', 'province', 'city', 'country', 'region', 'gender', 'avatar', 'big_head_url', 'small_head_url', 'head_img_url', 'extra_buffer'] if c in cols]
                if not select_cols:
                    return []
                query = 'SELECT rowid AS __rowid__, ' + ','.join(f'"{c}"' for c in select_cols) + f' FROM "{table}"'
                if limit:
                    query += f' LIMIT {int(limit)}'
                for row in conn.execute(query):
                    username = _text(row['username'] if 'username' in row.keys() else row['user_name'] if 'user_name' in row.keys() else '')
                    if not username:
                        continue
                    decision = classify_wechat_identity(username, source_family='contact', is_contact=True)
                    decisions[username] = decision
                    scope_counts[decision.scope_type] = scope_counts.get(decision.scope_type, 0) + 1
                    if not decision.allowed or decision.scope_type != 'contact':
                        excluded_counts[decision.scope_type] = excluded_counts.get(decision.scope_type, 0) + 1
                        continue
                    remark = _text(row['remark']) if 'remark' in row.keys() else ''
                    nickname = _text(row['nick_name']) if 'nick_name' in row.keys() else (_text(row['nickname']) if 'nickname' in row.keys() else '')
                    alias = _text(row['alias']) if 'alias' in row.keys() else ''
                    signature = _text(row['signature']) if 'signature' in row.keys() else ''
                    extra = parse_contact_extra_buffer(row['extra_buffer']) if 'extra_buffer' in row.keys() else None
                    extra_fields = extra.fields if extra is not None else {}
                    signature = signature or extra_fields.get('signature', '')
                    region_parts = []
                    for col in ['country', 'province', 'city', 'region']:
                        if col in row.keys() and _text(row[col]):
                            region_parts.append(_text(row[col]))
                    region = ' '.join(dict.fromkeys(region_parts)).strip() or extra_fields.get('region', '')
                    if not region:
                        region = ' '.join(v for v in [extra_fields.get('country', ''), extra_fields.get('province', ''), extra_fields.get('city', '')] if v).strip()
                    gender = _text(row['gender']) if 'gender' in row.keys() else ''
                    gender = gender or extra_fields.get('gender', '')
                    avatar_ref = ''
                    for col in ['avatar', 'big_head_url', 'small_head_url', 'head_img_url']:
                        if col in row.keys() and _text(row[col]):
                            avatar_ref = _text(row[col]); break
                    avatar_ref = avatar_ref or extra_fields.get('avatar_ref', '')
                    display = remark or nickname or alias or username
                    entity_id = _stable('customer', f'{self.account_id}:{username}')
                    citation = f'trove://wechat/{self.account_id}/contact/{_stable("contact", username)}'
                    out.append(ContactIdentity(self.account_id, username, entity_id, display, alias, remark, nickname, signature, region, gender, avatar_ref, citation))
        except sqlite3.DatabaseError:
            self.last_contacts = out
            return out
        self.last_scope_counts = scope_counts
        self.last_excluded_counts = excluded_counts
        self.last_scope_decisions = decisions
        self.last_contacts = out
        return out

    def persist_loaded_to_ontology(self, repo: MultimodalRepository) -> int:
        """Persist contacts already loaded from the immutable source snapshot.

        Keeping source I/O separate lets sync parse large WeChat databases
        before it acquires the Vault writer.  This method performs database
        mutations only.
        """
        entities: list[EntityRecord] = []
        observations: list[ObservationRecord] = []
        for contact in self.last_contacts:
            entities.append(EntityRecord(
                entity_id=contact.entity_id,
                entity_type='Customer',
                display_name=contact.display_name,
                identifiers={'wechat_username': contact.username, 'alias': contact.alias, 'remark': contact.remark, 'nickname': contact.nickname},
                confidence=0.95,
            ))
            for obs_type, value in [('wechat_username', contact.username), ('alias', contact.alias), ('remark', contact.remark), ('nickname', contact.nickname), ('signature', contact.signature), ('region', contact.region), ('gender', contact.gender), ('avatar_ref', contact.avatar_ref)]:
                if value:
                    observations.append(ObservationRecord(
                        observation_id=_stable('obs', f'{contact.entity_id}:{obs_type}:{value}'),
                        entity_id=contact.entity_id,
                        observation_type=obs_type,
                        value={'text': value},
                        status='active',
                        confidence=0.95 if obs_type != 'avatar_ref' else 0.75,
                        citation=contact.citation,
                        source_type='contact',
                    ))
        repo.upsert_contact_batch(entities, observations)
        return len(entities)

    def import_to_ontology(self, repo: MultimodalRepository, *, limit: int | None = None) -> int:
        self.load(limit=limit)
        return self.persist_loaded_to_ontology(repo)
