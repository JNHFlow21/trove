from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

IncludedScopeType = Literal['private_chat', 'group_chat', 'contact', 'moment', 'favorite']
ScopeType = Literal[
    'private_chat', 'group_chat', 'contact', 'moment', 'favorite',
    'excluded_public_account', 'excluded_subscription', 'excluded_system',
    'excluded_service', 'excluded_notification', 'excluded_mini_program',
    'excluded_file_helper', 'excluded_orphan_cache_media', 'excluded_unknown',
    'coverage_gap',
]

INCLUDED_SCOPE_TYPES: set[str] = {'private_chat', 'group_chat', 'contact', 'moment', 'favorite'}
EXCLUDED_SCOPE_TYPES: set[str] = {
    'excluded_public_account', 'excluded_subscription', 'excluded_system', 'excluded_service',
    'excluded_notification', 'excluded_mini_program', 'excluded_file_helper',
    'excluded_orphan_cache_media', 'excluded_unknown', 'coverage_gap',
}

SYSTEM_IDENTITIES = {
    'weixin', 'fmessage', 'medianote', 'floatbottle', 'newsapp', 'qqmail', 'qqsync',
    'lbsapp', 'shakeapp', 'voiceinputapp', 'feedsapp', 'masssendapp', 'voicevoipapp',
    'facebookapp', 'linkedinplugin', 'blogapp', 'readerapp', 'weibo', 'qqfriend',
    'officialaccounts', 'notification_messages', 'notifymessage', 'notification_messages',
}
FILE_HELPER_IDENTITIES = {'filehelper'}
SERVICE_HINTS = (
    'notifymessage', 'notification', 'service', 'servicenotify', 'notice', 'pay', 'wallet',
    'brandservice', 'helper_entry', 'message_fold', 'appmsg', 'teams', 'qqmail',
)
SUBSCRIPTION_HINTS = ('subscription', 'subscribe', 'subscribemsg', 'mpnews', 'newsapp', 'brandservice')
MINI_PROGRAM_HINTS = ('appbrand', 'weapp', 'mini_program', 'miniprogram', '@app', 'wxapp')
HUMAN_PREFIXES = ('wxid_', 'wx_', 'v1_', 'v2_')


@dataclass(frozen=True)
class ScopeDecision:
    allowed: bool
    scope_type: ScopeType
    reason: str
    raw_kind: str
    confidence: float = 1.0

    @property
    def family(self) -> str:
        return self.scope_type if self.allowed else 'excluded'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {'family': self.family}


def _norm(value: str | None) -> str:
    return (value or '').strip().lower()


def _decision(allowed: bool, scope_type: ScopeType, reason: str, raw_kind: str, confidence: float = 1.0) -> ScopeDecision:
    return ScopeDecision(allowed=allowed, scope_type=scope_type, reason=reason, raw_kind=raw_kind, confidence=confidence)


def classify_wechat_identity(
    username: str | None,
    *,
    source_family: str | None = None,
    has_chat_history: bool = False,
    is_contact: bool = False,
    is_orphan_cache: bool = False,
    table_role: str | None = None,
) -> ScopeDecision:
    """Classify a WeChat identity or source row against TROVE's five-family product scope.

    The classifier is intentionally deterministic and redaction-safe: it only returns
    low-cardinality reasons and never message bodies, raw paths, or contact names.
    Unknown categories are excluded/coverage-gap by default.
    """
    family = _norm(source_family)
    role = _norm(table_role)
    raw = _norm(username)
    raw_kind = 'empty' if not raw else 'chatroom' if raw.endswith('@chatroom') else 'identity'

    if is_orphan_cache or family in {'orphan_cache', 'cache_media'} or role == 'orphan_cache':
        return _decision(False, 'excluded_orphan_cache_media', 'orphan cache media has no accepted citation link', 'orphan_cache', 0.98)
    if family in {'moment', 'moments', 'sns'}:
        return _decision(True, 'moment', 'Moments are an included relationship evidence family', 'moment', 0.95)
    if family in {'favorite', 'favorites', 'fav'}:
        return _decision(True, 'favorite', 'Favorites are included as a local knowledge namespace', 'favorite', 0.95)

    if not raw:
        return _decision(False, 'coverage_gap', 'missing identity requires coverage-gap handling', raw_kind, 0.4)
    if raw.endswith('@chatroom'):
        return _decision(True, 'group_chat', 'group chat conversation is included', 'chatroom', 0.99)
    if raw.startswith('gh_'):
        return _decision(False, 'excluded_public_account', 'public-account identity is excluded', 'gh_*', 0.99)
    if raw in FILE_HELPER_IDENTITIES:
        return _decision(False, 'excluded_file_helper', 'filehelper is not relationship evidence', 'filehelper', 0.99)
    if raw in SYSTEM_IDENTITIES:
        scope = 'excluded_notification' if 'notify' in raw or 'notification' in raw else 'excluded_system'
        return _decision(False, scope, 'system identity is excluded', 'system_identity', 0.98)
    if any(hint in raw for hint in MINI_PROGRAM_HINTS):
        return _decision(False, 'excluded_mini_program', 'mini-program push/source is excluded', 'mini_program_marker', 0.9)
    if any(hint in raw for hint in SUBSCRIPTION_HINTS):
        return _decision(False, 'excluded_subscription', 'subscription/public-account folding source is excluded', 'subscription_marker', 0.9)
    if any(hint in raw for hint in SERVICE_HINTS):
        return _decision(False, 'excluded_service', 'service or notification-like identity is excluded', 'service_marker', 0.82)

    if family in {'contact', 'contacts'} or is_contact:
        return _decision(True, 'contact', 'relationship contact identity is included', 'contact_identity', 0.88)
    if has_chat_history:
        confidence = 0.86 if raw.startswith(HUMAN_PREFIXES) else 0.72
        return _decision(True, 'private_chat', 'private chat identity with history is included after exclusion checks', 'private_candidate', confidence)

    if raw.startswith(HUMAN_PREFIXES):
        return _decision(True, 'contact', 'human-shaped relationship identity is included as contact candidate', 'human_prefix', 0.72)
    return _decision(False, 'excluded_unknown', 'unknown identity does not default to private relationship evidence', 'unknown_identity', 0.35)


def classify_media_reference(source_type: str | None, citation: str | None = None) -> ScopeDecision:
    source = _norm(source_type)
    citation_l = _norm(citation)
    for family in ('private_chat', 'group_chat', 'contact', 'moment', 'favorite'):
        if source == family:
            return _decision(True, family, 'media is linked to an accepted source family', 'media_link', 0.95)  # type: ignore[arg-type]
    if source in {'message', 'chat'} and ('/chat/' in citation_l or '/media/' in citation_l):
        return _decision(True, 'private_chat', 'media is linked to an accepted chat/media citation', 'media_link', 0.75)
    if '/moment/' in citation_l:
        return _decision(True, 'moment', 'media is linked to an accepted Moment citation', 'media_link', 0.9)
    if '/favorite/' in citation_l:
        return _decision(True, 'favorite', 'media is linked to an accepted Favorite citation', 'media_link', 0.9)
    if '/contact/' in citation_l:
        return _decision(True, 'contact', 'media is linked to an accepted contact citation', 'media_link', 0.85)
    return _decision(False, 'excluded_orphan_cache_media', 'media has no accepted source citation link', 'orphan_cache', 0.8)


def scope_counts(decisions: list[ScopeDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        key = decision.scope_type
        counts[key] = counts.get(key, 0) + 1
    return counts


def public_scope_contract() -> dict[str, Any]:
    return {
        'included_families': ['private_chat', 'group_chat', 'contact', 'moment', 'favorite'],
        'excluded_by_default': sorted(EXCLUDED_SCOPE_TYPES),
        'favorites_policy': 'Favorites are local knowledge evidence and do not automatically project customer profile facts.',
        'unknown_policy': 'Unknown source categories are excluded or coverage gaps until classified with a documented reason.',
    }
