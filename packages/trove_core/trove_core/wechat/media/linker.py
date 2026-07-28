from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Iterable

from trove_core.store.repositories import MediaAssetLinkRecord, MediaAssetRecord, MultimodalRepository
from trove_core.wechat.media.resources import MediaReference
from trove_core.wechat.scope import classify_media_reference


def _stable(prefix: str, value: str) -> str:
    return f'{prefix}-{hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]}'


@dataclass(frozen=True)
class MediaLinkResult:
    assets_seen: int
    assets_upserted: int
    links_upserted: int
    accepted_links: int
    excluded_links: int
    excluded_counts: dict[str, int]
    changed_asset_ids: tuple[str, ...] = ()
    metrics: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MediaLinker:
    """Persist media assets and accepted-source links before decode/provider work."""

    def __init__(self, repo: MultimodalRepository):
        self.repo = repo

    @staticmethod
    def _classification_source_type(ref: MediaReference) -> str | None:
        if ref.source_type == 'group_chat':
            return 'wechat_metadata'
        if ref.modality == 'voice' and ref.source_type != 'private_chat':
            return 'wechat_metadata'
        return ref.source_type

    def link_references(
        self,
        refs: Iterable[MediaReference],
        *,
        source_states: Iterable[dict[str, Any]] = (),
    ) -> MediaLinkResult:
        assets_seen = 0
        accepted_links = 0
        excluded_links = 0
        excluded_counts: dict[str, int] = {}
        assets: list[MediaAssetRecord] = []
        links: list[MediaAssetLinkRecord] = []
        for ref in refs:
            assets_seen += 1
            assets.append(MediaAssetRecord(
                asset_id=ref.asset_id,
                account_id=ref.account_id,
                source_type=ref.source_type,
                source_id=ref.source_id,
                modality=ref.modality,
                media_type=ref.media_type,
                citation=ref.citation,
                local_type=ref.local_type,
                content_hash=ref.content_hash,
                path_ref=ref.path_ref,
                cache_state=ref.cache_state,
                processing_state='metadata_only' if ref.cache_state == 'metadata_only' else 'pending',
                metadata=(ref.metadata or {}) | {'path_hint_present': bool(ref.path_hint)},
            ))
            decision = classify_media_reference(self._classification_source_type(ref), ref.citation)
            if decision.allowed:
                accepted_links += 1
            else:
                excluded_links += 1
                excluded_counts[decision.scope_type] = excluded_counts.get(decision.scope_type, 0) + 1
            link_id = _stable('mlink', f'{ref.asset_id}:{ref.citation}:{ref.source_type}')
            links.append(MediaAssetLinkRecord(
                link_id=link_id,
                asset_id=ref.asset_id,
                account_id=ref.account_id,
                source_type=ref.source_type,
                source_citation=ref.citation,
                scope_type=decision.scope_type,
                accepted=decision.allowed,
                reason=decision.reason,
                metadata={'raw_kind': decision.raw_kind, 'confidence': decision.confidence},
            ))
        bulk = self.repo.upsert_media_graph(assets, links, source_states=source_states)
        return MediaLinkResult(
            assets_seen,
            int(bulk['assets_upserted']),
            int(bulk['links_upserted']),
            accepted_links,
            excluded_links,
            excluded_counts,
            tuple(bulk['changed_asset_ids']),
            dict(bulk['metrics']),
        )
