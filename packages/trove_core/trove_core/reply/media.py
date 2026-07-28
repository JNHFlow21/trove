from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from trove_core.media_fetch import fetch_media
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig

_MEDIA_KINDS = frozenset({
    'image', 'sticker', 'voice', 'video', 'file', 'attachment', 'document',
})
_FILE_MODALITIES = frozenset({'file', 'attachment', 'document'})
_CLOUD_ASR_PROVIDER = 'volcengine-asr-flash'
_CLOUD_ASR_MODEL = 'bigmodel:volc.bigasr.auc_turbo'


def _json_list(value: Any, *, limit: int = 20) -> list[Any]:
    if isinstance(value, list):
        return value[:limit]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed[:limit] if isinstance(parsed, list) else []


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _bounded_understanding(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        'caption': str(value.get('caption') or '')[:4_000],
        'visible_text': str(value.get('visible_text') or '')[:8_000],
        'audio_transcript': str(value.get('audio_transcript') or '')[:8_000],
        'objects': _json_list(
            value.get('objects', value.get('objects_json')),
        ),
        'business_signals': _json_list(
            value.get('business_signals', value.get('business_signals_json')),
        ),
        'keyframes': _json_list(
            value.get('keyframes', value.get('keyframes_json')),
        ),
        'model_id': str(value.get('model_id') or '')[:200],
        'prompt_version': str(value.get('prompt_version') or '')[:200],
        'confidence': _confidence(value.get('confidence')),
    }
    return {
        key: item
        for key, item in payload.items()
        if item is not None and item != '' and item != []
    }


def _hint_items(hint: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nested = hint.get('items')
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, Mapping)]
    return [hint]


class ReplyMediaResolver:
    """Resolve bounded reply media without leaking source paths to the model."""

    def __init__(
        self,
        config: VaultConfig,
        *,
        workspace: Any | None = None,
        max_items: int = 8,
        max_attachments: int = 3,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.max_items = max(1, min(int(max_items), 12))
        self.max_attachments = max(0, min(int(max_attachments), 4))
        self.store = SQLiteStore(config.paths.sqlite_path)

    def resolve(
        self,
        messages: Iterable[Any],
        *,
        new_message_citations: Iterable[str],
        hints: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        rows = list(messages)
        new = set(str(item) for item in new_message_citations)
        ordered = sorted(
            rows,
            key=lambda item: (
                item.citation not in new,
                -item.source_position,
                item.citation,
            ),
        )
        candidates: list[tuple[str, str, Mapping[str, Any] | None]] = []
        consumed_hint_citations: set[str] = set()
        for message in ordered:
            hint = hints.get(message.citation)
            if isinstance(hint, Mapping):
                consumed_hint_citations.add(message.citation)
                for item in _hint_items(hint):
                    candidates.append((message.citation, message.kind, item))
            elif message.kind in _MEDIA_KINDS:
                candidates.append((message.citation, message.kind, None))
        for citation in sorted(hints):
            if citation in consumed_hint_citations:
                continue
            hint = hints[citation]
            if not isinstance(hint, Mapping):
                continue
            for item in _hint_items(hint):
                candidates.append((citation, str(item.get('modality') or 'media'), item))

        output: list[Mapping[str, Any]] = []
        attachment_count = 0
        seen: set[tuple[str, str]] = set()
        for message_citation, message_kind, hint in candidates:
            if len(output) >= self.max_items:
                break
            asset_id = str((hint or {}).get('asset_id') or '')
            key = (message_citation, asset_id)
            if key in seen:
                continue
            seen.add(key)
            allow_attachment = (
                attachment_count < self.max_attachments
                and message_citation in new
            )
            item = self._resolve_one(
                message_citation,
                message_kind,
                hint,
                allow_attachment=allow_attachment,
            )
            if item.get('attachment'):
                attachment_count += 1
            output.append(item)
        return tuple(output)

    def _resolve_one(
        self,
        message_citation: str,
        message_kind: str,
        hint: Mapping[str, Any] | None,
        *,
        allow_attachment: bool,
    ) -> Mapping[str, Any]:
        if hint is None:
            return {
                'message_citation': message_citation,
                'citation': message_citation,
                'modality': message_kind,
                'state': 'pending_index',
                'reason': 'live_media_not_yet_resolved_in_vault',
                'reply_policy': 'do_not_infer_content',
                'trust': 'untrusted_evidence',
                'raw_paths_included': False,
            }
        citation = str(hint.get('citation') or message_citation)
        asset_id = str(hint.get('asset_id') or '')
        modality = str(hint.get('modality') or message_kind or 'media')
        if message_kind == 'sticker' and modality == 'image':
            modality = 'sticker'
        asset, understanding, observation, transcript = self._cached(
            asset_id,
        )
        if modality in _FILE_MODALITIES:
            metadata = _json_object(asset.get('metadata_json') if asset else None)
            return {
                'message_citation': message_citation,
                'citation': citation,
                'modality': modality,
                'state': 'metadata_only',
                'metadata': {
                    key: metadata[key]
                    for key in ('file_name', 'file_extension', 'file_size')
                    if key in metadata
                },
                'document_extract': 'disabled',
                'reply_policy': 'do_not_infer_file_contents',
                'trust': 'untrusted_evidence',
                'raw_paths_included': False,
            }
        if modality == 'voice':
            if transcript:
                return {
                    'message_citation': message_citation,
                    'citation': citation,
                    'modality': 'voice',
                    'state': 'understood',
                    'understanding': {
                        'audio_transcript': str(transcript.get('text') or '')[:8_000],
                        'language': str(transcript.get('language') or '')[:50],
                        'confidence': _confidence(
                            transcript.get('confidence'),
                        ),
                        'provider': _CLOUD_ASR_PROVIDER,
                        'model': _CLOUD_ASR_MODEL,
                    },
                    'trust': 'untrusted_derived_evidence',
                    'raw_paths_included': False,
                }
            available = bool(hint.get('available'))
            return {
                'message_citation': message_citation,
                'citation': citation,
                'modality': 'voice',
                'state': (
                    'pending_cloud_approval'
                    if available
                    else 'unavailable'
                ),
                'reason': (
                    'cloud_asr_approval_required'
                    if available
                    else 'voice_bytes_unavailable'
                ),
                'asr_policy': 'cloud_only',
                'local_asr_used': False,
                'reply_policy': 'do_not_infer_content',
                'trust': 'untrusted_evidence',
                'raw_paths_included': False,
            }

        cached = (
            _bounded_understanding(understanding)
            if understanding
            else _bounded_understanding(observation)
            if observation
            else {}
        )
        attachment: Mapping[str, Any] | None = None
        fetch_state = ''
        if allow_attachment and modality in {'image', 'sticker', 'video'}:
            attachment, fetched_understanding, fetch_state = self._attachment(
                citation,
                modality=modality,
            )
            if not cached and fetched_understanding:
                cached = _bounded_understanding(fetched_understanding)
        if cached:
            state = 'understood'
        elif attachment is not None:
            state = 'attached'
        elif bool(hint.get('available')):
            state = 'pending'
        else:
            state = 'unavailable'
        result: dict[str, Any] = {
            'message_citation': message_citation,
            'citation': citation,
            'modality': modality,
            'state': state,
            'reply_policy': (
                'use_attached_or_cached_evidence'
                if state in {'understood', 'attached'}
                else 'do_not_infer_content'
            ),
            'trust': (
                'untrusted_derived_evidence'
                if cached
                else 'untrusted_evidence'
            ),
            'raw_paths_included': False,
        }
        if cached:
            result['understanding'] = cached
        if attachment is not None:
            result['attachment'] = dict(attachment)
        if state in {'pending', 'unavailable'}:
            result['reason'] = fetch_state or (
                'media_understanding_pending'
                if state == 'pending'
                else 'media_bytes_unavailable'
            )
        return result

    def _cached(
        self,
        asset_id: str,
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        if not asset_id or not self.store.path.is_file():
            return None, None, None, None
        with self.store.connect() as conn:
            asset_row = conn.execute(
                'SELECT * FROM media_assets WHERE asset_id=? LIMIT 2',
                (asset_id,),
            ).fetchall()
            if len(asset_row) != 1:
                return None, None, None, None
            asset = dict(asset_row[0])
            content_hash = str(asset.get('content_hash') or '')
            understanding = None
            if content_hash and self.store._table_exists(
                conn, 'media_understanding',
            ):
                row = conn.execute(
                    """SELECT * FROM media_understanding
                        WHERE content_sha256=? AND status='active' LIMIT 1""",
                    (content_hash,),
                ).fetchone()
                understanding = dict(row) if row is not None else None
            observation = None
            if self.store._table_exists(conn, 'image_observations'):
                row = conn.execute(
                    """SELECT * FROM image_observations
                        WHERE asset_id=?
                          AND status NOT IN ('stale','rejected')
                          AND (
                            TRIM(COALESCE(caption,''))<>''
                            OR TRIM(COALESCE(visible_text,''))<>''
                          )
                        ORDER BY confidence DESC,updated_at DESC LIMIT 1""",
                    (asset_id,),
                ).fetchone()
                observation = dict(row) if row is not None else None
            transcript = None
            if (
                content_hash
                and self.store._table_exists(conn, 'transcripts')
                and self.store._table_exists(conn, 'provider_jobs')
            ):
                row = conn.execute(
                    """SELECT t.*
                         FROM transcripts t
                         JOIN provider_jobs p ON p.job_id=t.job_id
                        WHERE t.asset_id=? AND t.status='active'
                          AND p.provider=? AND p.model=?
                          AND p.status='completed' AND p.request_hash=?
                        ORDER BY t.created_at DESC LIMIT 1""",
                    (
                        asset_id,
                        _CLOUD_ASR_PROVIDER,
                        _CLOUD_ASR_MODEL,
                        content_hash,
                    ),
                ).fetchone()
                transcript = dict(row) if row is not None else None
        return asset, understanding, observation, transcript

    def _attachment(
        self,
        citation: str,
        *,
        modality: str,
    ) -> tuple[
        Mapping[str, Any] | None,
        Mapping[str, Any] | None,
        str,
    ]:
        if self.workspace is None or not hasattr(
            self.workspace, 'stage_media_evidence',
        ):
            return None, None, 'reply_workspace_media_unavailable'
        try:
            fetched = fetch_media(
                self.config.root,
                citation=citation,
                allow_remote=False,
                materialize_preview=True,
            )
        except Exception as exc:
            return None, None, 'media_fetch_' + type(exc).__name__
        if not fetched.get('ok'):
            return None, None, str(
                fetched.get('reason') or fetched.get('code') or 'media_unavailable',
            )[:160]
        understanding = fetched.get('understanding')
        candidates: list[Path] = []
        if modality in {'image', 'sticker'} and fetched.get('path'):
            candidates.append(Path(str(fetched['path'])))
        elif modality == 'video':
            preview = fetched.get('preview')
            if isinstance(preview, Mapping):
                candidates.extend(
                    Path(str(item))
                    for item in (preview.get('keyframe_paths') or [])[:1]
                    if item
                )
        for path in candidates:
            try:
                attachment = self.workspace.stage_media_evidence(path)
            except Exception as exc:
                return None, (
                    understanding if isinstance(understanding, Mapping) else None
                ), 'media_stage_' + type(exc).__name__
            return attachment, (
                understanding if isinstance(understanding, Mapping) else None
            ), ''
        return None, (
            understanding if isinstance(understanding, Mapping) else None
        ), 'media_preview_unavailable'


__all__ = ['ReplyMediaResolver']
