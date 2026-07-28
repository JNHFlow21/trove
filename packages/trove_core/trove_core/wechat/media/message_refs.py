from __future__ import annotations

import re

from trove_core.domain.messages import Message

from .resources import MediaReference, message_media_asset_id


MEDIA_RESOURCE_HINT_RE = re.compile(
    r'(?i)(?:^|[\s<>"\'])([^<>"\']+\.(?:amr|silk|m4a|mp3|wav|jpg|jpeg|png|gif|webp|heic|dat|mp4|mov))(?:[?\s<>"\']|$)'
)


def _message_path_hint(message: Message) -> str | None:
    text = str(message.content or '').strip()
    if not text or text.startswith('['):
        return None
    match = MEDIA_RESOURCE_HINT_RE.search(text)
    return match.group(1)[:512] if match else None


def _message_media_shape(message: Message) -> tuple[str, str] | None:
    if message.content_kind == 'voice':
        return 'voice', 'voice'
    if message.content_kind == 'image':
        return 'image', 'image'
    if message.content_kind == 'video':
        return 'video', 'video'
    if message.content_kind == 'sticker':
        return 'image', 'image'
    payload = message.normalized_payload or {}
    if message.content_kind == 'appmsg' and payload.get('normalized_type') == 'file':
        return 'file', 'document'
    return None


def message_media_references_for_messages(messages: list[Message]) -> list[MediaReference]:
    refs: list[MediaReference] = []
    for message in messages:
        shape = _message_media_shape(message)
        if shape is None:
            continue
        modality, media_type = shape
        citation = message.citation
        path_hint = None if modality == 'file' else _message_path_hint(message)
        payload_fields = (message.normalized_payload or {}).get('fields') or {}
        file_metadata = {
            key: payload_fields[key]
            for key in ('file_name', 'file_extension', 'file_size')
            if key in payload_fields
        }
        refs.append(MediaReference(
            asset_id=message_media_asset_id(citation, modality, media_type),
            account_id=message.account_id,
            source_type='private_chat' if message.conversation_type == 'private' else 'group_chat',
            source_id=citation,
            modality=modality,
            media_type=media_type,
            citation=citation,
            local_type=message.content_kind,
            path_hint=path_hint,
            cache_state='missing_local_cache' if path_hint else 'metadata_only',
            metadata={
                'registration': 'message_media',
                'message_citation': citation,
                'conversation_id': message.conversation_id,
                'conversation_type': message.conversation_type,
                'message_shard_id': message.shard_id,
                'message_local_id': message.local_id,
                'content_kind': message.content_kind,
            } | file_metadata,
        ))
    return refs


def voice_media_references_for_messages(messages: list[Message]) -> list[MediaReference]:
    return [ref for ref in message_media_references_for_messages(messages) if ref.modality == 'voice']


__all__ = ['message_media_references_for_messages', 'voice_media_references_for_messages']
