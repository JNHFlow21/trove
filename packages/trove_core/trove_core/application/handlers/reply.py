from __future__ import annotations

from typing import Any, Mapping

from .base import HandlerOutcome


def status(owner: Any, _payload: Mapping[str, Any]) -> HandlerOutcome:
    if owner is None or not callable(getattr(owner, 'reply_status', None)):
        return HandlerOutcome.failure(
            'capability_unavailable',
            'Reply runtime state is unavailable.',
        )
    return HandlerOutcome.success({'reply': owner.reply_status()})


def reviews(owner: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    if owner is None or not callable(getattr(owner, 'reply_reviews', None)):
        return HandlerOutcome.failure(
            'capability_unavailable',
            'Reply review state is unavailable.',
        )
    return HandlerOutcome.success({
        'reviews': owner.reply_reviews(
            state=str(payload.get('state') or 'pending'),
            limit=int(payload.get('limit') or 100),
        ),
    })


def activity(owner: Any, payload: Mapping[str, Any]) -> HandlerOutcome:
    if owner is None or not callable(getattr(owner, 'reply_activity', None)):
        return HandlerOutcome.failure(
            'capability_unavailable',
            'Reply activity is unavailable.',
        )
    return HandlerOutcome.success({
        'activity': owner.reply_activity(
            limit=int(payload.get('limit') or 100),
        ),
    })


__all__ = ['activity', 'reviews', 'status']
