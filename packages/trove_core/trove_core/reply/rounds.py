from __future__ import annotations

from dataclasses import replace
import secrets

from .models import ReplyEvent, RoundRecord, RoundTiming
from .store import ReplyStore


_MEDIA_KINDS = frozenset({'image', 'sticker', 'voice', 'video', 'file'})


def adaptive_quiet_ms(
    inbound_message_count: int,
    latest_kind: str,
    timing: RoundTiming,
) -> int:
    """Choose a deterministic bounded quiet window without inspecting text."""

    count = max(1, int(inbound_message_count))
    span = timing.quiet_max_ms - timing.quiet_default_ms
    if count >= 5:
        target = timing.quiet_max_ms
    elif count >= 3:
        target = timing.quiet_default_ms + (span * 2 // 3)
    elif count >= 2:
        target = timing.quiet_default_ms + (span // 3)
    else:
        target = timing.quiet_default_ms
    if latest_kind in _MEDIA_KINDS:
        target = max(target, timing.quiet_default_ms + (span * 2 // 3))
    return max(timing.quiet_min_ms, min(timing.quiet_max_ms, target))


class RoundCoordinator:
    """Durable adaptive collection and fairness for source-neutral reply events."""

    def __init__(
        self,
        store: ReplyStore,
        *,
        timing: RoundTiming | None = None,
    ) -> None:
        self.store = store
        self.timing = timing or RoundTiming()

    def observe(self, event: ReplyEvent, *, now: float) -> RoundRecord:
        count = min(len(event.messages), self.timing.max_messages)
        quiet_ms = adaptive_quiet_ms(count, event.latest_kind, self.timing)
        prestart = self.timing.generation_prestart_ms / 1_000.0
        minimum = self.timing.quiet_min_ms / 1_000.0
        quiet = quiet_ms / 1_000.0
        maximum = self.timing.max_collect_ms / 1_000.0

        with self.store.transaction() as conn:
            existing = self.store.find_round(
                event.account_id,
                event.conversation_id,
                connection=conn,
            )
            if existing is not None and event.source_position <= existing.source_position:
                return existing
            if existing is None:
                record = RoundRecord(
                    round_id='round_' + secrets.token_urlsafe(18),
                    account_id=event.account_id,
                    conversation_id=event.conversation_id,
                    target_ref=event.target_ref,
                    first_seen_at=now,
                    last_extended_at=now,
                    preparation_at=now + prestart,
                    earliest_ready_at=now + minimum,
                    ready_at=min(now + maximum, now + quiet),
                    deadline_at=now + maximum,
                    quiet_target_ms=quiet_ms,
                    source_position=event.source_position,
                    latest_fingerprint=event.latest_fingerprint,
                    inbound_message_count=count,
                    latest_kind=event.latest_kind,
                    revision=1,
                )
            else:
                record = replace(
                    existing,
                    target_ref=event.target_ref,
                    last_extended_at=now,
                    preparation_at=min(existing.deadline_at, now + prestart),
                    earliest_ready_at=now + minimum,
                    ready_at=min(existing.deadline_at, now + quiet),
                    quiet_target_ms=quiet_ms,
                    source_position=event.source_position,
                    latest_fingerprint=event.latest_fingerprint,
                    inbound_message_count=count,
                    latest_kind=event.latest_kind,
                    revision=existing.revision + 1,
                    attempts=0,
                    not_before=0.0,
                    last_error='',
                    blocked=False,
                )
                self.store.invalidate_superseded(
                    existing.round_id,
                    current_revision=record.revision,
                    now=now,
                    connection=conn,
                )
            saved = self.store.save_round(record, now=now, connection=conn)
            self.store.save_event(
                event,
                round_id=saved.round_id,
                round_revision=saved.revision,
                now=now,
                connection=conn,
            )
            return saved

    def ready(self, *, now: float) -> tuple[RoundRecord, ...]:
        return tuple(sorted(
            (item for item in self.store.list_rounds() if item.ready(now)),
            key=lambda item: (
                max(item.earliest_ready_at, item.ready_at, item.not_before),
                item.first_seen_at,
            ),
        ))

    def preparable(self, *, now: float) -> tuple[RoundRecord, ...]:
        return tuple(sorted(
            (item for item in self.store.list_rounds() if item.preparable(now)),
            key=lambda item: (
                max(item.preparation_at, item.not_before),
                item.first_seen_at,
            ),
        ))

    def fail(
        self,
        round_id: str,
        *,
        reason: str,
        retryable: bool,
        now: float,
    ) -> RoundRecord:
        current = self.store.get_round(round_id)
        attempts = current.attempts + 1
        blocked = not retryable or attempts >= 3
        not_before = (
            0.0
            if blocked
            else now + min(30.0, 0.5 * (2 ** min(attempts - 1, 6)))
        )
        updated = replace(
            current,
            attempts=attempts,
            not_before=not_before,
            last_error=str(reason)[:160],
            blocked=blocked,
        )
        return self.store.save_round(updated, now=now)

    def recover_generation_failures(self, *, now: float) -> int:
        recovered = 0
        for item in self.store.list_rounds():
            if not item.blocked or not item.last_error.startswith('generation_'):
                continue
            self.store.save_round(
                replace(
                    item,
                    attempts=0,
                    not_before=0.0,
                    last_error='',
                    blocked=False,
                ),
                now=now,
            )
            recovered += 1
        return recovered


__all__ = ['RoundCoordinator', 'adaptive_quiet_ms']
