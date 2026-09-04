from __future__ import annotations

from contextlib import closing

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping

from .generation import DraftGenerationCoordinator, ReplyGenerationError
from .models import (
    EvidenceMessage,
    ReplyDraft,
    ReplyEvent,
    RoundTiming,
    SendIntent,
)
from .rounds import RoundCoordinator
from .store import ReplyStore, ReplyStoreConflict, ReplyStoreNotFound


_DIGEST = re.compile(r'^[0-9a-f]{64}$')
ActionInvoker = Callable[[Mapping[str, Any]], Mapping[str, Any]]
_RETRYABLE_PRE_SEND_STAGES = (
    'sender_busy_timeout',
    'foreground_activation_failed',
    'foreground_focus_lost_before_navigate',
    'search_verification_failed',
    'foreground_focus_lost_after_navigate',
    'draft_verification_failed',
    'foreground_focus_lost_before_send',
    'resolve_client_',
    'driver_health_',
    'activate_foreground_',
    'locate_window_',
    'search_target_',
    'verify_search_',
    'navigate_',
    'verify_draft_',
    'pre_send_focus_check_',
)


class ReplyServiceError(RuntimeError):
    code = 'reply_service_unavailable'


def _private_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | int(getattr(os, 'O_NOFOLLOW', 0)),
            0o600,
        )
        encoded = (
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(',', ':'),
            )
            + '\n'
        ).encode('utf-8')
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError('short reply config write')
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ReplyServiceConfig:
    schema_version: int = 1
    armed: bool = False
    mode: str = 'review_queue'
    provider_id: str = 'wechat-source'
    account_id: str = ''
    source_account_id: str = ''
    account_id_sha256: str = ''
    conversation_namespace: str = ''
    key_store_secret: str = 'TROVE_WECHAT_KEY_STORE'
    poll_seconds: float = 1.0
    send_shortcut: str = 'unconfigured'
    max_reply_chars: int = 500
    cooldown_seconds: float = 15.0
    daily_send_limit: int = 300
    target_scope: str = 'allowlist'
    allowed_target_refs: tuple[str, ...] = ()
    agent_id: str = 'default-reply-agent'
    reply_backend: str = 'codex'
    model: str = 'gpt-5.6-terra'
    api_base_url: str = ''
    api_key_secret: str = ''
    style_profile_path: str = 'profile/style.md'
    session_idle_days: float = 3.0
    context_message_cap: int = 50
    generation_prestart_ms: int = 3_000
    round_quiet_min_ms: int = 6_000
    round_quiet_default_ms: int = 8_000
    round_quiet_max_ms: int = 15_000
    round_max_collect_ms: int = 60_000
    round_max_messages: int = 50

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ReplyServiceError('unsupported reply service config version')
        if type(self.armed) is not bool:
            raise ReplyServiceError('reply armed state must be boolean')
        if self.mode not in {'shadow', 'review_queue', 'live'}:
            raise ReplyServiceError('unsupported reply mode')
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ReplyServiceError('reply provider id is required')
        configured = bool(
            self.account_id
            or self.source_account_id
            or self.account_id_sha256
            or self.conversation_namespace
        )
        if configured:
            if (
                not self.account_id
                or not self.source_account_id
                or not self.conversation_namespace
                or _DIGEST.fullmatch(self.account_id_sha256) is None
                or hashlib.sha256(
                    self.source_account_id.encode('utf-8'),
                ).hexdigest()
                != self.account_id_sha256
                or self.account_id
                != (
                    'acct-'
                    + hashlib.sha256(
                        self.conversation_namespace.encode('utf-8'),
                    ).hexdigest()[:12]
                )
            ):
                raise ReplyServiceError('reply account binding is invalid')
        if self.armed and not configured:
            raise ReplyServiceError('reply account binding is required before arming')
        if (
            not isinstance(self.key_store_secret, str)
            or not self.key_store_secret
            or any(char.isspace() for char in self.key_store_secret)
        ):
            raise ReplyServiceError('reply key store secret name is invalid')
        if not 0.25 <= float(self.poll_seconds) <= 30:
            raise ReplyServiceError('reply polling interval is invalid')
        if self.send_shortcut not in {
            'unconfigured', 'return', 'command_return',
        }:
            raise ReplyServiceError('reply send shortcut is invalid')
        if self.armed and self.send_shortcut == 'unconfigured':
            raise ReplyServiceError('reply send shortcut is required before arming')
        if type(self.max_reply_chars) is not int or not 1 <= self.max_reply_chars <= 2_000:
            raise ReplyServiceError('reply length bound is invalid')
        if not 0 <= float(self.cooldown_seconds) <= 3_600:
            raise ReplyServiceError('reply cooldown bound is invalid')
        if (
            type(self.daily_send_limit) is not int
            or not 1 <= self.daily_send_limit <= 10_000
        ):
            raise ReplyServiceError('reply daily send limit is invalid')
        if self.target_scope not in {'allowlist', 'all_private_except_official'}:
            raise ReplyServiceError('reply target scope is invalid')
        if (
            any(_DIGEST.fullmatch(value) is None for value in self.allowed_target_refs)
            or len(set(self.allowed_target_refs)) != len(self.allowed_target_refs)
        ):
            raise ReplyServiceError('reply target allowlist is invalid')
        if self.target_scope == 'allowlist' and self.armed and not self.allowed_target_refs:
            raise ReplyServiceError('reply allowlist is empty')
        if re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', self.agent_id) is None:
            raise ReplyServiceError('reply agent id is invalid')
        if self.reply_backend not in {'codex', 'api'}:
            raise ReplyServiceError('reply backend is invalid')
        if not isinstance(self.model, str) or not self.model:
            raise ReplyServiceError('reply model is required')
        if self.reply_backend == 'api' and (
            not self.api_base_url or not self.api_key_secret
        ):
            raise ReplyServiceError('reply API configuration is incomplete')
        relative = Path(self.style_profile_path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ReplyServiceError('reply style path must be workspace-relative')
        if not 0.1 <= float(self.session_idle_days) <= 30:
            raise ReplyServiceError('reply session idle bound is invalid')
        if type(self.context_message_cap) is not int or not 1 <= self.context_message_cap <= 200:
            raise ReplyServiceError('reply context bound is invalid')
        RoundTiming(
            generation_prestart_ms=self.generation_prestart_ms,
            quiet_min_ms=self.round_quiet_min_ms,
            quiet_default_ms=self.round_quiet_default_ms,
            quiet_max_ms=self.round_quiet_max_ms,
            max_collect_ms=self.round_max_collect_ms,
            max_messages=self.round_max_messages,
        )

    @classmethod
    def path_for_vault(cls, vault_root: str | Path) -> Path:
        return Path(vault_root) / 'jobs' / 'reply' / 'config.json'

    @classmethod
    def load(cls, vault_root: str | Path) -> 'ReplyServiceConfig':
        path = cls.path_for_vault(vault_root)
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReplyServiceError('reply config is unreadable') from exc
        if not isinstance(payload, dict):
            raise ReplyServiceError('reply config must be an object')
        values = {
            name: payload[name]
            for name in cls.__dataclass_fields__
            if name in payload
        }
        if 'allowed_target_refs' in values:
            raw = values['allowed_target_refs']
            if not isinstance(raw, list):
                raise ReplyServiceError('reply allowlist must be an array')
            values['allowed_target_refs'] = tuple(str(item) for item in raw)
        try:
            return cls(**values)
        except TypeError as exc:
            raise ReplyServiceError('reply config fields are invalid') from exc

    @property
    def configured(self) -> bool:
        return bool(
            self.account_id
            and self.source_account_id
            and self.account_id_sha256
            and self.conversation_namespace
        )

    @property
    def timing(self) -> RoundTiming:
        return RoundTiming(
            generation_prestart_ms=self.generation_prestart_ms,
            quiet_min_ms=self.round_quiet_min_ms,
            quiet_default_ms=self.round_quiet_default_ms,
            quiet_max_ms=self.round_quiet_max_ms,
            max_collect_ms=self.round_max_collect_ms,
            max_messages=self.round_max_messages,
        )

    def save(self, vault_root: str | Path) -> None:
        _private_write_json(
            self.path_for_vault(vault_root),
            {
                **asdict(self),
                'allowed_target_refs': list(self.allowed_target_refs),
            },
        )

    def redacted(self) -> dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'armed': self.armed,
            'configured': self.configured,
            'mode': self.mode,
            'provider_id': self.provider_id,
            'account_ref': (
                hashlib.sha256(
                    self.account_id.encode('utf-8'),
                ).hexdigest()[:12] + '...'
                if self.account_id
                else None
            ),
            'poll_seconds': self.poll_seconds,
            'send_shortcut_configured': self.send_shortcut != 'unconfigured',
            'max_reply_chars': self.max_reply_chars,
            'cooldown_seconds': self.cooldown_seconds,
            'daily_send_limit': self.daily_send_limit,
            'target_scope': self.target_scope,
            'allowed_target_count': len(self.allowed_target_refs),
            'agent_id': self.agent_id,
            'reply_backend': self.reply_backend,
            'model': self.model,
            'style_profile_configured': bool(self.style_profile_path),
            'context_message_cap': self.context_message_cap,
            'timing': {
                'generation_prestart_ms': self.generation_prestart_ms,
                'quiet_min_ms': self.round_quiet_min_ms,
                'quiet_default_ms': self.round_quiet_default_ms,
                'quiet_max_ms': self.round_quiet_max_ms,
                'max_collect_ms': self.round_max_collect_ms,
                'max_messages': self.round_max_messages,
            },
            'secret_names': [
                self.key_store_secret,
                *(
                    [self.api_key_secret]
                    if self.reply_backend == 'api' and self.api_key_secret
                    else []
                ),
            ],
            'secret_values_included': False,
        }


class ReplyService:
    """One daemon-owned reply loop with durable evidence and send recovery."""

    def __init__(
        self,
        vault_root: str | Path,
        config: ReplyServiceConfig,
        *,
        action: ActionInvoker,
        generation: DraftGenerationCoordinator,
        store: ReplyStore | None = None,
        now: Callable[[], float] = time.time,
        generation_workers: int = 2,
    ) -> None:
        self.vault_root = Path(vault_root)
        self.config = config
        self.action = action
        self.store = store or ReplyStore.for_vault(self.vault_root)
        self.rounds = RoundCoordinator(self.store, timing=config.timing)
        self.generation = generation
        self.now = now
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(int(generation_workers), 4)),
            thread_name_prefix='trove-reply-generation',
        )
        self._generating: dict[str, tuple[int, Future[ReplyDraft]]] = {}
        self._last_poll_at: float | None = None
        self._last_error = ''
        self._closed = False

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    @property
    def keepalive_required(self) -> bool:
        return self.config.armed

    def start(self) -> bool:
        with self._lock:
            if self._closed:
                raise ReplyServiceError('reply service is closed')
            if not self.config.armed:
                return False
            if self.running:
                return False
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name='trove-reply-service',
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, *, timeout: float = 5.0) -> bool:
        with self._lock:
            thread = self._thread
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        with self._lock:
            if thread is self._thread and not self.running:
                self._thread = None
        return not self.running

    def close(self) -> None:
        if self._closed:
            return
        self.stop(timeout=5.0)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._closed = True

    def arm(self) -> dict[str, Any]:
        with self._lock:
            updated = replace(self.config, armed=True)
            updated.save(self.vault_root)
            self.config = updated
        self.start()
        return self.status()

    def disarm(self) -> dict[str, Any]:
        with self._lock:
            updated = replace(self.config, armed=False)
            updated.save(self.vault_root)
            self.config = updated
        self.stop(timeout=5.0)
        return self.status()

    def set_mode(self, mode: str) -> dict[str, Any]:
        if mode not in {'shadow', 'review_queue', 'live'}:
            raise ReplyServiceError('unsupported reply mode')
        with self._lock:
            if mode == self.config.mode:
                return self.status()
            if self.config.armed or self.running:
                raise ReplyServiceError(
                    'reply service must be stopped before changing mode'
                )
            pending = self.store.list_reviews(state='pending', limit=1)
            unresolved = self.store.list_sends(
                states=('prepared', 'dispatched', 'reconciling'),
                limit=1,
            )
            if pending or unresolved:
                raise ReplyServiceError(
                    'reply queue must be resolved before changing mode'
                )
            previous = self.config.mode
            updated = replace(self.config, mode=mode)
            updated.save(self.vault_root)
            self.config = updated
            self.store.add_activity(
                'mode_changed',
                state=f'{previous}_to_{mode}',
                now=float(self.now()),
            )
        return self.status()

    def decide_review(self, review_id: str, decision: str) -> dict[str, Any]:
        record = self.store.decide_review(
            review_id, decision=decision, now=self.now(),
        )
        draft = self.store.get_draft(record.draft_id)
        self.store.add_activity(
            'review_decided',
            state=decision,
            now=float(record.decided_at or self.now()),
            target_ref=draft.target_ref,
            conversation_id=draft.conversation_id,
            text=draft.text,
        )
        return self._review_payload(record, draft)

    @staticmethod
    def _send_retryable(operation: Any) -> bool:
        return bool(
            operation is not None
            and operation.state == 'failed'
            and operation.retry_count < 3
            and any(
                operation.stage == prefix
                or operation.stage.startswith(prefix)
                for prefix in _RETRYABLE_PRE_SEND_STAGES
            )
        )

    def _send_for_draft(self, draft: ReplyDraft) -> Any | None:
        operation_id, _idempotency_key = self._operation_identity(draft)
        try:
            return self.store.get_send(operation_id)
        except ReplyStoreNotFound:
            return None

    @staticmethod
    def _authorized_retry_dispatch(
        operation: Any | None,
        draft: ReplyDraft,
    ) -> bool:
        return bool(
            operation is not None
            and operation.draft_id == draft.draft_id
            and operation.state == 'prepared'
            and operation.stage == 'retry_authorized'
            and operation.retry_count > 0
            and draft.state == 'approved'
        )

    def retry_review(self, review_id: str) -> dict[str, Any]:
        review = self.store.get_review(review_id)
        draft = self.store.get_draft(review.draft_id)
        round_record = self.store.get_round(draft.round_id)
        operation = self._send_for_draft(draft)
        if (
            review.state != 'approved'
            or draft.state != 'approved'
            or draft.round_revision != round_record.revision
            or draft.source_position != round_record.source_position
            or not self._send_retryable(operation)
        ):
            raise ReplyStoreConflict('review send is not retryable')
        assert operation is not None
        intent = self.store.get_send_intent(operation.operation_id)
        preflight = self.action(
            self._send_payload(intent, 'retry_preflight')
        )
        if (
            preflight.get('state') != 'ready'
            or preflight.get('stage') != 'retry_preflight_passed'
            or preflight.get('operation_id') != intent.operation_id
            or preflight.get('idempotency_key') != intent.idempotency_key
            or preflight.get('target_ref') != draft.target_ref
            or preflight.get('expected_source_position')
            != intent.expected_source_position
            or preflight.get('draft_digest') != intent.draft_digest
        ):
            raise ReplyStoreConflict('provider retry preflight did not pass')
        now = float(self.now())
        reopened = self.store.reopen_failed_send(
            operation.operation_id,
            review_id=review.review_id,
            expected_stage=operation.stage,
            now=now,
            maximum_retries=3,
        )
        self.store.add_activity(
            'send_retry_authorized',
            state='prepared',
            now=now,
            target_ref=draft.target_ref,
            conversation_id=draft.conversation_id,
            text=draft.text,
        )
        payload = self._review_payload(review, draft)
        payload['send'] = {
            'state': reopened.state,
            'stage': reopened.stage,
            'retryable': False,
            'retry_count': reopened.retry_count,
        }
        return payload

    def status(self) -> dict[str, Any]:
        pending = self.store.list_reviews(state='pending', limit=1_000)
        unresolved = self.store.list_sends(
            states=('prepared', 'dispatched', 'reconciling'),
            limit=1_000,
        )
        provider: Mapping[str, Any] | None = None
        if self.config.configured:
            try:
                provider = self.action({'operation': 'status'})
            except Exception:
                provider = {
                    'state': 'blocked',
                    'ready': False,
                    'reason': 'provider_status_unavailable',
                }
        return {
            'state': (
                'running'
                if self.running and self.config.armed
                else 'stopped'
            ),
            'running': self.running,
            'armed': self.config.armed,
            'config': self.config.redacted(),
            'pending_reviews': len(pending),
            'unresolved_sends': len(unresolved),
            'last_poll_at': self._last_poll_at,
            'last_error': self._last_error or None,
            'provider': dict(provider or {}),
        }

    def reviews(
        self,
        *,
        state: str = 'pending',
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        results = []
        for review, draft in self.store.list_reviews(
            state=state, limit=limit,
        ):
            payload = self._review_payload(review, draft)
            operation = self._send_for_draft(draft)
            if operation is not None:
                payload['send'] = {
                    'state': operation.state,
                    'stage': operation.stage,
                    'retryable': self._send_retryable(operation),
                    'retry_count': operation.retry_count,
                }
            label = self._conversation_label(
                draft.account_id, draft.conversation_id,
            )
            if label:
                payload['draft']['display_name'] = label
            results.append(payload)
        return results

    def activity(self, *, limit: int = 100) -> list[dict[str, Any]]:
        results = []
        for raw in self.store.list_activity(limit=limit):
            item = dict(raw)
            if (
                not item.get('display_label')
                and item.get('conversation_id')
            ):
                item['display_label'] = self._conversation_label(
                    None, str(item['conversation_id']),
                )
            results.append(item)
        return results

    def _conversation_label(
        self,
        account_id: str | None,
        conversation_id: str,
    ) -> str | None:
        database = self.vault_root / 'index' / 'trove.sqlite'
        if not database.is_file():
            return None
        try:
            uri = database.resolve().as_uri() + '?mode=ro'
            with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
                if account_id:
                    row = conn.execute(
                        """SELECT title FROM conversations
                            WHERE account_id=? AND conversation_id=?
                            LIMIT 2""",
                        (account_id, conversation_id),
                    ).fetchall()
                else:
                    row = conn.execute(
                        """SELECT title FROM conversations
                            WHERE conversation_id=? LIMIT 2""",
                        (conversation_id,),
                    ).fetchall()
        except sqlite3.Error:
            return None
        if len(row) != 1:
            return None
        label = str(row[0][0] or '').strip()
        return label[:200] or None

    def tick(self) -> None:
        if not self.config.armed:
            return
        now = float(self.now())
        self._recover_sends(now)
        result = self.action({
            'operation': 'events',
            'account_id': self.config.account_id,
            'cursors': self.store.cursor_map(),
            'observed_at': now,
        })
        self._last_poll_at = now
        for acknowledgement in result.get('acknowledgements') or ():
            if not isinstance(acknowledgement, Mapping):
                continue
            target_ref = acknowledgement.get('target_ref')
            position = acknowledgement.get('source_position')
            if isinstance(target_ref, str) and type(position) is int:
                self.store.advance_cursor(target_ref, position, now=now)
        for raw in result.get('events') or ():
            event = self._event(raw)
            existing = self.store.find_round(
                event.account_id, event.conversation_id,
            )
            if not self._target_allowed(event.target_ref):
                self.store.advance_cursor(
                    event.target_ref, event.source_position, now=now,
                )
                self.store.add_activity(
                    'event_ignored',
                    state='target_not_allowed',
                    now=now,
                    target_ref=event.target_ref,
                    conversation_id=event.conversation_id,
                )
                continue
            if existing is None or event.source_position > existing.source_position:
                self.rounds.observe(event, now=now)
                self.store.add_activity(
                    'message_received',
                    state='observed',
                    now=now,
                    target_ref=event.target_ref,
                    conversation_id=event.conversation_id,
                    text='\n'.join(
                        item.text for item in event.messages if item.text
                    )[:8_000] or f'[{event.latest_kind}]',
                )
        self._collect_generations(now)
        self._start_generations(now)
        self._process_ready(now)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.tick()
                self._last_error = ''
            except Exception as exc:
                self._last_error = type(exc).__name__
                try:
                    self.store.add_activity(
                        'service_error',
                        state=type(exc).__name__,
                        now=float(self.now()),
                    )
                except Exception:
                    pass
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.05, float(self.config.poll_seconds) - elapsed))

    def _target_allowed(self, target_ref: str) -> bool:
        return (
            self.config.target_scope == 'all_private_except_official'
            or target_ref in self.config.allowed_target_refs
        )

    @staticmethod
    def _event(raw: Any) -> ReplyEvent:
        if not isinstance(raw, Mapping):
            raise ReplyServiceError('provider event is not an object')
        try:
            messages = tuple(
                EvidenceMessage(
                    citation=str(item['citation']),
                    source_position=int(item['source_position']),
                    observed_at=float(item['observed_at']),
                    kind=str(item['kind']),
                    text=(
                        str(item['text'])
                        if item.get('text') is not None
                        else None
                    ),
                )
                for item in raw['messages']
                if isinstance(item, Mapping)
                and item.get('trust') == 'untrusted_evidence'
            )
            return ReplyEvent(
                event_id=str(raw['event_id']),
                account_id=str(raw['account_id']),
                conversation_id=str(raw['conversation_id']),
                target_ref=str(raw['target_ref']),
                source_position=int(raw['source_position']),
                latest_fingerprint=str(raw['latest_fingerprint']),
                messages=messages,
                observed_at=float(raw['observed_at']),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplyServiceError('provider event contract is invalid') from exc

    def _collect_generations(self, now: float) -> None:
        for round_id, (revision, future) in list(self._generating.items()):
            if not future.done():
                continue
            self._generating.pop(round_id, None)
            try:
                draft = future.result()
            except Exception as exc:
                try:
                    current = self.store.get_round(round_id)
                except ReplyStoreNotFound:
                    continue
                if current.revision != revision:
                    continue
                reason = str(exc)
                retryable = (
                    isinstance(exc, ReplyGenerationError)
                    and reason not in {
                        'generation_event_mismatch',
                        'generation_context_mismatch',
                        'generation_stale',
                    }
                )
                self.rounds.fail(
                    round_id,
                    reason='generation_' + type(exc).__name__,
                    retryable=retryable,
                    now=now,
                )
                self.store.add_activity(
                    'generation_failed',
                    state=type(exc).__name__,
                    now=now,
                    target_ref=current.target_ref,
                    conversation_id=current.conversation_id,
                )
                continue
            self.store.add_activity(
                'reply_generated',
                state='generated',
                now=now,
                target_ref=draft.target_ref,
                conversation_id=draft.conversation_id,
                text=draft.text,
            )

    def _start_generations(self, now: float) -> None:
        cursors = self.store.cursor_map()
        for record in self.rounds.preparable(now=now):
            if cursors.get(record.target_ref, 0) >= record.source_position:
                continue
            current = self.store.current_draft(
                record.round_id, round_revision=record.revision,
            )
            active = self._generating.get(record.round_id)
            if current is not None or (
                active is not None and active[0] == record.revision
            ):
                continue
            try:
                event = self.store.get_event(
                    record.round_id, round_revision=record.revision,
                )
            except ReplyStoreNotFound:
                continue
            future = self._executor.submit(
                self.generation.generate, record, event,
            )
            self._generating[record.round_id] = (record.revision, future)

    def _process_ready(self, now: float) -> None:
        cursors = self.store.cursor_map()
        for record in self.rounds.ready(now=now):
            draft = self.store.current_draft(
                record.round_id, round_revision=record.revision,
            )
            if draft is None:
                continue
            if cursors.get(record.target_ref, 0) >= record.source_position:
                operation = self._send_for_draft(draft)
                if not self._authorized_retry_dispatch(operation, draft):
                    continue
            if draft.state == 'generated':
                if self.config.mode == 'shadow':
                    self.store.add_activity(
                        'shadow_draft',
                        state='observed_without_delivery',
                        now=now,
                        target_ref=draft.target_ref,
                        conversation_id=draft.conversation_id,
                        text=draft.text,
                    )
                    self.store.advance_cursor(
                        draft.target_ref, draft.source_position, now=now,
                    )
                elif self.config.mode == 'review_queue':
                    review = self.store.enqueue_review(draft.draft_id, now=now)
                    self.store.add_activity(
                        'review_queued',
                        state='pending',
                        now=now,
                        target_ref=draft.target_ref,
                        conversation_id=draft.conversation_id,
                        text=draft.text,
                    )
                    _ = review
                else:
                    self._send_draft(
                        draft,
                        grant_ref=(
                            f'auto_live_{record.round_id}_{record.revision}'
                        ),
                        now=now,
                    )
            elif draft.state == 'approved':
                review_rows = [
                    review
                    for review, item in self.store.list_reviews(
                        state='approved', limit=1_000,
                    )
                    if item.draft_id == draft.draft_id
                ]
                if len(review_rows) == 1:
                    self._send_draft(
                        draft, grant_ref=review_rows[0].review_id, now=now,
                    )
            elif draft.state == 'rejected':
                self.store.advance_cursor(
                    draft.target_ref, draft.source_position, now=now,
                )

    @staticmethod
    def _operation_identity(draft: ReplyDraft) -> tuple[str, str]:
        digest = hashlib.sha256(
            (
                'trove-reply-send-v1\0'
                + draft.draft_id
                + '\0'
                + draft.digest
            ).encode('utf-8')
        ).hexdigest()
        return 'send_' + digest[:32], digest

    def _intent(self, draft: ReplyDraft, grant_ref: str) -> SendIntent:
        operation_id, idempotency_key = self._operation_identity(draft)
        return SendIntent(
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            draft_id=draft.draft_id,
            account_id=draft.account_id,
            conversation_id=draft.conversation_id,
            target_ref=draft.target_ref,
            expected_source_position=draft.source_position,
            draft_digest=draft.digest,
            text=draft.text,
            grant_ref=grant_ref,
        )

    @staticmethod
    def _send_payload(intent: SendIntent, operation: str) -> dict[str, Any]:
        return {
            'operation': operation,
            'operation_id': intent.operation_id,
            'idempotency_key': intent.idempotency_key,
            'account_id': intent.account_id,
            'target_ref': intent.target_ref,
            'expected_source_position': intent.expected_source_position,
            'draft_digest': intent.draft_digest,
            'text': intent.text,
        }

    def _send_draft(
        self,
        draft: ReplyDraft,
        *,
        grant_ref: str,
        now: float,
    ) -> None:
        policy = self.store.send_policy_status(
            draft.target_ref,
            now=now,
            cooldown_seconds=float(self.config.cooldown_seconds),
            daily_send_limit=self.config.daily_send_limit,
        )
        if not policy['allowed']:
            reason = str(policy['reason'])
            if reason == 'target_cooldown':
                round_record = self.store.get_round(draft.round_id)
                self.store.save_round(
                    replace(
                        round_record,
                        not_before=max(
                            round_record.not_before,
                            float(policy['retry_at']),
                        ),
                    ),
                    now=now,
                )
            else:
                self.store.advance_cursor(
                    draft.target_ref, draft.source_position, now=now,
                )
            self.store.add_activity(
                'send_policy_blocked',
                state=reason,
                now=now,
                target_ref=draft.target_ref,
                conversation_id=draft.conversation_id,
            )
            return
        intent = self._intent(draft, grant_ref)
        operation, _replayed = self.store.prepare_send(intent, now=now)
        if operation.terminal:
            self.store.advance_cursor(
                draft.target_ref, draft.source_position, now=now,
            )
            return
        if operation.state in {'dispatched', 'reconciling'}:
            self._reconcile(intent, now)
            return
        self.store.mark_dispatched(
            operation.operation_id,
            external_ref=f'{self.config.provider_id}:{operation.operation_id}',
            now=now,
        )
        try:
            result = self.action(self._send_payload(intent, 'send'))
        except Exception:
            self.store.mark_reconciling(operation.operation_id, now=now)
            self._reconcile(intent, now)
            return
        self._finish_provider_result(intent, result, now)

    def _finish_provider_result(
        self,
        intent: SendIntent,
        result: Mapping[str, Any],
        now: float,
    ) -> None:
        state = str(result.get('state') or 'unknown')
        if state not in {'completed', 'failed', 'unknown'}:
            state = 'unknown'
        stage = str(result.get('stage') or 'provider_result_invalid')[:128]
        completed = (
            state == 'completed'
            and isinstance(result.get('proof'), Mapping)
            and result['proof'].get('remote_ack') is True
            and result['proof'].get('text_sha256') == intent.draft_digest
            and type(result['proof'].get('source_position')) is int
            and result['proof']['source_position']
            > intent.expected_source_position
        )
        if state == 'completed' and not completed:
            state = 'unknown'
            stage = 'completed_without_exact_proof'
        terminal = self.store.finish_send(
            intent.operation_id,
            state=state,
            stage=stage,
            now=now,
            result=dict(result) if state == 'completed' else None,
            error_code=(
                None if state == 'completed' else f'provider_send_{state}'
            ),
        )
        self.store.advance_cursor(
            intent.target_ref, intent.expected_source_position, now=now,
        )
        self.store.add_activity(
            'reply_sent' if terminal.state == 'completed' else 'send_terminal',
            state=terminal.state,
            now=now,
            target_ref=intent.target_ref,
            conversation_id=intent.conversation_id,
            text=intent.text,
        )

    def _reconcile(self, intent: SendIntent, now: float) -> None:
        operation = self.store.get_send(intent.operation_id)
        if operation.state == 'dispatched':
            self.store.mark_reconciling(operation.operation_id, now=now)
        try:
            result = self.action(self._send_payload(intent, 'reconcile'))
        except Exception:
            result = {
                'state': 'unknown',
                'stage': 'provider_reconciliation_failed',
            }
        self._finish_provider_result(intent, result, now)

    def _recover_sends(self, now: float) -> None:
        for operation in self.store.list_sends(
            states=('prepared', 'dispatched', 'reconciling'),
            limit=1_000,
        ):
            try:
                draft = self.store.get_draft(operation.draft_id)
                intent = self.store.get_send_intent(operation.operation_id)
            except (ReplyStoreNotFound, ReplyStoreConflict):
                continue
            if intent.operation_id != operation.operation_id:
                continue
            if operation.state == 'prepared':
                if self._authorized_retry_dispatch(operation, draft):
                    # Explicit retries pass through _process_ready so the
                    # latest conversation round/source watermark is checked
                    # after this tick's Provider event poll.
                    continue
                self._send_draft(
                    draft, grant_ref=intent.grant_ref, now=now,
                )
            else:
                self._reconcile(intent, now)

    @staticmethod
    def _review_payload(review: Any, draft: ReplyDraft) -> dict[str, Any]:
        return {
            'review_id': review.review_id,
            'state': review.state,
            'created_at': review.created_at,
            'decided_at': review.decided_at,
            'draft': {
                'draft_id': draft.draft_id,
                'account_id': draft.account_id,
                'conversation_id': draft.conversation_id,
                'target_ref': draft.target_ref,
                'source_position': draft.source_position,
                'text': draft.text,
                'backend': draft.backend,
                'model': draft.model,
                'created_at': draft.created_at,
            },
            'trust': 'untrusted_evidence',
        }


__all__ = [
    'ActionInvoker', 'ReplyService', 'ReplyServiceConfig', 'ReplyServiceError',
]
