from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

from .models import WeChatLiveConfig


_DIGEST = re.compile(r'[0-9a-f]{64}')


class WeChatActionError(RuntimeError):
    code = 'provider_action_rejected'


def _text(payload: Mapping[str, Any], name: str, *, maximum: int = 8_000) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value or len(value.encode('utf-8')) > maximum:
        raise WeChatActionError(f'{name}_invalid')
    return value


def _position(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value <= 0:
        raise WeChatActionError(f'{name}_invalid')
    return value


class WeChatActionAdapter:
    """Source-specific live event, delivery, and reconciliation adapter."""

    def __init__(self, config: WeChatLiveConfig, source: Any, sender: Any) -> None:
        self.config = config
        self.source = source
        self.sender = sender

    def invoke(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise WeChatActionError('action_payload_invalid')
        operation = payload.get('operation')
        if operation == 'status':
            return self._status()
        if operation == 'events':
            return self._events(payload)
        if operation == 'send':
            return self._send(payload)
        if operation == 'reconcile':
            return self._reconcile(payload)
        if operation == 'retry_preflight':
            return self._retry_preflight(payload)
        raise WeChatActionError('action_operation_unavailable')

    def _require_account(self, payload: Mapping[str, Any]) -> None:
        if payload.get('account_id') != self.config.account_id:
            raise WeChatActionError('account_mismatch')

    def _status(self) -> Mapping[str, Any]:
        readiness = self.sender.readiness()
        return {
            **readiness.to_dict(),
            'account_id': self.config.account_id,
            'enabled': self.config.enabled,
        }

    def _events(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_account(payload)
        cursors = payload.get('cursors')
        observed_at = payload.get('observed_at')
        if (
            not isinstance(cursors, Mapping)
            or any(
                not isinstance(key, str)
                or _DIGEST.fullmatch(key) is None
                or type(value) is not int
                or value < 0
                for key, value in cursors.items()
            )
        ):
            raise WeChatActionError('event_cursors_invalid')
        if (
            isinstance(observed_at, bool)
            or not isinstance(observed_at, (int, float))
            or not math.isfinite(float(observed_at))
            or observed_at < 0
        ):
            raise WeChatActionError('observed_at_invalid')
        result = self.source.events(dict(cursors), observed_at=float(observed_at))
        if not isinstance(result, Mapping):
            raise WeChatActionError('event_result_invalid')
        events = result.get('events')
        acknowledgements = result.get('acknowledgements')
        if not isinstance(events, list) or not isinstance(acknowledgements, list):
            raise WeChatActionError('event_result_invalid')
        for event in events:
            if (
                not isinstance(event, Mapping)
                or event.get('account_id') != self.config.account_id
                or _DIGEST.fullmatch(str(event.get('target_ref') or '')) is None
                or type(event.get('source_position')) is not int
                or not isinstance(event.get('messages'), list)
            ):
                raise WeChatActionError('event_contract_invalid')
            for message in event['messages']:
                if (
                    not isinstance(message, Mapping)
                    or message.get('trust') != 'untrusted_evidence'
                ):
                    raise WeChatActionError('event_trust_invalid')
        return {
            'events': events,
            'acknowledgements': acknowledgements,
            'account_id': self.config.account_id,
        }

    def _send_contract(
        self,
        payload: Mapping[str, Any],
        *,
        allow_source_advance: bool = False,
    ) -> tuple[Any, str, int]:
        if not self.config.enabled:
            raise WeChatActionError('action_not_armed')
        self._require_account(payload)
        _text(payload, 'operation_id')
        _text(payload, 'idempotency_key')
        target_ref = _text(payload, 'target_ref', maximum=256)
        if _DIGEST.fullmatch(target_ref) is None:
            raise WeChatActionError('target_ref_invalid')
        source_position = _position(payload, 'expected_source_position')
        text = _text(payload, 'text', maximum=self.config.max_reply_chars * 4)
        draft_digest = payload.get('draft_digest')
        if (
            not isinstance(draft_digest, str)
            or _DIGEST.fullmatch(draft_digest) is None
            or hashlib.sha256(text.encode('utf-8')).hexdigest() != draft_digest
        ):
            raise WeChatActionError('draft_digest_mismatch')
        try:
            identity = self.source.resolve_identity(target_ref)
        except Exception as exc:
            raise WeChatActionError('target_unavailable') from exc
        if identity.target_ref != target_ref or not identity.unique_search:
            raise WeChatActionError('target_identity_mismatch')
        try:
            current = self.source.current_position(identity.target_id)
        except Exception as exc:
            raise WeChatActionError('source_position_unavailable') from exc
        if (
            (not allow_source_advance and current != source_position)
            or (allow_source_advance and current < source_position)
        ):
            raise WeChatActionError('source_position_advanced')
        return identity, text, source_position

    def _send(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        identity, text, source_position = self._send_contract(payload)
        outcome = self.sender.send(
            identity,
            text,
            after_source_position=source_position,
            shortcut=self.config.send_shortcut,
        )
        base: dict[str, Any] = {
            'state': outcome.status,
            'stage': outcome.reason,
            'target_ref': identity.target_ref,
            'app_pid': int(outcome.app_pid),
        }
        if (
            outcome.status == 'completed'
            and outcome.reason == 'server_ack_verified'
            and outcome.echo_source_position > source_position
        ):
            base['proof'] = {
                'source_position': int(outcome.echo_source_position),
                'remote_ack': True,
                'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            }
            return base
        if outcome.status == 'completed':
            return {
                **base,
                'state': 'unknown',
                'stage': 'completed_without_exact_proof',
            }
        return base

    def _reconcile(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        identity, text, source_position = self._send_contract(
            payload, allow_source_advance=True,
        )
        try:
            echo = self.source.wait_for_outgoing_echo(
                identity.target_id,
                after_source_position=source_position,
                expected_text=text,
                timeout_seconds=1.0,
            )
        except Exception as exc:
            raise WeChatActionError('reconciliation_failed') from exc
        if (
            echo is not None
            and echo.is_outgoing
            and echo.server_acknowledged
            and echo.text.replace('\r\n', '\n').replace('\r', '\n').strip()
            == text.replace('\r\n', '\n').replace('\r', '\n').strip()
            and echo.source_position > source_position
        ):
            return {
                'state': 'completed',
                'stage': 'server_ack_reconciled',
                'target_ref': identity.target_ref,
                'proof': {
                    'source_position': int(echo.source_position),
                    'remote_ack': True,
                    'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                },
            }
        return {
            'state': 'unknown',
            'stage': 'server_ack_not_observed',
            'target_ref': identity.target_ref,
        }

    def _retry_preflight(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Prove exact source stability and absence of the prior send echo."""

        identity, text, source_position = self._send_contract(payload)
        try:
            echo = self.source.wait_for_outgoing_echo(
                identity.target_id,
                after_source_position=source_position,
                expected_text=text,
                timeout_seconds=1.0,
            )
        except Exception as exc:
            raise WeChatActionError('retry_preflight_failed') from exc
        try:
            current = self.source.current_position(identity.target_id)
        except Exception as exc:
            raise WeChatActionError('retry_preflight_failed') from exc
        binding = {
            'operation_id': payload['operation_id'],
            'idempotency_key': payload['idempotency_key'],
            'target_ref': identity.target_ref,
            'expected_source_position': source_position,
            'draft_digest': payload['draft_digest'],
        }
        if echo is not None:
            return {
                'state': 'blocked',
                'stage': 'prior_send_echo_observed',
                **binding,
            }
        if current != source_position:
            raise WeChatActionError('source_position_advanced')
        return {
            'state': 'ready',
            'stage': 'retry_preflight_passed',
            **binding,
        }


__all__ = ['WeChatActionAdapter', 'WeChatActionError']
