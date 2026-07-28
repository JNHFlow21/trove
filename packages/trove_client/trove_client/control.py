from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
import json
import time
from typing import TextIO
from typing import Any

from trove_core.approvals import ApprovalManager
from trove_core.managed_process import ManagedProcessManager
from trove_core.reply import ReplyServiceConfig, ReplyStore
from trove_core.security.operator_confirm import (
    OperatorConfirmationError,
    _open_controlling_terminal,
    decide_from_controlling_terminal,
)
from trove_daemon.lifecycle import RuntimeIdentity
from trove_daemon.operator_auth import (
    inspect_operator_app,
    save_operator_trust,
)


def stop_daemon(identity: RuntimeIdentity, *, timeout: float = 5.0) -> dict[str, Any]:
    return ManagedProcessManager(identity.runtime_dir).stop('daemon', timeout=timeout)


def decide_approval(
    vault_root: str | Path,
    approval_id: str,
    decision: str,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    manager = ApprovalManager(Path(vault_root))
    return decide_from_controlling_terminal(manager, approval_id, decision, note=note)


def control_reply_from_controlling_terminal(
    vault_root: str | Path,
    action: str,
    *,
    review_id: str | None = None,
    app_path: str | Path | None = None,
    mode: str | None = None,
    terminal: TextIO | None = None,
) -> dict[str, Any]:
    """Human-only reply control path with no stdin/env/--yes seam."""
    if action not in {
        'pair', 'arm', 'disarm', 'mode', 'approve', 'reject',
    }:
        raise OperatorConfirmationError('unsupported reply operator action')
    root = Path(vault_root)
    config = ReplyServiceConfig.load(root)
    context = (
        nullcontext(terminal)
        if terminal is not None
        else _open_controlling_terminal()
    )
    with context as tty:
        if tty is None or not tty.isatty():
            raise OperatorConfirmationError(
                'a live controlling terminal is required'
            )
        exact: dict[str, Any]
        if action == 'pair':
            if app_path is None:
                raise OperatorConfirmationError(
                    'operator app path is required'
                )
            identity = inspect_operator_app(app_path)
            exact = {
                'action': action,
                'operator_app': identity.redacted(),
            }
            expected = f'PAIR REPLY {identity.cdhash}'
        elif action in {'arm', 'disarm'}:
            exact = {
                'action': action,
                'reply_config': config.redacted(),
            }
            expected = f'{action.upper()} REPLY'
        elif action == 'mode':
            if mode not in {'shadow', 'review_queue', 'live'}:
                raise OperatorConfirmationError('reply mode is invalid')
            if config.armed:
                raise OperatorConfirmationError(
                    'reply service must be stopped before changing mode'
                )
            store = ReplyStore.for_vault(root)
            if (
                store.list_reviews(state='pending', limit=1)
                or store.list_sends(
                    states=('prepared', 'dispatched', 'reconciling'),
                    limit=1,
                )
            ):
                raise OperatorConfirmationError(
                    'reply queue must be resolved before changing mode'
                )
            exact = {
                'action': action,
                'from_mode': config.mode,
                'to_mode': mode,
                'reply_config': config.redacted(),
            }
            expected = f'SET REPLY MODE {mode}'
        else:
            if not isinstance(review_id, str) or not review_id:
                raise OperatorConfirmationError('reply review id is required')
            store = ReplyStore.for_vault(root)
            review = store.get_review(review_id)
            draft = store.get_draft(review.draft_id)
            if review.state != 'pending':
                raise OperatorConfirmationError(
                    'reply review is no longer pending'
                )
            decision = 'approved' if action == 'approve' else 'rejected'
            exact = {
                'action': action,
                'review_id': review_id,
                'draft_id': draft.draft_id,
                'account_id': draft.account_id,
                'conversation_id': draft.conversation_id,
                'target_ref': draft.target_ref,
                'source_position': draft.source_position,
                'text': draft.text,
                'decision': decision,
            }
            expected = f'{action.upper()} REPLY {review_id}'
        tty.write('TROVE reply operator decision (exact payload):\n')
        tty.write(
            json.dumps(
                exact, ensure_ascii=False, sort_keys=True, indent=2,
            )
            + '\n'
        )
        tty.write(f'Type exactly: {expected}\n> ')
        tty.flush()
        if tty.readline().rstrip('\r\n') != expected:
            raise OperatorConfirmationError(
                'operator confirmation did not match the exact request'
            )
        if action == 'pair':
            save_operator_trust(root, identity)
            return {
                'action': action,
                'operator_app': identity.redacted(),
                'daemon_restart_required': True,
            }
        if action in {'arm', 'disarm'}:
            updated = ReplyServiceConfig(
                **{
                    **config.__dict__,
                    'armed': action == 'arm',
                }
            )
            updated.save(root)
            return {
                'action': action,
                'config': updated.redacted(),
                'daemon_restart_required': True,
            }
        if action == 'mode':
            updated = ReplyServiceConfig(
                **{
                    **config.__dict__,
                    'mode': str(mode),
                }
            )
            updated.save(root)
            return {
                'action': action,
                'config': updated.redacted(),
                'daemon_restart_required': True,
            }
        decision = 'approved' if action == 'approve' else 'rejected'
        decided = store.decide_review(
            str(review_id), decision=decision, now=time.time(),
        )
        return {
            'action': action,
            'review_id': decided.review_id,
            'state': decided.state,
            'daemon_restart_required': False,
        }


__all__ = [
    'control_reply_from_controlling_terminal', 'decide_approval',
    'stop_daemon',
]
