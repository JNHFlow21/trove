from __future__ import annotations

from contextlib import nullcontext
import io
import json
from typing import Any, TextIO


class OperatorConfirmationError(RuntimeError):
    code = 'operator_confirmation_required'


def _open_controlling_terminal() -> TextIO:
    reader: io.FileIO | None = None
    writer: io.FileIO | None = None
    try:
        # Python's text ``r+`` wrapper requires a seekable stream on macOS,
        # while /dev/tty is intentionally non-seekable.  Bind independent
        # read/write handles into one duplex text stream instead.
        reader = io.FileIO('/dev/tty', 'r')
        writer = io.FileIO('/dev/tty', 'w')
        pair = io.BufferedRWPair(reader, writer)
        return io.TextIOWrapper(pair, encoding='utf-8', line_buffering=True)
    except (OSError, ValueError) as exc:
        for stream in (reader, writer):
            if stream is not None:
                stream.close()
        raise OperatorConfirmationError('a controlling terminal is required') from exc


def decide_from_controlling_terminal(
    manager: Any,
    approval_id: str,
    decision: str,
    *,
    note: str | None = None,
    terminal: TextIO | None = None,
) -> dict[str, Any]:
    """Make one human approval decision on the controlling terminal only.

    The function intentionally has no stdin, environment, or noninteractive
    confirmation parameter.  Adapters and Agents must never call it.
    """

    if decision not in {'approved', 'rejected'}:
        raise OperatorConfirmationError('decision must be approved or rejected')
    context = nullcontext(terminal) if terminal is not None else _open_controlling_terminal()
    with context as tty:
        if tty is None or not tty.isatty():
            raise OperatorConfirmationError('a live controlling terminal is required')
        record = manager.load(approval_id)
        verb = 'APPROVE' if decision == 'approved' else 'REJECT'
        exact = {
            'approval_id': approval_id,
            'action': record.action,
            'danger_class': record.danger_class,
            'payload': record.payload,
            'decision': decision,
        }
        tty.write('TROVE operator decision (exact payload):\n')
        tty.write(json.dumps(exact, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
        tty.write(f'Type exactly: {verb} {approval_id}\n> ')
        tty.flush()
        confirmation = tty.readline().rstrip('\r\n')
        if confirmation != f'{verb} {approval_id}':
            raise OperatorConfirmationError('operator confirmation did not match the exact request')
        decided = manager.decide(approval_id, decision, note=note)
        return decided.to_dict()


__all__ = [
    'OperatorConfirmationError', 'decide_from_controlling_terminal',
]
