from __future__ import annotations

from pathlib import Path

from trove_core.approvals import (
    ApprovalGrant,
    ApprovalValidationError,
    claim_approval_grant,
)
from trove_core.vault.config import VaultConfig
from trove_core.vault.writer_recovery import (
    WriterMarkerRecoveryResult,
    _recover_writer_marker_offline,
)


WRITER_MARKER_RECOVERY_ACTION = 'recover_writer_marker'
WRITER_MARKER_RECOVERY_DANGER_CLASS = 'delete_or_purge'


def writer_marker_recovery_payload(*, legacy_writers_stopped: bool) -> dict[str, bool]:
    """Return the only payload authorized for offline writer-marker recovery."""

    if type(legacy_writers_stopped) is not bool or legacy_writers_stopped is not True:
        raise ApprovalValidationError(
            'legacy_writers_stopped must be the exact boolean true',
            code='writer_marker_recovery_confirmation_required',
        )
    return {'legacy_writers_stopped': True}


def _claim_recovery_grant(
    grant: ApprovalGrant,
    cfg: VaultConfig,
    payload: dict[str, bool],
) -> None:
    """Claim once through the non-overridable application-boundary API."""

    claim_approval_grant(
        grant,
        cfg.root,
        action=WRITER_MARKER_RECOVERY_ACTION,
        danger_class=WRITER_MARKER_RECOVERY_DANGER_CLASS,
        payload=payload,
    )


def recover_writer_marker(
    vault: VaultConfig | str | Path,
    *,
    legacy_writers_stopped: bool,
    approval_grant: ApprovalGrant,
) -> WriterMarkerRecoveryResult:
    """Recover one dead writer marker through the approved offline protocol.

    This is the only public application entry point.  It rejects dictionaries,
    approval records, IDs, and other lookalikes: the caller must first consume
    an exact approval into an authentic ``ApprovalGrant``.
    """

    payload = writer_marker_recovery_payload(
        legacy_writers_stopped=legacy_writers_stopped,
    )
    if type(approval_grant) is not ApprovalGrant:
        raise ApprovalValidationError(
            'an authentic ApprovalGrant is required',
            code='invalid_grant',
        )
    cfg = (
        vault
        if isinstance(vault, VaultConfig)
        else VaultConfig.resolve(str(Path(vault).expanduser()), env={})
    )
    # Reject a wrong/cross-Vault grant before even preparing lock authority.
    # The same exact tuple is durably claimed after the
    # dead-owner proof and immediately before cleanup.
    approval_grant.validate_for(
        cfg.root,
        action=WRITER_MARKER_RECOVERY_ACTION,
        danger_class=WRITER_MARKER_RECOVERY_DANGER_CLASS,
        payload=payload,
    )
    return _recover_writer_marker_offline(
        cfg,
        legacy_writers_stopped=legacy_writers_stopped,
        claim=lambda: _claim_recovery_grant(approval_grant, cfg, payload),
    )
