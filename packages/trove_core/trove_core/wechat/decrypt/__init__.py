"""Local, redacted WeChat decrypt orchestration.

The package owns orchestration and manifests only. Secret values and raw WeChat
artifacts must never be persisted in the repo or in redacted reports.
"""

from .config import (
    ALLOWED_FILE_FAMILIES,
    OUT_OF_SCOPE_FILE_FAMILIES,
    DecryptConfig,
    DecryptFilePlan,
    DecryptPlan,
    SelectedAccount,
    classify_file_family,
)
from .preflight import build_decrypt_plan
from .runner import run_decrypt_plan
from .status import decrypt_status, known_keyed_account_refs, rollback_current

__all__ = [
    'ALLOWED_FILE_FAMILIES',
    'OUT_OF_SCOPE_FILE_FAMILIES',
    'DecryptConfig',
    'DecryptFilePlan',
    'DecryptPlan',
    'SelectedAccount',
    'classify_file_family',
    'build_decrypt_plan',
    'run_decrypt_plan',
    'decrypt_status',
    'known_keyed_account_refs',
    'rollback_current',
]
