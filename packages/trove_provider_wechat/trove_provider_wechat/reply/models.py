from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re


WORK_BUNDLE_ID = 'com.tencent.xinWeChat2'
WORK_APP_PATH = Path('/Applications/WeChat2.app')
WORK_EXECUTABLE_PATH = WORK_APP_PATH / 'Contents/MacOS/WeChat'


def stable_ref(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class WeChatLiveConfig:
    # account_id is the canonical TROVE account identifier. The source
    # account name is kept separate because it is private source-routing
    # material and must never become the Vault account identity.
    account_id: str
    account_id_sha256: str
    source_account_id: str = field(default='', repr=False)
    conversation_namespace: str = field(default='', repr=False)
    enabled: bool = False
    bundle_id: str = WORK_BUNDLE_ID
    app_path: str = str(WORK_APP_PATH)
    container_name: str = WORK_BUNDLE_ID
    send_shortcut: str = 'unconfigured'
    max_reply_chars: int = 500
    private_chats_enabled: bool = True
    groups_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.account_id, str) or not self.account_id:
            raise ValueError('live_account_id_required')
        source_account_id = self.source_account_id or self.account_id
        if (
            not isinstance(source_account_id, str)
            or not source_account_id
            or re.fullmatch(r'[0-9a-f]{64}', self.account_id_sha256) is None
            or hashlib.sha256(source_account_id.encode('utf-8')).hexdigest()
            != self.account_id_sha256
        ):
            raise ValueError('live_account_hash_required')
        if self.conversation_namespace:
            expected_account = (
                'acct-'
                + hashlib.sha256(
                    self.conversation_namespace.encode('utf-8'),
                ).hexdigest()[:12]
            )
            if self.account_id != expected_account:
                raise ValueError('live_vault_account_binding_invalid')
        if type(self.enabled) is not bool:
            raise ValueError('live_enabled_must_be_boolean')
        if (
            self.bundle_id != WORK_BUNDLE_ID
            or self.container_name != WORK_BUNDLE_ID
            or Path(self.app_path) != WORK_APP_PATH
        ):
            raise ValueError('private_or_unexpected_client_forbidden')
        if self.send_shortcut not in {'unconfigured', 'return', 'command_return'}:
            raise ValueError('invalid_send_shortcut')
        if self.enabled and self.send_shortcut == 'unconfigured':
            raise ValueError('armed_send_shortcut_required')
        if type(self.max_reply_chars) is not int or not 1 <= self.max_reply_chars <= 2_000:
            raise ValueError('invalid_max_reply_chars')
        if self.groups_enabled:
            raise ValueError('groups_not_supported')

    @property
    def source_account(self) -> str:
        return self.source_account_id or self.account_id

    @property
    def conversation_scope(self) -> str:
        return self.conversation_namespace or self.source_account


@dataclass(frozen=True)
class ContactIdentity:
    target_id: str = field(repr=False)
    target_ref: str
    search_query: str = field(repr=False)
    header_candidates: tuple[str, ...] = field(repr=False)
    unique_search: bool


@dataclass(frozen=True)
class LiveMessage:
    target_id: str = field(repr=False)
    target_ref: str
    source_name: str
    source_position: int
    server_id: str = field(repr=False)
    local_type: int
    create_time: int
    is_outgoing: bool
    text: str = field(repr=False)

    @property
    def fingerprint(self) -> str:
        return stable_ref(
            f'{self.target_id}\0{self.source_name}\0'
            f'{self.source_position}\0{self.server_id}'
        )

    @property
    def server_acknowledged(self) -> bool:
        return self.server_id.strip() not in {'', '0'}


@dataclass(frozen=True)
class SendOutcome:
    status: str
    reason: str
    target_ref: str
    app_pid: int
    echo_source_position: int = 0

    def __post_init__(self) -> None:
        if self.status not in {'completed', 'failed', 'unknown'}:
            raise ValueError('invalid_send_outcome')


@dataclass(frozen=True)
class SenderReadiness:
    available: bool
    armed: bool
    reason: str = ''
    app_pid: int = 0

    @property
    def ready(self) -> bool:
        return self.available and self.armed

    def to_dict(self) -> dict[str, object]:
        return {
            'state': 'ready' if self.ready else 'blocked',
            'ready': self.ready,
            'available': self.available,
            'armed': self.armed,
            'reason': self.reason,
            'app_pid': self.app_pid,
            'bundle_id': WORK_BUNDLE_ID,
            'delivery_mode': 'foreground_zero_click',
            'success_proof': 'server_ack_required',
        }


@dataclass(frozen=True)
class RunningApp:
    pid: int
    bundle_id: str
    app_path: Path
    executable_path: Path


@dataclass(frozen=True)
class WindowRef:
    window_id: int
    x: float
    y: float
    width: float
    height: float
