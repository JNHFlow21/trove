from __future__ import annotations

import fcntl
import os
from pathlib import Path
import secrets
import time
from typing import Any, Callable, Protocol

from .models import (
    ContactIdentity,
    SendOutcome,
    SenderReadiness,
    WeChatLiveConfig,
)


class InputDriver(Protocol):
    def ensure_ready(self) -> None: ...
    def hotkey(self, pid: int, window_id: int, keys: tuple[str, ...]) -> None: ...
    def press_key(self, pid: int, window_id: int, key: str) -> None: ...
    def type_text(self, pid: int, window_id: int, text: str) -> None: ...


class SenderUI(Protocol):
    def resolve_exact_running_app(self, bundle_id: str, app_path: str) -> Any: ...
    def frontmost_pid(self) -> int: ...
    def activate_exact_pid(self, pid: int) -> None: ...
    def main_window_for_pid(self, pid: int) -> Any: ...
    def pasteboard(self) -> Any: ...
    def restore_frontmost_pid(self, pid: int) -> None: ...


class EchoReader(Protocol):
    def wait_for_outgoing_echo(
        self,
        target_id: str,
        *,
        after_source_position: int,
        expected_text: str,
        timeout_seconds: float,
    ) -> Any: ...


class SenderLockTimeout(RuntimeError):
    pass


def _acquire_sender_lock(path: Path, *, timeout_seconds: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.chmod(path, 0o600)
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise SenderLockTimeout('sender_lock_timeout')
            time.sleep(0.05)


def _safe_error(exc: Exception) -> str:
    raw = str(exc).split(':', 1)[0]
    code = ''.join(char for char in raw if char.isalnum() or char in '_-')
    return f'{type(exc).__name__}:{code or "sender_unavailable"}'[:120]


class VerifiedSender:
    """Serialize UI delivery and accept success only after an exact DB echo."""

    def __init__(
        self,
        config: WeChatLiveConfig,
        reader: EchoReader,
        *,
        runtime_root: str | Path | None = None,
        driver: InputDriver | None = None,
        ui: SenderUI | None = None,
        lock_path: str | Path | None = None,
        lock_timeout_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.reader = reader
        if driver is None:
            from .cua_driver import CuaDriver
            driver = CuaDriver(runtime_root=runtime_root)
        if ui is None:
            from .macos_ui import MacOSSenderUI
            ui = MacOSSenderUI()
        self.driver = driver
        self.ui = ui
        base = (
            Path(runtime_root)
            if runtime_root is not None
            else Path.home() / 'Library/Application Support/TROVE/reply'
        )
        self.lock_path = Path(lock_path) if lock_path is not None else base / 'sender.lock'
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.sleep = sleep

    def readiness(self) -> SenderReadiness:
        armed = self.config.enabled and self.config.send_shortcut in {
            'return', 'command_return',
        }
        try:
            app = self.ui.resolve_exact_running_app(
                self.config.bundle_id, self.config.app_path,
            )
            self.driver.ensure_ready()
        except Exception as exc:
            return SenderReadiness(False, armed, _safe_error(exc), 0)
        return SenderReadiness(
            True,
            armed,
            '' if armed else 'sender_disarmed',
            int(app.pid),
        )

    @staticmethod
    def _normalized(value: str) -> str:
        return value.replace('\r\n', '\n').replace('\r', '\n').strip()

    def _copy_selection(self, pasteboard: Any, pid: int, window_id: int) -> str:
        sentinel = f'TROVE-CLIPBOARD-CHECK-{secrets.token_hex(12).upper()}'
        pasteboard.set_text(sentinel)
        self.driver.hotkey(pid, window_id, ('cmd', 'a'))
        self.driver.hotkey(pid, window_id, ('cmd', 'c'))
        self.sleep(0.08)
        copied = pasteboard.text()
        if copied == sentinel:
            raise RuntimeError('copy_event_not_delivered')
        return self._normalized(copied)

    def _clear_exact_draft(
        self,
        pasteboard: Any,
        pid: int,
        window_id: int,
        expected: str,
    ) -> None:
        if self._copy_selection(pasteboard, pid, window_id) != expected:
            raise RuntimeError('draft_cleanup_content_mismatch')
        self.driver.hotkey(pid, window_id, ('cmd', 'a'))
        self.driver.press_key(pid, window_id, 'delete')
        self.sleep(0.08)

    def send(
        self,
        identity: ContactIdentity,
        text: str,
        *,
        after_source_position: int,
        shortcut: str | None = None,
        echo_timeout_seconds: float = 12.0,
    ) -> SendOutcome:
        try:
            lock_fd = _acquire_sender_lock(
                self.lock_path,
                timeout_seconds=self.lock_timeout_seconds,
            )
        except SenderLockTimeout:
            return SendOutcome('failed', 'sender_busy_timeout', identity.target_ref, 0)
        try:
            return self._send_locked(
                identity,
                text,
                after_source_position=after_source_position,
                shortcut=shortcut,
                echo_timeout_seconds=echo_timeout_seconds,
            )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _send_locked(
        self,
        identity: ContactIdentity,
        text: str,
        *,
        after_source_position: int,
        shortcut: str | None,
        echo_timeout_seconds: float,
    ) -> SendOutcome:
        reply = self._normalized(text)
        if not reply or len(reply) > self.config.max_reply_chars:
            return SendOutcome('failed', 'invalid_reply_text', identity.target_ref, 0)
        send_shortcut = shortcut or self.config.send_shortcut
        if send_shortcut not in {'return', 'command_return'}:
            return SendOutcome(
                'failed', 'send_shortcut_unconfigured', identity.target_ref, 0,
            )
        if not identity.unique_search:
            return SendOutcome(
                'failed', 'target_search_not_unique', identity.target_ref, 0,
            )

        app_pid = 0
        previous_frontmost = 0
        send_event_posted = False
        draft_pasted = False
        stage = 'resolve_client'
        window: Any = None
        try:
            app = self.ui.resolve_exact_running_app(
                self.config.bundle_id, self.config.app_path,
            )
            app_pid = int(app.pid)
            previous_frontmost = self.ui.frontmost_pid()
            stage = 'driver_health'
            self.driver.ensure_ready()
            stage = 'activate_foreground'
            self.ui.activate_exact_pid(app_pid)
            if self.ui.frontmost_pid() != app_pid:
                return SendOutcome(
                    'failed', 'foreground_activation_failed',
                    identity.target_ref, app_pid,
                )
            stage = 'locate_window'
            window = self.ui.main_window_for_pid(app_pid)
            stage = 'search_target'
            self.driver.hotkey(app_pid, window.window_id, ('cmd', 'f'))
            self.driver.hotkey(app_pid, window.window_id, ('cmd', 'a'))
            self.driver.press_key(app_pid, window.window_id, 'delete')
            self.driver.type_text(app_pid, window.window_id, identity.search_query)
            if self.ui.frontmost_pid() != app_pid:
                return SendOutcome(
                    'failed', 'foreground_focus_lost_before_navigate',
                    identity.target_ref, app_pid,
                )
            stage = 'verify_search'
            with self.ui.pasteboard() as pasteboard:
                if (
                    self._copy_selection(
                        pasteboard, app_pid, window.window_id,
                    )
                    != self._normalized(identity.search_query)
                ):
                    return SendOutcome(
                        'failed', 'search_verification_failed',
                        identity.target_ref, app_pid,
                    )
            stage = 'navigate'
            self.driver.press_key(app_pid, window.window_id, 'return')
            self.sleep(0.2)

            with self.ui.pasteboard() as pasteboard:
                stage = 'compose_ready'
                if self.ui.frontmost_pid() != app_pid:
                    return SendOutcome(
                        'failed', 'foreground_focus_lost_after_navigate',
                        identity.target_ref, app_pid,
                    )
                self.driver.type_text(app_pid, window.window_id, reply)
                draft_pasted = True
                self.sleep(0.12)
                stage = 'verify_draft'
                if self._copy_selection(pasteboard, app_pid, window.window_id) != reply:
                    try:
                        self._clear_exact_draft(
                            pasteboard, app_pid, window.window_id, reply,
                        )
                    except Exception:
                        pass
                    draft_pasted = False
                    return SendOutcome(
                        'failed', 'draft_verification_failed',
                        identity.target_ref, app_pid,
                    )
                stage = 'pre_send_focus_check'
                if self.ui.frontmost_pid() != app_pid:
                    self._clear_exact_draft(
                        pasteboard, app_pid, window.window_id, reply,
                    )
                    draft_pasted = False
                    return SendOutcome(
                        'failed', 'foreground_focus_lost_before_send',
                        identity.target_ref, app_pid,
                    )
                stage = 'post_send_event'
                if send_shortcut == 'return':
                    self.driver.press_key(app_pid, window.window_id, 'return')
                else:
                    self.driver.hotkey(app_pid, window.window_id, ('cmd', 'return'))
                send_event_posted = True
                stage = 'wait_database_echo'
                echo = self.reader.wait_for_outgoing_echo(
                    identity.target_id,
                    after_source_position=after_source_position,
                    expected_text=reply,
                    timeout_seconds=echo_timeout_seconds,
                )
                if (
                    echo is not None
                    and echo.is_outgoing
                    and echo.server_acknowledged
                    and self._normalized(echo.text) == reply
                    and int(echo.source_position) > after_source_position
                ):
                    draft_pasted = False
                    return SendOutcome(
                        'completed', 'server_ack_verified',
                        identity.target_ref, app_pid, int(echo.source_position),
                    )
                stage = 'inspect_unsent_draft'
                remaining = self._copy_selection(
                    pasteboard, app_pid, window.window_id,
                )
                if remaining == reply:
                    self._clear_exact_draft(
                        pasteboard, app_pid, window.window_id, reply,
                    )
                    draft_pasted = False
                    return SendOutcome(
                        'failed', 'shortcut_did_not_send',
                        identity.target_ref, app_pid,
                    )
                return SendOutcome(
                    'unknown', 'send_event_without_server_ack',
                    identity.target_ref, app_pid,
                )
        except Exception as exc:
            if send_event_posted:
                return SendOutcome(
                    'unknown', f'post_send_{type(exc).__name__}',
                    identity.target_ref, app_pid,
                )
            if draft_pasted and window is not None:
                try:
                    with self.ui.pasteboard() as pasteboard:
                        self._clear_exact_draft(
                            pasteboard, app_pid, window.window_id, reply,
                        )
                except Exception:
                    pass
            return SendOutcome(
                'failed', f'{stage}_{_safe_error(exc)}',
                identity.target_ref, app_pid,
            )
        finally:
            try:
                if (
                    previous_frontmost
                    and previous_frontmost != app_pid
                    and self.ui.frontmost_pid() == app_pid
                ):
                    self.ui.restore_frontmost_pid(previous_frontmost)
            except Exception:
                pass


__all__ = ['VerifiedSender']
