from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from trove_provider_wechat.reply import (
    ContactIdentity,
    LiveMessage,
    RunningApp,
    WeChatLiveConfig,
    WindowRef,
)
from trove_provider_wechat.reply.sender import VerifiedSender


APP_PID = 123
TARGET_REF = hashlib.sha256(b'raw-target').hexdigest()


class FakeReader:
    def __init__(self, echo):
        self.echo = echo
        self.calls = []

    def wait_for_outgoing_echo(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.echo


class FakePasteboard:
    def __init__(self, reads):
        self.reads = list(reads)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def set_text(self, _value):
        return None

    def text(self):
        return self.reads.pop(0)


class FakeDriver:
    def __init__(self):
        self.hotkeys = []
        self.keys = []
        self.typed = []

    def ensure_ready(self):
        return None

    def hotkey(self, _pid, _window_id, keys):
        self.hotkeys.append(keys)

    def press_key(self, _pid, _window_id, key):
        self.keys.append(key)

    def type_text(self, _pid, _window_id, text):
        self.typed.append(text)


class FakeUI:
    def __init__(self, pasteboard, *, frontmost_pid=APP_PID):
        self._pasteboard = pasteboard
        self._frontmost_pid = frontmost_pid

    def resolve_exact_running_app(self, *_args):
        return RunningApp(
            APP_PID,
            'com.tencent.xinWeChat2',
            Path('/Applications/WeChat2.app'),
            Path('/Applications/WeChat2.app/Contents/MacOS/WeChat'),
        )

    def frontmost_pid(self):
        return self._frontmost_pid

    def activate_exact_pid(self, _pid):
        return None

    def main_window_for_pid(self, _pid):
        return WindowRef(1, 0, 0, 1_000, 800)

    def pasteboard(self):
        return self._pasteboard

    def restore_frontmost_pid(self, _pid):
        return None


def identity():
    return ContactIdentity(
        'raw-target', TARGET_REF, 'unique-search', ('Expected',), True,
    )


def echo():
    return LiveMessage(
        'raw-target', TARGET_REF, 'message-fixture', 8, 'server-8',
        1, 1_001, True, 'reply',
    )


class VerifiedSenderTests(unittest.TestCase):
    def config(self):
        return WeChatLiveConfig(
            account_id='account-fixture',
            account_id_sha256=hashlib.sha256(
                b'account-fixture',
            ).hexdigest(),
            enabled=True,
            send_shortcut='return',
        )

    def test_success_requires_exact_database_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver()
            reader = FakeReader(echo())
            sender = VerifiedSender(
                self.config(),
                reader,
                driver=driver,
                ui=FakeUI(FakePasteboard(['unique-search', 'reply'])),
                lock_path=Path(directory) / 'sender.lock',
                sleep=lambda _value: None,
            )
            result = sender.send(identity(), 'reply', after_source_position=7)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.echo_source_position, 8)
        self.assertEqual(driver.typed, ['unique-search', 'reply'])
        self.assertEqual(driver.keys.count('return'), 2)

    def test_missing_echo_and_missing_draft_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            sender = VerifiedSender(
                self.config(),
                FakeReader(None),
                driver=FakeDriver(),
                ui=FakeUI(FakePasteboard(['unique-search', 'reply', ''])),
                lock_path=Path(directory) / 'sender.lock',
                sleep=lambda _value: None,
            )
            result = sender.send(identity(), 'reply', after_source_position=7)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(result.reason, 'send_event_without_server_ack')

    def test_wrong_running_client_fails_before_typing(self):
        class WrongUI(FakeUI):
            def resolve_exact_running_app(self, *_args):
                raise RuntimeError('exact_work_app_match_count:0')

        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver()
            sender = VerifiedSender(
                self.config(),
                FakeReader(None),
                driver=driver,
                ui=WrongUI(FakePasteboard([])),
                lock_path=Path(directory) / 'sender.lock',
                sleep=lambda _value: None,
            )
            result = sender.send(identity(), 'reply', after_source_position=7)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(driver.typed, [])

    def test_search_must_read_back_exactly_before_navigation_or_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = FakeDriver()
            sender = VerifiedSender(
                self.config(),
                FakeReader(None),
                driver=driver,
                ui=FakeUI(FakePasteboard(['wrong-search'])),
                lock_path=Path(directory) / 'sender.lock',
                sleep=lambda _value: None,
            )
            result = sender.send(identity(), 'reply', after_source_position=7)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.reason, 'search_verification_failed')
        self.assertEqual(driver.typed, ['unique-search'])
        self.assertNotIn('return', driver.keys)


if __name__ == '__main__':
    unittest.main()
