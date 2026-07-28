from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trove_core.wechat.decrypt import key_capture
from trove_core.wechat.decrypt.key_capture import KeyCaptureConfig, build_key_store_secret_value, parse_key_store_secret
from trove_core.wechat.decrypt.runner import WeChatWCDBAESKeyStoreEngine
from trove_core.wechat.decrypt.secrets import AgentSwitchSecretResolver, SecretResolutionError


class _FakeScript:
    def on(self, _event, _callback):
        return None

    def load(self):
        return None


class _FakeSession:
    def __init__(self):
        self.detached = False

    def create_script(self, _source):
        return _FakeScript()

    def detach(self):
        self.detached = True


class _FakeDevice:
    def __init__(self):
        self.spawn_calls = []
        self.attach_calls = []
        self.resume_calls = []
        self.sessions = []

    def spawn(self, args):
        self.spawn_calls.append(args)
        return 4242

    def attach(self, pid):
        self.attach_calls.append(pid)
        session = _FakeSession()
        self.sessions.append(session)
        return session

    def resume(self, pid):
        self.resume_calls.append(pid)


class WeChatKeyCaptureTests(unittest.TestCase):
    def run_capture(self, config: KeyCaptureConfig, device: _FakeDevice, executable: Path):
        clock = iter([0.0, 2.0, 2.0])
        with (
            patch.dict(sys.modules, {'frida': SimpleNamespace(get_local_device=lambda: device)}),
            patch.object(key_capture, 'resolve_wechat_executable', return_value=executable),
            patch.object(key_capture.time, 'time', side_effect=lambda: next(clock)),
            patch.object(key_capture.time, 'sleep'),
        ):
            return key_capture.capture_key_store(config)

    def test_key_store_secret_payload_round_trips_without_paths(self):
        salt = 'a' * 32
        dk = 'b' * 64
        value = build_key_store_secret_value({salt: {'rounds': 256000, 'dk': dk, 'source': 'test'}})

        self.assertNotIn('\n', value)
        parsed = parse_key_store_secret(value)

        self.assertEqual(parsed[salt]['dk'], dk)
        self.assertNotIn('/Users/', value)
        self.assertNotIn('key_store.json', value)

    def test_runner_accepts_agent_switch_key_store_json_as_secret_value(self):
        salt = '1' * 32
        dk = '2' * 64
        secret_value = build_key_store_secret_value({salt: {'rounds': 256000, 'dk': dk}})
        engine = WeChatWCDBAESKeyStoreEngine()

        self.assertEqual(engine._load_keys(secret_value)[salt], dk)

    def test_wcdb_decrypt_streams_pages_without_reading_the_database_whole(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'message_0.db'
            destination = root / 'message_0.plain.db'
            source.write_bytes(os.urandom(WeChatWCDBAESKeyStoreEngine.page_size * 3))
            engine = WeChatWCDBAESKeyStoreEngine()

            with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file read')):
                engine._decrypt_to_file(source, destination, '0' * 64)

            self.assertEqual(destination.stat().st_size, source.stat().st_size)
            with destination.open('rb') as stream:
                self.assertEqual(stream.read(16), WeChatWCDBAESKeyStoreEngine.sqlite_header)

    def test_agent_switch_set_secret_uses_explicit_stdin_without_argv_or_env_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            capture = root / 'capture.json'
            helper = root / 'agent-switch-fake'
            ambient_canary = 'CANARY_' + 'AMBIENT_SECRET_4821'
            helper.write_text(
                '#!/usr/bin/env python3\n'
                'import json, os, sys\n'
                'if "--help" in sys.argv:\n'
                '    print("usage: secret set --stdin NAME")\n'
                '    raise SystemExit(0)\n'
                'value = sys.stdin.read()\n'
                f'open({str(capture)!r}, "w").write(json.dumps({{"argv": sys.argv[1:], "stdin": value, "env_contains": value in os.environ.values()}}))\n'
                'print("stored")\n',
                encoding='utf-8',
            )
            helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
            resolver = AgentSwitchSecretResolver(binary=str(helper))

            resolver.set_secret('TROVE_TEST_SECRET', 'not-a-real-secret')
            observed = json.loads(capture.read_text(encoding='utf-8'))

            self.assertEqual(observed['argv'], ['secret', 'set', '--stdin', 'TROVE_TEST_SECRET'])
            self.assertNotIn('not-a-real-secret', ' '.join(observed['argv']))
            self.assertEqual(observed['stdin'], 'not-a-real-secret')
            self.assertFalse(observed['env_contains'])

    def test_agent_switch_set_secret_fails_closed_when_safe_transport_is_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            marker = root / 'unsafe-set-invoked'
            helper = root / 'agent-switch-old'
            helper.write_text(
                '#!/usr/bin/env python3\n'
                'import sys\n'
                'if "--help" in sys.argv:\n'
                '    print("usage: secret set NAME VALUE")\n'
                '    raise SystemExit(0)\n'
                f'open({str(marker)!r}, "w").write("called")\n',
                encoding='utf-8',
            )
            helper.chmod(helper.stat().st_mode | stat.S_IXUSR)

            with self.assertRaises(SecretResolutionError) as raised:
                AgentSwitchSecretResolver(binary=str(helper)).set_secret('TROVE_TEST_SECRET', 'not-a-real-secret')

            self.assertEqual(raised.exception.code, 'secure_secret_transport_unavailable')
            self.assertFalse(marker.exists())

    def test_agent_switch_get_secret_uses_inherited_fd_without_standard_stream_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            capture = root / 'capture.json'
            helper = root / 'agent-switch-fake'
            ambient_canary = 'CANARY_' + 'AMBIENT_SECRET_4821'
            helper.write_text(
                '#!/usr/bin/env python3\n'
                'import json, os, sys\n'
                'if "--help" in sys.argv:\n'
                '    print("usage: secret get --fd N NAME")\n'
                '    raise SystemExit(0)\n'
                'fd = int(sys.argv[sys.argv.index("--fd") + 1])\n'
                'value = "not-a-real-secret"\n'
                f'open({str(capture)!r}, "w").write(json.dumps({{"argv": sys.argv[1:], "env_contains": value in os.environ.values(), "ambient_canary_present": {ambient_canary!r} in os.environ.values()}}))\n'
                'os.write(fd, value.encode("utf-8"))\n',
                encoding='utf-8',
            )
            helper.chmod(helper.stat().st_mode | stat.S_IXUSR)

            with patch.dict(os.environ, {'VOLCENGINE_ASR_API_KEY': ambient_canary}):
                value = AgentSwitchSecretResolver(binary=str(helper)).get_secret('TROVE_TEST_SECRET')
            observed = json.loads(capture.read_text(encoding='utf-8'))

            self.assertEqual(value, 'not-a-real-secret')
            self.assertEqual(observed['argv'][:3], ['secret', 'get', '--fd'])
            self.assertEqual(observed['argv'][-1], 'TROVE_TEST_SECRET')
            self.assertNotIn(value, ' '.join(observed['argv']))
            self.assertFalse(observed['env_contains'])
            self.assertFalse(observed['ambient_canary_present'])

    def test_agent_switch_get_secret_fails_closed_when_fd_transport_is_unsupported(self):
        with tempfile.TemporaryDirectory() as d:
            helper = Path(d) / 'agent-switch-old'
            helper.write_text(
                '#!/usr/bin/env python3\n'
                'import sys\n'
                'if "--help" in sys.argv:\n'
                '    print("usage: secret list")\n'
                '    raise SystemExit(0)\n',
                encoding='utf-8',
            )
            helper.chmod(helper.stat().st_mode | stat.S_IXUSR)

            with self.assertRaises(SecretResolutionError) as raised:
                AgentSwitchSecretResolver(binary=str(helper)).get_secret('TROVE_TEST_SECRET')

            self.assertEqual(raised.exception.code, 'secure_secret_transport_unavailable')

    def test_key_capture_reports_typed_failure_when_secret_transport_is_unavailable(self):
        class UnsupportedStore:
            def get_secret(self, _secret_name):
                raise SecretResolutionError('missing_key')

            def set_secret(self, _secret_name, _value):
                raise SecretResolutionError('secure_secret_transport_unavailable')

        captured = {
            'ok': True,
            'status': 'captured',
            'keys': {'a' * 32: {'rounds': 256000, 'dk': 'b' * 64}},
            'raw_content_included': False,
            'raw_paths_included': False,
        }
        with patch.object(key_capture, 'capture_key_store_isolated', return_value=captured):
            result = key_capture.capture_and_store_key_store(KeyCaptureConfig(), secret_store=UnsupportedStore())

        self.assertFalse(result['ok'])
        self.assertEqual(result['status'], 'secure_secret_transport_unavailable')
        self.assertFalse(result['secret_written'])
        self.assertNotIn('keys', result)
        self.assertFalse(result['raw_content_included'])

    def test_spawn_capture_activates_resumed_process_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            executable = Path(d) / 'WeChat'
            executable.touch()
            device = _FakeDevice()

            with patch.object(key_capture, '_activate_process_by_pid') as activate:
                result = self.run_capture(KeyCaptureConfig(wait_seconds=1), device, executable)

            self.assertEqual(result['status'], 'no_keys_captured')
            self.assertEqual(device.resume_calls, [4242])
            activate.assert_called_once_with(4242)

    def test_spawn_capture_can_opt_out_of_activation(self):
        with tempfile.TemporaryDirectory() as d:
            executable = Path(d) / 'WeChat'
            executable.touch()
            device = _FakeDevice()

            with patch.object(key_capture, '_activate_process_by_pid') as activate:
                self.run_capture(KeyCaptureConfig(wait_seconds=1, activate_spawned=False), device, executable)

            self.assertEqual(device.resume_calls, [4242])
            activate.assert_not_called()

    def test_spawn_attach_failure_resumes_wechat_instead_of_leaving_it_suspended(self):
        with tempfile.TemporaryDirectory() as d:
            executable = Path(d) / 'WeChat'
            executable.touch()
            device = _FakeDevice()
            device.attach = lambda _pid: (_ for _ in ()).throw(RuntimeError('permission denied'))

            with patch.object(key_capture, '_activate_process_by_pid') as activate:
                result = self.run_capture(KeyCaptureConfig(wait_seconds=1), device, executable)

            self.assertEqual(result['status'], 'frida_attach_failed')
            self.assertEqual(device.resume_calls, [4242])
            activate.assert_called_once_with(4242)

    def test_attach_mode_never_activates_or_resumes_existing_process(self):
        with tempfile.TemporaryDirectory() as d:
            executable = Path(d) / 'WeChat'
            executable.touch()
            device = _FakeDevice()
            clock = iter([0.0, 2.0, 2.0])

            with (
                patch.dict(sys.modules, {'frida': SimpleNamespace(get_local_device=lambda: device)}),
                patch.object(key_capture, 'resolve_wechat_executable', return_value=executable),
                patch.object(key_capture, '_wait_for_processes', return_value=[31337]),
                patch.object(key_capture, '_activate_process_by_pid') as activate,
                patch.object(key_capture.time, 'time', side_effect=lambda: next(clock)),
                patch.object(key_capture.time, 'sleep'),
            ):
                result = key_capture.capture_key_store(KeyCaptureConfig(mode='attach', wait_seconds=1))

            self.assertEqual(result['status'], 'no_keys_captured')
            self.assertEqual(device.attach_calls, [31337])
            self.assertEqual(device.spawn_calls, [])
            self.assertEqual(device.resume_calls, [])
            activate.assert_not_called()

    def test_activation_tolerates_osascript_launch_failure(self):
        with patch.object(key_capture.subprocess, 'run', side_effect=OSError('osascript unavailable')) as run:
            self.assertIsNone(key_capture._activate_process_by_pid(4242))

        self.assertEqual(run.call_args.args[0][:2], ['/usr/bin/osascript', '-e'])

    def test_spawn_target_uses_bundle_id_unless_app_path_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            executable = Path(d) / 'WeChat'
            executable.touch()

            bundle_device = _FakeDevice()
            with patch.object(key_capture, '_activate_process_by_pid'):
                self.run_capture(
                    KeyCaptureConfig(bundle_id='com.tencent.xinWeChat', wait_seconds=1),
                    bundle_device,
                    executable,
                )
            self.assertEqual(bundle_device.spawn_calls, [['com.tencent.xinWeChat']])

            path_device = _FakeDevice()
            with patch.object(key_capture, '_activate_process_by_pid'):
                self.run_capture(
                    KeyCaptureConfig(
                        bundle_id='com.tencent.xinWeChat',
                        wechat_app='/Applications/WeChat.app',
                        wait_seconds=1,
                    ),
                    path_device,
                    executable,
                )
            self.assertEqual(path_device.spawn_calls, [[str(executable)]])


if __name__ == '__main__':
    unittest.main()
