from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


DEFAULT_CUA_DRIVER = Path.home() / '.local/bin/cua-driver'
EXPECTED_CUA_EXECUTABLE = Path('/Applications/CuaDriver.app/Contents/MacOS/cua-driver')


class CuaDriver:
    """Privacy-safe wrapper; contact and draft text travel on stdin only."""

    def __init__(
        self,
        binary: str | Path = DEFAULT_CUA_DRIVER,
        *,
        runtime_root: str | Path | None = None,
    ) -> None:
        self.binary = Path(binary).expanduser()
        self.runtime_root = (
            Path(runtime_root)
            if runtime_root is not None
            else Path.home() / 'Library/Application Support/TROVE/reply'
        )

    def _validate_binary(self) -> None:
        if not self.binary.is_file():
            raise RuntimeError('cua_driver_missing')
        if self.binary.resolve() != EXPECTED_CUA_EXECUTABLE.resolve():
            raise RuntimeError('cua_driver_executable_mismatch')

    def _run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *args],
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )

    def _start_daemon(self, *, timeout_seconds: float = 5.0) -> None:
        lock_path = self.runtime_root / 'cua' / 'daemon-start.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(lock_path.parent, 0o700)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if self._run(['status'], timeout_seconds=2.0).returncode == 0:
                return
            environment = {
                'HOME': str(Path.home()),
                'PATH': '/usr/bin:/bin:/usr/sbin:/sbin',
                'TMPDIR': os.environ.get('TMPDIR', '/tmp'),
            }
            subprocess.Popen(
                [str(self.binary), 'serve', '--no-overlay'],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
            deadline = time.monotonic() + max(1.0, timeout_seconds)
            while time.monotonic() < deadline:
                if self._run(['status'], timeout_seconds=2.0).returncode == 0:
                    return
                time.sleep(0.1)
            raise RuntimeError('cua_driver_daemon_start_timeout')
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def call(
        self,
        tool: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        self._validate_binary()
        input_text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        result = self._run(
            ['call', str(tool)],
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )
        if result.returncode != 0 and 'daemon is not running' in result.stderr:
            self._start_daemon()
            result = self._run(
                ['call', str(tool)],
                input_text=input_text,
                timeout_seconds=timeout_seconds,
            )
        if result.returncode != 0:
            raise RuntimeError(f'cua_driver_{tool}_exit:{result.returncode}')
        try:
            response = json.loads(result.stdout.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'cua_driver_{tool}_non_json_response') from exc
        if not isinstance(response, dict) or response.get('error'):
            raise RuntimeError(f'cua_driver_{tool}_reported_error')
        return response

    def ensure_ready(self) -> None:
        # Delivery does not capture the screen. It relies on Accessibility
        # input, clipboard draft read-back, and an exact database echo. Keep
        # readiness read-only so it cannot raise TCC/direct-capture prompts.
        response = self.call('check_permissions', {'prompt': False})
        if not response.get('accessibility'):
            raise RuntimeError('cua_driver_accessibility_not_granted')
        source = response.get('source')
        if not isinstance(source, dict):
            raise RuntimeError('cua_driver_identity_missing')
        executable = Path(str(source.get('executable') or ''))
        if executable.resolve() != EXPECTED_CUA_EXECUTABLE.resolve():
            raise RuntimeError('cua_driver_identity_mismatch')

    def click(self, pid: int, window_id: int, x: float, y: float) -> None:
        self.call('click', {
            'pid': int(pid),
            'window_id': int(window_id),
            'x': float(x),
            'y': float(y),
            'delivery_mode': 'background',
        })

    def type_text(self, pid: int, window_id: int, text: str) -> None:
        self.call('type_text', {
            'pid': int(pid),
            'window_id': int(window_id),
            'text': str(text),
            'delay_ms': 5,
            'delivery_mode': 'background',
        })

    def hotkey(self, pid: int, window_id: int, keys: tuple[str, ...]) -> None:
        self.call('hotkey', {
            'pid': int(pid),
            'window_id': int(window_id),
            'keys': list(keys),
            'delivery_mode': 'background',
        })

    def press_key(self, pid: int, window_id: int, key: str) -> None:
        self.call('press_key', {
            'pid': int(pid),
            'window_id': int(window_id),
            'key': str(key),
            'delivery_mode': 'background',
        })
