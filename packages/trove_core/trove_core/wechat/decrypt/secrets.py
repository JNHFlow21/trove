from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Protocol

from trove_core.security.subprocess_env import agent_switch_subprocess_environment
from .redaction import redact_text


MAX_SECRET_BYTES = 64 * 1024


class SecretResolutionError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class SecretSink(Protocol):
    def set_secret(self, secret_name: str | None, value: str) -> None: ...


class SecretStore(SecretSink, Protocol):
    def get_secret(self, secret_name: str | None) -> str: ...


@dataclass(frozen=True)
class AgentSwitchSecretResolver:
    binary: str = 'agent-switch'

    def get_secret(self, secret_name: str | None) -> str:
        if not secret_name:
            raise SecretResolutionError('missing_key')
        if not self._supports_fd_secret_get():
            raise SecretResolutionError('secure_secret_transport_unavailable')
        read_fd, write_fd = os.pipe()
        try:
            proc = subprocess.Popen(
                [self.binary, 'secret', 'get', '--fd', str(write_fd), secret_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write_fd,),
                env=agent_switch_subprocess_environment(),
            )
        except OSError as exc:
            os.close(read_fd)
            os.close(write_fd)
            raise SecretResolutionError('secret_resolver_unavailable') from exc
        os.close(write_fd)
        chunks: list[bytes] = []
        read_errors: list[BaseException] = []

        def read_secret_pipe() -> None:
            try:
                with os.fdopen(read_fd, 'rb') as stream:
                    chunks.append(stream.read(MAX_SECRET_BYTES + 1))
            except BaseException as exc:  # pragma: no cover - defensive OS boundary
                read_errors.append(exc)

        reader = threading.Thread(target=read_secret_pipe, name='trove-secret-fd-reader', daemon=True)
        reader.start()
        try:
            _stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            reader.join(timeout=1)
            raise SecretResolutionError('secret_resolver_unavailable') from exc
        reader.join(timeout=1)
        if reader.is_alive() or read_errors:
            raise SecretResolutionError('secret_resolver_unavailable')
        if proc.returncode != 0:
            # Keep stderr out of reports except as redacted diagnostic in logs if a caller chooses.
            _ = redact_text(stderr.decode('utf-8', errors='replace'))
            raise SecretResolutionError('missing_key')
        payload = b''.join(chunks)
        if not payload or len(payload) > MAX_SECRET_BYTES or b'\x00' in payload or b'\r' in payload or b'\n' in payload:
            raise SecretResolutionError('missing_key')
        try:
            value = payload.decode('utf-8', errors='strict')
        except UnicodeDecodeError as exc:
            raise SecretResolutionError('missing_key') from exc
        return value

    def set_secret(self, secret_name: str | None, value: str) -> None:
        if not secret_name:
            raise SecretResolutionError('missing_secret_name')
        if not self._supports_stdin_secret_set():
            raise SecretResolutionError('secure_secret_transport_unavailable')
        try:
            proc = subprocess.run(
                [self.binary, 'secret', 'set', '--stdin', secret_name],
                input=value,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
                env=agent_switch_subprocess_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SecretResolutionError('secret_resolver_unavailable') from exc
        if proc.returncode != 0:
            _ = redact_text(proc.stderr)
            raise SecretResolutionError('secret_write_failed')

    def _supports_stdin_secret_set(self) -> bool:
        return self._supports_flag(['secret', 'set', '--help'], '--stdin')

    def _supports_fd_secret_get(self) -> bool:
        return self._supports_flag(['secret', 'get', '--help'], '--fd')

    def _supports_flag(self, arguments: list[str], flag: str) -> bool:
        try:
            proc = subprocess.run(
                [self.binary, *arguments],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
                env=agent_switch_subprocess_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SecretResolutionError('secret_resolver_unavailable') from exc
        help_text = f'{proc.stdout}\n{proc.stderr}'
        return proc.returncode == 0 and flag in help_text
