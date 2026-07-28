from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any


AGENT_SWITCH = Path.home() / '.local/bin/agent-switch'


class SecretUnavailable(RuntimeError):
    code = 'agent_switch_secret_unavailable'


def read_agent_switch_secret(
    name: str,
    *,
    executable: str | Path = AGENT_SWITCH,
) -> bytes:
    """Read one Agent Switch secret only through an inherited non-TTY fd."""
    if (
        not isinstance(name, str)
        or not name
        or any(char.isspace() for char in name)
    ):
        raise SecretUnavailable('secret name is invalid')
    read_fd, write_fd = os.pipe()
    try:
        process = subprocess.Popen(
            [
                str(Path(executable)),
                'secret',
                'get',
                '--fd',
                str(write_fd),
                name,
            ],
            pass_fds=(write_fd,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=False,
        )
        os.close(write_fd)
        write_fd = -1
        chunks: list[bytes] = []
        while True:
            chunk = os.read(read_fd, 65_536)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            return_code = process.wait(timeout=15)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait(timeout=5)
            raise SecretUnavailable('secret read timed out') from exc
        value = b''.join(chunks)
        if return_code != 0 or not value:
            raise SecretUnavailable('secret is unavailable')
        return value
    except OSError as exc:
        raise SecretUnavailable('Agent Switch is unavailable') from exc
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)


def load_key_store_secret(name: str) -> dict[str, dict[str, Any]]:
    raw = read_agent_switch_secret(name)
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SecretUnavailable('key store secret is invalid') from exc
    finally:
        raw = b''
    if isinstance(payload, dict) and isinstance(payload.get('keys'), dict):
        payload = payload['keys']
    if not isinstance(payload, dict):
        raise SecretUnavailable('key store secret is invalid')
    keys: dict[str, dict[str, Any]] = {}
    for salt, record in payload.items():
        if not isinstance(salt, str) or not isinstance(record, dict):
            continue
        derived = str(record.get('dk') or '')
        if derived:
            keys[salt.lower()] = {
                'dk': derived,
                'rounds': int(record.get('rounds') or 256_000),
            }
    if not keys:
        raise SecretUnavailable('key store secret has no usable keys')
    return keys


__all__ = [
    'AGENT_SWITCH', 'SecretUnavailable', 'load_key_store_secret',
    'read_agent_switch_secret',
]
