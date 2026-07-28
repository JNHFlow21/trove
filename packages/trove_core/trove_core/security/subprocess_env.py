from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


_AGENT_SWITCH_ENV_KEYS = frozenset({
    'HOME',
    'LANG',
    'LC_ALL',
    'LC_CTYPE',
    'PATH',
    'TMPDIR',
    'XDG_CONFIG_HOME',
})


def agent_switch_subprocess_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the minimal non-secret environment for Agent Switch commands."""

    source = os.environ if source is None else source
    env = {
        name: value
        for name in _AGENT_SWITCH_ENV_KEYS
        if type(value := source.get(name)) is str and '\x00' not in value
    }
    # LaunchAgents inherit a minimal PATH that commonly omits the per-user
    # Agent Switch installation. Keep secret resolution on the Agent Switch
    # control plane instead of silently making background jobs fall back to a
    # different provider.
    home = env.get('HOME')
    user_bin = str(Path(home).expanduser() / '.local' / 'bin') if home else None
    inherited_parts = [part for part in env.get('PATH', os.defpath).split(os.pathsep) if part]
    preferred_parts = [
        part for part in (user_bin, '/opt/homebrew/bin', '/usr/local/bin')
        if part
    ]
    path_parts = list(dict.fromkeys([*preferred_parts, *inherited_parts]))
    env['PATH'] = os.pathsep.join(path_parts)
    return env
