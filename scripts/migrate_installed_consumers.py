#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Sequence


TOOL_ID = 'agent-trove'
LEGACY_TOOL_ID = 'agent-trove' + '-wechat'
SKILLCTL = Path.home() / 'AgentWorkspace/skill-hub/scripts/skillctl'
EXPECTED_TOOL = {
    'id': TOOL_ID,
    'name': 'TROVE',
    'command': 'trove-mcp',
    'args': ['--pack', 'standard'],
    'requiredSecrets': [],
    'apps': {'claude': True, 'claude_desktop': True, 'codex': True, 'hermes': True},
    'description': 'Local TROVE v1 MCP standard capability pack.',
}
LEGACY_LAUNCH_AGENT_NAMES = (
    'com.trove.wechat.sync.plist',
    'com.trove.wechat.maintain.plist',
)
LEGACY_SKILL_NAMES = ('trove-chat-recall', 'trove-person-crm')


def discover_installed_consumers(
    *,
    home: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[dict[str, int], tuple[Path, ...]]:
    home = home or Path.home()
    launch_root = home / 'Library/LaunchAgents'
    launch_agents = tuple(
        path for name in LEGACY_LAUNCH_AGENT_NAMES
        if (path := launch_root / name).is_file()
    )
    skill_links = 0
    for parent in ('.agents/skills', '.codex/skills', '.claude/skills', '.hermes/skills'):
        for name in LEGACY_SKILL_NAMES:
            if (home / parent / name).exists() or (home / parent / name).is_symlink():
                skill_links += 1
    schedule_references = 0
    try:
        cron = runner(
            ['crontab', '-l'], check=False, capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
        )
        for line in str(cron.stdout or '').splitlines():
            lowered = line.lower()
            if 'trove' in lowered and any(token in lowered for token in ('trove_cli', 'trove-api', 'chat-recall')):
                schedule_references += 1
    except OSError:
        pass
    return ({
        'legacy_launch_agents': len(launch_agents),
        'legacy_schedule_references': schedule_references,
        'legacy_generated_skill_links': skill_links,
    }, launch_agents)


def _tool_matches_expected(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    command = tool.get('command')
    if not isinstance(command, str) or Path(command).name != 'trove-mcp':
        return False
    expected = dict(EXPECTED_TOOL)
    expected.pop('command')
    actual = dict(tool)
    actual.pop('command', None)
    return actual == expected


def audit(
    config: dict[str, Any],
    *,
    installed_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    tools = config.get('tools') if isinstance(config.get('tools'), list) else []
    matches = [item for item in tools if isinstance(item, dict) and item.get('id') == TOOL_ID]
    current = matches[0] if len(matches) == 1 else None
    counts = {
        'legacy_launch_agents': 0,
        'legacy_schedule_references': 0,
        'legacy_generated_skill_links': 0,
        **(installed_counts or {}),
    }
    matches_expected = len(matches) == 1 and _tool_matches_expected(current)
    legacy_references = (0 if matches_expected else 1) + sum(counts.values())
    return {
        'ok': matches_expected and legacy_references == 0,
        'tool_present_once': len(matches) == 1,
        'requires_migration': not matches_expected or legacy_references > 0,
        'secret_names_after': [],
        'legacy_entrypoint_references': legacy_references,
        **counts,
        'private_paths_included': False,
        'secret_values_included': False,
    }


def migrated_config(config: dict[str, Any], *, command: str = 'trove-mcp') -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    tools = result.get('tools')
    if not isinstance(tools, list):
        raise ValueError('Agent Switch tools must be a list')
    retained = [
        item for item in tools
        if not isinstance(item, dict) or item.get('id') not in {TOOL_ID, LEGACY_TOOL_ID}
    ]
    replacement = dict(EXPECTED_TOOL)
    replacement['command'] = command
    retained.append(replacement)
    result['tools'] = retained
    return result


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(prefix='.agent-switch-', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def apply(
    config_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    executable: str | None = None,
    skillctl: Path = SKILLCTL,
    legacy_launch_agents: Sequence[Path] = (),
    legacy_schedule_references: int = 0,
) -> dict[str, Any]:
    # Doctor must pass before central configuration changes.
    runner(['agent-switch', 'doctor'], check=True, stdout=subprocess.DEVNULL)
    candidate = executable or 'trove-mcp'
    binary = shutil.which(candidate)
    if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError('trove-mcp artifact is not installed')
    if legacy_schedule_references:
        raise RuntimeError('legacy external schedule requires operator migration')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    updated = migrated_config(config, command=str(Path(binary).resolve()))
    _atomic_write(config_path, updated)
    try:
        runner(['agent-switch', 'reconcile'], check=True, stdout=subprocess.DEVNULL)
        if skillctl.is_file():
            runner([str(skillctl), 'sync', 'trove', '--prune'], check=True, stdout=subprocess.DEVNULL)
    except BaseException:
        _atomic_write(config_path, config)
        runner(['agent-switch', 'reconcile'], check=False, stdout=subprocess.DEVNULL)
        raise
    for path in legacy_launch_agents:
        runner(
            ['launchctl', 'bootout', f'gui/{os.getuid()}', str(path)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        path.unlink(missing_ok=True)
    return audit(updated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path.home() / '.config/agent-switch/config.json')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    try:
        installed_counts, launch_agents = discover_installed_consumers()
        if args.apply:
            report = apply(
                args.config,
                legacy_launch_agents=launch_agents,
                legacy_schedule_references=installed_counts['legacy_schedule_references'],
            )
            after_counts, _ = discover_installed_consumers()
            report = audit(
                json.loads(args.config.read_text(encoding='utf-8')),
                installed_counts=after_counts,
            )
        else:
            report = audit(
                json.loads(args.config.read_text(encoding='utf-8')),
                installed_counts=installed_counts,
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        report = {'ok': False, 'error_code': 'installed_consumer_migration_failed', 'private_paths_included': False, 'secret_values_included': False}
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return 0 if report.get('ok') or (not args.apply and report.get('requires_migration')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
