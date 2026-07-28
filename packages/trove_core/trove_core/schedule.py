from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import getpass
import os
import plistlib
import re
import subprocess
import time
from typing import Any
import sys

from trove_core.vault.config import VaultConfig

SAFETY_NOTE = 'TROVE read-only WeChat snapshot sync + local index maintenance; no message sending.'
SYNC_LABEL = 'com.trove.wechat.sync'
MAINTAIN_LABEL = 'com.trove.wechat.maintain'


@dataclass(frozen=True)
class DualAccountScheduleOptions:
    sync_interval: str = '5m'
    config_path: Path | None = None
    runtime_python: Path | None = None
    dry_run: bool = False
    output_dir: Path | None = None


@dataclass(frozen=True)
class ScheduleInstallOptions:
    sync_interval: str = '1h'
    maintain_at: str = '03:00'
    watch: bool = False
    dry_run: bool = False
    output_dir: Path | None = None
    realtime_config: Path | None = None


@dataclass(frozen=True)
class ScheduleReport:
    ok: bool
    action: str
    dry_run: bool
    labels: list[str]
    files: list[str]
    installed: bool
    launchctl: dict[str, Any]
    safety_note: str = SAFETY_NOTE
    raw_content_included: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'pyproject.toml').exists() and (parent / 'scripts' / 'trove-python').exists():
            return parent
    return here.parents[4]


def trove_python_path() -> Path:
    return repo_root() / 'scripts' / 'trove-python'


def parse_interval_seconds(value: str) -> int:
    text = str(value or '').strip().lower()
    match = re.fullmatch(r'(\d+)\s*([smhd]?)', text)
    if not match:
        raise ValueError('sync interval must look like 3600, 60m, or 1h')
    amount = int(match.group(1))
    unit = match.group(2) or 's'
    factor = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}[unit]
    seconds = amount * factor
    if seconds < 60:
        raise ValueError('sync interval must be at least 60 seconds')
    return seconds


def parse_clock(value: str) -> dict[str, int]:
    match = re.fullmatch(r'(\d{1,2}):(\d{2})', str(value or '').strip())
    if not match:
        raise ValueError('maintain time must be HH:MM')
    hour = int(match.group(1))
    minute = int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError('maintain time must be a valid HH:MM')
    return {'Hour': hour, 'Minute': minute}


def launch_agents_dir() -> Path:
    return Path.home() / 'Library' / 'LaunchAgents'


def target_dir(cfg: VaultConfig, options: ScheduleInstallOptions) -> Path:
    if options.output_dir is not None:
        return options.output_dir.expanduser()
    if options.dry_run:
        return cfg.paths.jobs_dir / 'launchd-dry-run'
    return launch_agents_dir()


def build_sync_plist(cfg: VaultConfig, options: ScheduleInstallOptions) -> dict[str, Any]:
    if options.realtime_config is not None:
        args = [
            str(trove_python_path()), '-m', 'trove_cli.main', '--vault', str(cfg.root),
            'realtime-sync', '--config', str(options.realtime_config.expanduser()), '--json',
        ]
    else:
        args = [str(trove_python_path()), '-m', 'trove_cli.main', '--vault', str(cfg.root), 'sync', '--json']
    realtime_watch = options.realtime_config is not None or options.watch
    if realtime_watch:
        args.append('--watch')
    plist: dict[str, Any] = {
        'Label': SYNC_LABEL,
        'ProgramArguments': args,
        'RunAtLoad': True,
        'StandardOutPath': str(cfg.paths.logs_dir / 'launchd-sync.log'),
        'StandardErrorPath': str(cfg.paths.logs_dir / 'launchd-sync.err.log'),
        'EnvironmentVariables': {
            'TROVE_SAFETY_NOTE': SAFETY_NOTE,
        },
    }
    if options.realtime_config is not None:
        plist['EnvironmentVariables']['TROVE_SYNC_SNAPSHOT_MEDIA_ENABLED'] = '0'
    if realtime_watch:
        plist['KeepAlive'] = True
    else:
        plist['StartInterval'] = parse_interval_seconds(options.sync_interval)
    return plist


def build_dual_account_sync_plist(
    cfg: VaultConfig,
    options: DualAccountScheduleOptions,
) -> dict[str, Any]:
    config_path = options.config_path or cfg.paths.jobs_dir / 'dual_account_sync.private.json'
    runtime_python = options.runtime_python or Path(sys.executable)
    return {
        'Label': SYNC_LABEL,
        'ProgramArguments': [
            str(runtime_python), '-m', 'trove_core.jobs.dual_account_sync',
            '--vault', str(cfg.root), '--config', str(config_path),
        ],
        'RunAtLoad': True,
        'StartInterval': parse_interval_seconds(options.sync_interval),
        'ProcessType': 'Background',
        'LowPriorityIO': True,
        'ThrottleInterval': 60,
        'ExitTimeOut': 30,
        'StandardOutPath': str(cfg.paths.logs_dir / 'launchd-sync.log'),
        'StandardErrorPath': str(cfg.paths.logs_dir / 'launchd-sync.err.log'),
        'EnvironmentVariables': {
            'TROVE_SAFETY_NOTE': SAFETY_NOTE,
            'PATH': '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin',
        },
    }


def install_dual_account_schedule(
    vault_root: str | Path,
    *,
    options: DualAccountScheduleOptions | None = None,
) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    cfg.ensure()
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    options = options or DualAccountScheduleOptions()
    out_dir = (
        options.output_dir.expanduser()
        if options.output_dir is not None
        else (cfg.paths.jobs_dir / 'launchd-dry-run' if options.dry_run else launch_agents_dir())
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{SYNC_LABEL}.plist'
    path.write_text(
        plist_text(build_dual_account_sync_plist(cfg, options)),
        encoding='utf-8',
    )
    launchctl = {'ran': False, 'commands': []}
    installed = False
    if not options.dry_run:
        launchctl = bootstrap_launch_agents([path])
        installed = bool(launchctl.get('ok'))
    return ScheduleReport(
        ok=True if options.dry_run else installed,
        action='install',
        dry_run=options.dry_run,
        labels=[SYNC_LABEL],
        files=[path.name],
        installed=installed,
        launchctl=launchctl,
    ).to_dict()


def build_maintain_plist(cfg: VaultConfig, options: ScheduleInstallOptions) -> dict[str, Any]:
    return {
        'Label': MAINTAIN_LABEL,
        'ProgramArguments': [str(trove_python_path()), '-m', 'trove_cli.main', '--vault', str(cfg.root), 'maintain', '--json'],
        'RunAtLoad': False,
        'StartCalendarInterval': parse_clock(options.maintain_at),
        'StandardOutPath': str(cfg.paths.logs_dir / 'launchd-maintain.log'),
        'StandardErrorPath': str(cfg.paths.logs_dir / 'launchd-maintain.err.log'),
        'EnvironmentVariables': {
            'TROVE_SAFETY_NOTE': SAFETY_NOTE,
        },
    }


def plist_text(payload: dict[str, Any]) -> str:
    raw = plistlib.dumps(payload, sort_keys=False).decode('utf-8')
    comment = f'<!-- {SAFETY_NOTE} -->\n'
    return raw.replace('<plist version="1.0">\n', '<plist version="1.0">\n' + comment, 1)


def install_schedule(vault_root: str | Path | None = None, *, options: ScheduleInstallOptions | None = None) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root) if vault_root is not None else None, env={} if vault_root is not None else None)
    cfg.ensure()
    cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    options = options or ScheduleInstallOptions()
    out_dir = target_dir(cfg, options)
    out_dir.mkdir(parents=True, exist_ok=True)
    plists = {
        f'{SYNC_LABEL}.plist': build_sync_plist(cfg, options),
        f'{MAINTAIN_LABEL}.plist': build_maintain_plist(cfg, options),
    }
    files: list[str] = []
    for name, payload in plists.items():
        (out_dir / name).write_text(plist_text(payload), encoding='utf-8')
        files.append(name)
    launchctl = {'ran': False, 'commands': []}
    installed = False
    if not options.dry_run:
        launchctl = bootstrap_launch_agents([out_dir / name for name in files])
        installed = launchctl.get('ok', False)
    return ScheduleReport(
        ok=True if options.dry_run else bool(installed),
        action='install',
        dry_run=options.dry_run,
        labels=[SYNC_LABEL, MAINTAIN_LABEL],
        files=files,
        installed=installed,
        launchctl=launchctl,
    ).to_dict()


def uninstall_schedule(*, dry_run: bool = False, output_dir: Path | None = None) -> dict[str, Any]:
    out_dir = output_dir.expanduser() if output_dir is not None else launch_agents_dir()
    files = [out_dir / f'{SYNC_LABEL}.plist', out_dir / f'{MAINTAIN_LABEL}.plist']
    launchctl = {'ran': False, 'commands': []}
    removed: list[str] = []
    if not dry_run:
        launchctl = bootout_launch_agents([SYNC_LABEL, MAINTAIN_LABEL])
        for path in files:
            try:
                path.unlink()
                removed.append(path.name)
            except FileNotFoundError:
                pass
    else:
        removed = [path.name for path in files if path.exists()]
    return ScheduleReport(
        ok=True if dry_run else bool(launchctl.get('ok', False)),
        action='uninstall',
        dry_run=dry_run,
        labels=[SYNC_LABEL, MAINTAIN_LABEL],
        files=removed,
        installed=False,
        launchctl=launchctl,
    ).to_dict()


def schedule_status(*, output_dir: Path | None = None) -> dict[str, Any]:
    out_dir = output_dir.expanduser() if output_dir is not None else launch_agents_dir()
    files = {label: (out_dir / f'{label}.plist').exists() for label in (SYNC_LABEL, MAINTAIN_LABEL)}
    launchctl = list_launchctl_labels([SYNC_LABEL, MAINTAIN_LABEL])
    return {
        'ok': True,
        'action': 'status',
        'labels': [SYNC_LABEL, MAINTAIN_LABEL],
        'files_present': files,
        'launchctl': launchctl,
        'safety_note': SAFETY_NOTE,
        'raw_content_included': False,
    }


def gui_domain() -> str:
    return f'gui/{os.getuid()}'


def bootstrap_launch_agents(paths: list[Path]) -> dict[str, Any]:
    commands = []
    ok = True
    for path in paths:
        label = path.stem
        target = f'{gui_domain()}/{label}'
        enable = subprocess.run(
            ['launchctl', 'enable', target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        bootout = subprocess.run(
            ['launchctl', 'bootout', target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        bootstrap_attempts = 0
        while True:
            bootstrap_attempts += 1
            bootstrap = subprocess.run(
                ['launchctl', 'bootstrap', gui_domain(), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if bootstrap.returncode != 5 or bootstrap_attempts >= 5:
                break
            time.sleep(0.25)
        commands.append({
            'label_file': path.name,
            'enable_returncode': enable.returncode,
            'bootout_returncode': bootout.returncode,
            'returncode': bootstrap.returncode,
            'bootstrap_attempts': bootstrap_attempts,
            'stderr_bytes': len(bootstrap.stderr.encode('utf-8')) if bootstrap.stderr else 0,
        })
        ok = ok and enable.returncode == 0 and bootout.returncode in {0, 3, 113} and bootstrap.returncode == 0
    return {'ran': True, 'ok': ok, 'domain': gui_domain(), 'commands': commands}


def bootout_launch_agents(labels: list[str]) -> dict[str, Any]:
    commands = []
    ok = True
    for label in labels:
        target = f'{gui_domain()}/{label}'
        proc = subprocess.run(['launchctl', 'bootout', target], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        commands.append({'label': label, 'returncode': proc.returncode, 'stderr_bytes': len(proc.stderr.encode('utf-8')) if proc.stderr else 0})
        ok = ok and proc.returncode in {0, 3, 113}
    return {'ran': True, 'ok': ok, 'domain': gui_domain(), 'commands': commands}


def list_launchctl_labels(labels: list[str]) -> dict[str, Any]:
    if os.environ.get('TROVE_SKIP_LAUNCHCTL_STATUS') == '1':
        return {'ran': False, 'reason': 'skipped'}
    found: dict[str, bool] = {}
    for label in labels:
        proc = subprocess.run(['launchctl', 'print', f'{gui_domain()}/{label}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        found[label] = proc.returncode == 0
    return {'ran': True, 'domain': gui_domain(), 'loaded': found, 'user': getpass.getuser()}
