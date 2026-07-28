from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import multiprocessing as mp
import queue
import subprocess
import time
from typing import Any, Callable

from .redaction import stable_hash
from .secrets import AgentSwitchSecretResolver, SecretResolutionError, SecretStore


JS_HOOK = r'''
function buf2hex(buffer) {
    var a = new Uint8Array(buffer); var h = '';
    for (var i = 0; i < a.length; i++) h += ('0' + a[i].toString(16)).slice(-2);
    return h;
}
var found = false;
Process.enumerateModules().forEach(function(m) {
    if (found) return;
    m.enumerateExports().forEach(function(exp) {
        if (found) return;
        if (exp.name === "CCKeyDerivationPBKDF") {
            found = true;
            send({kind: "status", code: "hook_installed", module: m.name});
            Interceptor.attach(exp.address, {
                onEnter: function(args) {
                    this.saltLen = args[4].toInt32();
                    this.rounds = args[6].toInt32();
                    this.salt = args[3];
                    this.dk = args[7];
                    this.dkLen = args[8].toInt32();
                },
                onLeave: function(retval) {
                    if (this.saltLen !== 16) return;
                    if (this.dkLen !== 32) return;
                    send({
                        kind: "pbkdf2",
                        rounds: this.rounds,
                        salt: buf2hex(this.salt.readByteArray(this.saltLen)),
                        dk: buf2hex(this.dk.readByteArray(this.dkLen))
                    });
                }
            });
        }
    });
});
if (!found) send({kind: "status", code: "hook_not_found"});
'''


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class KeyCaptureConfig:
    secret_name: str = 'TROVE_WECHAT_KEY_STORE'
    wechat_app: str | None = None
    bundle_id: str | None = None
    mode: str = 'spawn'
    process_names: tuple[str, ...] = ('WeChat', 'WeChatAppEx')
    wait_seconds: int = 300
    min_usable_keys: int = 2
    settle_seconds: int = 8
    attach_attempts: int = 3
    attach_retry_delay: int = 2
    agent_switch_binary: str = 'agent-switch'
    capture_timeout_seconds: int | None = None
    activate_spawned: bool = True


def plist_value(app_path: Path, key: str) -> str:
    info_plist = app_path / 'Contents/Info.plist'
    if not info_plist.exists():
        return ''
    try:
        proc = subprocess.run(
            ['/usr/bin/defaults', 'read', str(info_plist.with_suffix('')), key],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return ''
    return proc.stdout.strip() if proc.returncode == 0 else ''


def executable_path(app_path: Path) -> Path | None:
    executable_name = plist_value(app_path, 'CFBundleExecutable') or 'WeChat'
    candidate = app_path / 'Contents/MacOS' / executable_name
    if candidate.exists():
        return candidate
    fallback = app_path / 'Contents/MacOS/WeChat'
    if fallback.exists():
        return fallback
    return None


def looks_like_wechat(app_path: Path) -> bool:
    bundle_id = plist_value(app_path, 'CFBundleIdentifier')
    executable = executable_path(app_path)
    name = app_path.name.lower()
    return bool(executable) and (
        'wechat' in name
        or '微信' in app_path.name
        or 'xinwechat' in bundle_id.lower()
        or 'wechat' in bundle_id.lower()
    )


def discover_wechat_apps() -> list[dict[str, str]]:
    roots = [Path('/Applications'), Path.home() / 'Applications', Path.home() / 'Desktop']
    apps: dict[str, dict[str, str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for app_path in sorted(root.glob('*.app')):
            if not looks_like_wechat(app_path):
                continue
            executable = executable_path(app_path)
            bundle_id = plist_value(app_path, 'CFBundleIdentifier')
            display_name = plist_value(app_path, 'CFBundleDisplayName') or plist_value(app_path, 'CFBundleName') or app_path.stem
            apps[str(app_path)] = {
                'app_ref': stable_hash(str(app_path), length=16),
                'bundle_id': bundle_id,
                'display_name': display_name,
                'executable_ref': stable_hash(str(executable), length=16) if executable else '',
            }
    return sorted(apps.values(), key=lambda item: (item.get('bundle_id') or '', item['app_ref']))


def _resolve_wechat_app(bundle_id: str | None = None, app_path: str | None = None) -> Path:
    if app_path:
        return Path(app_path).expanduser()
    candidates = [Path('/Applications/WeChat.app'), Path.home() / 'Applications/WeChat.app', Path.home() / 'Desktop/WeChat.app']
    if bundle_id:
        for root in [Path('/Applications'), Path.home() / 'Applications', Path.home() / 'Desktop']:
            if not root.exists():
                continue
            for candidate in sorted(root.glob('*.app')):
                if plist_value(candidate, 'CFBundleIdentifier') == bundle_id:
                    return candidate
        raise FileNotFoundError('wechat_app_not_found')
    for candidate in candidates:
        if (candidate / 'Contents/MacOS').exists():
            return candidate
    for root in [Path('/Applications'), Path.home() / 'Applications', Path.home() / 'Desktop']:
        if not root.exists():
            continue
        for candidate in sorted(root.glob('*.app')):
            if looks_like_wechat(candidate):
                return candidate
    raise FileNotFoundError('wechat_app_not_found')


def resolve_wechat_executable(bundle_id: str | None = None, app_path: str | None = None) -> Path:
    app = _resolve_wechat_app(bundle_id=bundle_id, app_path=app_path)
    executable = executable_path(app)
    if executable:
        return executable
    return app / 'Contents/MacOS/WeChat'


def _usable_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        rounds = int(payload.get('rounds'))
    except (TypeError, ValueError):
        return None
    salt = str(payload.get('salt') or '').strip().lower()
    dk = str(payload.get('dk') or '').strip().lower()
    if rounds != 256000:
        return None
    if len(salt) != 32 or len(dk) != 64:
        return None
    if not all(c in '0123456789abcdef' for c in salt + dk):
        return None
    now = _now()
    return {
        'rounds': rounds,
        'dk': dk,
        'first_seen_at': now,
        'last_seen_at': now,
        'source': 'local_frida_pbkdf2_capture',
    }


def parse_key_store_secret(value: str | None) -> dict[str, dict[str, Any]]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    raw = payload.get('keys') if isinstance(payload, dict) else payload
    if not isinstance(raw, dict):
        return {}
    keys: dict[str, dict[str, Any]] = {}
    for salt, item in raw.items():
        if not isinstance(item, dict):
            continue
        salt_text = str(salt).lower()
        dk = str(item.get('dk') or '').lower()
        if len(salt_text) == 32 and len(dk) == 64:
            keys[salt_text] = dict(item)
    return keys


def build_key_store_secret_value(keys: dict[str, dict[str, Any]]) -> str:
    return json.dumps({
        'schema': 'trove.wechat.key_store.v1',
        'updated_at': _now(),
        'warning': 'Sensitive local WeChat SQLCipher derived keys. Store only in Agent Switch; never commit or share.',
        'keys': dict(sorted(keys.items())),
    }, ensure_ascii=False, separators=(',', ':'))


def _activate_process_by_pid(pid: int) -> None:
    script = f'tell application "System Events" to set frontmost of first process whose unix id is {int(pid)} to true'
    try:
        subprocess.run(
            ['/usr/bin/osascript', '-e', script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        pass

def _matching_processes(device: Any, names: set[str]) -> list[int]:
    lowered = {name.lower() for name in names}
    pids: list[int] = []
    for process in device.enumerate_processes():
        if str(process.name).lower() in lowered:
            pids.append(int(process.pid))
    return sorted(set(pids))


def _wait_for_processes(device: Any, names: set[str], timeout: int) -> list[int]:
    start = time.time()
    while time.time() - start < timeout:
        pids = _matching_processes(device, names)
        if pids:
            return pids
        time.sleep(1)
    return []


def _attach_with_retry(device: Any, pid: int, attempts: int, delay: int) -> Any:
    last_error: Exception | None = None
    for _attempt in range(1, max(1, attempts) + 1):
        try:
            return device.attach(pid)
        except Exception as exc:  # pragma: no cover - depends on macOS runtime permissions
            last_error = exc
            time.sleep(max(0, delay))
    raise RuntimeError('frida_attach_failed') from last_error


def capture_key_store(config: KeyCaptureConfig, *, status_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Capture local WeChat PBKDF2 derived keys without writing secret values to disk."""
    try:
        import frida  # type: ignore[import-not-found]
    except ImportError as exc:
        return {'ok': False, 'status': 'frida_missing', 'install_hint': 'Install optional dependency: python -m pip install frida-tools', 'raw_content_included': False, 'raw_paths_included': False}

    status_callback = status_callback or (lambda _event: None)
    captured: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    started = time.time()
    device: Any | None = None
    spawned_pid: int | None = None
    spawned_resumed = False
    sessions: list[Any] = []

    def resume_spawned() -> None:
        nonlocal spawned_resumed
        if device is None or spawned_pid is None or spawned_resumed:
            return
        try:
            device.resume(spawned_pid)
            spawned_resumed = True
            if config.activate_spawned:
                time.sleep(1)
                _activate_process_by_pid(spawned_pid)
        except Exception:
            pass

    def record_event(event: dict[str, Any]) -> None:
        safe = {k: v for k, v in event.items() if k not in {'salt', 'dk'}}
        events.append(safe)
        status_callback(safe)

    def on_message(msg: dict[str, Any], _data: Any) -> None:
        if msg.get('type') == 'send':
            payload = msg.get('payload')
            if isinstance(payload, dict) and payload.get('kind') == 'pbkdf2':
                record = _usable_record(payload)
                if record:
                    salt = str(payload['salt']).lower()
                    previous = captured.get(salt, {})
                    if previous.get('first_seen_at'):
                        record['first_seen_at'] = previous['first_seen_at']
                    captured[salt] = record
                    record_event({'kind': 'pbkdf2', 'usable_keys': len(captured)})
            elif isinstance(payload, dict):
                record_event({'kind': 'status', 'code': payload.get('code')})
            else:
                record_event({'kind': 'status', 'code': 'message'})
        elif msg.get('type') == 'error':
            record_event({'kind': 'error', 'code': 'frida_script_error'})

    try:
        device = frida.get_local_device()
        bin_path = resolve_wechat_executable(bundle_id=config.bundle_id, app_path=config.wechat_app)
        spawn_target = config.bundle_id if config.bundle_id and not config.wechat_app else str(bin_path)
        if config.mode == 'spawn':
            if not bin_path.exists():
                return {'ok': False, 'status': 'wechat_binary_missing', 'raw_content_included': False, 'raw_paths_included': False}
            spawned_pid = int(device.spawn([spawn_target]))
            target_pids = [spawned_pid]
        elif config.mode == 'attach':
            names = set(config.process_names or ('WeChat', 'WeChatAppEx'))
            target_pids = _wait_for_processes(device, names, timeout=min(max(config.wait_seconds, 1), 120))
            if not target_pids:
                return {'ok': False, 'status': 'wechat_process_not_found', 'raw_content_included': False, 'raw_paths_included': False}
        else:
            return {'ok': False, 'status': 'invalid_mode', 'raw_content_included': False, 'raw_paths_included': False}
        for pid in target_pids[:8]:
            try:
                session = _attach_with_retry(device, pid, attempts=config.attach_attempts, delay=config.attach_retry_delay)
                script = session.create_script(JS_HOOK)
                script.on('message', on_message)
                script.load()
                sessions.append(session)
            except Exception:
                record_event({'kind': 'status', 'code': 'attach_failed'})
        if not sessions:
            resume_spawned()
            return {'ok': False, 'status': 'frida_attach_failed', 'raw_content_included': False, 'raw_paths_included': False}
        if spawned_pid is not None:
            resume_spawned()
        enough_seen_at: float | None = None
        while time.time() - started < max(1, config.wait_seconds):
            time.sleep(1)
            if config.min_usable_keys > 0 and len(captured) >= config.min_usable_keys:
                if enough_seen_at is None:
                    enough_seen_at = time.time()
                elif time.time() - enough_seen_at >= max(0, config.settle_seconds):
                    break
        for session in sessions:
            try:
                session.detach()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - depends on local app/runtime permissions
        resume_spawned()
        for session in sessions:
            try:
                session.detach()
            except Exception:
                pass
        return {'ok': False, 'status': 'capture_failed', 'error_code': exc.__class__.__name__, 'events': events[-10:], 'keys_captured': len(captured), 'raw_content_included': False, 'raw_paths_included': False}

    return {
        'ok': bool(captured),
        'status': 'captured' if captured else 'no_keys_captured',
        'keys': captured,
        'keys_captured': len(captured),
        'events': events[-10:],
        'elapsed_ms': round((time.time() - started) * 1000, 3),
        'raw_content_included': False,
        'raw_paths_included': False,
    }



def _capture_worker(config: KeyCaptureConfig, result_queue: Any) -> None:
    devnull_fd: int | None = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    except OSError:
        devnull_fd = None
    try:
        result_queue.put(capture_key_store(config))
    except BaseException as exc:  # pragma: no cover - defensive isolation boundary
        result_queue.put({'ok': False, 'status': 'capture_worker_failed', 'error_code': exc.__class__.__name__, 'raw_content_included': False, 'raw_paths_included': False})
    finally:
        try:
            import frida  # type: ignore[import-not-found]
            shutdown = getattr(frida, 'shutdown', None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


def capture_key_store_isolated(config: KeyCaptureConfig) -> dict[str, Any]:
    """Run Frida capture behind a killable process boundary.

    Frida attach/spawn can block inside native code when macOS debugging
    permissions are missing. Keeping it in a child process preserves the CLI
    contract: every attempt returns redacted JSON instead of hanging forever.
    """
    timeout = config.capture_timeout_seconds or (max(1, config.wait_seconds) + 30)
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_capture_worker, args=(config, result_queue), daemon=True)
    proc.start()
    proc.join(timeout=max(1, timeout))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():  # pragma: no cover - depends on platform process control
            proc.kill()
            proc.join(timeout=5)
        return {'ok': False, 'status': 'capture_timeout', 'timeout_seconds': timeout, 'raw_content_included': False, 'raw_paths_included': False}
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return {'ok': False, 'status': 'capture_no_result', 'exitcode': proc.exitcode, 'raw_content_included': False, 'raw_paths_included': False}
    if isinstance(result, dict):
        return result
    return {'ok': False, 'status': 'capture_invalid_result', 'raw_content_included': False, 'raw_paths_included': False}

def capture_and_store_key_store(config: KeyCaptureConfig, *, secret_store: SecretStore | None = None) -> dict[str, Any]:
    capture = capture_key_store_isolated(config)
    keys = capture.pop('keys', {})
    if not capture.get('ok'):
        return {**capture, 'secret_name': config.secret_name, 'secret_written': False}
    resolver = secret_store or AgentSwitchSecretResolver(binary=config.agent_switch_binary)
    existing: dict[str, dict[str, Any]] = {}
    try:
        existing = parse_key_store_secret(resolver.get_secret(config.secret_name))
    except SecretResolutionError:
        existing = {}
    now = _now()
    merged = dict(existing)
    for salt, item in keys.items():
        previous = merged.get(salt, {})
        merged[salt] = {
            **item,
            'first_seen_at': previous.get('first_seen_at') or item.get('first_seen_at') or now,
            'last_seen_at': now,
        }
    secret_value = build_key_store_secret_value(merged)
    try:
        resolver.set_secret(config.secret_name, secret_value)
    except SecretResolutionError as exc:
        return {**capture, 'ok': False, 'status': exc.code, 'secret_name': config.secret_name, 'secret_written': False, 'total_keys_after_merge': len(merged)}
    return {
        **capture,
        'status': 'stored',
        'secret_name': config.secret_name,
        'secret_written': True,
        'new_keys_captured': len(keys),
        'total_keys_after_merge': len(merged),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
