from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import struct
import subprocess
import time
from typing import Any, Callable, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen


_FORMAT = 'trove-managed-process'
_VERSION = 1
_NAME_RE = re.compile(r'[a-z][a-z0-9_-]{0,31}')
_NONCE_RE = re.compile(r'[0-9a-f]{32}')
_HASH_RE = re.compile(r'[0-9a-f]{64}')

# A managed child receives only the variables required to locate its runtime,
# Agent Switch, local caches, and explicit product configuration.  In
# particular, arbitrary ambient credentials and provider secret values are not
# inherited.  Secret-name selectors are handled separately below because their
# values are identifiers, not credentials.
_SAFE_BASE_ENV_KEYS = frozenset({
    'COMSPEC',
    'HOME',
    'LANG',
    'LANGUAGE',
    'LC_ALL',
    'LC_CTYPE',
    'LOCALAPPDATA',
    'PATH',
    'PATHEXT',
    'PYTHONIOENCODING',
    'PYTHONNOUSERSITE',
    'PYTHONPATH',
    'PYTHONUNBUFFERED',
    'PYTHONUTF8',
    'SSL_CERT_DIR',
    'SSL_CERT_FILE',
    'SYSTEMROOT',
    'TMP',
    'TMPDIR',
    'TEMP',
    'TZ',
    'VIRTUAL_ENV',
    'WINDIR',
    'XDG_CONFIG_HOME',
    'XDG_DATA_HOME',
})
_SAFE_TROVE_ENV_KEYS = frozenset({
    'TROVE_CLOUD_COST_CAP_RMB',
    'TROVE_CLOUD_EMBEDDING_DIMENSIONS',
    'TROVE_CLOUD_EMBEDDING_ENDPOINT',
    'TROVE_CLOUD_EMBEDDING_MODEL',
    'TROVE_CLOUD_EMBEDDING_PROVIDER',
    'TROVE_CLOUD_EMBEDDING_REQUEST_FORMAT',
    'TROVE_CLOUD_RERANK_ENDPOINT',
    'TROVE_CLOUD_RERANK_MODEL',
    'TROVE_CLOUD_RERANK_PROVIDER',
    'TROVE_CLOUD_RERANK_TOP_K',
    'TROVE_CONSOLE_PORT',
    'TROVE_BUILD_HASH',
    'TROVE_DISABLE_EMBED_DAEMON_CLIENT',
    'TROVE_DISABLE_LOCAL_EMBEDDING',
    'TROVE_DISABLE_SEARCH_WARMUP',
    'TROVE_EMBEDDING_CACHE',
    'TROVE_EMBEDDING_DAEMON_BATCH_REQUESTS',
    'TROVE_EMBEDDING_DAEMON_BATCH_TEXTS',
    'TROVE_EMBEDDING_DAEMON_BATCH_WAIT_MS',
    'TROVE_EMBEDDING_DAEMON_QUEUE_SIZE',
    'TROVE_EMBEDDING_DAEMON_TIMEOUT_MS',
    'TROVE_EMBEDDING_MODEL_PATH',
    'TROVE_EMBEDDING_SOCKET',
    'TROVE_ENABLE_CLOUD_ASR',
    'TROVE_ENABLE_CLOUD_EMBEDDING',
    'TROVE_ENABLE_CLOUD_RERANK',
    'TROVE_ENABLE_CLOUD_VISION',
    'TROVE_LOCAL_VLM_LOCAL_FILES_ONLY',
    'TROVE_MODEL_CACHE',
    'TROVE_REWRITE_SENDER_PREFILTER_LIMIT',
    'TROVE_VAULT_ROOT',
    'TROVE_VOLCENGINE_ARK_BASE_URL',
    'TROVE_VOLCENGINE_ARK_CHAT_PATH',
    'TROVE_VOLCENGINE_ARK_RESPONSES_PATH',
    'TROVE_VOLCENGINE_ARK_VISION_MODEL',
    'TROVE_VOLCENGINE_ASR_ENDPOINT',
    'TROVE_VOLCENGINE_ASR_MODEL_NAME',
    'TROVE_VOLCENGINE_ASR_RESOURCE_ID',
    'TROVE_WECHAT_COST_CAP_RMB',
})
_PROVIDER_SECRET_NAME_ENV_KEYS = frozenset({
    'TROVE_ASR_SECRET_NAME',
    'TROVE_VISION_SECRET_NAME',
    'TROVE_CLOUD_EMBEDDING_KEY_ENV',
    'TROVE_CLOUD_RERANK_KEY_ENV',
})
_DEFAULT_PROVIDER_SECRET_NAMES = frozenset({
    'DASHSCOPE_API_KEY',
    'TROVE_CLOUD_EMBEDDING_KEY',  # legacy constructor name; values remain forbidden
    'VOLCENGINE_ARK_API_KEY',
    'VOLCENGINE_ASR_API_KEY',
})
_SECRETISH_ENV_RE = re.compile(
    r'(?:^|_)(?:API_?KEY|AUTH|BEARER|COOKIE|CREDENTIALS?|KEY|PASS(?:WORD|WD)?|PRIVATE|SECRET|SESSION|SIGNATURE|TOKEN)(?:_|$)',
    re.IGNORECASE,
)


def _valid_env_text(value: object) -> bool:
    return type(value) is str and '\x00' not in value


def _configured_provider_secret_names(source: Mapping[str, str]) -> set[str]:
    names = set(_DEFAULT_PROVIDER_SECRET_NAMES)
    for selector in _PROVIDER_SECRET_NAME_ENV_KEYS:
        candidate = source.get(selector)
        if type(candidate) is str and 1 <= len(candidate) <= 256 and re.fullmatch(r'[A-Za-z0-9_]+', candidate):
            names.add(candidate)
    return names


def _minimal_child_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return the non-secret environment contract for a managed child."""

    configured_secret_names = _configured_provider_secret_names(source)
    child: dict[str, str] = {}
    for name in _SAFE_BASE_ENV_KEYS | _SAFE_TROVE_ENV_KEYS:
        value = source.get(name)
        if (
            name not in configured_secret_names
            and not _SECRETISH_ENV_RE.search(name)
            and _valid_env_text(value)
        ):
            child[name] = value

    # These four values are Agent Switch secret *names*.  Preserve only the
    # identifier grammar; never copy the environment entry named by them.
    for selector in _PROVIDER_SECRET_NAME_ENV_KEYS:
        value = source.get(selector)
        if type(value) is str and 1 <= len(value) <= 256 and re.fullmatch(r'[A-Za-z0-9_]+', value):
            child[selector] = value
    return child


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class ManagedProcessError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': False,
            'error': {'code': self.code, 'message': str(self)},
            'raw_content_included': False,
        }


@dataclass(frozen=True, slots=True)
class ManagedProcessRecord:
    name: str
    pid: int
    birth_time: str
    command_hash: str
    health_endpoint: str
    nonce: str
    created_at: str
    format: str = _FORMAT
    version: int = _VERSION

    @classmethod
    def from_dict(cls, value: object) -> 'ManagedProcessRecord':
        if type(value) is not dict:
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record is invalid.')
        required = {
            'birth_time', 'command_hash', 'created_at', 'format', 'health_endpoint',
            'name', 'nonce', 'pid', 'version',
        }
        if set(value) != required:
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record fields are invalid.')
        try:
            record = cls(**value)
        except TypeError as exc:
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record types are invalid.') from exc
        if (
            record.format != _FORMAT
            or record.version != _VERSION
            or type(record.name) is not str
            or not _NAME_RE.fullmatch(record.name)
            or type(record.pid) is not int
            or record.pid <= 1
            or type(record.birth_time) is not str
            or not 1 <= len(record.birth_time) <= 128
            or type(record.command_hash) is not str
            or not _HASH_RE.fullmatch(record.command_hash)
            or type(record.health_endpoint) is not str
            or not (
                record.health_endpoint.startswith('http://127.0.0.1:')
                or (
                    record.health_endpoint.startswith('unix:///')
                    and '\x00' not in record.health_endpoint
                    and len(os.fsencode(record.health_endpoint[7:])) <= 103
                )
            )
            or len(record.health_endpoint) > 1024
            or type(record.nonce) is not str
            or not _NONCE_RE.fullmatch(record.nonce)
            or type(record.created_at) is not str
            or not 1 <= len(record.created_at) <= 64
        ):
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record values are invalid.')
        return record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProcessInspector:
    """Read-only process identity probes used before every signal."""

    @staticmethod
    def running(pid: int | None) -> bool:
        if type(pid) is not int or pid <= 1:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        # A detached child may remain briefly as a zombie until Python reaps
        # its Popen object.  It is no longer running and must not make bounded
        # daemon drain/replace report a false timeout.
        try:
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'state='],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return True
        state = result.stdout.strip()
        if result.returncode != 0 or not state:
            return True
        return not state.startswith('Z')

    @staticmethod
    def _ps(pid: int, field: str) -> str | None:
        try:
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', f'{field}='],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        value = ' '.join(result.stdout.strip().split())
        return value if result.returncode == 0 and value else None

    def birth_time(self, pid: int) -> str | None:
        return self._ps(pid, 'lstart')

    def command_text(self, pid: int) -> str | None:
        return self._ps(pid, 'command')

    def command_hash(self, pid: int) -> str | None:
        command = self.command_text(pid)
        return hashlib.sha256(command.encode('utf-8')).hexdigest() if command else None


def health_probe(endpoint: str, nonce: str, *, timeout: float = 0.5) -> bool:
    if endpoint.startswith('unix://'):
        return _unix_health_probe(endpoint[7:], nonce, timeout=timeout)
    try:
        request = Request(endpoint, method='GET', headers={'Accept': 'application/json'})
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            raw = response.read(16 * 1024 + 1)
    except (OSError, URLError, TimeoutError, ValueError):
        return False
    if len(raw) > 16 * 1024:
        return False
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError):
        return False
    return type(payload) is dict and payload.get('ok') is True and payload.get('managed_nonce') == nonce


def _unix_health_probe(path: str, nonce: str, *, timeout: float) -> bool:
    if not path.startswith('/') or '\x00' in path or len(os.fsencode(path)) > 103:
        return False
    payload = json.dumps(
        {'type': 'health', 'managed_nonce': nonce},
        ensure_ascii=True, sort_keys=True, separators=(',', ':'),
    ).encode('ascii')
    if len(payload) > 4096:
        return False
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout)
    try:
        connection.connect(path)
        connection.sendall(struct.pack('>I', len(payload)) + payload)
        header = connection.recv(4)
        if len(header) != 4:
            return False
        length = struct.unpack('>I', header)[0]
        if not 1 <= length <= 16 * 1024:
            return False
        raw = bytearray()
        while len(raw) < length:
            chunk = connection.recv(length - len(raw))
            if not chunk:
                return False
            raw.extend(chunk)
        response = json.loads(raw.decode('utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    finally:
        connection.close()
    return (
        isinstance(response, dict)
        and response.get('ok') is True
        and response.get('managed_nonce') == nonce
        and response.get('transport') == 'unix'
    )


class ManagedProcessManager:
    def __init__(
        self,
        logs_dir: Path,
        *,
        inspector: ProcessInspector | None = None,
        probe: Callable[[str, str], bool] = health_probe,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.logs_dir = Path(logs_dir)
        self.inspector = inspector or ProcessInspector()
        self.probe = probe
        self.popen = popen
        self._children: dict[str, subprocess.Popen] = {}

    def record_path(self, name: str) -> Path:
        if type(name) is not str or not _NAME_RE.fullmatch(name):
            raise ValueError('managed process name is invalid')
        return self.logs_dir / f'trove-{name}.pid'

    def log_path(self, name: str) -> Path:
        self.record_path(name)
        return self.logs_dir / f'trove-{name}.log'

    def read(self, name: str) -> ManagedProcessRecord | None:
        path = self.record_path(name)
        try:
            listed = os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ManagedProcessError('managed_process_record_unavailable', 'Managed process record is unavailable.') from exc
        if stat.S_ISLNK(listed.st_mode) or not stat.S_ISREG(listed.st_mode) or listed.st_nlink != 1 or listed.st_mode & 0o077:
            raise ManagedProcessError('managed_process_record_unsafe', 'Managed process record is unsafe.')
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ManagedProcessError('managed_process_record_unavailable', 'Managed process record is unavailable.') from exc
        if len(raw) > 16 * 1024:
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record is too large.')
        try:
            payload = json.loads(raw.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record is invalid.') from exc
        record = ManagedProcessRecord.from_dict(payload)
        if record.name != name:
            raise ManagedProcessError('managed_process_record_invalid', 'Managed process record name does not match.')
        return record

    def _untrusted_pid(self, name: str) -> int | None:
        path = self.record_path(name)
        try:
            raw = path.read_text(encoding='utf-8').strip()
        except Exception:
            return None
        if raw.isascii() and raw.isdecimal():
            value = int(raw)
            return value if value > 1 else None
        try:
            payload = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return None
        value = payload.get('pid') if type(payload) is dict else None
        return value if type(value) is int and value > 1 else None

    def _write(self, record: ManagedProcessRecord) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.logs_dir, 0o700)
        path = self.record_path(record.name)
        temporary = path.with_name(f'.{path.name}.{record.nonce}.tmp')
        payload = (json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
        fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, 'O_NOFOLLOW'):
                flags |= os.O_NOFOLLOW
            fd = os.open(temporary, flags, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError('short managed process record write')
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(temporary, path)
            dir_fd = os.open(self.logs_dir, os.O_RDONLY | int(getattr(os, 'O_DIRECTORY', 0)))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            raise ManagedProcessError('managed_process_record_unavailable', 'Managed process record could not be stored.') from exc
        finally:
            if fd is not None:
                os.close(fd)
            temporary.unlink(missing_ok=True)

    def _identity_matches(self, record: ManagedProcessRecord, *, require_health: bool) -> bool:
        return bool(
            self.inspector.running(record.pid)
            and self.inspector.birth_time(record.pid) == record.birth_time
            and self.inspector.command_hash(record.pid) == record.command_hash
            and (not require_health or self.probe(record.health_endpoint, record.nonce))
        )

    def status(self, name: str) -> dict[str, Any]:
        path = self.record_path(name)
        try:
            record = self.read(name)
        except ManagedProcessError as exc:
            pid = self._untrusted_pid(name)
            return {
                'pid': pid,
                'running': self.inspector.running(pid),
                'responsive': False,
                'identity_verified': False,
                'reason_code': exc.code,
                'raw_content_included': False,
            }
        if record is None:
            return {
                'pid': None,
                'running': False,
                'responsive': False,
                'identity_verified': False,
                'reason_code': 'not_started',
                'raw_content_included': False,
            }
        running = self.inspector.running(record.pid)
        process_match = bool(
            running
            and self.inspector.birth_time(record.pid) == record.birth_time
            and self.inspector.command_hash(record.pid) == record.command_hash
        )
        responsive = bool(process_match and self.probe(record.health_endpoint, record.nonce))
        return {
            'pid': record.pid,
            'running': running,
            'responsive': responsive,
            'identity_verified': bool(process_match and responsive),
            'reason_code': None if process_match and responsive else ('health_unavailable' if process_match else 'identity_mismatch'),
            'health_endpoint': record.health_endpoint,
            'command_hash': record.command_hash,
            'birth_time': record.birth_time,
            'raw_content_included': False,
        }

    def start(
        self,
        name: str,
        command: list[str],
        *,
        health_endpoint: str,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        readiness_timeout: float = 8.0,
    ) -> dict[str, Any]:
        if not command or any(type(part) is not str or not part for part in command):
            raise ValueError('managed process command must be a non-empty string list')
        path = self.record_path(name)
        try:
            existing = self.read(name)
        except ManagedProcessError as exc:
            untrusted_pid = self._untrusted_pid(name)
            if self.inspector.running(untrusted_pid):
                raise ManagedProcessError('managed_process_identity_unverified', 'A live process has an unverified managed record; refusing to replace or signal it.') from exc
            path.unlink(missing_ok=True)
            existing = None
        if existing is not None:
            if self._identity_matches(existing, require_health=True):
                return {'ok': True, 'already_running': True, **self.status(name), 'log': self.log_path(name).name}
            if self.inspector.running(existing.pid):
                raise ManagedProcessError('managed_process_identity_unverified', 'A live process no longer matches its managed identity; refusing to replace or signal it.')
            path.unlink(missing_ok=True)

        self.logs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.logs_dir, 0o700)
        nonce = secrets.token_hex(16)
        source_env = os.environ if env is None else env
        process_env = _minimal_child_environment(source_env)
        process_env['TROVE_MANAGED_NONCE'] = nonce
        log_path = self.log_path(name)
        handle = log_path.open('ab')
        os.chmod(log_path, 0o600)
        try:
            process = self.popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(cwd),
                env=process_env,
                start_new_session=True,
            )
            self._children[name] = process
        finally:
            handle.close()

        record: ManagedProcessRecord | None = None
        deadline = time.monotonic() + max(0.1, float(readiness_timeout))
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ManagedProcessError('managed_process_startup_failed', 'Managed process exited before readiness.')
                birth_time = self.inspector.birth_time(process.pid)
                command_hash = self.inspector.command_hash(process.pid)
                if birth_time and command_hash:
                    record = ManagedProcessRecord(
                        name=name,
                        pid=process.pid,
                        birth_time=birth_time,
                        command_hash=command_hash,
                        health_endpoint=health_endpoint,
                        nonce=nonce,
                        created_at=_now(),
                    )
                    self._write(record)
                    break
                time.sleep(0.05)
            if record is None:
                raise ManagedProcessError('managed_process_identity_unavailable', 'Managed process identity was unavailable before timeout.')
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ManagedProcessError('managed_process_startup_failed', 'Managed process exited before readiness.')
                if self._identity_matches(record, require_health=True):
                    return {'ok': True, 'started': True, **self.status(name), 'log': log_path.name}
                time.sleep(0.05)
            raise ManagedProcessError('managed_process_readiness_failed', 'Managed process did not become ready before timeout.')
        except BaseException:
            # Only the exact Popen handle created in this call is terminated;
            # no pidfile-derived PID participates in startup rollback.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if record is not None:
                try:
                    persisted = self.read(name)
                except ManagedProcessError:
                    persisted = None
                if persisted == record:
                    path.unlink(missing_ok=True)
            self._children.pop(name, None)
            raise

    def stop(self, name: str, *, timeout: float = 5.0) -> dict[str, Any]:
        path = self.record_path(name)
        try:
            record = self.read(name)
        except ManagedProcessError as exc:
            pid = self._untrusted_pid(name)
            if not self.inspector.running(pid):
                path.unlink(missing_ok=True)
            return {
                'ok': False,
                'pid': pid,
                'was_running': self.inspector.running(pid),
                'running': self.inspector.running(pid),
                'identity_verified': False,
                'error': {'code': exc.code, 'message': 'Managed process identity is not verified; no signal was sent.'},
                'raw_content_included': False,
            }
        if record is None:
            return {'ok': True, 'pid': None, 'was_running': False, 'running': False, 'identity_verified': False}
        if not self.inspector.running(record.pid):
            path.unlink(missing_ok=True)
            child = self._children.pop(name, None)
            if child is not None:
                try:
                    child.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            return {'ok': True, 'pid': record.pid, 'was_running': False, 'running': False, 'identity_verified': False}
        if not self._identity_matches(record, require_health=True):
            return {
                'ok': False,
                'pid': record.pid,
                'was_running': True,
                'running': True,
                'identity_verified': False,
                'error': {'code': 'managed_process_identity_unverified', 'message': 'Managed process identity is not verified; no signal was sent.'},
                'raw_content_included': False,
            }
        # Recheck the nonce-bearing health endpoint immediately before the
        # signal.  Birth timestamps can have only one-second resolution on
        # some ``ps`` implementations; birth+command alone cannot distinguish
        # an identical command that reused the PID inside that second.
        if not self._identity_matches(record, require_health=True):
            return {
                'ok': False,
                'pid': record.pid,
                'was_running': True,
                'running': True,
                'identity_verified': False,
                'error': {'code': 'managed_process_identity_changed', 'message': 'Managed process identity changed; no signal was sent.'},
                'raw_content_included': False,
            }
        try:
            os.kill(record.pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + max(0.1, float(timeout))
        while self.inspector.running(record.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        running = self.inspector.running(record.pid)
        if not running:
            path.unlink(missing_ok=True)
            child = self._children.pop(name, None)
            if child is not None:
                try:
                    child.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        return {
            'ok': not running,
            'pid': record.pid,
            'was_running': True,
            'running': running,
            'identity_verified': True,
            'raw_content_included': False,
        }
