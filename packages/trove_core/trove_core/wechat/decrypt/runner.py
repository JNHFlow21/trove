from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import time
from typing import Protocol, Any

from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint

from .config import DecryptFilePlan, DecryptPlan
from .manifest import MANIFEST_NAME, write_account_identity, write_guard, write_manifest
from .path_safety import require_existing_under, require_output_under
from .redaction import redact_text, stable_hash
from .secrets import AgentSwitchSecretResolver, SecretResolutionError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def _own_wxid_from_account_root(path: Path) -> str:
    # Match the same boundary used by the historical importer. Container
    # account directories may append an underscore suffix that is not part of
    # the WeChat id; consuming it silently turns outgoing messages incoming.
    match = re.search(r'wxid_[A-Za-z0-9]+', path.name)
    return match.group(0) if match else ''


@dataclass(frozen=True)
class DecryptFileResult:
    account_ref_hash: str
    file_name: str
    file_family: str
    status: str
    error_code: str | None = None
    output_relative: str | None = None
    source_path_hash: str | None = None
    source_version_hash: str | None = None
    row_count: int | None = None
    schema_fingerprint: str | None = None
    stdout: str = ''
    stderr: str = ''
    raw_paths_included: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        output_relative = str(data.pop('output_relative') or '')
        data['output_file_name'] = Path(output_relative).name if output_relative else None
        data['stdout'] = redact_text(data.get('stdout') or '')
        data['stderr'] = redact_text(data.get('stderr') or '')
        data['raw_paths_included'] = False
        return data


class DecryptEngine(Protocol):
    def decrypt(self, source: Path, dest: Path, *, key: str | None, file_family: str) -> DecryptFileResult:
        ...


def sqlite_readable(path: Path) -> bool:
    try:
        uri = path.resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute('SELECT name FROM sqlite_master LIMIT 1').fetchone()
        return True
    except sqlite3.DatabaseError:
        return False


def sqlite_schema_fingerprint(path: Path) -> tuple[int | None, str | None]:
    try:
        uri = path.resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as conn:
            rows = list(conn.execute("SELECT name,sql FROM sqlite_master WHERE type IN ('table','index') ORDER BY name"))
            table_count = sum(1 for name, _sql in rows if name)
            fp = stable_hash('|'.join(f'{name}:{sql}' for name, sql in rows), length=20)
            return table_count, fp
    except sqlite3.DatabaseError:
        return None, None


class SQLCipherCLIEngine:
    """Decrypt using sqlcipher without putting key in argv/env/logs."""

    def __init__(self, *, binary: str = 'sqlcipher', allow_plaintext_copy: bool = True):
        self.binary = binary
        self.allow_plaintext_copy = allow_plaintext_copy

    @staticmethod
    def _sql_quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def decrypt(self, source: Path, dest: Path, *, key: str | None, file_family: str) -> DecryptFileResult:
        if file_family == 'media_kvdb':
            return DecryptFileResult('', source.name, file_family, 'skipped', error_code='unsupported_format')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.allow_plaintext_copy and sqlite_readable(source):
            shutil.copy2(source, dest)
            row_count, fp = sqlite_schema_fingerprint(dest)
            return DecryptFileResult('', source.name, file_family, 'copied_plaintext', output_relative=dest.name, row_count=row_count, schema_fingerprint=fp)
        if not key:
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='missing_key')
        if shutil.which(self.binary) is None:
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='decrypt_tool_missing')
        tmp = dest.with_suffix(dest.suffix + '.tmp')
        tmp.unlink(missing_ok=True)
        sql = '\n'.join([
            "PRAGMA " + "key = " + self._sql_quote(key) + ";",
            f"ATTACH DATABASE {self._sql_quote(str(tmp))} AS plaintext KEY '';",
            "SELECT sqlcipher_export('plaintext');",
            "DETACH DATABASE plaintext;",
            ".quit",
        ])
        started = time.time()
        try:
            proc = subprocess.run(
                [self.binary, str(source)],
                input=sql,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=120,
                env={k: v for k, v in os.environ.items() if 'KEY' not in k.upper() and 'SECRET' not in k.upper() and 'TOKEN' not in k.upper()},
            )
        except subprocess.TimeoutExpired:
            tmp.unlink(missing_ok=True)
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='decrypt_timeout')
        if proc.returncode != 0 or not tmp.exists() or not sqlite_readable(tmp):
            tmp.unlink(missing_ok=True)
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='wrong_key' if proc.returncode == 0 else 'decrypt_failed', stdout=redact_text(proc.stdout), stderr=redact_text(proc.stderr))
        os.replace(tmp, dest)
        row_count, fp = sqlite_schema_fingerprint(dest)
        return DecryptFileResult('', source.name, file_family, 'decrypted', output_relative=dest.name, row_count=row_count, schema_fingerprint=fp, stdout='', stderr='')


class WeChatWCDBAESKeyStoreEngine:
    """Decrypt WeChat WCDB-style SQLCipher files from salt->dk key-store data.

    The key store is read in-process and never included in reports. It maps the
    first 16 bytes of the encrypted DB (salt) to a 32-byte derived key hex.
    The store can come from a local private file or from an Agent Switch secret
    value passed through the existing secret-name resolver.
    """

    page_size = 4096
    reserve = 80
    iv_len = 16
    sqlite_header = b'SQLite format 3\x00'

    def __init__(self, key_store_path: Path | None = None, *, fallback: DecryptEngine | None = None):
        self.key_store_path = Path(key_store_path).expanduser() if key_store_path else None
        self.fallback = fallback or SQLCipherCLIEngine()
        self._file_keys: dict[str, str] | None = None
        self._secret_key_cache: dict[str, dict[str, str]] = {}

    @staticmethod
    def _keys_from_payload(payload: Any) -> dict[str, str]:
        raw = payload.get('keys') if isinstance(payload, dict) else payload
        keys: dict[str, str] = {}
        if isinstance(raw, dict):
            for salt, item in raw.items():
                if not isinstance(item, dict):
                    continue
                dk = str(item.get('dk') or '').strip().lower()
                salt_text = str(salt).strip().lower()
                if len(salt_text) == 32 and len(dk) == 64:
                    keys[salt_text] = dk
        return keys

    def _load_file_keys(self) -> dict[str, str]:
        if self._file_keys is not None:
            return self._file_keys
        if self.key_store_path is None:
            self._file_keys = {}
            return self._file_keys
        try:
            payload = json.loads(self.key_store_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            self._file_keys = {}
            return self._file_keys
        self._file_keys = self._keys_from_payload(payload)
        return self._file_keys

    def _load_secret_keys(self, key: str | None) -> dict[str, str]:
        if not key or not key.lstrip().startswith(('{', '[')):
            return {}
        if key in self._secret_key_cache:
            return self._secret_key_cache[key]
        try:
            payload = json.loads(key)
        except json.JSONDecodeError:
            self._secret_key_cache[key] = {}
            return {}
        parsed = self._keys_from_payload(payload)
        self._secret_key_cache[key] = parsed
        return parsed

    def _load_keys(self, key: str | None = None) -> dict[str, str]:
        keys = dict(self._load_file_keys())
        keys.update(self._load_secret_keys(key))
        return keys

    @classmethod
    def _decrypt_to_file(cls, source: Path, destination: Path, dk_hex: str) -> None:
        """Decrypt one WCDB database with a fixed one-page memory ceiling."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        enc_key = bytes.fromhex(dk_hex)
        with source.open('rb') as encrypted, destination.open('wb') as plaintext:
            index = 0
            while True:
                page = encrypted.read(cls.page_size)
                if not page:
                    break
                if len(page) != cls.page_size:
                    raise ValueError('encrypted database has a partial page')
                offset = 16 if index == 0 else 0
                enc_data = page[offset:cls.page_size - cls.reserve]
                iv = page[cls.page_size - cls.reserve:cls.page_size - cls.reserve + cls.iv_len]
                decryptor = Cipher(algorithms.AES(enc_key), modes.CBC(iv)).decryptor()
                plain = decryptor.update(enc_data) + decryptor.finalize()
                if index == 0:
                    plaintext.write(cls.sqlite_header)
                plaintext.write(plain)
                pad = cls.page_size - (len(cls.sqlite_header) if index == 0 else 0) - len(plain)
                if pad > 0:
                    plaintext.write(b'\x00' * pad)
                index += 1
            plaintext.flush()
            os.fsync(plaintext.fileno())

    def decrypt(self, source: Path, dest: Path, *, key: str | None, file_family: str) -> DecryptFileResult:
        if file_family == 'media_kvdb':
            return DecryptFileResult('', source.name, file_family, 'skipped', error_code='unsupported_format')
        if sqlite_readable(source):
            return self.fallback.decrypt(source, dest, key=key, file_family=file_family)
        try:
            with source.open('rb') as stream:
                salt = stream.read(16).hex().lower()
        except OSError:
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='source_unreadable')
        dk = self._load_keys(key).get(salt)
        if not dk:
            return self.fallback.decrypt(source, dest, key=key, file_family=file_family)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + '.tmp')
        tmp.unlink(missing_ok=True)
        try:
            self._decrypt_to_file(source, tmp, dk)
        except Exception:
            tmp.unlink(missing_ok=True)
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='decrypt_failed')
        if not sqlite_readable(tmp):
            tmp.unlink(missing_ok=True)
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='wrong_key')
        os.replace(tmp, dest)
        row_count, fp = sqlite_schema_fingerprint(dest)
        return DecryptFileResult('', source.name, file_family, 'decrypted', output_relative=dest.name, row_count=row_count, schema_fingerprint=fp)


class CopyPlaintextEngine:
    """Test/safe fixture engine: copies readable SQLite DBs, classifies kvdb unsupported."""

    def decrypt(self, source: Path, dest: Path, *, key: str | None, file_family: str) -> DecryptFileResult:
        if file_family == 'media_kvdb':
            return DecryptFileResult('', source.name, file_family, 'skipped', error_code='unsupported_format')
        if not sqlite_readable(source):
            return DecryptFileResult('', source.name, file_family, 'failed', error_code='wrong_key')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        row_count, fp = sqlite_schema_fingerprint(dest)
        return DecryptFileResult('', source.name, file_family, 'copied_plaintext', output_relative=dest.name, row_count=row_count, schema_fingerprint=fp)


def _source_version_hash(item: DecryptFilePlan, info: os.stat_result) -> str:
    """Fingerprint source identity without persisting a private path."""

    return stable_hash(
        json.dumps(
            [
                item.account_ref_hash,
                item.source_path.name,
                item.file_family,
                int(info.st_size),
                int(info.st_mtime_ns),
            ],
            ensure_ascii=True,
            separators=(',', ':'),
        ),
        # Keep the opaque signature below the manifest's generic long-hex
        # redaction threshold while retaining a collision-resistant 96 bits.
        length=24,
    )


def _previous_successful_files(base: Path) -> tuple[Path | None, dict[tuple[str, str, str, str, str], dict[str, Any]]]:
    """Load bounded, redacted reuse metadata from the published run only."""

    current = base / 'current'
    runs = base / 'runs'
    try:
        previous_run = require_existing_under(current, runs)
        if not previous_run.is_dir() or previous_run.is_symlink():
            return None, {}
        manifest_path = require_existing_under(previous_run / MANIFEST_NAME, previous_run)
        info = manifest_path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_size > 1024 * 1024:
            return None, {}
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        return None, {}
    if not isinstance(payload, dict) or payload.get('ok') is not True:
        return None, {}
    files = payload.get('files')
    if not isinstance(files, list):
        return None, {}
    reusable: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for item in files[:1024]:
        if not isinstance(item, dict) or item.get('status') not in {'decrypted', 'copied_plaintext', 'reused'}:
            continue
        key = (
            str(item.get('account_ref_hash') or ''),
            str(item.get('file_name') or ''),
            str(item.get('file_family') or ''),
            str(item.get('source_path_hash') or ''),
            str(item.get('source_version_hash') or ''),
        )
        if all(key):
            reusable[key] = item
    return previous_run, reusable


def _reuse_previous_output(
    previous_run: Path | None,
    previous_files: dict[tuple[str, str, str, str, str], dict[str, Any]],
    *,
    item: DecryptFilePlan,
    dest: Path,
    source_path_hash: str,
    source_version_hash: str,
) -> DecryptFileResult | None:
    if previous_run is None:
        return None
    previous = previous_files.get((
        item.account_ref_hash,
        item.source_path.name,
        item.file_family,
        source_path_hash,
        source_version_hash,
    ))
    if previous is None:
        return None
    try:
        prior_output = require_existing_under(previous_run / item.output_relative, previous_run)
        info = prior_output.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.link(prior_output, dest, follow_symlinks=False)
    except (OSError, ValueError):
        return None
    return DecryptFileResult(
        account_ref_hash=item.account_ref_hash,
        file_name=item.source_path.name,
        file_family=item.file_family,
        status='reused',
        output_relative=str(item.output_relative),
        source_path_hash=source_path_hash,
        source_version_hash=source_version_hash,
        row_count=previous.get('row_count') if isinstance(previous.get('row_count'), int) else None,
        schema_fingerprint=str(previous.get('schema_fingerprint') or '') or None,
    )


def _base_dir(vault_root: Path, output_source_name: str) -> Path:
    return vault_root / 'sources' / output_source_name


def _stage_current(base: Path, run_dir: Path) -> Path:
    """Prepare the complete current target without holding the Vault writer."""

    staged = base / f'.current-{run_dir.name}.tmp'
    if staged.is_dir() and not staged.is_symlink():
        shutil.rmtree(staged)
    else:
        staged.unlink(missing_ok=True)
    try:
        staged.symlink_to(run_dir, target_is_directory=True)
        return staged
    except OSError:
        staged.unlink(missing_ok=True)
        shutil.copytree(run_dir, staged)
        return staged


def _publish_current(base: Path, run_dir: Path, staged: Path) -> None:
    """Publish one already-staged current target using rename-only operations."""

    current = base / 'current'
    try:
        os.replace(staged, current)
    except OSError:
        if current.exists() or current.is_symlink():
            backup = base / f'.previous-current-{run_dir.name}'
            if backup.exists():
                if backup.is_dir() and not backup.is_symlink():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            current.rename(backup)
        os.replace(staged, current)


def _discard_unpublished_run(base: Path, run_dir: Path, staged: Path | None) -> None:
    """Best-effort cleanup after a created run fails before completion.

    The expensive decrypt phase intentionally runs outside the short Vault
    writer window.  Any build, staging, or publication failure must therefore
    discard its private snapshot instead of leaking a full copy every cycle.
    Filesystem state is authoritative: if ``current`` already resolves to this
    run, publication happened before the lock error and the run must survive.
    """

    if staged is not None and staged.parent == base:
        try:
            staged_info = staged.lstat()
        except OSError:
            staged_info = None
        if staged_info is not None:
            try:
                if stat.S_ISDIR(staged_info.st_mode) and not stat.S_ISLNK(staged_info.st_mode):
                    shutil.rmtree(staged)
                else:
                    staged.unlink()
            except OSError:
                pass

    runs = base / 'runs'
    if run_dir.parent != runs:
        return
    try:
        current_target = (base / 'current').resolve(strict=True)
        run_target = run_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        current_target = None
        run_target = None
    if current_target is not None and current_target == run_target:
        return
    try:
        run_info = run_dir.lstat()
    except OSError:
        return
    try:
        if stat.S_ISDIR(run_info.st_mode) and not stat.S_ISLNK(run_info.st_mode):
            shutil.rmtree(run_dir)
        else:
            run_dir.unlink()
    except OSError:
        pass


def _complete_decrypt_plan(
    plan: DecryptPlan,
    *,
    engine: DecryptEngine,
    secret_resolver: Any,
    switch_current: bool,
    started: float,
    run_id: str,
    base: Path,
    run_dir: Path,
) -> dict[str, Any]:
    previous_run, previous_files = _previous_successful_files(base)
    results: list[DecryptFileResult] = []
    selected_account_names: set[str] = set()
    account_output_names: dict[str, set[str]] = {}
    fatal_errors: list[str] = list(plan.errors)
    key_cache: dict[str, str | None] = {}
    for item in plan.files:
        output_account_name = item.output_relative.parts[0]
        account_output_names.setdefault(item.account_ref_hash, set()).add(output_account_name)
        try:
            source = require_existing_under(item.source_path, plan.config.live_root)
            dest = require_output_under(run_dir / item.output_relative, run_dir)
            source_info = source.stat()
        except (OSError, ValueError):
            results.append(DecryptFileResult(item.account_ref_hash, item.source_path.name, item.file_family, 'failed', error_code='path_escape', source_path_hash=stable_hash(item.source_path)))
            continue
        selected_account_names.add(output_account_name)
        source_path_hash = stable_hash(item.source_path)
        source_version_hash = _source_version_hash(item, source_info)
        reused = _reuse_previous_output(
            previous_run,
            previous_files,
            item=item,
            dest=dest,
            source_path_hash=source_path_hash,
            source_version_hash=source_version_hash,
        )
        if reused is not None:
            try:
                stable_source = _source_version_hash(item, source.stat()) == source_version_hash
            except OSError:
                stable_source = False
            if stable_source:
                results.append(reused)
                continue
            dest.unlink(missing_ok=True)
        key: str | None = None
        if item.file_family != 'media_kvdb':
            secret_name = item.secret_name
            if secret_name:
                if secret_name not in key_cache:
                    try:
                        key_cache[secret_name] = secret_resolver.get_secret(secret_name)
                    except SecretResolutionError as exc:
                        key_cache[secret_name] = None
                        # Defer missing key to per-file result.
                key = key_cache.get(secret_name)
        result = engine.decrypt(item.source_path, dest, key=key, file_family=item.file_family)
        results.append(DecryptFileResult(
            account_ref_hash=item.account_ref_hash,
            file_name=item.source_path.name,
            file_family=item.file_family,
            status=result.status,
            error_code=result.error_code,
            output_relative=str(item.output_relative) if result.status in {'decrypted', 'copied_plaintext'} else None,
            source_path_hash=source_path_hash,
            source_version_hash=source_version_hash,
            row_count=result.row_count,
            schema_fingerprint=result.schema_fingerprint,
            stdout=result.stdout,
            stderr=result.stderr,
        ))
    ok_statuses = {'decrypted', 'copied_plaintext', 'reused', 'skipped'}
    failed = [r for r in results if r.status not in ok_statuses]
    failed_account_refs = {r.account_ref_hash for r in failed}
    productive_account_refs = {
        r.account_ref_hash for r in results if r.status in {'decrypted', 'copied_plaintext', 'reused'}
    }
    complete_account_refs = productive_account_refs - failed_account_refs
    partial_success = bool(
        plan.config.allow_partial_accounts and failed_account_refs and complete_account_refs and not plan.errors
    )
    terminal_gaps: list[dict[str, Any]] = []
    if partial_success:
        for account_ref in failed_account_refs:
            for output_name in account_output_names.get(account_ref, set()):
                failed_output = run_dir / output_name
                if failed_output.is_dir() and not failed_output.is_symlink():
                    shutil.rmtree(failed_output)
        selected_account_names = {
            output_name
            for account_ref in complete_account_refs
            for output_name in account_output_names.get(account_ref, set())
        }
        terminal_gaps.append({
            'kind': 'account_key_unavailable',
            'account_count': len(failed_account_refs),
            'next_action': 'capture_keys_while_each_missing_local_account_is_open_then_rerun',
        })
    elif failed:
        fatal_errors.extend(sorted({r.error_code or r.status for r in failed}))
    if not results:
        fatal_errors.append('no_decryptable_files')
    succeeded = bool(results and not fatal_errors and selected_account_names)
    if succeeded and plan.config.persist_account_identity:
        for account_ref in complete_account_refs:
            own_wxids = {
                own_wxid
                for item in plan.files
                if item.account_ref_hash == account_ref
                if (own_wxid := _own_wxid_from_account_root(item.account_root))
            }
            own_wxids.update(
                own_wxid
                for selected in plan.config.selected_accounts
                if stable_hash(selected.account_id) == account_ref
                if (own_wxid := _own_wxid_from_account_root(Path(selected.account_id)))
            )
            for item in plan.files:
                if item.account_ref_hash != account_ref:
                    continue
                selected = plan.config.selected_for_root(item.account_root)
                if selected is None:
                    continue
                for candidate in (selected.account_id, selected.root_name or ''):
                    own_wxid = _own_wxid_from_account_root(Path(candidate))
                    if own_wxid:
                        own_wxids.add(own_wxid)
            if len(own_wxids) != 1:
                continue
            own_wxid = next(iter(own_wxids))
            for output_name in account_output_names.get(account_ref, set()):
                write_account_identity(
                    run_dir / output_name,
                    account_ref_hash=account_ref,
                    own_wxid=own_wxid,
                )
        write_guard(run_dir, account_dir_names=sorted(selected_account_names), run_id=run_id)
    status = 'completed_with_account_gaps' if partial_success and succeeded else ('completed' if succeeded else 'failed')
    manifest_payload = {
        'ok': succeeded,
        'status': status,
        'run_id': run_id,
        'created_at': _now(),
        'plan': plan.to_redacted_dict(),
        'files': [r.to_dict() for r in results],
        'summary': {
            'planned': len(plan.files),
            'decrypted': sum(1 for r in results if r.status == 'decrypted'),
            'copied_plaintext': sum(1 for r in results if r.status == 'copied_plaintext'),
            'reused': sum(1 for r in results if r.status == 'reused'),
            'skipped': sum(1 for r in results if r.status == 'skipped'),
            'failed': len(failed),
            'complete_accounts': len(complete_account_refs),
            'unavailable_accounts': len(failed_account_refs),
        },
        'terminal_gaps': terminal_gaps,
        'errors': fatal_errors[:50],
        'elapsed_ms': round((time.time() - started) * 1000, 3),
        'raw_content_included': False,
        'raw_paths_included': False,
    }
    staged_current = _stage_current(base, run_dir) if succeeded and switch_current else None
    with coordinated_vault_mutation(
        plan.config.vault_root,
        operation='decrypt_snapshot',
    ):
        manifest_path = write_manifest(run_dir, manifest_payload)
        if staged_current is not None:
            if staged_current.is_dir() and not staged_current.is_symlink():
                # The no-symlink fallback was copied before final publication;
                # add only the small completed manifest under the short lock.
                write_manifest(staged_current, manifest_payload)
            _publish_current(base, run_dir, staged_current)
    return {
        **manifest_payload,
        'run_ref': run_id,
        'manifest': manifest_path.name,
        'current_switched': bool(succeeded and switch_current),
    }


@mutation_entrypoint('decrypt_snapshot')
def run_decrypt_plan(
    plan: DecryptPlan,
    *,
    engine: DecryptEngine | None = None,
    secret_resolver: Any | None = None,
    switch_current: bool = True,
    write_session: VaultWriteSession | None = None,
) -> dict[str, Any]:
    if write_session is not None:
        write_session.validate_for(plan.config.vault_root)
        raise RuntimeError('decrypt work cannot run inside an outer writer session')
    # Fail before creating or copying a snapshot when the Vault is already
    # busy (or its compatibility marker needs recovery).  Publication takes a
    # fresh short lock later; cleanup below covers the intervening race.
    with coordinated_vault_mutation(
        plan.config.vault_root,
        operation='decrypt_snapshot',
    ):
        pass
    started = time.time()
    if engine is None:
        engine = WeChatWCDBAESKeyStoreEngine(plan.config.key_store_path)
    secret_resolver = secret_resolver or AgentSwitchSecretResolver()
    run_id = _run_id()
    base = _base_dir(plan.config.vault_root, plan.config.output_source_name)
    run_dir = base / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    staged_current = base / f'.current-{run_id}.tmp'
    try:
        return _complete_decrypt_plan(
            plan,
            engine=engine,
            secret_resolver=secret_resolver,
            switch_current=switch_current,
            started=started,
            run_id=run_id,
            base=base,
            run_dir=run_dir,
        )
    except BaseException:
        _discard_unpublished_run(base, run_dir, staged_current)
        raise
