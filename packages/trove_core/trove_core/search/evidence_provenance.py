from __future__ import annotations

from contextlib import closing

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROVENANCE_SCHEMA_VERSION = 1
EVIDENCE_MANIFEST_SCHEMA_VERSION = 1
_HEX = frozenset('0123456789abcdef')


class EvidenceProvenanceError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _is_hex(value: Any, *, lengths: set[int]) -> bool:
    text = str(value or '').lower()
    return len(text) in lengths and all(ch in _HEX for ch in text)


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ['git', *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=3.0,
            check=False,
        )
    except Exception:
        return None


def collect_git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    commit = _run_git(root, 'rev-parse', 'HEAD')
    status = _run_git(root, 'status', '--porcelain', '--untracked-files=normal')
    commit_sha = commit.stdout.strip().lower() if commit and commit.returncode == 0 else ''
    return {
        'commit_sha': commit_sha if _is_hex(commit_sha, lengths={40, 64}) else None,
        'dirty': None if status is None or status.returncode != 0 else bool(status.stdout),
    }


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf('SC_PAGE_SIZE'))
        pages = int(os.sysconf('SC_PHYS_PAGES'))
        total = page_size * pages
        return total if total > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def collect_platform_provenance() -> dict[str, Any]:
    processor = platform.processor().strip()
    return {
        'system': platform.system() or 'unknown',
        'release': platform.release() or 'unknown',
        'machine': platform.machine() or 'unknown',
        'python_implementation': platform.python_implementation() or 'unknown',
        'python_version': platform.python_version(),
        'cpu_count': int(os.cpu_count() or 0),
        'processor_sha256': stable_payload_sha256(processor or 'unknown'),
        'memory_bytes': _physical_memory_bytes(),
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def _safe_count(conn: sqlite3.Connection, table: str, *, active_only: bool = False) -> int:
    if not _table_exists(conn, table):
        return 0
    if table not in {'messages', 'evidence_chunks', 'vector_entries'}:
        raise EvidenceProvenanceError(f'unsupported provenance count table: {table}')
    where = " WHERE status='active'" if active_only else ''
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"{where}').fetchone()[0])


def _content_identity_sha256(conn: sqlite3.Connection) -> str:
    """Hash the complete retrieval corpus without emitting any row value."""

    digest = hashlib.sha256()
    queries = (
        (
            'messages',
            'SELECT citation,content,timestamp,content_kind FROM messages ORDER BY citation',
        ),
        (
            'evidence_chunks',
            'SELECT chunk_citation,parent_citation,content,status FROM evidence_chunks ORDER BY chunk_citation',
        ),
        (
            'vector_entries',
            'SELECT citation,provider,dimensions,coalesce(content_hash,\'\') FROM vector_entries ORDER BY citation',
        ),
    )
    for table, sql in queries:
        digest.update(table.encode('utf-8') + b'\0')
        if not _table_exists(conn, table):
            continue
        for row in conn.execute(sql):
            digest.update(canonical_json_bytes(list(row)))
            digest.update(b'\n')
    return digest.hexdigest()


def collect_store_provenance(sqlite_path: str | Path) -> dict[str, Any]:
    path = Path(sqlite_path).expanduser().resolve()
    if not path.exists():
        empty = {
            'schema_version': 0,
            'schema_manifest_sha256': stable_payload_sha256([]),
            'content_identity_sha256': stable_payload_sha256([]),
            'safe_metadata': {},
            'document_counts': {'messages': 0, 'chunks': 0, 'vectors': 0},
        }
        return {
            **empty,
            'index_generation_sha256': stable_payload_sha256(empty),
            'document_count': 0,
        }

    uri = path.as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        schema_rows = list(conn.execute(
            "SELECT type,name,coalesce(sql,'') FROM sqlite_master "
            "WHERE type IN ('table','index','trigger','view') ORDER BY type,name"
        ))
        schema_manifest = [
            {'type': str(row[0]), 'name_sha256': stable_payload_sha256(str(row[1])), 'sql_sha256': stable_payload_sha256(str(row[2]))}
            for row in schema_rows
        ]
        user_version = int(conn.execute('PRAGMA user_version').fetchone()[0])
        safe_metadata: dict[str, str] = {}
        if _table_exists(conn, 'schema_meta'):
            allowed = ('schema_version', 'fts_tokenizer', 'message_fts_rows', 'chunk_fts_rows')
            placeholders = ','.join('?' for _ in allowed)
            for key, value in conn.execute(
                f'SELECT key,value FROM schema_meta WHERE key IN ({placeholders}) ORDER BY key',
                allowed,
            ):
                safe_metadata[str(key)] = str(value)
        try:
            schema_version = int(safe_metadata.get('schema_version', user_version))
        except ValueError:
            schema_version = user_version
        counts = {
            'messages': _safe_count(conn, 'messages'),
            'chunks': _safe_count(conn, 'evidence_chunks', active_only=True),
            'vectors': _safe_count(conn, 'vector_entries'),
        }
        content_identity_sha256 = _content_identity_sha256(conn)

    generation_basis = {
        'schema_version': schema_version,
        'schema_manifest_sha256': stable_payload_sha256(schema_manifest),
        'content_identity_sha256': content_identity_sha256,
        'safe_metadata': safe_metadata,
        'document_counts': counts,
    }
    return {
        **generation_basis,
        'index_generation_sha256': stable_payload_sha256(generation_basis),
        'document_count': counts['chunks'] or counts['messages'],
    }


def collect_provider_provenance(provider: object | None) -> dict[str, Any]:
    if provider is None:
        return {
            'provider_sha256': stable_payload_sha256('none'),
            'model_sha256': stable_payload_sha256('none'),
            'dimensions': 0,
        }
    provider_identity = '|'.join([
        provider.__class__.__module__,
        provider.__class__.__qualname__,
        str(getattr(provider, 'name', '') or ''),
        str(getattr(provider, 'provider_name', '') or ''),
    ])
    model_identity = (
        getattr(provider, 'model_id', None)
        or getattr(provider, 'model', None)
        or getattr(provider, 'model_path', None)
        or 'unknown'
    )
    return {
        'provider_sha256': stable_payload_sha256(provider_identity),
        'model_sha256': stable_payload_sha256(str(model_identity)),
        'dimensions': int(getattr(provider, 'dimensions', 0) or 0),
    }


def build_artifact_provenance(
    *,
    repo_root: str | Path,
    sqlite_path: str | Path,
    case_pack_path: str | Path,
    seed: int,
    fixture_id: str,
    provider: object | None,
    temperature: str,
    warmups: int,
    rounds: int,
) -> dict[str, Any]:
    if temperature not in {'cold', 'warm'}:
        raise EvidenceProvenanceError('temperature must be cold or warm')
    if warmups < 0 or rounds < 1:
        raise EvidenceProvenanceError('warmups must be >= 0 and rounds must be >= 1')
    if temperature == 'cold' and warmups:
        raise EvidenceProvenanceError('cold evidence cannot claim warmup runs')
    case_path = Path(case_pack_path).expanduser()
    store = collect_store_provenance(sqlite_path)
    case_pack_sha256 = sha256_file(case_path)
    fixture_basis = {
        'fixture_label_sha256': stable_payload_sha256(str(fixture_id or 'synthetic_or_redacted')),
        'schema_version': store['schema_version'],
        'index_generation_sha256': store['index_generation_sha256'],
        'document_counts': store['document_counts'],
    }
    provenance = {
        'schema_version': PROVENANCE_SCHEMA_VERSION,
        'git': collect_git_provenance(repo_root),
        'platform': collect_platform_provenance(),
        'fixture': {
            'kind': 'synthetic_or_redacted',
            'sha256': stable_payload_sha256(fixture_basis),
        },
        'seed': int(seed),
        'case_pack_sha256': case_pack_sha256,
        'store': store,
        'provider': collect_provider_provenance(provider),
        'execution': {
            'temperature': temperature,
            'warmups': int(warmups),
            'rounds': int(rounds),
            'includes_engine_build': temperature == 'cold',
        },
        'privacy': {
            'raw_fixture_identity_included': False,
            'raw_case_pack_included': False,
            'private_paths_included': False,
            'provider_names_included': False,
            'model_names_included': False,
        },
    }
    validate_artifact_provenance(provenance, release=False)
    return provenance


def validate_artifact_provenance(provenance: Any, *, release: bool) -> None:
    if not isinstance(provenance, dict):
        raise EvidenceProvenanceError('missing artifact provenance')
    if provenance.get('schema_version') != PROVENANCE_SCHEMA_VERSION:
        raise EvidenceProvenanceError('unsupported artifact provenance schema')
    git = provenance.get('git')
    platform_data = provenance.get('platform')
    fixture = provenance.get('fixture')
    store = provenance.get('store')
    provider = provenance.get('provider')
    execution = provenance.get('execution')
    for name, value in (
        ('git', git), ('platform', platform_data), ('fixture', fixture),
        ('store', store), ('provider', provider), ('execution', execution),
    ):
        if not isinstance(value, dict):
            raise EvidenceProvenanceError(f'missing provenance field: {name}')
    if not _is_hex(git.get('commit_sha'), lengths={40, 64}):
        raise EvidenceProvenanceError('invalid provenance git commit')
    if not isinstance(git.get('dirty'), bool):
        raise EvidenceProvenanceError('invalid provenance git dirty flag')
    for key in ('system', 'release', 'machine', 'python_implementation', 'python_version'):
        if not str(platform_data.get(key) or '').strip():
            raise EvidenceProvenanceError(f'missing provenance platform field: {key}')
    if not isinstance(platform_data.get('cpu_count'), int) or platform_data['cpu_count'] < 1:
        raise EvidenceProvenanceError('invalid provenance cpu_count')
    if not _is_hex(platform_data.get('processor_sha256'), lengths={64}):
        raise EvidenceProvenanceError('invalid provenance processor hash')
    if platform_data.get('memory_bytes') is not None and int(platform_data['memory_bytes']) <= 0:
        raise EvidenceProvenanceError('invalid provenance memory_bytes')
    if fixture.get('kind') != 'synthetic_or_redacted' or not _is_hex(fixture.get('sha256'), lengths={64}):
        raise EvidenceProvenanceError('invalid provenance fixture')
    if not isinstance(provenance.get('seed'), int):
        raise EvidenceProvenanceError('missing provenance seed')
    for key in ('case_pack_sha256',):
        if not _is_hex(provenance.get(key), lengths={64}):
            raise EvidenceProvenanceError(f'invalid provenance hash: {key}')
    if not isinstance(store.get('schema_version'), int) or store['schema_version'] < 0:
        raise EvidenceProvenanceError('invalid provenance schema version')
    for key in ('schema_manifest_sha256', 'content_identity_sha256', 'index_generation_sha256'):
        if not _is_hex(store.get(key), lengths={64}):
            raise EvidenceProvenanceError(f'invalid provenance store hash: {key}')
    if not isinstance(store.get('document_count'), int) or store['document_count'] < 0:
        raise EvidenceProvenanceError('invalid provenance document count')
    for key in ('provider_sha256', 'model_sha256'):
        if not _is_hex(provider.get(key), lengths={64}):
            raise EvidenceProvenanceError(f'invalid provenance provider field: {key}')
    if execution.get('temperature') not in {'cold', 'warm'}:
        raise EvidenceProvenanceError('invalid provenance temperature')
    if not isinstance(execution.get('warmups'), int) or execution['warmups'] < 0:
        raise EvidenceProvenanceError('invalid provenance warmups')
    if not isinstance(execution.get('rounds'), int) or execution['rounds'] < 1:
        raise EvidenceProvenanceError('invalid provenance rounds')
    if not isinstance(execution.get('includes_engine_build'), bool):
        raise EvidenceProvenanceError('invalid provenance engine-build flag')
    if execution['temperature'] == 'cold' and execution['warmups']:
        raise EvidenceProvenanceError('cold provenance cannot include warmups')
    if execution['includes_engine_build'] != (execution['temperature'] == 'cold'):
        raise EvidenceProvenanceError('provenance temperature does not match engine-build scope')
    if release and git['dirty']:
        raise EvidenceProvenanceError('release evidence must come from a clean worktree')


def evidence_manifest_path(artifact_path: str | Path) -> Path:
    path = Path(artifact_path).expanduser()
    return path.with_name(path.name + '.manifest.json')


def build_evidence_manifest(artifact_path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    path = Path(artifact_path).expanduser()
    provenance = report.get('provenance')
    manifest = {
        'schema_version': EVIDENCE_MANIFEST_SCHEMA_VERSION,
        'artifact_type': 'evidence_manifest_redacted',
        'artifact_file': path.name,
        'artifact_sha256': sha256_file(path),
        'artifact_bytes': int(path.stat().st_size),
        'provenance_sha256': stable_payload_sha256(provenance) if provenance is not None else None,
        'privacy': {
            'artifact_content_included': False,
            'private_paths_included': False,
            'token_values_included': False,
        },
    }
    return manifest


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_evidence_artifact(report: dict[str, Any], artifact_path: str | Path) -> Path:
    path = Path(artifact_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode('utf-8') + b'\n'
    _atomic_write(path, encoded)
    manifest = build_evidence_manifest(path, report)
    manifest_path = evidence_manifest_path(path)
    _atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode('utf-8') + b'\n',
    )
    return manifest_path


def verify_evidence_manifest(artifact_path: str | Path, *, required: bool) -> dict[str, Any] | None:
    path = Path(artifact_path).expanduser()
    manifest_path = evidence_manifest_path(path)
    if not manifest_path.exists():
        if required:
            raise EvidenceProvenanceError(f'missing evidence manifest: {path.name}')
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProvenanceError(f'invalid evidence manifest: {path.name}') from exc
    if manifest.get('schema_version') != EVIDENCE_MANIFEST_SCHEMA_VERSION:
        raise EvidenceProvenanceError('unsupported evidence manifest schema')
    if manifest.get('artifact_type') != 'evidence_manifest_redacted':
        raise EvidenceProvenanceError('invalid evidence manifest type')
    if manifest.get('artifact_file') != path.name:
        raise EvidenceProvenanceError('evidence manifest artifact name mismatch')
    if manifest.get('artifact_sha256') != sha256_file(path):
        raise EvidenceProvenanceError('evidence manifest artifact hash mismatch')
    if int(manifest.get('artifact_bytes') or -1) != int(path.stat().st_size):
        raise EvidenceProvenanceError('evidence manifest artifact size mismatch')
    report = json.loads(path.read_text(encoding='utf-8'))
    expected_provenance_hash = stable_payload_sha256(report.get('provenance')) if report.get('provenance') is not None else None
    if manifest.get('provenance_sha256') != expected_provenance_hash:
        raise EvidenceProvenanceError('evidence manifest provenance hash mismatch')
    return manifest
