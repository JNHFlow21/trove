from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from typing import Any, Callable, ContextManager

from trove_core.store.sqlite_store import SQLiteStore, vector_document_text
from trove_core.search.evidence_provenance import (
    sha256_bytes,
    stable_payload_sha256,
    verify_evidence_manifest,
)
from trove_core.vector.ledger import VectorIndexLedger
from .score_calibration import (
    VectorScoreCalibrationError,
    embedding_identity,
    index_identity,
    score_calibration_status,
    validate_score_calibration_artifact,
)

VECTOR_TEXT_VERSION = 3
ZVEC_COLLECTION_CONTRACT_VERSION = 1
ZVEC_ADAPTIVE_OVERFETCH_MAX = 1600
_ZVEC_MAX_WRITE_BATCH = 1024

_ZVEC_PUSHDOWN_FIELDS = {
    'account_id',
    'conversation_id',
    'conversation_type',
    'since',
    'until',
}


def _upsert_zvec_docs(collection: Any, docs: list[Any]) -> int:
    """Upsert docs without exceeding zvec's per-call write batch limit."""

    for start in range(0, len(docs), _ZVEC_MAX_WRITE_BATCH):
        collection.upsert(docs[start:start + _ZVEC_MAX_WRITE_BATCH])
    return len(docs)


def _zvec_string_literal(value: str) -> str | None:
    """Encode one ZVEC SQL literal without allowing expression injection."""

    if len(value) > 4096 or any(ord(char) < 32 for char in value):
        return None
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _zvec_filter_plan(filters: dict[str, str]) -> tuple[str | None, dict[str, str], list[str], list[str]]:
    clauses: list[str] = []
    residual: dict[str, str] = {}
    pushed: list[str] = []
    ignored: list[str] = []
    for key, raw_value in filters.items():
        value = str(raw_value)
        if not value:
            ignored.append(key)
            continue
        if key in {'source_family', 'scope_type'} and value == 'all':
            ignored.append(key)
            continue
        literal = _zvec_string_literal(value)
        if key in _ZVEC_PUSHDOWN_FIELDS and literal is not None:
            if key == 'since':
                clauses.append(f'timestamp >= {literal}')
            elif key == 'until':
                clauses.append(f'timestamp <= {literal}')
            else:
                clauses.append(f'{key} = {literal}')
            pushed.append(key)
        elif key == 'sender' and literal is not None and not any(char in value for char in ('%', '_')):
            # ZVEC LIKE preserves SQLiteStore's substring sender-name contract;
            # wildcard-bearing input remains a residual filter so user text is
            # never interpreted as a pattern.
            clauses.append(f'(sender_name LIKE {_zvec_string_literal("%" + value + "%")} OR sender_id = {literal})')
            pushed.append(key)
        else:
            residual[key] = value
    return (' AND '.join(clauses) or None), residual, sorted(pushed), sorted(ignored)


def _provider_metadata(provider: Any | None) -> dict[str, Any]:
    if provider is None:
        return {}
    identity = embedding_identity(provider)
    metadata = {
        'embedding_provider': str(getattr(provider, 'provider_name', getattr(provider, 'name', provider.__class__.__name__))),
        'embedding_model': str(getattr(provider, 'model_id', getattr(provider, 'model', '')) or ''),
        'embedding_dimensions': int(getattr(provider, 'dimensions', 0) or 0),
        'embedding_request_format': str(getattr(provider, 'request_format', '') or ''),
        'embedding_identity_sha256': identity['sha256'],
    }
    if bool(getattr(provider, 'supports_sparse', False)):
        metadata['embedding_sparse'] = True
        metadata['embedding_query_instruct_sha256'] = hashlib.sha256(
            str(getattr(provider, 'query_instruct', '') or '').encode('utf-8')
        ).hexdigest()
    return metadata


def _provider_contract_mismatch(metadata: dict[str, Any], provider: Any | None, *, require_present: bool = False) -> bool:
    expected = _provider_metadata(provider)
    for key, value in expected.items():
        if value not in {'', 0}:
            current = metadata.get(key)
            if current != value and (require_present or current is not None):
                return True
    dimensions = expected.get('embedding_dimensions') or 0
    if dimensions:
        current_dim = metadata.get('dimensions')
        if current_dim != dimensions and (require_present or current_dim is not None):
            return True
    return False


def _sidecar_generation_revision(value: Any) -> int | None:
    """Parse a mirrored revision without letting corrupt JSON break status."""

    try:
        revision = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return revision if revision >= 1 else None


def _index_contract_mismatch(metadata: dict[str, Any], provider: Any | None) -> bool:
    if metadata.get('collection_contract_version') != ZVEC_COLLECTION_CONTRACT_VERSION:
        return True
    if metadata.get('vector_text_version') not in {None, VECTOR_TEXT_VERSION}:
        return True
    return _provider_contract_mismatch(metadata, provider, require_present=True)


def _embedding_text(row: Any) -> str:
    try:
        if hasattr(row, 'keys') and 'vector_text' in row.keys():
            return str(row['vector_text'] or '')
    except Exception:
        pass
    return vector_document_text(row)


class _DenseOnlyProvider:
    """Reuse one hybrid query embedding without changing provider identity."""

    supports_sparse = False

    def __init__(self, provider: Any, dense_query: list[float]):
        self._provider = provider
        self._dense_query = dense_query

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def embed_query(self, _text: str) -> list[float]:
        return self._dense_query

    def embed(self, text: str) -> list[float]:
        return self._provider.embed(text)


class ZVecStore:
    """Optional local ZVEC adapter.

    Importing this class never requires ZVEC. If the dependency is absent, the adapter reports
    unavailable and callers should continue with FTS/exact retrieval.
    """
    def __init__(
        self,
        collection_path: str | Path,
        store: SQLiteStore | None = None,
        *,
        memory_limit_mb: int = 1024,
        ledger_backend: str = 'zvec',
    ):
        self.collection_path = str(Path(collection_path).expanduser())
        self.store = store
        self.ledger_backend = str(ledger_backend)
        self.ledger = VectorIndexLedger(store, backend=self.ledger_backend) if store is not None else None
        self.memory_limit_mb = memory_limit_mb
        self._collection = None
        self._search_local = threading.local()
        try:
            import zvec  # type: ignore
        except Exception as exc:
            self._zvec = None
            self._error = exc
        else:
            self._zvec = zvec
            self._error = None
            try:
                zvec.init(log_type=None, memory_limit_mb=memory_limit_mb)
            except RuntimeError:
                # ZVEC can only be initialized once per process. Reuse that process-global init.
                pass
            except Exception as exc:
                self._zvec = None
                self._error = exc

    @property
    def available(self) -> bool:
        return self._zvec is not None

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.available else f'ZVEC is not installed or failed to import: {self._error}'

    @property
    def reason_code(self) -> str | None:
        if self.available:
            return None
        if isinstance(self._error, ModuleNotFoundError):
            return 'zvec_import_unavailable'
        return 'zvec_init_failed'

    @property
    def metadata_path(self) -> Path:
        return Path(str(self.collection_path) + '.trove-meta.json')

    @property
    def progress_path(self) -> Path:
        return Path(str(self.collection_path) + '.trove-progress.json')

    @property
    def swap_marker_path(self) -> Path:
        return Path(str(self.collection_path) + '.trove-swap.json')

    def _generation_paths(self, collection_path: str | Path) -> dict[str, Path]:
        base = Path(collection_path)
        return {
            'collection': base,
            'metadata': Path(str(base) + '.trove-meta.json'),
            'progress': Path(str(base) + '.trove-progress.json'),
        }

    def _final_generation_paths(self) -> dict[str, Path]:
        return self._generation_paths(self.collection_path)

    def _backup_generation_paths(self) -> dict[str, Path]:
        return {
            'collection': Path(str(self.collection_path) + '.trove-backup'),
            'metadata': Path(str(self.metadata_path) + '.trove-backup'),
            'progress': Path(str(self.progress_path) + '.trove-backup'),
        }

    def _read_metadata(self) -> dict[str, Any]:
        try:
            if self.metadata_path.exists():
                return json.loads(self.metadata_path.read_text(encoding='utf-8'))
        except Exception:
            return {}
        return {}

    def _authoritative_score_metadata(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Overlay the SQLite ledger revision and reject a stale sidecar."""

        recovery_reason = self._atomic_recovery_reason()
        if recovery_reason is not None:
            raise VectorScoreCalibrationError(recovery_reason)
        current = dict(metadata if metadata is not None else self._read_metadata())
        published_baseline = bool(
            current.get('complete') is True
            or current.get('catchup_pending') is True
        )
        if (
            current.get('schema_version') != 4
            or current.get('backend') != 'zvec'
            or not published_baseline
        ):
            raise VectorScoreCalibrationError('vector_score_calibration_index_incomplete')
        generation_id = str(current.get('generation_id') or '')
        generation = self.ledger.generation(generation_id) if self.ledger is not None and generation_id else None
        if generation is None or generation.status != 'active':
            raise VectorScoreCalibrationError('vector_score_calibration_index_incomplete')
        mirrored = _sidecar_generation_revision(current.get('generation_revision'))
        if mirrored != generation.revision:
            raise VectorScoreCalibrationError('vector_generation_revision_mismatch')
        current['backend'] = 'zvec'
        current['generation_revision'] = generation.revision
        return current

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'wb', closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        safe = dict(metadata)
        safe['updated_at'] = time.time()
        self._atomic_write_json(self.metadata_path, safe)

    def _read_progress(self) -> dict[str, Any]:
        try:
            if self.progress_path.exists():
                return json.loads(self.progress_path.read_text(encoding='utf-8'))
        except Exception:
            return {}
        return {}

    def _write_progress(self, progress: dict[str, Any]) -> None:
        safe = dict(progress)
        safe['updated_at'] = time.time()
        self._atomic_write_json(self.progress_path, safe)

    def _write_swap_marker(
        self,
        *,
        phase: str,
        tmp_path: Path | None = None,
        operation: str = 'rebuild',
        previous_generation_id: str | None = None,
        new_generation_id: str | None = None,
    ) -> None:
        payload = {
            'phase': phase,
            'operation': operation,
            'tmp_collection_path': str(tmp_path) if tmp_path is not None else None,
            'previous_generation_id': previous_generation_id,
            'new_generation_id': new_generation_id,
            'updated_at': time.time(),
        }
        self._atomic_write_json(self.swap_marker_path, payload)

    def _read_swap_marker(self) -> dict[str, Any]:
        try:
            if self.swap_marker_path.exists():
                return json.loads(self.swap_marker_path.read_text(encoding='utf-8'))
        except Exception:
            return {}
        return {}

    def _atomic_failpoint(self, name: str) -> None:
        if os.environ.get('TROVE_ZVEC_ATOMIC_CRASHPOINT') == name:
            os._exit(99)
        if os.environ.get('TROVE_ZVEC_ATOMIC_FAILPOINT') == name:
            raise RuntimeError(f'simulated_zvec_atomic_failpoint:{name}')

    def _remove_path(self, path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _remove_generation(self, paths: dict[str, Path]) -> None:
        for key in ('progress', 'metadata', 'collection'):
            self._remove_path(paths[key])

    def _tmp_generation_bases(self) -> list[Path]:
        base = Path(self.collection_path)
        prefix = f'{base.name}.trove-tmp-'
        out: dict[str, Path] = {}
        for path in base.parent.glob(f'{prefix}*'):
            name = path.name
            for suffix in ('.trove-meta.json', '.trove-progress.json'):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if name.startswith(prefix):
                out[name] = base.parent / name
        return [out[name] for name in sorted(out)]

    def _remove_tmp_generations(self, *, except_path: Path | None = None) -> int:
        keep = except_path.resolve() if except_path else None
        removed = 0
        for tmp_base in self._tmp_generation_bases():
            if keep is not None and tmp_base.resolve() == keep:
                continue
            tmp = self._generation_paths(tmp_base)
            if any(path.exists() for path in tmp.values()):
                generation_id = ''
                for sidecar in (tmp['metadata'], tmp['progress']):
                    try:
                        generation_id = str(json.loads(sidecar.read_text(encoding='utf-8')).get('generation_id') or '')
                    except (OSError, json.JSONDecodeError):
                        continue
                    if generation_id:
                        break
                if generation_id and self.ledger is not None:
                    self.ledger.discard(generation_id)
                self._remove_generation(tmp)
                removed += 1
        return removed

    def _rename_existing(self, src: Path, dst: Path) -> bool:
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._remove_path(dst)
        src.rename(dst)
        return True

    def _restore_backup_generation(self, final: dict[str, Path], backup: dict[str, Path]) -> None:
        self._remove_path(final['collection'])
        if backup['collection'].exists():
            backup['collection'].rename(final['collection'])
        for key in ('metadata', 'progress'):
            if backup[key].exists():
                self._remove_path(final[key])
                backup[key].rename(final[key])

    def _tmp_generation_complete(self, tmp: dict[str, Path]) -> bool:
        return tmp['collection'].exists() and tmp['metadata'].exists() and tmp['progress'].exists()

    def _atomic_recovery_required(self) -> bool:
        return bool(
            self.swap_marker_path.exists()
            or any(path.exists() for path in self._backup_generation_paths().values())
            or self._tmp_generation_bases()
        )

    def _atomic_recovery_reason(self) -> str | None:
        # A temporary generation is built outside the publication lease while
        # the old active generation remains readable. Only a marker or backup
        # means publication touched the active filesystem and read paths must
        # fail closed. Writers still use _atomic_recovery_required() to clean
        # orphaned temporary generations.
        marker_present = self.swap_marker_path.exists()
        backup_present = any(path.exists() for path in self._backup_generation_paths().values())
        if not marker_present and not backup_present:
            return None
        marker = self._read_swap_marker()
        if str(marker.get('operation') or '') == 'incremental':
            return 'vector_incremental_replay_required'
        return 'zvec_atomic_recovery_required'

    def recover_atomic_rebuild(self) -> dict[str, Any]:
        """Repair a previously interrupted full ZVEC generation swap.

        Recovery is intentionally mutating and is only called by explicit
        indexing/rebuild paths, never by read-only status/search.
        """
        self._invalidate_collection_cache()
        marker = self._read_swap_marker()
        final = self._final_generation_paths()
        backup = self._backup_generation_paths()
        tmp_path_raw = marker.get('tmp_collection_path') if marker else None
        tmp = self._generation_paths(tmp_path_raw) if tmp_path_raw else {}
        backup_exists = any(path.exists() for path in backup.values())
        final_exists = final['collection'].exists()

        if not marker:
            removed_tmp = self._remove_tmp_generations()
            if backup_exists:
                if final_exists:
                    self._remove_generation(backup)
                    return {'status': 'cleaned_stale_backup', 'tmp_generations_removed': removed_tmp}
                self._restore_backup_generation(final, backup)
                return {'status': 'restored_backup', 'tmp_generations_removed': removed_tmp}
            return {'status': 'noop', 'tmp_generations_removed': removed_tmp}

        phase = str(marker.get('phase') or '')
        operation = str(marker.get('operation') or 'rebuild')
        previous_generation_id = str(marker.get('previous_generation_id') or '')
        new_generation_id = str(marker.get('new_generation_id') or '')
        if operation == 'incremental':
            return {
                'status': 'incremental_replay_required',
                'generation_id': new_generation_id or previous_generation_id or None,
            }
        try:
            if phase == 'final_ready':
                if self.ledger is not None and new_generation_id:
                    self.ledger.activate(new_generation_id)
                self._remove_generation(backup)
                if tmp:
                    self._remove_generation(tmp)
                removed_tmp = self._remove_tmp_generations()
                if self.ledger is not None:
                    self.ledger.prune_retired()
                self.swap_marker_path.unlink(missing_ok=True)
                return {'status': 'finalized', 'tmp_generations_removed': removed_tmp}

            if phase == 'final_files_ready' and self.ledger is not None and new_generation_id:
                active_generation = self.ledger.active_generation()
                if active_generation is not None and active_generation.generation_id == new_generation_id:
                    # The process can die after the SQLite activation commit but
                    # before advancing the filesystem marker. The active ledger
                    # is the transaction boundary: finish publishing the new
                    # files instead of restoring files for the retired ledger.
                    self._remove_generation(backup)
                    if tmp:
                        self._remove_generation(tmp)
                    removed_tmp = self._remove_tmp_generations()
                    self.ledger.prune_retired()
                    self.swap_marker_path.unlink(missing_ok=True)
                    return {'status': 'finalized', 'phase': phase, 'tmp_generations_removed': removed_tmp}

            if (
                phase == 'final_files_ready'
                and not backup_exists
                and all(final[key].exists() for key in ('collection', 'metadata', 'progress'))
                and self.ledger is not None
                and new_generation_id
                and self.ledger.generation(new_generation_id) is not None
            ):
                self.ledger.activate(new_generation_id)
                if tmp:
                    self._remove_generation(tmp)
                removed_tmp = self._remove_tmp_generations()
                self.ledger.prune_retired()
                self.swap_marker_path.unlink(missing_ok=True)
                return {'status': 'finalized_initial', 'phase': phase, 'tmp_generations_removed': removed_tmp}

            if backup_exists:
                self._restore_backup_generation(final, backup)
                if self.ledger is not None and new_generation_id:
                    self.ledger.discard(new_generation_id)
                if tmp:
                    self._remove_generation(tmp)
                self.swap_marker_path.unlink(missing_ok=True)
                removed_tmp = self._remove_tmp_generations()
                return {'status': 'restored_backup', 'phase': phase, 'tmp_generations_removed': removed_tmp}

            if tmp and self._tmp_generation_complete(tmp):
                for key in ('collection', 'metadata', 'progress'):
                    if not final[key].exists() and tmp[key].exists():
                        tmp[key].rename(final[key])
                    elif tmp[key].exists():
                        self._remove_path(tmp[key])
                if self.ledger is not None and new_generation_id:
                    self.ledger.activate(new_generation_id)
                removed_tmp = self._remove_tmp_generations()
                if self.ledger is not None:
                    self.ledger.prune_retired()
                self.swap_marker_path.unlink(missing_ok=True)
                return {'status': 'finalized_initial', 'phase': phase, 'tmp_generations_removed': removed_tmp}

            if tmp:
                self._remove_generation(tmp)
            if self.ledger is not None and new_generation_id:
                self.ledger.discard(new_generation_id)
            if not final['metadata'].exists() or not final['progress'].exists():
                self._remove_generation(final)
            self.swap_marker_path.unlink(missing_ok=True)
            removed_tmp = self._remove_tmp_generations()
            return {'status': 'discarded_incomplete_tmp', 'phase': phase, 'tmp_generations_removed': removed_tmp}
        finally:
            self._invalidate_collection_cache()

    def status(self, provider: Any | None = None) -> dict:
        exists = Path(self.collection_path).exists()
        recovery_reason = self._atomic_recovery_reason()
        metadata = self._read_metadata()
        progress = self._read_progress()
        # With no collection there is nothing to compare against.  Avoid an
        # exact evidence/message COUNT over the entire corpus on every routine
        # maintain just to report ``zvec_collection_missing``.
        expected_count = self._expected_document_count() if exists else None
        generation_id = str(metadata.get('generation_id') or '')
        generation = self.ledger.generation(generation_id) if self.ledger is not None and generation_id else None
        indexed_count = generation.indexed_count if generation is not None else 0
        generation_revision = generation.revision if generation is not None else 0
        mirrored_revision_raw = metadata.get('generation_revision')
        mirrored_revision = _sidecar_generation_revision(mirrored_revision_raw)
        generation_revision_mismatch = bool(
            generation is not None
            and mirrored_revision != generation_revision
        )
        expected_provider = _provider_metadata(provider)
        metadata_present = bool(metadata)
        ledger_present = generation is not None and generation.status == 'active'
        # ``metadata_complete`` describes the published generation contract,
        # not whether it has caught up with newly imported source rows. An
        # active schema-v4 generation remains incrementally writable while its
        # count is temporarily behind the source.
        declared_complete = metadata.get('complete') is True
        declared_catchup = metadata.get('catchup_pending') is True
        metadata_complete = bool(
            metadata.get('schema_version') == 4
            and ledger_present
            and (declared_complete or declared_catchup)
        )
        provider_mismatch = bool(exists and _provider_contract_mismatch(metadata, provider, require_present=True))
        stale = bool(exists and metadata.get('vector_text_version') != VECTOR_TEXT_VERSION)
        collection_contract_mismatch = bool(
            exists
            and metadata.get('collection_contract_version') != ZVEC_COLLECTION_CONTRACT_VERSION
        )
        metadata_incomplete = bool(exists and not metadata_complete)
        catchup_pending = bool(
            exists
            and metadata_complete
            and not stale
            and not provider_mismatch
            and (
                not declared_complete
                or (expected_count is not None and indexed_count < expected_count)
            )
        )
        incomplete = metadata_incomplete
        rebuild_required = stale or incomplete or provider_mismatch or collection_contract_mismatch
        complete = bool(
            exists
            and metadata_present
            and metadata_complete
            and declared_complete
            and not stale
            and not incomplete
            and not catchup_pending
            and not provider_mismatch
            and recovery_reason is None
            and (expected_count is None or indexed_count >= expected_count)
        )
        authoritative_metadata = dict(metadata)
        if generation is not None:
            authoritative_metadata['backend'] = 'zvec'
            authoritative_metadata['generation_revision'] = generation_revision
        calibration = score_calibration_status(authoritative_metadata, provider)
        calibration_required = bool(
            exists
            and metadata_complete
            and not rebuild_required
            and calibration.get('state') != 'available'
        )
        health = 'degraded' if calibration_required or generation_revision_mismatch or recovery_reason else 'ok'
        unavailable = self.unavailable_reason
        # Status is used on read-only paths (search/eval/API health).  Do not
        # open the collection here: some native ZVEC recovery paths clean crash
        # residue files on open, which would turn a status/read query into a
        # Vault mutation.  Mutating validation belongs in explicit maintain/rebuild.
        return {
            'backend': 'zvec',
            'available': self.available,
            'collection_exists': exists,
            'health': health,
            'state': 'available' if self.available and exists and health == 'ok' else ('degraded' if health == 'degraded' else 'unavailable_fallback'),
            'reason_code': self.reason_code if not self.available else (
                recovery_reason if recovery_reason is not None
                else 'zvec_rebuild_required' if rebuild_required
                else ('zvec_catchup_pending' if catchup_pending else (
                    'vector_generation_revision_mismatch' if generation_revision_mismatch
                    else calibration.get('reason_code') if calibration_required
                    else (None if exists else 'zvec_collection_missing')
                ))
            ),
            'unavailable_reason': unavailable,
            'schema_version': metadata.get('schema_version'),
            'generation_id': generation_id or None,
            'generation_revision': generation_revision or None,
            'metadata_generation_revision': mirrored_revision_raw,
            'generation_revision_mismatch': generation_revision_mismatch,
            'recovery_required': recovery_reason is not None,
            'recovery_reason_code': recovery_reason,
            'ledger_authority': 'sqlite/vector_index_ledger',
            'ledger_present': ledger_present,
            'vector_text_version': metadata.get('vector_text_version'),
            'expected_vector_text_version': VECTOR_TEXT_VERSION,
            'collection_contract_version': metadata.get('collection_contract_version'),
            'expected_collection_contract_version': ZVEC_COLLECTION_CONTRACT_VERSION,
            'collection_contract_mismatch': collection_contract_mismatch,
            'stale': stale,
            'incomplete': incomplete,
            'complete': complete,
            'metadata_present': metadata_present,
            'metadata_complete': metadata_complete,
            'provider_mismatch': provider_mismatch,
            'rebuild_required': rebuild_required,
            'catchup_pending': catchup_pending,
            'calibration_required': calibration_required,
            'score_calibration': calibration,
            'dimensions': metadata.get('dimensions'),
            'embedding_provider': metadata.get('embedding_provider'),
            'embedding_model': metadata.get('embedding_model'),
            'embedding_dimensions': metadata.get('embedding_dimensions') or metadata.get('dimensions'),
            'expected_embedding': expected_provider,
            'indexed_count': indexed_count,
            'expected_document_count': expected_count,
            'last_indexed_at': metadata.get('updated_at'),
            'progress': progress,
        }

    def atomic_rebuild(
        self,
        provider,
        *,
        store: SQLiteStore | None = None,
        ledger_store: SQLiteStore | None = None,
        batch_size: int = 256,
        max_messages: int | None = None,
        generation_publish: Callable[[], ContextManager[Any]] | None = None,
    ) -> int:
        """Build a complete replacement generation, then atomically publish it.

        The old generation remains readable until the new collection, metadata,
        and progress files have all been built under a temporary name.
        """
        if not self.available:
            raise RuntimeError(self.unavailable_reason)
        active_store = store or self.store
        if active_store is None:
            raise RuntimeError('ZVEC indexing requires a SQLiteStore.')
        publish = generation_publish or nullcontext
        if self._atomic_recovery_required():
            with publish():
                self.recover_atomic_rebuild()
        tmp_path = Path(str(self.collection_path) + f'.trove-tmp-{os.getpid()}-{int(time.time() * 1000)}')
        tmp = self._generation_paths(tmp_path)
        self._remove_generation(tmp)
        builder = self.__class__(
            tmp_path,
            store=active_store,
            memory_limit_mb=self.memory_limit_mb,
            ledger_backend=self.ledger_backend,
        )
        builder._zvec = self._zvec
        builder._error = self._error
        try:
            indexed = builder.index_all_messages(
                provider,
                store=active_store,
                ledger_store=ledger_store,
                batch_size=batch_size,
                max_messages=max_messages,
                activate_generation=False,
            )
            # Building the temporary collection is intentionally outside the
            # publication lease: active readers keep using the old complete
            # generation.  Only the verified swap and its recovery window are
            # exclusive.
            with publish():
                self._swap_tmp_generation_into_place(tmp_path)
        except Exception:
            with publish():
                self.recover_atomic_rebuild()
            self._remove_generation(tmp)
            raise
        finally:
            builder._invalidate_collection_cache()
            self._invalidate_collection_cache()
        return indexed

    def _swap_tmp_generation_into_place(self, tmp_path: Path) -> None:
        final = self._final_generation_paths()
        backup = self._backup_generation_paths()
        tmp = self._generation_paths(tmp_path)
        if not self._tmp_generation_complete(tmp):
            raise RuntimeError('ZVEC temporary generation is incomplete')
        try:
            tmp_metadata = json.loads(tmp['metadata'].read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError('ZVEC temporary generation metadata is invalid') from exc
        new_generation_id = str(tmp_metadata.get('generation_id') or '')
        if not new_generation_id or self.ledger is None:
            raise RuntimeError('ZVEC temporary generation has no authoritative ledger')
        ready = self.ledger.generation(new_generation_id)
        if ready is None or ready.status != 'ready':
            raise RuntimeError('ZVEC temporary generation ledger is not ready')
        active = self.ledger.active_generation()
        previous_generation_id = active.generation_id if active is not None else None
        self._remove_generation(backup)
        marker_args = {
            'tmp_path': tmp_path,
            'operation': 'rebuild',
            'previous_generation_id': previous_generation_id,
            'new_generation_id': new_generation_id,
        }
        self._write_swap_marker(phase='prepared', **marker_args)
        try:
            self._atomic_failpoint('after_prepare')

            self._rename_existing(final['collection'], backup['collection'])
            self._write_swap_marker(phase='final_collection_to_backup', **marker_args)
            self._atomic_failpoint('after_final_collection_to_backup')

            self._rename_existing(final['metadata'], backup['metadata'])
            self._write_swap_marker(phase='final_metadata_to_backup', **marker_args)
            self._atomic_failpoint('after_final_metadata_to_backup')

            self._rename_existing(final['progress'], backup['progress'])
            self._write_swap_marker(phase='final_progress_to_backup', **marker_args)
            self._atomic_failpoint('after_final_progress_to_backup')

            self._rename_existing(tmp['collection'], final['collection'])
            self._write_swap_marker(phase='tmp_collection_to_final', **marker_args)
            self._atomic_failpoint('after_tmp_collection_to_final')

            self._rename_existing(tmp['metadata'], final['metadata'])
            self._write_swap_marker(phase='tmp_metadata_to_final', **marker_args)
            self._atomic_failpoint('after_tmp_metadata_to_final')

            self._rename_existing(tmp['progress'], final['progress'])
            self._write_swap_marker(phase='final_files_ready', **marker_args)
            self._atomic_failpoint('after_tmp_progress_to_final')

            self.ledger.activate(new_generation_id)
            self._atomic_failpoint('after_ledger_activation_before_marker')
            self._write_swap_marker(phase='final_ready', **marker_args)
            self._atomic_failpoint('after_ledger_activation')

            self._remove_generation(backup)
            self.ledger.prune_retired()
            self.swap_marker_path.unlink(missing_ok=True)
        except Exception:
            self.recover_atomic_rebuild()
            raise

    def index_all_messages(
        self,
        provider,
        *,
        store: SQLiteStore | None = None,
        ledger_store: SQLiteStore | None = None,
        batch_size: int = 256,
        max_messages: int | None = None,
        citations=None,
        activate_generation: bool = True,
    ) -> int:
        """Reconcile ZVEC against SQLite using a delta-proportional ledger.

        The JSON sidecar is deliberately constant-size. Citation hashes and the
        generation lifecycle live in indexed SQLite tables so an incremental
        update only reads and writes the requested citations.
        """
        if not self.available:
            raise RuntimeError(self.unavailable_reason)
        active_store = store or self.store
        if active_store is None:
            raise RuntimeError('ZVEC indexing requires a SQLiteStore.')
        batch_size = max(1, int(batch_size))
        if max_messages is not None:
            max_messages = max(0, int(max_messages))
        citation_filter = None if citations is None else list(dict.fromkeys(str(c) for c in citations if c))
        if citation_filter is not None and not citation_filter:
            return 0

        active_ledger_store = ledger_store or active_store
        self.store = active_ledger_store
        self.ledger = VectorIndexLedger(active_ledger_store, backend=self.ledger_backend)
        ledger = self.ledger

        incremental_replay = False
        if self._atomic_recovery_required():
            # Rebuild recovery may replace the final sidecars. Incremental
            # recovery intentionally leaves its marker in place for replay.
            recovery = self.recover_atomic_rebuild()
            incremental_replay = recovery.get('status') == 'incremental_replay_required'
        metadata = self._read_metadata()
        collection_exists = Path(self.collection_path).exists()
        generation_id = str(metadata.get('generation_id') or '')
        generation = ledger.generation(generation_id) if generation_id else None
        contract_mismatch = collection_exists and _index_contract_mismatch(metadata, provider)

        incremental_ready = bool(
            collection_exists
            and metadata.get('schema_version') == 4
            and (
                metadata.get('complete') is True
                or metadata.get('catchup_pending') is True
            )
            and generation is not None
            and generation.status == 'active'
            and not contract_mismatch
        )
        if citation_filter is not None and not incremental_ready:
            raise RuntimeError(
                'ZVEC incremental indexing requires a complete existing collection; '
                'run a full vector rebuild first.'
            )

        legacy_or_incompatible = bool(
            collection_exists
            and (
                contract_mismatch
                or metadata.get('schema_version') != 4
                or generation is None
            )
        )
        if legacy_or_incompatible:
            if not activate_generation:
                self._reset_collection()
                metadata = {}
                collection_exists = False
                generation_id = ''
                generation = None
            else:
                # Never rewrite an already-published collection in place when
                # its contract/ledger is unknown. Build and publish a new
                # generation through the crash-safe swap path instead.
                return self.atomic_rebuild(
                    provider,
                    store=active_store,
                    batch_size=batch_size,
                    max_messages=max_messages,
                )

        expected_count = self._expected_document_count(active_store)
        new_generation = generation is None
        if new_generation:
            dimensions = int(getattr(provider, 'dimensions', 0) or 0)
            if dimensions <= 0:
                dimensions = len(provider.embed('trove dimension probe'))
                provider.dimensions = dimensions
            provider_contract = _provider_metadata(provider)
            generation_id = uuid.uuid4().hex
            generation = ledger.begin_generation(
                generation_id,
                vector_text_version=VECTOR_TEXT_VERSION,
                embedding_provider=str(provider_contract.get('embedding_provider') or ''),
                embedding_model=str(provider_contract.get('embedding_model') or ''),
                dimensions=dimensions,
                expected_count=expected_count,
            )

        active_generation = ledger.active_generation()
        mutating_active_generation = bool(
            active_generation is not None
            and active_generation.generation_id == generation_id
        )
        marker_args = {
            'operation': 'incremental',
            'previous_generation_id': generation_id if mutating_active_generation else None,
            'new_generation_id': generation_id,
        }
        if mutating_active_generation:
            self._write_swap_marker(phase='incremental_prepared', **marker_args)

        def flush_active_delta(*, phase: str) -> None:
            if not mutating_active_generation:
                return
            collection.flush()
            self._write_swap_marker(phase=phase, **marker_args)
            self._atomic_failpoint('after_incremental_collection_flush')

        current_count = generation.indexed_count
        progress_base = {
            'generation_id': generation_id,
            'expected_document_count': expected_count,
            'max_messages': max_messages,
            'dirty_count': len(citation_filter) if citation_filter is not None else None,
            'complete': False,
        }
        self._write_progress({
            **progress_base,
            'state': 'opening',
            'visited': 0,
            'changed_indexed_count': 0,
            'deleted_count': 0,
            'indexed_count': current_count,
        })
        # This writer owns the incremental marker it just created (or is
        # replaying). All other direct/read openings remain fail-closed.
        collection = self._open_or_create(
            provider,
            allow_recovery=mutating_active_generation or incremental_replay,
        )

        indexed = 0
        deleted = 0
        visited = 0
        truncated_by_max_messages = False
        scan_started = citation_filter is None and generation is not None
        if scan_started:
            ledger.begin_full_scan()

        try:
            if citation_filter is not None:
                candidates = ledger.citations_for_dirty(generation_id, citation_filter)
                if candidates:
                    active = self._active_vector_citations_for_dirty(active_store, citation_filter)
                    stale_citations = [citation for citation in candidates if citation not in active]
                    if stale_citations:
                        deleted += self._delete_documents(collection, stale_citations)
                        flush_active_delta(phase='incremental_collection_flushed')
                        ledger.apply_delta(generation_id, deletes=stale_citations, expected_count=expected_count)

            self._write_progress({
                **progress_base,
                'state': 'running',
                'visited': visited,
                'changed_indexed_count': indexed,
                'deleted_count': deleted,
                'indexed_count': ledger.generation(generation_id).indexed_count,
            })
            self._atomic_failpoint('during_tmp_generation_progress')

            pending_rows: list[tuple[dict[str, Any], str, str]] = []

            def apply_upsert_batch() -> None:
                nonlocal indexed, pending_rows
                if not pending_rows:
                    return
                citations_in_batch = [citation for _, citation, _ in pending_rows]
                if scan_started:
                    ledger.record_seen(citations_in_batch)
                current_hashes = ledger.hashes(generation_id, citations_in_batch)
                changed_rows: list[dict[str, Any]] = []
                for source, citation, digest in pending_rows:
                    if current_hashes.get(citation) == digest:
                        continue
                    changed = dict(source)
                    changed['vector_text'] = _embedding_text(source)
                    changed['content_hash'] = digest
                    changed_rows.append(changed)
                if changed_rows:
                    indexed += self._upsert_batch(collection, provider, changed_rows)
                    flush_active_delta(phase='incremental_collection_flushed')
                    ledger.apply_delta(
                        generation_id,
                        upserts=((row['citation'], row['content_hash']) for row in changed_rows),
                        expected_count=expected_count,
                    )
                pending_rows = []

            for source_row in active_store.iter_vector_documents(
                batch_size=batch_size,
                citations=citation_filter,
            ):
                if max_messages is not None and visited >= max_messages:
                    truncated_by_max_messages = True
                    break
                visited += 1
                citation = str(source_row['citation'])
                text = _embedding_text(source_row)
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                pending_rows.append((dict(source_row), citation, digest))
                if len(pending_rows) >= batch_size:
                    apply_upsert_batch()
                    row_count = ledger.generation(generation_id).indexed_count
                    self._write_progress({
                        **progress_base,
                        'state': 'running',
                        'visited': visited,
                        'changed_indexed_count': indexed,
                        'deleted_count': deleted,
                        'indexed_count': row_count,
                    })
            apply_upsert_batch()

            max_messages_limited = bool(
                max_messages is not None
                and (
                    truncated_by_max_messages
                    or (expected_count is not None and visited < expected_count)
                )
            )
            if scan_started and not max_messages_limited:
                for stale_batch in ledger.stale_after_full_scan(generation_id, limit=batch_size):
                    deleted += self._delete_documents(collection, stale_batch)
                    flush_active_delta(phase='incremental_collection_flushed')
                    ledger.apply_delta(generation_id, deletes=stale_batch, expected_count=expected_count)

            collection.flush()
            if mutating_active_generation:
                self._write_swap_marker(phase='incremental_ledger_committed', **marker_args)
                self._atomic_failpoint('after_incremental_ledger')
            ledger.apply_delta(generation_id, expected_count=expected_count)
            refreshed = ledger.generation(generation_id)
            if refreshed is None:
                raise RuntimeError('vector generation disappeared during indexing')
            complete = bool(
                not max_messages_limited
                and (expected_count is None or refreshed.indexed_count >= expected_count)
            )
            publishable_generation = refreshed.status in {'building', 'ready'}
            published_generation = False
            if publishable_generation and complete:
                ledger.mark_ready(generation_id, expected_count=expected_count)
                if activate_generation:
                    ledger.activate(generation_id)
                    published_generation = True
                refreshed = ledger.generation(generation_id)
                if refreshed is None:
                    raise RuntimeError('vector generation disappeared during publication')

            next_metadata = {
                'schema_version': 4,
                'generation_id': generation_id,
                'generation_revision': refreshed.revision,
                'collection_contract_version': ZVEC_COLLECTION_CONTRACT_VERSION,
                'vector_text_version': VECTOR_TEXT_VERSION,
                'dimensions': int(getattr(provider, 'dimensions', 0) or 0),
                **_provider_metadata(provider),
                'indexed_count': refreshed.indexed_count,
                'changed_indexed_count': indexed,
                'deleted_count': deleted,
                'expected_document_count': expected_count,
                'complete': complete,
                'catchup_pending': bool(mutating_active_generation and not complete),
                'max_messages_limited': max_messages_limited,
                'last_dirty_count': len(citation_filter) if citation_filter is not None else None,
                'backend': 'zvec',
            }
            prior_calibration = metadata.get('score_calibration')
            if isinstance(prior_calibration, dict):
                calibrated_next = {**next_metadata, 'score_calibration': prior_calibration}
                if score_calibration_status(calibrated_next, provider).get('state') == 'available':
                    # Inserts and deletes advance the crash-detection revision
                    # but keep the model/metric/vector-text score domain.
                    next_metadata['score_calibration'] = prior_calibration
            self._write_metadata(next_metadata)
            self._write_progress({
                **progress_base,
                'state': 'complete' if complete else 'partial',
                'visited': visited,
                'changed_indexed_count': indexed,
                'deleted_count': deleted,
                'indexed_count': refreshed.indexed_count,
                'complete': complete,
            })
            if mutating_active_generation:
                self.swap_marker_path.unlink(missing_ok=True)
            if published_generation:
                ledger.prune_retired()
            return indexed
        except BaseException:
            if new_generation:
                ledger.discard(generation_id)
            raise
        finally:
            if scan_started:
                ledger.end_full_scan()

    def _active_vector_citations_for_dirty(self, store: SQLiteStore, citations: list[str]) -> set[str]:
        active: set[str] = set()
        store.initialize()
        with store.connect() as conn:
            has_chunks = store._table_exists(conn, 'evidence_chunks') and int(conn.execute('SELECT COUNT(*) FROM evidence_chunks').fetchone()[0]) > 0
            for start in range(0, len(citations), 500):
                batch = citations[start:start + 500]
                if not batch:
                    continue
                placeholders = ','.join('?' for _ in batch)
                if has_chunks:
                    for row in conn.execute(
                        f"""SELECT chunk_citation FROM evidence_chunks
                            WHERE status='active'
                              AND (parent_citation IN ({placeholders}) OR chunk_citation IN ({placeholders}))""",
                        [*batch, *batch],
                    ):
                        active.add(str(row['chunk_citation']))
                elif store._table_exists(conn, 'messages'):
                    for row in conn.execute(
                        f"SELECT citation FROM messages WHERE citation IN ({placeholders})",
                        batch,
                    ):
                        active.add(str(row['citation']))
        return active

    def apply_precomputed_delta(
        self,
        provider,
        *,
        rows: list[dict[str, Any]],
        deletes: list[str],
        expected_count: int | None,
    ) -> dict[str, int]:
        """Commit an already embedded incremental delta without provider calls."""

        if self.ledger is None:
            raise RuntimeError('ZVEC incremental commit requires a ledger')
        metadata = self._read_metadata()
        generation_id = str(metadata.get('generation_id') or '')
        generation = self.ledger.generation(generation_id) if generation_id else None
        if (
            not generation_id
            or generation is None
            or generation.status != 'active'
            or metadata.get('schema_version') != 4
            or (
                metadata.get('complete') is not True
                and metadata.get('catchup_pending') is not True
            )
            or _index_contract_mismatch(metadata, provider)
        ):
            raise RuntimeError('ZVEC incremental indexing requires a complete matching collection')
        collection = self._open_existing(allow_recovery=True)
        marker_args = {
            'operation': 'incremental',
            'previous_generation_id': generation_id,
            'new_generation_id': generation_id,
        }
        self._write_swap_marker(phase='incremental_prepared', **marker_args)
        try:
            deleted = self._delete_documents(collection, list(dict.fromkeys(deletes)))
            indexed = self._upsert_precomputed_batch(collection, rows)
            collection.flush()
            self._write_swap_marker(phase='incremental_collection_flushed', **marker_args)
            self.ledger.apply_delta(
                generation_id,
                upserts=((str(row['citation']), str(row['content_hash'])) for row in rows),
                deletes=deletes,
                expected_count=expected_count,
            )
            refreshed = self.ledger.generation(generation_id)
            if refreshed is None:
                raise RuntimeError('vector generation disappeared during incremental commit')
            complete = bool(expected_count is None or refreshed.indexed_count >= expected_count)
            next_metadata = {
                **metadata,
                'schema_version': 4,
                'generation_id': generation_id,
                'generation_revision': refreshed.revision,
                'collection_contract_version': ZVEC_COLLECTION_CONTRACT_VERSION,
                'vector_text_version': VECTOR_TEXT_VERSION,
                'dimensions': int(getattr(provider, 'dimensions', 0) or 0),
                **_provider_metadata(provider),
                'indexed_count': refreshed.indexed_count,
                'changed_indexed_count': indexed,
                'deleted_count': deleted,
                'expected_document_count': expected_count,
                'complete': complete,
                'catchup_pending': not complete,
                'max_messages_limited': False,
                'last_dirty_count': len(rows) + len(deletes),
                'backend': 'zvec',
            }
            # Incremental changes keep the same model, metric, dimensions, and
            # vector-text contract, so their similarity score domain is stable.
            # Preserve the generation calibration; full rebuilds publish a new
            # generation id and still require a fresh calibration.
            self._write_metadata(next_metadata)
            self._write_progress({
                'generation_id': generation_id,
                'state': 'complete' if complete else 'partial',
                'visited': len(rows),
                'changed_indexed_count': indexed,
                'deleted_count': deleted,
                'indexed_count': refreshed.indexed_count,
                'expected_document_count': expected_count,
                'complete': complete,
            })
            self._write_swap_marker(phase='incremental_ledger_committed', **marker_args)
            self.swap_marker_path.unlink(missing_ok=True)
            return {'indexed': indexed, 'deleted': deleted}
        except BaseException:
            self._invalidate_collection_cache()
            raise

    def _delete_documents(self, collection, citations: list[str]) -> int:
        if not citations:
            return 0
        deleted = 0
        for start in range(0, len(citations), 500):
            batch = citations[start:start + 500]
            ids = [self._doc_id(citation) for citation in batch]
            try:
                collection.delete(ids)
            except TypeError:
                for doc_id in ids:
                    collection.delete(doc_id)
            deleted += len(batch)
        return deleted

    def _invalidate_collection_cache(self) -> None:
        self._collection = None

    def _reset_collection(self) -> None:
        self._invalidate_collection_cache()
        path = Path(self.collection_path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        if self.progress_path.exists():
            self.progress_path.unlink()

    def reset_collection(self) -> None:
        self._write_progress({
            'state': 'resetting',
            'visited': 0,
            'changed_indexed_count': 0,
            'indexed_count': 0,
            'expected_document_count': None,
            'max_messages': None,
            'complete': False,
        })
        self._reset_collection()
        self._write_progress({
            'state': 'reset',
            'visited': 0,
            'changed_indexed_count': 0,
            'indexed_count': 0,
            'expected_document_count': None,
            'max_messages': None,
            'complete': False,
        })

    def apply_score_calibration_file(self, artifact_path: str | Path, *, provider: Any) -> dict[str, Any]:
        """Bind one manifested, immutable dev artifact to this generation."""

        path = Path(artifact_path).expanduser()
        manifest = verify_evidence_manifest(path, required=True)
        if not isinstance(manifest, dict):  # pragma: no cover - required=True
            raise VectorScoreCalibrationError('vector_score_calibration_manifest_invalid')
        try:
            artifact_bytes = path.read_bytes()
            if sha256_bytes(artifact_bytes) != manifest.get('artifact_sha256'):
                raise VectorScoreCalibrationError('vector_score_calibration_manifest_mismatch')
            artifact = json.loads(artifact_bytes.decode('utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VectorScoreCalibrationError('vector_score_calibration_artifact_invalid') from exc
        metadata = self._authoritative_score_metadata()
        if not metadata or metadata.get('complete') is not True:
            raise VectorScoreCalibrationError('vector_score_calibration_index_incomplete')
        calibration = validate_score_calibration_artifact(
            artifact,
            metadata=metadata,
            provider=provider,
            # This is a local runtime trust decision, not release evidence.
            # The manifest and generation/model bindings remain mandatory,
            # while a dirty source tree is recorded in provenance rather than
            # making local calibration impossible during development.
            release=False,
        )
        calibration['artifact_sha256'] = str(manifest.get('artifact_sha256') or '')
        calibration['artifact_manifest_sha256'] = stable_payload_sha256(manifest)
        latest_metadata = self._authoritative_score_metadata()
        if index_identity(latest_metadata) != index_identity(metadata):
            raise VectorScoreCalibrationError('vector_score_calibration_index_mismatch')
        next_metadata = dict(latest_metadata)
        next_metadata['score_calibration'] = calibration
        self._write_metadata(next_metadata)
        return score_calibration_status(next_metadata, provider)

    def calibration_candidates(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = ZVEC_ADAPTIVE_OVERFETCH_MAX,
        provider: Any,
    ) -> list[tuple[Any, float]]:
        """Return local scored rows only for explicit dev calibration tooling.

        This bypasses the score floor but not cardinality bounds, filters, model
        identity, or finite-score validation. It is intentionally separate from
        the product search API so uncalibrated vectors cannot become evidence.
        """

        return self._search_scored(
            query,
            filters=filters,
            limit=limit,
            provider=provider,
            require_calibration=False,
        )

    def search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10, provider=None):
        return [
            row
            for row, _score in self._search_scored(
                query,
                filters=filters,
                limit=limit,
                provider=provider,
                require_calibration=True,
            )
        ]

    def _search_scored(
        self,
        query: str,
        *,
        filters: dict[str, str] | None,
        limit: int,
        provider: Any,
        require_calibration: bool,
    ) -> list[tuple[Any, float]]:
        from trove_core.bounds import BoundedLimit, RETRIEVAL_CANDIDATES

        if require_calibration:
            limit = BoundedLimit(limit, field='limit', spec=RETRIEVAL_CANDIDATES)
        else:
            try:
                limit = max(1, min(int(limit), ZVEC_ADAPTIVE_OVERFETCH_MAX))
            except (TypeError, ValueError) as exc:
                raise ValueError('calibration limit must be an integer') from exc
        if not self.available:
            raise RuntimeError(self.unavailable_reason)
        if provider is None:
            raise RuntimeError('ZVEC search requires an embedding provider.')
        if self.store is None:
            raise RuntimeError('ZVEC search requires a SQLiteStore.')
        if require_calibration and bool(getattr(provider, 'supports_sparse', False)):
            return self._search_hybrid_scored(
                query,
                filters=filters,
                limit=int(limit),
                provider=provider,
            )
        metadata = self._authoritative_score_metadata()
        score_domain_identity = index_identity(metadata)
        if _provider_contract_mismatch(metadata, provider, require_present=True):
            raise VectorScoreCalibrationError('vector_score_calibration_model_mismatch')
        calibration = score_calibration_status(metadata, provider)
        if require_calibration and calibration.get('state') != 'available':
            self._search_local.status = {
                'bounded': True,
                'score_calibration': calibration,
                'returned_count': 0,
                'unscored_count': 0,
                'score_rejected_count': 0,
            }
            raise VectorScoreCalibrationError(
                str(calibration.get('reason_code') or 'vector_score_calibration_missing')
            )
        score_floor = float(calibration['inclusive_min_score']) if require_calibration else None
        collection = self._open_existing()
        embed_query = getattr(provider, 'embed_query', provider.embed)
        qv = embed_query(query)
        zvec = self._zvec
        filters = filters or {}
        filter_expression, residual_filters, pushed_keys, ignored_keys = _zvec_filter_plan(filters)
        max_depth = ZVEC_ADAPTIVE_OVERFETCH_MAX
        depth = int(limit) if not residual_filters else min(max_depth, max(int(limit) * 3, 50))
        attempts: list[int] = []
        pushdown_rejected = False
        scored_rows: list[tuple[Any, float]] = []
        exhausted = False
        score_floor_exhausted = False
        score_rejected_count = 0
        unscored_count = 0
        while True:
            attempts.append(depth)
            qparam = None
            try:
                qparam = zvec.HnswQueryParam(ef=min(max_depth, max(200, depth * 2)))
            except Exception:
                qparam = None
            try:
                docs = collection.query(
                    zvec.Query('embedding', vector=qv, param=qparam),
                    topk=depth,
                    filter=filter_expression,
                    include_vector=False,
                    output_fields=['citation'],
                )
            except ValueError:
                if not filter_expression or pushdown_rejected:
                    raise
                # Older/newer ZVEC filter dialects must degrade safely rather
                # than fail search.  All filters become bounded residual work.
                pushdown_rejected = True
                filter_expression = None
                residual_filters = dict(filters)
                pushed_keys = []
                ignored_keys = []
                depth = min(max_depth, max(int(limit) * 3, 50))
                continue
            current_metadata = self._authoritative_score_metadata()
            if index_identity(current_metadata) != score_domain_identity:
                raise VectorScoreCalibrationError('vector_score_calibration_index_mismatch')
            if require_calibration:
                current_calibration = score_calibration_status(current_metadata, provider)
                if current_calibration.get('state') != 'available':
                    raise VectorScoreCalibrationError(
                        str(current_calibration.get('reason_code') or 'vector_score_calibration_missing')
                    )
            doc_scores: list[float] = []
            for doc in docs:
                try:
                    score = float(doc.score)
                except (AttributeError, TypeError, ValueError):
                    score = math.nan
                if not math.isfinite(score):
                    unscored_count += 1
                doc_scores.append(score)
            if unscored_count:
                self._search_local.status = {
                    'bounded': True,
                    'score_calibration': calibration if require_calibration else {'state': 'calibration_mode'},
                    'returned_count': 0,
                    'unscored_count': unscored_count,
                    'score_rejected_count': 0,
                }
                raise VectorScoreCalibrationError('vector_unscored')
            citations = [str((doc.fields or {}).get('citation') or doc.id) for doc in docs]
            evidence_by_citation = self.store.evidence_by_citations(citations)
            scored_rows = []
            score_floor_exhausted = False
            score_rejected_count = 0
            for citation, score in zip(citations, doc_scores):
                if score_floor is not None and score < score_floor:
                    score_rejected_count += 1
                    score_floor_exhausted = True
                    continue
                row = evidence_by_citation.get(citation)
                if row is None or not self.store._filter_row(row, residual_filters):
                    continue
                scored_rows.append((row, score))
                if len(scored_rows) >= limit:
                    break
            exhausted = len(docs) < depth or score_floor_exhausted
            if len(scored_rows) >= limit or exhausted or depth >= max_depth or not residual_filters:
                break
            depth = min(max_depth, depth * 2)

        self._search_local.status = {
            'filter_pushdown': bool(pushed_keys),
            'pushdown_keys': pushed_keys,
            'residual_keys': sorted(residual_filters),
            'ignored_keys': ignored_keys,
            'pushdown_rejected': pushdown_rejected,
            'adaptive_overfetch': bool(residual_filters),
            'attempt_depths': attempts,
            'max_depth': max_depth,
            'bounded': True,
            'exhausted': exhausted,
            'complete': bool(len(scored_rows) >= limit or exhausted),
            'returned_count': len(scored_rows),
            'score_calibration': calibration if require_calibration else {'state': 'calibration_mode'},
            'score_floor_exhausted': score_floor_exhausted,
            'score_rejected_count': score_rejected_count,
            'unscored_count': unscored_count,
        }
        return scored_rows

    def _search_hybrid_scored(
        self,
        query: str,
        *,
        filters: dict[str, str] | None,
        limit: int,
        provider: Any,
    ) -> list[tuple[Any, float]]:
        """Fuse calibrated dense and native sparse ranks without mixing score scales."""

        from trove_core.bounds import RETRIEVAL_CANDIDATES

        hybrid_depth = min(RETRIEVAL_CANDIDATES.maximum, max(limit * 3, 50))
        hybrid = provider.embed_query_hybrid(query)
        dense_rows = self._search_scored(
            query,
            filters=filters,
            limit=hybrid_depth,
            provider=_DenseOnlyProvider(provider, hybrid.dense),
            require_calibration=True,
        )
        dense_status = dict(getattr(self._search_local, 'status', {}) or {})
        if not hybrid.sparse:
            dense_status['hybrid'] = {
                'enabled': True,
                'dense_candidates': len(dense_rows),
                'sparse_candidates': 0,
                'fusion': 'rrf',
            }
            self._search_local.status = dense_status
            return dense_rows[:limit]

        zvec = self._zvec
        collection = self._open_existing()
        filters = filters or {}
        filter_expression, residual_filters, pushed_keys, ignored_keys = _zvec_filter_plan(filters)
        depth = hybrid_depth
        try:
            sparse_docs = collection.query(
                zvec.Query('sparse_embedding', vector=hybrid.sparse),
                topk=depth,
                filter=filter_expression,
                include_vector=False,
                output_fields=['citation'],
            )
            sparse_pushdown_rejected = False
        except ValueError:
            if not filter_expression:
                raise
            sparse_docs = collection.query(
                zvec.Query('sparse_embedding', vector=hybrid.sparse),
                topk=depth,
                filter=None,
                include_vector=False,
                output_fields=['citation'],
            )
            residual_filters = dict(filters)
            pushed_keys = []
            ignored_keys = []
            sparse_pushdown_rejected = True

        sparse_citations = [str((doc.fields or {}).get('citation') or doc.id) for doc in sparse_docs]
        sparse_evidence = self.store.evidence_by_citations(sparse_citations)
        sparse_rows: list[Any] = []
        for citation in sparse_citations:
            row = sparse_evidence.get(citation)
            if row is not None and self.store._filter_row(row, residual_filters):
                sparse_rows.append(row)

        def citation_of(row: Any) -> str:
            try:
                return str(row['citation'] or '')
            except Exception:
                return ''

        rows_by_citation: dict[str, Any] = {}
        scores: dict[str, float] = {}
        for rows in ([row for row, _score in dense_rows], sparse_rows):
            for rank, row in enumerate(rows, start=1):
                citation = citation_of(row)
                if not citation:
                    continue
                rows_by_citation.setdefault(citation, row)
                scores[citation] = scores.get(citation, 0.0) + 1.0 / (60.0 + rank)
        ranked = sorted(scores, key=lambda citation: (-scores[citation], citation))[:limit]
        self._search_local.status = {
            **dense_status,
            'hybrid': {
                'enabled': True,
                'dense_candidates': len(dense_rows),
                'sparse_candidates': len(sparse_rows),
                'fused_candidates': len(scores),
                'fusion': 'rrf',
                'sparse_pushdown_keys': pushed_keys,
                'sparse_residual_keys': sorted(residual_filters),
                'sparse_ignored_keys': ignored_keys,
                'sparse_pushdown_rejected': sparse_pushdown_rejected,
            },
            'returned_count': len(ranked),
        }
        return [(rows_by_citation[citation], scores[citation]) for citation in ranked]

    def last_search_status(self) -> dict[str, Any]:
        """Return redacted per-thread search telemetry for the calling route."""

        return dict(getattr(self._search_local, 'status', {}) or {})

    def _open_existing(self, *, allow_recovery: bool = False):
        if not self.available:
            raise RuntimeError(self.unavailable_reason)
        recovery_reason = self._atomic_recovery_reason()
        if recovery_reason is not None and not allow_recovery:
            self._invalidate_collection_cache()
            raise VectorScoreCalibrationError(recovery_reason)
        if not Path(self.collection_path).exists():
            self._invalidate_collection_cache()
            raise RuntimeError('ZVEC collection does not exist; run vector indexing first.')
        if self._collection is None:
            self._collection = self._zvec.open(self.collection_path)
        return self._collection

    def _open_or_create(self, provider, *, allow_recovery: bool = False):
        path = Path(self.collection_path)
        if path.exists():
            return self._open_existing(allow_recovery=allow_recovery)
        path.parent.mkdir(parents=True, exist_ok=True)
        dim = int(getattr(provider, 'dimensions', 0) or 0)
        if dim <= 0:
            sample = provider.embed('trove dimension probe')
            dim = len(sample)
            provider.dimensions = dim
        index_param = self._index_param()
        vector_schemas = [
            self._zvec.VectorSchema('embedding', self._zvec.DataType.VECTOR_FP32, dimension=dim, index_param=index_param)
        ]
        if bool(getattr(provider, 'supports_sparse', False)):
            vector_schemas.append(
                self._zvec.VectorSchema('sparse_embedding', self._zvec.DataType.SPARSE_VECTOR_FP32)
            )
        schema = self._zvec.CollectionSchema(
            'trove_messages',
            fields=[
                self._zvec.FieldSchema('citation', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('account_id', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('conversation_id', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('conversation_type', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('sender_id', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('sender_name', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('timestamp', self._zvec.DataType.STRING),
            ],
            vectors=vector_schemas,
        )
        self._collection = self._zvec.create_and_open(self.collection_path, schema)
        return self._collection

    def _index_param(self):
        try:
            return self._zvec.HnswIndexParam(metric_type=self._zvec.MetricType.IP)
        except Exception:
            return self._zvec.FlatIndexParam(metric_type=self._zvec.MetricType.IP)

    def _upsert_batch(self, collection, provider, rows: list[Any]) -> int:
        if not rows:
            return 0
        hybrid = bool(getattr(provider, 'supports_sparse', False))
        if hybrid:
            embeddings = provider.embed_hybrid_many(
                [_embedding_text(row) for row in rows], text_type='document'
            )
        else:
            embeddings = provider.embed_many([_embedding_text(row) for row in rows])
        docs = []
        for row, embedded in zip(rows, embeddings):
            dense = embedded.dense if hybrid else embedded
            vectors = {'embedding': [float(x) for x in dense]}
            if hybrid:
                vectors['sparse_embedding'] = {
                    int(index): float(value) for index, value in embedded.sparse.items()
                }
            docs.append(self._zvec.Doc(
                id=self._doc_id(row['citation']),
                fields={
                    'citation': row['citation'],
                    'account_id': row['account_id'],
                    'conversation_id': row['conversation_id'],
                    'conversation_type': row['conversation_type'],
                    'sender_id': row['sender_id'],
                    'sender_name': row['sender_name'],
                    'timestamp': row['timestamp'],
                },
                vectors=vectors,
            ))
        _upsert_zvec_docs(collection, docs)
        return len(docs)

    def _upsert_precomputed_batch(self, collection, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        docs = [
            self._zvec.Doc(
                id=self._doc_id(str(row['citation'])),
                fields={
                    'citation': row['citation'],
                    'account_id': row['account_id'],
                    'conversation_id': row['conversation_id'],
                    'conversation_type': row['conversation_type'],
                    'sender_id': row['sender_id'],
                    'sender_name': row['sender_name'],
                    'timestamp': row['timestamp'],
                },
                vectors={
                    'embedding': [float(value) for value in row['vector']],
                    **(
                        {'sparse_embedding': {int(index): float(value) for index, value in row['sparse_vector'].items()}}
                        if isinstance(row.get('sparse_vector'), dict) else {}
                    ),
                },
            )
            for row in rows
        ]
        _upsert_zvec_docs(collection, docs)
        return len(docs)

    def _doc_id(self, citation: str) -> str:
        return 'm' + hashlib.sha256(citation.encode('utf-8')).hexdigest()[:31]

    def _expected_document_count(self, store: SQLiteStore | None = None) -> int | None:
        active_store = store or self.store
        if active_store is None or not active_store.path.exists():
            return None
        try:
            with active_store.connect() as conn:
                if active_store._table_exists(conn, 'evidence_chunks'):
                    chunks = int(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE status='active'").fetchone()[0])
                    if chunks > 0:
                        return chunks
                if active_store._table_exists(conn, 'messages'):
                    return int(conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0])
        except Exception:
            return None
        return None
