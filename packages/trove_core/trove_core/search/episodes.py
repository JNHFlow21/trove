from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Iterable

from trove_core.providers.cloud_policy import cloud_retrieval_policy
from trove_core.providers.factory import EnvironmentAgentSwitchSecretResolver
from trove_core.store.sqlite_store import SQLiteStore, open_store, vector_document_text
from trove_core.vector.score_calibration import embedding_identity
from trove_core.vector.zvec_store import _zvec_string_literal


EPISODE_CONTRACT_VERSION = 2
EPISODE_WINDOW_BEFORE = 3
EPISODE_WINDOW_AFTER = 3
EPISODE_STRIDE = 3
EPISODE_MESSAGE_CHARS = 220
EPISODE_BUILD_BATCH_SIZE = 120
# The Beijing realtime price for text-embedding-v4 is RMB 0.5 / million input
# tokens.  Keep the price snapshot next to the cap so a future pricing change is
# an explicit code/config review rather than an accidental unbounded run.
EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION = 0.5
EPISODE_CLOUD_COST_CAP_RMB = 80.0
EPISODE_QUERY_INSTRUCT = (
    'Given a multi-hop Chinese chat-history query, retrieve a conversation episode '
    'that contains the complete evidence chain.'
)
SELECTOR_MODEL = 'qwen3.6-flash'
SELECTOR_ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
SELECTOR_INPUT_USD_PER_MILLION = 0.165
SELECTOR_OUTPUT_USD_PER_MILLION = 0.99
_ZVEC_MAX_WRITE_BATCH = 1024


def _httpx_post(*args: Any, **kwargs: Any) -> Any:
    from trove_core.providers.http_pool import post

    return post(*args, **kwargs)


@dataclass(frozen=True)
class EpisodeHit:
    episode_id: str
    citations: tuple[str, ...]
    score: float


def _upsert_zvec_docs(collection: Any, docs: list[Any]) -> int:
    """Upsert docs without exceeding zvec's per-call write batch limit."""

    for start in range(0, len(docs), _ZVEC_MAX_WRITE_BATCH):
        collection.upsert(docs[start:start + _ZVEC_MAX_WRITE_BATCH])
    return len(docs)


def episode_collection_path(vector_dir: Path) -> Path:
    return vector_dir / 'zvec' / 'episodes-cloud'


class EpisodeCloudBudgetExceeded(RuntimeError):
    """A resumable episode build reached its durable cloud request budget."""

    def __init__(self) -> None:
        super().__init__('episode_cloud_cost_cap_reached')


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _value(row: Any, key: str) -> str:
    try:
        return str(row[key] or '')
    except Exception:
        return ''


def _episode_row(rows: list[Any], center: int) -> dict[str, Any] | None:
    window = rows[max(0, center - EPISODE_WINDOW_BEFORE):center + EPISODE_WINDOW_AFTER + 1]
    citations = tuple(_value(row, 'citation') for row in window if _value(row, 'citation'))
    if not citations:
        return None
    anchor = _value(rows[center], 'citation')
    episode_id = 'e' + hashlib.sha256(
        ('\x1f'.join((_value(rows[center], 'account_id'), _value(rows[center], 'conversation_id'), anchor))).encode('utf-8')
    ).hexdigest()[:31]
    lines = []
    for row in window:
        line = ' | '.join(
            part for part in (
                _value(row, 'timestamp'),
                _value(row, 'sender_name'),
                _value(row, 'content')[:EPISODE_MESSAGE_CHARS],
            ) if part
        )
        if line:
            lines.append(line)
    return {
        'episode_id': episode_id,
        'account_id': _value(rows[center], 'account_id'),
        'conversation_id': _value(rows[center], 'conversation_id'),
        'timestamp': _value(rows[center], 'timestamp'),
        'citations': citations,
        'text': '会话证据片段:\n' + '\n'.join(lines),
    }


def _episode_rows(rows: list[Any]) -> Iterable[dict[str, Any]]:
    for center in range(0, len(rows), EPISODE_STRIDE):
        if item := _episode_row(rows, center):
            yield item


def _incremental_episode_rows(
    rows: list[Any],
    dirty_citations: set[str],
    *,
    has_tombstone: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Use a bounded tail update for append-only dirty suffixes.

    A normal realtime import appends one contiguous suffix. Existing episode
    anchors stay stable, so only the last overlapping windows need new cloud
    embeddings. Mid-conversation inserts and deletions can shift every later
    stride anchor and deliberately retain the full-conversation fallback.
    """

    if not rows:
        return 'full', []
    positions = [
        index for index, row in enumerate(rows)
        if _value(row, 'citation') in dirty_citations
    ]
    contiguous_suffix = bool(positions) and positions == list(range(positions[0], len(rows)))
    if has_tombstone or not contiguous_suffix:
        return 'full', list(_episode_rows(rows))
    first_affected = max(0, positions[0] - EPISODE_WINDOW_AFTER)
    start_center = (first_affected // EPISODE_STRIDE) * EPISODE_STRIDE
    planned = [
        item for center in range(start_center, len(rows), EPISODE_STRIDE)
        if (item := _episode_row(rows, center)) is not None
    ]
    return 'tail', planned


def iter_episode_documents(sqlite_path: Path) -> Iterable[dict[str, Any]]:
    store = open_store(sqlite_path, readonly=True)
    conn = store.connect_once()
    try:
        cursor = conn.execute(
            """SELECT citation,account_id,conversation_id,timestamp,sender_name,content,local_id
                 FROM messages
                ORDER BY account_id,conversation_id,timestamp,local_id,citation"""
        )
        current_key: tuple[str, str] | None = None
        rows: list[Any] = []
        for row in cursor:
            key = (_value(row, 'account_id'), _value(row, 'conversation_id'))
            if current_key is not None and key != current_key:
                yield from _episode_rows(rows)
                rows = []
            current_key = key
            rows.append(row)
        if rows:
            yield from _episode_rows(rows)
    finally:
        conn.close()
        store.close()


class EpisodeZVecStore:
    def __init__(self, path: str | Path, *, store: SQLiteStore | None = None):
        self.path = Path(path)
        self.metadata_path = Path(str(self.path) + '.trove-meta.json')
        self.store = store
        self._collection = None
        try:
            import zvec  # type: ignore
        except Exception as exc:
            self._zvec = None
            self._error = exc
        else:
            self._zvec = zvec
            self._error = None

    @property
    def staging_path(self) -> Path:
        return Path(str(self.path) + '.trove-building')

    @property
    def staging_metadata_path(self) -> Path:
        return Path(str(self.staging_path) + '.trove-meta.json')

    def _metadata(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def status(self, provider: Any | None = None) -> dict[str, Any]:
        metadata = self._metadata()
        try:
            staging = json.loads(self.staging_metadata_path.read_text(encoding='utf-8'))
        except Exception:
            staging = {}
        exists = self.path.exists()
        identity = embedding_identity(provider)['sha256'] if provider is not None else None
        mismatch = bool(exists and identity and metadata.get('embedding_identity_sha256') != identity)
        complete = bool(metadata.get('complete') is True)
        return {
            'state': 'available' if exists and complete and not mismatch else 'unavailable_fallback',
            'collection_exists': exists,
            'complete': complete,
            'indexed_count': int(metadata.get('indexed_count') or 0),
            'expected_document_count': int(metadata.get('expected_document_count') or 0),
            'provider_mismatch': mismatch,
            'contract_version': metadata.get('contract_version'),
            'staging': {
                'state': staging.get('state'),
                'resumable': bool(
                    self.staging_path.exists()
                    and staging.get('complete') is not True
                    and staging.get('state') in {
                        'building', 'embedding_complete_pending_upsert',
                        'paused_budget', 'paused_provider_error', 'paused_source_changed',
                    }
                ),
                'complete': bool(staging.get('complete') is True),
                'indexed_count': int(staging.get('indexed_count') or 0),
                'expected_document_count': int(staging.get('expected_document_count') or 0),
                'estimated_cost_rmb': float((staging.get('budget') or {}).get('estimated_cost_rmb') or 0.0),
                'cost_cap_rmb': float((staging.get('budget') or {}).get('cost_cap_rmb') or 0.0),
                'raw_content_included': False,
            },
            'raw_content_included': False,
        }

    def _open(self):
        if self._zvec is None or not self.path.exists():
            raise RuntimeError('episode_zvec_unavailable')
        if self._collection is None:
            self._collection = self._zvec.open(str(self.path))
        return self._collection

    @staticmethod
    def _collection_count(collection: Any) -> int:
        try:
            return int(collection.stats.doc_count)
        except Exception:
            docs = getattr(collection, 'docs_by_id', None)
            if isinstance(docs, dict):
                return len(docs)
            raise RuntimeError('episode_stage_count_unavailable')

    @staticmethod
    def _budget_payload(
        *,
        cost_cap_rmb: float,
        consumed_input_tokens: int = 0,
        provider_reported_input_tokens: int = 0,
        provider_calls: int = 0,
        failed_attempt_reserved_tokens: int = 0,
        inflight_reserved_tokens: int = 0,
    ) -> dict[str, Any]:
        return {
            'cost_cap_rmb': round(float(cost_cap_rmb), 6),
            'unit_price_rmb_per_million_tokens': EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION,
            'consumed_input_tokens': max(0, int(consumed_input_tokens)),
            'provider_reported_input_tokens': max(0, int(provider_reported_input_tokens)),
            'provider_calls': max(0, int(provider_calls)),
            'failed_attempt_reserved_tokens': max(0, int(failed_attempt_reserved_tokens)),
            'inflight_reserved_tokens': max(0, int(inflight_reserved_tokens)),
            'estimated_cost_rmb': round(
                max(0, int(consumed_input_tokens))
                * EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION / 1_000_000,
                6,
            ),
        }

    def _quarantine_incompatible_stage(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if not self.staging_path.exists() and not self.staging_metadata_path.exists():
            return {}
        if not metadata:
            raise RuntimeError('episode_stage_metadata_missing')
        stamp = f'{int(time.time() * 1000)}'
        stale_path = Path(str(self.staging_path) + f'.stale-{stamp}')
        stale_meta = Path(str(stale_path) + '.trove-meta.json')
        if self.staging_path.exists():
            self.staging_path.rename(stale_path)
        if self.staging_metadata_path.exists():
            self.staging_metadata_path.rename(stale_meta)
        budget = metadata.get('budget') if isinstance(metadata.get('budget'), dict) else {}
        # Carry the conservative spend ledger into the replacement stage.  A
        # source/provider mismatch must never silently reset the user's cap.
        return dict(budget)

    def rebuild(
        self,
        provider: Any,
        *,
        cfg: Any,
        source_snapshot: str,
        cost_cap_rmb: float = EPISODE_CLOUD_COST_CAP_RMB,
    ) -> dict[str, Any]:
        if self._zvec is None:
            raise RuntimeError('episode_zvec_unavailable')
        if not bool(getattr(provider, 'supports_sparse', False)):
            raise RuntimeError('episode_hybrid_provider_required')
        if not math.isfinite(float(cost_cap_rmb)) or not 0 < float(cost_cap_rmb) <= 10_000:
            raise ValueError('episode_cloud_cost_cap_invalid')
        tmp = self.staging_path
        tmp_meta = self.staging_metadata_path
        backup = Path(str(self.path) + '.trove-backup')
        backup_meta = Path(str(self.metadata_path) + '.trove-backup')
        # Recover an interrupted prior swap before touching the resumable
        # staging collection.  A backup is never discarded merely because a
        # later build started.
        if backup.exists() and not self.path.exists():
            backup.rename(self.path)
        if backup_meta.exists() and not self.metadata_path.exists():
            backup_meta.rename(self.metadata_path)
        if backup.exists() or backup_meta.exists():
            raise RuntimeError('episode_previous_swap_requires_recovery')
        dim = int(getattr(provider, 'dimensions', 0) or 0)
        if dim <= 0:
            dim = len(provider.embed('trove episode dimension probe'))
            provider.dimensions = dim
        identity = embedding_identity(provider)['sha256']
        contract = {
            'contract_version': EPISODE_CONTRACT_VERSION,
            'embedding_identity_sha256': identity,
            'dimensions': dim,
            'source_snapshot_sha256': source_snapshot,
            'window_before': EPISODE_WINDOW_BEFORE,
            'window_after': EPISODE_WINDOW_AFTER,
            'stride': EPISODE_STRIDE,
            'message_chars': EPISODE_MESSAGE_CHARS,
        }
        publication_started = False
        try:
            stage_metadata = json.loads(tmp_meta.read_text(encoding='utf-8'))
        except FileNotFoundError:
            stage_metadata = {}
        except Exception as exc:
            raise RuntimeError('episode_stage_metadata_invalid') from exc
        compatible = bool(stage_metadata) and all(stage_metadata.get(key) == value for key, value in contract.items())
        carried_budget: dict[str, Any] = {}
        if (tmp.exists() or tmp_meta.exists()) and not compatible:
            carried_budget = self._quarantine_incompatible_stage(stage_metadata)

        try:
            index_param = self._zvec.HnswIndexParam(metric_type=self._zvec.MetricType.IP)
        except Exception:
            index_param = self._zvec.FlatIndexParam(metric_type=self._zvec.MetricType.IP)
        schema = self._zvec.CollectionSchema(
            'trove_episodes',
            fields=[
                self._zvec.FieldSchema('episode_id', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('account_id', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('conversation_id', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('timestamp', self._zvec.DataType.STRING),
                self._zvec.FieldSchema('citations_json', self._zvec.DataType.STRING),
            ],
            vectors=[
                self._zvec.VectorSchema('embedding', self._zvec.DataType.VECTOR_FP32, dimension=dim, index_param=index_param),
                self._zvec.VectorSchema('sparse_embedding', self._zvec.DataType.SPARSE_VECTOR_FP32),
            ],
        )
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if compatible and tmp.exists():
            collection = self._zvec.open(str(tmp))
            indexed = self._collection_count(collection)
            expected = int(stage_metadata.get('expected_document_count') or 0)
            if indexed < int(stage_metadata.get('indexed_count') or 0) or indexed > expected:
                raise RuntimeError('episode_stage_progress_invalid')
            stage_metadata['indexed_count'] = indexed
        else:
            from trove_core.runtime import _raise_if_vector_source_changed

            expected = 0
            source_chars = 0
            for item in iter_episode_documents(cfg.paths.sqlite_path):
                expected += 1
                source_chars += len(item['text'])
            _raise_if_vector_source_changed(cfg, source_snapshot)
            collection = self._zvec.create_and_open(str(tmp), schema)
            budget = self._budget_payload(
                cost_cap_rmb=cost_cap_rmb,
                consumed_input_tokens=int(carried_budget.get('consumed_input_tokens') or 0),
                provider_reported_input_tokens=int(carried_budget.get('provider_reported_input_tokens') or 0),
                provider_calls=int(carried_budget.get('provider_calls') or 0),
                failed_attempt_reserved_tokens=int(carried_budget.get('failed_attempt_reserved_tokens') or 0),
            )
            stage_metadata = {
                'schema_version': 2,
                **contract,
                'complete': False,
                'state': 'building',
                'indexed_count': 0,
                'expected_document_count': expected,
                'source_character_count': source_chars,
                'budget': budget,
                'raw_content_included': False,
            }
            _atomic_json(tmp_meta, stage_metadata)
            indexed = 0

        budget = stage_metadata.get('budget') if isinstance(stage_metadata.get('budget'), dict) else {}
        persisted_cap = float(budget.get('cost_cap_rmb') or cost_cap_rmb)
        # A resume may lower the cap, but it may not silently raise it.  Raising
        # a campaign cap requires an explicit new stage/campaign.
        effective_cap = min(float(cost_cap_rmb), persisted_cap)
        inflight = int(budget.get('inflight_reserved_tokens') or 0)
        if inflight:
            budget['failed_attempt_reserved_tokens'] = int(budget.get('failed_attempt_reserved_tokens') or 0) + inflight
            budget['inflight_reserved_tokens'] = 0
        budget['cost_cap_rmb'] = effective_cap
        stage_metadata['budget'] = budget
        _atomic_json(tmp_meta, stage_metadata)

        provider_start_tokens = int(getattr(provider, 'input_tokens', 0) or 0)
        provider_start_calls = int(getattr(provider, 'provider_calls', 0) or 0)
        reported_start = int(budget.get('provider_reported_input_tokens') or 0)
        calls_start = int(budget.get('provider_calls') or 0)

        def write_progress(*, state: str, complete: bool = False) -> None:
            current_tokens = max(0, int(getattr(provider, 'input_tokens', provider_start_tokens) or 0) - provider_start_tokens)
            current_calls = max(0, int(getattr(provider, 'provider_calls', provider_start_calls) or 0) - provider_start_calls)
            budget['provider_reported_input_tokens'] = reported_start + current_tokens
            budget['provider_calls'] = calls_start + current_calls
            budget['estimated_cost_rmb'] = round(
                int(budget.get('consumed_input_tokens') or 0)
                * EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION / 1_000_000,
                6,
            )
            stage_metadata.update({
                'state': state,
                'complete': bool(complete),
                'indexed_count': indexed,
                'budget': budget,
                'raw_content_included': False,
            })
            _atomic_json(tmp_meta, stage_metadata)

        batch: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal indexed, batch
            if not batch:
                return
            from trove_core.runtime import _raise_if_vector_source_changed

            try:
                _raise_if_vector_source_changed(cfg, source_snapshot)
            except Exception:
                write_progress(state='paused_source_changed')
                raise
            reserved = sum(len(item['text'].encode('utf-8')) for item in batch) + 1024
            consumed = int(budget.get('consumed_input_tokens') or 0)
            cap_tokens = int(effective_cap / EPISODE_EMBEDDING_PRICE_RMB_PER_MILLION * 1_000_000)
            if consumed + reserved > cap_tokens:
                write_progress(state='paused_budget')
                raise EpisodeCloudBudgetExceeded()
            budget['consumed_input_tokens'] = consumed + reserved
            budget['inflight_reserved_tokens'] = reserved
            write_progress(state='building')
            before_tokens = int(getattr(provider, 'input_tokens', 0) or 0)
            try:
                embeddings = provider.embed_hybrid_many([item['text'] for item in batch], text_type='document')
            except Exception:
                budget['inflight_reserved_tokens'] = 0
                budget['failed_attempt_reserved_tokens'] = int(budget.get('failed_attempt_reserved_tokens') or 0) + reserved
                write_progress(state='paused_provider_error')
                raise
            actual = max(0, int(getattr(provider, 'input_tokens', before_tokens) or 0) - before_tokens)
            if hasattr(provider, 'input_tokens'):
                budget['consumed_input_tokens'] = consumed + actual
            budget['inflight_reserved_tokens'] = 0
            write_progress(state='embedding_complete_pending_upsert')
            docs = []
            for item, embedding in zip(batch, embeddings):
                docs.append(self._zvec.Doc(
                    id=item['episode_id'],
                    fields={
                        'episode_id': item['episode_id'],
                        'account_id': item['account_id'],
                        'conversation_id': item['conversation_id'],
                        'timestamp': item['timestamp'],
                        'citations_json': json.dumps(item['citations'], ensure_ascii=False, separators=(',', ':')),
                    },
                    vectors={
                        'embedding': embedding.dense,
                        'sparse_embedding': embedding.sparse,
                    },
                ))
            _upsert_zvec_docs(collection, docs)
            indexed += len(docs)
            batch = []
            write_progress(state='building')

        try:
            scanned = 0
            for item in iter_episode_documents(cfg.paths.sqlite_path):
                if scanned < indexed:
                    scanned += 1
                    continue
                scanned += 1
                batch.append(item)
                if len(batch) >= EPISODE_BUILD_BATCH_SIZE:
                    flush()
            flush()
            collection.flush()
            if indexed != expected or scanned != expected or self._collection_count(collection) != expected:
                write_progress(state='incomplete_count_mismatch')
                raise RuntimeError('episode_collection_incomplete')
            metadata = {
                **stage_metadata,
                'complete': True,
                'state': 'ready_to_publish',
                'indexed_count': indexed,
                'expected_document_count': expected,
            }
            _atomic_json(tmp_meta, metadata)
            from trove_core.runtime import _raise_if_vector_source_changed
            from trove_core.vault.generation import coordinated_vault_generation_publish
            from trove_core.vault.mutations import coordinated_vault_mutation

            with coordinated_vault_mutation(cfg, operation='vector_rebuild'):
                _raise_if_vector_source_changed(cfg, source_snapshot)
                with coordinated_vault_generation_publish(cfg, operation='vector-rebuild-episodes'):
                    publication_started = True
                    if self.path.exists():
                        self.path.rename(backup)
                    if self.metadata_path.exists():
                        self.metadata_path.rename(backup_meta)
                    tmp.rename(self.path)
                    tmp_meta.rename(self.metadata_path)
                    if backup.exists():
                        shutil.rmtree(backup)
                    backup_meta.unlink(missing_ok=True)
            self._collection = None
            return self.status(provider) | {'ok': True}
        except Exception:
            if publication_started:
                # Move a newly activated-but-not-fully-published collection
                # back to staging before restoring the prior active pair.  The
                # paid new vectors and the last known-good active collection
                # both survive every swap failure.
                if self.path.exists() and backup.exists():
                    if not tmp.exists():
                        self.path.rename(tmp)
                    else:
                        self.path.rename(Path(str(tmp) + f'.swap-failed-{int(time.time() * 1000)}'))
                    backup.rename(self.path)
                elif not self.path.exists() and backup.exists():
                    backup.rename(self.path)
                elif self.path.exists() and not backup.exists() and not tmp.exists():
                    self.path.rename(tmp)
                if self.metadata_path.exists() and backup_meta.exists():
                    if not tmp_meta.exists():
                        self.metadata_path.rename(tmp_meta)
                    else:
                        self.metadata_path.rename(Path(str(tmp_meta) + f'.swap-failed-{int(time.time() * 1000)}'))
                    backup_meta.rename(self.metadata_path)
                elif not self.metadata_path.exists() and backup_meta.exists():
                    backup_meta.rename(self.metadata_path)
                elif self.metadata_path.exists() and not backup_meta.exists() and not tmp_meta.exists():
                    self.metadata_path.rename(tmp_meta)
            # The stable staging collection and its redacted progress/cost
            # ledger are intentionally retained.  No incomplete collection is
            # ever moved into the active path, and the next invocation resumes
            # without re-paying for completed batches.
            raise

    def search(self, query: str, *, provider: Any, filters: dict[str, str] | None = None, limit: int = 3) -> list[EpisodeHit]:
        if self.status(provider)['state'] != 'available':
            return []
        hybrid = provider.embed_hybrid_many(
            [query], text_type='query', instruct=EPISODE_QUERY_INSTRUCT
        )[0]
        collection = self._open()
        expression_parts = []
        for key in ('account_id', 'conversation_id'):
            value = (filters or {}).get(key)
            literal = _zvec_string_literal(str(value)) if value else None
            if literal:
                expression_parts.append(f'{key} = {literal}')
        expression = ' AND '.join(expression_parts) or None
        depth = min(120, max(20, int(limit) * 10))
        dense = collection.query(
            self._zvec.Query('embedding', vector=hybrid.dense), topk=depth,
            filter=expression, include_vector=False,
            output_fields=['episode_id', 'citations_json'],
        )
        sparse = collection.query(
            self._zvec.Query('sparse_embedding', vector=hybrid.sparse), topk=depth,
            filter=expression, include_vector=False,
            output_fields=['episode_id', 'citations_json'],
        ) if hybrid.sparse else []
        docs: dict[str, Any] = {}
        scores: dict[str, float] = {}
        for ranked in (dense, sparse):
            for rank, doc in enumerate(ranked, start=1):
                docs.setdefault(doc.id, doc)
                scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (60.0 + rank)
        episode_ids = sorted(scores, key=lambda value: (-scores[value], value))[:max(1, int(limit))]
        hits: list[EpisodeHit] = []
        for episode_id in episode_ids:
            try:
                citations = tuple(str(value) for value in json.loads(docs[episode_id].fields['citations_json']))
            except Exception:
                continue
            hits.append(EpisodeHit(episode_id, citations, scores[episode_id]))
        return hits

    def sync_dirty_conversations(self, provider: Any, *, cfg: Any, citations: list[str]) -> dict[str, Any]:
        """Re-embed only conversations touched by the dirty citation set."""

        if not citations or self.status(provider)['state'] != 'available':
            return {'state': 'skipped', 'conversation_count': 0, 'indexed_count': 0}
        from trove_core.runtime import _raise_if_vector_source_changed, _vector_source_snapshot
        from trove_core.vault.generation import coordinated_vault_generation_publish
        from trove_core.vault.mutations import coordinated_vault_mutation

        source_snapshot = _vector_source_snapshot(cfg)
        parent_citations = list(dict.fromkeys(str(value).split('#chunk-', 1)[0] for value in citations if value))
        keys: set[tuple[str, str]] = set()
        prepared: dict[tuple[str, str], list[dict[str, Any]]] = {}
        dirty_by_key: dict[tuple[str, str], set[str]] = {}
        tombstone_keys: set[tuple[str, str]] = set()
        update_mode: dict[tuple[str, str], str] = {}
        store = open_store(cfg.paths.sqlite_path, readonly=True)
        try:
            with store.connect() as conn:
                for start in range(0, len(parent_citations), 400):
                    batch = parent_citations[start:start + 400]
                    placeholders = ','.join('?' for _ in batch)
                    for table in ('messages', 'sync_citation_tombstones'):
                        if not store._table_exists(conn, table):
                            continue
                        for row in conn.execute(
                            f'SELECT citation,account_id,conversation_id FROM {table} WHERE citation IN ({placeholders})',
                            batch,
                        ):
                            key = (_value(row, 'account_id'), _value(row, 'conversation_id'))
                            if all(key):
                                keys.add(key)
                                if table == 'messages':
                                    dirty_by_key.setdefault(key, set()).add(_value(row, 'citation'))
                                else:
                                    tombstone_keys.add(key)
                for key in sorted(keys):
                    rows = list(conn.execute(
                        """SELECT citation,account_id,conversation_id,timestamp,sender_name,content,local_id
                             FROM messages WHERE account_id=? AND conversation_id=?
                            ORDER BY timestamp,local_id,citation""",
                        key,
                    ))
                    mode, items = _incremental_episode_rows(
                        rows,
                        dirty_by_key.get(key, set()),
                        has_tombstone=key in tombstone_keys,
                    )
                    update_mode[key] = mode
                    prepared[key] = items
        finally:
            store.close()

        all_items = [item for key in sorted(prepared) for item in prepared[key]]
        embeddings = provider.embed_hybrid_many(
            [item['text'] for item in all_items], text_type='document'
        ) if all_items else []
        docs_by_key: dict[tuple[str, str], list[Any]] = {key: [] for key in keys}
        for item, embedding in zip(all_items, embeddings):
            key = (item['account_id'], item['conversation_id'])
            docs_by_key[key].append(self._zvec.Doc(
                id=item['episode_id'],
                fields={
                    'episode_id': item['episode_id'],
                    'account_id': item['account_id'],
                    'conversation_id': item['conversation_id'],
                    'timestamp': item['timestamp'],
                    'citations_json': json.dumps(item['citations'], ensure_ascii=False, separators=(',', ':')),
                },
                vectors={'embedding': embedding.dense, 'sparse_embedding': embedding.sparse},
            ))
        _raise_if_vector_source_changed(cfg, source_snapshot)
        collection = self._open()
        with coordinated_vault_mutation(cfg, operation='vector_index'):
            _raise_if_vector_source_changed(cfg, source_snapshot)
            with coordinated_vault_generation_publish(cfg, operation='vector-index-episodes'):
                for account_id, conversation_id in sorted(keys):
                    account = _zvec_string_literal(account_id)
                    conversation = _zvec_string_literal(conversation_id)
                    if update_mode.get((account_id, conversation_id)) == 'full' and account and conversation:
                        collection.delete_by_filter(f'account_id = {account} AND conversation_id = {conversation}')
                    docs = docs_by_key[(account_id, conversation_id)]
                    if docs:
                        _upsert_zvec_docs(collection, docs)
                collection.flush()
                metadata = self._metadata()
                metadata.update({
                    'complete': True,
                    'indexed_count': int(collection.stats.doc_count),
                    'expected_document_count': int(collection.stats.doc_count),
                    'source_snapshot_sha256': source_snapshot,
                    'embedding_identity_sha256': embedding_identity(provider)['sha256'],
                })
                _atomic_json(self.metadata_path, metadata)
        return {
            'state': 'available',
            'conversation_count': len(keys),
            'indexed_count': len(all_items),
            'collection_count': int(collection.stats.doc_count),
            'tail_conversation_count': sum(mode == 'tail' for mode in update_mode.values()),
            'full_conversation_count': sum(mode == 'full' for mode in update_mode.values()),
        }


SELECTOR_SYSTEM_PROMPT = """You select a minimal evidence set for a private Chinese chat-history retrieval query.
Return one JSON object only and never answer the query.
Schema: {"selected_indices": [integer, integer]}
Rules:
- Select exactly 2 or 3 distinct candidate indices.
- The selected messages must jointly cover all facts, stages, entities, numbers, and time relations requested by the query.
- Prefer a coherent same-conversation or chronological evidence chain over individually similar distractors.
- `episode_rank` is the retrieved episode rank (0 is strongest) and `episode_position` is chronological position inside that episode; use them only to preserve a complete chain, never as a substitute for textual evidence.
- Use only provided indices. Do not invent evidence and do not call tools.
"""


class BoundedEvidenceSelector:
    def __init__(
        self,
        vault_root: str | Path,
        *,
        timeout: float = 8.0,
        post: Callable[..., Any] | None = None,
        secret_resolver: Any | None = None,
    ):
        self.vault_root = Path(vault_root)
        self.timeout = max(1.0, min(float(timeout), 30.0))
        self._post = post
        self._secret_resolver = secret_resolver or EnvironmentAgentSwitchSecretResolver()
        self._credential: str | None = None
        self._credential_lock = threading.Lock()

    def _credential_value(self) -> str:
        with self._credential_lock:
            if self._credential is None:
                self._credential = self._secret_resolver.resolve('DASHSCOPE_API_KEY')
            return self._credential

    def select(
        self,
        query: str,
        rows: list[Any],
        *,
        candidate_metadata: list[dict[str, int]] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        if not cloud_retrieval_policy(self.vault_root)['enabled'] or len(rows) < 2:
            return rows, {'state': 'skipped', 'reason_code': 'policy_disabled_or_insufficient_candidates'}
        documents = [vector_document_text(row)[:1600] for row in rows[:21]]
        credential = self._credential_value()
        metadata = list(candidate_metadata or [])
        candidates = []
        for index, document in enumerate(documents):
            item: dict[str, Any] = {'index': index, 'text': document}
            if index < len(metadata):
                for key in ('episode_rank', 'episode_position'):
                    value = metadata[index].get(key)
                    if type(value) is int and value >= 0:
                        item[key] = value
            candidates.append(item)

        started = time.perf_counter()
        response = (self._post or _httpx_post)(
            SELECTOR_ENDPOINT,
            headers={'Authorization': f'Bearer {credential}', 'Content-Type': 'application/json'},
            json={
                'model': SELECTOR_MODEL,
                'messages': [
                    {'role': 'system', 'content': SELECTOR_SYSTEM_PROMPT},
                    {'role': 'user', 'content': json.dumps({
                        'query': query,
                        'candidates': candidates,
                    }, ensure_ascii=False)},
                ],
                'temperature': 0,
                'response_format': {'type': 'json_object'},
                'enable_thinking': False,
                'max_tokens': 256,
            },
            timeout=self.timeout,
        )
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        data = response.json()
        if response.status_code >= 400:
            if response.status_code in {401, 403}:
                # A rotated Agent Switch credential should recover on the next
                # request instead of poisoning this long-lived runtime object.
                with self._credential_lock:
                    self._credential = None
            raise RuntimeError(f'evidence_selector_http_{response.status_code}')
        choices = data.get('choices') if isinstance(data, dict) else None
        content = ((choices[0].get('message') or {}).get('content')) if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        try:
            parsed = json.loads(content) if isinstance(content, str) else {}
            indices = parsed.get('selected_indices')
        except Exception as exc:
            raise RuntimeError('evidence_selector_invalid_json') from exc
        if not isinstance(indices, list):
            raise RuntimeError('evidence_selector_indices_missing')
        selected: list[int] = []
        for value in indices:
            if type(value) is not int or value < 0 or value >= len(documents):
                raise RuntimeError('evidence_selector_index_invalid')
            if value not in selected:
                selected.append(value)
        if not 2 <= len(selected) <= 3:
            raise RuntimeError('evidence_selector_count_invalid')
        selected_set = set(selected)
        ordered = [*(rows[index] for index in selected), *(row for index, row in enumerate(rows) if index not in selected_set)]
        usage = data.get('usage') if isinstance(data, dict) else {}
        input_tokens = int(usage.get('prompt_tokens') or 0) if isinstance(usage, dict) else 0
        output_tokens = int(usage.get('completion_tokens') or 0) if isinstance(usage, dict) else 0
        return ordered, {
            'state': 'available',
            'model': SELECTOR_MODEL,
            'selected_count': len(selected),
            'elapsed_ms': elapsed,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost_usd': round(
                input_tokens * SELECTOR_INPUT_USD_PER_MILLION / 1_000_000
                + output_tokens * SELECTOR_OUTPUT_USD_PER_MILLION / 1_000_000,
                9,
            ),
            'raw_content_included': False,
            'bundle_metadata_included': bool(candidate_metadata),
        }
