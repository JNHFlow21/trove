from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import uuid

from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultWriteSession
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from trove_core.vault.tracing import redact_value


@dataclass(frozen=True)
class ImportProcessConfig:
    config_id: str = 'default'
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 120
    multimodal: str = 'metadata_only'
    vector_index: str = 'diagnose_only'
    wiki_projection: str = 'cited_only'
    evaluation: str = 'fixture_and_redacted'
    allow_cloud_asr: bool = False
    allow_cloud_vision: bool = False
    cloud_retrieval: str = 'disabled'
    parent_child_evidence: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.chunk_max_chars < 120 or self.chunk_max_chars > 4000:
            errors.append('chunk_max_chars must be between 120 and 4000')
        if self.chunk_overlap_chars < 0 or self.chunk_overlap_chars >= self.chunk_max_chars:
            errors.append('chunk_overlap_chars must be >=0 and smaller than chunk_max_chars')
        if self.multimodal not in {'metadata_only', 'fixture_only', 'cloud_gated'}:
            errors.append('multimodal must be metadata_only, fixture_only, or cloud_gated')
        if self.vector_index not in {'off', 'diagnose_only', 'incremental', 'rebuild_with_approval'}:
            errors.append('vector_index must be off, diagnose_only, incremental, or rebuild_with_approval')
        if (self.allow_cloud_asr or self.allow_cloud_vision) and self.multimodal != 'cloud_gated':
            errors.append('cloud ASR/Vision requires multimodal=cloud_gated and a separate readiness gate')
        if self.cloud_retrieval not in {'disabled', 'enabled'}:
            errors.append('cloud_retrieval must be disabled or enabled')
        return errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['redacted_hash'] = self.redacted_hash()
        return data

    def redacted_hash(self) -> str:
        text = json.dumps(redact_value(asdict(self)), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def default_process_config() -> ImportProcessConfig:
    return ImportProcessConfig(config_id='pcfg-default')


def process_config_from_payload(payload: dict[str, Any] | None = None) -> ImportProcessConfig:
    payload = dict(payload or {})
    if not payload.get('config_id'):
        payload['config_id'] = 'pcfg-' + uuid.uuid4().hex[:12]
    allowed = set(ImportProcessConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return ImportProcessConfig(**filtered)


@mutation_entrypoint('process_config_write')
def write_process_config(
    vault_root: str | Path,
    config: ImportProcessConfig,
    *,
    write_session: VaultWriteSession | None = None,
) -> Path:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    with coordinated_vault_mutation(
        cfg,
        operation='process_config_write',
        write_session=write_session,
    ):
        cfg.ensure()
        path = cfg.paths.jobs_dir / 'process_configs' / f'{config.config_id}.redacted.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        latest = cfg.paths.jobs_dir / 'process_configs' / 'latest.redacted.json'
        latest.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def read_latest_process_config(vault_root: str | Path) -> dict[str, Any]:
    cfg = VaultConfig.resolve(str(vault_root), env={})
    path = cfg.paths.jobs_dir / 'process_configs' / 'latest.redacted.json'
    if not path.exists():
        return {'status': 'missing', 'config': default_process_config().to_dict()}
    return {'status': 'ok', 'config': json.loads(path.read_text(encoding='utf-8'))}
