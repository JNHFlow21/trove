from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
import os
import re
import subprocess
from typing import Any

from .config import ProviderConfig, agent_switch_secret_names, configured_cost_cap_rmb

PRIVATE_PATH_RE = re.compile(r'/Users/[A-Za-z0-9_.-]+/[^\s"\'<>)]*')
MEDIA_NAME_RE = re.compile(r'(?i)\b[^\s/]+\.(jpg|jpeg|png|gif|webp|heic|mp3|m4a|wav|amr|silk|mp4|mov|dat)\b')
SECRETISH_RE = re.compile(r'(?i)(api[_-]?key|secret|token|authorization)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}')
PROVIDER_PAYLOAD_RE = re.compile(r'(?i)(provider_payload|raw_payload|audio_info|input_tokens|output_tokens|image_observation|transcript)')
RAW_TEXT_RE = re.compile(r'(?i)(raw_message|raw transcript|原文|聊天原文|message_content)')


@dataclass(frozen=True)
class CloudReadinessInput:
    repo_root: Path
    vault_root: Path | None = None
    cost_cap_rmb: float | None = None
    estimated_cost_rmb: float | None = None
    doc_verification_date: str | None = None
    provider_docs_ok: bool = False
    selected_account_ids: list[str] = field(default_factory=list)
    discovered_account_ids: list[str] = field(default_factory=list)
    undecryptable_account_ids: list[str] = field(default_factory=list)
    coverage_gap_account_ids: list[str] = field(default_factory=list)
    redaction_probe: str = ''
    require_clean_git: bool = True
    require_usage_store: bool = True


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str
    hard_stop: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CloudReadinessReport:
    ok: bool
    generated_at: str
    hard_stops: list[ReadinessIssue]
    warnings: list[ReadinessIssue]
    provider_status: dict[str, Any]
    cost: dict[str, Any]
    scope: dict[str, Any]
    redaction: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'generated_at': self.generated_at,
            'hard_stops': [i.to_dict() for i in self.hard_stops],
            'warnings': [i.to_dict() for i in self.warnings],
            'provider_status': self.provider_status,
            'cost': self.cost,
            'scope': self.scope,
            'redaction': self.redaction,
        }


def _today() -> str:
    return date.today().isoformat()


def _git_clean(repo_root: Path) -> tuple[bool, list[str]]:
    try:
        out = subprocess.check_output(['git', '-C', str(repo_root), 'status', '--porcelain'], text=True)
    except Exception as exc:
        return False, [f'git status unavailable: {exc.__class__.__name__}']
    lines = [line for line in out.splitlines() if line.strip()]
    return not lines, lines[:20]


def _workspace_safe(repo_root: Path) -> bool:
    text = str(repo_root.resolve())
    return (
        '/Library/Mobile Documents/' not in text
        and '/Knowledge_OS/' not in text
        and not text.endswith('/Knowledge_OS')
    )


def redaction_issues(text: str) -> list[str]:
    if not text:
        return []
    issues: list[str] = []
    if PRIVATE_PATH_RE.search(text):
        issues.append('private_path')
    if SECRETISH_RE.search(text):
        issues.append('secret_or_token')
    if PROVIDER_PAYLOAD_RE.search(text):
        issues.append('provider_payload_or_transcript_marker')
    if RAW_TEXT_RE.search(text):
        issues.append('raw_text_marker')
    if MEDIA_NAME_RE.search(text):
        issues.append('media_filename')
    return issues


def check_cloud_processing_readiness(
    params: CloudReadinessInput,
    *,
    env: dict[str, str] | None = None,
    agent_switch_names: set[str] | None = None,
) -> CloudReadinessReport:
    env = env if env is not None else os.environ
    names = agent_switch_names if agent_switch_names is not None else agent_switch_secret_names()
    provider_cfg = ProviderConfig.resolve(env, agent_switch_names=names, check_agent_switch=False)
    provider_status = provider_cfg.to_redacted_dict(env, agent_switch_names=names, check_agent_switch=False)
    issues: list[ReadinessIssue] = []
    warnings: list[ReadinessIssue] = []

    if not _workspace_safe(params.repo_root):
        issues.append(ReadinessIssue('workspace_unsafe', 'Repo root must be a local checkout, not iCloud or Knowledge_OS.'))
    if params.require_clean_git:
        clean, dirty = _git_clean(params.repo_root)
        if not clean:
            issues.append(ReadinessIssue('git_dirty', 'Git working tree is not clean; real cloud processing cannot start with unstaged/uncommitted files.'))
            warnings.append(ReadinessIssue('git_dirty_sample', '; '.join(dirty), hard_stop=False))

    asr = provider_status['providers']['asr']
    vision = provider_status['providers']['vision']
    if not asr['configured']:
        issues.append(ReadinessIssue('asr_secret_missing', f'{asr["secret"]["name"]} is not available through Agent Switch FD transport.'))
    if not vision['configured']:
        issues.append(ReadinessIssue('vision_secret_missing', f'{vision["secret"]["name"]} is not available through Agent Switch FD transport.'))
    if provider_cfg.asr_model_name != 'bigmodel' or provider_cfg.asr_resource_id != 'volc.bigasr.auc_turbo':
        issues.append(ReadinessIssue('asr_provider_not_pinned', 'ASR provider must be pinned to model_name=bigmodel and resource_id=volc.bigasr.auc_turbo.'))
    if provider_cfg.ark_vision_model != 'doubao-seed-2-0-lite-260215':
        issues.append(ReadinessIssue('vision_model_not_default', 'Ark Vision Lite default must be doubao-seed-2-0-lite-260215 unless a later verified plan changes it.'))

    if not params.provider_docs_ok or params.doc_verification_date != _today():
        issues.append(ReadinessIssue('provider_docs_stale_or_missing', 'Official ASR/Vision docs must be verified on the current execution date before upload.'))

    cap = params.cost_cap_rmb if params.cost_cap_rmb is not None else configured_cost_cap_rmb(env)
    if cap is None:
        warnings.append(ReadinessIssue(
            'asr_cost_unlimited',
            'Cloud ASR has no blocking RMB ceiling; usage is still estimated and recorded.',
            hard_stop=False,
        ))
    if params.estimated_cost_rmb is None:
        issues.append(ReadinessIssue('cost_estimate_missing', 'Preflight estimated cost is required before real cloud processing.'))
    elif cap is not None and params.estimated_cost_rmb > cap:
        issues.append(ReadinessIssue('cost_estimate_exceeds_cap', 'Estimated cloud cost exceeds configured local cap.'))

    usage_store_ok = False
    if params.vault_root is not None:
        jobs_dir = params.vault_root / 'jobs'
        proof_dir = params.vault_root / 'proof'
        try:
            jobs_dir.mkdir(parents=True, exist_ok=True)
            proof_dir.mkdir(parents=True, exist_ok=True)
            usage_store_ok = jobs_dir.exists() and os.access(jobs_dir, os.W_OK)
        except Exception:
            usage_store_ok = False
    if params.require_usage_store and not usage_store_ok:
        issues.append(ReadinessIssue('usage_store_unavailable', 'Provider usage accounting must be persistable in the runtime Vault jobs area.'))

    selected = set(params.selected_account_ids)
    discovered = set(params.discovered_account_ids)
    undecryptable = set(params.undecryptable_account_ids)
    gaps = set(params.coverage_gap_account_ids)
    unauthorized = sorted(discovered - selected) if selected else []
    if unauthorized:
        issues.append(ReadinessIssue('unauthorized_account_in_scope', 'Discovered accounts outside the selected approved scope.'))
    missing_gaps = sorted(undecryptable - gaps)
    if missing_gaps:
        issues.append(ReadinessIssue('undecryptable_without_coverage_gap', 'Undecryptable accounts must be recorded as coverage gaps.'))

    redaction = redaction_issues(params.redaction_probe)
    if redaction:
        issues.append(ReadinessIssue('redaction_probe_failed', 'Redaction probe contains forbidden raw/private/provider/media material.'))

    cost = {
        'currency': 'RMB',
        'cap_configured': cap is not None,
        'cap_rmb': cap,
        'estimated_cost_rmb': params.estimated_cost_rmb,
        'estimate_within_cap': bool(params.estimated_cost_rmb is not None and (cap is None or params.estimated_cost_rmb <= cap)),
    }
    scope = {
        'selected_account_count': len(selected),
        'discovered_account_count': len(discovered),
        'unauthorized_account_count': len(unauthorized),
        'undecryptable_account_count': len(undecryptable),
        'coverage_gap_count': len(gaps),
        'unauthorized_account_ids': unauthorized,
        'undecryptable_missing_coverage_gap_ids': missing_gaps,
    }
    redaction_payload = {'ok': not redaction, 'issues': redaction, 'raw_probe_included': False}
    return CloudReadinessReport(
        ok=not issues,
        generated_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        hard_stops=issues,
        warnings=warnings,
        provider_status=provider_status,
        cost=cost,
        scope=scope,
        redaction=redaction_payload,
    )
