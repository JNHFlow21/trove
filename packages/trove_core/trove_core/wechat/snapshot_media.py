from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import shutil
from typing import Any

from trove_core.wechat.media_mapping_assessment import d0_mapping_conclusion

DEFAULT_WECHAT_FILES_ROOT = Path('~/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files').expanduser()
DEFAULT_MAX_COPY_BYTES = 20 * 1024 * 1024 * 1024


def _extract_wxid(value: str) -> str:
    match = re.search(r'wxid_[A-Za-z0-9]+(?:_[0-9a-fA-F]{4})?', str(value or ''))
    return match.group(0) if match else ''


def _account_hash(value: str) -> str:
    return hashlib.sha256(str(value or '').encode('utf-8')).hexdigest()[:12]


@dataclass(frozen=True)
class SnapshotMediaFile:
    source: Path
    target: Path
    size_bytes: int


def _iter_allowed_roots(account_dir: Path) -> list[tuple[Path, Path]]:
    roots: list[tuple[Path, Path]] = []
    cache_root = account_dir / 'cache'
    if cache_root.exists():
        for month in sorted(cache_root.iterdir()):
            if not month.is_dir() or not re.fullmatch(r'\d{4}-\d{2}', month.name):
                continue
            src = month / 'sns' / 'img'
            if src.exists():
                roots.append((src, Path('cache') / month.name / 'sns' / 'img'))
    for rel in (Path('business') / 'sns' / 'bkg', Path('business') / 'sns' / 'publish'):
        src = account_dir / rel
        if src.exists():
            roots.append((src, rel))
    return roots


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file():
            continue
        files.append(path)
    return sorted(files)


def _copy_needed(source: Path, target: Path) -> bool:
    if not target.exists():
        return True
    try:
        src_stat = source.stat()
        dst_stat = target.stat()
    except OSError:
        return True
    return src_stat.st_size != dst_stat.st_size or int(src_stat.st_mtime) > int(dst_stat.st_mtime)


def build_snapshot_media_plan(snapshot_dir: Path, *, wechat_root: Path | None = None) -> dict[str, Any]:
    """Plan allowlist-only SNS media cache copies into an existing snapshot."""
    snapshot_dir = Path(snapshot_dir).expanduser()
    wechat_root = Path(wechat_root or DEFAULT_WECHAT_FILES_ROOT).expanduser()
    report: dict[str, Any] = {
        'enabled': True,
        'status': 'planned',
        'source_root_exists': wechat_root.exists(),
        'snapshot_exists': snapshot_dir.exists(),
        'account_sources': 0,
        'account_targets': 0,
        'matched_accounts': 0,
        'files_total': 0,
        'bytes_total': 0,
        'raw_paths_included': False,
        'd0_mapping_conclusion': d0_mapping_conclusion(),
    }
    if not wechat_root.exists() or not snapshot_dir.exists():
        report['status'] = 'skipped'
        report['reason'] = 'source_or_snapshot_missing'
        return report | {'files': []}
    source_by_wxid: dict[str, Path] = {}
    for child in sorted(wechat_root.iterdir()):
        if not child.is_dir():
            continue
        wxid = _extract_wxid(child.name)
        if wxid:
            source_by_wxid.setdefault(wxid, child)
    targets: list[Path] = []
    for child in sorted(snapshot_dir.iterdir()):
        if not child.is_dir():
            continue
        wxid = _extract_wxid(child.name)
        if wxid:
            targets.append(child)
    files: list[SnapshotMediaFile] = []
    matched_accounts = 0
    matched_hashes: list[str] = []
    for target_account in targets:
        wxid = _extract_wxid(target_account.name)
        source_account = source_by_wxid.get(wxid)
        if source_account is None:
            continue
        matched_accounts += 1
        matched_hashes.append(_account_hash(wxid))
        for source_root, rel_root in _iter_allowed_roots(source_account):
            for source in _iter_files(source_root):
                rel = source.relative_to(source_root)
                try:
                    size = source.stat().st_size
                except OSError:
                    continue
                files.append(SnapshotMediaFile(source=source, target=target_account / rel_root / rel, size_bytes=int(size)))
    report.update({
        'account_sources': len(source_by_wxid),
        'account_targets': len(targets),
        'matched_accounts': matched_accounts,
        'matched_account_hashes': sorted(set(matched_hashes)),
        'files_total': len(files),
        'bytes_total': sum(item.size_bytes for item in files),
    })
    return report | {'files': files}


def refresh_snapshot_media_cache(
    snapshot_dir: Path,
    *,
    wechat_root: Path | None = None,
    max_bytes: int = DEFAULT_MAX_COPY_BYTES,
) -> dict[str, Any]:
    plan = build_snapshot_media_plan(snapshot_dir, wechat_root=wechat_root)
    files = list(plan.pop('files', []))
    if plan.get('status') == 'skipped':
        return plan
    total = int(plan.get('bytes_total') or 0)
    if total > max_bytes:
        plan.update({'status': 'skipped', 'reason': 'copy_size_exceeds_limit', 'max_bytes': max_bytes})
        return plan
    copied_files = 0
    copied_bytes = 0
    skipped_existing = 0
    for item in files:
        if not _copy_needed(item.source, item.target):
            skipped_existing += 1
            continue
        item.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, item.target)
        copied_files += 1
        copied_bytes += item.size_bytes
    plan.update({
        'status': 'copied',
        'max_bytes': max_bytes,
        'copied_files': copied_files,
        'copied_bytes': copied_bytes,
        'skipped_existing_files': skipped_existing,
        'raw_paths_included': False,
    })
    return plan
