from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

from .dat_decoder import decode_wechat_dat_file, sniff_image_type
from .hash_store import safe_derivative_path, sha256_file

VISION_READABLE_SUFFIXES = {'.heic', '.heif', '.webp'}


@dataclass(frozen=True)
class ImageResolveResult:
    status: str
    wrapper_type: str
    input_hash: str | None = None
    output_hash: str | None = None
    derivative_ref: str | None = None
    image_type: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_image_file(input_path: Path, vault_root: Path, *, asset_id: str) -> ImageResolveResult:
    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        return ImageResolveResult(status='missing_local_cache', wrapper_type='missing', error_code='missing_file')
    input_hash = sha256_file(input_path)
    if input_path.suffix.lower() == '.dat':
        decoded = decode_wechat_dat_file(input_path)
        if decoded.output_bytes is None:
            return ImageResolveResult(status=decoded.status, wrapper_type='wechat_dat_xor', input_hash=input_hash, error_code=decoded.error_code)
        ext = decoded.image_type or 'img'
        out = safe_derivative_path(vault_root, 'media', 'decoded', f'{asset_id}.{ext}')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(decoded.output_bytes)
        return ImageResolveResult(status='decoded' if decoded.status == 'decoded' else 'copied', wrapper_type='wechat_dat_xor', input_hash=input_hash, output_hash=sha256_file(out), derivative_ref=str(out.relative_to(Path(vault_root).expanduser().resolve())), image_type=ext)
    head = input_path.read_bytes()[:16]
    image_type = sniff_image_type(head)
    if not image_type and input_path.suffix.lower() in VISION_READABLE_SUFFIXES:
        image_type = input_path.suffix.lower().lstrip('.')
    if not image_type:
        return ImageResolveResult(status='decode_failed', wrapper_type='plain_file', input_hash=input_hash, error_code='unsupported_image_header')
    out = safe_derivative_path(vault_root, 'media', 'decoded', f'{asset_id}.{image_type}')
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, out)
    return ImageResolveResult(status='copied', wrapper_type='plain_file', input_hash=input_hash, output_hash=sha256_file(out), derivative_ref=str(out.relative_to(Path(vault_root).expanduser().resolve())), image_type=image_type)
