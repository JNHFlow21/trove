from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import subprocess

from .hash_store import safe_derivative_path, sha256_file

SUPPORTED_ASR_SUFFIXES = {'.wav', '.mp3', '.m4a', '.amr', '.silk'}


@dataclass(frozen=True)
class AudioResolveResult:
    status: str
    input_hash: str | None = None
    output_hash: str | None = None
    derivative_ref: str | None = None
    codec: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_audio_file(input_path: Path, vault_root: Path, *, asset_id: str, ffmpeg: str = 'ffmpeg') -> AudioResolveResult:
    input_path = Path(input_path).expanduser()
    if not input_path.exists():
        return AudioResolveResult(status='missing_local_cache', error_code='missing_file')
    input_hash = sha256_file(input_path)
    root = Path(vault_root).expanduser().resolve()
    out = safe_derivative_path(root, 'media', 'audio', f'{asset_id}.wav')
    out.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == '.wav':
        shutil.copyfile(input_path, out)
        return AudioResolveResult(status='copied', input_hash=input_hash, output_hash=sha256_file(out), derivative_ref=str(out.relative_to(root)), codec='wav')
    ffmpeg_input = input_path
    ffmpeg_prefix: list[str] = []
    pcm_path: Path | None = None
    if input_path.suffix.lower() == '.silk':
        try:
            import pysilk
        except ImportError:
            return AudioResolveResult(status='decode_failed', input_hash=input_hash, error_code='silk_decoder_missing')
        pcm_path = out.with_suffix('.pcm')
        try:
            pcm_path.unlink(missing_ok=True)
            with input_path.open('rb') as source, pcm_path.open('wb') as pcm:
                pysilk.decode(source, pcm, 24000)
            if not pcm_path.is_file() or pcm_path.stat().st_size <= 0:
                raise ValueError('empty decoded pcm')
        except (OSError, RuntimeError, ValueError):
            pcm_path.unlink(missing_ok=True)
            return AudioResolveResult(status='decode_failed', input_hash=input_hash, error_code='silk_decode_failed')
        ffmpeg_input = pcm_path
        ffmpeg_prefix = ['-f', 's16le', '-ar', '24000', '-ac', '1']
    try:
        subprocess.run(
            [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', *ffmpeg_prefix, '-i', str(ffmpeg_input), '-ar', '16000', '-ac', '1', str(out)],
            check=True, capture_output=True, timeout=60,
        )
    except FileNotFoundError:
        return AudioResolveResult(status='decode_failed', input_hash=input_hash, error_code='ffmpeg_missing')
    except subprocess.SubprocessError:
        return AudioResolveResult(status='decode_failed', input_hash=input_hash, error_code='ffmpeg_failed')
    finally:
        if pcm_path is not None:
            pcm_path.unlink(missing_ok=True)
    return AudioResolveResult(status='normalized', input_hash=input_hash, output_hash=sha256_file(out), derivative_ref=str(out.relative_to(root)), codec='wav')
