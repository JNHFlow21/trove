from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re

JPEG_MAGIC = b'\xff\xd8\xff'
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
GIF_MAGIC = b'GIF8'
MAGICS = [('jpg', JPEG_MAGIC), ('png', PNG_MAGIC), ('gif', GIF_MAGIC), ('hevc', b'wxgf')]
V1_MAGIC = b'\x07\x08V1\x08\x07'
V2_MAGIC = b'\x07\x08V2\x08\x07'
V1_AES_KEY = b'cfcd208495d565ef'


@dataclass(frozen=True)
class DatDecodeResult:
    status: str
    image_type: str | None = None
    xor_key: int | None = None
    output_bytes: bytes | None = None
    error_code: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data['output_bytes'] = None if self.output_bytes is None else f'<{len(self.output_bytes)} bytes>'
        return data


def sniff_image_type(data: bytes) -> str | None:
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    for image_type, magic in MAGICS:
        if data.startswith(magic):
            return image_type
    return None


def detect_xor_key(data: bytes) -> tuple[int, str] | None:
    if not data:
        return None
    for image_type, magic in MAGICS:
        key = data[0] ^ magic[0]
        sample = bytes(b ^ key for b in data[:len(magic)])
        if sample == magic:
            return key, image_type
    return None


def _decode_v2_bytes(data: bytes, *, aes_key: bytes, xor_key: int) -> DatDecodeResult:
    if len(data) < 15 or len(aes_key) != 16 or not 0 <= xor_key <= 255:
        return DatDecodeResult(status='decode_failed', error_code='v2_invalid_header')
    aes_size = int.from_bytes(data[6:10], 'little')
    xor_size = int.from_bytes(data[10:14], 'little')
    aligned_aes_size = aes_size + (16 - (aes_size % 16))
    aes_end = 15 + aligned_aes_size
    raw_end = len(data) - xor_size
    if aes_end > len(data) or raw_end < aes_end:
        return DatDecodeResult(status='decode_failed', error_code='v2_invalid_header')
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        decryptor = Cipher(algorithms.AES(aes_key), modes.ECB()).decryptor()
        decrypted = decryptor.update(data[15:aes_end]) + decryptor.finalize()
    except (ImportError, ValueError):
        return DatDecodeResult(status='decode_failed', error_code='v2_decrypt_failed')
    if not decrypted:
        return DatDecodeResult(status='decode_failed', error_code='v2_decrypt_failed')
    padding = decrypted[-1]
    if padding < 1 or padding > 16 or decrypted[-padding:] != bytes([padding]) * padding:
        return DatDecodeResult(status='decode_failed', error_code='v2_decrypt_failed')
    decrypted = decrypted[:-padding]
    decoded_tail = bytes(value ^ xor_key for value in data[raw_end:])
    output = decrypted + data[aes_end:raw_end] + decoded_tail
    image_type = sniff_image_type(output)
    if image_type is None:
        return DatDecodeResult(status='decode_failed', error_code='v2_decrypt_failed')
    return DatDecodeResult(
        status='decoded', image_type=image_type, xor_key=xor_key, output_bytes=output,
    )


def decode_wechat_dat_bytes(
    data: bytes,
    *,
    v2_aes_key: bytes | None = None,
    v2_xor_key: int | None = None,
) -> DatDecodeResult:
    direct = sniff_image_type(data)
    if direct:
        return DatDecodeResult(status='already_image', image_type=direct, output_bytes=data)
    if data.startswith(V1_MAGIC):
        return _decode_v2_bytes(data, aes_key=V1_AES_KEY, xor_key=0x88)
    if data.startswith(V2_MAGIC):
        if v2_aes_key is None or v2_xor_key is None:
            return DatDecodeResult(status='decode_failed', error_code='v2_key_unavailable')
        return _decode_v2_bytes(data, aes_key=v2_aes_key, xor_key=v2_xor_key)
    detected = detect_xor_key(data)
    if not detected:
        return DatDecodeResult(status='decode_failed', error_code='unknown_dat_wrapper')
    key, image_type = detected
    decoded = bytes(b ^ key for b in data)
    if sniff_image_type(decoded) != image_type:
        return DatDecodeResult(status='decode_failed', error_code='invalid_decoded_header')
    return DatDecodeResult(status='decoded', image_type=image_type, xor_key=key, output_bytes=decoded)


def _normalize_wxid(value: str) -> str:
    value = value.strip()
    if value.startswith('wxid_'):
        return 'wxid_' + value.removeprefix('wxid_').split('_', 1)[0]
    base, separator, suffix = value.rpartition('_')
    if separator and len(suffix) == 4 and all(char in '0123456789abcdefABCDEF' for char in suffix):
        return base
    return value


def _v2_key_candidates(path: Path) -> list[tuple[bytes, int]]:
    resolved = path.resolve(strict=False)
    account_root = next((parent for parent in resolved.parents if parent.parent.name == 'xwechat_files'), None)
    if account_root is None:
        return []
    documents = account_root.parent.parent
    kvcomm = documents / 'app_data' / 'net' / 'kvcomm'
    if not kvcomm.is_dir():
        return []
    account_names = list(dict.fromkeys((account_root.name, _normalize_wxid(account_root.name))))
    codes: set[int] = set()
    try:
        entries = list(kvcomm.iterdir())
    except OSError:
        return []
    for entry in entries:
        match = re.match(r'^key_(\d+)_', entry.name)
        if match:
            codes.add(int(match.group(1)))
    candidates: list[tuple[bytes, int]] = []
    for account_name in account_names:
        for code in sorted(codes):
            digest = hashlib.md5(f'{code}{account_name}'.encode()).hexdigest()[:16].encode()
            candidates.append((digest, code & 0xFF))
    return candidates


def decode_wechat_dat_bytes_for_path(data: bytes, path: Path) -> DatDecodeResult:
    if data.startswith(V2_MAGIC):
        for aes_key, xor_key in _v2_key_candidates(Path(path)):
            decoded = decode_wechat_dat_bytes(data, v2_aes_key=aes_key, v2_xor_key=xor_key)
            if decoded.output_bytes is not None:
                return decoded
        return DatDecodeResult(status='decode_failed', error_code='v2_key_unavailable')
    return decode_wechat_dat_bytes(data)


def decode_wechat_dat_file(path: Path) -> DatDecodeResult:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return DatDecodeResult(status='missing_local_cache', error_code='missing_file')
    return decode_wechat_dat_bytes_for_path(data, Path(path))
