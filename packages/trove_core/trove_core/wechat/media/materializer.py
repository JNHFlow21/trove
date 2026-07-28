from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
from io import BytesIO
import ipaddress
import os
from pathlib import Path
import socket
import ssl
import stat
import tempfile
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlsplit

from trove_core.approvals import ApprovalGrant, require_claimed_approval_grant
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig, path_is_under
from trove_core.wechat.media.locator import MediaLocatorResult, locate_media_asset
from trove_core.wechat.media.source_registry import resolve_snapshot_root


MAX_MEDIA_BYTES = 64 * 1024 * 1024
MAX_REDIRECTS = 3
REMOTE_TIMEOUT_SECONDS = 15
WECHAT_CDN_SUFFIXES = ('qpic.cn', 'qlogo.cn', 'wechat.com', 'weixin.qq.com', 'wx.qq.com')
WECHAT_CDN_EXACT_HOSTS = {'vweixinf.tc.qq.com', 'wxapp.tc.qq.com'}
_SAFE_ERROR_CODES = {
    'source_outside_bound_root', 'invalid_source_relative_path', 'source_not_regular_file',
    'media_size_limit_exceeded', 'empty_media', 'remote_host_resolves_non_global',
    'remote_host_unresolved', 'remote_locator_invalid', 'remote_host_not_allowlisted',
    'remote_redirect_missing_location', 'remote_redirect_limit_exceeded',
    'remote_content_type_invalid', 'media_content_type_invalid', 'materialized_hash_collision',
    'source_snapshot_unavailable', 'locator_routes_exhausted', 'local_video_cache_missing',
    'v2_key_unavailable', 'v2_invalid_header', 'v2_decrypt_failed', 'unknown_dat_wrapper',
}


@dataclass(frozen=True)
class MaterializationResult:
    ok: bool
    status: str
    asset_id: str
    path: Path | None = None
    path_ref: str | None = None
    content_sha256: str | None = None
    mime: str | None = None
    route: str | None = None
    reason: str | None = None
    approval_payload: dict[str, Any] | None = None
    bytes_written: int = 0

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'status': self.status,
            'asset_id': self.asset_id,
            'content_sha256': self.content_sha256,
            'mime': self.mime,
            'route': self.route,
            'reason': self.reason,
            'approval_payload': self.approval_payload,
            'bytes_written': self.bytes_written,
            'raw_paths_included': False,
            'remote_url_included': False,
        }


def remote_fetch_approval_payload(
    *,
    citation: str,
    asset_id: str,
    locator: MediaLocatorResult,
    max_bytes: int = MAX_MEDIA_BYTES,
) -> dict[str, Any]:
    if locator.status != 'remote' or not locator.locator_hash or not locator.snapshot_revision or not locator.remote_url:
        raise ValueError('remote locator is incomplete')
    host = (urlsplit(locator.remote_url).hostname or '').lower()
    return {
        'citation': str(citation),
        'asset_id': str(asset_id),
        'locator_hash': locator.locator_hash,
        'snapshot_revision': locator.snapshot_revision,
        'host_hash': hashlib.sha256(host.encode('utf-8')).hexdigest(),
        'max_bytes': int(max_bytes),
        'purpose': 'lazy_media_materialization',
    }


def _secure_open_under(root: Path, path: Path) -> tuple[int, os.stat_result]:
    root = root.resolve(strict=True)
    path = path.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise OSError('source_outside_bound_root') from exc
    if not relative.parts or any(part in {'', '.', '..'} for part in relative.parts):
        raise OSError('invalid_source_relative_path')
    flags_dir = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    directory_fd = os.open(root, flags_dir)
    try:
        for part in relative.parts[:-1]:
            next_fd = os.open(part, flags_dir, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(relative.parts[-1], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(file_fd)
            raise OSError('source_not_regular_file')
        return file_fd, info
    finally:
        os.close(directory_fd)


def _sniff(data: bytes, *, suffix: str, modality: str) -> tuple[str, str] | None:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png', '.png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg', '.jpg'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif', '.gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp', '.webp'
    if data.startswith(b'wxgf'):
        return 'image/heic', '.hevc'
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return 'audio/wav', '.wav'
    if data.startswith(b'#!AMR'):
        return 'audio/amr', '.amr'
    if data.startswith((b'#!SILK_V3', b'\x02#!SILK_V3')):
        return 'audio/silk', '.silk'
    if data.startswith(b'ID3'):
        return 'audio/mpeg', '.mp3'
    if len(data) >= 12 and data[4:8] == b'ftyp':
        return 'video/mp4', '.mp4'
    if data.startswith(b'%PDF-'):
        return 'application/pdf', '.pdf'
    if modality == 'image' and suffix.lower() in {'.dat', '.heic', '.heif'}:
        return 'application/octet-stream', suffix.lower()
    if modality in {'file', 'attachment', 'document'}:
        return 'application/octet-stream', suffix.lower() if suffix and len(suffix) <= 10 else '.bin'
    return None


def _copy_reader(reader: BinaryIO, output: BinaryIO, *, max_bytes: int) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    written = 0
    prefix = bytearray()
    while True:
        chunk = reader.read(min(1024 * 1024, max_bytes + 1 - written))
        if not chunk:
            break
        written += len(chunk)
        if written > max_bytes:
            raise OSError('media_size_limit_exceeded')
        if len(prefix) < 4096:
            prefix.extend(chunk[:4096 - len(prefix)])
        digest.update(chunk)
        output.write(chunk)
    output.flush()
    os.fsync(output.fileno())
    if written <= 0:
        raise OSError('empty_media')
    return digest.hexdigest(), written, bytes(prefix)


def _safe_error_code(exc: BaseException) -> str:
    text = str(exc)
    if text in _SAFE_ERROR_CODES:
        return text
    if text.startswith('remote_http_status_') and text.removeprefix('remote_http_status_').isdigit():
        return text
    return 'materialization_io_error'


def _host_allowed(host: str) -> bool:
    host = host.rstrip('.').lower()
    return host in WECHAT_CDN_EXACT_HOSTS or any(
        host == suffix or host.endswith('.' + suffix) for suffix in WECHAT_CDN_SUFFIXES
    )


def _global_addresses(host: str) -> list[str]:
    addresses: list[str] = []
    for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        address = str(info[4][0])
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not ip.is_global:
            raise OSError('remote_host_resolves_non_global')
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise OSError('remote_host_unresolved')
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, *, timeout: float):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def _fetch_remote(url: str, output: BinaryIO, *, max_bytes: int) -> tuple[str, int, bytes, str]:
    current = url
    for _hop in range(MAX_REDIRECTS + 1):
        parsed = urlsplit(current)
        host = (parsed.hostname or '').rstrip('.').lower()
        if parsed.scheme != 'https' or not host or parsed.username or parsed.password or parsed.fragment or parsed.port not in {None, 443}:
            raise OSError('remote_locator_invalid')
        if not _host_allowed(host):
            raise OSError('remote_host_not_allowlisted')
        address = _global_addresses(host)[0]
        connection = _PinnedHTTPSConnection(host, address, timeout=REMOTE_TIMEOUT_SECONDS)
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        try:
            connection.request('GET', path, headers={
                'Host': host,
                'Accept': 'image/*,video/*,audio/*,application/octet-stream',
                'User-Agent': 'TROVE-local-media/1',
                'Connection': 'close',
            })
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader('Location')
                response.read(1024)
                if not location:
                    raise OSError('remote_redirect_missing_location')
                current = urljoin(current, location)
                continue
            if response.status != 200:
                raise OSError(f'remote_http_status_{response.status}')
            content_length = response.getheader('Content-Length')
            if content_length and int(content_length) > max_bytes:
                raise OSError('media_size_limit_exceeded')
            digest, written, prefix = _copy_reader(response, output, max_bytes=max_bytes)
            return digest, written, prefix, str(response.getheader('Content-Type') or '').split(';', 1)[0].lower()
        finally:
            connection.close()
    raise OSError('remote_redirect_limit_exceeded')


def _publish_materialized(
    cfg: VaultConfig,
    store: SQLiteStore,
    *,
    asset_id: str,
    temp_path: Path,
    digest: str,
    written: int,
    prefix: bytes,
    suffix: str,
    modality: str,
    route: str,
    publish: bool = True,
) -> MaterializationResult:
    detected = _sniff(prefix, suffix=suffix, modality=modality)
    if detected is None:
        raise OSError('media_content_type_invalid')
    mime, extension = detected
    destination = cfg.root / 'media' / 'materialized' / digest[:2] / f'{digest}{extension}'
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        destination.parent.chmod(0o700)
    except OSError:
        pass
    if destination.exists():
        if destination.stat().st_size != written:
            raise OSError('materialized_hash_collision')
        temp_path.unlink(missing_ok=True)
    else:
        os.replace(temp_path, destination)
        destination.chmod(0o600)
    path_ref = str(destination.relative_to(cfg.root))
    result = MaterializationResult(
        True, 'materialized', asset_id, destination, path_ref, digest, mime, route,
        bytes_written=written,
    )
    if publish:
        publish_materialization_result(store, result)
    return result


def publish_materialization_result(
    store: SQLiteStore,
    result: MaterializationResult,
    *,
    expected_path_ref: str | None = None,
) -> bool:
    """CAS-publish a prepared content-addressed file into SQLite."""

    if not result.ok or not result.path_ref:
        return False
    with store.connect() as conn:
        params: list[Any] = [result.path_ref, result.content_sha256, result.asset_id]
        predicate = ''
        if expected_path_ref is not None:
            predicate = " AND COALESCE(path_ref,'')=?"
            params.append(str(expected_path_ref))
        cursor = conn.execute(
            f"""UPDATE media_assets
                   SET path_ref=?,content_hash=COALESCE(?,content_hash),cache_state='cached',
                       processing_state='pending',updated_at=datetime('now')
                 WHERE asset_id=?{predicate}""",
            params,
        )
        if max(0, cursor.rowcount) != 1:
            conn.rollback()
            return False
        conn.execute(
            """UPDATE media_source_bindings
                  SET locator_state='materialized',updated_at=datetime('now')
                WHERE asset_id=?""",
            (result.asset_id,),
        )
        conn.commit()
    return True


def materialize_media_asset(
    cfg: VaultConfig,
    store: SQLiteStore,
    asset: Any,
    *,
    citation: str,
    allow_remote: bool = False,
    approval_grant: ApprovalGrant | None = None,
    approval_payload: dict[str, Any] | None = None,
    max_bytes: int = MAX_MEDIA_BYTES,
    publish: bool = True,
) -> MaterializationResult:
    asset_id = str(asset['asset_id'])
    modality = str(asset['modality'] or '')
    locator = locate_media_asset(cfg, store, asset)
    if locator.status == 'unavailable':
        return MaterializationResult(False, 'unavailable', asset_id, reason=locator.reason, route=locator.route)
    if locator.route == 'vault_cache' and locator.path is not None:
        path_ref = str(locator.path.relative_to(cfg.root))
        return MaterializationResult(
            True, 'cached', asset_id, locator.path, path_ref,
            str(asset['content_hash'] or '') or None, route='vault_cache',
            bytes_written=int(locator.path.stat().st_size),
        )

    if locator.status == 'remote':
        expected = remote_fetch_approval_payload(
            citation=citation,
            asset_id=asset_id,
            locator=locator,
            max_bytes=max_bytes,
        )
        if not allow_remote or approval_grant is None:
            return MaterializationResult(
                False, 'awaiting_approval', asset_id, route=locator.route,
                reason='remote_fetch_approval_required', approval_payload=expected,
            )
        if approval_payload != expected:
            return MaterializationResult(
                False, 'awaiting_approval', asset_id, route=locator.route,
                reason='remote_fetch_approval_mismatch', approval_payload=expected,
            )
        require_claimed_approval_grant(
            approval_grant,
            cfg.root,
            action='wechat_cdn_fetch',
            danger_class='remote_media_fetch',
            payload=expected,
        )

    temp_dir = cfg.root / 'media' / 'tmp'
    temp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        temp_dir.chmod(0o700)
    except OSError:
        pass
    fd, temp_name = tempfile.mkstemp(prefix='.materialize-', dir=temp_dir)
    os.fchmod(fd, 0o600)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, 'w+b') as output:
            if locator.status == 'remote' and locator.remote_url:
                digest, written, prefix, response_type = _fetch_remote(locator.remote_url, output, max_bytes=max_bytes)
                if response_type and not (
                    response_type.startswith(('image/', 'video/', 'audio/'))
                    or response_type == 'application/octet-stream'
                ):
                    raise OSError('remote_content_type_invalid')
                suffix = Path(urlsplit(locator.remote_url).path).suffix
            elif locator.path is not None and locator.snapshot_revision:
                source_root = locator.source_root
                if source_root is None:
                    source_root, _ = resolve_snapshot_root(cfg, store, locator.snapshot_revision)
                if source_root is None:
                    raise OSError('source_snapshot_unavailable')
                source_fd, source_stat = _secure_open_under(source_root, locator.path)
                try:
                    if source_stat.st_size > max_bytes:
                        raise OSError('media_size_limit_exceeded')
                    with os.fdopen(source_fd, 'rb') as source:
                        digest, written, prefix = _copy_reader(source, output, max_bytes=max_bytes)
                finally:
                    # fdopen owns the descriptor after construction.
                    pass
                suffix = locator.path.suffix
                if modality == 'image' and suffix.lower() == '.dat':
                    from trove_core.wechat.media.dat_decoder import decode_wechat_dat_bytes_for_path

                    output.seek(0)
                    decoded = decode_wechat_dat_bytes_for_path(output.read(), locator.path)
                    if decoded.output_bytes is None:
                        raise OSError(decoded.error_code or 'unknown_dat_wrapper')
                    output.seek(0)
                    output.truncate(0)
                    digest, written, prefix = _copy_reader(
                        BytesIO(decoded.output_bytes), output, max_bytes=max_bytes,
                    )
                    suffix = f'.{decoded.image_type or "img"}'
            elif locator.embedded_bytes is not None and locator.snapshot_revision:
                digest, written, prefix = _copy_reader(
                    BytesIO(locator.embedded_bytes), output, max_bytes=max_bytes,
                )
                suffix = '.silk'
            else:
                raise OSError('locator_routes_exhausted')
        return _publish_materialized(
            cfg,
            store,
            asset_id=asset_id,
            temp_path=temp_path,
            digest=digest,
            written=written,
            prefix=prefix,
            suffix=suffix,
            modality=modality,
            route=str(locator.route or 'unknown'),
            publish=publish,
        )
    except (OSError, ValueError) as exc:
        temp_path.unlink(missing_ok=True)
        return MaterializationResult(False, 'unavailable', asset_id, route=locator.route, reason=_safe_error_code(exc))
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
