#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping
import zipfile


MANIFEST_KEYS = frozenset({
    'schema_version', 'artifact_type', 'source_git_sha', 'source_dirty', 'protocol',
    'runtime_build_hash', 'catalog_hash', 'provider_package_hash', 'runtime',
    'provider', 'distribution_set_sha256', 'privacy',
})


class DistributionVerificationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile) -> list[str]:
    names: list[str] = []
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        mode = item.external_attr >> 16
        if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(mode):
            raise DistributionVerificationError('unsafe wheel member')
        names.append(item.filename)
    return names


def _entry_points(archive: zipfile.ZipFile, names: list[str]) -> configparser.ConfigParser:
    matches = [name for name in names if name.endswith('.dist-info/entry_points.txt')]
    parser = configparser.ConfigParser()
    if matches:
        if len(matches) != 1:
            raise DistributionVerificationError('entry point metadata is ambiguous')
        parser.read_string(archive.read(matches[0]).decode('utf-8'))
    return parser


def _metadata(archive: zipfile.ZipFile, names: list[str]):
    matches = [name for name in names if name.endswith('.dist-info/METADATA')]
    if len(matches) != 1:
        raise DistributionVerificationError('wheel metadata is missing')
    return Parser().parsestr(archive.read(matches[0]).decode('utf-8'))


def _verify_runtime(path: Path, expected: Mapping[str, Any]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = _safe_members(archive)
        if any(
            name.startswith(('trove_provider_wechat/', 'trove_api/', 'apps/'))
            or name.endswith(('.js', '.mjs', '.cjs', '.ts', '.tsx'))
            for name in names
        ):
            raise DistributionVerificationError('runtime wheel contains a forbidden surface')
        required = {
            'trove_protocol/', 'trove_core/', 'trove_client/', 'trove_daemon/',
            'trove_cli/', 'trove_mcp/',
        }
        if not all(any(name.startswith(prefix) for name in names) for prefix in required):
            raise DistributionVerificationError('runtime wheel is incomplete')
        entry_points = _entry_points(archive, names)
        if not entry_points.has_section('console_scripts') or dict(entry_points['console_scripts']) != {
            'trove': 'trove_cli.main:main',
            'trove-mcp': 'trove_mcp.server:main',
            'troved': 'trove_daemon.main:main',
        }:
            raise DistributionVerificationError('runtime public executables are invalid')
        metadata = _metadata(archive, names)
        if metadata['Name'] != 'trove-runtime' or metadata['Version'] != expected['version']:
            raise DistributionVerificationError('runtime name/version mismatch')


def _verify_provider(path: Path, expected: Mapping[str, Any], package_hash: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = _safe_members(archive)
        package_members = [name for name in names if not '.dist-info/' in name]
        if not package_members or any(not name.startswith('trove_provider_wechat/') for name in package_members):
            raise DistributionVerificationError('provider wheel contains unrelated runtime code')
        entry_points = _entry_points(archive, names)
        if not entry_points.has_section('trove.providers') or dict(entry_points['trove.providers']) != {
            'wechat-source': 'trove_provider_wechat:create_provider',
        }:
            raise DistributionVerificationError('provider distribution entry point is invalid')
        metadata = _metadata(archive, names)
        requirements = metadata.get_all('Requires-Dist') or []
        if (
            metadata['Name'] != 'trove-provider-wechat'
            or metadata['Version'] != expected['version']
            or not any(re.fullmatch(r'trove-runtime\s*(?:\(==1\.0\.0\)|==1\.0\.0)', item) for item in requirements)
        ):
            raise DistributionVerificationError('provider dependency boundary is invalid')
        manifest = json.loads(archive.read('trove_provider_wechat/manifest.json'))
        seal = archive.read('trove_provider_wechat/package.seal')
        if (
            manifest.get('provider_id') != expected['provider_id']
            or manifest.get('package_sha256') != package_hash
            or hashlib.sha256(seal).hexdigest() != package_hash
        ):
            raise DistributionVerificationError('provider manifest/seal mismatch')


def verify_distribution(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_symlink():
        raise DistributionVerificationError('distribution manifest is unsafe')
    listed_manifest = manifest_path.stat()
    if listed_manifest.st_uid != os.getuid() or listed_manifest.st_mode & 0o077:
        raise DistributionVerificationError('distribution manifest permissions are unsafe')
    root = manifest_path.resolve(strict=True).parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise DistributionVerificationError('distribution manifest schema is invalid')
    if (
        manifest.get('schema_version') != 1
        or manifest.get('artifact_type') != 'trove_distribution_set'
        or manifest.get('protocol') != 'trove/1'
        or not re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('source_git_sha') or ''))
        or type(manifest.get('source_dirty')) is not bool
    ):
        raise DistributionVerificationError('distribution manifest identity is invalid')
    for field in ('runtime_build_hash', 'catalog_hash', 'provider_package_hash', 'distribution_set_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', str(manifest.get(field) or '')):
            raise DistributionVerificationError('distribution hash is invalid')
    if manifest.get('privacy') != {
        'private_paths_included': False,
        'secret_values_included': False,
        'vault_content_included': False,
    }:
        raise DistributionVerificationError('distribution privacy contract is invalid')
    core = {key: manifest[key] for key in MANIFEST_KEYS - {'schema_version', 'artifact_type', 'distribution_set_sha256', 'privacy'}}
    expected_set_hash = hashlib.sha256(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    if expected_set_hash != manifest['distribution_set_sha256']:
        raise DistributionVerificationError('distribution set hash mismatch')
    artifacts: dict[str, Path] = {}
    for kind in ('runtime', 'provider'):
        item = manifest.get(kind)
        if not isinstance(item, dict) or set(item) != (
            {'file', 'sha256', 'version'} if kind == 'runtime'
            else {'file', 'sha256', 'version', 'provider_id'}
        ):
            raise DistributionVerificationError('artifact metadata is invalid')
        name = item.get('file')
        if not isinstance(name, str) or Path(name).name != name or not name.endswith('.whl'):
            raise DistributionVerificationError('artifact filename is unsafe')
        artifact = root / name
        listed = artifact.stat() if artifact.exists() and not artifact.is_symlink() else None
        if (
            listed is None or not artifact.is_file()
            or listed.st_uid != os.getuid() or listed.st_mode & 0o077
            or _sha256(artifact) != item.get('sha256')
        ):
            raise DistributionVerificationError('artifact hash mismatch')
        artifacts[kind] = artifact
    _verify_runtime(artifacts['runtime'], manifest['runtime'])
    _verify_provider(artifacts['provider'], manifest['provider'], manifest['provider_package_hash'])
    return {
        'ok': True,
        'distribution_set_sha256': manifest['distribution_set_sha256'],
        'runtime_sha256': manifest['runtime']['sha256'],
        'provider_sha256': manifest['provider']['sha256'],
        'artifact_count': 2,
        'private_paths_included': False,
        'secret_values_included': False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_distribution(args.manifest.expanduser())
    except Exception:
        report = {
            'ok': False, 'error_code': 'distribution_verification_failed',
            'private_paths_included': False, 'secret_values_included': False,
        }
    print(json.dumps(report, sort_keys=True, separators=(',', ':')))
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
