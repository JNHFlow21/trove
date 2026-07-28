#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_VERSION = '1.0.0'
PROVIDER_VERSION = '1.0.0'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(env: dict[str, str]) -> tuple[str, bool]:
    configured = env.get('TROVE_RELEASE_GIT_SHA')
    if configured is not None:
        if not re.fullmatch(r'[0-9a-f]{40}', configured):
            raise ValueError('TROVE_RELEASE_GIT_SHA must be a full commit SHA')
        current = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=False,
            capture_output=True, text=True,
        )
        if current.returncode == 0:
            if current.stdout.strip() != configured:
                raise ValueError('TROVE_RELEASE_GIT_SHA does not match the checkout')
            dirty = bool(subprocess.run(
                ['git', 'status', '--porcelain'], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip())
            return configured, dirty
        return configured, False
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ['git', 'status', '--porcelain'], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return sha, dirty


def _build_wheel(project: Path, output: Path) -> Path:
    before = set(output.glob('*.whl'))
    subprocess.run([
        sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
        '--wheel-dir', str(output), str(project),
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    created = set(output.glob('*.whl')) - before
    if len(created) != 1:
        raise RuntimeError('wheel build did not produce exactly one artifact')
    return created.pop()


def _copy_build_sources(staging: Path) -> tuple[Path, Path]:
    runtime = staging / 'runtime-source'
    provider = staging / 'provider-source'
    runtime.mkdir()
    provider.mkdir()
    for name in ('pyproject.toml', 'README.md'):
        shutil.copyfile(ROOT / name, runtime / name)
    packages = runtime / 'packages'
    packages.mkdir()
    for name in (
        'trove_protocol', 'trove_core', 'trove_client', 'trove_daemon',
        'trove_cli', 'trove_mcp',
    ):
        shutil.copytree(ROOT / 'packages' / name, packages / name, ignore=shutil.ignore_patterns(
            '__pycache__', '*.pyc', 'build', '*.egg-info',
        ))
    shutil.copyfile(ROOT / 'packages/trove_provider_wechat/pyproject.toml', provider / 'pyproject.toml')
    shutil.copytree(
        ROOT / 'packages/trove_provider_wechat/trove_provider_wechat',
        provider / 'trove_provider_wechat',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
    )
    return runtime, provider


def build_distribution(output: Path, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    environment = dict(os.environ if env is None else env)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('distribution output must be absent or empty')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    source_sha, source_dirty = _source_sha(environment)
    with tempfile.TemporaryDirectory(prefix='trove-wheel-build-') as directory:
        staging = Path(directory)
        runtime_source, provider_source = _copy_build_sources(staging)
        runtime_wheel = _build_wheel(runtime_source, staging)
        provider_wheel = _build_wheel(provider_source, staging)
        expected = {
            'trove_runtime': (runtime_wheel, RUNTIME_VERSION),
            'trove_provider_wechat': (provider_wheel, PROVIDER_VERSION),
        }
        for prefix, (wheel, version) in expected.items():
            normalized = version.replace('.', '_')
            if not wheel.name.startswith(f'{prefix}-{version}') and not wheel.name.startswith(f'{prefix}-{normalized}'):
                raise RuntimeError('wheel name/version mismatch')
        copied = []
        for wheel in (runtime_wheel, provider_wheel):
            destination = output / wheel.name
            shutil.copyfile(wheel, destination)
            os.chmod(destination, 0o600)
            copied.append(destination)

    for path in (ROOT / 'packages/trove_protocol', ROOT / 'packages/trove_daemon'):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    from trove_daemon.lifecycle import build_identity, catalog_identity

    provider_manifest = json.loads((
        ROOT / 'packages/trove_provider_wechat/trove_provider_wechat/manifest.json'
    ).read_text(encoding='utf-8'))
    runtime, provider = copied
    core = {
        'source_git_sha': source_sha,
        'source_dirty': source_dirty,
        'protocol': 'trove/1',
        'runtime_build_hash': build_identity(),
        'catalog_hash': catalog_identity(),
        'provider_package_hash': provider_manifest['package_sha256'],
        'runtime': {
            'file': runtime.name, 'sha256': _sha256(runtime), 'version': RUNTIME_VERSION,
        },
        'provider': {
            'file': provider.name, 'sha256': _sha256(provider),
            'version': PROVIDER_VERSION, 'provider_id': provider_manifest['provider_id'],
        },
    }
    set_hash = hashlib.sha256(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    manifest = {
        'schema_version': 1,
        'artifact_type': 'trove_distribution_set',
        **core,
        'distribution_set_sha256': set_hash,
        'privacy': {
            'private_paths_included': False,
            'secret_values_included': False,
            'vault_content_included': False,
        },
    }
    manifest_path = output / 'distribution-manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )
    os.chmod(manifest_path, 0o600)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_distribution(args.out.expanduser())
    except Exception:
        print(json.dumps({'ok': False, 'error_code': 'distribution_build_failed'}, sort_keys=True))
        return 2
    print(json.dumps({
        'ok': True,
        'manifest_file': 'distribution-manifest.json',
        'distribution_set_sha256': manifest['distribution_set_sha256'],
        'source_dirty': manifest['source_dirty'],
    }, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
