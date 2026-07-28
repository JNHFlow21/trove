from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
_DISTRIBUTION: Path | None = None


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def distribution_dir() -> Path:
    global _DISTRIBUTION
    if _DISTRIBUTION is None:
        root = Path(tempfile.mkdtemp(prefix='trove-test-distribution-'))
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=False,
            capture_output=True, text=True,
        )
        configured_sha = os.environ.get('TROVE_RELEASE_GIT_SHA')
        if revision.returncode == 0:
            sha = revision.stdout.strip()
            source = Path(tempfile.mkdtemp(prefix='trove-test-source-'))
            atexit.register(shutil.rmtree, source, True)
            subprocess.run(
                ['git', 'checkout-index', '--all', f'--prefix={source}/'],
                cwd=ROOT, check=True,
            )
        elif configured_sha:
            sha = configured_sha
            source = ROOT
        else:
            raise RuntimeError('distribution tests require Git or TROVE_RELEASE_GIT_SHA')
        package_paths = [
            source / 'packages' / name
            for name in (
                'trove_protocol', 'trove_core', 'trove_client', 'trove_daemon',
                'trove_cli', 'trove_mcp',
            )
        ]
        env = clean_env() | {
            'TROVE_RELEASE_GIT_SHA': sha,
            'PYTHONPATH': os.pathsep.join(map(str, package_paths)),
        }
        subprocess.run([
            str(ROOT / '.venv/bin/python'), str(source / 'scripts/build_distribution.py'),
            '--out', str(root),
        ], cwd=source, env=env, check=True, stdout=subprocess.DEVNULL)
        _DISTRIBUTION = root
        atexit.register(shutil.rmtree, root, True)
    return _DISTRIBUTION


def create_venv(*, provider: bool) -> Path:
    root = Path(tempfile.mkdtemp(prefix='trove-test-install-'))
    atexit.register(shutil.rmtree, root, True)
    subprocess.run([sys.executable, '-m', 'venv', str(root)], check=True, env=clean_env())
    distribution = distribution_dir()
    wheels = sorted(distribution.glob('trove_runtime-*.whl'))
    if provider:
        wheels.extend(sorted(distribution.glob('trove_provider_wechat-*.whl')))
    subprocess.run([
        str(root / 'bin/python'), '-m', 'pip', 'install',
        '--find-links', str(distribution), *map(str, wheels),
    ], check=True, env=clean_env(), stdout=subprocess.DEVNULL)
    return root


__all__ = ['ROOT', 'clean_env', 'create_venv', 'distribution_dir']
