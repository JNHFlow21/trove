from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapRuntimeExtrasTests(unittest.TestCase):
    def resolve_extras(self, system_name: str, *, explicit: str | None = None) -> str:
        command = (
            'TROVE_BOOTSTRAP_SOURCE_ONLY=1; '
            'source scripts/bootstrap_runtime.sh; '
            f'resolve_install_extras {system_name}'
        )
        env = os.environ.copy()
        if explicit is None:
            env.pop('TROVE_RUNTIME_INSTALL_EXTRAS', None)
        else:
            env['TROVE_RUNTIME_INSTALL_EXTRAS'] = explicit
        proc = subprocess.run(
            ['bash', '-c', command],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_non_darwin_default_extras_exclude_macos_local_vision(self):
        self.assertEqual(self.resolve_extras('Linux'), '')

    def test_darwin_default_extras_are_product_runtime_only(self):
        self.assertEqual(self.resolve_extras('Darwin'), 'local-vision,local-embedding,zvec')

    def test_explicit_runtime_extras_are_respected(self):
        self.assertEqual(self.resolve_extras('Linux', explicit='custom-extra'), 'custom-extra')


if __name__ == '__main__':
    unittest.main()
