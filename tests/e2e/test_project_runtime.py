from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ProjectRuntimeTests(unittest.TestCase):
    def test_trove_python_wrapper_uses_repo_venv(self):
        wrapper = ROOT / 'scripts' / 'trove-python'
        expected = ROOT / '.venv' / 'bin' / 'python'
        if not expected.exists():
            self.fail('project .venv is required; run bash scripts/bootstrap_runtime.sh')
        proc = subprocess.run(
            [str(wrapper), '-c', 'import json, os, sys; print(json.dumps({"executable": sys.executable, "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"), "pythonpath": os.environ.get("PYTHONPATH", "")}))'],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(Path(payload['executable']), expected)
        self.assertEqual(payload['python_no_user_site'], '1')
        paths = payload['pythonpath'].split(os.pathsep)
        self.assertEqual(paths[0], str(ROOT / 'packages' / 'trove_protocol'))
        self.assertNotIn(str(ROOT / 'packages' / 'trove_api'), paths)

    def test_direct_script_trampolines_into_project_runtime(self):
        proc = subprocess.run(
            ['python3', 'scripts/runtime_doctor.py', '--json'],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload['uses_project_python'])
        self.assertEqual(Path(payload['python_executable']), ROOT / '.venv' / 'bin' / 'python')

    def test_git_hooks_use_project_runtime_wrapper(self):
        for name in ['pre-commit', 'pre-push']:
            hook = (ROOT / '.githooks' / name).read_text(encoding='utf-8')
            self.assertIn('./scripts/trove-python', hook)
            self.assertNotIn('python3 scripts/', hook)

    def test_root_check_runner_uses_project_runtime_wrapper(self):
        source = (ROOT / 'scripts' / 'check.py').read_text(encoding='utf-8')
        self.assertIn("PYTHON = ROOT / 'scripts' / 'trove-python'", source)
        self.assertNotIn('npm', source)
        proc = subprocess.run(
            [str(ROOT / 'scripts' / 'trove-python'), 'scripts/check.py', 'contract', '--list'],
            cwd=ROOT, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('public-surface', proc.stdout)


if __name__ == '__main__':
    unittest.main()
