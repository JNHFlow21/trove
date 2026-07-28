from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'run_lazy_profile_enrichment_acceptance.py'


def _load_script():
    spec = importlib.util.spec_from_file_location('lazy_profile_acceptance_contract', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LazyProfileAcceptanceContractTests(unittest.TestCase):
    def test_private_selector_is_fd_only_and_no_customer_argument_exists(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), '--help'],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('--input-fd', result.stdout)
        self.assertNotIn('--customer', result.stdout)

    def test_private_input_is_bounded_exact_and_closes_inherited_fd(self):
        module = _load_script()
        read_fd, write_fd = os.pipe()
        os.write(write_fd, json.dumps({
            'customer': 'Synthetic Customer',
            'actor': 'operator',
            'session': 'contract',
            'allow_cloud_asr': False,
        }).encode('utf-8'))
        os.close(write_fd)
        value = module._read_private_input(read_fd)
        self.assertEqual(value['customer'], 'Synthetic Customer')
        with self.assertRaises(OSError):
            os.fstat(read_fd)
        with self.assertRaises(ValueError):
            module._read_private_input(0)

    def test_proof_writer_is_0600_and_rejects_private_markers(self):
        module = _load_script()
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / module.REPORT_RELATIVE_PATH
            module._safe_write(target, {
                'schema': 'lazy-profile-enrichment-acceptance/v1',
                'ok': True,
                'counts': {'completed': 1},
            })
            self.assertEqual(os.stat(target).st_mode & 0o077, 0)
            self.assertEqual(json.loads(target.read_text(encoding='ascii'))['ok'], True)
            for value in (
                {'citation': 'trove://fixture'},
                {'url': 'https://invalid.example'},
                {'path': '/Volumes/private/source'},
                {'raw_text': 'fixture body'},
            ):
                with self.assertRaises(ValueError):
                    module._safe_write(target, value)

    def test_purge_audit_can_be_required_by_release_acceptance(self):
        source = SCRIPT.read_text(encoding='utf-8')
        self.assertIn("parser.add_argument('--require-purge-audit'", source)
        self.assertIn("report['lifecycle']['purge_verified']", source)


if __name__ == '__main__':
    unittest.main()
