from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MultimodalFixtureFlowTests(unittest.TestCase):
    def test_real_acceptance_script_writes_recoverable_redacted_blocker_report(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            out = vault / 'proof' / 'multimodal' / 'acceptance.redacted.json'
            proc = subprocess.run([
                sys.executable,
                'scripts/run_multimodal_real_acceptance.py',
                '--vault', str(vault),
                '--out', str(out),
                '--selected-account-id', 'acct-a',
                '--discovered-account-id', 'acct-a',
                '--undecryptable-account-id', 'acct-gap',
                '--coverage-gap-account-id', 'acct-gap',
                '--small-batch',
            ], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(payload['status'], 'blocked')
            self.assertFalse(payload['cloud_upload_started'])
            self.assertFalse(payload['privacy']['provider_payloads_included'])
            text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn('/Users/', text)
            self.assertNotIn('Bearer ', text)


if __name__ == '__main__':
    unittest.main()
