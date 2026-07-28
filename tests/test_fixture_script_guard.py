from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class FixtureScriptGuardTests(unittest.TestCase):
    def test_valid_fixture_reset_can_write_synthetic_jsonl(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'fixture'
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / 'scripts' / 'generate_fixture_vault.py'),
                    '--vault',
                    str(vault),
                    '--reset',
                    '--jsonl',
                ],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result)
            self.assertTrue((vault / '.trove-fixture-vault.json').is_file())
            lines = (vault / 'fixtures' / 'synthetic' / 'messages.jsonl').read_text(encoding='utf-8').splitlines()
            self.assertGreaterEqual(len(lines), 12)

    def test_reset_refuses_unmarked_nonempty_root_without_recursive_delete(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "real-vault"
            vault.mkdir()
            sentinel = vault / "real-data.bin"
            sentinel.write_bytes(b"keep-me")

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "generate_fixture_vault.py"),
                    "--vault",
                    str(vault),
                    "--reset",
                    "--jsonl",
                ],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result)
            payload = json.loads(result.stderr)
            self.assertEqual(payload["error"]["code"], "fixture_vault_guard_rejected")
            self.assertEqual(payload["error"]["reason_code"], "fixture_marker_missing_nonempty_root")
            self.assertNotIn(str(vault), result.stdout + result.stderr)
            self.assertEqual(sentinel.read_bytes(), b"keep-me")
            self.assertEqual([path.name for path in vault.iterdir()], ["real-data.bin"])


if __name__ == "__main__":
    unittest.main()
