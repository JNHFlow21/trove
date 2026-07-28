from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.privacy_scan import scan


class PrivacyScanCloudRuntimeTests(unittest.TestCase):
    def test_blocks_runtime_media_transcripts_and_provider_payloads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'safe.md').write_text('# ok\n', encoding='utf-8')
            (root / 'proof' / 'raw').mkdir(parents=True)
            (root / 'proof' / 'raw' / 'provider_payload.json').write_text('{}', encoding='utf-8')
            (root / 'transcripts').mkdir()
            (root / 'transcripts' / 'voice_1.txt').write_text('redacted test', encoding='utf-8')
            (root / 'media_understanding.jsonl').write_text('{}\n', encoding='utf-8')
            (root / 'media').mkdir()
            (root / 'media' / 'photo.jpg').write_bytes(b'not real image')
            findings = scan(root)
        joined = '\n'.join(findings)
        self.assertIn('provider payload path', joined)
        self.assertIn('runtime cloud/transcript/provider payload path', joined)
        self.assertIn('media_understanding.jsonl: denied runtime cloud/transcript/provider payload path', joined)
        self.assertIn('raw media file is forbidden', joined)

    def test_allows_source_docs_that_describe_redacted_cloud_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'docs').mkdir()
            (root / 'docs' / 'testing.md').write_text('provider status reports secret names only\n', encoding='utf-8')
            self.assertEqual(scan(root), [])

    def test_blocks_runtime_derived_data_cache_and_acceptance_proof_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            candidates = (
                root / 'media' / 'previews' / 'preview.txt',
                root / 'media' / 'materialized' / 'asset.bin',
                root / 'approvals' / 'approval.json',
                root / 'proof' / 'lazy-profile-enrichment' / 'acceptance-report.redacted.json',
                root / 'profile_snapshots' / 'snapshot.json',
                root / 'profile_enrichment' / 'manifest.json',
            )
            for path in candidates:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n', encoding='utf-8')
            findings = scan(root)
        joined = '\n'.join(findings)
        for path in candidates:
            self.assertIn(path.relative_to(root).as_posix(), joined)
        self.assertGreaterEqual(joined.count('denied runtime derived-data/cache/proof path'), len(candidates))

    def test_blocks_real_wechat_citations_in_docs_but_allows_fixture_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'docs' / 'perf').mkdir(parents=True)
            (root / 'docs' / 'perf' / 'fixture.json').write_text(
                '"trove://wechat/acct-work/conv-sales-review/message_0/10"\n',
                encoding='utf-8',
            )
            self.assertEqual(scan(root), [])
            (root / 'docs' / 'perf' / 'leak.json').write_text(
                '"trove://wechat/acct-0123456789ab/conv-fedcba987654/message_1/11221"\n',
                encoding='utf-8',
            )
            findings = scan(root)
        self.assertIn('raw real WeChat citation detected in docs', '\n'.join(findings))


if __name__ == '__main__':
    unittest.main()
