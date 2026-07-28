from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from scripts.dependency_manifest import build_manifest
from scripts.migrate_from_kos import migrate
from scripts.preflight_workspace import inspect_target
from scripts.privacy_scan import scan


class MigrationSafetyTests(unittest.TestCase):
    def test_preflight_rejects_knowledge_os_target(self):
        report = inspect_target(Path('/tmp/Knowledge_OS/wechat/trove'), owner='test')
        self.assertFalse(report['ok'])
        self.assertTrue(report['errors'])

    def test_migration_allowlist_excludes_sensitive_and_live_sender_files(self):
        with tempfile.TemporaryDirectory() as s, tempfile.TemporaryDirectory() as t:
            source = Path(s)
            (source / 'PROJECT.md').write_text('safe', encoding='utf-8')
            (source / 'wechat-chat-analysis/scripts').mkdir(parents=True)
            (source / 'wechat-chat-analysis/scripts/vault_cli.py').write_text('print("ok")', encoding='utf-8')
            (source / 'wechat-chat-analysis/scripts/wechat_auto_reply.py').write_text('do not copy', encoding='utf-8')
            (source / 'wechat-chat-analysis/decrypted').mkdir(parents=True)
            (source / 'wechat-chat-analysis/decrypted/key_store.json').write_text('{}', encoding='utf-8')
            (source / 'wechat-chat-analysis/output').mkdir(parents=True)
            (source / 'wechat-chat-analysis/output/messages.jsonl').write_text('{}\n', encoding='utf-8')
            report = migrate(source, Path(t))
            copied = set(report['copied'])
            self.assertIn('PROJECT.md', copied)
            self.assertIn('wechat-chat-analysis/scripts/vault_cli.py', copied)
            copied_text = '\n'.join(copied)
            self.assertNotIn('wechat_auto_reply.py', copied_text)
            self.assertNotIn('key_store.json', copied_text)
            self.assertNotIn('messages.jsonl', copied_text)
            self.assertEqual(report['deleted_from_source'], [])

    def test_dependency_manifest_reports_missing_local_import(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'a.py').write_text('import b\n', encoding='utf-8')
            (root / 'b.py').write_text('VALUE=1\n', encoding='utf-8')
            manifest = build_manifest(root, [Path('a.py')])
            self.assertFalse(manifest['ok'])
            self.assertEqual(manifest['missing_local_imports'][0]['required_path'], 'b.py')

    def test_privacy_scan_detects_sensitive_content_and_allows_portable_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'README.md').write_text('$HOME/Trove/trove is portable\n', encoding='utf-8')
            self.assertEqual(scan(root), [])
            (root / 'leak.txt').write_text('/Users/' + 'alice/private/chat ' + '0123456789abcdef' * 3 + '\n', encoding='utf-8')
            findings = scan(root)
            self.assertTrue(any('private absolute path' in f for f in findings))
            self.assertTrue(any('long hex' in f for f in findings))
            (root / 'chat.db').write_bytes(b'SQLite format 3\x00' + b'0' * 100)
            self.assertTrue(any('SQLite header' in f or 'denied sensitive filename' in f for f in scan(root)))

    def test_privacy_scan_rejects_contact_details_and_realistic_account_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = (
                'phone=' + '139' + '12345678' + '\n'
                + 'email=' + 'person' + '@private-domain.invalid' + '\n'
                + 'account=' + 'wxid_' + 'a1b2c3d4e5f6g7' + '\n'
            )
            (root / 'contact.txt').write_text(payload, encoding='utf-8')
            findings = scan(root)
            self.assertTrue(any('phone-like value' in item for item in findings))
            self.assertTrue(any('non-example email address' in item for item in findings))
            self.assertTrue(any('non-synthetic WeChat identifier' in item for item in findings))

    def test_privacy_scan_allows_only_hash_labeled_full_digests_in_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / 'tests' / 'fixtures' / 'synthetic' / 'evidence'
            bundle.mkdir(parents=True)
            digest = 'a1' * 32
            evidence = bundle / 'evidence.json'
            evidence.write_text(json.dumps({'artifact_sha256': digest}), encoding='utf-8')
            self.assertEqual(scan(root), [])

            evidence.write_text(json.dumps({'query': digest}), encoding='utf-8')
            self.assertTrue(any('long hex' in finding for finding in scan(root)))

    def test_v1_release_manifest_allows_hash_fields_but_not_content_fields(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            release = root / 'tests' / 'fixtures' / 'synthetic' / 'evidence' / 'release-manifest.json'
            release.parent.mkdir(parents=True)
            digest = 'ab' * 32
            release.write_text(json.dumps({
                'source_git_sha': 'a' * 40,
                'distribution_set_sha256': digest,
                'evidence_sha256': {'test_summary_sha256': digest},
            }), encoding='utf-8')
            self.assertEqual(scan(root), [])

            release.write_text(json.dumps({
                'distribution_set_sha256': digest,
                'content': digest,
            }), encoding='utf-8')
            self.assertTrue(any('long hex' in finding for finding in scan(root)))


if __name__ == '__main__':
    unittest.main()
