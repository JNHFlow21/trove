from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_cli.main import main


class MultimodalCliContractTests(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, json.loads(buf.getvalue())

    def test_legacy_batch_voice_commands_fail_before_creating_approval(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'

            code, data = self.run_cli([
                '--vault', str(vault), 'media', 'transcribe',
                '--budget', '5', '--yes', '--json',
            ])
            self.assertEqual(code, 2)
            self.assertEqual(data['error']['code'], 'cloud_asr_per_citation_required')
            self.assertFalse((vault / 'approvals').exists())

            code, data = self.run_cli([
                '--vault', str(vault), 'voice-transcribe',
                '--conversation-id', 'conv-a', '--yes', '--json',
            ])
            self.assertEqual(code, 2)
            self.assertEqual(data['error']['code'], 'cloud_asr_per_citation_required')
            self.assertFalse((vault / 'approvals').exists())

    def test_media_inventory_import_contacts_profile_and_fixture_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'source'
            source.mkdir()
            db = source / 'message_resource.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE resource(local_id INTEGER, local_type TEXT, path TEXT)')
            conn.execute('INSERT INTO resource VALUES(?,?,?)', (1, '3', 'missing.jpg'))
            conn.commit(); conn.close()
            code, data = self.run_cli(['media-inventory', str(source), '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(data['counts']['image'], 1)
            self.assertFalse(data['raw_paths_included'])

            contact_db = root / 'contact.db'
            conn = sqlite3.connect(contact_db)
            conn.execute('CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, signature TEXT, big_head_url TEXT)')
            conn.execute('INSERT INTO contact VALUES(?,?,?,?,?,?)', ('wxid-a', '示例教育', '示例', 'example_edu', '预算审批中', 'avatar-hash'))
            conn.commit(); conn.close()
            vault = root / 'vault'
            code, data = self.run_cli(['--vault', str(vault), 'import-contacts', str(contact_db), '--account-id', 'acct-a', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(data['imported_contacts'], 1)
            code, profile = self.run_cli(['--vault', str(vault), 'customer-profile', '--customer', '示例教育', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(profile['type'], 'customer_profile')
            self.assertTrue(profile['sections']['identity'])
            code, manifest = self.run_cli([
                '--vault', str(vault), 'profile-enrichment', '--actor', 'operator', '--session', 'cli-session',
                'plan', '--customer', '示例教育', '--mode', 'complete', '--item-budget', '20', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual(manifest['type'], 'profile_enrichment_manifest')
            self.assertFalse(manifest['raw_content_included'])
            self.assertTrue(manifest['profile_snapshot']['created'])
            code, status = self.run_cli([
                '--vault', str(vault), 'profile-enrichment', '--actor', 'operator', '--session', 'cli-session',
                'status', '--run-id', manifest['run_id'], '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual(status['run_id'], manifest['run_id'])
            self.assertEqual(status['state'], 'complete')
            code, snapshot = self.run_cli([
                '--vault', str(vault), 'profile-enrichment', '--actor', 'operator', '--session', 'cli-session',
                'finalize', '--run-id', manifest['run_id'], '--json',
            ])
            self.assertEqual(code, 0)
            self.assertTrue(snapshot['cache_hit'])
            code, snapshot_status = self.run_cli([
                '--vault', str(vault), 'profile-enrichment', '--actor', 'operator', '--session', 'cli-session',
                'snapshot-status', '--customer', '示例教育', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual(snapshot_status['completeness_state'], 'complete')
            self.assertFalse(snapshot_status['stale'])
            code, auto_enabled = self.run_cli([
                '--vault', str(vault), 'profile-auto', 'enable',
                '--customer', '示例教育', '--debounce-seconds', '0', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertTrue(auto_enabled['enabled'])
            code, auto_refresh = self.run_cli([
                '--vault', str(vault), 'profile-auto', 'refresh',
                '--customer', '示例教育', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual(auto_refresh['refresh']['created_snapshots'], 1)
            code, history = self.run_cli([
                '--vault', str(vault), 'profile-snapshots', 'list',
                '--customer', '示例教育', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual([item['version'] for item in history['items']], [2, 1])
            code, historical = self.run_cli([
                '--vault', str(vault), 'profile-snapshots', 'get',
                '--customer', '示例教育', '--version', '1', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual(historical['version'], 1)
            code, delta = self.run_cli([
                '--vault', str(vault), 'profile-snapshots', 'diff',
                '--customer', '示例教育', '--from-version', '1', '--to-version', '2', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertEqual((delta['from_version'], delta['to_version']), (1, 2))
            code, auto_disabled = self.run_cli([
                '--vault', str(vault), 'profile-auto', 'disable',
                '--customer', '示例教育', '--json',
            ])
            self.assertEqual(code, 0)
            self.assertFalse(auto_disabled['enabled'])

            audio = root / 'voice.wav'; audio.write_bytes(b'RIFFfixture')
            code, job = self.run_cli(['--vault', str(vault), 'voice-transcribe', '--audio-path', str(audio), '--asset-id', 'asset-v', '--citation', 'trove://voice', '--transcript', '示例教育语音', '--json'])
            self.assertEqual(code, 2)
            self.assertEqual(job['error']['code'], 'fake_asr_test_only')
            image = root / 'img.jpg'; image.write_bytes(b'\xff\xd8\xfffixture')
            code, job = self.run_cli(['--vault', str(vault), 'image-observe', '--image-path', str(image), '--asset-id', 'asset-i', '--citation', 'trove://image', '--caption', '示例教育图片', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(job['status'], 'completed')
            code, jobs = self.run_cli(['--vault', str(vault), 'provider-jobs', '--json'])
            self.assertEqual(code, 0)
            self.assertGreaterEqual(len(jobs['jobs']), 1)
            self.assertFalse(any(row.get('provider') == 'fake-asr' for row in jobs['jobs']))

    def test_media_observe_cli_passes_caption_flags(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('trove_core.application.sensitive_commands.run_image_observation_budget', return_value={'ok': True, 'caption_enabled': False, 'caption_budget': 7}) as observe:
                code, data = self.run_cli(['--vault', d, 'media', 'observe', '--budget', '3', '--no-caption', '--caption-budget', '7', '--include-images', '--yes', '--json'])
            self.assertEqual(code, 0)
            self.assertFalse(data['caption_enabled'])
            self.assertEqual(observe.call_args.kwargs['budget'], 3)
            self.assertFalse(observe.call_args.kwargs['caption'])
            self.assertEqual(observe.call_args.kwargs['caption_budget'], 7)
            self.assertTrue(observe.call_args.kwargs['include_images'])

    def test_media_annotate_cli_writes_cache_status_only(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'cli.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(__import__('base64').b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='))
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            citation = 'trove://wechat/acct-a/conv-a/message_0/1#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-cli-annotate',
                account_id='acct-a',
                source_type='message',
                source_id='source-cli',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/cli.png',
                cache_state='cached',
            ))

            code, data = self.run_cli([
                '--vault', str(vault), 'media', 'annotate', citation,
                '--caption', 'fixture caption',
                '--objects-json', '[{"label":"fixture-object"}]',
                '--business-signals-json', '[]',
                '--confidence', '0.7',
                '--model-id', 'agent-fixture-v1',
                '--prompt-version', 'p1',
                '--json',
            ])
            self.assertEqual(code, 0)
            self.assertTrue(data['ok'])
            self.assertEqual(data['status'], 'cached')
            self.assertFalse(data['raw_content_included'])
            self.assertNotIn('fixture caption', json.dumps(data, ensure_ascii=False))
            code, status = self.run_cli(['--vault', str(vault), 'media', 'understanding-status', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(status['active'], 1)
            code, invalidated = self.run_cli(['--vault', str(vault), 'media', 'annotate', '--invalidate', '--model-id', 'agent-fixture-v1', '--yes', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(invalidated['invalidated'], 1)


if __name__ == '__main__':
    unittest.main()
