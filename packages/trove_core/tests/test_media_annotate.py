from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from trove_core.agent_tools.tools import media_annotate
from trove_core.media_fetch import fetch_media
from trove_core.media_understanding import annotate_media_understanding, invalidate_media_understanding, media_understanding_status
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig

PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
)


class MediaAnnotateTests(unittest.TestCase):
    def test_identical_bytes_reuse_understanding_but_project_each_citation_once(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'shared.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citations = [
                'trove://wechat/acct/conv/s/1#image-0',
                'trove://wechat/acct/conv/s/2#image-0',
            ]
            for index, citation in enumerate(citations):
                repo.upsert_media_asset(MediaAssetRecord(
                    f'asset-shared-projection-{index}', 'acct', 'message', f'msg-{index}',
                    'image', 'image', citation, path_ref='sources/shared.png', cache_state='cached',
                ))
                media_annotate(
                    vault, citation=citation, caption='共享海报', visible_text='共享可搜索文字',
                    objects=[{'label': 'poster'}], business_signals=[{'type': 'pricing'}],
                    model_id='agent-shared', prompt_version='p1',
                )
            media_annotate(
                vault, citation=citations[1], caption='共享海报', visible_text='共享可搜索文字',
                objects=[{'label': 'poster'}], business_signals=[{'type': 'pricing'}],
                model_id='agent-shared', prompt_version='p1',
            )
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_understanding').fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM image_observations WHERE status='active'").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM evidence_chunks WHERE source_type='image_observation' AND status='active'").fetchone()[0], 2)
                row = conn.execute('SELECT objects_json,business_signals_json FROM image_observations WHERE citation=?', (citations[1],)).fetchone()
                self.assertEqual(len(json.loads(row['objects_json'])), 1)
                self.assertEqual(len(json.loads(row['business_signals_json'])), 1)

    def test_prompt_upgrade_records_history_and_supersedes_evidence_projection(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'upgrade.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct/conv/s/3#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-upgrade-projection', 'acct', 'message', 'msg-upgrade', 'image', 'image', citation,
                path_ref='sources/upgrade.png', cache_state='cached',
            ))
            media_annotate(vault, citation=citation, caption='旧视觉理解', model_id='agent-v1', prompt_version='p1')
            result = media_annotate(
                vault, citation=citation, caption='新视觉理解', visible_text='新版OCRneedle',
                model_id='agent-v2', prompt_version='p2',
            )
            self.assertEqual(result['history_count'], 1)
            with store.connect() as conn:
                rows = list(conn.execute('SELECT status,model_id,prompt_version FROM image_observations ORDER BY created_at,observation_id'))
            self.assertEqual({(row['status'], row['model_id'], row['prompt_version']) for row in rows}, {
                ('superseded', 'agent-v1', 'p1'), ('active', 'agent-v2', 'p2'),
            })
            hits = store.chunk_search('新版OCRneedle', filters={'source_type': 'image_observation'}, limit=3)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]['parent_citation'], citation)

    def test_task_bound_annotation_rejects_hash_asset_and_remote_execution_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'bound.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct/conv/s/4#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                'asset-bound-projection', 'acct', 'message', 'msg-bound', 'image', 'image', citation,
                path_ref='sources/bound.png', cache_state='cached',
            ))
            wrong_hash = annotate_media_understanding(
                vault, citation=citation, caption='blocked', model_id='agent', prompt_version='p1',
                expected_content_sha256='0' * 64, expected_asset_id='asset-bound-projection',
            )
            wrong_asset = annotate_media_understanding(
                vault, citation=citation, caption='blocked', model_id='agent', prompt_version='p1',
                expected_content_sha256=hashlib.sha256(PNG_1X1).hexdigest(), expected_asset_id='other-asset',
            )
            with self.assertRaises(ValueError):
                annotate_media_understanding(
                    vault, citation=citation, caption='blocked', model_id='agent', prompt_version='p1',
                    execution_location='remote',
                )
            self.assertEqual(wrong_hash['code'], 'annotation_content_hash_mismatch')
            self.assertEqual(wrong_asset['code'], 'annotation_asset_mismatch')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM media_understanding').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM image_observations').fetchone()[0], 0)

    def test_annotate_then_fetch_hits_without_echoing_content(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'annotate.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct-a/conv-a/message_0/1#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-annotate',
                account_id='acct-a',
                source_type='message',
                source_id='source-annotate',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/annotate.png',
                cache_state='cached',
            ))
            before = fetch_media(vault, citation)
            self.assertTrue(before['ok'], before)
            self.assertIsNone(before['understanding'])

            result = media_annotate(
                vault,
                citation=citation,
                caption='fixture caption',
                visible_text='fixture text',
                objects=[{'label': 'fixture-object'}],
                business_signals=[{'type': 'fixture-signal'}],
                confidence=0.9,
                model_id='agent-fixture-v1',
                prompt_version='p1',
            )
            self.assertTrue(result['ok'], result)
            self.assertFalse(result['raw_content_included'])
            self.assertNotIn('fixture caption', json.dumps(result, ensure_ascii=False))
            self.assertFalse(result['propose_channel']['auto_proposed'])

            after = fetch_media(vault, citation)
            self.assertEqual(after['understanding']['caption'], 'fixture caption')
            self.assertEqual(after['understanding']['business_signals'][0]['type'], 'fixture-signal')
            status = media_understanding_status(vault)
            self.assertEqual(status['active'], 1)
            self.assertEqual(status['modality_distribution'], {'image': 1})
            self.assertEqual(status['model_distribution'], {'agent-fixture-v1@p1': 1})
            self.assertGreaterEqual(status['total_fetch_hits'], 1)
            invalidated = invalidate_media_understanding(vault, model_id='agent-fixture-v1')
            self.assertEqual(invalidated['invalidated'], 1)
            self.assertFalse(invalidated['auto_rerun'])
            self.assertIsNone(fetch_media(vault, citation)['understanding'])



    def test_understanding_status_is_read_only_when_table_missing(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            cfg.paths.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(cfg.paths.sqlite_path) as conn:
                conn.execute('CREATE TABLE placeholder(id INTEGER PRIMARY KEY)')
                conn.commit()

            with patch.object(SQLiteStore, 'initialize', side_effect=AssertionError('status must not initialize')):
                status = media_understanding_status(vault)

            self.assertEqual(status['total'], 0)
            self.assertEqual(status['active'], 0)
            self.assertEqual(status['modality_distribution'], {})
            with sqlite3.connect(cfg.paths.sqlite_path) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables, {'placeholder'})

    def test_understanding_status_missing_db_does_not_create_index(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            self.assertFalse(cfg.paths.sqlite_path.exists())
            status = media_understanding_status(vault)
            self.assertEqual(status['total'], 0)
            self.assertFalse(cfg.paths.sqlite_path.exists())

    def test_annotate_hash_only_does_not_materialize_or_update_asset_hash(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'hashonly.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct-a/conv-a/message_0/10#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-hashonly',
                account_id='acct-a',
                source_type='message',
                source_id='source-hashonly',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/hashonly.png',
                cache_state='cached',
            ))
            decoded_dir = vault / 'media' / 'decoded'
            before_decoded = sorted(p.name for p in decoded_dir.glob('*')) if decoded_dir.exists() else []
            with store.connect() as conn:
                before_asset = dict(conn.execute('SELECT * FROM media_assets WHERE asset_id=?', ('asset-hashonly',)).fetchone())

            result = media_annotate(
                vault,
                citation=citation,
                caption='fixture caption',
                model_id='agent-hashonly-v1',
                prompt_version='p1',
            )

            self.assertTrue(result['ok'], result)
            self.assertEqual(result['content_sha256'], hashlib.sha256(PNG_1X1).hexdigest())
            after_decoded = sorted(p.name for p in decoded_dir.glob('*')) if decoded_dir.exists() else []
            self.assertEqual(after_decoded, before_decoded)
            with store.connect() as conn:
                after_asset = dict(conn.execute('SELECT * FROM media_assets WHERE asset_id=?', ('asset-hashonly',)).fetchone())
            self.assertEqual(after_asset, before_asset)

    def test_dat_annotate_hashes_decoded_bytes_in_memory(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            source = vault / 'sources' / 'wrapped.dat'
            source.parent.mkdir(parents=True, exist_ok=True)
            key = 0x5A
            source.write_bytes(bytes(b ^ key for b in PNG_1X1))
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct-a/conv-a/message_0/11#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-dat-hashonly',
                account_id='acct-a',
                source_type='message',
                source_id='source-dat-hashonly',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/wrapped.dat',
                cache_state='cached',
            ))
            decoded_dir = vault / 'media' / 'decoded'
            before_decoded = sorted(p.name for p in decoded_dir.glob('*')) if decoded_dir.exists() else []
            with store.connect() as conn:
                before_asset = dict(conn.execute('SELECT * FROM media_assets WHERE asset_id=?', ('asset-dat-hashonly',)).fetchone())

            result = media_annotate(
                vault,
                citation=citation,
                caption='fixture caption',
                model_id='agent-hashonly-v1',
                prompt_version='p1',
            )

            self.assertTrue(result['ok'], result)
            self.assertEqual(result['content_sha256'], hashlib.sha256(PNG_1X1).hexdigest())
            after_decoded = sorted(p.name for p in decoded_dir.glob('*')) if decoded_dir.exists() else []
            self.assertEqual(after_decoded, before_decoded)
            with store.connect() as conn:
                after_asset = dict(conn.execute('SELECT * FROM media_assets WHERE asset_id=?', ('asset-dat-hashonly',)).fetchone())
            self.assertEqual(after_asset, before_asset)

    def test_partial_annotation_preserves_existing_fields(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'partial.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            citation = 'trove://wechat/acct-a/conv-a/message_0/9#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-partial',
                account_id='acct-a',
                source_type='message',
                source_id='source-partial',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/partial.png',
                cache_state='cached',
            ))
            media_annotate(vault, citation=citation, caption='first caption', model_id='agent-v1', prompt_version='p1')
            media_annotate(vault, citation=citation, caption='', visible_text='second text', model_id='agent-v1', prompt_version='p1')
            fetched = fetch_media(vault, citation)
            self.assertEqual(fetched['understanding']['caption'], 'first caption')
            self.assertEqual(fetched['understanding']['visible_text'], 'second text')


    def test_replace_true_clears_explicit_empty_fields(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'replace.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            citation = 'trove://wechat/acct-a/conv-a/message_0/12#image-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-replace',
                account_id='acct-a',
                source_type='message',
                source_id='source-replace',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/replace.png',
                cache_state='cached',
            ))
            media_annotate(
                vault,
                citation=citation,
                caption='first caption',
                visible_text='first text',
                objects=[{'label': 'fixture-object'}],
                business_signals=[{'type': 'fixture-signal'}],
                keyframes=[{'time_seconds': 0, 'description': 'opening'}],
                audio_transcript='first transcript',
                confidence=0.7,
                model_id='agent-replace-v1',
                prompt_version='p1',
            )
            media_annotate(
                vault,
                citation=citation,
                caption='',
                visible_text='',
                objects=[],
                business_signals=[],
                keyframes=[],
                audio_transcript='',
                confidence=0.0,
                model_id='agent-replace-v1',
                prompt_version='p1',
                replace=True,
            )

            fetched = fetch_media(vault, citation)['understanding']
            self.assertEqual(fetched['caption'], '')
            self.assertEqual(fetched['visible_text'], '')
            self.assertEqual(fetched['objects'], [])
            self.assertEqual(fetched['business_signals'], [])
            self.assertEqual(fetched['keyframes'], [])
            self.assertEqual(fetched['audio_transcript'], '')
            self.assertEqual(fetched['confidence'], 0.0)

    def test_annotate_validates_structured_fields(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                media_annotate(
                    d,
                    citation='trove://missing',
                    objects='{"not":"a-list"}',
                    model_id='agent-fixture-v1',
                    prompt_version='p1',
                )


    def test_annotate_rejects_unsupported_structured_fields(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                media_annotate(
                    d,
                    citation='trove://missing#image-0',
                    objects=[{'label': 'fixture', 'unexpected': 'blocked'}],
                    model_id='agent-fixture-v1',
                    prompt_version='p1',
                )
            with self.assertRaises(ValueError):
                media_annotate(
                    d,
                    citation='trove://missing#image-0',
                    business_signals=[{'type': 123}],
                    model_id='agent-fixture-v1',
                    prompt_version='p1',
                )
            with self.assertRaises(ValueError):
                media_annotate(
                    d,
                    citation='trove://missing#image-0',
                    objects=[{'label': 'fixture', 'attributes': {'nested': {'blocked': {'again': True}}}}],
                    model_id='agent-fixture-v1',
                    prompt_version='p1',
                )

    def test_parent_citation_requires_unique_media_asset(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image0 = vault / 'sources' / 'parent0.png'
            image1 = vault / 'sources' / 'parent1.png'
            image0.parent.mkdir(parents=True, exist_ok=True)
            image0.write_bytes(PNG_1X1)
            image1.write_bytes(PNG_1X1)
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            parent = 'trove://wechat/acct-a/conv-a/message_0/20'
            for idx, path in enumerate((image0, image1)):
                repo.upsert_media_asset(MediaAssetRecord(
                    asset_id=f'asset-parent-{idx}',
                    account_id='acct-a',
                    source_type='message',
                    source_id=f'source-parent-{idx}',
                    modality='image',
                    media_type='image',
                    citation=f'{parent}#image-{idx}',
                    path_ref=f'sources/parent{idx}.png',
                    cache_state='cached',
                ))

            ambiguous = media_annotate(
                vault,
                citation=parent,
                caption='fixture caption',
                model_id='agent-parent-v1',
                prompt_version='p1',
            )
            self.assertFalse(ambiguous['ok'])
            self.assertEqual(ambiguous['code'], 'ambiguous_media_citation')
            self.assertFalse(ambiguous['raw_content_included'])

            exact = media_annotate(
                vault,
                citation=f'{parent}#image-1',
                caption='fixture caption',
                model_id='agent-parent-v1',
                prompt_version='p1',
            )
            self.assertTrue(exact['ok'], exact)
            self.assertEqual(exact['asset_id'], 'asset-parent-1')


    def test_unique_parent_citation_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'single-parent.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            parent = 'trove://wechat/acct-a/conv-a/message_0/22'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-single-parent',
                account_id='acct-a',
                source_type='message',
                source_id='source-single-parent',
                modality='image',
                media_type='image',
                citation=f'{parent}#image-0',
                path_ref='sources/single-parent.png',
                cache_state='cached',
            ))

            result = media_annotate(
                vault,
                citation=parent,
                caption='fixture caption',
                model_id='agent-parent-v1',
                prompt_version='p1',
            )
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['asset_id'], 'asset-single-parent')

    def test_invalid_citation_fragment_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'invalid-fragment.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            repo = MultimodalRepository(SQLiteStore(cfg.paths.sqlite_path))
            parent = 'trove://wechat/acct-a/conv-a/message_0/21'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-invalid-fragment',
                account_id='acct-a',
                source_type='message',
                source_id='source-invalid-fragment',
                modality='image',
                media_type='image',
                citation=f'{parent}#image-0',
                path_ref='sources/invalid-fragment.png',
                cache_state='cached',
            ))

            result = media_annotate(
                vault,
                citation=f'{parent}#chunk-0',
                caption='fixture caption',
                model_id='agent-parent-v1',
                prompt_version='p1',
            )
            self.assertFalse(result['ok'])
            self.assertEqual(result['code'], 'invalid_media_citation')

    def test_video_annotate_and_fetch_returns_keyframes(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            video = vault / 'sources' / 'clip.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'fixture video bytes')
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct-a/conv-a/message_0/2#video-0'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-video-annotate',
                account_id='acct-a',
                source_type='message',
                source_id='source-video',
                modality='video',
                media_type='video',
                citation=citation,
                path_ref='sources/clip.mp4',
                cache_state='cached',
            ))
            result = media_annotate(
                vault,
                citation=citation,
                caption='fixture video caption',
                keyframes=[{'time_seconds': 0, 'description': 'opening'}],
                audio_transcript='fixture transcript',
                confidence=0.8,
                model_id='agent-video-v1',
                prompt_version='p-video',
            )
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['modality'], 'video')
            fetched = fetch_media(vault, citation)
            self.assertTrue(fetched['ok'], fetched)
            self.assertEqual(fetched['mime'], 'video/mp4')
            self.assertTrue(Path(fetched['path']).is_file())
            self.assertIn('preview', fetched)
            self.assertEqual(fetched['understanding']['keyframes'][0]['time_seconds'], 0)
            self.assertEqual(fetched['understanding']['audio_transcript'], 'fixture transcript')
            with self.assertRaises(ValueError):
                media_annotate(
                    d,
                    citation='trove://missing',
                    objects=[1],
                    model_id='agent-fixture-v1',
                    prompt_version='p1',
                )


if __name__ == '__main__':
    unittest.main()
