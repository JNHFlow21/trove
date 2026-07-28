from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from trove_core.asr.fake import FakeASRProvider
from trove_core.media_fetch import fetch_media
from trove_core.media_pipeline import ensure_voice_transcript
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.wechat.import_job import run_import_job
from trove_core.wechat.importers.wechat_decrypted import msg_table_for
from trove_core.wechat.decrypt.manifest import write_account_identity
from trove_core.wechat.media.materializer import materialize_media_asset
from trove_core.wechat.media.locator import _remote_message_sticker_url
from packages.trove_core.tests.test_message_media_registration import create_multimodal_message_source


class LazyMediaLocatorTests(unittest.TestCase):
    def test_sticker_cdn_url_is_recovered_from_exact_message_without_persisting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            account = Path(directory) / 'com.tencent.xinWeChat__wxid_ownerfixture'
            account.mkdir()
            table = msg_table_for('wxid_fixtureb')
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute('CREATE TABLE Name2Id (user_name TEXT)')
                conn.execute('INSERT INTO Name2Id VALUES (?)', ('wxid_fixtureb',))
                conn.execute(f'CREATE TABLE "{table}" (local_id INTEGER,message_content BLOB)')
                conn.execute(
                    f'INSERT INTO "{table}" VALUES (?,?)',
                    (7, b'<msg><emoji cdnurl="http://vweixinf.tc.qq.com/sticker.gif" encrypturl="http://untrusted.invalid/encrypted" /></msg>'),
                )
                conn.commit()

            recovered = _remote_message_sticker_url(account, {
                'message_shard_id': 'message_0',
                'conversation_id': 'conv-' + hashlib.sha256(
                    f'{account.name}:wxid_fixtureb'.encode(),
                ).hexdigest()[:12],
                'message_local_id': 7,
            })

            self.assertEqual(recovered, 'https://vweixinf.tc.qq.com/sticker.gif')

    def test_video_with_resource_metadata_but_no_local_cache_has_typed_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-missing-video'
            account = create_multimodal_message_source(snapshot)
            table = msg_table_for('wxid_personmedia')
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN server_id INTEGER')
                conn.execute(f'UPDATE "{table}" SET server_id=? WHERE local_id=3', (33445566,))
                conn.commit()
            with sqlite3.connect(account / 'message_resource.db') as conn:
                conn.execute('CREATE TABLE MessageResourceInfo (message_svr_id INTEGER,packed_info BLOB)')
                conn.execute(
                    'INSERT INTO MessageResourceInfo VALUES (?,?)',
                    (33445566, b'\x12\x22\x0a\x20' + b'5952151887f79d432023f406c221ffbb'),
                )
                conn.commit()
            with sqlite3.connect(account / 'hardlink.db') as conn:
                conn.execute('CREATE TABLE dir2id (username TEXT PRIMARY KEY)')
                conn.execute(
                    'CREATE TABLE video_hardlink_info_v4 (file_name TEXT,file_size INTEGER,dir1 INTEGER,dir2 INTEGER,type INTEGER)'
                )
                conn.commit()
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                video = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='video' AND m.conversation_type='private'"
                ).fetchone()

            with mock.patch(
                'trove_core.wechat.media.locator._wechat_documents_root', return_value=root / 'Documents',
            ):
                result = materialize_media_asset(cfg, store, video, citation=video['message_citation'])

            self.assertFalse(result.ok)
            self.assertEqual(result.reason, 'local_video_cache_missing')

    def test_video_materializes_from_exact_live_hardlink_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-live-video'
            account = create_multimodal_message_source(snapshot)
            table = msg_table_for('wxid_personmedia')
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN server_id INTEGER')
                conn.execute(f'UPDATE "{table}" SET server_id=? WHERE local_id=3', (22334455,))
                conn.commit()
            resource_key = 'ab' * 16
            with sqlite3.connect(account / 'message_resource.db') as conn:
                conn.execute('CREATE TABLE MessageResourceInfo (message_svr_id INTEGER,packed_info BLOB)')
                conn.execute(
                    'INSERT INTO MessageResourceInfo VALUES (?,?)',
                    (22334455, b'\x12\x22\x0a\x20' + resource_key.encode()),
                )
                conn.commit()
            video_bytes = b'\x00\x00\x00\x18ftypmp42' + b'x' * 32
            with sqlite3.connect(account / 'hardlink.db') as conn:
                conn.execute('CREATE TABLE dir2id (username TEXT PRIMARY KEY)')
                conn.execute('INSERT INTO dir2id(rowid,username) VALUES (?,?)', (1, '2026-01'))
                conn.execute(
                    'CREATE TABLE video_hardlink_info_v4 (file_name TEXT,file_size INTEGER,dir1 INTEGER,dir2 INTEGER,type INTEGER)'
                )
                conn.execute(
                    'INSERT INTO video_hardlink_info_v4 VALUES (?,?,?,?,?)',
                    (resource_key + '.mp4', len(video_bytes), 1, 0, 3),
                )
                conn.commit()
            live_documents = root / 'Documents'
            live_video = live_documents / 'xwechat_files' / 'wxid_ownerfixture' / 'msg' / 'video' / '2026-01' / f'{resource_key}.mp4'
            live_video.parent.mkdir(parents=True)
            live_video.write_bytes(video_bytes)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                video = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='video' AND m.conversation_type='private'"
                ).fetchone()

            with mock.patch(
                'trove_core.wechat.media.locator._wechat_documents_root', return_value=live_documents,
            ):
                result = materialize_media_asset(cfg, store, video, citation=video['message_citation'])

            self.assertTrue(result.ok, result.to_redacted_dict())
            self.assertEqual(result.route, 'live_wechat_hardlink_cache')
            self.assertEqual(Path(result.path).read_bytes(), video_bytes)

    def test_file_materializes_from_exact_live_wechat_file_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-live-file'
            account = create_multimodal_message_source(snapshot)
            opaque_account = snapshot / 'acct-opaque-fixture'
            account.rename(opaque_account)
            account = opaque_account
            write_account_identity(
                account,
                account_ref_hash='0123456789abcdef',
                own_wxid='wxid_ownerfixture',
            )
            file_bytes = b'x' * 42
            with sqlite3.connect(account / 'hardlink.db') as conn:
                conn.execute('CREATE TABLE dir2id (username TEXT PRIMARY KEY)')
                conn.execute('INSERT INTO dir2id(rowid,username) VALUES (?,?)', (1, '2026-01'))
                conn.execute(
                    'CREATE TABLE file_hardlink_info_v4 (file_name TEXT,file_size INTEGER,dir1 INTEGER,dir2 INTEGER,type INTEGER)'
                )
                conn.execute(
                    'INSERT INTO file_hardlink_info_v4 VALUES (?,?,?,?,?)',
                    ('方案.pdf', len(file_bytes), 1, 0, 1),
                )
                conn.commit()
            live_documents = root / 'Documents'
            live_file = live_documents / 'xwechat_files' / 'wxid_ownerfixture' / 'msg' / 'file' / '2026-01' / '方案.pdf'
            live_file.parent.mkdir(parents=True)
            live_file.write_bytes(file_bytes)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                document = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='file' AND m.conversation_type='private'"
                ).fetchone()

            with mock.patch(
                'trove_core.wechat.media.locator._wechat_documents_root', return_value=live_documents,
            ):
                result = materialize_media_asset(cfg, store, document, citation=document['message_citation'])

            self.assertTrue(result.ok, result.to_redacted_dict())
            self.assertEqual(result.route, 'live_wechat_file_cache')
            self.assertEqual(Path(result.path).read_bytes(), file_bytes)

    def test_indexed_file_without_live_bytes_is_a_typed_lazy_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-missing-file'
            account = create_multimodal_message_source(snapshot)
            with sqlite3.connect(account / 'hardlink.db') as conn:
                conn.execute('CREATE TABLE dir2id (username TEXT PRIMARY KEY)')
                conn.execute('INSERT INTO dir2id(rowid,username) VALUES (?,?)', (1, '2026-01'))
                conn.execute(
                    'CREATE TABLE file_hardlink_info_v4 (file_name TEXT,file_size INTEGER,dir1 INTEGER,dir2 INTEGER,type INTEGER)'
                )
                conn.execute('INSERT INTO file_hardlink_info_v4 VALUES (?,?,?,?,?)', ('方案.pdf', 42, 1, 0, 1))
                conn.commit()
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                document = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='file' AND m.conversation_type='private'"
                ).fetchone()

            with mock.patch(
                'trove_core.wechat.media.locator._wechat_documents_root', return_value=root / 'Documents',
            ):
                result = materialize_media_asset(cfg, store, document, citation=document['message_citation'])

            self.assertFalse(result.ok)
            self.assertEqual(result.route, 'live_wechat_file_cache')
            self.assertEqual(result.reason, 'local_file_cache_missing')

    def test_file_without_hardlink_snapshot_still_reports_client_cache_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-pre-download-file'
            create_multimodal_message_source(snapshot)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                document = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='file' AND m.conversation_type='private'"
                ).fetchone()

            with mock.patch(
                'trove_core.wechat.media.locator._wechat_documents_root', return_value=root / 'Documents',
            ):
                result = materialize_media_asset(cfg, store, document, citation=document['message_citation'])

            self.assertFalse(result.ok)
            self.assertEqual(result.route, 'live_wechat_file_cache')
            self.assertEqual(result.reason, 'local_file_cache_missing')

    def test_image_materializes_from_exact_live_hardlink_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-live-image'
            account = create_multimodal_message_source(snapshot)
            (account / 'media' / 'private.jpg').unlink()
            table = msg_table_for('wxid_personmedia')
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN server_id INTEGER')
                conn.execute(
                    f'UPDATE "{table}" SET packed_info_data=NULL,server_id=? WHERE local_id=1',
                    (123456789,),
                )
                conn.commit()
            resource_key = 'cd' * 16
            with sqlite3.connect(account / 'message_resource.db') as conn:
                conn.execute('CREATE TABLE MessageResourceInfo (message_svr_id INTEGER,packed_info BLOB)')
                conn.execute(
                    'INSERT INTO MessageResourceInfo VALUES (?,?)',
                    (123456789, b'\x12\x22\x0a\x20' + resource_key.encode()),
                )
                conn.commit()
            with sqlite3.connect(account / 'hardlink.db') as conn:
                conn.execute('CREATE TABLE dir2id (username TEXT PRIMARY KEY)')
                conn.execute('INSERT INTO dir2id(rowid,username) VALUES (?,?)', (1, 'conversation-cache'))
                conn.execute('INSERT INTO dir2id(rowid,username) VALUES (?,?)', (2, '2026-01'))
                conn.execute(
                    'CREATE TABLE image_hardlink_info_v4 (file_name TEXT,file_size INTEGER,dir1 INTEGER,dir2 INTEGER)'
                )
                image_bytes = b'\xff\xd8\xff\xe0' + b'x' * 32
                uin = 1509864304
                aes_key = hashlib.md5(f'{uin}wxid_ownerfixture'.encode()).hexdigest()[:16].encode()
                padding = 16 - (len(image_bytes) % 16)
                padded = image_bytes + bytes([padding]) * padding
                encryptor = Cipher(algorithms.AES(aes_key), modes.ECB()).encryptor()
                ciphertext = encryptor.update(padded) + encryptor.finalize()
                dat_bytes = b'\x07\x08V2\x08\x07' + len(image_bytes).to_bytes(4, 'little') + (0).to_bytes(4, 'little') + b'\x01' + ciphertext
                conn.execute(
                    'INSERT INTO image_hardlink_info_v4 VALUES (?,?,?,?)',
                    (resource_key + '.dat', len(dat_bytes), 1, 2),
                )
                conn.commit()
            live_documents = root / 'Documents'
            live_image = live_documents / 'xwechat_files' / 'wxid_ownerfixture' / 'msg' / 'attach' / 'conversation-cache' / '2026-01' / 'Img' / f'{resource_key}.dat'
            live_image.parent.mkdir(parents=True)
            live_image.write_bytes(dat_bytes)
            kvcomm = live_documents / 'app_data' / 'net' / 'kvcomm'
            kvcomm.mkdir(parents=True)
            (kvcomm / f'key_{uin}_fixture.statistic').touch()
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                image = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='image' AND m.conversation_type='private'"
                ).fetchone()

            with mock.patch(
                'trove_core.wechat.media.locator._wechat_documents_root', return_value=live_documents,
            ):
                result = materialize_media_asset(cfg, store, image, citation=image['message_citation'])

            self.assertTrue(result.ok, result.to_redacted_dict())
            self.assertEqual(result.route, 'live_wechat_hardlink_cache')
            self.assertEqual(Path(result.path).read_bytes(), image_bytes)

    def test_voice_materializes_from_exact_embedded_voice_info_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-embedded-voice'
            account = create_multimodal_message_source(snapshot)
            (account / 'media' / 'voice.wav').unlink()
            with sqlite3.connect(account / 'message_0.db') as conn:
                table = msg_table_for('wxid_personmedia')
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN server_id INTEGER')
                conn.execute(
                    f'UPDATE "{table}" SET packed_info_data=NULL,server_id=? WHERE local_id=2',
                    (987654321,),
                )
                conn.commit()
            silk = b'\x02#!SILK_V3' + b'\x00' * 64
            with sqlite3.connect(account / 'media_0.db') as conn:
                conn.execute(
                    'CREATE TABLE VoiceInfo (chat_name_id INTEGER,create_time INTEGER,local_id INTEGER,svr_id INTEGER,voice_data BLOB,data_index INTEGER)'
                )
                conn.execute(
                    'INSERT INTO VoiceInfo VALUES (?,?,?,?,?,?)',
                    (1, 1710000060, 2, 987654321, silk, 0),
                )
                conn.commit()
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                voice = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='voice' AND m.conversation_type='private'"
                ).fetchone()

            result = materialize_media_asset(cfg, store, voice, citation=voice['message_citation'])

            self.assertTrue(result.ok, result.to_redacted_dict())
            self.assertEqual(result.route, 'source_voice_info_blob')
            path = Path(result.path)
            self.assertEqual(path.suffix, '.silk')
            self.assertEqual(path.read_bytes(), silk)

    def test_voice_materializes_when_server_id_drifts_but_chat_local_time_match(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-voice-server-drift'
            account = create_multimodal_message_source(snapshot)
            (account / 'media' / 'voice.wav').unlink()
            with sqlite3.connect(account / 'message_0.db') as conn:
                table = msg_table_for('wxid_personmedia')
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN server_id INTEGER')
                conn.execute(
                    f'UPDATE "{table}" SET packed_info_data=NULL,server_id=? WHERE local_id=2',
                    (987654321,),
                )
                conn.commit()
            silk = b'\x02#!SILK_V3' + b'\x01' * 64
            with sqlite3.connect(account / 'media_0.db') as conn:
                conn.execute('CREATE TABLE Name2Id (user_name TEXT)')
                conn.execute(
                    'INSERT INTO Name2Id(rowid,user_name) VALUES (?,?)',
                    (1, 'wxid_personmedia'),
                )
                conn.execute(
                    'CREATE TABLE VoiceInfo (chat_name_id INTEGER,create_time INTEGER,local_id INTEGER,svr_id INTEGER,voice_data BLOB,data_index INTEGER)'
                )
                conn.execute(
                    'INSERT INTO VoiceInfo VALUES (?,?,?,?,?,?)',
                    (1, 1710000060, 2, 123456789, silk, 0),
                )
                conn.commit()
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                voice = conn.execute(
                    "SELECT ma.*,m.citation AS message_citation FROM media_assets ma JOIN messages m ON m.citation=ma.citation "
                    "WHERE ma.modality='voice' AND m.conversation_type='private'"
                ).fetchone()

            result = materialize_media_asset(
                cfg,
                store,
                voice,
                citation=voice['message_citation'],
            )

            self.assertTrue(result.ok, result.to_redacted_dict())
            self.assertEqual(result.route, 'source_voice_info_blob')
            self.assertEqual(Path(result.path).read_bytes(), silk)

    def test_image_and_voice_materialize_from_bound_snapshot_once(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-local'
            create_multimodal_message_source(snapshot)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                image = conn.execute("SELECT citation FROM messages WHERE content_kind='image' AND conversation_type='private'").fetchone()
                voice = conn.execute("SELECT citation FROM messages WHERE content_kind='voice' AND conversation_type='private'").fetchone()

            first_image = fetch_media(vault, image['citation'])
            second_image = fetch_media(vault, image['citation'])
            pending_voice = ensure_voice_transcript(vault, citation=voice['citation'])
            done_voice = ensure_voice_transcript(
                vault,
                citation=voice['citation'],
            )

            self.assertTrue(first_image['ok'], first_image)
            self.assertEqual(first_image['content_sha256'], second_image['content_sha256'])
            self.assertTrue(Path(first_image['path']).resolve().is_relative_to(vault.resolve()))
            self.assertEqual(pending_voice['status'], 'pending_transcript')
            self.assertEqual(done_voice['status'], 'pending_transcript')
            with store.connect() as conn:
                rows = list(conn.execute(
                    "SELECT path_ref,cache_state,content_hash FROM media_assets WHERE modality IN ('image','voice') AND source_type='private_chat'"
                ))
            self.assertTrue(all(row['cache_state'] == 'cached' for row in rows))
            self.assertTrue(all(row['path_ref'] and not Path(row['path_ref']).is_absolute() for row in rows))
            self.assertEqual(len(list((vault / 'media' / 'materialized').rglob('*.*'))), 2)

    def test_fetch_materialize_and_decode_run_without_global_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-lock-probe'
            create_multimodal_message_source(snapshot)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                citation = conn.execute(
                    "SELECT citation FROM messages WHERE content_kind='image' AND conversation_type='private'"
                ).fetchone()[0]

            from trove_core import media_fetch as media_fetch_module
            original_materialize = media_fetch_module.materialize_media_asset
            original_resolve = media_fetch_module.resolve_image_file
            phases: list[str] = []

            def probe(phase: str) -> None:
                with VaultOperationCoordinator(cfg).write(owner=f'probe-{phase}'):
                    phases.append(phase)

            def materialize(*args, **kwargs):
                probe('materialize')
                return original_materialize(*args, **kwargs)

            def resolve(*args, **kwargs):
                probe('decode')
                return original_resolve(*args, **kwargs)

            with mock.patch('trove_core.media_fetch.materialize_media_asset', materialize), \
                 mock.patch('trove_core.media_fetch.resolve_image_file', resolve):
                result = fetch_media(vault, citation)

            self.assertTrue(result['ok'], result)
            self.assertEqual(phases, ['materialize', 'decode'])

    def test_missing_snapshot_is_a_typed_terminal_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-missing'
            create_multimodal_message_source(snapshot)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                citation = conn.execute("SELECT citation FROM messages WHERE content_kind='image' AND conversation_type='private'").fetchone()[0]
            import shutil
            shutil.rmtree(snapshot)

            result = fetch_media(vault, citation)

            self.assertFalse(result['ok'])
            self.assertEqual(result['reason'], 'source_snapshot_unavailable')
            self.assertFalse(result['raw_paths_included'])

    def test_symlink_escape_is_rejected_and_never_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            snapshot = vault / 'decrypted' / 'runs' / 'run-symlink'
            account = create_multimodal_message_source(snapshot)
            outside = vault / 'outside.png'
            outside.write_bytes(b'\x89PNG\r\n\x1a\n' + b'x' * 32)
            image = account / 'media' / 'private.jpg'
            image.unlink()
            image.symlink_to(outside)
            run_import_job(vault, [snapshot], reset_index=True)
            store = SQLiteStore(cfg.paths.sqlite_path)
            with store.connect() as conn:
                citation = conn.execute("SELECT citation FROM messages WHERE content_kind='image' AND conversation_type='private'").fetchone()[0]

            result = fetch_media(vault, citation)

            self.assertFalse(result['ok'])
            self.assertIn(result['reason'], {'locator_routes_exhausted', 'source_snapshot_changed'})
            self.assertFalse((vault / 'media' / 'materialized').exists())


if __name__ == '__main__':
    unittest.main()
