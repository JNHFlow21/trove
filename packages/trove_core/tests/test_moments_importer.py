from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.knowledge.customer_profile import build_customer_profile
from trove_core.runtime import build_search_engine
from trove_core.search.query import SearchRequest
from trove_core.store.repositories import EntityRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.auxiliary_import import _moment_parent_citation_sql, import_auxiliary_sources
from trove_core.wechat.importers.moments import MomentsImporter
from trove_core.media_fetch import fetch_media
import hashlib
import base64


class MomentsImporterTests(unittest.TestCase):
    def test_moment_media_parent_expression_is_exact_and_indexable(self):
        expression = _moment_parent_citation_sql('media_asset_links.source_citation')
        self.assertNotIn('LIKE', expression.upper())
        conn = sqlite3.connect(':memory:')
        try:
            rows = []
            for citation in (
                'trove://wechat/acct/moment/m1',
                'trove://wechat/acct/moment/m1#image-0',
                'trove://wechat/acct/moment/m1#video-2',
            ):
                row = conn.execute(
                    f'SELECT {expression} FROM (SELECT ? AS source_citation) AS media_asset_links',
                    (citation,),
                ).fetchone()
                rows.append(row[0])
        finally:
            conn.close()
        self.assertEqual(rows, ['trove://wechat/acct/moment/m1'] * 3)

    def test_whitelist_imports_timeline_only_and_counts_others(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'sns.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMessage_tmp3(local_id INTEGER, create_time INTEGER, type INTEGER, feed_id TEXT, from_username TEXT, from_nickname TEXT, content TEXT, comment_id TEXT)')
            conn.execute('CREATE TABLE SnsAdTimeLine(tid TEXT, username TEXT, content TEXT, create_time INTEGER)')
            conn.execute('CREATE TABLE SnsTopItem_1(tid TEXT, username TEXT, summary TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('sns1', 'wxid-a', '客户发布新校区动态', b''))
            conn.execute('INSERT INTO SnsMessage_tmp3 VALUES(?,?,?,?,?,?,?,?)', (1, 1760000000, 1, 'sns1', 'wxid-b', 'Nick', '赞', '99'))
            conn.execute('INSERT INTO SnsAdTimeLine VALUES(?,?,?,?)', ('ad1', 'ad-user', 'ad body', 1760000000))
            conn.execute('INSERT INTO SnsTopItem_1 VALUES(?,?,?)', ('top1', 'wxid-a', 'summary'))
            conn.commit(); conn.close()
            importer = MomentsImporter(db, account_id='acct-a')
            rows = importer.load()
            self.assertEqual(len(rows), 1)
            self.assertIn('客户发布', rows[0].text)
            self.assertEqual(importer.last_report['interaction_source_rows'], 1)
            self.assertEqual(importer.last_report['excluded_counts']['moment_ad'], 1)
            self.assertEqual(importer.last_report['skipped_tables']['SnsTopItem_1'], 1)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            self.assertEqual(importer.import_to_store(repo), 1)
            self.assertEqual(importer.last_report['persistence_commits'], 1)
            with repo.store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM moment_items').fetchone()[0], 1)
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM moment_interactions').fetchone()[0], 1)
                row = dbconn.execute('SELECT interaction_type, actor_id, actor_name, text, metadata_json FROM moment_interactions').fetchone()
                self.assertEqual(row['interaction_type'], 'like')
                self.assertEqual(row['actor_id'], 'wxid-b')
                self.assertEqual(row['actor_name'], 'Nick')
                self.assertEqual(row['text'], '')
            self.assertNotIn('99', row['text'])

    def test_batch_persistence_rejects_cross_account_parent_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            with self.assertRaisesRegex(ValueError, 'account_id'):
                repo.upsert_moment_batch(
                    [{'moment_id': 'm1', 'account_id': 'acct-a', 'citation': 'trove://wechat/acct-a/moment/m1'}],
                    [{
                        'interaction_id': 'i1', 'moment_id': 'm1', 'account_id': 'acct-b',
                        'citation': 'trove://wechat/acct-b/moment/m1/interaction/i1',
                        'interaction_type': 'comment',
                    }],
                )
            with repo.store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM moment_items').fetchone()[0], 0)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM moment_interactions').fetchone()[0], 0)


    def test_timeline_xml_parses_author_timestamp_text_and_media(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'sns.db'
            xml = """<TimelineObject>
              <id>native-1</id><username>wxid-author</username><createTime>1760000000</createTime>
              <contentDesc>XML 正文</contentDesc>
              <ContentObject><mediaList><media><id>m1</id><type>2</type><thumb>https://thumb.invalid/a.jpg</thumb><url>https://cdn.invalid/a.jpg</url></media></mediaList></ContentObject>
            </TimelineObject>"""
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('tid-fallback', 'fallback-author', xml, b''))
            conn.commit(); conn.close()
            importer = MomentsImporter(db, account_id='acct-a')
            rows = importer.load()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].moment_id, 'moment-' + __import__('hashlib').sha256(b'acct-a:native-1').hexdigest()[:12])
            self.assertEqual(rows[0].author_id, 'wxid-author')
            self.assertTrue(rows[0].timestamp.startswith('2025-10-09T'))
            self.assertEqual(rows[0].text, 'XML 正文')
            self.assertEqual(rows[0].media_refs[0]['idx'], 0)
            self.assertEqual(rows[0].media_refs[0]['media_type'], 'image')
            self.assertIn('url_hash', rows[0].media_refs[0])
            self.assertIn('thumb_hash', rows[0].media_refs[0])
            self.assertNotIn('https://cdn.invalid', str(rows[0].media_refs[0]))
            self.assertEqual(importer.last_report['parse_success'], 1)
            self.assertEqual(importer.last_report['media_refs_count'], 1)
            self.assertEqual(importer.last_report['media_refs_nonempty_count'], 1)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            importer.import_to_store(repo)
            with repo.store.connect() as dbconn:
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_assets WHERE source_type='moment'").fetchone()[0], 1)
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_asset_links WHERE source_type='moment' AND accepted=1").fetchone()[0], 1)
                asset = dbconn.execute("SELECT citation, cache_state FROM media_assets WHERE source_type='moment'").fetchone()
                self.assertTrue(asset['citation'].endswith('#image-0'))
                self.assertEqual(asset['cache_state'], 'missing_local_cache')

    def test_duplicate_moment_media_urls_create_distinct_image_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'sns.db'
            xml = """<TimelineObject>
              <id>native-dup-media</id><username>wxid-author</username><createTime>1760000000</createTime>
              <contentDesc>重复图片</contentDesc>
              <ContentObject><mediaList>
                <media><id>m1</id><type>2</type><url>https://cdn.invalid/same.jpg</url></media>
                <media><id>m2</id><type>2</type><url>https://cdn.invalid/same.jpg</url></media>
              </mediaList></ContentObject>
            </TimelineObject>"""
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-dup-media', 'wxid-author', xml, b''))
            conn.commit(); conn.close()

            store = SQLiteStore(root / 'vault.sqlite')
            repo = MultimodalRepository(store)
            MomentsImporter(db, account_id='acct-a').import_to_store(repo)

            with store.connect() as dbconn:
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_assets WHERE source_type='moment'").fetchone()[0], 2)
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_asset_links WHERE source_type='moment' AND accepted=1").fetchone()[0], 2)
                rows = list(dbconn.execute("SELECT citation, source_id FROM media_assets WHERE source_type='moment' ORDER BY citation"))
                self.assertTrue(rows[0]['citation'].endswith('#image-0'))
                self.assertTrue(rows[1]['citation'].endswith('#image-1'))
                self.assertTrue(rows[0]['source_id'].endswith('#image-0'))
                self.assertTrue(rows[1]['source_id'].endswith('#image-1'))

    def test_single_image_moments_keep_per_moment_image_zero_citations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'sns.db'
            xml1 = '<TimelineObject><id>native-a</id><username>wxid-author</username><createTime>1760000000</createTime><contentDesc>A</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>https://cdn.invalid/a.jpg</url></media></mediaList></ContentObject></TimelineObject>'
            xml2 = '<TimelineObject><id>native-b</id><username>wxid-author</username><createTime>1760000001</createTime><contentDesc>B</contentDesc><ContentObject><mediaList><media><id>m2</id><type>2</type><url>https://cdn.invalid/b.jpg</url></media></mediaList></ContentObject></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-a', 'wxid-author', xml1, b''))
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-b', 'wxid-author', xml2, b''))
            conn.commit(); conn.close()
            store = SQLiteStore(root / 'vault.sqlite')
            repo = MultimodalRepository(store)
            MomentsImporter(db, account_id='acct-a').import_to_store(repo)
            with store.connect() as dbconn:
                citations = [row['citation'] for row in dbconn.execute("SELECT citation FROM media_assets WHERE source_type='moment' ORDER BY citation")]
            self.assertEqual(len(citations), 2)
            self.assertTrue(all(citation.endswith('#image-0') for citation in citations))

    def test_cached_moment_video_fetches_as_video(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            account_dir = vault / 'sources' / 'acct'
            account_dir.mkdir(parents=True)
            db = account_dir / 'sns.db'
            cache_key = 'ef' + hashlib.sha256(b'wechat-sns-video-key').hexdigest()[:30]
            cache_dir = account_dir / 'cache' / '2026-01' / 'sns' / 'img' / cache_key[:2]
            cache_dir.mkdir(parents=True)
            (cache_dir / cache_key[2:]).write_bytes(b'\x00\x00\x00\x18ftypisomfixture-video')
            xml = '<TimelineObject><id>native-video</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>视频</contentDesc><ContentObject><mediaList><media><id>video1</id><type>6</type><url>https://cdn.invalid/v.mp4</url></media></mediaList></ContentObject></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMediaCache(feed_id TEXT, media_id TEXT, cache_key TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-video', 'wxid-a', xml, b''))
            conn.execute('INSERT INTO SnsMediaCache VALUES(?,?,?)', ('native-video', 'video1', cache_key))
            conn.commit(); conn.close()
            repo = MultimodalRepository(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            MomentsImporter(db, account_id='acct-a').import_to_store(repo)
            with repo.store.connect() as dbconn:
                asset = dbconn.execute("SELECT citation, modality, cache_state FROM media_assets WHERE source_type='moment'").fetchone()
                self.assertEqual(asset['modality'], 'video')
                self.assertEqual(asset['cache_state'], 'source_available')
                self.assertTrue(asset['citation'].endswith('#video-0'))
                citation = asset['citation']
            fetched = fetch_media(vault, citation)
            self.assertTrue(fetched['ok'], fetched)
            self.assertEqual(fetched['mime'], 'video/mp4')

    def test_interaction_hex_does_not_false_map_sns_cache(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            account_dir = vault / 'sources' / 'acct'
            account_dir.mkdir(parents=True)
            db = account_dir / 'sns.db'
            cache_key = 'fa' + hashlib.sha256(b'false-cache-key').hexdigest()[:30]
            cache_dir = account_dir / 'cache' / '2026-01' / 'sns' / 'img' / cache_key[:2]
            cache_dir.mkdir(parents=True)
            (cache_dir / cache_key[2:]).write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8z8AARQABywH6f6n6AAAAAElFTkSuQmCC'))
            xml = '<TimelineObject><id>native-false</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>图</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>https://cdn.invalid/false.jpg</url></media></mediaList></ContentObject></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMessage_tmp3(local_id INTEGER, create_time INTEGER, type INTEGER, feed_id TEXT, from_username TEXT, from_nickname TEXT, content TEXT, comment_id TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-false', 'wxid-a', xml, b''))
            conn.execute('INSERT INTO SnsMessage_tmp3 VALUES(?,?,?,?,?,?,?,?)', (1, 1760000000, 2, 'native-false', 'wxid-b', 'Nick', cache_key, '99'))
            conn.commit(); conn.close()
            repo = MultimodalRepository(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            MomentsImporter(db, account_id='acct-a').import_to_store(repo)
            with repo.store.connect() as dbconn:
                asset = dbconn.execute("SELECT cache_state, path_ref FROM media_assets WHERE source_type='moment'").fetchone()
                self.assertEqual(asset['cache_state'], 'inventory_only')
                self.assertFalse(asset['path_ref'])


    def test_message_tmp3_comments_map_by_feed_id_and_filter_id_fields(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'sns.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMessage_tmp3(local_id INTEGER, create_time INTEGER, type INTEGER, feed_id TEXT, from_username TEXT, from_nickname TEXT, content TEXT, comment_id TEXT, comment64_id TEXT, comment_flag TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-1', 'wxid-author', '<TimelineObject><id>native-feed</id><username>wxid-author</username><createTime>1760000000</createTime><contentDesc>正文</contentDesc></TimelineObject>', b''))
            conn.execute('INSERT INTO SnsMessage_tmp3 VALUES(?,?,?,?,?,?,?,?,?,?)', (1, 1760000001, 2, 'feed-1', 'wxid-actor', 'Actor', '真实评论', '99', '100', 'flag'))
            conn.execute('INSERT INTO SnsMessage_tmp3 VALUES(?,?,?,?,?,?,?,?,?,?)', (2, 1760000002, 9, 'missing-feed', 'wxid-x', 'X', '其它类型', '101', '102', 'flag'))
            conn.commit(); conn.close()
            importer = MomentsImporter(db, account_id='acct-a')
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            importer.import_to_store(repo)
            with repo.store.connect() as dbconn:
                rows = list(dbconn.execute('SELECT moment_id, interaction_type, actor_id, actor_name, text, metadata_json FROM moment_interactions ORDER BY timestamp'))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['interaction_type'], 'comment')
                self.assertEqual(rows[0]['actor_id'], 'wxid-actor')
                self.assertEqual(rows[0]['actor_name'], 'Actor')
                self.assertEqual(rows[0]['text'], '真实评论')
                self.assertNotIn('99', rows[0]['text'])
                self.assertEqual(rows[1]['interaction_type'], '9')
                self.assertIn('"orphan": true', rows[1]['metadata_json'])
            self.assertEqual(importer.last_report['orphan_interactions'], 1)

    def test_auxiliary_moment_import_prune_keeps_new_interaction_citations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            account_dir = root / 'acct'
            account_dir.mkdir()
            db = account_dir / 'sns.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMessage_tmp3(local_id INTEGER, create_time INTEGER, type INTEGER, feed_id TEXT, from_username TEXT, from_nickname TEXT, content TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-1', 'wxid-author', '<TimelineObject><id>native-feed</id><username>wxid-author</username><createTime>1760000000</createTime><contentDesc>正文</contentDesc></TimelineObject>', b''))
            conn.execute('INSERT INTO SnsMessage_tmp3 VALUES(?,?,?,?,?,?,?)', (1, 1760000001, 2, 'feed-1', 'wxid-actor', 'Actor', '真实评论'))
            conn.commit(); conn.close()
            store = SQLiteStore(root / 'vault.sqlite')
            repo = MultimodalRepository(store)
            import_auxiliary_sources(account_dir, account_id='acct-a', store=store, repo=repo, only={'moment'})
            with store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM moment_interactions').fetchone()[0], 1)

    def test_auxiliary_moment_import_prunes_stale_moment_media_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            account_dir = root / 'acct'
            account_dir.mkdir()
            db = account_dir / 'sns.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            with_media = '<TimelineObject><id>native-media-prune</id><username>wxid-author</username><createTime>1760000000</createTime><contentDesc>带图</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>https://cdn.invalid/prune.jpg</url></media></mediaList></ContentObject></TimelineObject>'
            without_media = '<TimelineObject><id>native-media-prune</id><username>wxid-author</username><createTime>1760000001</createTime><contentDesc>无图</contentDesc></TimelineObject>'
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-media-prune', 'wxid-author', with_media, b''))
            conn.commit(); conn.close()

            store = SQLiteStore(root / 'vault.sqlite')
            repo = MultimodalRepository(store)
            import_auxiliary_sources(account_dir, account_id='acct-a', store=store, repo=repo, only={'moment'})
            with store.connect() as dbconn:
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_assets WHERE source_type='moment'").fetchone()[0], 1)
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_asset_links WHERE source_type='moment'").fetchone()[0], 1)

            conn = sqlite3.connect(db)
            conn.execute('UPDATE SnsTimeLine SET content=?', (without_media,))
            conn.commit(); conn.close()
            import_auxiliary_sources(account_dir, account_id='acct-a', store=store, repo=repo, only={'moment'})
            with store.connect() as dbconn:
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_assets WHERE source_type='moment'").fetchone()[0], 0)
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_asset_links WHERE source_type='moment'").fetchone()[0], 0)

    def test_auxiliary_moment_import_keeps_cross_account_native_ids_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            xml_template = '<TimelineObject><id>shared-native-media</id><username>wxid-author</username><createTime>1760000000</createTime><contentDesc>共享</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>https://cdn.invalid/{name}.jpg</url></media></mediaList></ContentObject></TimelineObject>'
            store = SQLiteStore(root / 'vault.sqlite')
            repo = MultimodalRepository(store)
            for account_id in ('acct-a', 'acct-b'):
                account_dir = root / account_id
                account_dir.mkdir()
                db = account_dir / 'sns.db'
                conn = sqlite3.connect(db)
                conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
                conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-shared', 'wxid-author', xml_template.format(name=account_id), b''))
                conn.commit(); conn.close()
                import_auxiliary_sources(account_dir, account_id=account_id, store=store, repo=repo, only={'moment'})
            import_auxiliary_sources(root / 'acct-a', account_id='acct-a', store=store, repo=repo, only={'moment'})

            with store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM moment_items').fetchone()[0], 2)
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_asset_links WHERE source_type='moment'").fetchone()[0], 2)
                self.assertEqual(dbconn.execute("SELECT COUNT(*) FROM media_assets WHERE source_type='moment'").fetchone()[0], 2)
                rows = list(dbconn.execute('SELECT account_id, moment_id FROM moment_items ORDER BY account_id'))
                self.assertEqual([row['account_id'] for row in rows], ['acct-a', 'acct-b'])
                self.assertNotEqual(rows[0]['moment_id'], rows[1]['moment_id'])

    def test_cross_account_interactions_follow_parent_account(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = SQLiteStore(root / 'vault.sqlite')
            repo = MultimodalRepository(store)
            for account_id in ('acct-a', 'acct-b'):
                db = root / f'{account_id}.db'
                conn = sqlite3.connect(db)
                conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
                conn.execute('CREATE TABLE SnsMessage_tmp3(local_id INTEGER, create_time INTEGER, type INTEGER, feed_id TEXT, from_username TEXT, from_nickname TEXT, content TEXT)')
                conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-shared', 'wxid-author', '<TimelineObject><id>shared-native</id><username>wxid-author</username><createTime>1760000000</createTime><contentDesc>正文</contentDesc></TimelineObject>', b''))
                conn.execute('INSERT INTO SnsMessage_tmp3 VALUES(?,?,?,?,?,?,?)', (1, 1760000001, 2, 'feed-shared', 'wxid-actor', 'Actor', '评论'))
                conn.commit(); conn.close()
                MomentsImporter(db, account_id=account_id).import_to_store(repo)

            with store.connect() as dbconn:
                moments = list(dbconn.execute('SELECT account_id, moment_id FROM moment_items ORDER BY account_id'))
                interactions = list(dbconn.execute('SELECT mi.account_id, mi.moment_id, m.account_id AS parent_account_id FROM moment_interactions mi JOIN moment_items m ON mi.moment_id=m.moment_id ORDER BY mi.account_id'))
            self.assertEqual(len(moments), 2)
            self.assertEqual(len(interactions), 2)
            self.assertEqual([row['account_id'] for row in moments], ['acct-a', 'acct-b'])
            self.assertEqual([(row['account_id'], row['parent_account_id']) for row in interactions], [('acct-a', 'acct-a'), ('acct-b', 'acct-b')])
            self.assertNotEqual(interactions[0]['moment_id'], interactions[1]['moment_id'])

    def test_repository_rejects_cross_account_parent_interaction(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))
            repo.insert_moment_item(moment_id='moment-scoped', account_id='acct-a', citation='trove://wechat/acct-a/moment/moment-scoped')
            with self.assertRaises(ValueError):
                repo.insert_moment_interaction(
                    interaction_id='interaction-bad',
                    moment_id='moment-scoped',
                    account_id='acct-b',
                    citation='trove://wechat/acct-b/moment/moment-scoped/interaction/interaction-bad',
                    interaction_type='comment',
                )


    def test_duplicate_native_id_uses_latest_then_longest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'sns.db'
            xml_old = '<TimelineObject><id>dup-native</id><username>wxid-a</username><createTime>1700000000</createTime><contentDesc>短</contentDesc></TimelineObject>'
            xml_new = '<TimelineObject><id>dup-native</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>更长的正文</contentDesc></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-old', 'wxid-a', xml_old, b''))
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-new', 'wxid-a', xml_new, b''))
            conn.commit(); conn.close()
            importer = MomentsImporter(db, account_id='acct-a')
            rows = importer.load()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].text, '更长的正文')
            self.assertEqual(importer.last_report['imported_moments'], 1)
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            importer.import_to_store(repo)
            with repo.store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM moment_items').fetchone()[0], 1)
                self.assertEqual(dbconn.execute('SELECT text FROM moment_items').fetchone()[0], '更长的正文')


    def test_moment_media_fetch_uses_local_sns_cache_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            account_dir = vault / 'sources' / 'acct'
            account_dir.mkdir(parents=True)
            db = account_dir / 'sns.db'
            url = 'https://cdn.invalid/cached-image.jpg'
            cache_key = 'ab' + hashlib.sha256(b'wechat-sns-cache-key').hexdigest()[:30]
            cache_dir = account_dir / 'cache' / '2026-01' / 'sns' / 'img' / cache_key[:2]
            cache_dir.mkdir(parents=True)
            # 1x1 PNG
            (cache_dir / cache_key[2:]).write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8z8AARQABywH6f6n6AAAAAElFTkSuQmCC'))
            xml = f'<TimelineObject><id>native-cache</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>带图</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>{url}</url></media></mediaList></ContentObject></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMediaCache(feed_id TEXT, media_id TEXT, cache_key TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-cache', 'wxid-a', xml, b''))
            conn.execute('INSERT INTO SnsMediaCache VALUES(?,?,?)', ('native-cache', 'm1', cache_key))
            conn.commit(); conn.close()
            repo = MultimodalRepository(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            importer = MomentsImporter(db, account_id='acct-a')
            importer.import_to_store(repo)
            with repo.store.connect() as dbconn:
                asset = dbconn.execute("SELECT citation, cache_state, path_ref FROM media_assets WHERE source_type='moment'").fetchone()
                self.assertEqual(asset['cache_state'], 'source_available')
                self.assertFalse(asset['path_ref'])
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM sns_cache_mappings').fetchone()[0], 1)
                citation = asset['citation']
            self.assertEqual(importer.last_report['sns_cache_mapping_status'], 'mapped')
            self.assertEqual(importer.last_report['sns_cache_d0_mapping_conclusion']['readable_sns_db'], 'no_mapping')
            self.assertIn('media_0.kvdb', importer.last_report['sns_cache_d0_mapping_conclusion']['encrypted_candidates_not_inspected'])
            self.assertGreater(importer.last_report['sns_cache_cached_conversion_rate'], 0)
            fetched = fetch_media(vault, citation)
            self.assertTrue(fetched['ok'], fetched)
            self.assertEqual(fetched['status'], 'available')
            self.assertFalse(fetched['raw_content_included'])

    def test_moment_media_does_not_guess_url_md5_cache_key_without_sns_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            account_dir = vault / 'sources' / 'acct'
            account_dir.mkdir(parents=True)
            db = account_dir / 'sns.db'
            url = 'https://cdn.invalid/legacy-md5-guess.jpg'
            md5 = hashlib.md5(url.encode('utf-8')).hexdigest()
            cache_dir = account_dir / 'cache' / '2026-01' / 'sns' / 'img' / md5[:2]
            cache_dir.mkdir(parents=True)
            (cache_dir / md5[2:]).write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8z8AARQABywH6f6n6AAAAAElFTkSuQmCC'))
            xml = f'<TimelineObject><id>native-inventory</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>带图</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>{url}</url></media></mediaList></ContentObject></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-inventory', 'wxid-a', xml, b''))
            conn.commit(); conn.close()
            repo = MultimodalRepository(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            importer = MomentsImporter(db, account_id='acct-a')
            importer.import_to_store(repo)
            with repo.store.connect() as dbconn:
                asset = dbconn.execute("SELECT citation, cache_state, path_ref FROM media_assets WHERE source_type='moment'").fetchone()
                self.assertEqual(asset['cache_state'], 'inventory_only')
                self.assertFalse(asset['path_ref'])
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM sns_cache_mappings').fetchone()[0], 0)
            self.assertEqual(importer.last_report['sns_cache_mapping_status'], 'inventory_only')
            self.assertEqual(importer.last_report['sns_cache_d0_mapping_conclusion']['cache_files'], 'no_embedded_mapping')
            self.assertEqual(importer.last_report['sns_cache_cached_conversion_rate'], 0)

    def test_moment_media_fetch_reports_missing_cache_structurally(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            account_dir = vault / 'sources' / 'acct'
            account_dir.mkdir(parents=True)
            db = account_dir / 'sns.db'
            xml = '<TimelineObject><id>native-missing</id><username>wxid-a</username><createTime>1760000000</createTime><contentDesc>缺图</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type><url>https://cdn.invalid/missing.jpg</url></media></mediaList></ContentObject></TimelineObject>'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-missing', 'wxid-a', xml, b''))
            conn.commit(); conn.close()
            repo = MultimodalRepository(SQLiteStore(vault / 'index' / 'trove.sqlite'))
            importer = MomentsImporter(db, account_id='acct-a')
            importer.import_to_store(repo)
            with repo.store.connect() as dbconn:
                citation = dbconn.execute("SELECT citation FROM media_assets WHERE source_type='moment'").fetchone()['citation']
            fetched = fetch_media(vault, citation)
            self.assertFalse(fetched['ok'])
            self.assertEqual(fetched['reason'], 'remote_fetch_approval_required')
            self.assertEqual(fetched['code'], 'approval_required')

    def test_moment_media_lazy_workflow_surfaces_search_and_profile_hints_without_caption_precompute(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            account_dir = vault / 'sources' / 'acct'
            account_dir.mkdir(parents=True)
            db = account_dir / 'sns.db'
            url = 'https://cdn.invalid/lazy-workflow.jpg'
            cache_key = 'cd' + hashlib.sha256(b'lazy-workflow-cache-key').hexdigest()[:30]
            cache_dir = account_dir / 'cache' / '2026-01' / 'sns' / 'img' / cache_key[:2]
            cache_dir.mkdir(parents=True)
            (cache_dir / cache_key[2:]).write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8z8AARQABywH6f6n6AAAAAElFTkSuQmCC'))
            xml = f'''<TimelineObject>
              <id>native-lazy-workflow</id><username>wxid-author</username><createTime>1760000000</createTime>
              <contentDesc>示例教育 momenthintneedle 带图更新</contentDesc>
              <ContentObject><mediaList>
                <media><id>m1</id><type>2</type><url>{url}</url></media>
                <media><id>m2</id><type>2</type><url>https://cdn.invalid/not-in-cache.jpg</url></media>
              </mediaList></ContentObject>
            </TimelineObject>'''
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE SnsTimeLine(tid TEXT, user_name TEXT, content TEXT, pack_info_buf BLOB)')
            conn.execute('CREATE TABLE SnsMediaCache(feed_id TEXT, media_id TEXT, cache_key TEXT)')
            conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed-lazy', 'wxid-author', xml, b''))
            conn.execute('INSERT INTO SnsMediaCache VALUES(?,?,?)', ('native-lazy-workflow', 'm1', cache_key))
            conn.commit(); conn.close()

            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            repo.upsert_entity(EntityRecord(entity_id='customer-1', entity_type='Customer', display_name='示例教育', identifiers={'wechat_id': 'wxid-author'}))
            importer = MomentsImporter(db, account_id='acct-a')
            importer.import_to_store(repo)
            store.rebuild_evidence_chunks_for_source_types(['moment'])

            default_response = build_search_engine(cfg).search(SearchRequest('momenthintneedle', limit=1, include_vector=False, semantic='off')).to_dict()
            self.assertIsNone(default_response['results'][0]['media_hint'])

            response = build_search_engine(cfg).search(SearchRequest('momenthintneedle', limit=1, include_vector=False, semantic='off', include_media_hints=True)).to_dict()
            self.assertEqual(response['total'], 1)
            hint = response['results'][0]['media_hint']
            self.assertEqual(hint['image_count'], 2)
            self.assertEqual(hint['available_count'], 1)
            self.assertFalse(hint['raw_paths_included'])
            self.assertEqual({item['cache_state'] for item in hint['items']}, {'source_available', 'inventory_only'})

            profile = build_customer_profile(store, '示例教育', limit=3)
            self.assertTrue(profile['sections']['moments'])
            self.assertEqual(profile['sections']['moments'][0]['media_hint']['image_count'], 2)
            with store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM image_observations').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
