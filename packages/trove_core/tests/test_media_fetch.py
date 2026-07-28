from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from trove_core.media_fetch import fetch_media
from trove_core.runtime import build_search_engine
from trove_core.search.context import ContextService
from trove_core.search.query import SearchRequest
from trove_core.store.repositories import MediaAssetLinkRecord, MediaAssetRecord, MediaUnderstandingRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='
)


class MediaFetchTests(unittest.TestCase):
    def test_search_image_message_returns_media_hint_and_fetches_vault_preview(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'poster.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(
                [Account('acct-a', 'A', 'A')],
                [Conversation('conv-a', 'acct-a', '私聊', 'private')],
                [Message('acct-a', 'A', 'conv-a', '私聊', 'private', 'u1', '客户', datetime(2026, 1, 1, tzinfo=timezone.utc), '请看这张图片 mediafetchneedle', 'message_0', 1)],
            )
            repo = MultimodalRepository(store)
            citation = 'trove://wechat/acct-a/conv-a/message_0/1'
            repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image',
                account_id='acct-a',
                source_type='message',
                source_id='msg-1',
                modality='image',
                media_type='image',
                citation=citation,
                path_ref='sources/poster.png',
                cache_state='cached',
                metadata={'message_citation': citation},
            ))
            repo.upsert_media_asset_link(MediaAssetLinkRecord('link-image', 'asset-image', 'acct-a', 'message', citation, 'private_chat', True, 'accepted'))

            default_response = build_search_engine(cfg).search(SearchRequest('mediafetchneedle', limit=1, semantic='off')).to_dict()
            self.assertIsNone(default_response['results'][0]['media_hint'])

            response = build_search_engine(cfg).search(SearchRequest('mediafetchneedle', limit=1, semantic='off', include_media_hints=True)).to_dict()
            self.assertEqual(response['results'][0]['media_hint']['type'], 'image')
            self.assertFalse(response['results'][0]['media_hint']['raw_paths_included'])

            fetched = fetch_media(vault, response['results'][0]['citation'])
            self.assertTrue(fetched['ok'])
            self.assertEqual(fetched['mime'], 'image/png')
            self.assertEqual((fetched['width'], fetched['height']), (1, 1))
            self.assertTrue(Path(fetched['path']).is_file())
            self.assertTrue(Path(fetched['path']).resolve().is_relative_to(vault.resolve()))

            context = ContextService(store).fetch(citation)
            self.assertEqual(context['messages'][0]['media_hint']['type'], 'image')

    def test_fetch_returns_understanding_by_content_hash_for_shared_assets(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            image = vault / 'sources' / 'shared.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(PNG_1X1)
            content_sha = hashlib.sha256(PNG_1X1).hexdigest()
            store = SQLiteStore(cfg.paths.sqlite_path)
            repo = MultimodalRepository(store)
            citations = [
                'trove://wechat/acct-a/conv-a/message_0/1#image-0',
                'trove://wechat/acct-a/moment/moment-1#image-0',
            ]
            for i, citation in enumerate(citations):
                repo.upsert_media_asset(MediaAssetRecord(
                    asset_id=f'asset-shared-{i}',
                    account_id='acct-a',
                    source_type='moment' if 'moment' in citation else 'message',
                    source_id=f'source-{i}',
                    modality='image',
                    media_type='image',
                    citation=citation,
                    path_ref='sources/shared.png',
                    cache_state='cached',
                    metadata={'message_citation': citation},
                ))
            repo.upsert_media_understanding(MediaUnderstandingRecord(
                content_sha256=content_sha,
                modality='image',
                caption='fixture caption',
                visible_text='fixture text',
                objects=[{'label': 'fixture-object'}],
                business_signals=[{'type': 'fixture-signal'}],
                model_id='agent-fixture-v1',
                prompt_version='p1',
                confidence=0.91,
                source_citations=[citations[0]],
            ))

            first = fetch_media(vault, citations[0])
            second = fetch_media(vault, citations[1])
            self.assertTrue(first['ok'], first)
            self.assertTrue(second['ok'], second)
            self.assertEqual(first['content_sha256'], content_sha)
            self.assertEqual(second['content_sha256'], content_sha)
            self.assertEqual(first['understanding']['caption'], 'fixture caption')
            self.assertEqual(second['understanding']['model_id'], 'agent-fixture-v1')
            self.assertEqual(second['understanding']['objects'][0]['label'], 'fixture-object')
            self.assertNotIn('source_citations', second['understanding'])
            self.assertEqual(second['understanding']['source_citations_count'], 2)
            with store.connect() as conn:
                row = conn.execute('SELECT source_citations_json FROM media_understanding WHERE content_sha256=?', (content_sha,)).fetchone()
                self.assertEqual(set(json.loads(row['source_citations_json'])), set(citations))
                projections = list(conn.execute(
                    """SELECT citation,status FROM image_observations
                         WHERE content_sha256=? ORDER BY citation""",
                    (content_sha,),
                ))
                self.assertEqual(
                    [(row['citation'], row['status']) for row in projections],
                    [(citation, 'active') for citation in sorted(citations)],
                )

    def test_missing_image_returns_media_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d) / 'vault'
            VaultConfig.resolve(str(vault), env={}).ensure()
            result = fetch_media(vault, 'trove://wechat/acct-a/conv-a/message_0/404')
            self.assertEqual(result['code'], 'media_unavailable')
            self.assertIsNone(result['understanding'])


if __name__ == '__main__':
    unittest.main()
