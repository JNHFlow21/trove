from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore


class MediaAssetDedupTests(unittest.TestCase):
    def test_null_content_hash_does_not_merge_distinct_image_citations(self):
        with tempfile.TemporaryDirectory() as d:
            repo = MultimodalRepository(SQLiteStore(Path(d) / 'vault.sqlite'))

            first = repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image-1',
                account_id='acct-a',
                source_type='message',
                source_id='shared-source',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv/s/1#image',
                content_hash=None,
            ))
            second = repo.upsert_media_asset(MediaAssetRecord(
                asset_id='asset-image-2',
                account_id='acct-a',
                source_type='message',
                source_id='shared-source',
                modality='image',
                media_type='image',
                citation='trove://wechat/acct-a/conv/s/2#image',
                content_hash=None,
            ))

            self.assertEqual(first['asset_id'], 'asset-image-1')
            self.assertEqual(second['asset_id'], 'asset-image-2')
            with repo.store.connect() as conn:
                count = conn.execute('SELECT COUNT(*) AS n FROM media_assets').fetchone()['n']
            self.assertEqual(count, 2)


if __name__ == '__main__':
    unittest.main()
