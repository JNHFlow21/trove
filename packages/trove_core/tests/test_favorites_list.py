from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers import favorites as favorite_handlers
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault


def _favorite(favorite_id: str, epoch: str, **overrides):
    row = {
        'favorite_id': favorite_id,
        'account_id': 'acct-work',
        'citation': f'trove://wechat/acct-work/favorite/{favorite_id}',
        'timestamp': epoch,
        'title': '',
        'text': f'收藏内容 {favorite_id}',
        'metadata': {'table': 'fav_db_item', 'rowid': 7},
    }
    row.update(overrides)
    return row


class _FavoritesVaultCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_vault(self.vault, reset=True)
        self.config = VaultConfig.resolve(str(self.vault))
        favorite_handlers._reset_cursor_store_for_tests()
        self.repo = MultimodalRepository(SQLiteStore(self.config.paths.sqlite_path))

    def tearDown(self) -> None:
        self.repo.store.close_all()
        self.temp.cleanup()

    def seed(self, favorites):
        self.repo.upsert_favorite_batch(favorites)
        self.repo.store.close_all()

    def dispatcher(self):
        return build_default_dispatcher(self.vault)

    def _seed_defaults(self):
        self.seed([
            _favorite('fav-0001', '1000001', title='报价', text='第一批报价单'),
            _favorite('fav-0002', '1000002', text='长' * 400),
            _favorite('fav-0003', '1000003', text='含链接 https://example.invalid/x'),
            _favorite('fav-0004', '1000004', media_refs=[{'field': 'image', 'state': 'metadata_only'}]),
            _favorite('fav-0005', '1000005', title='百分号 100% 进度', metadata={}),
        ])


class FavoritesListTests(_FavoritesVaultCase):
    def test_list_orders_desc_and_shapes_items(self):
        self._seed_defaults()
        response = self.dispatcher().dispatch('trove.favorites_list', {}, request_id='req-list')
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['matched_total'], 5)
        citations = [item['citation'] for item in data['favorites']]
        self.assertEqual(
            citations,
            [f'trove://wechat/acct-work/favorite/fav-000{i}' for i in (5, 4, 3, 2, 1)],
        )
        self.assertEqual(response['coverage'], {'state': 'complete', 'returned': 5, 'remaining': 0})
        self.assertEqual(response['page'], {'has_more': False})
        media_item = data['favorites'][1]
        self.assertEqual(media_item['kind'], 'media')
        self.assertEqual(media_item['media_count'], 1)
        note_item = data['favorites'][2]
        self.assertEqual(note_item['kind'], 'note')
        self.assertEqual(note_item['source'], {'table': 'fav_db_item', 'rowid': 7})
        self.assertEqual(data['favorites'][0]['source'], None)
        self.assertTrue(all(item['trust'] == 'untrusted_evidence' for item in data['favorites']))
        self.assertTrue(all(item['account_id'] == 'acct-work' for item in data['favorites']))

    def test_list_truncates_text_and_title(self):
        self._seed_defaults()
        response = self.dispatcher().dispatch(
            'trove.favorites_list', {'limit': 5}, request_id='req-excerpt',
        )
        self.assertTrue(response['ok'])
        items = {item['citation']: item for item in response['data']['favorites']}
        long_item = items['trove://wechat/acct-work/favorite/fav-0002']
        self.assertTrue(long_item['text_truncated'])
        self.assertLessEqual(len(long_item['text']), 240)
        short_item = items['trove://wechat/acct-work/favorite/fav-0001']
        self.assertFalse(short_item['text_truncated'])
        self.assertEqual(short_item['title'], '报价')
        self.assertFalse(short_item['title_truncated'])

    def test_keyword_filter_matches_text_and_escapes_wildcards(self):
        self._seed_defaults()
        response = self.dispatcher().dispatch(
            'trove.favorites_list', {'keyword': '报价'}, request_id='req-keyword',
        )
        self.assertTrue(response['ok'])
        citations = [item['citation'] for item in response['data']['favorites']]
        self.assertEqual(citations, ['trove://wechat/acct-work/favorite/fav-0001'])
        self.assertEqual(response['data']['matched_total'], 1)
        wildcard = self.dispatcher().dispatch(
            'trove.favorites_list', {'keyword': '100%'}, request_id='req-wildcard',
        )
        self.assertTrue(wildcard['ok'])
        citations = [item['citation'] for item in wildcard['data']['favorites']]
        self.assertEqual(citations, ['trove://wechat/acct-work/favorite/fav-0005'])
        missing = self.dispatcher().dispatch(
            'trove.favorites_list', {'keyword': '不存在'}, request_id='req-keyword-miss',
        )
        self.assertTrue(missing['ok'])
        self.assertEqual(missing['data']['matched_total'], 0)
        self.assertEqual(missing['coverage']['state'], 'complete')

    def test_kind_filter_selects_derived_class(self):
        self._seed_defaults()
        media = self.dispatcher().dispatch(
            'trove.favorites_list', {'kind': 'media'}, request_id='req-kind-media',
        )
        self.assertTrue(media['ok'])
        self.assertEqual(
            [item['citation'] for item in media['data']['favorites']],
            ['trove://wechat/acct-work/favorite/fav-0004'],
        )
        notes = self.dispatcher().dispatch(
            'trove.favorites_list', {'kind': 'note'}, request_id='req-kind-note',
        )
        self.assertEqual(notes['data']['matched_total'], 4)
        self.assertTrue(all(item['kind'] == 'note' for item in notes['data']['favorites']))

    def test_time_range_accepts_iso_and_epoch_bounds(self):
        self._seed_defaults()
        expected = ['trove://wechat/acct-work/favorite/fav-0004', 'trove://wechat/acct-work/favorite/fav-0003']
        iso = self.dispatcher().dispatch(
            'trove.favorites_list',
            {'since': '1970-01-12T13:46:43Z', 'until': '1970-01-12T13:46:45Z'},
            request_id='req-range-iso',
        )
        self.assertTrue(iso['ok'], iso.get('error'))
        self.assertEqual([item['citation'] for item in iso['data']['favorites']], expected)
        self.assertEqual(iso['data']['scope']['since'], '1000003')
        epoch = self.dispatcher().dispatch(
            'trove.favorites_list', {'since': '1000003', 'until': '1000005'},
            request_id='req-range-epoch',
        )
        self.assertEqual([item['citation'] for item in epoch['data']['favorites']], expected)

    def test_invalid_time_bound_fails_closed(self):
        self._seed_defaults()
        response = self.dispatcher().dispatch(
            'trove.favorites_list', {'since': '去年'}, request_id='req-bad-time',
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'invalid_request')

    def test_cursor_paginates_without_overlap(self):
        self._seed_defaults()
        dispatcher = self.dispatcher()
        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            payload = {'limit': 2}
            if cursor:
                payload['cursor'] = cursor
            response = dispatcher.dispatch('trove.favorites_list', payload, request_id=f'req-page-{pages}')
            self.assertTrue(response['ok'], response.get('error'))
            seen.extend(item['citation'] for item in response['data']['favorites'])
            pages += 1
            cursor = (response.get('page') or {}).get('next_cursor')
            if not response['page']['has_more']:
                self.assertEqual(response['coverage']['state'], 'complete')
                self.assertEqual(response['coverage']['remaining'], 0)
                break
            self.assertEqual(response['coverage']['state'], 'partial')
            self.assertIsNotNone(cursor)
        self.assertEqual(pages, 3)
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(seen), len(set(seen)))

    def test_cursor_is_bound_to_filters_and_handle(self):
        self._seed_defaults()
        dispatcher = self.dispatcher()
        first = dispatcher.dispatch('trove.favorites_list', {'limit': 2}, request_id='req-bind-1')
        cursor = first['page']['next_cursor']
        changed = dispatcher.dispatch(
            'trove.favorites_list',
            {'limit': 2, 'cursor': cursor, 'keyword': '报价'},
            request_id='req-bind-2',
        )
        self.assertEqual(changed['error']['code'], 'cursor_mismatch')
        garbage = dispatcher.dispatch(
            'trove.favorites_list', {'limit': 2, 'cursor': 'x' * 43}, request_id='req-bind-3',
        )
        self.assertEqual(garbage['error']['code'], 'cursor_invalid')

    def test_limit_bounds_are_enforced(self):
        self._seed_defaults()
        dispatcher = self.dispatcher()
        for limit in (0, 101):
            response = dispatcher.dispatch(
                'trove.favorites_list', {'limit': limit}, request_id=f'req-limit-{limit}',
            )
            self.assertFalse(response['ok'])
            self.assertEqual(response['error']['code'], 'invalid_request')

    def test_account_scope_narrows_results(self):
        self.seed([
            _favorite('fav-work', '1000001'),
            _favorite('fav-personal', '1000002', account_id='acct-personal',
                      citation='trove://wechat/acct-personal/favorite/fav-personal'),
        ])
        scoped = self.dispatcher().dispatch(
            'trove.favorites_list', {'account_id': 'acct-personal'}, request_id='req-scope',
        )
        self.assertTrue(scoped['ok'])
        self.assertEqual(scoped['data']['matched_total'], 1)
        self.assertEqual(
            scoped['data']['favorites'][0]['citation'],
            'trove://wechat/acct-personal/favorite/fav-personal',
        )

    def test_favorites_queries_use_bounded_indexes(self):
        self._seed_defaults()
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    (
                        'SELECT favorite_id FROM favorites ORDER BY timestamp DESC, favorite_id DESC LIMIT ?',
                        (50,),
                    ),
                    (
                        'SELECT favorite_id FROM favorites WHERE timestamp>=? AND timestamp<?'
                        ' ORDER BY timestamp DESC, favorite_id DESC LIMIT ?',
                        ('1000002', '1000005', 50),
                    ),
                    (
                        'SELECT COUNT(*) FROM favorites WHERE timestamp>=?',
                        ('1000002',),
                    ),
                ):
                    plans.extend(
                        str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params)
                    )
        finally:
            store.close()
        plan_text = ' | '.join(plans)
        self.assertIn('idx_favorites_time', plan_text)
        self.assertNotIn('TEMP B-TREE', plan_text)

    def test_missing_vault_returns_bounded_empty_result(self):
        missing = Path(self.temp.name) / 'missing-vault'
        config = VaultConfig.resolve(str(missing), env={})
        response = favorite_handlers.favorites_list(config, {})
        self.assertTrue(response.ok)
        self.assertEqual(response.data['favorites'], [])
        self.assertEqual(response.coverage['state'], 'complete')

        via_dispatcher = build_default_dispatcher(missing).dispatch(
            'trove.favorites_list', {}, request_id='req-empty',
        )
        self.assertTrue(via_dispatcher['ok'])
        self.assertEqual(via_dispatcher['data']['matched_total'], 0)
        self.assertEqual(via_dispatcher['coverage']['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
