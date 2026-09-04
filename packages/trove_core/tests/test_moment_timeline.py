from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.application.handlers import moments as moment_handlers
from trove_core.store.repositories import EntityRecord, MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault


def _moment(moment_id: str, author: str, day: int, **overrides):
    row = {
        'moment_id': moment_id,
        'account_id': 'acct-work',
        'citation': f'trove://wechat/acct-work/moment/{moment_id}',
        'author_id': author,
        'timestamp': f'2026-06-{day:02d}T08:00:00Z',
        'text': f'动态 {moment_id}',
    }
    row.update(overrides)
    return row


def _interaction(interaction_id: str, moment_id: str, actor_id: str, *, kind: str = 'like', **overrides):
    row = {
        'interaction_id': interaction_id,
        'moment_id': moment_id,
        'account_id': 'acct-work',
        'citation': f'trove://wechat/acct-work/moment/{moment_id}/interaction/{interaction_id}',
        'interaction_type': kind,
        'actor_id': actor_id,
        'actor_name': '',
        'timestamp': '2026-06-20T09:00:00Z',
    }
    row.update(overrides)
    return row


class _MomentVaultCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.vault = Path(self.temp.name) / 'vault'
        index_fixture_vault(self.vault, reset=True)
        self.config = VaultConfig.resolve(str(self.vault))
        moment_handlers._reset_cursor_store_for_tests()
        self.repo = MultimodalRepository(SQLiteStore(self.config.paths.sqlite_path))

    def tearDown(self) -> None:
        self.repo.store.close_all()
        self.temp.cleanup()

    def seed(self, moments, interactions=(), entities=()):
        for entity in entities:
            self.repo.upsert_entity(entity)
        self.repo.upsert_moment_batch(moments, interactions)
        self.repo.store.close_all()

    def dispatcher(self):
        return build_default_dispatcher(self.vault)


class MomentTimelineTests(_MomentVaultCase):
    def _seed_author(self):
        self.seed(
            [
                _moment('moment-0001', 'wxid-author-a', 11, text='短文'),
                _moment('moment-0002', 'wxid-author-a', 12, text='长' * 400, media_refs=[{'idx': 0}, {'idx': 1}]),
                _moment('moment-0003', 'wxid-author-a', 13),
                _moment('moment-0004', 'wxid-author-a', 14),
                _moment('moment-0005', 'wxid-author-a', 15, link={'title': '链接标题', 'url': 'https://example.invalid/x'}),
                _moment('moment-deleted', 'wxid-author-a', 16, text='不可见', status='deleted'),
                _moment('moment-other', 'wxid-author-b', 12),
            ],
            [
                _interaction('int-like', 'moment-0003', 'wxid-fan-1', kind='like'),
                _interaction('int-comment', 'moment-0003', 'wxid-fan-2', kind='comment', text='写得真好'),
            ],
            [
                EntityRecord(
                    entity_id='ent-author-a', entity_type='Person', display_name='阿May',
                    identifiers={'nickname': '阿May', 'user_id': 'wxid-author-a'},
                ),
            ],
        )

    def test_timeline_resolves_author_name_and_orders_desc(self):
        self._seed_author()
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': '阿May'}, request_id='req-timeline',
        )
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['scope']['author_id'], 'wxid-author-a')
        self.assertEqual(data['matched_total'], 5)
        citations = [item['citation'] for item in data['moments']]
        self.assertEqual(
            citations,
            [f'trove://wechat/acct-work/moment/moment-000{i}' for i in (5, 4, 3, 2, 1)],
        )
        self.assertEqual(response['coverage']['state'], 'complete')
        self.assertEqual(response['page'], {'has_more': False})
        linked = data['moments'][0]
        self.assertEqual(linked['link'], {'title': '链接标题', 'url': 'https://example.invalid/x'})
        interacted = data['moments'][2]
        self.assertEqual(interacted['interactions'], {'likes': 1, 'comments': 1, 'total': 2})
        self.assertTrue(all(item['trust'] == 'untrusted_evidence' for item in data['moments']))

    def test_timeline_truncates_text_and_counts_media(self):
        self._seed_author()
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': 'wxid-author-a', 'limit': 5}, request_id='req-excerpt',
        )
        self.assertTrue(response['ok'])
        items = {item['citation']: item for item in response['data']['moments']}
        long_item = items['trove://wechat/acct-work/moment/moment-0002']
        self.assertTrue(long_item['text_truncated'])
        self.assertLessEqual(len(long_item['text']), 240)
        self.assertEqual(long_item['media_count'], 2)
        short_item = items['trove://wechat/acct-work/moment/moment-0001']
        self.assertFalse(short_item['text_truncated'])
        self.assertEqual(short_item['media_count'], 0)

    def test_timeline_exact_author_id_needs_no_entity(self):
        self.seed([_moment('moment-solo', 'wxid-no-entity', 11)])
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': 'wxid-no-entity'}, request_id='req-exact',
        )
        self.assertTrue(response['ok'])
        self.assertEqual(response['data']['matched_total'], 1)

    def test_timeline_unknown_target_is_typed_no_results(self):
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': '不存在的人'}, request_id='req-missing',
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'no_results')
        self.assertEqual(response['error']['details']['candidates'], [])

    def test_timeline_ambiguous_author_returns_candidates_without_picking(self):
        self.seed(
            [
                _moment('moment-a1', 'wxid-shared-1', 11),
                _moment('moment-a2', 'wxid-shared-2', 12),
            ],
            entities=[
                EntityRecord(
                    entity_id='ent-shared-1', entity_type='Person', display_name='一',
                    identifiers={'nickname': '重名', 'user_id': 'wxid-shared-1'},
                ),
                EntityRecord(
                    entity_id='ent-shared-2', entity_type='Person', display_name='二',
                    identifiers={'nickname': '重名', 'user_id': 'wxid-shared-2'},
                ),
            ],
        )
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': '重名'}, request_id='req-ambiguous',
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'ambiguous_target')
        candidates = response['error']['details']['candidates']
        self.assertEqual({item['id'] for item in candidates}, {'wxid-shared-1', 'wxid-shared-2'})
        resolved = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': 'wxid-shared-2'}, request_id='req-pinned',
        )
        self.assertTrue(resolved['ok'])
        self.assertEqual(resolved['data']['scope']['author_id'], 'wxid-shared-2')

    def test_timeline_resolves_contact_remark(self):
        self.seed(
            [_moment('moment-remark', 'wxid-remark-a', 11)],
            entities=[
                EntityRecord(
                    entity_id='ent-remark-a', entity_type='Person', display_name='总编王老师',
                    identifiers={'nickname': '王老师', 'remark': '总编王老师', 'user_id': 'wxid-remark-a'},
                ),
            ],
        )
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': '总编王老师'}, request_id='req-remark',
        )
        self.assertTrue(response['ok'], response.get('error'))
        self.assertEqual(response['data']['scope']['author_id'], 'wxid-remark-a')

    def test_timeline_resolves_compound_remark_substring(self):
        self.seed(
            [_moment('moment-compound', 'wxid-compound-a', 11)],
            entities=[
                EntityRecord(
                    entity_id='ent-compound-a', entity_type='Person', display_name='Alice陈安然',
                    identifiers={
                        'nickname': 'Alice', 'alias': 'alice75882',
                        'remark': 'Alice陈安然', 'user_id': 'wxid-compound-a',
                    },
                ),
            ],
        )
        dispatcher = self.dispatcher()
        for target in ('陈安然', 'Alice陈安然', 'Alice', 'alice75882'):
            response = dispatcher.dispatch(
                'trove.moment_timeline', {'target': target}, request_id=f'req-compound-{target}',
            )
            self.assertTrue(response['ok'], (target, response.get('error')))
            self.assertEqual(response['data']['scope']['author_id'], 'wxid-compound-a')

    def test_timeline_fuzzy_substring_stays_ambiguous_with_two_authors(self):
        self.seed(
            [
                _moment('moment-chief-1', 'wxid-chief-1', 11),
                _moment('moment-chief-2', 'wxid-chief-2', 12),
            ],
            entities=[
                EntityRecord(
                    entity_id='ent-chief-1', entity_type='Person', display_name='主编阿红',
                    identifiers={'remark': '主编阿红', 'user_id': 'wxid-chief-1'},
                ),
                EntityRecord(
                    entity_id='ent-chief-2', entity_type='Person', display_name='主编阿蓝',
                    identifiers={'remark': '主编阿蓝', 'user_id': 'wxid-chief-2'},
                ),
            ],
        )
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': '主编'}, request_id='req-fuzzy-ambiguous',
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'ambiguous_target')
        candidates = response['error']['details']['candidates']
        self.assertEqual({item['id'] for item in candidates}, {'wxid-chief-1', 'wxid-chief-2'})
        self.assertTrue(all(item['match'] == 'entity_fuzzy' for item in candidates))

    def test_timeline_fuzzy_never_invents_author_without_moments(self):
        self.seed(
            [_moment('moment-real', 'wxid-real-author', 11)],
            entities=[
                EntityRecord(
                    entity_id='ent-real-author', entity_type='Person', display_name='书玉本人',
                    identifiers={'remark': '陈安然', 'user_id': 'wxid-real-author'},
                ),
                EntityRecord(
                    entity_id='ent-quiet', entity_type='Person', display_name='陈安然二号',
                    identifiers={'remark': '陈安然二号', 'user_id': 'wxid-no-moments'},
                ),
            ],
        )
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': '陈安然'}, request_id='req-fuzzy-narrow',
        )
        self.assertTrue(response['ok'], response.get('error'))
        self.assertEqual(response['data']['scope']['author_id'], 'wxid-real-author')

    def test_timeline_cursor_paginates_without_overlap(self):
        self._seed_author()
        dispatcher = self.dispatcher()
        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            payload = {'target': '阿May', 'limit': 2}
            if cursor:
                payload['cursor'] = cursor
            response = dispatcher.dispatch('trove.moment_timeline', payload, request_id=f'req-page-{pages}')
            self.assertTrue(response['ok'], response.get('error'))
            seen.extend(item['citation'] for item in response['data']['moments'])
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

    def test_timeline_cursor_is_bound_to_filters_and_handle(self):
        self._seed_author()
        dispatcher = self.dispatcher()
        first = dispatcher.dispatch(
            'trove.moment_timeline', {'target': '阿May', 'limit': 2}, request_id='req-bind-1',
        )
        cursor = first['page']['next_cursor']
        changed = dispatcher.dispatch(
            'trove.moment_timeline',
            {'target': '阿May', 'limit': 2, 'cursor': cursor, 'since': '2026-06-12T00:00:00Z'},
            request_id='req-bind-2',
        )
        self.assertEqual(changed['error']['code'], 'cursor_mismatch')
        garbage = dispatcher.dispatch(
            'trove.moment_timeline',
            {'target': '阿May', 'limit': 2, 'cursor': 'x' * 43},
            request_id='req-bind-3',
        )
        self.assertEqual(garbage['error']['code'], 'cursor_invalid')

    def test_timeline_time_range_filter(self):
        self._seed_author()
        response = self.dispatcher().dispatch(
            'trove.moment_timeline',
            {
                'target': 'wxid-author-a',
                'since': '2026-06-12T00:00:00Z',
                'until': '2026-06-14T00:00:00Z',
            },
            request_id='req-range',
        )
        self.assertTrue(response['ok'])
        citations = [item['citation'] for item in response['data']['moments']]
        self.assertEqual(
            citations,
            ['trove://wechat/acct-work/moment/moment-0003', 'trove://wechat/acct-work/moment/moment-0002'],
        )

    def test_timeline_excludes_non_active_status(self):
        self._seed_author()
        response = self.dispatcher().dispatch(
            'trove.moment_timeline', {'target': 'wxid-author-a', 'limit': 50}, request_id='req-status',
        )
        citations = [item['citation'] for item in response['data']['moments']]
        self.assertNotIn('trove://wechat/acct-work/moment/moment-deleted', citations)
        self.assertEqual(response['data']['matched_total'], 5)

    def test_timeline_account_scope_narrows_results(self):
        self.seed([
            _moment('moment-work', 'wxid-both', 11),
            _moment('moment-personal', 'wxid-both', 12, account_id='acct-personal',
                    citation='trove://wechat/acct-personal/moment/moment-personal'),
        ])
        scoped = self.dispatcher().dispatch(
            'trove.moment_timeline',
            {'target': 'wxid-both', 'account_id': 'acct-personal'},
            request_id='req-scope',
        )
        self.assertTrue(scoped['ok'])
        self.assertEqual(scoped['data']['matched_total'], 1)
        self.assertEqual(
            scoped['data']['moments'][0]['citation'],
            'trove://wechat/acct-personal/moment/moment-personal',
        )

    def test_timeline_limit_bounds_are_enforced(self):
        self._seed_author()
        dispatcher = self.dispatcher()
        for limit in (0, 101):
            response = dispatcher.dispatch(
                'trove.moment_timeline', {'target': 'wxid-author-a', 'limit': limit},
                request_id=f'req-limit-{limit}',
            )
            self.assertFalse(response['ok'])
            self.assertEqual(response['error']['code'], 'invalid_request')

    def test_timeline_queries_use_bounded_indexes(self):
        self._seed_author()
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    (
                        "SELECT moment_id FROM moment_items WHERE author_id=? AND status='active'"
                        ' ORDER BY timestamp DESC, moment_id DESC LIMIT ?',
                        ('wxid-author-a', 50),
                    ),
                    (
                        "SELECT COUNT(*) FROM moment_items WHERE author_id=? AND status='active'",
                        ('wxid-author-a',),
                    ),
                    (
                        "SELECT moment_id, interaction_type, COUNT(*) FROM moment_interactions"
                        " WHERE status='active' AND moment_id IN (?,?) GROUP BY moment_id, interaction_type",
                        ('moment-0001', 'moment-0002'),
                    ),
                ):
                    plans.extend(
                        str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params)
                    )
        finally:
            store.close()
        plan_text = ' | '.join(plans)
        self.assertNotIn('SCAN moment_items', plan_text)
        self.assertNotIn('SCAN moment_interactions', plan_text)
        self.assertIn('idx_moment_items_author_time', plan_text)
        self.assertIn('idx_moment_interactions_moment', plan_text)

    def test_missing_vault_returns_bounded_empty_result(self):
        missing = Path(self.temp.name) / 'missing-vault'
        config = VaultConfig.resolve(str(missing), env={})
        response = moment_handlers.moment_timeline(config, {'target': 'anyone'})
        self.assertTrue(response.ok)
        self.assertEqual(response.data['moments'], [])
        self.assertEqual(response.coverage['state'], 'complete')

        via_dispatcher = build_default_dispatcher(missing).dispatch(
            'trove.moment_timeline', {'target': 'anyone'}, request_id='req-empty',
        )
        self.assertFalse(via_dispatcher['ok'])
        self.assertEqual(via_dispatcher['error']['code'], 'no_results')
        self.assertEqual(via_dispatcher['error']['details']['candidates'], [])


class MomentInteractionTests(_MomentVaultCase):
    def _seed_interactions(self):
        self.seed(
            [
                _moment('moment-host', 'wxid-author-a', 14, text='正文'),
                _moment('moment-else', 'wxid-author-a', 15),
                _moment('moment-gone', 'wxid-author-a', 16, status='deleted'),
            ],
            [
                _interaction('int-1', 'moment-host', 'wxid-fan-1', kind='like', timestamp='2026-06-14T09:00:00Z'),
                _interaction('int-2', 'moment-host', 'wxid-fan-2', kind='comment', text='第一条评论', timestamp='2026-06-14T10:00:00Z'),
                _interaction('int-3', 'moment-host', 'wxid-fan-2', kind='comment', text='第二条评论', timestamp='2026-06-14T11:00:00Z'),
                _interaction('int-4', 'moment-else', 'wxid-fan-2', kind='like', timestamp='2026-06-15T09:00:00Z'),
            ],
        )

    def test_interactions_by_citation_include_moment_summary(self):
        self._seed_interactions()
        response = self.dispatcher().dispatch(
            'trove.moment_interactions',
            {'citation': 'trove://wechat/acct-work/moment/moment-host'},
            request_id='req-inter-cite',
        )
        self.assertTrue(response['ok'], response.get('error'))
        data = response['data']
        self.assertEqual(data['moment']['author_id'], 'wxid-author-a')
        self.assertEqual(data['moment']['text'], '正文')
        self.assertEqual(data['matched_total'], 3)
        types = [item['interaction_type'] for item in data['interactions']]
        self.assertEqual(types, ['comment', 'comment', 'like'])
        first = data['interactions'][0]
        self.assertEqual(first['text'], '第二条评论')
        self.assertEqual(first['actor_id'], 'wxid-fan-2')
        self.assertEqual(
            first['moment_citation'], 'trove://wechat/acct-work/moment/moment-host',
        )

    def test_interactions_by_actor_resolves_name_and_history(self):
        self._seed_interactions()
        response = self.dispatcher().dispatch(
            'trove.moment_interactions', {'target': 'wxid-fan-2'}, request_id='req-actor',
        )
        self.assertTrue(response['ok'])
        data = response['data']
        self.assertEqual(data['scope']['actor_id'], 'wxid-fan-2')
        self.assertEqual(data['matched_total'], 3)
        self.assertNotIn('moment', data)

    def test_interactions_actor_resolution_by_stored_name(self):
        self.seed(
            [_moment('moment-host', 'wxid-author-a', 14)],
            [
                _interaction('int-n1', 'moment-host', 'wxid-named-1', actor_name='小北'),
                _interaction('int-n2', 'moment-host', 'wxid-named-1', kind='comment', text='冒泡', timestamp='2026-06-14T10:00:00Z'),
            ],
        )
        response = self.dispatcher().dispatch(
            'trove.moment_interactions', {'target': '小北'}, request_id='req-actor-name',
        )
        self.assertTrue(response['ok'])
        self.assertEqual(response['data']['scope']['actor_id'], 'wxid-named-1')
        self.assertEqual(response['data']['matched_total'], 2)

    def test_interactions_actor_resolution_by_remark_substring(self):
        self.seed(
            [_moment('moment-host', 'wxid-author-a', 14)],
            [
                _interaction('int-f1', 'moment-host', 'wxid-fan-9', actor_name='Alice'),
                _interaction('int-f2', 'moment-host', 'wxid-fan-9', kind='comment', text='冒泡', timestamp='2026-06-14T10:00:00Z'),
            ],
            entities=[
                EntityRecord(
                    entity_id='ent-fan-9', entity_type='Person', display_name='Alice陈安然',
                    identifiers={'nickname': 'Alice', 'remark': 'Alice陈安然', 'user_id': 'wxid-fan-9'},
                ),
            ],
        )
        dispatcher = self.dispatcher()
        for target in ('陈安然', 'Alice陈安然'):
            response = dispatcher.dispatch(
                'trove.moment_interactions', {'target': target}, request_id=f'req-actor-remark-{target}',
            )
            self.assertTrue(response['ok'], (target, response.get('error')))
            self.assertEqual(response['data']['scope']['actor_id'], 'wxid-fan-9')
            self.assertEqual(response['data']['matched_total'], 2)

    def test_interactions_ambiguous_actor_returns_candidates(self):
        self.seed(
            [_moment('moment-host', 'wxid-author-a', 14)],
            [
                _interaction('int-d1', 'moment-host', 'wxid-twin-1', actor_name='Alice'),
                _interaction('int-d2', 'moment-host', 'wxid-twin-2', actor_name='Alice', timestamp='2026-06-14T10:00:00Z'),
            ],
        )
        response = self.dispatcher().dispatch(
            'trove.moment_interactions', {'target': 'Alice'}, request_id='req-actor-ambiguous',
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'ambiguous_target')
        candidates = response['error']['details']['candidates']
        self.assertEqual({item['id'] for item in candidates}, {'wxid-twin-1', 'wxid-twin-2'})

    def test_interactions_unknown_or_deleted_moment_is_no_results(self):
        self._seed_interactions()
        dispatcher = self.dispatcher()
        missing = dispatcher.dispatch(
            'trove.moment_interactions',
            {'citation': 'trove://wechat/acct-work/moment/moment-nope'},
            request_id='req-inter-missing',
        )
        self.assertEqual(missing['error']['code'], 'no_results')
        deleted = dispatcher.dispatch(
            'trove.moment_interactions',
            {'citation': 'trove://wechat/acct-work/moment/moment-gone'},
            request_id='req-inter-deleted',
        )
        self.assertEqual(deleted['error']['code'], 'no_results')

    def test_interactions_require_citation_or_target(self):
        response = self.dispatcher().dispatch(
            'trove.moment_interactions', {}, request_id='req-inter-empty',
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error']['code'], 'invalid_request')

    def test_interactions_paginate_with_cursor(self):
        self._seed_interactions()
        dispatcher = self.dispatcher()
        first = dispatcher.dispatch(
            'trove.moment_interactions',
            {'citation': 'trove://wechat/acct-work/moment/moment-host', 'limit': 2},
            request_id='req-inter-page-1',
        )
        self.assertTrue(first['ok'])
        self.assertTrue(first['page']['has_more'])
        self.assertEqual(first['coverage']['remaining'], 1)
        second = dispatcher.dispatch(
            'trove.moment_interactions',
            {
                'citation': 'trove://wechat/acct-work/moment/moment-host',
                'limit': 2,
                'cursor': first['page']['next_cursor'],
            },
            request_id='req-inter-page-2',
        )
        self.assertTrue(second['ok'])
        self.assertFalse(second['page']['has_more'])
        self.assertEqual(len(second['data']['interactions']), 1)
        first_ids = {item['citation'] for item in first['data']['interactions']}
        second_ids = {item['citation'] for item in second['data']['interactions']}
        self.assertFalse(first_ids & second_ids)

    def test_interactions_time_filter_and_scope(self):
        self._seed_interactions()
        response = self.dispatcher().dispatch(
            'trove.moment_interactions',
            {
                'citation': 'trove://wechat/acct-work/moment/moment-host',
                'since': '2026-06-14T09:30:00Z',
                'until': '2026-06-14T10:30:00Z',
            },
            request_id='req-inter-range',
        )
        self.assertTrue(response['ok'])
        self.assertEqual(response['data']['matched_total'], 1)
        self.assertEqual(response['data']['interactions'][0]['text'], '第一条评论')
        wrong_scope = self.dispatcher().dispatch(
            'trove.moment_interactions',
            {
                'citation': 'trove://wechat/acct-work/moment/moment-host',
                'account_id': 'acct-personal',
            },
            request_id='req-inter-scope',
        )
        self.assertEqual(wrong_scope['error']['code'], 'no_results')

    def test_actor_queries_use_bounded_indexes(self):
        self._seed_interactions()
        store = SQLiteStore(self.config.paths.sqlite_path, readonly=True)
        store.initialize()
        try:
            with store.connect() as conn:
                plans = []
                for sql, params in (
                    (
                        "SELECT interaction_id FROM moment_interactions WHERE actor_id=? AND status='active'"
                        ' ORDER BY timestamp DESC, interaction_id DESC LIMIT ?',
                        ('wxid-fan-2', 50),
                    ),
                    (
                        "SELECT 1 FROM moment_interactions WHERE actor_id=? AND status='active' LIMIT 1",
                        ('wxid-fan-2',),
                    ),
                ):
                    plans.extend(
                        str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params)
                    )
        finally:
            store.close()
        plan_text = ' | '.join(plans)
        self.assertNotIn('SCAN moment_interactions', plan_text)
        self.assertIn('idx_moment_interactions_actor_time', plan_text)


if __name__ == '__main__':
    unittest.main()
