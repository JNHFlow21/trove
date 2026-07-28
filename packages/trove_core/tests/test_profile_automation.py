from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.knowledge.profile_automation import (
    ProfileAutomationService,
    process_profile_refresh_queue,
)
from trove_core.knowledge.profile_snapshots import (
    diff_profile_snapshots,
    get_profile_snapshot,
    list_profile_snapshots,
)
from trove_core.store.repositories import EntityRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


class ProfileAutomationTests(unittest.TestCase):
    def _fixture(self, root: str):
        vault = Path(root) / 'vault'
        cfg = VaultConfig.resolve(str(vault), env={})
        cfg.ensure()
        store = SQLiteStore(cfg.paths.sqlite_path)
        MultimodalRepository(store).upsert_entity(EntityRecord(
            entity_id='customer-auto',
            entity_type='Customer',
            display_name='自动画像客户',
            identifiers={'wechat_id': 'wxid-auto', 'remark': '自动画像客户'},
        ))
        first = Message(
            'acct', 'A', 'wxid-auto', '自动画像客户', 'private', 'wxid-auto', '自动画像客户',
            datetime(2026, 1, 1, tzinfo=timezone.utc), '希望下周继续跟进方案', 's', 1,
        )
        repo = WeChatRepository(store)
        repo.replace_fixture(
            [Account('acct', 'A', 'A')],
            [Conversation('wxid-auto', 'acct', '自动画像客户', 'private')],
            [first],
        )
        return cfg, store, repo

    def test_explicit_subscription_auto_saves_only_material_profile_changes(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, repo = self._fixture(root)
            enabled = agent_tools.profile_automation_enable(
                cfg.root, '自动画像客户', debounce_seconds=0,
            )
            self.assertTrue(enabled['enabled'])
            self.assertEqual(enabled['queue_state'], 'pending')

            first = process_profile_refresh_queue(cfg, limit=5)
            self.assertEqual(first['created_snapshots'], 1)
            self.assertEqual(list_profile_snapshots(store, '自动画像客户')['count'], 1)

            # A duplicate queue event must be idempotent when the safe profile
            # projection did not materially change.
            ProfileAutomationService(store).enqueue_all(reason='duplicate_fixture', debounce_override_seconds=0)
            unchanged = process_profile_refresh_queue(cfg, limit=5)
            self.assertEqual(unchanged['created_snapshots'], 0)
            self.assertEqual(unchanged['cache_hits'], 1)
            self.assertEqual(list_profile_snapshots(store, '自动画像客户')['count'], 1)

            second = Message(
                'acct', 'A', 'wxid-auto', '自动画像客户', 'private', 'wxid-auto', '自动画像客户',
                datetime(2026, 1, 2, tzinfo=timezone.utc), '新增需求是整理预算和审批计划', 's', 2,
            )
            repo.replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-auto', 'acct', '自动画像客户', 'private')],
                [second],
            )
            ProfileAutomationService(store).enqueue_all(reason='sync_delta', debounce_override_seconds=0)
            refreshed = process_profile_refresh_queue(cfg, limit=5)
            self.assertEqual(refreshed['created_snapshots'], 1)
            versions = list_profile_snapshots(store, '自动画像客户')
            self.assertEqual([item['version'] for item in versions['items']], [2, 1])
            self.assertFalse(get_profile_snapshot(store, '自动画像客户', version=2)['stale'])

    def test_disable_prevents_future_automatic_queueing(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, _ = self._fixture(root)
            agent_tools.profile_automation_enable(cfg.root, '自动画像客户', debounce_seconds=0)
            process_profile_refresh_queue(cfg, limit=5)
            disabled = agent_tools.profile_automation_disable(cfg.root, '自动画像客户')
            self.assertFalse(disabled['enabled'])
            queued = ProfileAutomationService(store).enqueue_all(
                reason='sync_delta', debounce_override_seconds=0,
            )
            self.assertEqual(queued['queued'], 0)

    def test_queue_generation_cas_retries_instead_of_publishing_stale_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, _ = self._fixture(root)
            service = ProfileAutomationService(store)
            service.enable('自动画像客户', debounce_seconds=0)
            claim = service.claim_due(now=datetime.now(timezone.utc) + timedelta(seconds=1))
            self.assertIsNotNone(claim)
            prepared = service.prepare(claim)
            service.enqueue_all(reason='newer_sync', debounce_override_seconds=0)
            result = service.publish(claim, prepared)
            self.assertEqual(result['status'], 'retry_required')
            with store.connect() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM profile_snapshots').fetchone()[0], 0)

    def test_expired_claim_reissue_fences_the_old_worker(self):
        with tempfile.TemporaryDirectory() as root:
            _cfg, store, _ = self._fixture(root)
            service = ProfileAutomationService(store)
            started = datetime(2026, 1, 1, tzinfo=timezone.utc)
            service.enable('自动画像客户', debounce_seconds=0, now=started)
            old_claim = service.claim_due(now=started)
            old_prepared = service.prepare(old_claim)
            new_claim = service.claim_due(now=started + timedelta(seconds=301))

            self.assertGreater(new_claim['generation'], old_claim['generation'])
            self.assertFalse(service.fail(old_claim, 'late_worker', now=started + timedelta(seconds=302))['applied'])
            self.assertEqual(service.publish(old_claim, old_prepared)['status'], 'retry_required')
            self.assertTrue(service.publish(new_claim, service.prepare(new_claim))['ok'])

    def test_capped_worker_reports_remaining_due_backlog(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, _ = self._fixture(root)
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-second', entity_type='Customer', display_name='第二位画像客户',
                identifiers={'wechat_id': 'wxid-second'},
            ))
            service = ProfileAutomationService(store)
            service.enable('自动画像客户', debounce_seconds=0)
            service.enable('第二位画像客户', debounce_seconds=0)

            partial = process_profile_refresh_queue(cfg, limit=1)

            self.assertFalse(partial['ok'])
            self.assertEqual(partial['status'], 'backlog_remaining')
            self.assertEqual(partial['remaining_due'], 1)
            self.assertFalse(partial['drained'])
            completed = process_profile_refresh_queue(cfg, limit=5)
            self.assertTrue(completed['drained'])

    def test_repeated_sync_events_preserve_the_first_debounce_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            _cfg, store, _ = self._fixture(root)
            service = ProfileAutomationService(store)
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            service.enable('自动画像客户', debounce_seconds=180, now=start)
            self.assertIsNotNone(service.claim_due(now=start))
            with store.connect() as conn:
                conn.execute('DELETE FROM profile_refresh_queue')
                conn.commit()

            service.enqueue_all(reason='first_sync', now=start)
            service.enqueue_all(reason='second_sync', now=start + timedelta(seconds=60))

            with store.connect() as conn:
                row = conn.execute(
                    'SELECT generation,available_at FROM profile_refresh_queue'
                ).fetchone()
            self.assertEqual(row['generation'], 2)
            self.assertEqual(row['available_at'], '2026-01-01T00:03:00Z')
            self.assertTrue(service.has_due(now=start + timedelta(seconds=180)))

    def test_message_delta_queues_only_the_impacted_subscription(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, _ = self._fixture(root)
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-other',
                entity_type='Customer',
                display_name='另一位客户',
                identifiers={'wechat_id': 'wxid-other'},
            ))
            agent_tools.profile_automation_enable(
                cfg.root, '自动画像客户', debounce_seconds=0,
            )
            agent_tools.profile_automation_enable(
                cfg.root, '另一位客户', debounce_seconds=0,
            )
            process_profile_refresh_queue(cfg, limit=5)

            queued = ProfileAutomationService(store).enqueue_impacted(
                {'wxid-auto'}, reason='sync_message_delta', debounce_override_seconds=0,
            )

            self.assertEqual(queued['subscriptions_checked'], 2)
            self.assertEqual(queued['queued'], 1)
            with store.connect() as conn:
                rows = list(conn.execute('SELECT entity_id FROM profile_refresh_queue'))
            self.assertEqual([row['entity_id'] for row in rows], ['customer-auto'])

    def test_alias_index_queues_subscription_without_scanning_raw_alias_shape(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, _ = self._fixture(root)
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-alias',
                entity_type='Customer',
                display_name='Alias Display',
                identifiers={'remark': 'Alias Remark'},
            ))
            agent_tools.profile_automation_enable(
                cfg.root, 'Alias Display', debounce_seconds=0,
            )
            process_profile_refresh_queue(cfg, limit=5)

            queued = ProfileAutomationService(store).enqueue_impacted(
                {'Alias Remark'}, reason='sync_message_delta', debounce_override_seconds=0,
            )

            self.assertEqual(queued['queued'], 1)
            with store.connect() as conn:
                self.assertEqual(
                    conn.execute('SELECT entity_id FROM profile_refresh_queue').fetchone()[0],
                    'customer-alias',
                )

    def test_refresh_now_processes_the_requested_entity_not_an_older_queue_item(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, _ = self._fixture(root)
            MultimodalRepository(store).upsert_entity(EntityRecord(
                entity_id='customer-other',
                entity_type='Customer',
                display_name='Other Customer',
                identifiers={'wechat_id': 'wxid-other'},
            ))
            agent_tools.profile_automation_enable(
                cfg.root, 'Other Customer', debounce_seconds=0,
            )
            agent_tools.profile_automation_enable(
                cfg.root, '自动画像客户', debounce_seconds=0,
            )

            refreshed = agent_tools.profile_automation_refresh_now(
                cfg.root, '自动画像客户',
            )

            self.assertTrue(refreshed['ok'])
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM profile_snapshots WHERE entity_id='customer-auto'"
                ).fetchone()[0], 1)
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM profile_snapshots WHERE entity_id='customer-other'"
                ).fetchone()[0], 0)
                self.assertIsNotNone(conn.execute(
                    "SELECT 1 FROM profile_refresh_queue WHERE entity_id='customer-other'"
                ).fetchone())

    def test_history_get_and_diff_are_bounded_cited_and_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, repo = self._fixture(root)
            agent_tools.profile_automation_enable(cfg.root, '自动画像客户', debounce_seconds=0)
            process_profile_refresh_queue(cfg, limit=5)
            repo.replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-auto', 'acct', '自动画像客户', 'private')],
                [Message(
                    'acct', 'A', 'wxid-auto', '自动画像客户', 'private', 'wxid-auto', '自动画像客户',
                    datetime(2026, 1, 3, tzinfo=timezone.utc), '下一步是周五复盘报价', 's', 3,
                )],
            )
            ProfileAutomationService(store).enqueue_all(reason='sync_delta', debounce_override_seconds=0)
            process_profile_refresh_queue(cfg, limit=5)

            old = get_profile_snapshot(store, '自动画像客户', version=1)
            self.assertEqual(old['version'], 1)
            self.assertTrue(old['stale'])
            self.assertEqual(old['freshness_state'], 'stale')
            self.assertIn(old['completeness_state'], {'current', 'current_with_deferred_enrichment'})
            self.assertTrue(old['profile']['sections'])
            self.assertFalse(old['raw_content_included'])
            delta = diff_profile_snapshots(store, '自动画像客户', from_version=1, to_version=2)
            self.assertEqual((delta['from_version'], delta['to_version']), (1, 2))
            self.assertGreater(delta['changes_count'], 0)
            self.assertFalse(delta['raw_content_included'])
            history = list_profile_snapshots(store, '自动画像客户')['items']
            self.assertEqual(
                [item['freshness_state'] for item in history], ['current', 'stale'],
            )

    def test_content_reversion_creates_a_new_version_instead_of_pointing_at_old_history(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, repo = self._fixture(root)
            original = Message(
                'acct', 'A', 'wxid-auto', '自动画像客户', 'private',
                'wxid-auto', '自动画像客户',
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                '希望下周继续跟进方案', 's', 1,
            )
            agent_tools.profile_automation_enable(cfg.root, '自动画像客户', debounce_seconds=0)
            process_profile_refresh_queue(cfg, limit=5)
            first = get_profile_snapshot(store, '自动画像客户', version=1)
            changed = Message(
                'acct', 'A', 'wxid-auto', '自动画像客户', 'private',
                'wxid-auto', '自动画像客户',
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                '临时变更后的画像内容', 's', 2,
            )
            repo.replace_fixture(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-auto', 'acct', '自动画像客户', 'private')],
                [changed],
            )
            ProfileAutomationService(store).enqueue_all(
                reason='changed', debounce_override_seconds=0,
            )
            process_profile_refresh_queue(cfg, limit=5)
            repo.apply_delta(
                [Account('acct', 'A', 'A')],
                [Conversation('wxid-auto', 'acct', '自动画像客户', 'private')],
                [original],
                deleted_citations=[changed.citation],
            )
            ProfileAutomationService(store).enqueue_all(
                reason='reverted', debounce_override_seconds=0,
            )
            process_profile_refresh_queue(cfg, limit=5)

            latest = get_profile_snapshot(store, '自动画像客户')
            self.assertEqual(latest['version'], 3)
            self.assertEqual(latest['content_hash'], first['content_hash'])
            self.assertNotEqual(latest['profile_id'], first['profile_id'])
            self.assertFalse(latest['stale'])

    def test_automatic_history_retention_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            cfg, store, repo = self._fixture(root)
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO profile_snapshots(
                           profile_id,entity_id,version,projection_json,content_hash,source_revision,
                           schema_version,completeness_state,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        'profile-formal', 'customer-auto', 1, '{}', 'formal-hash',
                        'formal-source', 'customer-profile/v2', 'complete',
                        '2025-12-31T00:00:00Z',
                    ),
                )
                conn.commit()
            agent_tools.profile_automation_enable(cfg.root, '自动画像客户', debounce_seconds=0)
            with patch(
                'trove_core.knowledge.profile_automation.AUTOMATIC_SNAPSHOT_HISTORY_LIMIT', 2,
            ):
                process_profile_refresh_queue(cfg, limit=5)
                for index in (2, 3):
                    repo.replace_fixture(
                        [Account('acct', 'A', 'A')],
                        [Conversation('wxid-auto', 'acct', '自动画像客户', 'private')],
                        [Message(
                            'acct', 'A', 'wxid-auto', '自动画像客户', 'private',
                            'wxid-auto', '自动画像客户',
                            datetime(2026, 1, index, tzinfo=timezone.utc),
                            f'自动画像滚动版本 {index}', 's', index,
                        )],
                    )
                    ProfileAutomationService(store).enqueue_all(
                        reason='retention_fixture', debounce_override_seconds=0,
                    )
                    process_profile_refresh_queue(cfg, limit=5)

            versions = list_profile_snapshots(store, '自动画像客户')
            self.assertEqual(versions['count'], 3)
            self.assertEqual([item['version'] for item in versions['items']], [4, 3, 1])
            self.assertEqual(versions['items'][-1]['profile_id'], 'profile-formal')

    def test_retries_stop_after_a_bounded_failure_count_and_new_work_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            _cfg, store, _ = self._fixture(root)
            service = ProfileAutomationService(store)
            current = datetime(2026, 1, 1, tzinfo=timezone.utc)
            service.enable('自动画像客户', debounce_seconds=0, now=current)
            for attempt in range(1, 6):
                claim = service.claim_due(now=current)
                self.assertIsNotNone(claim)
                failed = service.fail(claim, 'fixture_failure', now=current)
                self.assertEqual(failed['terminal'], attempt == 5)
                current += timedelta(seconds=301)
            with store.connect() as conn:
                row = conn.execute(
                    'SELECT state,attempt_count FROM profile_refresh_queue'
                ).fetchone()
            self.assertEqual((row['state'], row['attempt_count']), ('failed', 5))
            self.assertFalse(service.has_due(now=current))

            service.enqueue_all(
                reason='new_sync', debounce_override_seconds=0, now=current,
            )
            with store.connect() as conn:
                row = conn.execute(
                    'SELECT state,attempt_count FROM profile_refresh_queue'
                ).fetchone()
            self.assertEqual((row['state'], row['attempt_count']), ('pending', 0))


if __name__ == '__main__':
    unittest.main()
