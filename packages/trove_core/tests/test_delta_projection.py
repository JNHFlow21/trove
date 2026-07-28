from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trove_core.media_pipeline import enqueue_media_jobs
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.resources import MediaReference, discover_media_assets_delta
from trove_core.wechat.models import Account, Conversation, Message


class DeltaProjectionComplexityTests(unittest.TestCase):
    @staticmethod
    def _fixture(rows: int) -> tuple[list[Account], list[Conversation], list[Message]]:
        account = Account('acct-delta', 'Delta', 'Delta')
        conversation = Conversation('conv-delta', account.account_id, 'Delta conversation', 'private')
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        messages = [
            Message(
                account.account_id,
                account.label,
                conversation.conversation_id,
                conversation.title,
                conversation.type,
                'sender',
                'Sender',
                start + timedelta(seconds=index),
                f'synthetic message {index}',
                'message_0',
                index,
            )
            for index in range(rows)
        ]
        return [account], [conversation], messages

    def test_10k_message_edit_add_delete_are_citation_proportional(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'trove.sqlite')
            repo = WeChatRepository(store)
            accounts, conversations, messages = self._fixture(10_000)
            seeded = repo.apply_delta(accounts, conversations, messages)
            self.assertEqual(seeded['messages_changed'], 10_000)

            edited = Message(
                **{
                    **messages[4_321].__dict__,
                    'content': 'synthetic edited citation only',
                }
            )
            edit = repo.apply_delta(accounts, conversations, [edited])
            self.assertEqual(edit['messages_changed'], 1)
            self.assertEqual(edit['chunks']['citations'], 1)
            self.assertEqual(edit['metrics']['candidate_rows'], 1)
            self.assertEqual(edit['metrics']['rows_scanned'], 1)
            self.assertLessEqual(edit['metrics']['sql_statements'], 12)
            self.assertEqual(edit['metrics']['commits'], 1)

            identical = repo.apply_delta(accounts, conversations, [edited])
            self.assertEqual(identical['messages_changed'], 0)
            self.assertEqual(identical['metrics']['commits'], 0)
            self.assertEqual(identical['metrics']['rows_written'], 0)
            self.assertEqual(identical['metrics']['wal_bytes'], 0)

            added = Message(
                **{
                    **messages[-1].__dict__,
                    'local_id': 10_001,
                    'timestamp': messages[-1].timestamp + timedelta(seconds=1),
                    'content': 'synthetic one-row append',
                }
            )
            addition = repo.apply_delta(accounts, conversations, [added])
            self.assertEqual(addition['messages_changed'], 1)
            self.assertEqual(addition['metrics']['rows_scanned'], 1)

            deletion = repo.apply_delta(
                [],
                [],
                [],
                deleted_citations=[messages[7_777].citation],
            )
            self.assertEqual(deletion['tombstones'], 1)
            self.assertEqual(deletion['chunks']['citations'], 1)
            self.assertEqual(deletion['metrics']['candidate_rows'], 1)
            self.assertEqual(deletion['metrics']['rows_scanned'], 1)
            with store.connect() as conn:
                self.assertIsNone(conn.execute('SELECT 1 FROM messages WHERE citation=?', (messages[7_777].citation,)).fetchone())
                self.assertIsNotNone(conn.execute('SELECT 1 FROM sync_citation_tombstones WHERE citation=?', (messages[7_777].citation,)).fetchone())

    def test_complete_source_inventory_emits_deletion_tombstone_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'trove.sqlite')
            repo = WeChatRepository(store)
            accounts, conversations, messages = self._fixture(2)
            repo.apply_delta(
                accounts,
                conversations,
                messages,
                source_key='synthetic-source',
                source_snapshot_complete=True,
            )

            report = repo.apply_delta(
                accounts,
                conversations,
                messages[:1],
                source_key='synthetic-source',
                source_snapshot_complete=True,
            )

            self.assertEqual(report['source_rows_deleted'], 1)
            self.assertEqual(report['tombstones'], 1)
            self.assertEqual(report['tombstone_citations'], [messages[1].citation])
            self.assertEqual(report['metrics']['commits'], 1)

    def test_conversation_metadata_is_the_only_child_fanout(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'trove.sqlite')
            repo = WeChatRepository(store)
            accounts, conversations, messages = self._fixture(32)
            repo.apply_delta(accounts, conversations, messages)
            renamed = [Conversation('conv-delta', 'acct-delta', 'Renamed once', 'private')]

            report = repo.apply_delta(accounts, renamed, [])

            self.assertEqual(report['metadata_conversations'], 1)
            self.assertEqual(report['citations_changed'], 32)
            self.assertEqual(report['chunks']['citations'], 32)

    def test_non_payload_cleanup_uses_one_set_based_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'trove.sqlite')
            repo = WeChatRepository(store)
            accounts, conversations, messages = self._fixture(2_000)
            repo.apply_delta(accounts, conversations, messages)
            timestamp = '2026-01-01T00:00:00Z'
            with store.connect() as conn:
                conn.executemany(
                    """INSERT INTO message_payloads(
                           citation,appmsg_type,normalized_type,parse_status,normalized_json,display_text,
                           source_hash,parser_version,unsupported_reason,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            message.citation,
                            None,
                            'link',
                            'parsed',
                            '{}',
                            '[appmsg/link] stale fixture',
                            'a' * 64,
                            'fixture-v1',
                            None,
                            timestamp,
                            timestamp,
                        )
                        for message in messages
                    ],
                )
                conn.commit()

            payload_deletes: list[str] = []
            with store.connect() as conn:
                conn.set_trace_callback(
                    lambda statement: payload_deletes.append(statement)
                    if statement.lstrip().upper().startswith('DELETE FROM MESSAGE_PAYLOADS')
                    else None
                )
            report = repo.apply_delta(accounts, conversations, messages)
            with store.connect() as conn:
                conn.set_trace_callback(None)
                remaining = int(conn.execute('SELECT COUNT(*) FROM message_payloads').fetchone()[0])

            self.assertEqual(remaining, 0)
            self.assertEqual(len(payload_deletes), 1)
            self.assertEqual(report['metrics']['rows_written'], len(messages))
            self.assertEqual(report['metrics']['commits'], 1)

    def test_media_discovery_empty_and_append_paths_are_delta_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_db = root / 'message_resource.db'
            with sqlite3.connect(source_db) as conn:
                conn.execute('CREATE TABLE resource_detail(local_id INTEGER, local_type TEXT, path TEXT, ignored_payload TEXT)')
                conn.executemany(
                    'INSERT INTO resource_detail VALUES(?,?,?,?)',
                    [(index, '3', f'synthetic-{index}.dat', 'not selected') for index in range(1_000)],
                )
            store = SQLiteStore(root / 'trove.sqlite')
            linker = MediaLinker(MultimodalRepository(store))
            initial = discover_media_assets_delta(root, store=store, account_id='acct-media')
            linker.link_references(initial.refs, source_states=initial.source_states)

            empty = discover_media_assets_delta(root, store=store, account_id='acct-media')
            empty_persist = linker.link_references(empty.refs, source_states=empty.source_states)
            empty_queue = enqueue_media_jobs(store, asset_ids=empty_persist.changed_asset_ids)
            self.assertEqual(empty.counters['source_rows_scanned'], 0)
            self.assertEqual(empty.counters['candidate_rows'], 0)
            self.assertEqual(empty_persist.metrics['commits'], 0)
            self.assertEqual(empty_persist.metrics['rows_written'], 0)
            self.assertEqual(empty_persist.metrics['wal_bytes'], 0)
            self.assertEqual(empty_queue['metrics']['commits'], 0)

            with sqlite3.connect(source_db) as conn:
                conn.execute("INSERT INTO resource_detail VALUES(1001,'3','synthetic-new.dat','not selected')")
            appended = discover_media_assets_delta(root, store=store, account_id='acct-media')
            appended_persist = linker.link_references(appended.refs, source_states=appended.source_states)
            self.assertEqual(appended.counters['appended_tables'], 1)
            self.assertEqual(appended.counters['source_rows_scanned'], 1)
            self.assertEqual(appended.counters['candidate_rows'], 1)
            self.assertEqual(appended_persist.assets_upserted, 1)
            self.assertEqual(appended_persist.metrics['commits'], 1)

    def test_media_bulk_identical_conflict_is_noop_and_queue_scopes_one_of_10k(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'trove.sqlite')
            repo = MultimodalRepository(store)
            accounts, conversations, messages = self._fixture(1)
            WeChatRepository(store).apply_delta(accounts, conversations, messages)
            assets = [
                MediaAssetRecord(
                    asset_id=f'asset-{index}',
                    account_id='acct-delta',
                    source_type='message',
                    source_id=f'source-{index}',
                    modality='voice' if index == 9_999 else 'image',
                    media_type='voice' if index == 9_999 else 'image',
                    citation=messages[0].citation if index == 9_999 else f'trove://synthetic/image/{index}',
                )
                for index in range(10_000)
            ]
            first = repo.upsert_media_graph(assets, [])
            second = repo.upsert_media_graph(assets, [])
            self.assertEqual(first['assets_upserted'], 10_000)
            self.assertEqual(second['assets_upserted'], 0)
            self.assertEqual(second['metrics']['commits'], 0)
            self.assertEqual(second['metrics']['rows_written'], 0)
            self.assertEqual(second['metrics']['wal_bytes'], 0)

            queued = enqueue_media_jobs(store, asset_ids=['asset-9999'])
            self.assertEqual(queued['seen'], 1)
            self.assertEqual(queued['queued'], 1)
            self.assertEqual(queued['metrics']['candidate_rows'], 1)
            self.assertEqual(queued['metrics']['rows_scanned'], 1)

    def test_media_graph_and_private_voice_job_commit_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / 'trove.sqlite')
            accounts, conversations, messages = self._fixture(1)
            WeChatRepository(store).apply_delta(accounts, conversations, messages)
            linker = MediaLinker(MultimodalRepository(store))
            ref = MediaReference(
                asset_id='asset-atomic-voice',
                account_id='acct-delta',
                source_type='private_chat',
                source_id=messages[0].citation,
                modality='voice',
                media_type='voice',
                citation=messages[0].citation,
            )
            with store.connect() as conn:
                conn.execute(
                    """CREATE TRIGGER fail_atomic_media_job BEFORE INSERT ON media_jobs
                       BEGIN SELECT RAISE(ABORT, 'synthetic queue failure'); END"""
                )
                conn.commit()

            with self.assertRaises(sqlite3.IntegrityError):
                linker.link_references([ref])
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM media_assets WHERE asset_id='asset-atomic-voice'",
                ).fetchone()[0], 0)
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM media_asset_links WHERE asset_id='asset-atomic-voice'",
                ).fetchone()[0], 0)
                conn.execute('DROP TRIGGER fail_atomic_media_job')
                conn.commit()

            report = linker.link_references([ref])
            self.assertEqual(report.metrics['commits'], 1)
            self.assertEqual(report.metrics['rows_written'], 3)
            with store.connect() as conn:
                job = conn.execute(
                    "SELECT status FROM media_jobs WHERE asset_id='asset-atomic-voice'",
                ).fetchone()
            self.assertIsNotNone(job)
            self.assertEqual(job['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
