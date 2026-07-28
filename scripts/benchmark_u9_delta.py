#!/usr/bin/env python3
"""Synthetic U9 complexity gate (10k default, 100k nightly-ready)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile

from trove_core.media_pipeline import enqueue_media_jobs
from trove_core.store.repositories import MediaAssetRecord, MultimodalRepository, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.media.linker import MediaLinker
from trove_core.wechat.media.resources import discover_media_assets_delta
from trove_core.wechat.models import Account, Conversation, Message


def message_fixture(rows: int) -> tuple[list[Account], list[Conversation], list[Message]]:
    account = Account('acct-benchmark', 'Benchmark', 'Benchmark')
    conversation = Conversation('conv-benchmark', account.account_id, 'Synthetic benchmark', 'private')
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    messages = [
        Message(
            account.account_id,
            account.label,
            conversation.conversation_id,
            conversation.title,
            conversation.type,
            'sender',
            'Sender',
            base + timedelta(seconds=index),
            f'synthetic message {index}',
            'message_0',
            index,
        )
        for index in range(rows)
    ]
    return [account], [conversation], messages


def run(rows: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='trove-u9-synthetic-') as directory:
        root = Path(directory)
        store = SQLiteStore(root / 'trove.sqlite')
        wechat = WeChatRepository(store)
        accounts, conversations, messages = message_fixture(rows)
        seed = wechat.apply_delta(accounts, conversations, messages)

        target = rows // 2
        edited = Message(**{**messages[target].__dict__, 'content': 'synthetic exact edit'})
        edit = wechat.apply_delta(accounts, conversations, [edited])
        identical = wechat.apply_delta(accounts, conversations, [edited])
        added = Message(**{
            **messages[-1].__dict__,
            'local_id': rows + 1,
            'timestamp': messages[-1].timestamp + timedelta(seconds=1),
            'content': 'synthetic exact append',
        })
        addition = wechat.apply_delta(accounts, conversations, [added])
        deletion = wechat.apply_delta([], [], [], deleted_citations=[messages[target + 1].citation])

        source_db = root / 'message_resource.db'
        with sqlite3.connect(source_db) as conn:
            conn.execute('CREATE TABLE resource_detail(local_id INTEGER,local_type TEXT,path TEXT,ignored_payload TEXT)')
            conn.executemany(
                'INSERT INTO resource_detail VALUES(?,?,?,?)',
                ((index, '3', f'synthetic-{index}.dat', 'not selected') for index in range(rows)),
            )
        linker = MediaLinker(MultimodalRepository(store))
        media_seed = discover_media_assets_delta(root, store=store, account_id='acct-media-benchmark')
        media_seed_write = linker.link_references(media_seed.refs, source_states=media_seed.source_states)
        media_empty = discover_media_assets_delta(root, store=store, account_id='acct-media-benchmark')
        media_empty_write = linker.link_references(media_empty.refs, source_states=media_empty.source_states)
        media_empty_queue = enqueue_media_jobs(store, asset_ids=media_empty_write.changed_asset_ids)
        with sqlite3.connect(source_db) as conn:
            conn.execute(
                'INSERT INTO resource_detail VALUES(?,?,?,?)',
                (rows + 1, '3', 'synthetic-appended.dat', 'not selected'),
            )
        media_append = discover_media_assets_delta(root, store=store, account_id='acct-media-benchmark')
        media_append_write = linker.link_references(media_append.refs, source_states=media_append.source_states)

        voice = MediaAssetRecord(
            'asset-queue-delta',
            'acct-benchmark',
            'message',
            'synthetic-voice',
            'voice',
            'voice',
            messages[0].citation,
        )
        MultimodalRepository(store).upsert_media_graph([voice], [])
        queue_delta = enqueue_media_jobs(store, asset_ids=[voice.asset_id])

        gates = {
            'edit_one_candidate': edit['metrics']['candidate_rows'] == 1 and edit['chunks']['citations'] == 1,
            'edit_noop_zero_commit': identical['metrics']['commits'] == 0 and identical['metrics']['rows_written'] == 0 and identical['metrics']['wal_bytes'] == 0,
            'append_one_candidate': addition['metrics']['candidate_rows'] == 1,
            'delete_one_tombstone': deletion['tombstones'] == 1 and deletion['chunks']['citations'] == 1,
            'empty_media_zero_scan': media_empty.counters['source_rows_scanned'] == 0,
            'empty_media_zero_commit': media_empty_write.metrics['commits'] == 0 and media_empty_queue['metrics']['commits'] == 0,
            'media_append_one_scan': media_append.counters['source_rows_scanned'] == 1 and media_append.counters['candidate_rows'] == 1,
            'queue_one_candidate': queue_delta['metrics']['candidate_rows'] == 1 and queue_delta['metrics']['rows_scanned'] == 1,
        }
        return {
            'ok': all(gates.values()),
            'rows': rows,
            'gates': gates,
            'message': {
                'seed': seed['metrics'],
                'edit': edit['metrics'],
                'identical': identical['metrics'],
                'append': addition['metrics'],
                'delete': deletion['metrics'],
            },
            'media': {
                'seed_discovery': media_seed.counters,
                'seed_persistence': media_seed_write.metrics,
                'empty_discovery': media_empty.counters,
                'empty_persistence': media_empty_write.metrics,
                'empty_queue': media_empty_queue['metrics'],
                'append_discovery': media_append.counters,
                'append_persistence': media_append_write.metrics,
                'queue_delta': queue_delta['metrics'],
            },
            'synthetic_only': True,
            'raw_content_included': False,
            'raw_paths_included': False,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=10_000)
    args = parser.parse_args()
    if args.rows < 2:
        parser.error('--rows must be at least 2')
    report = run(args.rows)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
