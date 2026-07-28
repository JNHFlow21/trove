from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from trove_core.approvals import ApprovalManager, claim_approval_grant
from trove_core.embedding.fake_provider import FakeEmbeddingProvider
from trove_core.runtime import (
    VectorFullRebuildRequired,
    VectorIndexSourceChanged,
    _vector_source_snapshot,
    index_vectors,
    vector_cloud_approval_payload,
)
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.migrations import migrate_schema
from trove_core.store.schema import SCHEMA_VERSION, VECTOR_SOURCE_TRIGGER_NAMES
from trove_core.vault.config import VaultConfig
from trove_core.wechat.indexer import index_fixture_vault


class _CloudEmbeddingProvider:
    name = 'synthetic-cloud:embedding-v1'
    provider_name = 'synthetic-cloud'
    model = 'embedding-v1'
    dimensions = 4
    endpoint = 'https://example.invalid/embeddings'
    egress_kind = 'cloud_embedding_upload'

    def __init__(self, on_first_embed=None) -> None:
        self.calls = 0
        self._on_first_embed = on_first_embed

    def embed(self, _text: str) -> list[float]:
        self.calls += 1
        if self.calls == 1 and self._on_first_embed is not None:
            self._on_first_embed()
        return [1.0, 0.0, 0.0, 0.0]


class VectorSourceCASTests(unittest.TestCase):
    def _approved_request(self, cfg: VaultConfig, provider: _CloudEmbeddingProvider):
        payload = vector_cloud_approval_payload(
            cfg,
            provider,
            backend='sqlite',
            batch_size=16,
            max_messages=2,
            purge=False,
        )
        grant = ApprovalManager(cfg.root).require(
            'cloud_vector_index',
            'cloud_embedding_upload',
            payload,
            one_step_approval=True,
        )
        claim_approval_grant(
            grant,
            cfg.root,
            action='cloud_vector_index',
            danger_class='cloud_embedding_upload',
            payload=payload,
        )
        return payload, grant

    def test_source_revision_tracks_insert_update_delete_for_messages_and_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            store = SQLiteStore(cfg.paths.sqlite_path)

            def revision() -> int:
                with store.connect() as conn:
                    return int(conn.execute(
                        "SELECT value FROM schema_meta WHERE key='vector_source_revision'"
                    ).fetchone()[0])

            message_citation = 'trove://fixture/vector-cas/message'
            before = revision()
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO messages(
                           citation,account_id,account_label,conversation_id,conversation_title,
                           conversation_type,sender_id,sender_name,timestamp,content,content_kind,
                           shard_id,local_id,sent_by_me,source_type,direction
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        message_citation, 'cas-account', 'CAS', 'cas-conversation', 'CAS conversation',
                        'private', 'cas-sender', 'CAS sender', '2026-01-01T00:00:00Z',
                        'inserted source', 'text', 'cas-shard', 1, 0, 'wechat', 'inbound',
                    ),
                )
                conn.commit()
            self.assertEqual(revision(), before + 1)

            with store.connect() as conn:
                conn.execute('UPDATE messages SET content=? WHERE citation=?', ('updated source', message_citation))
                conn.commit()
            self.assertEqual(revision(), before + 2)

            with store.connect() as conn:
                conn.execute('DELETE FROM messages WHERE citation=?', (message_citation,))
                conn.commit()
            self.assertEqual(revision(), before + 3)

            chunk_citation = 'trove://fixture/vector-cas/chunk'
            with store.connect() as conn:
                conn.execute(
                    """INSERT INTO evidence_chunks(
                           chunk_id,chunk_citation,parent_citation,account_id,account_label,
                           source_type,source_id,title,actor,timestamp,content,chunk_index,
                           metadata_json,status,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        'cas-chunk', chunk_citation, 'trove://fixture/vector-cas/parent',
                        'cas-account', 'CAS', 'message', 'cas-conversation', 'CAS conversation',
                        'CAS sender', '2026-01-01T00:00:00Z', 'inserted chunk', 0, '{}',
                        'active', '2026-01-01T00:00:00Z',
                    ),
                )
                conn.commit()
            self.assertEqual(revision(), before + 4)

            with store.connect() as conn:
                conn.execute(
                    'UPDATE evidence_chunks SET content=? WHERE chunk_citation=?',
                    ('updated chunk', chunk_citation),
                )
                conn.commit()
            self.assertEqual(revision(), before + 5)

            with store.connect() as conn:
                conn.execute('DELETE FROM evidence_chunks WHERE chunk_citation=?', (chunk_citation,))
                conn.commit()
            self.assertEqual(revision(), before + 6)

    def test_incompatible_incremental_generation_fails_before_scan_or_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            provider = FakeEmbeddingProvider(dimensions=16)
            status = {
                'collection_exists': True,
                'rebuild_required': True,
                'provider_mismatch': True,
                'incomplete': False,
                'reason_code': 'zvec_rebuild_required',
            }

            with patch('trove_core.runtime.ZVecStore.status', return_value=status), \
                 patch('trove_core.runtime._prepare_incremental_vectors') as prepare:
                with self.assertRaisesRegex(VectorFullRebuildRequired, 'zvec_rebuild_required'):
                    index_vectors(
                        cfg,
                        provider,
                        backend='zvec',
                        citations=['trove://fixture/dirty'],
                    )

            prepare.assert_not_called()

    def test_v24_upgrade_installs_revision_without_scanning_vector_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vault.sqlite'
            SQLiteStore(path).initialize()
            with sqlite3.connect(path) as conn:
                for name in VECTOR_SOURCE_TRIGGER_NAMES:
                    conn.execute(f'DROP TRIGGER "{name}"')
                conn.execute("DELETE FROM schema_meta WHERE key='vector_source_revision'")
                conn.execute("UPDATE schema_meta SET value='24' WHERE key='schema_version'")
                conn.execute('PRAGMA user_version=24')
                conn.commit()

                statements: list[str] = []
                conn.set_trace_callback(statements.append)
                try:
                    self.assertEqual(migrate_schema(conn), SCHEMA_VERSION)
                finally:
                    conn.set_trace_callback(None)

                source_selects = [
                    statement for statement in statements
                    if statement.lstrip().upper().startswith('SELECT')
                    and (' FROM MESSAGES' in statement.upper() or ' FROM EVIDENCE_CHUNKS' in statement.upper())
                ]
                self.assertEqual(source_selects, [])
                self.assertEqual(conn.execute(
                    "SELECT value FROM schema_meta WHERE key='vector_source_revision'"
                ).fetchone()[0], '0')
                installed = {
                    str(row[0]) for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'vector_source_%'"
                    )
                }
                self.assertEqual(installed, set(VECTOR_SOURCE_TRIGGER_NAMES))

    def test_unrelated_profile_and_media_writes_during_embedding_still_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            index_fixture_vault(vault, reset=True)
            cfg = VaultConfig.resolve(str(vault), env={})
            store = SQLiteStore(cfg.paths.sqlite_path)
            snapshot_before = _vector_source_snapshot(cfg)

            def write_unrelated_rows() -> None:
                with store.connect() as conn:
                    conn.execute(
                        """INSERT INTO media_source_state(
                               source_key,file_fingerprint,table_fingerprint,row_watermark,row_count,updated_at
                           ) VALUES('vector-cas-media','file','table',1,1,'2026-01-01T00:00:00Z')"""
                    )
                    conn.execute(
                        """INSERT INTO entities(
                               entity_id,entity_type,display_name,identifiers_json,status,confidence,
                               created_at,updated_at
                           ) VALUES('vector-cas-person','person','CAS person','{}','active',1,
                                    '2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')"""
                    )
                    conn.execute(
                        """INSERT INTO profile_snapshots(
                               profile_id,entity_id,version,projection_json,content_hash,source_revision,
                               schema_version,completeness_state,evidence_citations_json,
                               enrichment_summary_json,gaps_json,created_at
                           ) VALUES('vector-cas-profile','vector-cas-person',1,'{}','hash','source',
                                    'customer-profile/test','complete','[]','{}','[]',
                                    '2026-01-01T00:00:00Z')"""
                    )
                    conn.commit()

            provider = _CloudEmbeddingProvider(on_first_embed=write_unrelated_rows)
            payload, grant = self._approved_request(cfg, provider)
            result = index_vectors(
                cfg,
                provider,
                backend='sqlite',
                batch_size=16,
                max_messages=2,
                approval_grant=grant,
                approval_payload=payload,
            )

            self.assertTrue(result['ok'])
            self.assertGreater(provider.calls, 0)
            self.assertEqual(_vector_source_snapshot(cfg), snapshot_before)
            with store.connect() as conn:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM profile_snapshots WHERE profile_id='vector-cas-profile'"
                ).fetchone()[0], 1)
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM media_source_state WHERE source_key='vector-cas-media'"
                ).fetchone()[0], 1)

    def test_message_or_evidence_change_during_embedding_rejects_publish(self):
        mutations = {
            'message': "UPDATE messages SET content=content || ' changed during embedding' WHERE id=(SELECT MIN(id) FROM messages)",
            'evidence': "UPDATE evidence_chunks SET content=content || ' changed during embedding' WHERE rowid=(SELECT MIN(rowid) FROM evidence_chunks WHERE status='active')",
        }
        for label, sql in mutations.items():
            with self.subTest(source=label), tempfile.TemporaryDirectory() as directory:
                vault = Path(directory) / 'vault'
                index_fixture_vault(vault, reset=True)
                cfg = VaultConfig.resolve(str(vault), env={})
                store = SQLiteStore(cfg.paths.sqlite_path)

                def change_source() -> None:
                    with store.connect() as conn:
                        cursor = conn.execute(sql)
                        self.assertEqual(cursor.rowcount, 1)
                        conn.commit()

                provider = _CloudEmbeddingProvider(on_first_embed=change_source)
                payload, grant = self._approved_request(cfg, provider)
                with self.assertRaises(VectorIndexSourceChanged):
                    index_vectors(
                        cfg,
                        provider,
                        backend='sqlite',
                        batch_size=16,
                        max_messages=2,
                        approval_grant=grant,
                        approval_payload=payload,
                    )
                self.assertGreater(provider.calls, 0)


if __name__ == '__main__':
    unittest.main()
