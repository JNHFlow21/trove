from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.runtime import _reconcile_sorted_vector_documents
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vector.ledger import VectorIndexLedger


class VectorIndexLedgerTests(unittest.TestCase):
    def _begin(self, ledger: VectorIndexLedger, generation_id: str, *, expected_count: int = 0):
        return ledger.begin_generation(
            generation_id,
            vector_text_version=3,
            embedding_provider='fake',
            embedding_model='fixture',
            dimensions=8,
            expected_count=expected_count,
        )

    def test_single_citation_delta_stays_bounded_with_large_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ledger = VectorIndexLedger(store)
            self._begin(ledger, 'large', expected_count=10_000)
            for start in range(0, 10_000, 1_000):
                ledger.apply_delta(
                    'large',
                    upserts=(
                        (f'fixture://message/{value}', f'hash-{value}')
                        for value in range(start, start + 1_000)
                    ),
                )

            metrics = ledger.apply_delta(
                'large',
                upserts=[('fixture://message/5000', 'changed-hash')],
                expected_count=10_000,
            )

            self.assertEqual(metrics['candidate_rows'], 1)
            self.assertEqual(metrics['count_delta'], 0)
            self.assertEqual(ledger.generation('large').indexed_count, 10_000)
            self.assertEqual(
                ledger.hashes('large', ['fixture://message/5000']),
                {'fixture://message/5000': 'changed-hash'},
            )

    def test_revision_changes_only_for_actual_transactional_delta(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ledger = VectorIndexLedger(store)
            generation = self._begin(ledger, 'revision', expected_count=1)
            self.assertEqual(generation.revision, 1)

            inserted = ledger.apply_delta(
                'revision',
                upserts=[('fixture://message/1', 'hash-a')],
                expected_count=1,
            )
            self.assertEqual(inserted['revision_incremented'], 1)
            self.assertEqual(ledger.generation('revision').revision, 2)

            replayed = ledger.apply_delta(
                'revision',
                upserts=[('fixture://message/1', 'hash-a')],
                expected_count=1,
            )
            self.assertEqual(replayed['revision_incremented'], 0)
            self.assertEqual(ledger.generation('revision').revision, 2)

            expected_only = ledger.apply_delta('revision', expected_count=2)
            self.assertEqual(expected_only['revision_incremented'], 0)
            self.assertEqual(ledger.generation('revision').revision, 2)

            changed = ledger.apply_delta(
                'revision',
                upserts=[('fixture://message/1', 'hash-b')],
            )
            self.assertEqual(changed['revision_incremented'], 1)
            self.assertEqual(ledger.generation('revision').revision, 3)

            missing_delete = ledger.apply_delta('revision', deletes=['fixture://missing'])
            self.assertEqual(missing_delete['revision_incremented'], 0)
            self.assertEqual(ledger.generation('revision').revision, 3)

            deleted = ledger.apply_delta('revision', deletes=['fixture://message/1'])
            self.assertEqual(deleted['revision_incremented'], 1)
            self.assertEqual(ledger.generation('revision').revision, 4)

            replayed_delete = ledger.apply_delta('revision', deletes=['fixture://message/1'])
            self.assertEqual(replayed_delete['revision_incremented'], 0)
            self.assertEqual(ledger.generation('revision').revision, 4)

    def test_discard_and_retired_prune_explicitly_remove_rows_with_foreign_keys_off(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ledger = VectorIndexLedger(store)
            self._begin(ledger, 'discard-me', expected_count=1)
            ledger.apply_delta('discard-me', upserts=[('fixture://discard', 'hash')])
            with store.connect() as conn:
                self.assertEqual(int(conn.execute('PRAGMA foreign_keys').fetchone()[0]), 0)

            self.assertEqual(ledger.discard('discard-me'), 1)
            with store.connect() as conn:
                self.assertEqual(int(conn.execute(
                    "SELECT COUNT(*) FROM vector_index_ledger WHERE generation_id='discard-me'"
                ).fetchone()[0]), 0)
                self.assertEqual(int(conn.execute(
                    "SELECT COUNT(*) FROM vector_index_generations WHERE generation_id='discard-me'"
                ).fetchone()[0]), 0)

            for generation_id in ('old', 'current'):
                self._begin(ledger, generation_id, expected_count=1)
                ledger.apply_delta(
                    generation_id,
                    upserts=[(f'fixture://{generation_id}', f'hash-{generation_id}')],
                )
                ledger.mark_ready(generation_id, expected_count=1)
                ledger.activate(generation_id)

            self.assertEqual(ledger.generation('old').status, 'retired')
            self.assertEqual(ledger.prune_retired(), 1)
            self.assertIsNone(ledger.generation('old'))
            self.assertEqual(ledger.active_generation().generation_id, 'current')
            with store.connect() as conn:
                self.assertEqual(int(conn.execute(
                    "SELECT COUNT(*) FROM vector_index_ledger WHERE generation_id='old'"
                ).fetchone()[0]), 0)
                self.assertEqual(int(conn.execute(
                    "SELECT COUNT(*) FROM vector_index_ledger WHERE generation_id='current'"
                ).fetchone()[0]), 1)

    def test_activation_rejects_building_generation(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ledger = VectorIndexLedger(store)
            self._begin(ledger, 'building')

            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                ledger.activate('building')
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                ledger.mark_ready('building', expected_count=1)
            with self.assertRaisesRegex(RuntimeError, 'does not accept deltas'):
                ledger.apply_delta('missing', upserts=[('fixture://orphan', 'hash')])

            self.assertEqual(ledger.generation('building').status, 'building')
            self.assertIsNone(ledger.active_generation())

    def test_dirty_parent_lookup_returns_only_exact_and_chunk_range(self):
        with tempfile.TemporaryDirectory() as d:
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            ledger = VectorIndexLedger(store)
            self._begin(ledger, 'dirty', expected_count=6)
            parent = 'fixture://message/1'
            ledger.apply_delta('dirty', upserts=[
                (parent, 'parent-hash'),
                (parent + '#chunk-0001', 'chunk-1'),
                (parent + '#chunk-0002', 'chunk-2'),
                (parent + '#chunk.other', 'not-a-child'),
                ('fixture://message/10#chunk-0001', 'other-prefix'),
                ('fixture://message/2', 'other-parent'),
            ])

            citations = ledger.citations_for_dirty('dirty', [parent])

            self.assertEqual(citations, [
                parent,
                parent + '#chunk-0001',
                parent + '#chunk-0002',
            ])
            self.assertEqual(
                list(ledger.iter_entries('dirty', batch_size=2)),
                sorted([
                    (parent, 'parent-hash'),
                    (parent + '#chunk-0001', 'chunk-1'),
                    (parent + '#chunk-0002', 'chunk-2'),
                    (parent + '#chunk.other', 'not-a-child'),
                    ('fixture://message/10#chunk-0001', 'other-prefix'),
                    ('fixture://message/2', 'other-parent'),
                ]),
            )

    def test_sorted_reconciliation_keeps_only_changes_and_stale_keys(self):
        import hashlib

        def digest(value: str) -> str:
            return hashlib.sha256(value.encode('utf-8')).hexdigest()

        source = [
            {'citation': 'fixture://a', 'vector_text': 'unchanged'},
            {'citation': 'fixture://c', 'vector_text': 'changed'},
            {'citation': 'fixture://d', 'vector_text': 'new'},
        ]
        ledger = [
            ('fixture://a', digest('unchanged')),
            ('fixture://b', digest('stale')),
            ('fixture://c', digest('old')),
        ]

        changed, deletes, count = _reconcile_sorted_vector_documents(source, ledger)

        self.assertEqual([row['citation'] for row in changed], ['fixture://c', 'fixture://d'])
        self.assertEqual(deletes, ['fixture://b'])
        self.assertEqual(count, 3)
        self.assertTrue(all('content_hash' in row for row in changed))


if __name__ == '__main__':
    unittest.main()
