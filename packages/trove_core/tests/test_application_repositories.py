from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from trove_core.application.repositories import RepositoryFacade, SQLiteUnitOfWork
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.models import Account, Conversation, Message


def _vault(tmp_path: Path) -> VaultConfig:
    cfg = VaultConfig.resolve(str(tmp_path / 'vault'))
    cfg.paths.index_dir.mkdir(parents=True)
    store = SQLiteStore(cfg.paths.sqlite_path)
    store.initialize()
    store.upsert_accounts([Account('a1', 'Account', 'Account')])
    store.upsert_conversations([
        Conversation('c1', 'a1', 'Alice', 'private'),
        Conversation('c2', 'a1', 'Alice Team', 'group'),
    ])
    store.upsert_messages([
        Message('a1', 'Account', 'c1', 'Alice', 'private', 'u1', 'Alice', datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc), 'hello', 's1', 1),
        Message('a1', 'Account', 'c2', 'Alice Team', 'group', 'u2', 'Bob', datetime(2026, 1, 3, 3, 4, 5, tzinfo=timezone.utc), 'world', 's1', 2),
    ])
    store.close_all()
    return cfg


class ApplicationRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.cfg = _vault(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_vertical_repositories_share_one_unit_of_work(self) -> None:
        with SQLiteUnitOfWork(self.cfg) as uow:
            facade = RepositoryFacade.from_uow(uow)
            self.assertEqual([row['conversation_id'] for row in facade.messages.list_conversations(limit=10)], ['c1', 'c2'])
            self.assertEqual(facade.search.list_contacts(limit=10), [])
            citation = 'trove://wechat/a1/c1/s1/1'
            self.assertIsNotNone(facade.evidence.by_citation(citation))
            self.assertTrue(facade.evidence.citation_matches(
                citation,
                {'conversation_id': 'c1', 'since': '2026-01-01T00:00:00Z', 'until': '2026-02-01T00:00:00Z'},
            ))
            self.assertFalse(facade.evidence.citation_matches(citation, {'conversation_id': 'c2'}))

    def test_contact_resolution_candidates_are_repository_owned(self) -> None:
        with SQLiteUnitOfWork(self.cfg) as uow:
            exact = uow.messages.conversation_candidates('Alice', limit=10)
            self.assertEqual([(row['conversation_id'], row['match_source']) for row in exact], [
                ('c1', 'conversation'),
                ('c2', 'conversation'),
            ])
            self.assertEqual(uow.messages.conversation_candidates('nobody', limit=10), [])

    def test_unit_of_work_closes_owned_store(self) -> None:
        uow = SQLiteUnitOfWork(self.cfg)
        with uow:
            self.assertGreaterEqual(uow.store.active_connection_count, 1)
            uow.messages.list_conversations(limit=1)
            self.assertGreaterEqual(uow.store.active_connection_count, 1)
        self.assertEqual(uow.store.active_connection_count, 0)
