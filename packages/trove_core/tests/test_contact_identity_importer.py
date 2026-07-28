from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.wechat.importers.contacts import ContactIdentityImporter


class ContactIdentityImporterTests(unittest.TestCase):
    def test_contact_metadata_creates_identity_observations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / 'contact.db'
            conn = sqlite3.connect(db)
            conn.execute('CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, signature TEXT, big_head_url TEXT)')
            conn.execute('INSERT INTO contact VALUES(?,?,?,?,?,?)', ('wxid-a', '示例教育示例人物丁', '示例人物丁', 'example_edu', '负责校区采购', 'avatar-hash-only'))
            conn.commit(); conn.close()
            importer = ContactIdentityImporter(db, account_id='acct-a')
            contacts = importer.load()
            self.assertEqual(len(contacts), 1)
            self.assertEqual(contacts[0].display_name, '示例教育示例人物丁')
            repo = MultimodalRepository(SQLiteStore(root / 'vault.sqlite'))
            self.assertEqual(importer.import_to_ontology(repo), 1)
            with repo.store.connect() as dbconn:
                self.assertEqual(dbconn.execute('SELECT COUNT(*) FROM entities').fetchone()[0], 1)
                self.assertGreaterEqual(dbconn.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 4)
                self.assertGreaterEqual(dbconn.execute('SELECT COUNT(*) FROM entity_identifiers').fetchone()[0], 3)


if __name__ == '__main__':
    unittest.main()
