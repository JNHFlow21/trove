from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import sqlite3

from trove_core.wechat.source_inventory import inventory

class SourceInventoryTests(unittest.TestCase):
    def test_redacts_paths_and_marks_importable_decrypted_current(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'decrypted' / 'current' / 'acct'
            root.mkdir(parents=True)
            (root / 'message_0.db').write_bytes(b'not sqlite but source-like')
            candidates = inventory([root])
            self.assertEqual(len(candidates), 1)
            c = candidates[0]
            self.assertEqual(c.category, 'runtime decrypted DB copies')
            self.assertTrue(c.importable)
            self.assertNotIn('/Users/', c.redacted_path)

    def test_key_material_is_sensitive_and_not_importable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'key_store.json'
            p.write_text('{}')
            c = inventory([p])[0]
            self.assertTrue(c.sensitive)
            self.assertFalse(c.importable)

    def test_scope_counts_are_redacted_for_account_dir(self):
        with tempfile.TemporaryDirectory() as d:
            acct = Path(d) / 'decrypted' / 'current' / 'acct'
            acct.mkdir(parents=True)
            with sqlite3.connect(acct / 'contact.db') as conn:
                conn.execute('CREATE TABLE contact(username TEXT)')
                conn.execute('INSERT INTO contact VALUES (?)', ('wxid-a',))
                conn.execute('INSERT INTO contact VALUES (?)', ('gh_public',))
            with sqlite3.connect(acct / 'message_0.db') as conn:
                conn.execute('CREATE TABLE Name2Id(user_name TEXT)')
                conn.execute('INSERT INTO Name2Id VALUES (?)', ('wxid-a',))
                conn.execute('INSERT INTO Name2Id VALUES (?)', ('room@chatroom',))
                conn.execute('INSERT INTO Name2Id VALUES (?)', ('notifymessage',))
            c = inventory([acct])[0]
            self.assertGreaterEqual(c.scope_counts.get('private_chat', 0), 1)
            self.assertGreaterEqual(c.scope_counts.get('group_chat', 0), 1)
            self.assertGreaterEqual(c.excluded_counts.get('excluded_public_account', 0), 1)
