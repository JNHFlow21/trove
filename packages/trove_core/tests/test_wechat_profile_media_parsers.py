from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from trove_core.store.sqlite_store import SQLiteStore
from trove_core.store.repositories import MultimodalRepository
from trove_core.wechat.importers.contacts import ContactIdentityImporter
from trove_core.wechat.media.resources import discover_media_assets
from trove_core.wechat.parsers.contact_extra import parse_contact_extra_buffer
from trove_core.wechat.parsers.packed_info import parse_packed_info_blob


class WeChatProfileMediaParserTests(unittest.TestCase):
    def test_contact_extra_parser_only_promotes_known_schema(self):
        parsed = parse_contact_extra_buffer(json.dumps({'signature': '签名needle', 'region': '示例城市甲', 'gender': '2'}).encode())
        unknown = parse_contact_extra_buffer(b'\x01random readable signature maybe fake')

        self.assertEqual(parsed.fields['signature'], '签名needle')
        self.assertEqual(parsed.fields['region'], '示例城市甲')
        self.assertEqual(unknown.fields, {})
        self.assertEqual(unknown.diagnostics['status'], 'unknown_schema')

    def test_contact_importer_writes_extra_buffer_observations(self):
        with tempfile.TemporaryDirectory() as d:
            contact_db = Path(d) / 'contact.db'
            with sqlite3.connect(contact_db) as conn:
                conn.execute('CREATE TABLE contact(username TEXT, remark TEXT, nick_name TEXT, alias TEXT, extra_buffer BLOB)')
                conn.execute('INSERT INTO contact VALUES(?,?,?,?,?)', ('wxid_friend', '客户', '', '', json.dumps({'signature': '签名needle', 'region': '示例城市甲'}).encode()))
                conn.commit()
            store = SQLiteStore(Path(d) / 'trove.sqlite')
            repo = MultimodalRepository(store)

            count = ContactIdentityImporter(contact_db, account_id='acct-a').import_to_ontology(repo)

            self.assertEqual(count, 1)
            with store.connect() as conn:
                rows = list(conn.execute("SELECT observation_type,value_json FROM observations WHERE source_type='contact'"))
            values = {(r['observation_type'], r['value_json']) for r in rows}
            self.assertTrue(any(t == 'signature' and '签名needle' in v for t, v in values))
            self.assertTrue(any(t == 'region' and '示例城市甲' in v for t, v in values))

    def test_packed_info_parser_and_media_discovery_use_path_hints(self):
        with tempfile.TemporaryDirectory() as d:
            acct = Path(d) / 'acct'
            media_dir = acct / 'media'
            media_dir.mkdir(parents=True)
            (media_dir / 'voice.amr').write_bytes(b'voice')
            with sqlite3.connect(acct / 'message_resource.db') as conn:
                conn.execute('CREATE TABLE MessageResourceDetail(resource_id TEXT, type TEXT, packed_info BLOB)')
                payload = json.dumps({'file_path': 'media/voice.amr', 'type': 'voice'}).encode()
                conn.execute('INSERT INTO MessageResourceDetail VALUES(?,?,?)', ('r1', 'voice', payload))
                conn.commit()

            parsed = parse_packed_info_blob(json.dumps({'file_path': 'media/voice.amr'}).encode())
            refs = discover_media_assets(acct, account_id='acct-a')

            self.assertEqual(parsed.path_hints, ['media/voice.amr'])
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].modality, 'voice')
            self.assertEqual(refs[0].cache_state, 'source_available')
            self.assertIsNone(refs[0].path_ref)


if __name__ == '__main__':
    unittest.main()
