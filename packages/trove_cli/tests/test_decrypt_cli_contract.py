from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from trove_cli.main import main


def make_account(acct: Path) -> None:
    acct.mkdir(parents=True)
    with sqlite3.connect(acct / 'contact.db') as conn:
        conn.execute('CREATE TABLE contact(username TEXT, remark TEXT)')
        conn.execute('INSERT INTO contact VALUES(?,?)', ('wxid_cli', 'CLI'))
        conn.commit()
    table = 'Msg_' + hashlib.md5(b'wxid_cli').hexdigest()
    with sqlite3.connect(acct / 'message_0.db') as conn:
        conn.execute('CREATE TABLE Name2Id(user_name TEXT,is_session INTEGER)')
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES(?,?,?)', (1, 'wxid_cli', 1))
        conn.execute(f'CREATE TABLE {table}(local_id INTEGER, real_sender_id INTEGER, create_time INTEGER, message_content TEXT)')
        conn.execute(f'INSERT INTO {table} VALUES(?,?,?,?)', (1, 1, 1710000000, 'cli decrypt import'))
        conn.commit()


class DecryptCliContractTests(unittest.TestCase):
    def run_cli(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, json.loads(buf.getvalue())

    def test_decrypt_preflight_run_status_are_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            live = root / 'live'
            vault = root / 'vault'
            make_account(live / 'com.tencent.xinWeChat__wxid_cli')

            pre_code, preflight = self.run_cli(['--vault', str(vault), 'decrypt', 'preflight', '--live-root', str(live), '--selected-account', 'wxid_cli:com.tencent.xinWeChat__wxid_cli', '--json'])
            auto_code, auto_preflight = self.run_cli([
                '--vault', str(vault), 'decrypt', 'preflight', '--live-root', str(live),
                '--all-discovered-wechat-accounts', '--json',
            ])
            run_code, run = self.run_cli(['--vault', str(vault), 'decrypt', 'run', '--live-root', str(live), '--selected-account', 'wxid_cli:com.tencent.xinWeChat__wxid_cli', '--yes', '--json'])
            known_code, known_preflight = self.run_cli([
                '--vault', str(vault), 'decrypt', 'preflight', '--live-root', str(live),
                '--known-keyed-wechat-accounts', '--json',
            ])
            status_code, status = self.run_cli(['--vault', str(vault), 'decrypt', 'status', '--json'])

            self.assertEqual(pre_code, 0)
            self.assertEqual(auto_code, 0)
            self.assertEqual(run_code, 0)
            self.assertEqual(known_code, 0)
            self.assertEqual(status_code, 0)
            self.assertTrue(preflight['ok'])
            self.assertTrue(auto_preflight['ok'])
            self.assertEqual(auto_preflight['config']['selected_account_count'], 1)
            self.assertTrue(run['ok'])
            self.assertEqual(known_preflight['config']['selected_account_count'], 1)
            self.assertTrue(status['ok'])
            joined = json.dumps({'preflight': preflight, 'auto': auto_preflight, 'known': known_preflight, 'run': run, 'status': status}, ensure_ascii=False)
            self.assertNotIn(str(live), joined)
            self.assertNotIn('com.tencent.xinWeChat__wxid_cli', joined)
            self.assertNotIn('cli decrypt import', joined)
            self.assertFalse(run['raw_paths_included'])

    def test_key_capture_activation_defaults_on_and_supports_opt_out(self):
        configs = []

        def fake_capture(config):
            configs.append(config)
            return {'ok': True, 'status': 'stored', 'raw_content_included': False, 'raw_paths_included': False}

        with patch('trove_core.wechat.decrypt.key_capture.capture_and_store_key_store', side_effect=fake_capture):
            default_code, _ = self.run_cli(['decrypt', 'key-capture', '--yes', '--json'])
            opt_out_code, _ = self.run_cli(['decrypt', 'key-capture', '--no-activate', '--yes', '--json'])

        self.assertEqual(default_code, 0)
        self.assertEqual(opt_out_code, 0)
        self.assertTrue(configs[0].activate_spawned)
        self.assertFalse(configs[1].activate_spawned)


if __name__ == '__main__':
    unittest.main()
