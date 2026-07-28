from __future__ import annotations

from dataclasses import replace
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.store.repositories import WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.coordinator import VaultOperationCoordinator
from trove_core.wechat.appmsg_backfill import appmsg_backfill_plan, backfill_appmsg_payloads, recover_appmsg_payload
from trove_core.wechat.importers.wechat_decrypted import WeChatDecryptedAccountImporter, msg_table_for
from trove_core.wechat.import_job import run_import_job


def create_appmsg_source(root: Path) -> Path:
    account = root / 'com.tencent.xinWeChat__wxid_ownerfixture'
    account.mkdir(parents=True)
    with sqlite3.connect(account / 'contact.db') as conn:
        conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT)')
        conn.execute('CREATE TABLE chatroom_member (chatroom TEXT, member TEXT)')
        conn.execute(
            'INSERT INTO contact(username,remark,nick_name,alias) VALUES (?,?,?,?)',
            ('wxid_personfixture', 'AppMsg Fixture', '', ''),
        )
        conn.commit()
    table = msg_table_for('wxid_personfixture')
    with sqlite3.connect(account / 'message_0.db') as conn:
        conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INTEGER)')
        conn.execute('INSERT INTO Name2Id(rowid,user_name,is_session) VALUES (?,?,?)', (1, 'wxid_personfixture', 1))
        conn.execute(f'''CREATE TABLE {table} (
            local_id INTEGER, local_type INTEGER, real_sender_id INTEGER, create_time INTEGER,
            message_content TEXT, compress_content BLOB, WCDB_CT_message_content BLOB
        )''')
        conn.execute(
            f'INSERT INTO {table}(local_id,local_type,real_sender_id,create_time,message_content) VALUES (?,?,?,?,?)',
            (
                1, 49, 1, 1710000000,
                '<msg><appmsg><type>5</type><title>修复后的应用卡片</title>'
                '<url>https://example.com/private?token=never-persist</url></appmsg></msg>',
            ),
        )
        conn.commit()
    return account


class AppMsgBackfillTests(unittest.TestCase):
    def test_profile_enrichment_marks_missing_historical_appmsg_source_as_terminal_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            source = vault / 'decrypted' / 'runs' / 'appmsg-missing-source'
            account = create_appmsg_source(source)
            run_import_job(vault, [source], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                conn.execute(
                    "UPDATE message_payloads SET parse_status='malformed',normalized_type='unsupported',normalized_json='{}',display_text='[appmsg]',unsupported_reason='malformed_xml'"
                )
                conn.commit()
            table = msg_table_for('wxid_personfixture')
            with sqlite3.connect(account / 'message_0.db') as conn:
                conn.execute(f'DELETE FROM "{table}" WHERE local_id=1')
                conn.commit()
            manifest = agent_tools.profile_enrichment_plan(
                vault, 'AppMsg Fixture', actor='operator', session='missing-source-session', item_budget=20,
            )
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='missing-source-session',
                worker='local-worker', execution_location='local',
            )

            result = agent_tools.profile_enrichment_appmsg_execute(
                vault, claim['task']['task_id'], actor='operator', session='missing-source-session',
                worker='local-worker', claim_token=claim['agent_action']['claim_token'],
            )

            self.assertFalse(result['ok'])
            self.assertEqual(result['reason'], 'source_appmsg_not_found')
            self.assertEqual(result['profile_enrichment']['state'], 'complete_with_terminal_gaps')
            status = agent_tools.profile_enrichment_status(
                vault, manifest['run_id'], actor='operator', session='missing-source-session',
            )
            self.assertEqual(status['state'], 'complete_with_terminal_gaps')

    def test_profile_enrichment_executes_appmsg_recovery_and_completes_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            source = vault / 'decrypted' / 'runs' / 'appmsg-profile'
            create_appmsg_source(source)
            run_import_job(vault, [source], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                citation = conn.execute("SELECT citation FROM messages WHERE content_kind='appmsg'").fetchone()[0]
                conn.execute("UPDATE messages SET content='[appmsg]' WHERE citation=?", (citation,))
                conn.execute(
                    "UPDATE message_payloads SET parse_status='malformed',normalized_type='unsupported',normalized_json='{}',display_text='[appmsg]',unsupported_reason='malformed_xml' WHERE citation=?",
                    (citation,),
                )
                conn.commit()
            manifest = agent_tools.profile_enrichment_plan(
                vault, 'AppMsg Fixture', actor='operator', session='appmsg-session', item_budget=20,
            )
            claim = agent_tools.profile_enrichment_claim(
                vault, manifest['run_id'], actor='operator', session='appmsg-session',
                worker='local-worker', execution_location='local',
            )

            result = agent_tools.profile_enrichment_appmsg_execute(
                vault, claim['task']['task_id'], actor='operator', session='appmsg-session',
                worker='local-worker', claim_token=claim['agent_action']['claim_token'],
            )

            self.assertTrue(result['ok'], result)
            self.assertEqual(result['execution_path'], 'local_appmsg_recovery')
            self.assertEqual(result['profile_enrichment']['task']['state'], 'completed')
            status = agent_tools.profile_enrichment_status(
                vault, manifest['run_id'], actor='operator', session='appmsg-session',
            )
            self.assertEqual(status['state'], 'complete')

    def test_exact_recovery_repairs_one_historical_malformed_payload_from_bound_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / 'vault'
            source = vault / 'decrypted' / 'runs' / 'appmsg-exact'
            create_appmsg_source(source)
            run_import_job(vault, [source], reset_index=True)
            store = SQLiteStore(vault / 'index' / 'trove.sqlite')
            with store.connect() as conn:
                citation = conn.execute("SELECT citation FROM messages WHERE content_kind='appmsg'").fetchone()[0]
                conn.execute("UPDATE messages SET content='[appmsg]' WHERE citation=?", (citation,))
                conn.execute(
                    "UPDATE message_payloads SET parse_status='malformed',normalized_type='unsupported',normalized_json='{}',display_text='[appmsg]',unsupported_reason='malformed_xml' WHERE citation=?",
                    (citation,),
                )
                conn.commit()

            from trove_core.wechat import appmsg_backfill as appmsg_module
            original_recovery = appmsg_module._exact_appmsg_from_snapshot

            def recovery_without_writer(*args, **kwargs):
                with VaultOperationCoordinator(vault).write(owner='probe-appmsg-source'):
                    pass
                return original_recovery(*args, **kwargs)

            with patch(
                'trove_core.wechat.appmsg_backfill._exact_appmsg_from_snapshot',
                recovery_without_writer,
            ):
                result = recover_appmsg_payload(vault, citation)

            self.assertTrue(result['ok'], result)
            self.assertEqual(result['parse_status'], 'parsed')
            self.assertEqual(result['changed'], 1)
            self.assertFalse(result['raw_content_included'])
            with store.connect() as conn:
                row = conn.execute(
                    'SELECT m.content,p.parse_status FROM messages m JOIN message_payloads p ON p.citation=m.citation WHERE m.citation=?',
                    (citation,),
                ).fetchone()
            self.assertIn('修复后的应用卡片', row['content'])
            self.assertEqual(row['parse_status'], 'parsed')

    def test_backfill_repairs_existing_placeholder_and_rerun_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'current'
            account = create_appmsg_source(source)
            accounts, conversations, messages = WeChatDecryptedAccountImporter(account).load()
            normalized = messages[0]
            placeholder = replace(normalized, content='[appmsg]', normalized_payload=None)
            vault = root / 'vault'
            cfg = VaultConfig.resolve(str(vault), env={})
            cfg.ensure()
            store = SQLiteStore(cfg.paths.sqlite_path)
            WeChatRepository(store).replace_fixture(accounts, conversations, [placeholder])

            dry_run = appmsg_backfill_plan(vault, source)
            with store.connect() as conn:
                before = conn.execute('SELECT content FROM messages').fetchone()['content']
            first = backfill_appmsg_payloads(vault, source)
            second = backfill_appmsg_payloads(vault, source)

            self.assertEqual(dry_run['would_change'], 1)
            self.assertEqual(before, '[appmsg]')
            self.assertEqual(first['backfill']['changed'], 1)
            self.assertEqual(second['backfill']['changed'], 0)
            self.assertFalse(first['raw_content_included'])
            self.assertFalse(first['raw_paths_included'])
            with store.connect() as conn:
                row = conn.execute(
                    'SELECT m.content,p.normalized_json,p.parse_status FROM messages m JOIN message_payloads p ON p.citation=m.citation'
                ).fetchone()
            self.assertIn('修复后的应用卡片', row['content'])
            self.assertEqual(row['parse_status'], 'parsed')
            self.assertNotIn('never-persist', row['normalized_json'])
            self.assertNotIn('/private', row['normalized_json'])


if __name__ == '__main__':
    unittest.main()
