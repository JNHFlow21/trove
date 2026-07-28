from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trove_core.agent_tools import tools as agent_tools
from trove_core.approvals import ApprovalManager
from trove_core.media_fetch import fetch_media
from trove_core.store.repositories import MultimodalRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.wechat.importers.moments import MomentsImporter
from trove_core.wechat.media.materializer import _host_allowed


PNG = b'\x89PNG\r\n\x1a\n' + b'fixture-remote-png'


def create_remote_moment(vault: Path, url: str) -> tuple[str, SQLiteStore]:
    cfg = VaultConfig.resolve(str(vault), env={})
    cfg.ensure()
    account = vault / 'sources' / 'account-a'
    account.mkdir(parents=True)
    db = account / 'sns.db'
    xml = (
        '<TimelineObject><id>remote-native</id><username>wxid-a</username><createTime>1760000000</createTime>'
        '<contentDesc>remote</contentDesc><ContentObject><mediaList><media><id>m1</id><type>2</type>'
        f'<url>{url}</url></media></mediaList></ContentObject></TimelineObject>'
    )
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE SnsTimeLine(tid TEXT,user_name TEXT,content TEXT,pack_info_buf BLOB)')
        conn.execute('INSERT INTO SnsTimeLine VALUES(?,?,?,?)', ('feed', 'wxid-a', xml, b''))
        conn.commit()
    store = SQLiteStore(cfg.paths.sqlite_path)
    MomentsImporter(db, account_id='acct-a').import_to_store(MultimodalRepository(store))
    with store.connect() as conn:
        citation = conn.execute("SELECT citation FROM media_assets WHERE source_type='moment'").fetchone()[0]
    return str(citation), store


class LazyMediaRemoteFetchTests(unittest.TestCase):
    def test_known_message_sticker_hosts_are_allowlisted_exactly(self):
        self.assertTrue(_host_allowed('vweixinf.tc.qq.com'))
        self.assertTrue(_host_allowed('wxapp.tc.qq.com'))
        self.assertFalse(_host_allowed('attacker.tc.qq.com'))

    def test_exact_approval_allows_one_bounded_remote_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            citation, store = create_remote_moment(vault, 'https://mmbiz.qpic.cn/sz_mmbiz_png/fixture?wx_fmt=png')
            prepared = fetch_media(vault, citation, allow_remote=True)
            payload = prepared['approval_payload']
            manager = ApprovalManager(vault)
            record = manager.request('wechat_cdn_fetch', 'remote_media_fetch', payload)
            manager.decide(record.approval_id, 'approved')

            def fake_fetch(url, output, *, max_bytes):
                self.assertIn('qpic.cn', url)
                self.assertGreaterEqual(max_bytes, len(PNG))
                output.write(PNG)
                output.flush()
                return hashlib.sha256(PNG).hexdigest(), len(PNG), PNG, 'image/png'

            with patch('trove_core.wechat.media.materializer._fetch_remote', side_effect=fake_fetch) as network:
                result = agent_tools.media_fetch(
                    vault,
                    citation,
                    allow_remote=True,
                    approval_id=record.approval_id,
                )
                cached = agent_tools.media_fetch(vault, citation, allow_remote=True)

            self.assertEqual(prepared['code'], 'approval_required')
            self.assertNotIn('qpic.cn', str(payload))
            self.assertTrue(result['ok'], result)
            self.assertTrue(cached['ok'])
            network.assert_called_once()
            with store.connect() as conn:
                asset = conn.execute("SELECT path_ref,content_hash,cache_state FROM media_assets WHERE source_type='moment'").fetchone()
            self.assertFalse(Path(asset['path_ref']).is_absolute())
            self.assertEqual(asset['content_hash'], hashlib.sha256(PNG).hexdigest())
            self.assertEqual(asset['cache_state'], 'cached')

    def test_ssrf_locator_is_rejected_after_exact_approval_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / 'vault'
            citation, _store = create_remote_moment(vault, 'https://127.0.0.1/private.png')
            prepared = fetch_media(vault, citation, allow_remote=True)
            payload = prepared['approval_payload']
            manager = ApprovalManager(vault)
            record = manager.request('wechat_cdn_fetch', 'remote_media_fetch', payload)
            manager.decide(record.approval_id, 'approved')

            with patch('socket.create_connection', side_effect=AssertionError('network must not run')) as network:
                result = agent_tools.media_fetch(
                    vault,
                    citation,
                    allow_remote=True,
                    approval_id=record.approval_id,
                )

            self.assertFalse(result['ok'])
            self.assertEqual(result['reason'], 'remote_host_not_allowlisted')
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
