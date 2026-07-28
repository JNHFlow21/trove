from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from trove_protocol.provider import ProviderManifest
from trove_provider_wechat import create_provider

from .fixtures import write_account


PACKAGE = Path(__file__).resolve().parents[1] / 'trove_provider_wechat'


class ProviderMultiAccountTests(unittest.TestCase):
    def test_two_accounts_have_independent_records_citations_and_watermarks(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_account(root, 'account-a', label='A', token='alpha')
            removed = write_account(root, 'account-b', label='B', token='beta', rows=3)
            provider = create_provider(manifest, source_root=root)
            imported = {
                account['account_id']: provider.invoke('read', {
                    'operation': 'changes', 'account_id': account['account_id'],
                })
                for account in provider.accounts()
            }
            self.assertEqual(set(imported), {'account-a', 'account-b'})
            self.assertEqual(len(imported['account-a']['records']), 2)
            self.assertEqual(len(imported['account-b']['records']), 3)
            self.assertTrue(all('/account-a/' in row['citation'] for row in imported['account-a']['records']))
            self.assertTrue(all('/account-b/' in row['citation'] for row in imported['account-b']['records']))
            self.assertNotEqual(imported['account-a']['watermark'], imported['account-b']['watermark'])
            retained_b = list(imported['account-b']['records'])
            shutil.rmtree(removed)
            self.assertEqual([item['account_id'] for item in provider.accounts()], ['account-a'])
            self.assertEqual(imported['account-b']['records'], retained_b)


if __name__ == '__main__':
    unittest.main()
