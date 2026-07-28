from __future__ import annotations

import unittest

from trove_core.wechat.scope import classify_media_reference, classify_wechat_identity, public_scope_contract


class WeChatScopeClassifierTests(unittest.TestCase):
    def test_positive_five_families(self):
        self.assertEqual(classify_wechat_identity('wxid_alice', has_chat_history=True).scope_type, 'private_chat')
        self.assertEqual(classify_wechat_identity('room@chatroom').scope_type, 'group_chat')
        self.assertEqual(classify_wechat_identity('wxid_alice', source_family='contact').scope_type, 'contact')
        self.assertEqual(classify_wechat_identity('', source_family='moment').scope_type, 'moment')
        self.assertEqual(classify_wechat_identity('gh_saved_article', source_family='favorite').scope_type, 'favorite')

    def test_excluded_identities_do_not_become_private(self):
        cases = {
            'gh_demo': 'excluded_public_account',
            'weixin': 'excluded_system',
            'notifymessage': 'excluded_notification',
            'fmessage': 'excluded_system',
            'medianote': 'excluded_system',
            'filehelper': 'excluded_file_helper',
            'brandservice_fold': 'excluded_subscription',
            'wxapp_service': 'excluded_mini_program',
        }
        for username, expected in cases.items():
            with self.subTest(username=username):
                decision = classify_wechat_identity(username, has_chat_history=True)
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.scope_type, expected)

    def test_unknown_without_history_is_excluded(self):
        decision = classify_wechat_identity('mystery-source')
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.scope_type, 'excluded_unknown')

    def test_orphan_media_excluded_until_linked_to_accepted_citation(self):
        self.assertFalse(classify_media_reference('wechat_metadata', 'trove://wechat/acct/media/cache').allowed)
        self.assertTrue(classify_media_reference('moment', 'trove://wechat/acct/moment/m1').allowed)
        self.assertTrue(classify_media_reference('favorite', 'trove://wechat/acct/favorite/f1').allowed)

    def test_public_contract_documents_favorites_policy(self):
        contract = public_scope_contract()
        self.assertIn('private_chat', contract['included_families'])
        self.assertIn('Favorites are local knowledge evidence', contract['favorites_policy'])


if __name__ == '__main__':
    unittest.main()
