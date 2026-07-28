from __future__ import annotations

import json
import unittest

from trove_core.wechat.parsers.appmsg import MAX_SOURCE_BYTES, parse_appmsg


class AppMsgParserTests(unittest.TestCase):
    def test_wechat_group_sender_prefix_is_removed_before_xml_parse(self):
        link = parse_appmsg(
            'wxid_senderfixture:\n'
            '<msg><appmsg><type>5</type><title>群聊链接</title>'
            '<url>https://example.com/private?token=never-persist</url></appmsg></msg>'
        )
        location = parse_appmsg(
            'wxid_senderfixture:\n'
            '<msg><location x="31.2" y="121.4" label="群聊位置" /></msg>'
        )

        self.assertEqual((link.parse_status, link.normalized_type), ('parsed', 'link'))
        self.assertEqual(link.fields['title'], '群聊链接')
        self.assertEqual((location.parse_status, location.normalized_type), ('parsed', 'location'))
        self.assertEqual(location.fields['location_label'], '群聊位置')

    def test_known_non_appmsg_wrappers_preserve_location_call_and_generic_card_context(self):
        location = parse_appmsg('<msg><location x="31.2" y="121.4" label="见面地点" /></msg>')
        call = parse_appmsg(
            '<voipmsg><VoIPBubbleMsg><msg><room_type>1</room_type><msg_type>2</msg_type>'
            '<duration>95</duration></msg></VoIPBubbleMsg></voipmsg>'
        )
        generic = parse_appmsg('<msg><appmsg><type>62</type><title>保留卡片标题</title></appmsg></msg>')

        self.assertEqual((location.parse_status, location.normalized_type), ('parsed', 'location'))
        self.assertEqual(location.fields['location_label'], '见面地点')
        self.assertEqual((call.parse_status, call.normalized_type), ('parsed', 'call'))
        self.assertEqual(call.fields['duration_seconds'], 95)
        self.assertEqual((generic.parse_status, generic.normalized_type), ('parsed', 'generic_card'))
        self.assertEqual(generic.fields['title'], '保留卡片标题')

    def test_link_strips_query_and_keeps_non_clickable_identity(self):
        parsed = parse_appmsg(
            '<msg><appmsg><type>5</type><title>发布说明</title><des>安全摘要</des>'
            '<url>https://Example.COM/private/path?token=secret-value&amp;sig=abc</url></appmsg></msg>'
        )

        self.assertEqual(parsed.parse_status, 'parsed')
        self.assertEqual(parsed.normalized_type, 'link')
        self.assertEqual(parsed.fields['link_identity']['scheme'], 'https')
        self.assertEqual(parsed.fields['link_identity']['host'], 'example.com')
        self.assertEqual(len(parsed.fields['link_identity']['path_hash']), 64)
        serialized = json.dumps(parsed.to_dict(), ensure_ascii=False)
        self.assertNotIn('secret-value', serialized)
        self.assertNotIn('private/path', serialized)
        self.assertNotIn('https://', serialized)

    def test_supported_families_have_bounded_allowed_fields(self):
        fixtures = {
            'image_card': '<msg><appmsg><type>8</type><appattach><totallen>1681660</totallen><fileext>pic</fileext><cdnthumburl>signed-secret</cdnthumburl></appattach></appmsg></msg>',
            'mini_program': '<msg><appmsg><type>33</type><title>小程序</title><weappinfo><appid>wx123</appid><username>gh_abc</username><pagepath>pages/a?token=nope</pagepath></weappinfo></appmsg></msg>',
            'note': '<msg><appmsg><type>24</type><title>收藏笔记</title><des>安全摘要</des><recorditem>不保存笔记正文</recorditem></appmsg></msg>',
            'location': '<msg><appmsg><type>48</type><title>位置</title><location x="31.2304" y="121.4737" label="会面地点" /></appmsg></msg>',
            'quote': '<msg><appmsg><type>57</type><title>回复</title><refermsg><type>1</type><displayname>客户</displayname><title>原消息标题</title><content>不保存原文</content></refermsg></appmsg></msg>',
            'merged_chat': '<msg><appmsg><type>19</type><title>聊天记录</title><des>3条消息</des><recorditem>不保存合并正文</recorditem></appmsg></msg>',
            'file': '<msg><appmsg><type>6</type><title>方案.pdf</title><appattach><totallen>12345</totallen><fileext>pdf</fileext><attachid>signed-secret</attachid></appattach></appmsg></msg>',
        }
        for expected_type, raw in fixtures.items():
            with self.subTest(expected_type=expected_type):
                parsed = parse_appmsg(raw)
                self.assertEqual(parsed.normalized_type, expected_type)
                self.assertEqual(parsed.parse_status, 'parsed')
                if expected_type == 'location':
                    self.assertEqual(parsed.fields['location_label'], '会面地点')
                    self.assertEqual(parsed.fields['latitude'], 31.2304)
                if expected_type == 'file':
                    self.assertEqual(parsed.fields['file_extension'], 'pdf')
                    self.assertEqual(parsed.fields['file_size'], 12345)
                if expected_type == 'image_card':
                    self.assertEqual(parsed.fields['file_extension'], 'pic')
                    self.assertEqual(parsed.fields['file_size'], 1681660)
                    self.assertEqual(parsed.display_text, '[appmsg/image_card] pic 1681660B')
                serialized = json.dumps(parsed.to_dict(), ensure_ascii=False)
                self.assertNotIn('不保存原文', serialized)
                self.assertNotIn('不保存合并正文', serialized)
                self.assertNotIn('不保存笔记正文', serialized)
                self.assertNotIn('signed-secret', serialized)

    def test_malformed_unknown_and_unsafe_xml_are_typed(self):
        malformed = parse_appmsg('<msg><appmsg>')
        self.assertEqual((malformed.parse_status, malformed.unsupported_reason), ('malformed', 'malformed_xml'))
        unknown = parse_appmsg('<msg><appmsg><type>99999</type><title>保留标题</title></appmsg></msg>')
        self.assertEqual((unknown.parse_status, unknown.unsupported_reason), ('unsupported', 'unsupported_appmsg_type'))
        unsafe = parse_appmsg('<!DOCTYPE x [<!ENTITY x "boom">]><msg><appmsg><title>&x;</title></appmsg></msg>')
        self.assertEqual((unsafe.parse_status, unsafe.unsupported_reason), ('rejected', 'unsafe_xml_construct'))
        too_large = parse_appmsg(b'x' * (MAX_SOURCE_BYTES + 1))
        self.assertEqual(too_large.unsupported_reason, 'payload_too_large')


if __name__ == '__main__':
    unittest.main()
