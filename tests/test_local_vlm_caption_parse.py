from __future__ import annotations

import unittest

from trove_core.local_vlm.mlx_vlm_provider import parse_caption_text


class LocalVLMCaptionParseP2Tests(unittest.TestCase):
    def test_digit_prefixed_captions_are_not_stripped_as_list_numbers(self):
        self.assertEqual(parse_caption_text('2026预算海报')[0], '2026预算海报')
        self.assertEqual(parse_caption_text('3D模型截图')[0], '3D模型截图')
        self.assertEqual(parse_caption_text('1. 描述')[0], '描述')

    def test_json_only_response_is_rejected(self):
        for raw in (
            '{"caption": "2026预算海报", "labels": ["预算"]}',
            '{\n  "caption": "2026预算海报",\n  "labels": ["预算"]\n}',
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'empty_caption_output'):
                parse_caption_text(raw)


if __name__ == '__main__':
    unittest.main()
