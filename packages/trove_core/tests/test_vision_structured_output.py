from __future__ import annotations

import json
import unittest

from trove_core.vision.structured_output import parse_structured_vision_text


class VisionStructuredOutputTests(unittest.TestCase):
    def test_validates_required_caption_and_confidence(self):
        data = parse_structured_vision_text(json.dumps({'caption': '报价单截图', 'visible_text': '价格', 'objects': ['document'], 'business_signals': ['pricing'], 'entity_mentions': ['示例'], 'confidence': 0.8}, ensure_ascii=False))
        self.assertEqual(data['caption'], '报价单截图')
        self.assertEqual(data['confidence'], 0.8)
        with self.assertRaises(ValueError):
            parse_structured_vision_text('{bad json')
        with self.assertRaises(ValueError):
            parse_structured_vision_text(json.dumps({'caption': '', 'confidence': 0.2}))


if __name__ == '__main__':
    unittest.main()
