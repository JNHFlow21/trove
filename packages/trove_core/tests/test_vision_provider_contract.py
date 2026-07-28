from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trove_core.vision.base import VisionRequest
from trove_core.vision.volcengine_ark import VolcengineArkVisionProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.payload).encode('utf-8')


class VisionProviderContractTests(unittest.TestCase):
    def test_ark_responses_request_uses_input_image_and_usage_cost(self):
        seen = {}
        output = {'caption': 'fixture product image', 'visible_text': '报价', 'objects': ['document'], 'business_signals': ['pricing'], 'entity_mentions': ['示例教育'], 'confidence': 0.82}
        def fake_urlopen(req, timeout=0):
            seen['url'] = req.full_url
            seen['headers'] = dict(req.header_items())
            seen['body'] = json.loads(req.data.decode('utf-8'))
            return FakeResponse({'output_text': json.dumps(output), 'usage': {'input_tokens': 1000, 'output_tokens': 500, 'total_tokens': 1500, 'input_tokens_details': {'cached_tokens': 100}}})
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / 'img.jpg'; img.write_bytes(b'\xff\xd8\xfffixture')
            provider = VolcengineArkVisionProvider(api_key='fake-key', urlopen=fake_urlopen)
            result = provider.observe(VisionRequest(asset_id='asset-i', image_path=img, citation='trove://citation'))
        self.assertTrue(seen['url'].endswith('/responses'))
        self.assertEqual(seen['body']['model'], 'doubao-seed-2-0-lite-260215')
        content_types = [c['type'] for c in seen['body']['input'][0]['content']]
        self.assertIn('input_image', content_types)
        self.assertIn('input_text', content_types)
        self.assertEqual(result.caption, 'fixture product image')
        self.assertEqual(result.usage.input_tokens, 1000)
        self.assertEqual(result.usage.estimated_cost_rmb, 0.002412)
        self.assertFalse(result.raw_provider_payload_stored)
        self.assertNotIn('fake-key', str(result.to_dict()))


if __name__ == '__main__':
    unittest.main()
