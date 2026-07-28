from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trove_core.asr.base import ASRRequest
from trove_core.asr.volcengine_flash import VolcengineASRFlashProvider


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.payload).encode('utf-8')


class ASRProviderContractTests(unittest.TestCase):
    def test_volcengine_flash_request_is_pinned_and_costed(self):
        seen = {}
        def fake_urlopen(req, timeout=0):
            seen['timeout'] = timeout
            seen['headers'] = dict(req.header_items())
            seen['body'] = json.loads(req.data.decode('utf-8'))
            return FakeResponse({'result': {'text': 'fixture transcript', 'additions': {'duration': 3000}}, 'audio_info': {'duration': 3000}})
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / 'voice.wav'
            audio.write_bytes(b'RIFFfixture')
            provider = VolcengineASRFlashProvider(api_key='fake-key', urlopen=fake_urlopen, timeout=7)
            result = provider.transcribe(ASRRequest(asset_id='asset-1', audio_path=audio, citation='trove://citation'))
        self.assertEqual(seen['body']['request']['model_name'], 'bigmodel')
        header_keys = {k.lower(): v for k, v in seen['headers'].items()}
        self.assertEqual(header_keys['x-api-resource-id'], 'volc.bigasr.auc_turbo')
        self.assertEqual(header_keys['x-api-sequence'], '-1')
        self.assertIn('data', seen['body']['audio'])
        self.assertEqual(result.text, 'fixture transcript')
        self.assertEqual(result.usage.duration_seconds, 3.0)
        self.assertEqual(result.usage.estimated_cost_rmb, 0.00375)
        self.assertNotIn('fake-key', str(result.to_dict()))


if __name__ == '__main__':
    unittest.main()
