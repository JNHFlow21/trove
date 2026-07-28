from __future__ import annotations

import base64
import os
import tempfile
import time
import unittest
from pathlib import Path

from trove_core.local_vlm.base import ImageCaptionRequest
from trove_core.local_vlm.fake import FakeLocalVLMCaptionProvider
from trove_core.local_vlm.mlx_vlm_provider import MlxVLMCaptionProvider, parse_caption_text


class LocalVLMProviderTests(unittest.TestCase):
    def test_parse_caption_text_limits_caption_and_extracts_labels(self):
        caption, labels = parse_caption_text('这是一张展示课程优惠和预算信息的中文海报截图，画面中有醒目的标题和表格。额外长句应被截断以保证紧凑。\n标签：海报，预算，课程，优惠，截图，超额')
        self.assertLessEqual(len(caption), 60)
        self.assertEqual(labels, ['海报', '预算', '课程', '优惠', '截图'])
        with self.assertRaises(ValueError):
            parse_caption_text('')

    def test_fake_provider_contract_is_local_and_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / 'fixture.jpg'
            img.write_bytes(b'\xff\xd8\xfffixture')
            provider = FakeLocalVLMCaptionProvider(caption='本地中文描述', labels=['本地'])
            result = provider.caption(ImageCaptionRequest(asset_id='asset-i', image_path=img, citation='trove://image'))
        self.assertEqual(result.caption, '本地中文描述')
        self.assertEqual(result.labels, ['本地'])
        self.assertEqual(result.usage.estimated_cost_rmb, 0.0)
        self.assertNotIn('Bearer ', str(result.to_dict()))

    def test_mlx_vlm_real_smoke_optional(self):
        if os.environ.get('TROVE_RUN_LOCAL_VLM_SMOKE') != '1':
            self.skipTest('set TROVE_RUN_LOCAL_VLM_SMOKE=1 to run local mlx-vlm smoke')
        # 1x1 PNG, generated into a temp directory so raw media never enters Git.
        png = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/l2cZ2QAAAABJRU5ErkJggg=='
        )
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / 'fixture.png'
            img.write_bytes(png)
            provider = MlxVLMCaptionProvider(cache_dir=Path(d) / 'models')
            started = time.perf_counter()
            try:
                result = provider.caption(ImageCaptionRequest(asset_id='smoke-image', image_path=img))
            except RuntimeError as exc:
                self.skipTest(f'local mlx-vlm unavailable: {exc}')
            elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertTrue(result.caption)
        self.assertLessEqual(len(result.caption), 60)
        print(f'local_vlm_smoke_ms={elapsed_ms:.1f}')


if __name__ == '__main__':
    unittest.main()
