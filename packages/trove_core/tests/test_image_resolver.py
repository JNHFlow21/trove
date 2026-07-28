from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trove_core.wechat.media.image_resolver import resolve_image_file


class ImageResolverTests(unittest.TestCase):
    def test_decodes_dat_to_vault_derivative(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'sample.dat'
            jpeg = b'\xff\xd8\xff\xe0fixture'
            source.write_bytes(bytes(b ^ 0x21 for b in jpeg))
            vault = root / 'vault'
            result = resolve_image_file(source, vault, asset_id='asset-1')
            self.assertEqual(result.status, 'decoded')
            self.assertEqual(result.image_type, 'jpg')
            self.assertTrue((vault / result.derivative_ref).exists())
            self.assertNotIn('..', result.derivative_ref)

    def test_missing_image_is_resumable_state(self):
        with tempfile.TemporaryDirectory() as d:
            result = resolve_image_file(Path(d) / 'missing.dat', Path(d) / 'vault', asset_id='asset-1')
            self.assertEqual(result.status, 'missing_local_cache')
            self.assertEqual(result.error_code, 'missing_file')


if __name__ == '__main__':
    unittest.main()
