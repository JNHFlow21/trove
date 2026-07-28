from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from trove_core.wechat.media.dat_decoder import decode_wechat_dat_bytes, decode_wechat_dat_file


class DatDecoderTests(unittest.TestCase):
    def test_mac_wechat_v2_derives_account_key_and_decodes(self):
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / 'Documents'
            account_name = 'wxid_ownerfixture_c198'
            image = b'\xff\xd8\xff\xe0' + b'image-payload' * 3
            uin = 1509864304
            key = hashlib.md5(f'{uin}wxid_ownerfixture'.encode()).hexdigest()[:16].encode()
            padding = 16 - (len(image) % 16)
            padded = image + bytes([padding]) * padding
            encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
            ciphertext = encryptor.update(padded) + encryptor.finalize()
            payload = b'\x07\x08V2\x08\x07' + len(image).to_bytes(4, 'little') + (0).to_bytes(4, 'little') + b'\x01' + ciphertext
            path = documents / 'xwechat_files' / account_name / 'msg' / 'attach' / 'a' / '2026-01' / 'Img' / 'fixture.dat'
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            kvcomm = documents / 'app_data' / 'net' / 'kvcomm'
            kvcomm.mkdir(parents=True)
            (kvcomm / f'key_{uin}_fixture.statistic').touch()

            result = decode_wechat_dat_file(path)

            self.assertEqual(result.status, 'decoded')
            self.assertEqual(result.image_type, 'jpg')
            self.assertEqual(result.output_bytes, image)

    def test_xor_encoded_jpeg_decodes(self):
        jpeg = b'\xff\xd8\xff\xe0fixture'
        key = 0x37
        encoded = bytes(b ^ key for b in jpeg)
        result = decode_wechat_dat_bytes(encoded)
        self.assertEqual(result.status, 'decoded')
        self.assertEqual(result.image_type, 'jpg')
        self.assertEqual(result.xor_key, key)
        self.assertEqual(result.output_bytes, jpeg)

    def test_unknown_wrapper_is_precise_failure(self):
        result = decode_wechat_dat_bytes(b'not-an-image')
        self.assertEqual(result.status, 'decode_failed')
        self.assertEqual(result.error_code, 'unknown_dat_wrapper')


if __name__ == '__main__':
    unittest.main()
