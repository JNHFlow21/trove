from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from trove_core.wechat.media.audio_resolver import normalize_audio_file


class AudioResolverTests(unittest.TestCase):
    def test_wechat_silk_is_decoded_to_pcm_before_ffmpeg_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'voice.silk'
            source.write_bytes(b'\x02#!SILK_V3fixture')
            vault = root / 'vault'
            calls: list[tuple[bytes, int]] = []

            def decode(input_file, output_file, sample_rate):
                calls.append((input_file.read(), sample_rate))
                output_file.write(b'\x00\x00' * 32)

            def run_ffmpeg(args, **_kwargs):
                Path(args[-1]).write_bytes(b'RIFF\x00\x00\x00\x00WAVEfmt ')
                return SimpleNamespace(returncode=0)

            with mock.patch.dict(sys.modules, {'pysilk': SimpleNamespace(decode=decode)}), mock.patch(
                'trove_core.wechat.media.audio_resolver.subprocess.run', side_effect=run_ffmpeg,
            ):
                result = normalize_audio_file(source, vault, asset_id='asset-silk')

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 24000)
            self.assertEqual(result.status, 'normalized')
            self.assertEqual(result.codec, 'wav')
            self.assertTrue((vault / result.derivative_ref).exists())
            self.assertFalse(list((vault / 'media' / 'audio').glob('*.pcm')))

    def test_wav_is_copied_to_vault_audio_derivative(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'voice.wav'
            source.write_bytes(b'RIFF\x00\x00\x00\x00WAVEfmt ')
            vault = root / 'vault'
            result = normalize_audio_file(source, vault, asset_id='asset-voice')
            self.assertEqual(result.status, 'copied')
            self.assertEqual(result.codec, 'wav')
            self.assertTrue((vault / result.derivative_ref).exists())

    def test_missing_audio_is_resumable_state(self):
        with tempfile.TemporaryDirectory() as d:
            result = normalize_audio_file(Path(d) / 'missing.amr', Path(d) / 'vault', asset_id='asset-voice')
            self.assertEqual(result.status, 'missing_local_cache')


if __name__ == '__main__':
    unittest.main()
