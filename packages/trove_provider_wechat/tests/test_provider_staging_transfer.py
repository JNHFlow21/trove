from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from trove_daemon.provider_loader import ProviderLoadError, StagingTransfer
from trove_protocol.provider import ProviderManifest
from trove_provider_wechat import create_provider


PACKAGE = Path(__file__).resolve().parents[1] / 'trove_provider_wechat'


class ProviderStagingTransferTests(unittest.TestCase):
    def test_64_mib_media_uses_owner_only_staging_not_json_or_base64(self):
        manifest = ProviderManifest.from_dict(json.loads((PACKAGE / 'manifest.json').read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'media.bin'
            source.write_bytes(b'fixture-block' * (64 * 1024 * 1024 // len(b'fixture-block')) + b'x')
            self.assertLessEqual(source.stat().st_size, 64 * 1024 * 1024)
            transfer = StagingTransfer(root / 'staging')
            grant = transfer.allocate()
            provider = create_provider(manifest, media_assets={'asset-a': source})
            metadata = provider.invoke('media', {
                'asset_id': 'asset-a', 'staging_path': str(grant.path),
            })
            self.assertFalse(metadata['blob_in_json'])
            self.assertNotIn('data', metadata)
            accepted = transfer.accept(
                grant.handle, size=metadata['size'], sha256=metadata['sha256'],
                cas_dir=root / 'cas',
            )
            self.assertEqual(accepted.stat().st_size, source.stat().st_size)
            self.assertEqual(os.stat(accepted).st_mode & 0o777, 0o600)

    def test_unsafe_short_oversize_and_hash_mismatch_are_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = StagingTransfer(root / 'staging', max_bytes=8)
            for mode in ('short', 'oversize', 'hash'):
                grant = transfer.allocate()
                grant.path.write_bytes(b'fixture')
                size = 8 if mode == 'short' else (9 if mode == 'oversize' else 7)
                digest = hashlib.sha256(b'wrong' if mode == 'hash' else b'fixture').hexdigest()
                with self.subTest(mode=mode), self.assertRaises(ProviderLoadError):
                    transfer.accept(grant.handle, size=size, sha256=digest, cas_dir=root / 'cas')
                self.assertFalse(grant.path.exists())

    def test_nonregular_symlink_and_non_owner_only_mode_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = StagingTransfer(root / 'staging')

            grant = transfer.allocate()
            grant.path.unlink()
            grant.path.mkdir()
            with self.assertRaises(ProviderLoadError):
                transfer.accept(grant.handle, size=0, sha256=hashlib.sha256(b'').hexdigest(), cas_dir=root / 'cas')
            self.assertFalse(grant.path.exists())

            outside = root / 'outside'
            outside.write_bytes(b'fixture')
            grant = transfer.allocate()
            grant.path.unlink()
            grant.path.symlink_to(outside)
            with self.assertRaises(ProviderLoadError):
                transfer.accept(grant.handle, size=7, sha256=hashlib.sha256(b'fixture').hexdigest(), cas_dir=root / 'cas')
            self.assertTrue(outside.exists())
            self.assertFalse(grant.path.exists())

            grant = transfer.allocate()
            grant.path.write_bytes(b'fixture')
            os.chmod(grant.path, 0o644)
            with self.assertRaises(ProviderLoadError):
                transfer.accept(grant.handle, size=7, sha256=hashlib.sha256(b'fixture').hexdigest(), cas_dir=root / 'cas')
            self.assertFalse(grant.path.exists())


if __name__ == '__main__':
    unittest.main()
