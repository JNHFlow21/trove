from __future__ import annotations

import hashlib
import os
import tempfile
import time
import unittest
from pathlib import Path

from trove_protocol.blobs import BlobContractError, BlobTransfer, validate_staged_blob


class BlobContractTests(unittest.TestCase):
    def test_owner_only_regular_file_with_matching_size_and_hash_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'blob'
            path.write_bytes(b'fixture-bytes')
            os.chmod(path, 0o600)
            transfer = BlobTransfer(
                path=str(path), size=13,
                sha256=hashlib.sha256(b'fixture-bytes').hexdigest(),
                expires_at=time.time() + 30,
            )
            self.assertEqual(validate_staged_blob(transfer, staging_root=root), path.resolve())

    def test_missing_hash_mismatch_expired_and_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'blob'
            path.write_bytes(b'x')
            os.chmod(path, 0o600)
            valid = BlobTransfer(
                path=str(path), size=1, sha256=hashlib.sha256(b'x').hexdigest(),
                expires_at=time.time() + 30,
            )
            cases = (
                replace_transfer(valid, path=str(root / 'missing')),
                replace_transfer(valid, size=2),
                replace_transfer(valid, sha256='0' * 64),
                replace_transfer(valid, expires_at=1),
                replace_transfer(valid, path=str(root.parent / 'escape')),
            )
            for transfer in cases:
                with self.subTest(transfer=transfer):
                    with self.assertRaises(BlobContractError):
                        validate_staged_blob(transfer, staging_root=root)

    def test_symlink_and_non_owner_only_permissions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            target.write_bytes(b'x')
            os.chmod(target, 0o644)
            link = root / 'link'
            link.symlink_to(target)
            for path in (target, link):
                transfer = BlobTransfer(
                    path=str(path), size=1, sha256=hashlib.sha256(b'x').hexdigest(),
                    expires_at=time.time() + 30,
                )
                with self.assertRaises(BlobContractError):
                    validate_staged_blob(transfer, staging_root=root)


def replace_transfer(transfer: BlobTransfer, **changes) -> BlobTransfer:
    payload = transfer.to_dict() | changes
    return BlobTransfer.from_dict(payload)


if __name__ == '__main__':
    unittest.main()
