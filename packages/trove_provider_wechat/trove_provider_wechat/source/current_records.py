from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .current_importer import WeChatDecryptedAccountImporter


def _source_watermark(directory: Path) -> str:
    """Fingerprint source state without parsing or retaining message rows."""
    digest = hashlib.sha256()
    candidates = [directory / 'contact.db', *sorted(directory.glob('message_*.db'))]
    for path in candidates:
        try:
            info = path.stat()
        except OSError:
            continue
        digest.update(path.name.encode('utf-8'))
        digest.update(str(info.st_size).encode('ascii'))
        digest.update(str(info.st_mtime_ns).encode('ascii'))
    return digest.hexdigest()


def is_current_account(directory: Path) -> bool:
    return (
        directory.is_dir()
        and not directory.is_symlink()
        and any(directory.glob('message_*.db'))
        and (directory / 'contact.db').is_file()
    )


def current_accounts(root: Path) -> list[dict[str, Any]]:
    accounts: list[dict[str, Any]] = []
    if not root.is_dir():
        return accounts
    for directory in sorted(path for path in root.iterdir() if is_current_account(path)):
        importer = WeChatDecryptedAccountImporter(directory)
        accounts.append({
            'account_id': importer.account_id,
            'label': importer.account_label,
            # Enumeration is a startup/health boundary. Exact counts belong to
            # Vault queries and must not force a full source decode here.
            'message_count': 0,
            'watermark': _source_watermark(directory),
        })
    return accounts


def current_account_records(directory: Path) -> tuple[list[dict[str, Any]], str]:
    importer = WeChatDecryptedAccountImporter(directory)
    accounts, _, messages = importer.load()
    account_label = accounts[0].label if accounts else importer.account_label
    rows: list[dict[str, Any]] = []
    for message in sorted(messages, key=lambda item: (item.timestamp, item.shard_id, item.local_id)):
        record = message.safe_dict()
        record['account_label'] = account_label
        record['trust'] = 'untrusted_evidence'
        rows.append(record)
    # A cursor represents source state, not the cost of reserializing every
    # message. This keeps account enumeration O(file-count) and bounded.
    return rows, _source_watermark(directory)


__all__ = ['current_account_records', 'current_accounts', 'is_current_account']
