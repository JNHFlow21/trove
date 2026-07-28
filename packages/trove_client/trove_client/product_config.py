from __future__ import annotations

import os
from pathlib import Path

from trove_core.product_config import ProductConfigError, load_product_config


def resolve_vault_root(value: str | None, *, create: bool = False) -> Path:
    if value is not None:
        selected = Path(value).expanduser()
    elif create:
        config = load_product_config(for_write=True)
        if config.vault_root is None:
            raise ProductConfigError('vault_unconfigured', 'product configuration has no Vault')
        selected = config.vault_root
    else:
        config = load_product_config()
        if config.vault_root is None:
            raise FileNotFoundError('no readable Vault is configured')
        selected = config.vault_root
    if create:
        selected.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(selected, 0o700)
    return selected.resolve(strict=True)


__all__ = ['resolve_vault_root']
