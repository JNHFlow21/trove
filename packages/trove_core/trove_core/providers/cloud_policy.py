from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from trove_core.wechat.process_config import process_config_from_payload, read_latest_process_config


def cloud_retrieval_policy(vault_root: str | Path) -> dict[str, Any]:
    latest = read_latest_process_config(vault_root)
    config = process_config_from_payload(dict(latest.get('config') or {}))
    enabled = config.cloud_retrieval == 'enabled'
    return {
        'state': 'enabled' if enabled else 'disabled',
        'enabled': enabled,
        'consent_scope': 'vault-continuous-retrieval-v1' if enabled else None,
        'embedding': 'aliyun:text-embedding-v4:dense+sparse' if enabled else 'local',
        'rerank': 'aliyun:qwen3-rerank:top20' if enabled else 'off',
        # Multi-hop ranks complete precomputed episode bundles with the same
        # default reranker. Keep the legacy selector field explicit so status
        # clients do not mistake the removed Flash call for live spend.
        'multi_hop_selector': 'off',
        'multi_hop_episode': (
            'zvec:text-embedding-v4:top10+qwen3-rerank'
            if enabled else 'off'
        ),
        'secret_name': 'DASHSCOPE_API_KEY' if enabled else None,
        'secret_value_included': False,
        'raw_content_included': False,
    }


def cloud_retrieval_environment(
    vault_root: str | Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    snapshot = dict(os.environ if env is None else env)
    if cloud_retrieval_policy(vault_root)['enabled']:
        snapshot['TROVE_ENABLE_CLOUD_EMBEDDING'] = '1'
        snapshot['TROVE_ENABLE_CLOUD_RERANK'] = '1'
    return snapshot
