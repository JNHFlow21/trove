from __future__ import annotations

from typing import Any

ENCRYPTED_MAPPING_CANDIDATES_NOT_INSPECTED = (
    'media_0.db',
    'message_resource.db',
    'hardlink.db',
    'head_image.db',
    'media_0.kvdb',
)

# D0 is deliberately layered: readable stores and cache files only bound what was
# inspected; encrypted DB/KV candidates remain outside TROVE's no-decrypt product
# boundary and are listed as the next external decryption targets.
D0_MAPPING_CONCLUSION = {
    'readable_sns_db': 'no_mapping',
    'cache_files': 'no_embedded_mapping',
    'encrypted_candidates_not_inspected': list(ENCRYPTED_MAPPING_CANDIDATES_NOT_INSPECTED),
}


def d0_mapping_conclusion() -> dict[str, Any]:
    return {
        'readable_sns_db': D0_MAPPING_CONCLUSION['readable_sns_db'],
        'cache_files': D0_MAPPING_CONCLUSION['cache_files'],
        'encrypted_candidates_not_inspected': list(ENCRYPTED_MAPPING_CANDIDATES_NOT_INSPECTED),
    }
