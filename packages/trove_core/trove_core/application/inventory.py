from __future__ import annotations

from trove_core.approvals import SENSITIVE_CAPABILITY_INVENTORY


# Auditable source of truth.  Values are code locations, not user-facing
# capability claims; CI verifies that adapters do not bypass these boundaries.
APPLICATION_COMMAND_INVENTORY = {
    'full_import': 'trove_core.application.sensitive_commands.execute_full_import',
    'reset_index_cache': 'trove_core.application.sensitive_commands.execute_reset_index_cache',
    'scope_rebuild': 'trove_core.application.sensitive_commands.execute_scope_rebuild',
    'vector_purge_rebuild': 'trove_core.application.sensitive_commands.execute_vector_mutation',
    'vector_rebuild': 'trove_core.application.sensitive_commands.execute_vector_mutation',
    'content_kind_backfill': 'trove_core.application.sensitive_commands.execute_content_kind_backfill',
    'appmsg_backfill': 'trove_core.application.sensitive_commands.execute_appmsg_backfill',
    'derived_data_purge': 'trove_core.application.sensitive_commands.execute_derived_data_purge',
    'message_media_backfill': 'trove_core.application.sensitive_commands.execute_message_media_backfill',
    'wechat_cdn_fetch': 'trove_core.application.sensitive_commands.execute_wechat_cdn_fetch',
    'entity_reconcile': 'trove_core.agent_tools.tools.identity_reconcile',
    'media_understanding_invalidate': 'trove_core.application.sensitive_commands.execute_media_understanding_invalidate',
    'recover_writer_marker': 'trove_core.application.writer_recovery.execute_writer_marker_recovery',
    'voice_cloud_asr': 'trove_core.application.cloud_commands.execute_cloud_voice_transcript',
    'image_cloud_vision': 'trove_core.application.cloud_commands.execute_cloud_image_observation',
    'cloud_embedding_probe': 'scripts.probe_cloud_embedding_text._execute_probe',
    'cloud_vector_index': 'trove_core.application.cloud_commands.execute_cloud_vector_index',
    'cloud_rerank': 'trove_core.application.cloud_commands.execute_cloud_rerank',
    'real_media_processing': 'trove_core.application.sensitive_commands.execute_real_voice_transcription',
    'files_archive': 'trove_core.application.sensitive_commands.execute_files_archive',
    'observe_approve': 'trove_core.application.sensitive_commands.execute_observation_status',
    'observe_retire': 'trove_core.application.sensitive_commands.execute_observation_status',
}


def validate_application_command_inventory() -> None:
    missing = set(SENSITIVE_CAPABILITY_INVENTORY) - set(APPLICATION_COMMAND_INVENTORY)
    stale = set(APPLICATION_COMMAND_INVENTORY) - set(SENSITIVE_CAPABILITY_INVENTORY)
    if missing or stale:
        raise RuntimeError(
            f'application command inventory mismatch: missing={sorted(missing)} stale={sorted(stale)}'
        )
