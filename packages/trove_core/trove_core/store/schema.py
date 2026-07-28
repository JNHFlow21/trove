from __future__ import annotations

SCHEMA_VERSION = 28
FTS_TOKENIZER_VERSION = 'trigram/v1'
VECTOR_SOURCE_REVISION_KEY = 'vector_source_revision'

TABLES = {
    'accounts': 'accounts',
    'conversations': 'conversations',
    'messages': 'messages',
    'message_payloads': 'message_payloads',
    'source_snapshots': 'source_snapshots',
    'media_source_bindings': 'media_source_bindings',
    'message_fts': 'message_fts',
    'chunk_fts': 'chunk_fts',
    'media_assets': 'media_assets',
    'media_asset_links': 'media_asset_links',
    'media_decode_results': 'media_decode_results',
    'media_jobs': 'media_jobs',
    'provider_jobs': 'provider_jobs',
    'transcripts': 'transcripts',
    'image_observations': 'image_observations',
    'sns_cache_mappings': 'sns_cache_mappings',
    'media_understanding': 'media_understanding',
    'moment_items': 'moment_items',
    'moment_interactions': 'moment_interactions',
    'favorites': 'favorites',
    'evidence_items': 'evidence_items',
    'entities': 'entities',
    'entity_identifiers': 'entity_identifiers',
    'observations': 'observations',
    'relationships': 'relationships',
    'profile_snapshots': 'profile_snapshots',
    'profile_enrichment_runs': 'profile_enrichment_runs',
    'profile_enrichment_tasks': 'profile_enrichment_tasks',
    'profile_automation_subscriptions': 'profile_automation_subscriptions',
    'profile_refresh_queue': 'profile_refresh_queue',
    'evidence_chunks': 'evidence_chunks',
    'local_trace_events': 'local_trace_events',
    'approval_records': 'approval_records',
    'operation_journal': 'operation_journal',
    'derived_data_purge_audit': 'derived_data_purge_audit',
    'sync_state': 'sync_state',
    'sync_dirty_citations': 'sync_dirty_citations',
    'sync_aux_state': 'sync_aux_state',
    'sync_citation_tombstones': 'sync_citation_tombstones',
    'sync_message_source_rows': 'sync_message_source_rows',
    'media_source_state': 'media_source_state',
    'media_source_rows': 'media_source_rows',
    'vector_entries': 'vector_entries',
    'vector_index_generations': 'vector_index_generations',
    'vector_index_ledger': 'vector_index_ledger',
}

BASE_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS accounts (
        account_id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        display_name TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT NOT NULL,
        account_id TEXT NOT NULL,
        title TEXT NOT NULL,
        type TEXT NOT NULL,
        member_count INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (account_id, conversation_id)
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        citation TEXT NOT NULL UNIQUE,
        account_id TEXT NOT NULL,
        account_label TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        conversation_title TEXT NOT NULL,
        conversation_type TEXT NOT NULL,
        sender_id TEXT NOT NULL,
        sender_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        content TEXT NOT NULL,
        content_kind TEXT NOT NULL DEFAULT 'text',
        shard_id TEXT NOT NULL,
        local_id INTEGER NOT NULL,
        sent_by_me INTEGER NOT NULL,
        source_type TEXT NOT NULL,
        direction TEXT NOT NULL,
        UNIQUE (account_id, conversation_id, shard_id, local_id)
    )""",
]

MULTIMODAL_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS message_payloads (
        citation TEXT PRIMARY KEY,
        appmsg_type INTEGER,
        normalized_type TEXT NOT NULL,
        parse_status TEXT NOT NULL,
        normalized_json TEXT NOT NULL DEFAULT '{}',
        display_text TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        unsupported_reason TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(citation) REFERENCES messages(citation) ON UPDATE CASCADE ON DELETE CASCADE,
        CHECK(parse_status IN ('parsed', 'unsupported', 'malformed', 'rejected'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_message_payloads_status_type ON message_payloads(parse_status, normalized_type)""",
    """CREATE TABLE IF NOT EXISTS source_snapshots (
        snapshot_revision TEXT PRIMARY KEY,
        root_ref TEXT,
        manifest_hash TEXT NOT NULL,
        guard_run_id_hash TEXT,
        state TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK(state IN ('available', 'external_unbound', 'unavailable', 'changed'))
    )""",
    """CREATE TABLE IF NOT EXISTS media_source_bindings (
        asset_id TEXT PRIMARY KEY,
        snapshot_revision TEXT NOT NULL,
        account_dir_hash TEXT NOT NULL,
        source_coordinates_json TEXT NOT NULL DEFAULT '{}',
        locator_state TEXT NOT NULL DEFAULT 'bound',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id) ON UPDATE CASCADE ON DELETE CASCADE,
        FOREIGN KEY(snapshot_revision) REFERENCES source_snapshots(snapshot_revision),
        CHECK(locator_state IN ('bound', 'snapshot_unavailable', 'routes_exhausted', 'materialized'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_media_source_bindings_snapshot ON media_source_bindings(snapshot_revision, account_dir_hash)""",
    """CREATE TABLE IF NOT EXISTS media_assets (
        asset_id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        modality TEXT NOT NULL,
        media_type TEXT NOT NULL,
        local_type TEXT,
        citation TEXT NOT NULL,
        content_hash TEXT,
        path_ref TEXT,
        cache_state TEXT NOT NULL DEFAULT 'unknown',
        processing_state TEXT NOT NULL DEFAULT 'pending',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_media_assets_unique_ref_hash ON media_assets(account_id, source_type, source_id, modality, media_type, content_hash) WHERE content_hash IS NOT NULL AND content_hash != ''""",
    """CREATE INDEX IF NOT EXISTS idx_media_assets_ref_lookup ON media_assets(account_id, source_type, source_id, modality, media_type)""",
    """CREATE INDEX IF NOT EXISTS idx_media_assets_account_modality ON media_assets(account_id, modality, cache_state)""",
    """CREATE INDEX IF NOT EXISTS idx_media_assets_citation ON media_assets(citation)""",
    """CREATE TABLE IF NOT EXISTS media_asset_links (
        link_id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        account_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_citation TEXT NOT NULL,
        scope_type TEXT NOT NULL,
        accepted INTEGER NOT NULL,
        reason TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_media_asset_links_asset ON media_asset_links(asset_id, accepted)""",
    """CREATE INDEX IF NOT EXISTS idx_media_asset_links_citation ON media_asset_links(source_citation)""",
    """CREATE TABLE IF NOT EXISTS media_decode_results (
        decode_id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        status TEXT NOT NULL,
        wrapper_type TEXT,
        input_hash TEXT,
        output_hash TEXT,
        derivative_ref TEXT,
        error_code TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id)
    )""",
    """CREATE TABLE IF NOT EXISTS media_jobs (
        job_id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        retry_count INTEGER NOT NULL DEFAULT 0,
        error_code TEXT,
        last_duration_ms REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(asset_id, job_type),
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_media_jobs_status ON media_jobs(job_type, status, updated_at)""",
    """CREATE TABLE IF NOT EXISTS provider_jobs (
        job_id TEXT PRIMARY KEY,
        asset_id TEXT,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0,
        usage_json TEXT NOT NULL DEFAULT '{}',
        cost_rmb REAL NOT NULL DEFAULT 0,
        request_hash TEXT,
        error_code TEXT,
        citation TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_provider_jobs_status ON provider_jobs(job_type, status, updated_at)""",
    """CREATE TABLE IF NOT EXISTS transcripts (
        transcript_id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        job_id TEXT,
        citation TEXT NOT NULL,
        text TEXT NOT NULL,
        language TEXT,
        confidence REAL NOT NULL DEFAULT 0,
        duration_seconds REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id),
        FOREIGN KEY(job_id) REFERENCES provider_jobs(job_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_transcripts_citation ON transcripts(citation)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_one_active_asset ON transcripts(asset_id) WHERE status='active'""",
    """CREATE TABLE IF NOT EXISTS image_observations (
        observation_id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        job_id TEXT,
        citation TEXT NOT NULL,
        caption TEXT NOT NULL,
        visible_text TEXT NOT NULL DEFAULT '',
        objects_json TEXT NOT NULL DEFAULT '[]',
        business_signals_json TEXT NOT NULL DEFAULT '[]',
        content_sha256 TEXT NOT NULL DEFAULT '',
        model_id TEXT NOT NULL DEFAULT '',
        prompt_version TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'proposed',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT '',
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id),
        FOREIGN KEY(job_id) REFERENCES provider_jobs(job_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_image_observations_citation ON image_observations(citation)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_image_observations_projection_identity ON image_observations(asset_id,citation,content_sha256,model_id,prompt_version)""",
    """CREATE TABLE IF NOT EXISTS sns_cache_mappings (
        mapping_id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        moment_id TEXT NOT NULL,
        source_citation TEXT NOT NULL,
        media_idx INTEGER,
        path_ref TEXT,
        mapping_source TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(account_id, cache_key, source_citation)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_sns_cache_mappings_cache ON sns_cache_mappings(account_id, cache_key)""",
    """CREATE INDEX IF NOT EXISTS idx_sns_cache_mappings_citation ON sns_cache_mappings(source_citation)""",
    """CREATE TABLE IF NOT EXISTS media_understanding (
        content_sha256 TEXT PRIMARY KEY,
        modality TEXT NOT NULL,
        caption TEXT NOT NULL DEFAULT '',
        visible_text TEXT NOT NULL DEFAULT '',
        objects_json TEXT NOT NULL DEFAULT '[]',
        business_signals_json TEXT NOT NULL DEFAULT '[]',
        keyframes_json TEXT NOT NULL DEFAULT '[]',
        audio_transcript TEXT NOT NULL DEFAULT '',
        model_id TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0,
        origin TEXT NOT NULL DEFAULT 'lazy_agent',
        status TEXT NOT NULL DEFAULT 'active',
        source_citations_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        fetch_hit_count INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_media_understanding_modality ON media_understanding(modality, status)""",
    """CREATE INDEX IF NOT EXISTS idx_media_understanding_model ON media_understanding(model_id, prompt_version, status)""",
    """CREATE TABLE IF NOT EXISTS moment_items (
        moment_id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        author_id TEXT,
        citation TEXT NOT NULL UNIQUE,
        timestamp TEXT,
        text TEXT NOT NULL DEFAULT '',
        link_json TEXT NOT NULL DEFAULT '{}',
        media_refs_json TEXT NOT NULL DEFAULT '[]',
        comments_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'active',
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS moment_interactions (
        interaction_id TEXT PRIMARY KEY,
        moment_id TEXT NOT NULL,
        account_id TEXT NOT NULL,
        citation TEXT NOT NULL UNIQUE,
        interaction_type TEXT NOT NULL,
        actor_id TEXT NOT NULL DEFAULT '',
        actor_name TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL DEFAULT '',
        timestamp TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        FOREIGN KEY(moment_id) REFERENCES moment_items(moment_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_moment_interactions_moment ON moment_interactions(moment_id, interaction_type)""",
    """CREATE TABLE IF NOT EXISTS favorites (
        favorite_id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        citation TEXT NOT NULL UNIQUE,
        timestamp TEXT,
        title TEXT NOT NULL DEFAULT '',
        text TEXT NOT NULL DEFAULT '',
        media_refs_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        entity_type TEXT NOT NULL,
        display_name TEXT NOT NULL,
        identifiers_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'active',
        confidence REAL NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(entity_type, display_name)""",
    """CREATE TABLE IF NOT EXISTS entity_identifiers (
        entity_id TEXT NOT NULL,
        identifier_type TEXT NOT NULL,
        normalized_value TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0,
        citation TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(entity_id, identifier_type, normalized_value, source),
        FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_entity_identifiers_lookup ON entity_identifiers(normalized_value, identifier_type, confidence)""",
    """CREATE INDEX IF NOT EXISTS idx_entity_identifiers_entity ON entity_identifiers(entity_id, identifier_type)""",
    """CREATE TABLE IF NOT EXISTS observations (
        observation_id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        observation_type TEXT NOT NULL,
        value_json TEXT NOT NULL,
        status TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0,
        citation TEXT NOT NULL,
        source_type TEXT NOT NULL,
        valid_from TEXT,
        supersedes_observation_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(entity_id) REFERENCES entities(entity_id),
        CHECK(status IN ('proposed', 'active', 'superseded', 'rejected', 'merge_candidate', 'merged', 'needs_review'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_observations_entity_status ON observations(entity_id, status, observation_type)""",
    """CREATE INDEX IF NOT EXISTS idx_observations_citation ON observations(citation)""",
    """CREATE TABLE IF NOT EXISTS relationships (
        relationship_id TEXT PRIMARY KEY,
        subject_entity_id TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object_entity_id TEXT,
        object_ref TEXT,
        citation TEXT,
        confidence REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'active',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY(subject_entity_id) REFERENCES entities(entity_id),
        FOREIGN KEY(object_entity_id) REFERENCES entities(entity_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject_entity_id, predicate)""",
    """CREATE TABLE IF NOT EXISTS profile_snapshots (
        profile_id TEXT PRIMARY KEY,
        entity_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        projection_json TEXT NOT NULL,
        content_hash TEXT NOT NULL DEFAULT '',
        source_revision TEXT NOT NULL DEFAULT 'legacy',
        run_id TEXT,
        schema_version TEXT NOT NULL DEFAULT 'customer-profile/legacy',
        completeness_state TEXT NOT NULL DEFAULT 'stale',
        evidence_citations_json TEXT NOT NULL DEFAULT '[]',
        enrichment_summary_json TEXT NOT NULL DEFAULT '{}',
        gaps_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_profile_snapshots_entity_version ON profile_snapshots(entity_id,version)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_snapshots_entity_version_unique ON profile_snapshots(entity_id,version)""",
    """CREATE INDEX IF NOT EXISTS idx_profile_snapshots_entity_hash ON profile_snapshots(entity_id,content_hash)""",
    """CREATE TABLE IF NOT EXISTS profile_enrichment_runs (
        run_id TEXT PRIMARY KEY,
        plan_key TEXT NOT NULL UNIQUE,
        entity_id TEXT NOT NULL,
        mode TEXT NOT NULL,
        state TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        actor_hash TEXT NOT NULL,
        session_hash TEXT NOT NULL,
        consent_hash TEXT NOT NULL,
        execution_location TEXT NOT NULL,
        processor_identity TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        purpose TEXT NOT NULL,
        item_budget INTEGER NOT NULL,
        cost_budget_rmb REAL NOT NULL,
        estimated_cost_rmb REAL NOT NULL DEFAULT 0,
        actual_cost_rmb REAL NOT NULL DEFAULT 0,
        deferred_count INTEGER NOT NULL DEFAULT 0,
        manifest_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        revoked_at TEXT,
        FOREIGN KEY(entity_id) REFERENCES entities(entity_id),
        CHECK(mode IN ('standard', 'complete')),
        CHECK(execution_location IN ('local', 'remote')),
        CHECK(state IN ('pending', 'running', 'awaiting_approval', 'awaiting_agent', 'paused_budget', 'complete', 'complete_with_terminal_gaps', 'cancelled'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_profile_enrichment_runs_entity ON profile_enrichment_runs(entity_id, updated_at)""",
    """CREATE TABLE IF NOT EXISTS profile_enrichment_tasks (
        task_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        asset_id TEXT,
        citation TEXT NOT NULL,
        modality TEXT NOT NULL,
        relevance_reason TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        content_hash TEXT,
        state TEXT NOT NULL,
        next_tool TEXT NOT NULL,
        approval_required INTEGER NOT NULL DEFAULT 0,
        approval_id TEXT,
        approval_scope_hash TEXT,
        processor_identity TEXT,
        prompt_version TEXT NOT NULL DEFAULT 'profile-enrichment/v1',
        attempt_count INTEGER NOT NULL DEFAULT 0,
        claim_token_hash TEXT,
        delivery_token_hash TEXT,
        delivery_consumed_at TEXT,
        lease_owner_hash TEXT,
        lease_expires_at TEXT,
        heartbeat_at TEXT,
        completion_key TEXT,
        terminal_reason TEXT,
        estimated_cost_rmb REAL NOT NULL DEFAULT 0,
        actual_cost_rmb REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        FOREIGN KEY(run_id) REFERENCES profile_enrichment_runs(run_id) ON DELETE CASCADE,
        FOREIGN KEY(asset_id) REFERENCES media_assets(asset_id) ON UPDATE CASCADE ON DELETE SET NULL,
        UNIQUE(run_id, citation, modality),
        CHECK(state IN ('pending', 'materializing', 'awaiting_agent', 'awaiting_approval', 'processing', 'completed', 'unavailable', 'retryable_failure', 'paused_budget', 'cancelled'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_profile_enrichment_tasks_queue ON profile_enrichment_tasks(run_id, state, updated_at)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_enrichment_tasks_completion ON profile_enrichment_tasks(completion_key) WHERE completion_key IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS profile_automation_subscriptions (
        entity_id TEXT PRIMARY KEY,
        selector TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        debounce_seconds INTEGER NOT NULL DEFAULT 180,
        consent_scope TEXT NOT NULL DEFAULT 'explicit-profile-auto-maintenance-v1',
        last_profile_id TEXT,
        last_refresh_at TEXT,
        last_error_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE ON UPDATE CASCADE,
        FOREIGN KEY(last_profile_id) REFERENCES profile_snapshots(profile_id) ON DELETE SET NULL ON UPDATE CASCADE,
        CHECK(enabled IN (0,1)),
        CHECK(debounce_seconds BETWEEN 0 AND 3600)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_profile_automation_enabled ON profile_automation_subscriptions(enabled,updated_at)""",
    """CREATE TABLE IF NOT EXISTS profile_refresh_queue (
        entity_id TEXT PRIMARY KEY,
        generation INTEGER NOT NULL DEFAULT 1,
        state TEXT NOT NULL DEFAULT 'pending',
        reason TEXT NOT NULL,
        available_at TEXT NOT NULL,
        claimed_at TEXT,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        last_error_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(entity_id) REFERENCES profile_automation_subscriptions(entity_id) ON DELETE CASCADE ON UPDATE CASCADE,
        CHECK(state IN ('pending','processing','failed'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_profile_refresh_queue_due ON profile_refresh_queue(state,available_at,updated_at)""",
    """CREATE TABLE IF NOT EXISTS evidence_items (
        evidence_id TEXT PRIMARY KEY,
        citation TEXT NOT NULL UNIQUE,
        account_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        actor TEXT NOT NULL DEFAULT '',
        timestamp TEXT,
        content TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_items_source ON evidence_items(source_type, account_id, timestamp)""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_items_citation ON evidence_items(citation)""",

    """CREATE TABLE IF NOT EXISTS evidence_chunks (
        chunk_id TEXT PRIMARY KEY,
        chunk_citation TEXT NOT NULL UNIQUE,
        parent_citation TEXT NOT NULL,
        account_id TEXT NOT NULL DEFAULT '',
        account_label TEXT NOT NULL DEFAULT '',
        source_type TEXT NOT NULL,
        source_id TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL DEFAULT '',
        actor TEXT NOT NULL DEFAULT '',
        timestamp TEXT,
        content TEXT NOT NULL DEFAULT '',
        chunk_index INTEGER NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_chunks_parent ON evidence_chunks(parent_citation, chunk_index)""",
    """CREATE INDEX IF NOT EXISTS idx_evidence_chunks_source ON evidence_chunks(source_type, account_id, timestamp)""",
    """CREATE TABLE IF NOT EXISTS local_trace_events (
        trace_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE INDEX IF NOT EXISTS idx_local_trace_events ON local_trace_events(created_at, stage, status)""",
    """CREATE TABLE IF NOT EXISTS approval_records (
        approval_id TEXT PRIMARY KEY,
        action TEXT NOT NULL,
        danger_class TEXT NOT NULL,
        status TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        requested_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        decided_at TEXT,
        decision_note TEXT,
        consumed_at TEXT,
        consumption_id TEXT
    )""",
    """CREATE INDEX IF NOT EXISTS idx_approval_records_status ON approval_records(status, requested_at)""",
    """CREATE TABLE IF NOT EXISTS operation_journal (
        operation_id TEXT PRIMARY KEY,
        capability_id TEXT NOT NULL,
        request_digest TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        replay_policy TEXT NOT NULL,
        state TEXT NOT NULL,
        stage TEXT NOT NULL,
        owner TEXT NOT NULL,
        result_json TEXT,
        error_json TEXT,
        continuation_token_hash TEXT,
        external_ref TEXT,
        version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(capability_id, idempotency_key),
        CHECK(replay_policy IN ('idempotent','journaled','never')),
        CHECK(state IN ('pending','running','awaiting_agent','reconciling','completed','failed','cancelled')),
        CHECK(owner IN ('daemon','provider','agent','none'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_operation_journal_state_updated ON operation_journal(state, updated_at)""",
    """CREATE TABLE IF NOT EXISTS derived_data_purge_audit (
        purge_id TEXT PRIMARY KEY,
        scope_type TEXT NOT NULL,
        scope_hash TEXT NOT NULL,
        lifecycle_version TEXT NOT NULL,
        status TEXT NOT NULL,
        counts_json TEXT NOT NULL DEFAULT '{}',
        backup_policy TEXT NOT NULL,
        audit_retention_until TEXT NOT NULL,
        created_at TEXT NOT NULL,
        CHECK(scope_type IN ('entity','source','run','task')),
        CHECK(status IN ('completed','failed'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_derived_data_purge_audit_retention ON derived_data_purge_audit(audit_retention_until,created_at)""",
]

SYNC_VECTOR_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sync_state (
        account_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        shard_id TEXT NOT NULL,
        max_local_id INTEGER NOT NULL DEFAULT -1,
        max_create_time INTEGER NOT NULL DEFAULT -1,
        max_timestamp TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL,
        PRIMARY KEY(account_id, conversation_id, shard_id)
    )""",
    """CREATE TABLE IF NOT EXISTS sync_dirty_citations (
        citation TEXT PRIMARY KEY,
        account_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        source_type TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS sync_aux_state (
        source_key TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS sync_citation_tombstones (
        citation TEXT PRIMARY KEY,
        account_id TEXT NOT NULL DEFAULT '',
        conversation_id TEXT NOT NULL DEFAULT '',
        source_type TEXT NOT NULL DEFAULT '',
        deleted_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_sync_tombstones_deleted ON sync_citation_tombstones(deleted_at, citation)""",
    """CREATE TABLE IF NOT EXISTS sync_message_source_rows (
        source_key TEXT NOT NULL,
        citation TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(source_key, citation)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_sync_message_source_citation ON sync_message_source_rows(citation, source_key)""",
    """CREATE TABLE IF NOT EXISTS media_source_state (
        source_key TEXT PRIMARY KEY,
        file_fingerprint TEXT NOT NULL,
        table_fingerprint TEXT NOT NULL,
        row_watermark INTEGER NOT NULL DEFAULT 0,
        row_count INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS media_source_rows (
        source_key TEXT NOT NULL,
        row_id INTEGER NOT NULL,
        row_fingerprint TEXT NOT NULL,
        asset_id TEXT NOT NULL DEFAULT '',
        citation TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL,
        PRIMARY KEY(source_key, row_id)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_media_source_rows_asset ON media_source_rows(asset_id, source_key)""",
    """CREATE TABLE IF NOT EXISTS vector_entries (
        citation TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        dimensions INTEGER NOT NULL,
        vector_json TEXT NOT NULL,
        content_hash TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS vector_index_generations (
        backend TEXT NOT NULL,
        generation_id TEXT NOT NULL,
        status TEXT NOT NULL,
        vector_text_version INTEGER NOT NULL,
        embedding_provider TEXT NOT NULL DEFAULT '',
        embedding_model TEXT NOT NULL DEFAULT '',
        dimensions INTEGER NOT NULL DEFAULT 0,
        expected_count INTEGER,
        indexed_count INTEGER NOT NULL DEFAULT 0,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        activated_at TEXT,
        PRIMARY KEY(backend, generation_id),
        CHECK(status IN ('building','ready','active','retired'))
    )""",
    """CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_generation_active ON vector_index_generations(backend) WHERE status='active'""",
    """CREATE INDEX IF NOT EXISTS idx_vector_generation_status ON vector_index_generations(backend,status,created_at)""",
    """CREATE TABLE IF NOT EXISTS vector_index_ledger (
        backend TEXT NOT NULL,
        generation_id TEXT NOT NULL,
        citation TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'indexed',
        updated_at TEXT NOT NULL,
        PRIMARY KEY(backend, generation_id, citation),
        FOREIGN KEY(backend,generation_id) REFERENCES vector_index_generations(backend,generation_id) ON DELETE CASCADE,
        CHECK(state IN ('indexed','deleted'))
    )""",
    """CREATE INDEX IF NOT EXISTS idx_vector_ledger_citation ON vector_index_ledger(backend,generation_id,state,citation)""",
]

SEARCH_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_messages_filter_time ON messages(account_id, conversation_id, conversation_type, source_type, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_context_window ON messages(account_id, conversation_id, timestamp, shard_id, local_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_sender_time ON messages(sender_id, sender_name, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_source_time ON messages(source_type, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_citation ON messages(citation)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_account_type ON conversations(account_id, type)",
    # conversation_id is only the second column of the conversations primary
    # key and of idx_messages_filter_time.  Contact resolution and scoped
    # fallback commonly know the conversation id without an account id, so
    # those paths otherwise scan the complete table.
    "CREATE INDEX IF NOT EXISTS idx_conversations_id_account ON conversations(conversation_id, account_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation_time ON messages(conversation_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_chunks_source_parent ON evidence_chunks(source_type, parent_citation)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_chunks_filter_time ON evidence_chunks(account_id, source_type, source_id, actor, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_chunks_source_id_status_time ON evidence_chunks(source_id, status, timestamp)",
]

# The vector publish CAS must follow only rows that can change the document
# stream sent to an embedding provider.  Keeping this monotonic revision in the
# same SQLite transaction as the source mutation makes the final comparison
# constant-time without treating unrelated profile/media writes as changes.
VECTOR_SOURCE_TRIGGER_NAMES = [
    'vector_source_messages_ai',
    'vector_source_messages_ad',
    'vector_source_messages_au',
    'vector_source_chunks_ai',
    'vector_source_chunks_ad',
    'vector_source_chunks_au',
]

VECTOR_SOURCE_REVISION_SCHEMA = [
    """CREATE TRIGGER IF NOT EXISTS vector_source_messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO schema_meta(key,value) VALUES('vector_source_revision','1')
        ON CONFLICT(key) DO UPDATE
        SET value=CAST(CAST(schema_meta.value AS INTEGER)+1 AS TEXT);
    END""",
    """CREATE TRIGGER IF NOT EXISTS vector_source_messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO schema_meta(key,value) VALUES('vector_source_revision','1')
        ON CONFLICT(key) DO UPDATE
        SET value=CAST(CAST(schema_meta.value AS INTEGER)+1 AS TEXT);
    END""",
    """CREATE TRIGGER IF NOT EXISTS vector_source_messages_au
        AFTER UPDATE OF citation,conversation_title,conversation_type,sender_name,
                        timestamp,content,content_kind,source_type,direction ON messages
        WHEN OLD.citation IS NOT NEW.citation
          OR OLD.conversation_title IS NOT NEW.conversation_title
          OR OLD.conversation_type IS NOT NEW.conversation_type
          OR OLD.sender_name IS NOT NEW.sender_name
          OR OLD.timestamp IS NOT NEW.timestamp
          OR OLD.content IS NOT NEW.content
          OR OLD.content_kind IS NOT NEW.content_kind
          OR OLD.source_type IS NOT NEW.source_type
          OR OLD.direction IS NOT NEW.direction
    BEGIN
        INSERT INTO schema_meta(key,value) VALUES('vector_source_revision','1')
        ON CONFLICT(key) DO UPDATE
        SET value=CAST(CAST(schema_meta.value AS INTEGER)+1 AS TEXT);
    END""",
    """CREATE TRIGGER IF NOT EXISTS vector_source_chunks_ai AFTER INSERT ON evidence_chunks
        WHEN NEW.status='active'
    BEGIN
        INSERT INTO schema_meta(key,value) VALUES('vector_source_revision','1')
        ON CONFLICT(key) DO UPDATE
        SET value=CAST(CAST(schema_meta.value AS INTEGER)+1 AS TEXT);
    END""",
    """CREATE TRIGGER IF NOT EXISTS vector_source_chunks_ad AFTER DELETE ON evidence_chunks
        WHEN OLD.status='active'
    BEGIN
        INSERT INTO schema_meta(key,value) VALUES('vector_source_revision','1')
        ON CONFLICT(key) DO UPDATE
        SET value=CAST(CAST(schema_meta.value AS INTEGER)+1 AS TEXT);
    END""",
    """CREATE TRIGGER IF NOT EXISTS vector_source_chunks_au
        AFTER UPDATE OF chunk_citation,parent_citation,title,actor,timestamp,content,source_type,status
        ON evidence_chunks
        WHEN (OLD.status='active' OR NEW.status='active')
         AND (OLD.chunk_citation IS NOT NEW.chunk_citation
          OR OLD.parent_citation IS NOT NEW.parent_citation
          OR OLD.title IS NOT NEW.title
          OR OLD.actor IS NOT NEW.actor
          OR OLD.timestamp IS NOT NEW.timestamp
          OR OLD.content IS NOT NEW.content
          OR OLD.source_type IS NOT NEW.source_type
          OR OLD.status IS NOT NEW.status)
    BEGIN
        INSERT INTO schema_meta(key,value) VALUES('vector_source_revision','1')
        ON CONFLICT(key) DO UPDATE
        SET value=CAST(CAST(schema_meta.value AS INTEGER)+1 AS TEXT);
    END""",
]

FTS_TRIGGER_NAMES = [
    'message_fts_ai',
    'message_fts_ad',
    'message_fts_au',
    'chunk_fts_ai',
    'chunk_fts_ad',
    'chunk_fts_au',
]

TRIGRAM_FTS_SCHEMA = [
    """CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
        citation UNINDEXED,
        content,
        sender_name,
        conversation_title,
        tokenize='trigram',
        content='messages',
        content_rowid='id'
    )""",
    """CREATE TRIGGER IF NOT EXISTS message_fts_ai AFTER INSERT ON messages BEGIN
        INSERT INTO message_fts(rowid,citation,content,sender_name,conversation_title)
        VALUES (new.id,new.citation,new.content,new.sender_name,new.conversation_title);
    END""",
    """CREATE TRIGGER IF NOT EXISTS message_fts_ad AFTER DELETE ON messages BEGIN
        INSERT INTO message_fts(message_fts,rowid,citation,content,sender_name,conversation_title)
        VALUES('delete',old.id,old.citation,old.content,old.sender_name,old.conversation_title);
    END""",
    """CREATE TRIGGER IF NOT EXISTS message_fts_au AFTER UPDATE ON messages BEGIN
        INSERT INTO message_fts(message_fts,rowid,citation,content,sender_name,conversation_title)
        VALUES('delete',old.id,old.citation,old.content,old.sender_name,old.conversation_title);
        INSERT INTO message_fts(rowid,citation,content,sender_name,conversation_title)
        VALUES (new.id,new.citation,new.content,new.sender_name,new.conversation_title);
    END""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        chunk_citation UNINDEXED,
        content,
        title,
        actor,
        tokenize='trigram',
        content='evidence_chunks',
        content_rowid='rowid'
    )""",
    """CREATE TRIGGER IF NOT EXISTS chunk_fts_ai AFTER INSERT ON evidence_chunks BEGIN
        INSERT INTO chunk_fts(rowid,chunk_citation,content,title,actor)
        VALUES (new.rowid,new.chunk_citation,new.content,new.title,new.actor);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_fts_ad AFTER DELETE ON evidence_chunks BEGIN
        INSERT INTO chunk_fts(chunk_fts,rowid,chunk_citation,content,title,actor)
        VALUES('delete',old.rowid,old.chunk_citation,old.content,old.title,old.actor);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_fts_au AFTER UPDATE ON evidence_chunks BEGIN
        INSERT INTO chunk_fts(chunk_fts,rowid,chunk_citation,content,title,actor)
        VALUES('delete',old.rowid,old.chunk_citation,old.content,old.title,old.actor);
        INSERT INTO chunk_fts(rowid,chunk_citation,content,title,actor)
        VALUES (new.rowid,new.chunk_citation,new.content,new.title,new.actor);
    END""",
]

# Persistent DDL has one owner. Runtime/sync/vector modules consume this manifest;
# they never create their own tables or indexes.
PERSISTENT_SCHEMA = [
    *BASE_SCHEMA,
    *MULTIMODAL_SCHEMA,
    *SYNC_VECTOR_SCHEMA,
    *SEARCH_INDEXES,
    *VECTOR_SOURCE_REVISION_SCHEMA,
]
COMPLETE_SCHEMA_MANIFEST = [*PERSISTENT_SCHEMA, *TRIGRAM_FTS_SCHEMA]

REQUIRED_COLUMNS = {
    'messages': {'content_kind'},
    'moment_interactions': {'actor_name'},
    'sync_dirty_citations': {'source_type'},
    'vector_entries': {'content_hash'},
    'vector_index_generations': {'revision'},
    'approval_records': {'consumed_at', 'consumption_id'},
}

EXPECTED_INDEX_COLUMNS = {
    'idx_messages_filter_time': ('account_id', 'conversation_id', 'conversation_type', 'source_type', 'timestamp'),
    'idx_messages_context_window': ('account_id', 'conversation_id', 'timestamp', 'shard_id', 'local_id'),
    'idx_conversations_id_account': ('conversation_id', 'account_id'),
    'idx_messages_conversation_time': ('conversation_id', 'timestamp'),
    'idx_evidence_chunks_parent': ('parent_citation', 'chunk_index'),
    'idx_evidence_chunks_source_parent': ('source_type', 'parent_citation'),
    'idx_evidence_chunks_filter_time': ('account_id', 'source_type', 'source_id', 'actor', 'timestamp'),
    'idx_evidence_chunks_source_id_status_time': ('source_id', 'status', 'timestamp'),
}
