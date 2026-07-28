from __future__ import annotations
from pathlib import Path
import tempfile

from trove_core.store.repositories import ImageObservationRecord, MediaAssetRecord, MultimodalRepository, ProviderJobRecord, TranscriptRecord, WeChatRepository
from trove_core.store.sqlite_store import SQLiteStore
from trove_core.vault.config import VaultConfig
from trove_core.vault.mutations import coordinated_vault_mutation, mutation_entrypoint
from .fixture_factory import generate_fixture, FixtureData
from .fixture_guard import FixtureVaultGuardError, fixture_vault_session, normalize_fixture_root, prepare_fixture_vault


VOICE_FIXTURES = [
    ('voice-1', 'trove://wechat/acct-work/conv-example_edu-private/message_0/1#voice-fixture-1', '语音证据 示例教育确认校长下周三参加试点评审'),
    ('voice-2', 'trove://wechat/acct-work/conv-example_edu-private/message_0/2#voice-fixture-2', '语音证据 林老师说预算审批可以先走基础版'),
    ('voice-3', 'trove://wechat/acct-work/conv-sales-review/message_0/10#voice-fixture-3', '语音证据 成员甲提醒新版报价要突出三个月试点'),
    ('voice-4', 'trove://wechat/acct-work/conv-trove-team/message_0/20#voice-fixture-4', '语音证据 产品会议决定本地检索必须 evidence first'),
    ('voice-5', 'trove://wechat/acct-work/conv-sales-review/message_0/11#voice-fixture-5', '语音证据 客户提到暑期招生海报需要 OCR 复核'),
    ('voice-6', 'trove://wechat/acct-work/conv-sales-review/message_0/12#voice-fixture-6', '语音证据 售后回访记录希望周五前给实施排期'),
    ('voice-7', 'trove://wechat/acct-work/conv-sales-review/message_0/11#voice-fixture-7', '语音证据 运营小周建议先降低首单风险'),
    ('voice-8', 'trove://wechat/acct-personal/conv-family/message_0/1#voice-fixture-8', '语音证据 家庭消息不进入客户画像'),
]

IMAGE_FIXTURES = [
    ('image-1', 'trove://wechat/acct-work/conv-example_edu-private/message_0/1#image-fixture-1', '图片OCR证据 示例教育基础版报价单'),
    ('image-2', 'trove://wechat/acct-work/conv-example_edu-private/message_1/1#image-fixture-2', '图片OCR证据 三个月试点排期表'),
    ('image-3', 'trove://wechat/acct-work/conv-sales-review/message_0/10#image-fixture-3', '图片OCR证据 预算审批流程截图'),
    ('image-4', 'trove://wechat/acct-work/conv-trove-team/message_0/22#image-fixture-4', '图片OCR证据 桌面UI设置面板草图'),
    ('image-5', 'trove://wechat/acct-work/conv-trove-team/message_0/21#image-fixture-5', '图片OCR证据 本地API token说明'),
    ('image-6', 'trove://wechat/acct-work/conv-sales-review/message_0/12#image-fixture-6', '图片OCR证据 新版报价责任分工'),
    ('image-7', 'trove://wechat/acct-work/conv-sales-review/message_0/11#image-fixture-7', '图片OCR证据 降低首单风险复盘'),
    ('image-8', 'trove://wechat/acct-personal/conv-family/message_0/1#image-fixture-8', '图片OCR证据 家庭采购清单不进客户筛选'),
]

_FIXTURE_GENERATED_AT = '2026-06-21T00:00:00Z'


def _add_fixture_multimodal(store: SQLiteStore) -> None:
    repo = MultimodalRepository(store)
    for asset_id, citation, text in VOICE_FIXTURES:
        repo.upsert_media_asset(MediaAssetRecord(
            asset_id=f'fixture-{asset_id}',
            account_id='acct-work' if 'acct-work' in citation else 'acct-personal',
            source_type='message',
            source_id=citation.split('#', 1)[0],
            modality='voice',
            media_type='voice',
            citation=citation.rsplit('#', 1)[0],
            content_hash=asset_id,
            cache_state='metadata_only',
            processing_state='done',
        ))
        repo.record_provider_job(ProviderJobRecord(
            job_id=f'fixture-cloud-job-{asset_id}',
            asset_id=f'fixture-{asset_id}',
            provider='volcengine-asr-flash',
            model='bigmodel:volc.bigasr.auc_turbo',
            job_type='asr',
            status='completed',
            request_hash=asset_id,
            citation=citation,
        ))
        repo.insert_transcript(TranscriptRecord(
            transcript_id=f'fixture-transcript-{asset_id}',
            asset_id=f'fixture-{asset_id}',
            citation=citation,
            text=text,
            language='zh',
            confidence=1.0,
            duration_seconds=3.0,
            job_id=f'fixture-cloud-job-{asset_id}',
        ))
    for asset_id, citation, visible_text in IMAGE_FIXTURES:
        repo.upsert_media_asset(MediaAssetRecord(
            asset_id=f'fixture-{asset_id}',
            account_id='acct-work' if 'acct-work' in citation else 'acct-personal',
            source_type='message',
            source_id=citation.split('#', 1)[0],
            modality='image',
            media_type='image',
            citation=citation.rsplit('#', 1)[0],
            content_hash=asset_id,
            cache_state='metadata_only',
            processing_state='done',
        ))
        repo.insert_image_observation(ImageObservationRecord(
            observation_id=f'fixture-imageobs-{asset_id}',
            asset_id=f'fixture-{asset_id}',
            citation=citation,
            caption='',
            visible_text=visible_text,
            confidence=1.0,
            status='active',
        ))


def _stabilize_fixture_database(store: SQLiteStore) -> None:
    """Remove wall-clock noise so an identical fixture is byte-reusable."""

    connection = store.connect()
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    for table in tables:
        columns = {
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        for column in ("created_at", "updated_at"):
            if column in columns:
                cursor = connection.execute(
                    f'UPDATE "{table}" SET "{column}"=?',
                    (_FIXTURE_GENERATED_AT,),
                )
                cursor.close()
    cursor = connection.execute(
        """UPDATE evidence_chunks
           SET timestamp=?
           WHERE source_type IN ('transcript', 'image_observation')""",
        (_FIXTURE_GENERATED_AT,),
    )
    cursor.close()
    connection.commit()
    cursor = connection.execute('VACUUM')
    cursor.close()


def _fixture_config(vault_root: Path) -> VaultConfig:
    cfg = VaultConfig.resolve(str(normalize_fixture_root(vault_root)), env={})
    cfg.validate_runtime_path()
    cfg.require_configured_for_write(action='fixture indexing')
    return cfg


@mutation_entrypoint('fixture_generation')
def _index_fixture_data(
    cfg: VaultConfig,
    data: FixtureData,
    *,
    reset: bool,
    add_multimodal: bool,
    write_jsonl: bool,
) -> dict:
    identity = prepare_fixture_vault(cfg.root)
    # Every fixture generation is built away from the target Vault. `reset`
    # remains an API compatibility flag; publication is always an immutable
    # staged generation followed by an atomic, guarded replacement.
    _ = reset
    with tempfile.TemporaryDirectory(prefix='trove-fixture-stage-') as stage_directory:
        stage_root = Path(stage_directory)
        staged_sqlite = stage_root / 'index' / 'trove.sqlite'
        staged_sqlite.parent.mkdir(parents=True)
        store = SQLiteStore(staged_sqlite)
        try:
            changed = WeChatRepository(store).replace_fixture(data.accounts, data.conversations, data.messages)
            chunks = store.rebuild_evidence_chunks()
            if add_multimodal:
                _add_fixture_multimodal(store)
            _stabilize_fixture_database(store)
            counts = store.counts()
            checkpoint = store.connect().execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
            if checkpoint is None or int(checkpoint[0]) != 0:
                raise FixtureVaultGuardError('fixture_stage_checkpoint_failed')
        finally:
            store.close()
        staged_jsonl = stage_root / 'fixtures' / 'synthetic' / 'messages.jsonl'
        if write_jsonl:
            data.to_jsonl(staged_jsonl)

        with fixture_vault_session(cfg.root, expected_identity=identity, allow_create=False) as guard:
            with coordinated_vault_mutation(cfg, operation='fixture_generation'):
                guard.recover_pending_generation()
                guard.validate_current()
                guard.ensure_product_directories()
                was_provisional = guard.identity.provisional
                if was_provisional:
                    guard.validate_provisional_layout()
                artifact = guard.publish_sqlite_generation(staged_sqlite)
                ready_publication = None
                try:
                    if artifact.reused:
                        artifact.verify(guard.root_fd, integrity=True)
                    else:
                        ready_publication = guard.mark_generation_ready(artifact)
                        guard.finalize_generation(artifact, ready_publication)
                finally:
                    artifact.close()
                    if ready_publication is not None:
                        ready_publication.close()
                if write_jsonl:
                    guard.publish_file(
                        staged_jsonl,
                        ('fixtures', 'synthetic', 'messages.jsonl'),
                        require_absent=was_provisional,
                    )
                guard.validate_current()
    return {'vault': str(cfg.root), 'sqlite': str(cfg.paths.sqlite_path), 'changed': changed, 'chunks': chunks, 'counts': counts}


@mutation_entrypoint('fixture_generation')
def index_fixture_vault(
    vault_root: Path,
    seed: int = 20260621,
    reset: bool = False,
    write_jsonl: bool = False,
) -> dict:
    return _index_fixture_data(
        _fixture_config(vault_root),
        generate_fixture(seed),
        reset=reset,
        add_multimodal=True,
        write_jsonl=write_jsonl,
    )


@mutation_entrypoint('fixture_generation')
def index_fixture_data(vault_root: Path, data: FixtureData, reset: bool = False) -> dict:
    return _index_fixture_data(
        _fixture_config(vault_root),
        data,
        reset=reset,
        add_multimodal=False,
        write_jsonl=False,
    )
