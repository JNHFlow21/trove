from __future__ import annotations

from trove_core.store.repositories import MultimodalRepository, ObservationRecord


def set_observation_status(repo: MultimodalRepository, observation_id: str, status: str) -> dict:
    if status not in repo.OBSERVATION_STATUSES:
        raise ValueError(f'invalid observation status: {status}')
    with repo.store.connect() as conn:
        row = conn.execute('SELECT * FROM observations WHERE observation_id=?', (observation_id,)).fetchone()
        if row is None:
            raise KeyError(observation_id)
        conn.execute('UPDATE observations SET status=?, updated_at=datetime(\'now\') WHERE observation_id=?', (status, observation_id))
        conn.commit()
        return dict(conn.execute('SELECT * FROM observations WHERE observation_id=?', (observation_id,)).fetchone())


def supersede_observation(repo: MultimodalRepository, old_observation_id: str, new_observation: ObservationRecord) -> dict:
    set_observation_status(repo, old_observation_id, 'superseded')
    if not new_observation.supersedes_observation_id:
        new_observation = ObservationRecord(
            observation_id=new_observation.observation_id,
            entity_id=new_observation.entity_id,
            observation_type=new_observation.observation_type,
            value=new_observation.value,
            status=new_observation.status,
            confidence=new_observation.confidence,
            citation=new_observation.citation,
            source_type=new_observation.source_type,
            valid_from=new_observation.valid_from,
            supersedes_observation_id=old_observation_id,
        )
    return repo.add_observation(new_observation)
