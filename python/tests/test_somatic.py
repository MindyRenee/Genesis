"""Somatic state-dependent recall — the body a memory was encoded in."""

from __future__ import annotations

from genesis_conscious.memory.engine import MemoryEngine, MemoryRecord
from genesis_conscious.self.somatic import SomaticSnapshot, somatic_congruence


def test_congruence_identical_is_one():
    snap = SomaticSnapshot(arousal=0.6, valence=0.2, tone=0.5)
    assert somatic_congruence(snap, snap) == 1.0


def test_congruence_missing_is_neutral():
    snap = SomaticSnapshot()
    assert somatic_congruence(None, snap) == 0.5
    assert somatic_congruence(snap, None) == 0.5
    assert somatic_congruence(snap.to_dict(), None) == 0.5


def test_congruence_diverges_with_distance():
    calm = SomaticSnapshot(arousal=0.2, valence=0.8)
    tense = SomaticSnapshot(arousal=0.9, valence=-0.8)
    near = SomaticSnapshot(arousal=0.25, valence=0.75)
    assert somatic_congruence(calm, near) > somatic_congruence(calm, tense)


def test_snapshot_roundtrip_ignores_legacy_keys():
    snap = SomaticSnapshot(arousal=0.7, valence=-0.3)
    assert SomaticSnapshot.from_dict(snap.to_dict()) is not None
    # Snapshots written before the age field was removed still load.
    legacy = dict(snap.to_dict())
    legacy["age_episodes"] = 42.0
    rebuilt = SomaticSnapshot.from_dict(legacy)
    assert rebuilt is not None and rebuilt.arousal == snap.arousal
    assert SomaticSnapshot.from_dict(None) is None
    assert SomaticSnapshot.from_dict({}) is None


def test_somatic_rerank_prefers_congruent_body():
    from genesis_client.types import SimilarEpisode

    class _Client:
        def find_similar(self, query, limit=5):
            return [
                SimilarEpisode(episode_id=1, hamming_distance=10, timestamp=0, salience=0.5),
                SimilarEpisode(episode_id=2, hamming_distance=10, timestamp=0, salience=0.5),
            ]

        def store_episode(self, **kwargs):
            raise AssertionError("not used")

    engine = MemoryEngine(client=_Client())  # type: ignore[arg-type]
    engine._records[1] = MemoryRecord(
        episode_id=1,
        somatic=SomaticSnapshot(arousal=0.9, valence=0.8).to_dict(),
    )
    engine._records[2] = MemoryRecord(
        episode_id=2,
        somatic=SomaticSnapshot(arousal=0.1, valence=-0.8).to_dict(),
    )
    engine.somatic_provider = lambda: SomaticSnapshot(arousal=0.9, valence=0.8).to_dict()
    engine.somatic_weight = 0.3
    ranked = engine._rank_by_somatic_congruence(
        [
            SimilarEpisode(episode_id=1, hamming_distance=10, timestamp=0, salience=0.5),
            SimilarEpisode(episode_id=2, hamming_distance=10, timestamp=0, salience=0.5),
        ]
    )
    assert [r.episode_id for r in ranked] == [1, 2]
    # And without a provider the order is untouched.
    engine2 = MemoryEngine(client=_Client())  # type: ignore[arg-type]
    untouched = engine2._rank_by_somatic_congruence(
        [
            SimilarEpisode(episode_id=1, hamming_distance=10, timestamp=0, salience=0.5),
        ]
    )
    assert [r.episode_id for r in untouched] == [1]


def test_somatic_survives_serialization_roundtrip():
    class _Client:
        def find_similar(self, query, limit=5):
            return []

        def store_episode(self, **kwargs):
            raise AssertionError("not used")

    engine = MemoryEngine(client=_Client())  # type: ignore[arg-type]
    engine._records[7] = MemoryRecord(
        episode_id=7,
        somatic=SomaticSnapshot(arousal=0.3, valence=0.4).to_dict(),
    )
    payload = engine.serialize_records()
    engine2 = MemoryEngine(client=_Client())  # type: ignore[arg-type]
    engine2.restore_records(payload)
    rec = engine2._records[7]
    assert rec.somatic is not None
    assert abs(rec.somatic["arousal"] - 0.3) < 1e-9
    # Records written with a legacy developmental_stage key still load.
    payload["records"][0]["developmental_stage"] = "TRUST_VS_MISTRUST"
    engine3 = MemoryEngine(client=_Client())  # type: ignore[arg-type]
    engine3.restore_records(payload)
    assert engine3._records[7].somatic is not None
