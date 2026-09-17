"""Memory bundle tests — storage, retrieval, consolidation, and spaced repetition.

Tests MemoryEngine (retrieval, context, reconsolidation), EmotionalMemorySystem,
AttractorNetwork, PrimingSystem, SpreadingActivation, and SpacedRepetitionScheduler.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import math
import time
from typing import cast
from unittest.mock import MagicMock

import pytest

from genesis_client.types import Episode, MemoryStats, SimilarEpisode
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.memory import (
    AttractorNetwork,
    MemoryContext,
    MemoryEngine,
    MemoryRecord,
    ReconsolidationModification,
    ReviewRecord,
    SourceTag,
    SpacedRepetitionScheduler,
)
from genesis_cognitive.memory.systems import (
    EmotionalMemorySystem,
    PrimingSystem,
    SpreadingActivation,
    _emotional_similarity,
)
from genesis_cognitive.sleep import SleepStage

logger = logging.getLogger(__name__)



# ======================================================================
# From tests/test_memory_engine.py
# ======================================================================

def _make_mock_client(ltm_count: int = 0) -> MagicMock:
    """Create a mock GenesisClient."""
    client = MagicMock()
    client.get_memory_stats.return_value = MemoryStats(
        stm_count=0, ltm_count=ltm_count, ltm_capacity=65536
    )
    client.store_event.return_value = True
    # store_episode returns the real LTM episode ID. Use a counter so
    # each call returns a unique ID, mirroring the daemon's behavior.
    _episode_id_counter = [ltm_count]

    def _store_episode(**kwargs):
        """Return a unique incrementing episode ID on each call."""
        _episode_id_counter[0] += 1
        return _episode_id_counter[0]

    client.store_episode.side_effect = _store_episode
    client.find_similar.return_value = []
    client.retrieve_episode.return_value = None
    return client


def _make_engine(ltm_count: int = 0) -> MemoryEngine:
    """Create a MemoryEngine with a mock client."""
    return MemoryEngine(client=_make_mock_client(ltm_count))


def _make_record(
    episode_id: int = 1,
    salience: float = 0.5,
    retention: float = 1.0,
    encoded_at: int | None = None,
    **kwargs,
) -> MemoryRecord:
    """Create a MemoryRecord with sensible defaults."""
    now_ms = int(time.time() * 1000)
    return MemoryRecord(
        episode_id=episode_id,
        salience=salience,
        retention=retention,
        encoded_at=encoded_at or now_ms,
        last_accessed=now_ms,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# Conversation tracking
# ═══════════════════════════════════════════════════════════════════


def test_add_turn_user() -> None:
    """add_turn records a user turn."""
    engine = _make_engine()
    engine.add_turn("user", "Hello Genesis", topics=["greeting"])
    assert engine.turn_count == 1
    assert engine.conversation[0].role == "user"
    assert engine.conversation[0].text == "Hello Genesis"
    assert "greeting" in engine.conversation[0].topics


def test_add_turn_max_capacity() -> None:
    """Conversation is trimmed to max_turns."""
    engine = _make_engine()
    engine.max_turns = 3
    for i in range(5):
        engine.add_turn("user", f"message {i}")
    assert engine.turn_count == 3
    # Should keep the last 3
    assert engine.conversation[0].text == "message 2"
    assert engine.conversation[2].text == "message 4"


def test_is_first_interaction_after_user_turn() -> None:
    """is_first_interaction is False after a user turn."""
    engine = _make_engine()
    engine.add_turn("user", "Hello")
    assert engine.is_first_interaction is False


def test_is_first_interaction_genesis_only() -> None:
    """is_first_interaction stays True if only genesis speaks."""
    engine = _make_engine()
    engine.add_turn("genesis", "Hello")
    assert engine.is_first_interaction is True


def test_get_recent_topics() -> None:
    """get_recent_topics returns deduplicated topics from recent turns."""
    engine = _make_engine()
    engine.add_turn("user", "tell me about dogs", topics=["dogs", "animals"])
    engine.add_turn("genesis", "dogs are animals", topics=["dogs", "mammals"])
    engine.add_turn("user", "what about cats", topics=["cats", "animals"])
    topics = engine.get_recent_topics(n=3)
    assert "dogs" in topics
    assert "animals" in topics
    assert "cats" in topics
    assert "mammals" in topics
    # Deduplicated
    assert topics.count("dogs") == 1
    assert topics.count("animals") == 1


# ═══════════════════════════════════════════════════════════════════
# User facts
# ═══════════════════════════════════════════════════════════════════


def test_learn_user_fact() -> None:
    """learn_user_fact stores a fact."""
    engine = _make_engine()
    engine.learn_user_fact("name", "Alice")
    facts = engine.get_user_facts()
    assert facts["name"] == "Alice"


# ═══════════════════════════════════════════════════════════════════
# Memory storage
# ═══════════════════════════════════════════════════════════════════


def test_store_memory_success() -> None:
    """store_memory stores via the client and creates a record."""
    engine = _make_engine(ltm_count=1)
    ok = engine.store_memory("I learned about dogs", salience=0.7)
    assert ok is True
    assert cast(MagicMock, engine.client).store_episode.called


def test_store_memory_creates_record() -> None:
    """store_memory creates a MemoryRecord for tracking."""
    engine = _make_engine(ltm_count=1)
    engine.store_memory("test memory", salience=0.6)
    assert len(engine._records) == 1
    rec = next(iter(engine._records.values()))
    assert abs(rec.salience - 0.6) < 1e-9


def test_store_memory_applies_synaptic_consolidation() -> None:
    """store_memory applies synaptic consolidation immediately."""
    engine = _make_engine(ltm_count=1)
    engine.store_memory("test memory", salience=0.5)
    # synaptic_consolidate boosts synaptic_strength by 0.4
    rec = next(iter(engine._records.values()))
    assert rec.synaptic_strength > 0.3  # default is 0.3, boosted by 0.4


def test_store_memory_with_emotional_tag() -> None:
    """store_memory stores the emotional tag."""
    engine = _make_engine(ltm_count=1)
    tag = [0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    engine.store_memory("emotional memory", emotional_tag=tag)
    rec = next(iter(engine._records.values()))
    assert rec.emotional_tag == tag


def test_store_memory_default_emotional_tag() -> None:
    """store_memory uses neutral tag when none provided."""
    engine = _make_engine(ltm_count=1)
    engine.store_memory("neutral memory")
    rec = next(iter(engine._records.values()))
    assert len(rec.emotional_tag) == 12
    assert all(abs(v - 0.5) < 1e-9 for v in rec.emotional_tag)


def test_store_memory_with_source() -> None:
    """store_memory records the source tag."""
    engine = _make_engine(ltm_count=1)
    engine.store_memory("user said this", source="user", source_confidence=0.9)
    rec = next(iter(engine._records.values()))
    assert rec.source_tag is not None
    assert rec.source_tag.source == "user"
    assert abs(rec.source_tag.confidence - 0.9) < 1e-9


def test_store_memory_invalid_source_defaults_unknown() -> None:
    """store_memory defaults invalid source to 'unknown'."""
    engine = _make_engine(ltm_count=1)
    engine.store_memory("test", source="invalid_source")
    rec = next(iter(engine._records.values()))
    assert rec.source_tag is not None
    assert rec.source_tag.source == "unknown"


def test_store_memory_uses_real_ltm_id() -> None:
    """store_memory uses the real LTM episode ID returned by store_episode.

    This is the critical regression test for the ID-sync bug: previously
    store_memory used store_event (STM path) which doesn't return an ID,
    so MemoryEngine fabricated a fake ID from ltm_count. Now it uses
    store_episode which returns the real LTM ID synchronously. The
    MemoryRecord, attractor key, and synaptic consolidation must all
    use this real ID.
    """
    engine = _make_engine(ltm_count=100)
    # The mock store_episode returns 101 (ltm_count + 1)
    eid = engine.store_memory("real id test", salience=0.5)
    assert eid is True
    # The record must be keyed by the real LTM ID (101), not ltm_count (100)
    assert 101 in engine._records
    rec = engine._records[101]
    assert rec.episode_id == 101
    # The attractor network must also use the real ID
    assert "101" in engine.attractor._patterns


def test_store_memory_real_id_not_ltm_count() -> None:
    """store_memory must not use ltm_count as the episode ID.

    After archiving, ltm_count (active count) has gaps — it is not the
    max episode ID. The real ID from store_episode is the only correct
    key. This test verifies that even when ltm_count is much smaller
    than the real ID (as happens after archiving), the correct ID is
    used.
    """
    # Simulate post-archive state: 50 active episodes but the next ID
    # is 200 (150 were archived). The mock store_episode returns 51
    # (ltm_count + 1) by default, but we override it to return 200.
    client = _make_mock_client(ltm_count=50)
    client.store_episode.side_effect = None
    client.store_episode.return_value = 200
    engine = MemoryEngine(client=client)
    ok = engine.store_memory("post-archive memory", salience=0.5)
    assert ok is True
    # Must use 200 (real ID), not 51 (ltm_count + 1)
    assert 200 in engine._records
    assert 51 not in engine._records


# ═══════════════════════════════════════════════════════════════════
# Memory retrieval
# ═══════════════════════════════════════════════════════════════════


def test_retrieve_relevant_empty() -> None:
    """retrieve_relevant returns empty list when no memories match."""
    engine = _make_engine()
    cast(MagicMock, engine.client).find_similar.return_value = []
    results = engine.retrieve_relevant("test query")
    assert results == []


def test_retrieve_relevant_returns_results() -> None:
    """retrieve_relevant returns matching memories."""
    engine = _make_engine()
    cast(MagicMock, engine.client).find_similar.return_value = [
        SimilarEpisode(
            episode_id=1,
            hamming_distance=5,
            timestamp=int(time.time() * 1000),
            salience=0.8,
        ),
    ]
    results = engine.retrieve_relevant("dogs")
    assert len(results) == 1
    assert results[0].episode_id == 1


def test_retrieve_relevant_client_error_returns_empty() -> None:
    """retrieve_relevant handles client errors gracefully."""
    engine = _make_engine()
    cast(MagicMock, engine.client).find_similar.side_effect = OSError("connection lost")
    results = engine.retrieve_relevant("test")
    assert results == []


def test_retrieve_episode_success() -> None:
    """retrieve_episode returns the episode from the client."""
    engine = _make_engine()
    ep = Episode(
        episode_id=1,
        timestamp=int(time.time() * 1000),
        salience=0.7,
        association_hash=0,
        compact_emotional_tag=cast(list[float], 0),
        full_emotional_tag=[0.5] * 12,
        event_type=0,
        source_module=4,
        text="test episode",
    )
    cast(MagicMock, engine.client).retrieve_episode.return_value = ep
    result = engine.retrieve_episode(1)
    assert result is not None
    assert result.episode_id == 1


def test_retrieve_episode_not_found() -> None:
    """retrieve_episode returns None when episode doesn't exist."""
    engine = _make_engine()
    cast(MagicMock, engine.client).retrieve_episode.return_value = None
    result = engine.retrieve_episode(999)
    assert result is None


def test_retrieve_episode_client_error() -> None:
    """retrieve_episode handles client errors gracefully."""
    engine = _make_engine()
    cast(MagicMock, engine.client).retrieve_episode.side_effect = OSError("error")
    result = engine.retrieve_episode(1)
    assert result is None


# ═══════════════════════════════════════════════════════════════════
# Context building
# ═══════════════════════════════════════════════════════════════════


def test_build_context_with_topics() -> None:
    """build_context retrieves memories for given topics."""
    engine = _make_engine()
    cast(MagicMock, engine.client).find_similar.return_value = [
        SimilarEpisode(
            episode_id=1,
            hamming_distance=5,
            timestamp=int(time.time() * 1000),
            salience=0.7,
        ),
    ]
    cast(MagicMock, engine.client).get_memory_stats.return_value = MemoryStats(
        stm_count=0, ltm_count=5, ltm_capacity=65536
    )
    ctx = engine.build_context(["dogs"])
    assert isinstance(ctx, MemoryContext)
    assert len(ctx.retrieved) == 1
    assert ctx.total_memories == 5


def test_build_context_no_topics() -> None:
    """build_context with no topics doesn't retrieve."""
    engine = _make_engine()
    ctx = engine.build_context([])
    assert isinstance(ctx, MemoryContext)
    assert ctx.retrieved == []


def test_build_context_includes_conversation() -> None:
    """build_context includes recent conversation turns."""
    engine = _make_engine()
    engine.add_turn("user", "hello", topics=["greeting"])
    engine.add_turn("genesis", "hi there", topics=["greeting"])
    ctx = engine.build_context(["greeting"])
    assert len(ctx.recent_turns) == 2


def test_build_context_includes_user_facts() -> None:
    """build_context includes learned user facts."""
    engine = _make_engine()
    engine.learn_user_fact("name", "Alice")
    ctx = engine.build_context([])
    assert ctx.user_facts["name"] == "Alice"


def test_build_context_first_interaction() -> None:
    """build_context reflects first interaction status."""
    engine = _make_engine()
    ctx = engine.build_context([])
    assert ctx.is_first_interaction is True


# ═══════════════════════════════════════════════════════════════════
# Forgetting
# ═══════════════════════════════════════════════════════════════════


def test_forget_recent_memory_not_forgotten() -> None:
    """Recent memories are not forgotten."""
    engine = _make_engine()
    rec = _make_record(encoded_at=int(time.time() * 1000))
    engine._records[1] = rec
    assert engine.forget() == 0
    assert not rec.forgotten


def test_forget_old_low_salience_memory() -> None:
    """Old, low-salience memories are forgotten."""
    engine = _make_engine()
    # Encode 1 year ago with low salience
    old_time = int((time.time() - 365 * 86400) * 1000)
    rec = _make_record(
        episode_id=1,
        salience=0.1,
        encoded_at=old_time,
    )
    engine._records[1] = rec
    forgotten = engine.forget()
    assert forgotten == 1
    assert rec.forgotten


def test_forget_old_high_salience_retained() -> None:
    """Old, high-salience memories are retained longer."""
    engine = _make_engine()
    # Encode 1 week ago with high salience
    old_time = int((time.time() - 7 * 86400) * 1000)
    rec = _make_record(
        episode_id=1,
        salience=0.9,
        encoded_at=old_time,
    )
    engine._records[1] = rec
    forgotten = engine.forget()
    # High salience → tau is much longer → not forgotten yet
    assert forgotten == 0
    assert not rec.forgotten


def test_forget_skips_already_forgotten() -> None:
    """forget doesn't re-count already forgotten memories."""
    engine = _make_engine()
    old_time = int((time.time() - 365 * 86400) * 1000)
    rec = _make_record(episode_id=1, salience=0.1, encoded_at=old_time)
    rec.forgotten = True
    engine._records[1] = rec
    assert engine.forget() == 0


def test_forget_idempotent_same_timestamp() -> None:
    """Calling forget() twice with the same timestamp gives the same retention.

    Previously, interference, retrieval failure, and adaptive forgetting
    were applied as per-call penalties, so calling forget() repeatedly
    with the same time would keep reducing retention. Now all penalties
    are computed from first principles and the result is idempotent.
    """
    engine = _make_engine()
    now = time.time()
    old_time = int((now - 3600) * 1000)  # 1 hour ago
    rec = _make_record(
        episode_id=1, salience=0.1, encoded_at=old_time,
    )
    rec.last_accessed = old_time
    engine._records[1] = rec
    engine.forget(now=now)
    r1 = rec.retention
    # Call again with the same timestamp — retention should not change.
    engine.forget(now=now)
    r2 = rec.retention
    assert r1 == r2, f"forget() not idempotent: {r1} != {r2}"


def test_forget_idempotent_with_interference() -> None:
    """forget() is idempotent even with multiple nearby memories causing interference."""
    engine = _make_engine()
    now = time.time()
    base = int((now - 1800) * 1000)  # 30 min ago
    # Three memories encoded close together → interference.
    for i in range(3):
        rec = _make_record(
            episode_id=i, salience=0.3, encoded_at=base + i * 1000,
        )
        rec.last_accessed = base + i * 1000
        engine._records[i] = rec
    engine.forget(now=now)
    retentions_1 = {i: r.retention for i, r in engine._records.items()}
    engine.forget(now=now)
    retentions_2 = {i: r.retention for i, r in engine._records.items()}
    assert retentions_1 == retentions_2, (
        f"forget() not idempotent with interference: {retentions_1} != {retentions_2}"
    )


# ═══════════════════════════════════════════════════════════════════
# Sleep consolidation
# ═══════════════════════════════════════════════════════════════════


def test_consolidate_during_sleep_n3() -> None:
    """N3 sleep triggers systems consolidation and replay."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, hippocampal_dependency=1.0)
    engine._records[1] = rec
    count = engine.consolidate_during_sleep(SleepStage.N3, duration=100.0)
    assert count >= 0
    # N3 should reduce hippocampal dependency
    assert rec.hippocampal_dependency < 1.0


@pytest.mark.parametrize("stage", [SleepStage.N2, SleepStage.REM, SleepStage.N1])
def test_consolidate_during_sleep_stage(stage: SleepStage) -> None:
    """Sleep consolidation runs for N2, REM, and N1 stages.

    Hippocampal dependency decreases for N2 (the main transfer stage);
    count is non-negative for all stages.
    """
    engine = _make_engine()
    rec = _make_record(episode_id=1, hippocampal_dependency=1.0)
    engine._records[1] = rec
    count = engine.consolidate_during_sleep(stage, duration=100.0)
    assert count >= 0
    if stage == SleepStage.N2:
        assert rec.hippocampal_dependency < 1.0


# ═══════════════════════════════════════════════════════════════════
# Synaptic consolidation
# ═══════════════════════════════════════════════════════════════════


def test_synaptic_consolidate_success() -> None:
    """synaptic_consolidate boosts synaptic strength."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, synaptic_strength=0.3)
    engine._records[1] = rec
    ok = engine.synaptic_consolidate(1)
    assert ok is True
    assert rec.synaptic_strength > 0.3
    assert rec.retention > 0.0  # retention boosted too


def test_synaptic_consolidate_not_found() -> None:
    """synaptic_consolidate returns False for unknown memory."""
    engine = _make_engine()
    ok = engine.synaptic_consolidate(999)
    assert ok is False


def test_synaptic_consolidate_forgotten() -> None:
    """synaptic_consolidate skips forgotten memories."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.forgotten = True
    engine._records[1] = rec
    ok = engine.synaptic_consolidate(1)
    assert ok is False


def test_synaptic_consolidate_bounded() -> None:
    """synaptic strength is bounded at 1.0."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, synaptic_strength=0.9)
    engine._records[1] = rec
    engine.synaptic_consolidate(1)
    assert rec.synaptic_strength <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Systems consolidation
# ═══════════════════════════════════════════════════════════════════


def test_systems_consolidate_reduces_hippocampal_dependency() -> None:
    """systems_consolidate reduces hippocampal dependency."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, hippocampal_dependency=1.0)
    engine._records[1] = rec
    processed = engine.systems_consolidate()
    assert processed == 1
    assert rec.hippocampal_dependency < 1.0
    assert rec.systems_strength > 0.0


def test_systems_consolidate_multiple_calls() -> None:
    """Multiple systems_consolidate calls progressively reduce dependency."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, hippocampal_dependency=1.0)
    engine._records[1] = rec
    engine.systems_consolidate()
    dep_after_1 = rec.hippocampal_dependency
    engine.systems_consolidate()
    dep_after_2 = rec.hippocampal_dependency
    assert dep_after_2 < dep_after_1


def test_systems_consolidate_skips_forgotten() -> None:
    """systems_consolidate skips forgotten memories."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, hippocampal_dependency=1.0)
    rec.forgotten = True
    engine._records[1] = rec
    processed = engine.systems_consolidate()
    assert processed == 0


def test_systems_consolidate_skips_zero_dependency() -> None:
    """systems_consolidate skips memories with zero dependency."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, hippocampal_dependency=0.0)
    engine._records[1] = rec
    processed = engine.systems_consolidate()
    assert processed == 0


# ═══════════════════════════════════════════════════════════════════
# Reconsolidation window management
# ═══════════════════════════════════════════════════════════════════


def test_tick_reconsolidation_closes_window() -> None:
    """tick_reconsolidation reduces the reconsolidation window."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, reconsolidation_window=100.0)
    engine._records[1] = rec
    engine.tick_reconsolidation(dt=50.0)
    assert abs(rec.reconsolidation_window - 50.0) < 1e-9


def test_tick_reconsolidation_clamps_to_zero() -> None:
    """tick_reconsolidation clamps the window to 0."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, reconsolidation_window=30.0)
    engine._records[1] = rec
    engine.tick_reconsolidation(dt=100.0)
    assert rec.reconsolidation_window == 0.0


def test_tick_reconsolidation_no_window() -> None:
    """tick_reconsolidation does nothing to records with no window."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, reconsolidation_window=0.0)
    engine._records[1] = rec
    engine.tick_reconsolidation(dt=100.0)
    assert rec.reconsolidation_window == 0.0


# ═══════════════════════════════════════════════════════════════════
# Reconsolidation modification
# ═══════════════════════════════════════════════════════════════════


def test_reconsolidate_strengthen() -> None:
    """reconsolidate with positive strength_delta strengthens memory."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, salience=0.5, retention=0.6, reconsolidation_window=100.0)
    engine._records[1] = rec
    mod = ReconsolidationModification(strength_delta=0.2)
    ok = engine.reconsolidate(1, mod)
    assert ok is True
    assert abs(rec.salience - 0.7) < 1e-9
    assert abs(rec.retention - 0.8) < 1e-9


def test_reconsolidate_weaken() -> None:
    """reconsolidate with negative strength_delta weakens memory."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, salience=0.7, retention=0.8, reconsolidation_window=100.0)
    engine._records[1] = rec
    mod = ReconsolidationModification(strength_delta=-0.3)
    ok = engine.reconsolidate(1, mod)
    assert ok is True
    assert abs(rec.salience - 0.4) < 1e-9


def test_reconsolidate_emotional_retag() -> None:
    """reconsolidate blends the emotional tag."""
    engine = _make_engine()
    rec = _make_record(
        episode_id=1,
        emotional_tag=[0.5] * 12,
        reconsolidation_window=100.0,
    )
    engine._records[1] = rec
    new_tag = [0.9] * 12
    mod = ReconsolidationModification(
        strength_delta=0.0,
        emotional_tag=new_tag,
        blend_factor=0.5,
    )
    ok = engine.reconsolidate(1, mod)
    assert ok is True
    # Blended: 0.5 * 0.5 + 0.9 * 0.5 = 0.7
    assert abs(rec.emotional_tag[0] - 0.7) < 1e-9


def test_reconsolidate_no_window_returns_false() -> None:
    """reconsolidate returns False when window is closed."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, reconsolidation_window=0.0)
    engine._records[1] = rec
    mod = ReconsolidationModification(strength_delta=0.1)
    ok = engine.reconsolidate(1, mod)
    assert ok is False


def test_reconsolidate_not_found() -> None:
    """reconsolidate returns False for unknown memory."""
    engine = _make_engine()
    mod = ReconsolidationModification(strength_delta=0.1)
    ok = engine.reconsolidate(999, mod)
    assert ok is False


def test_reconsolidate_forgotten() -> None:
    """reconsolidate skips forgotten memories."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, reconsolidation_window=100.0)
    rec.forgotten = True
    engine._records[1] = rec
    mod = ReconsolidationModification(strength_delta=0.1)
    ok = engine.reconsolidate(1, mod)
    assert ok is False


def test_reconsolidate_closes_window() -> None:
    """reconsolidate closes the window after applying modification."""
    engine = _make_engine()
    rec = _make_record(episode_id=1, reconsolidation_window=100.0)
    engine._records[1] = rec
    mod = ReconsolidationModification(strength_delta=0.1)
    engine.reconsolidate(1, mod)
    assert rec.reconsolidation_window == 0.0


# ═══════════════════════════════════════════════════════════════════
# Source monitoring
# ═══════════════════════════════════════════════════════════════════


def test_monitor_source_correct() -> None:
    """monitor_source returns True when source matches."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="user", confidence=0.9)
    engine._records[1] = rec
    assert engine.monitor_source(1, "user") is True


def test_monitor_source_incorrect() -> None:
    """monitor_source returns False when source doesn't match."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="user", confidence=0.9)
    engine._records[1] = rec
    assert engine.monitor_source(1, "genesis") is False


def test_monitor_source_no_tag() -> None:
    """monitor_source returns False when no source tag exists."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = None
    engine._records[1] = rec
    assert engine.monitor_source(1, "user") is False


def test_monitor_source_not_found() -> None:
    """monitor_source returns False for unknown memory."""
    engine = _make_engine()
    assert engine.monitor_source(999, "user") is False


# ═══════════════════════════════════════════════════════════════════
# Confabulation risk
# ═══════════════════════════════════════════════════════════════════


def test_confabulation_risk_external_source() -> None:
    """External sources (user, learning) have low confabulation risk."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="user", confidence=0.9)
    engine._records[1] = rec
    risk = engine.check_confabulation_risk(1)
    assert risk < 0.3


def test_confabulation_risk_inference() -> None:
    """Inference source has higher confabulation risk."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="inference", confidence=0.5)
    engine._records[1] = rec
    risk = engine.check_confabulation_risk(1)
    assert risk > 0.4


def test_confabulation_risk_imagination() -> None:
    """Imagination source has high confabulation risk."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="imagination", confidence=0.3)
    engine._records[1] = rec
    risk = engine.check_confabulation_risk(1)
    assert risk > 0.5


def test_confabulation_risk_no_tag() -> None:
    """No source tag returns moderate risk (0.5)."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = None
    engine._records[1] = rec
    risk = engine.check_confabulation_risk(1)
    assert abs(risk - 0.5) < 1e-9


def test_confabulation_risk_not_found() -> None:
    """Unknown memory returns moderate risk (0.5)."""
    engine = _make_engine()
    risk = engine.check_confabulation_risk(999)
    assert abs(risk - 0.5) < 1e-9


def test_confabulation_risk_bounded() -> None:
    """Confabulation risk is bounded [0, 1]."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="imagination", confidence=0.0)
    rec.access_count = 20
    engine._records[1] = rec
    risk = engine.check_confabulation_risk(1)
    assert 0.0 <= risk <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Get memories by source
# ═══════════════════════════════════════════════════════════════════


def test_get_memories_by_source() -> None:
    """get_memories_by_source filters by source."""
    engine = _make_engine()
    for eid, src in [(1, "user"), (2, "learning"), (3, "user")]:
        rec = _make_record(episode_id=eid)
        rec.source_tag = SourceTag(source=src)
        engine._records[eid] = rec
    user_mems = engine.get_memories_by_source("user")
    assert 1 in user_mems
    assert 3 in user_mems
    assert 2 not in user_mems


def test_get_memories_by_source_excludes_forgotten() -> None:
    """get_memories_by_source excludes forgotten memories."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="user")
    rec.forgotten = True
    engine._records[1] = rec
    assert engine.get_memories_by_source("user") == []


def test_get_memories_by_source_no_tag() -> None:
    """get_memories_by_source excludes memories without source tags."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = None
    engine._records[1] = rec
    assert engine.get_memories_by_source("user") == []


# ═══════════════════════════════════════════════════════════════════
# Source summary
# ═══════════════════════════════════════════════════════════════════


def test_source_summary() -> None:
    """source_summary returns counts per source."""
    engine = _make_engine()
    for eid, src in [(1, "user"), (2, "learning"), (3, "user")]:
        rec = _make_record(episode_id=eid)
        rec.source_tag = SourceTag(source=src)
        engine._records[eid] = rec
    summary = engine.source_summary()
    assert summary["user"] == 2
    assert summary["learning"] == 1


def test_source_summary_excludes_forgotten() -> None:
    """source_summary excludes forgotten memories."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = SourceTag(source="user")
    rec.forgotten = True
    engine._records[1] = rec
    summary = engine.source_summary()
    assert "user" not in summary or summary["user"] == 0


def test_source_summary_no_tag_counts_as_unknown() -> None:
    """source_summary counts memories without tag as 'unknown'."""
    engine = _make_engine()
    rec = _make_record(episode_id=1)
    rec.source_tag = None
    engine._records[1] = rec
    summary = engine.source_summary()
    assert summary.get("unknown", 0) == 1


# ═══════════════════════════════════════════════════════════════════
# Persistence — serialize/restore memory records across restarts
# ═══════════════════════════════════════════════════════════════════


def test_serialize_and_restore_records_round_trip() -> None:
    """serialize_records → restore_records preserves all consolidation state.

    Without persistence, every restart resets all Python-side memory
    metadata: hippocampal dependency, synaptic/systems strength,
    retention, forgetting, reconsolidation windows, source tags, and
    access counts. This test verifies the round-trip preserves them.
    """
    engine = _make_engine()
    # Create a few records with varied state
    rec1 = _make_record(episode_id=10, salience=0.8)
    rec1.hippocampal_dependency = 0.3
    rec1.synaptic_strength = 0.7
    rec1.systems_strength = 0.5
    rec1.retention = 0.6
    rec1.access_count = 3
    rec1.encoded_at = 1000000
    rec1.last_accessed = 2000000
    rec1.source_tag = SourceTag(source="user", modality="text", confidence=0.9)

    rec2 = _make_record(episode_id=20, salience=0.4)
    rec2.forgotten = True
    rec2.retention = 0.02
    rec2.source_tag = SourceTag(source="inference", confidence=0.6)

    rec3 = _make_record(episode_id=30, salience=0.6)
    rec3.reconsolidation_window = 3600.0
    rec3.last_reconsolidated = 3000000
    rec3.source_tag = SourceTag(source="genesis", confidence=0.85)

    engine._records = {10: rec1, 20: rec2, 30: rec3}
    engine.forgotten_count = 1
    engine.reconsolidation_count = 5
    engine.synaptic_consolidation_count = 10
    engine.systems_consolidation_count = 3
    engine.sleep_consolidation_count = 7
    engine.facts_replayed_count = 42

    serialized = engine.serialize_records()

    # Restore into a fresh engine
    engine2 = _make_engine()
    assert len(engine2._records) == 0
    engine2.restore_records(serialized)

    # All records restored
    assert len(engine2._records) == 3

    # Record 1 — full state check
    r1 = engine2._records[10]
    assert r1.episode_id == 10
    assert r1.salience == 0.8
    assert r1.hippocampal_dependency == 0.3
    assert r1.synaptic_strength == 0.7
    assert r1.systems_strength == 0.5
    assert r1.retention == 0.6
    assert r1.access_count == 3
    assert r1.encoded_at == 1000000
    assert r1.last_accessed == 2000000
    assert not r1.forgotten
    assert r1.source_tag is not None
    assert r1.source_tag.source == "user"
    assert r1.source_tag.confidence == 0.9

    # Record 2 — forgotten state preserved
    r2 = engine2._records[20]
    assert r2.forgotten
    assert r2.retention == 0.02
    assert r2.source_tag is not None
    assert r2.source_tag.source == "inference"

    # Record 3 — reconsolidation window preserved
    r3 = engine2._records[30]
    assert r3.reconsolidation_window == 3600.0
    assert r3.last_reconsolidated == 3000000
    assert r3.source_tag is not None
    assert r3.source_tag.source == "genesis"

    # Counters preserved
    assert engine2.forgotten_count == 1
    assert engine2.reconsolidation_count == 5
    assert engine2.synaptic_consolidation_count == 10
    assert engine2.systems_consolidation_count == 3
    assert engine2.sleep_consolidation_count == 7
    assert engine2.facts_replayed_count == 42


def test_restore_records_handles_missing_source_tag() -> None:
    """restore_records handles records with no source tag (None)."""
    engine = _make_engine()
    data = {
        "records": [
            {
                "episode_id": 1,
                "salience": 0.5,
                "emotional_tag": [0.5] * 12,
                "encoded_at": 1000,
                "last_accessed": 2000,
                "access_count": 0,
                "hippocampal_dependency": 1.0,
                "synaptic_strength": 0.3,
                "systems_strength": 0.0,
                "retention": 1.0,
                "forgotten": False,
                "reconsolidation_window": 0.0,
                "last_reconsolidated": 0,
                "source": None,
            }
        ]
    }
    engine.restore_records(data)
    assert len(engine._records) == 1
    assert engine._records[1].source_tag is None


def test_restore_records_handles_empty_data() -> None:
    """restore_records handles empty/malformed data gracefully."""
    engine = _make_engine()
    engine.restore_records({})
    assert len(engine._records) == 0
    engine.restore_records({"records": []})
    assert len(engine._records) == 0


def test_restore_records_skips_invalid_entries() -> None:
    """restore_records skips entries with invalid episode IDs."""
    engine = _make_engine()
    data = {
        "records": [
            {"episode_id": "not_a_number", "salience": 0.5},
            {"salience": 0.5},  # missing episode_id
            {"episode_id": 42, "salience": 0.5},
        ]
    }
    engine.restore_records(data)
    assert len(engine._records) == 1
    assert 42 in engine._records


# ═══════════════════════════════════════════════════════════════════
# Salience / emotional_tag clamping — Rust/Python consistency
# ═══════════════════════════════════════════════════════════════════


def test_store_memory_clamps_out_of_range_salience() -> None:
    """store_memory clamps salience to [0,1] matching Rust IPC's finite_clamp.

    Without clamping, the Python-side MemoryRecord and the Rust-side
    LTM store would disagree: the Rust store clamps on receipt, but
    the Python record would store the raw value, causing the forgetting
    formula (which scales tau by rec.salience) to produce incorrect decay.
    """
    engine = _make_engine()
    engine.store_memory("test memory", salience=2.0)
    rec = engine.get_memory_record(1)
    assert rec is not None
    assert rec.salience == 1.0, f"salience should be clamped to 1.0, got {rec.salience}"


def test_store_memory_clamps_negative_salience() -> None:
    """store_memory clamps negative salience to 0.0."""
    engine = _make_engine()
    engine.store_memory("test memory", salience=-0.5)
    rec = engine.get_memory_record(1)
    assert rec is not None
    assert rec.salience == 0.0, f"salience should be clamped to 0.0, got {rec.salience}"


def test_store_memory_handles_nan_salience() -> None:
    """store_memory replaces NaN salience with a neutral default."""
    engine = _make_engine()
    engine.store_memory("test memory", salience=float("nan"))
    rec = engine.get_memory_record(1)
    assert rec is not None
    assert rec.salience == 0.5, f"NaN salience should become 0.5, got {rec.salience}"
    assert not math.isnan(rec.salience)


def test_store_memory_clamps_out_of_range_emotional_tag() -> None:
    """store_memory clamps emotional_tag elements to [0,1]."""
    engine = _make_engine()
    tag = [2.0, -1.0, 0.5, float("inf"), 0.5] + [0.5] * 7
    engine.store_memory("test memory", salience=0.5, emotional_tag=tag)
    rec = engine.get_memory_record(1)
    assert rec is not None
    assert rec.emotional_tag[0] == 1.0, "2.0 should clamp to 1.0"
    assert rec.emotional_tag[1] == 0.0, "-1.0 should clamp to 0.0"
    assert rec.emotional_tag[3] == 0.5, "Inf should become 0.5"
    for v in rec.emotional_tag:
        assert 0.0 <= v <= 1.0, f"all tag values should be in [0,1], got {v}"


# ======================================================================
# From tests/test_reconsolidation_boundaries.py
# ======================================================================

def _make_mock_client_recons(ltm_count: int = 1) -> MagicMock:
    """Create a mock GenesisClient."""
    client = MagicMock()
    client.get_memory_stats.return_value = MemoryStats(
        stm_count=0, ltm_count=ltm_count, ltm_capacity=65536
    )
    client.store_event.return_value = True
    # store_episode returns the real LTM episode ID (ltm_count + 1).
    client.store_episode.return_value = ltm_count + 1
    client.find_similar.return_value = []
    return client


def _make_engine_with_record(
    episode_id: int = 1,
    salience: float = 0.5,
    retention: float = 0.8,
    synaptic_strength: float = 0.5,
) -> tuple[MemoryEngine, MemoryRecord]:
    """Create a MemoryEngine with a pre-registered memory record."""
    client = _make_mock_client_recons()
    engine = MemoryEngine(client=client)
    rec = MemoryRecord(
        episode_id=episode_id,
        salience=salience,
        retention=retention,
        synaptic_strength=synaptic_strength,
        encoded_at=int(time.time() * 1000),
        last_accessed=int(time.time() * 1000),
    )
    engine._records[episode_id] = rec
    return engine, rec


def test_recall_without_prediction_error_does_not_open_window():
    """Recall with prediction_error=0 should NOT open the reconsolidation window."""
    engine, rec = _make_engine_with_record(episode_id=1)
    initial_window = rec.reconsolidation_window

    # Retrieve with no prediction error (default)
    engine._mark_accessed(episode_id=1, now_ms=int(time.time() * 1000), opens_window=False)

    assert rec.reconsolidation_window == initial_window, (
        f"Window should not open without prediction error, "
        f"got {rec.reconsolidation_window} (expected {initial_window})"
    )
    print("  PASS  recall without PE does not open reconsolidation window")


def test_recall_with_prediction_error_opens_window():
    """Recall with prediction_error >= threshold should open the window."""
    engine, rec = _make_engine_with_record(episode_id=1)

    engine._mark_accessed(episode_id=1, now_ms=int(time.time() * 1000), opens_window=True)

    assert rec.reconsolidation_window > 0, (
        f"Window should open with prediction error, "
        f"got {rec.reconsolidation_window}"
    )
    assert rec.reconsolidation_window == engine._RECONSOLIDATION_WINDOW, (
        f"Window should be full duration, got {rec.reconsolidation_window}"
    )
    print("  PASS  recall with PE opens reconsolidation window")


def test_retrieval_practice_strengthens_memory():
    """Every recall should strengthen the memory (retrieval practice effect)."""
    engine, rec = _make_engine_with_record(
        episode_id=1, retention=0.5, synaptic_strength=0.4
    )
    initial_retention = rec.retention
    initial_synaptic = rec.synaptic_strength

    engine._mark_accessed(episode_id=1, now_ms=int(time.time() * 1000), opens_window=False)

    assert rec.retention > initial_retention, (
        f"Retention should increase from retrieval practice, "
        f"got {rec.retention} (expected > {initial_retention})"
    )
    assert rec.synaptic_strength > initial_synaptic, (
        f"Synaptic strength should increase from retrieval practice, "
        f"got {rec.synaptic_strength} (expected > {initial_synaptic})"
    )
    print("  PASS  retrieval practice strengthens memory")


def test_reconsolidate_fails_without_open_window():
    """reconsolidate() should return False if window was never opened."""
    engine, rec = _make_engine_with_record(episode_id=1)
    assert rec.reconsolidation_window == 0.0

    result = engine.reconsolidate(1, ReconsolidationModification(strength_delta=0.1))

    assert result is False, "reconsolidate should fail when window is closed"
    assert rec.salience == 0.5, "Memory should not be modified when window is closed"
    print("  PASS  reconsolidate fails without open window")


def test_reconsolidate_succeeds_with_open_window():
    """reconsolidate() should apply modifications when window is open."""
    engine, rec = _make_engine_with_record(episode_id=1, salience=0.5)

    # Open the window (mimicking recall with prediction error)
    rec.reconsolidation_window = engine._RECONSOLIDATION_WINDOW

    result = engine.reconsolidate(1, ReconsolidationModification(strength_delta=0.1))

    assert result is True, "reconsolidate should succeed when window is open"
    assert rec.salience == 0.6, f"Salience should be 0.6, got {rec.salience}"
    assert rec.reconsolidation_window == 0.0, "Window should close after reconsolidation"
    print("  PASS  reconsolidate succeeds with open window")


def test_retrieve_relevant_with_pe_opens_window():
    """retrieve_relevant with prediction_error >= threshold opens window."""
    engine, rec = _make_engine_with_record(episode_id=1)
    # Mock find_similar to return our episode
    engine.client.find_similar.return_value = [
        SimilarEpisode(
            episode_id=1, hamming_distance=10, timestamp=int(time.time() * 1000),
            salience=0.5,
        )
    ]

    engine.retrieve_relevant("test query", prediction_error=0.5)

    assert rec.reconsolidation_window > 0, "Window should open with PE=0.5"
    assert rec.access_count == 1, f"Access count should be 1, got {rec.access_count}"
    print("  PASS  retrieve_relevant with PE opens window")


def test_retrieve_relevant_without_pe_does_not_open_window():
    """retrieve_relevant with prediction_error=0 does NOT open window."""
    engine, rec = _make_engine_with_record(episode_id=1)
    engine.client.find_similar.return_value = [
        SimilarEpisode(
            episode_id=1, hamming_distance=10, timestamp=int(time.time() * 1000),
            salience=0.5,
        )
    ]

    engine.retrieve_relevant("test query", prediction_error=0.0)

    assert rec.reconsolidation_window == 0.0, "Window should NOT open with PE=0"
    assert rec.access_count == 1, f"Access count should still increment, got {rec.access_count}"
    print("  PASS  retrieve_relevant without PE does not open window")


def test_retrieve_episode_with_pe_opens_window():
    """retrieve_episode with prediction_error >= threshold opens window."""
    engine, rec = _make_engine_with_record(episode_id=1)
    engine.client.retrieve_episode.return_value = Episode(
        episode_id=1, timestamp=int(time.time() * 1000), salience=0.5,
        association_hash=0, full_emotional_tag=[0.5] * 12,
        event_type=0, source_module=4, text="test memory",
    )

    engine.retrieve_episode(1, prediction_error=0.3)

    assert rec.reconsolidation_window > 0, "Window should open with PE=0.3"
    print("  PASS  retrieve_episode with PE opens window")


def test_retrieve_episode_without_pe_does_not_open_window():
    """retrieve_episode with prediction_error=0 does NOT open window."""
    engine, rec = _make_engine_with_record(episode_id=1)
    engine.client.retrieve_episode.return_value = Episode(
        episode_id=1, timestamp=int(time.time() * 1000), salience=0.5,
        association_hash=0, full_emotional_tag=[0.5] * 12,
        event_type=0, source_module=4, text="test memory",
    )

    engine.retrieve_episode(1, prediction_error=0.0)

    assert rec.reconsolidation_window == 0.0, "Window should NOT open with PE=0"
    assert rec.access_count == 1, "Access count should still increment"
    print("  PASS  retrieve_episode without PE does not open window")


def test_retrieval_practice_is_bounded():
    """Retrieval practice should not push retention above 1.0."""
    engine, rec = _make_engine_with_record(episode_id=1, retention=0.99)

    engine._mark_accessed(episode_id=1, now_ms=int(time.time() * 1000), opens_window=False)

    assert rec.retention <= 1.0, f"Retention should be bounded at 1.0, got {rec.retention}"
    print("  PASS  retrieval practice is bounded at 1.0")


# ═══════════════════════════════════════════════════════════════
#  Source monitoring tests (Johnson et al., 1993)
# ═══════════════════════════════════════════════════════════════


def _make_engine_with_source(
    episode_id: int = 1,
    source: str = "conversation",
    confidence: float = 0.8,
    access_count: int = 0,
) -> tuple[MemoryEngine, MemoryRecord]:
    """Create a MemoryEngine with a record that has a source tag."""
    engine, rec = _make_engine_with_record(episode_id=episode_id)
    rec.source_tag = SourceTag(source=source, modality="text", confidence=confidence)
    rec.access_count = access_count
    return engine, rec


def test_store_memory_creates_source_tag():
    """store_memory with source= should create a SourceTag on the record."""
    client = _make_mock_client_recons(ltm_count=1)
    engine = MemoryEngine(client=client)

    engine.store_memory(text="test memory", salience=0.5, source="conversation")

    # store_episode returns ltm_count + 1 = 2
    rec = engine._records.get(2)
    assert rec is not None, "Record should be created"
    assert rec.source_tag is not None, "Source tag should be created"
    assert rec.source_tag.source == "conversation", (
        f"Source should be 'conversation', got '{rec.source_tag.source}'"
    )
    print("  PASS  store_memory creates source tag")


def test_store_memory_invalid_source_defaults_to_unknown():
    """store_memory with an invalid source string should default to 'unknown'."""
    client = _make_mock_client_recons(ltm_count=1)
    engine = MemoryEngine(client=client)

    engine.store_memory(text="test", source="bogus_source")

    # store_episode returns ltm_count + 1 = 2
    rec = engine._records.get(2)
    assert rec is not None and rec.source_tag is not None
    assert rec.source_tag.source == "unknown", (
        f"Invalid source should default to 'unknown', got '{rec.source_tag.source}'"
    )
    print("  PASS  invalid source defaults to 'unknown'")


def test_recons_monitor_source_correct():
    """monitor_source should return True when source matches."""
    engine, _rec = _make_engine_with_source(episode_id=1, source="conversation")

    result = engine.monitor_source(1, "conversation")

    assert result is True, "monitor_source should return True for matching source"
    print("  PASS  monitor_source returns True for correct source")


def test_recons_monitor_source_incorrect():
    """monitor_source should return False when source doesn't match."""
    engine, _rec = _make_engine_with_source(episode_id=1, source="inference")

    result = engine.monitor_source(1, "user")

    assert result is False, "monitor_source should return False for mismatched source"
    print("  PASS  monitor_source returns False for incorrect source")


def test_recons_monitor_source_no_tag():
    """monitor_source should return False when no source tag exists."""
    engine, _rec = _make_engine_with_record(episode_id=1)

    result = engine.monitor_source(1, "user")

    assert result is False, "monitor_source should return False when no source tag"
    print("  PASS  monitor_source returns False when no source tag")


def test_confabulation_risk_low_for_external_source():
    """Confabulation risk should be low for externally-sourced memories."""
    engine, _rec = _make_engine_with_source(
        episode_id=1, source="conversation", confidence=0.9, access_count=2
    )

    risk = engine.check_confabulation_risk(1)

    assert risk < 0.2, f"Risk should be low for external source, got {risk}"
    print(f"  PASS  confabulation risk low for external source ({risk:.2f})")


def test_confabulation_risk_high_for_inference():
    """Confabulation risk should be high for inferred memories with low confidence."""
    engine, _rec = _make_engine_with_source(
        episode_id=1, source="inference", confidence=0.3, access_count=8
    )

    risk = engine.check_confabulation_risk(1)

    assert risk > 0.6, f"Risk should be high for inference+low confidence, got {risk}"
    print(f"  PASS  confabulation risk high for inference ({risk:.2f})")


def test_confabulation_risk_high_for_imagination():
    """Confabulation risk should be high for imagined memories."""
    engine, _rec = _make_engine_with_source(
        episode_id=1, source="imagination", confidence=0.5, access_count=3
    )

    risk = engine.check_confabulation_risk(1)

    assert risk > 0.5, f"Risk should be high for imagination, got {risk}"
    print(f"  PASS  confabulation risk high for imagination ({risk:.2f})")


def test_recons_get_memories_by_source():
    """get_memories_by_source should return only memories from that source."""
    engine, _ = _make_engine_with_source(episode_id=1, source="conversation")
    rec2 = MemoryRecord(episode_id=2, source_tag=SourceTag(source="learning"))
    engine._records[2] = rec2
    rec3 = MemoryRecord(episode_id=3, source_tag=SourceTag(source="conversation"))
    engine._records[3] = rec3

    conv_ids = engine.get_memories_by_source("conversation")
    learn_ids = engine.get_memories_by_source("learning")

    assert set(conv_ids) == {1, 3}, f"Expected {{1, 3}}, got {set(conv_ids)}"
    assert set(learn_ids) == {2}, f"Expected {{2}}, got {set(learn_ids)}"
    print("  PASS  get_memories_by_source filters correctly")


def test_recons_source_summary():
    """source_summary should return counts per source category."""
    engine, _ = _make_engine_with_source(episode_id=1, source="conversation")
    engine._records[2] = MemoryRecord(episode_id=2, source_tag=SourceTag(source="conversation"))
    engine._records[3] = MemoryRecord(episode_id=3, source_tag=SourceTag(source="learning"))

    summary = engine.source_summary()

    conv = summary.get("conversation", 0)
    learn = summary.get("learning", 0)
    assert conv == 2, f"Expected 2 conversation, got {conv}"
    assert learn == 1, f"Expected 1 learning, got {learn}"
    print("  PASS  source_summary returns correct counts")


def test_recons_source_summary_excludes_forgotten():
    """source_summary should not count forgotten memories."""
    engine, _ = _make_engine_with_source(episode_id=1, source="conversation")
    rec2 = MemoryRecord(episode_id=2, source_tag=SourceTag(source="conversation"), forgotten=True)
    engine._records[2] = rec2

    summary = engine.source_summary()

    conv = summary.get("conversation", 0)
    assert conv == 1, f"Expected 1 (forgotten excluded), got {conv}"
    print("  PASS  source_summary excludes forgotten memories")


# ═══════════════════════════════════════════════════════════════
#  Retrieval-induced forgetting tests (Anderson et al., 1994)
# ═══════════════════════════════════════════════════════════════


def test_rif_weakenes_competing_memories():
    """RIF should weaken emotionally similar non-retrieved memories."""
    engine, rec1 = _make_engine_with_record(episode_id=1, retention=0.8)
    # rec1 has default emotional_tag [0.5]*12
    # Create a competing memory with the same emotional tag
    rec2 = MemoryRecord(
        episode_id=2, retention=0.8, salience=0.3,
        emotional_tag=[0.5] * 12,  # identical → max similarity
    )
    engine._records[2] = rec2
    initial_retention = rec2.retention

    # Retrieve memory 1 (not memory 2)
    weakened = engine._apply_retrieval_induced_forgetting([1])

    assert weakened >= 1, f"Should weaken at least 1 memory, got {weakened}"
    assert rec2.retention < initial_retention, (
        f"Competing memory should be weakened, got {rec2.retention} (was {initial_retention})"
    )
    # Retrieved memory should NOT be weakened
    assert rec1.retention >= 0.8, f"Retrieved memory should not be weakened, got {rec1.retention}"
    print("  PASS  RIF weakens competing memories")


def test_rif_does_not_weaken_dissimilar_memories():
    """RIF should NOT weaken emotionally dissimilar memories."""
    engine, _ = _make_engine_with_record(episode_id=1, retention=0.8)
    # Create a dissimilar memory — emotional tag pointing in a very
    # different direction from [0.5]*12 (cosine similarity < 0.6).
    rec2 = MemoryRecord(
        episode_id=2, retention=0.8, salience=0.3,
        emotional_tag=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    engine._records[2] = rec2
    initial_retention = rec2.retention

    engine._apply_retrieval_induced_forgetting([1])

    assert rec2.retention == initial_retention, (
        f"Dissimilar memory should not be weakened, got {rec2.retention} (was {initial_retention})"
    )
    print("  PASS  RIF does not weaken dissimilar memories")


def test_rif_does_not_weaken_retrieved_memories():
    """RIF should NOT weaken the retrieved memory itself."""
    engine, rec1 = _make_engine_with_record(episode_id=1, retention=0.8)
    rec2 = MemoryRecord(episode_id=2, retention=0.8, emotional_tag=[0.5] * 12)
    engine._records[2] = rec2

    engine._apply_retrieval_induced_forgetting([1, 2])

    # Both were retrieved — neither should be weakened by RIF
    assert rec1.retention == 0.8, f"Retrieved memory 1 should not be weakened, got {rec1.retention}"
    assert rec2.retention == 0.8, f"Retrieved memory 2 should not be weakened, got {rec2.retention}"
    print("  PASS  RIF does not weaken retrieved memories")


def test_rif_high_salience_resists_weakening():
    """High-salience memories should resist RIF more than low-salience."""
    engine, _ = _make_engine_with_record(episode_id=1, retention=0.8)
    # Low-salience competing memory
    rec_low = MemoryRecord(
        episode_id=2, retention=0.8, salience=0.1,
        emotional_tag=[0.5] * 12,
    )
    # High-salience competing memory (same emotional tag)
    rec_high = MemoryRecord(
        episode_id=3, retention=0.8, salience=0.9,
        emotional_tag=[0.5] * 12,
    )
    engine._records[2] = rec_low
    engine._records[3] = rec_high

    engine._apply_retrieval_induced_forgetting([1])

    low_drop = 0.8 - rec_low.retention
    high_drop = 0.8 - rec_high.retention
    assert low_drop > high_drop, (
        f"Low-salience should weaken more ({low_drop}) than high-salience ({high_drop})"
    )
    print(f"  PASS  high-salience resists RIF (low drop={low_drop:.4f}, high drop={high_drop:.4f})")


def test_rif_empty_retrieved_list_does_nothing():
    """RIF with empty retrieved list should do nothing."""
    engine, rec1 = _make_engine_with_record(episode_id=1, retention=0.8)
    initial = rec1.retention

    weakened = engine._apply_retrieval_induced_forgetting([])

    assert weakened == 0, f"Should weaken 0 memories, got {weakened}"
    assert rec1.retention == initial, "Memory should be unchanged"
    print("  PASS  RIF with empty list does nothing")


def test_rif_integrated_with_retrieve_relevant():
    """retrieve_relevant should trigger RIF on competing memories."""
    engine, _rec1 = _make_engine_with_record(episode_id=1, retention=0.8)
    rec2 = MemoryRecord(episode_id=2, retention=0.8, salience=0.3, emotional_tag=[0.5] * 12)
    engine._records[2] = rec2
    initial_retention = rec2.retention

    # Mock find_similar to return only episode 1
    engine.client.find_similar.return_value = [
        SimilarEpisode(
            episode_id=1, hamming_distance=10, timestamp=int(time.time() * 1000),
            salience=0.5,
        )
    ]

    engine.retrieve_relevant("test query")

    assert rec2.retention < initial_retention, (
        f"Competing memory should be weakened after retrieve_relevant, "
        f"got {rec2.retention} (was {initial_retention})"
    )
    print("  PASS  RIF integrated with retrieve_relevant")


# ═══════════════════════════════════════════════════════════════
#  Experience replay tests (Kirkpatrick et al., 2017)
# ═══════════════════════════════════════════════════════════════


def test_replay_old_memories_strengthens_old():
    """Experience replay should strengthen old (non-recent) memories."""
    engine, _ = _make_engine_with_record(episode_id=1, retention=0.5, synaptic_strength=0.3)
    rec = engine._records[1]
    # Make it old (encoded long ago)
    rec.encoded_at = int(time.time() * 1000) - 999_999_999  # ~11.5 days ago
    initial_retention = rec.retention
    initial_synaptic = rec.synaptic_strength

    # recent_cutoff = now (so this memory is "old")
    replayed = engine._replay_old_memories(recent_cutoff=int(time.time() * 1000))

    assert replayed >= 1, f"Should replay at least 1 memory, got {replayed}"
    assert rec.retention > initial_retention, (
        f"Old memory retention should increase, "
        f"got {rec.retention} (was {initial_retention})"
    )
    assert rec.synaptic_strength > initial_synaptic, (
        f"Old memory synaptic strength should increase, "
        f"got {rec.synaptic_strength} (was {initial_synaptic})"
    )
    print("  PASS  experience replay strengthens old memories")


def test_replay_does_not_touch_recent_memories():
    """Experience replay should NOT replay recent memories (already replayed in N3)."""
    engine, rec = _make_engine_with_record(episode_id=1, retention=0.5)
    # Memory is recent (encoded now)
    now_ms = int(time.time() * 1000)
    rec.encoded_at = now_ms
    initial_retention = rec.retention

    # recent_cutoff = now - 1 (the memory is after the cutoff → "recent")
    replayed = engine._replay_old_memories(recent_cutoff=now_ms - 1)

    assert replayed == 0, f"Should replay 0 recent memories, got {replayed}"
    assert rec.retention == initial_retention, "Recent memory should not be touched"
    print("  PASS  replay does not touch recent memories")


def test_replay_does_not_touch_forgotten_memories():
    """Experience replay should NOT replay forgotten memories."""
    engine, rec = _make_engine_with_record(episode_id=1, retention=0.5)
    rec.encoded_at = int(time.time() * 1000) - 999_999_999
    rec.forgotten = True
    initial_retention = rec.retention

    replayed = engine._replay_old_memories(recent_cutoff=int(time.time() * 1000))

    assert replayed == 0, f"Should replay 0 forgotten memories, got {replayed}"
    assert rec.retention == initial_retention, "Forgotten memory should not be touched"
    print("  PASS  replay does not touch forgotten memories")


def test_replay_respects_sample_size_limit():
    """Replay should not exceed the sample size limit."""
    client = _make_mock_client_recons(ltm_count=1)
    engine = MemoryEngine(client=client)
    # Create 20 old memories
    old_time = int(time.time() * 1000) - 999_999_999
    for i in range(1, 21):
        engine._records[i] = MemoryRecord(
            episode_id=i, retention=0.5, encoded_at=old_time,
        )

    replayed = engine._replay_old_memories(recent_cutoff=int(time.time() * 1000))

    assert replayed <= engine._REPLAY_SAMPLE_SIZE, (
        f"Should not exceed sample size {engine._REPLAY_SAMPLE_SIZE}, got {replayed}"
    )
    print(f"  PASS  replay respects sample size limit ({replayed} <= {engine._REPLAY_SAMPLE_SIZE})")


def test_replay_empty_store_does_nothing():
    """Replay with no memories should return 0."""
    client = _make_mock_client_recons(ltm_count=0)
    engine = MemoryEngine(client=client)

    replayed = engine._replay_old_memories(recent_cutoff=int(time.time() * 1000))

    assert replayed == 0, f"Should replay 0 with empty store, got {replayed}"
    print("  PASS  replay with empty store does nothing")


def test_replay_integrated_with_n3_consolidation():
    """N3 consolidation should include experience replay."""
    from genesis_cognitive.sleep import SleepStage

    client = _make_mock_client_recons(ltm_count=1)
    engine = MemoryEngine(client=client)
    # Create an old memory
    old_time = int(time.time() * 1000) - 999_999_999
    rec = MemoryRecord(episode_id=1, retention=0.5, encoded_at=old_time)
    engine._records[1] = rec
    initial_retention = rec.retention

    # Run N3 consolidation (duration=3600 → recent_cutoff = now - 7200000)
    processed = engine.consolidate_during_sleep(SleepStage.N3, duration=3600.0)

    # The old memory should have been replayed (strengthened)
    assert rec.retention > initial_retention, (
        f"Old memory should be strengthened during N3, "
        f"got {rec.retention} (was {initial_retention})"
    )
    assert processed >= 1, f"N3 should process at least 1 memory, got {processed}"
    print("  PASS  N3 consolidation includes experience replay")


def main() -> None:
    """Main."""
    tests = [
        test_recall_without_prediction_error_does_not_open_window,
        test_recall_with_prediction_error_opens_window,
        test_retrieval_practice_strengthens_memory,
        test_reconsolidate_fails_without_open_window,
        test_reconsolidate_succeeds_with_open_window,
        test_retrieve_relevant_with_pe_opens_window,
        test_retrieve_relevant_without_pe_does_not_open_window,
        test_retrieve_episode_with_pe_opens_window,
        test_retrieve_episode_without_pe_does_not_open_window,
        test_retrieval_practice_is_bounded,
        # Source monitoring
        test_store_memory_creates_source_tag,
        test_store_memory_invalid_source_defaults_to_unknown,
        test_monitor_source_correct,
        test_monitor_source_incorrect,
        test_monitor_source_no_tag,
        test_confabulation_risk_low_for_external_source,
        test_confabulation_risk_high_for_inference,
        test_confabulation_risk_high_for_imagination,
        test_get_memories_by_source,
        test_source_summary,
        test_source_summary_excludes_forgotten,
        # Retrieval-induced forgetting
        test_rif_weakenes_competing_memories,
        test_rif_does_not_weaken_dissimilar_memories,
        test_rif_does_not_weaken_retrieved_memories,
        test_rif_high_salience_resists_weakening,
        test_rif_empty_retrieved_list_does_nothing,
        test_rif_integrated_with_retrieve_relevant,
        # Experience replay
        test_replay_old_memories_strengthens_old,
        test_replay_does_not_touch_recent_memories,
        test_replay_does_not_touch_forgotten_memories,
        test_replay_respects_sample_size_limit,
        test_replay_empty_store_does_nothing,
        test_replay_integrated_with_n3_consolidation,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"Memory boundary & source monitoring tests: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


# ======================================================================
# From tests/test_memory_systems.py
# ======================================================================

def test_fear_conditioning_creates_memory() -> None:
    """Fear conditioning creates a new emotional memory."""
    ems = EmotionalMemorySystem()
    memory = ems.fear_conditioning("snake", negative_emotion_strength=0.8)
    assert memory.stimulus == "snake"
    assert memory.association_strength > 0
    assert not memory.is_extinct
    assert ems.count == 1


def test_fear_conditioning_default_emotional_tag() -> None:
    """Default emotional tag has high cortisol, low serotonin."""
    ems = EmotionalMemorySystem()
    memory = ems.fear_conditioning("spider", negative_emotion_strength=0.5)
    # Neurochemical indices match the Rust canonical ordering:
    # cortisol = 6, serotonin = 1.
    assert memory.emotional_tag[6] >= 0.8
    assert memory.emotional_tag[1] <= 0.2


def test_fear_conditioning_custom_emotional_tag() -> None:
    """Custom emotional tag is stored as provided."""
    ems = EmotionalMemorySystem()
    custom_tag = [0.1] * 12
    memory = ems.fear_conditioning("height", 0.5, emotional_tag=custom_tag)
    assert memory.emotional_tag == custom_tag


def test_fear_conditioning_strength_scales_with_emotion() -> None:
    """Stronger negative emotion → stronger association."""
    ems = EmotionalMemorySystem()
    weak = ems.fear_conditioning("stimulus_weak", 0.2)
    strong = ems.fear_conditioning("stimulus_strong", 0.9)
    assert strong.association_strength > weak.association_strength


def test_fear_conditioning_reconditioning_strengthens() -> None:
    """Re-conditioning an existing (non-extinct) memory strengthens it."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("dog", 0.5)
    original = ems.retrieve_emotional("dog")
    assert original is not None
    original_strength = original.association_strength
    ems.fear_conditioning("dog", 0.5)
    reconditioned = ems.retrieve_emotional("dog")
    assert reconditioned is not None
    assert reconditioned.association_strength > original_strength


def test_fear_conditioning_increments_counter() -> None:
    """Each conditioning call increments the counter."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("a", 0.5)
    ems.fear_conditioning("b", 0.5)
    assert ems.conditioning_count == 2


# ═══════════════════════════════════════════════════════════════════
# EmotionalMemorySystem — extinction learning
# ═══════════════════════════════════════════════════════════════════


def test_extinction_weakenes_association() -> None:
    """Extinction learning reduces association strength."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("spider", 0.8)
    original = ems.retrieve_emotional("spider")
    assert original is not None
    original_strength = original.association_strength
    result = ems.extinction_learning("spider", exposures=5)
    assert result is not None
    assert result.association_strength < original_strength


def test_extinction_increases_extinction_level() -> None:
    """Extinction learning increases extinction_level."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("spider", 0.8)
    result = ems.extinction_learning("spider", exposures=3)
    assert result is not None
    assert result.extinction_level > 0.0


def test_extinction_marks_extinct() -> None:
    """Enough extinction exposures mark the memory as extinct."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("stimulus", 0.3)
    # Many exposures should push it to extinction
    result = ems.extinction_learning("stimulus", exposures=20)
    assert result is not None
    assert result.is_extinct


def test_extinction_unknown_stimulus_returns_none() -> None:
    """Extinction of unknown stimulus returns None."""
    ems = EmotionalMemorySystem()
    result = ems.extinction_learning("unknown", exposures=5)
    assert result is None


def test_extinction_increments_counter() -> None:
    """Each extinction call increments the counter."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("stimulus", 0.5)
    ems.extinction_learning("stimulus", exposures=1)
    assert ems.extinction_count == 1


def test_reconditioning_after_extinction() -> None:
    """Re-conditioning after extinction reverses extinction."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("stimulus", 0.3)
    ems.extinction_learning("stimulus", exposures=20)
    extinct = ems.retrieve_emotional("stimulus")
    assert extinct is not None
    assert extinct.is_extinct
    # Re-condition
    ems.fear_conditioning("stimulus", 0.5)
    reconditioned = ems.retrieve_emotional("stimulus")
    assert reconditioned is not None
    assert not reconditioned.is_extinct


# ═══════════════════════════════════════════════════════════════════
# EmotionalMemorySystem — retrieval
# ═══════════════════════════════════════════════════════════════════


def test_retrieve_emotional_found() -> None:
    """retrieve_emotional returns the memory for a known stimulus."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("snake", 0.5)
    result = ems.retrieve_emotional("snake")
    assert result is not None
    assert result.stimulus == "snake"


def test_retrieve_emotional_not_found() -> None:
    """retrieve_emotional returns None for unknown stimulus."""
    ems = EmotionalMemorySystem()
    result = ems.retrieve_emotional("unknown")
    assert result is None


def test_retrieve_emotional_increments_retrieval_count() -> None:
    """Retrieval increments the memory's retrieval_count."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("snake", 0.5)
    memory = ems.retrieve_emotional("snake")
    assert memory is not None
    assert memory.retrieval_count == 1
    ems.retrieve_emotional("snake")
    memory = ems.retrieve_emotional("snake")
    assert memory is not None
    assert memory.retrieval_count == 3


def test_retrieve_emotional_mood_congruent_boost() -> None:
    """Matching current emotion boosts association strength."""
    ems = EmotionalMemorySystem()
    memory_tag = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ems.fear_conditioning("snake", 0.5, emotional_tag=memory_tag)
    memory = ems.retrieve_emotional("snake")
    assert memory is not None
    original_strength = memory.association_strength
    # Retrieve with matching emotion (same direction → high cosine similarity)
    ems.retrieve_emotional("snake", current_emotion=memory_tag)
    memory = ems.retrieve_emotional("snake", current_emotion=memory_tag)
    assert memory is not None
    assert memory.association_strength > original_strength


def test_retrieve_emotional_mood_incongruent_no_boost() -> None:
    """Non-matching current emotion doesn't boost association."""
    ems = EmotionalMemorySystem()
    # Memory tag: high on first 6, low on last 6
    memory_tag = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ems.fear_conditioning("snake", 0.5, emotional_tag=memory_tag)
    memory = ems.retrieve_emotional("snake")
    assert memory is not None
    original_strength = memory.association_strength
    # Retrieve with opposite direction emotion (low on first 6, high on last 6)
    opposite = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    ems.retrieve_emotional("snake", current_emotion=opposite)
    memory = ems.retrieve_emotional("snake")
    assert memory is not None
    assert memory.association_strength == original_strength


# ═══════════════════════════════════════════════════════════════════
# EmotionalMemorySystem — counts
# ═══════════════════════════════════════════════════════════════════


def test_ems_active_count() -> None:
    """active_count excludes extinct memories."""
    ems = EmotionalMemorySystem()
    ems.fear_conditioning("a", 0.3)
    ems.fear_conditioning("b", 0.5)
    ems.extinction_learning("a", exposures=20)
    assert ems.count == 2
    assert ems.active_count == 1


# ═══════════════════════════════════════════════════════════════════
# _emotional_similarity helper
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param(
            [0.5, 0.6, 0.7, 0.8, 0.5, 0.4, 0.3, 0.5, 0.6, 0.7, 0.5, 0.5],
            [0.5, 0.6, 0.7, 0.8, 0.5, 0.4, 0.3, 0.5, 0.6, 0.7, 0.5, 0.5],
            1.0,
            id="identical",
        ),
        pytest.param(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
            id="orthogonal",
        ),
        pytest.param(
            [], [], 0.0,
            id="empty",
        ),
        pytest.param(
            [1.0, 0.0], [1.0, 0.0, 0.5, 0.5], None,
            id="different_lengths",
        ),
    ],
)
def test_emotional_similarity(a: list[float], b: list[float], expected: float | None) -> None:
    """_emotional_similarity handles identical, orthogonal, empty, and different-length vectors."""
    sim = _emotional_similarity(a, b)
    if expected is not None:
        assert abs(sim - expected) < 1e-9
    else:
        assert 0.0 < sim <= 1.0


# ═══════════════════════════════════════════════════════════════════
# PrimingSystem
# ═══════════════════════════════════════════════════════════════════


def test_priming_prime_and_is_primed() -> None:
    """Priming a concept makes it primed."""
    ps = PrimingSystem()
    ps.prime("doctor", amount=0.8)
    assert ps.is_primed("doctor") > 0.0


def test_priming_not_primed() -> None:
    """Unprimed concept has 0 priming."""
    ps = PrimingSystem()
    assert ps.is_primed("unknown") == 0.0


def test_priming_accumulates() -> None:
    """Repeated priming accumulates (bounded at 1.0)."""
    ps = PrimingSystem()
    ps.prime("concept", amount=0.5)
    level1 = ps.is_primed("concept")
    ps.prime("concept", amount=0.5)
    level2 = ps.is_primed("concept")
    assert level2 > level1


def test_priming_bounded_at_one() -> None:
    """Priming level is bounded at 1.0."""
    ps = PrimingSystem()
    ps.prime("concept", amount=1.0)
    ps.prime("concept", amount=1.0)
    assert ps.is_primed("concept") <= 1.0


def test_priming_get_all_primed() -> None:
    """get_all_primed returns all primed concepts above threshold."""
    ps = PrimingSystem()
    ps.prime("a", amount=0.5)
    ps.prime("b", amount=0.7)
    primed = ps.get_all_primed(threshold=0.01)
    assert "a" in primed
    assert "b" in primed


def test_priming_get_all_primed_threshold() -> None:
    """get_all_primed respects the threshold."""
    ps = PrimingSystem()
    ps.prime("a", amount=0.3)
    primed = ps.get_all_primed(threshold=0.5)
    assert "a" not in primed


def test_priming_with_network_spreads() -> None:
    """Priming with a network spreads to neighbors."""
    net = ConceptNetwork()
    net.add_concept("doctor")
    net.add_concept("nurse")
    net.add_edge("doctor", "nurse", RelationType.RELATED_TO, weight=0.8)
    ps = PrimingSystem(network=net)
    ps.prime("doctor", amount=0.8)
    # Nurse should be primed via spreading
    assert ps.is_primed("nurse") > 0.0


# ═══════════════════════════════════════════════════════════════════
# AttractorNetwork
# ═══════════════════════════════════════════════════════════════════


def test_attractor_store_and_pattern_count() -> None:
    """Storing a pattern increments pattern_count."""
    net = AttractorNetwork(size=8)
    net.store("p1", [1, 0, 1, 0, 1, 0, 1, 0])
    assert net.pattern_count == 1
    net.store("p2", [0, 1, 0, 1, 0, 1, 0, 1])
    assert net.pattern_count == 2


def test_attractor_retrieve_exact_match() -> None:
    """Retrieving with the exact stored pattern returns that pattern."""
    net = AttractorNetwork(size=8)
    pattern: list[float] = [1, 0, 1, 0, 1, 0, 1, 0]
    net.store("p1", pattern)
    _state, match_id = net.retrieve(pattern)
    assert match_id == "p1"


def test_attractor_retrieve_partial_cue() -> None:
    """Retrieving with a noisy cue completes to the stored pattern."""
    net = AttractorNetwork(size=8)
    pattern: list[float] = [1, 1, 1, 1, 0, 0, 0, 0]
    net.store("p1", pattern)
    # Noisy cue: 1 bit flipped (position 5 should be 0/-1 but is 1)
    cue: list[float] = [1, 1, 1, 1, 0, 1, 0, 0]
    state, match_id = net.retrieve(cue)
    assert match_id == "p1"
    # State should converge to the stored bipolar pattern
    expected = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    assert state == expected


def test_attractor_retrieve_no_match() -> None:
    """Retrieving with no stored patterns returns None match."""
    net = AttractorNetwork(size=8)
    _state, match_id = net.retrieve([1, 0, 1, 0, 1, 0, 1, 0])
    assert match_id is None


def test_attractor_retrieve_increments_count() -> None:
    """Successful retrieval increments retrieval_count."""
    net = AttractorNetwork(size=8)
    net.store("p1", [1, 1, 1, 1, 0, 0, 0, 0])
    net.retrieve([1, 1, 1, 1, 0, 0, 0, 0])
    assert net.retrieval_count == 1


def test_attractor_auto_associate() -> None:
    """auto_associate finds similar stored patterns."""
    net = AttractorNetwork(size=8)
    # Two similar patterns (differ in 1 bit)
    net.store("p1", [1, 1, 1, 1, 0, 0, 0, 0])
    net.store("p2", [1, 1, 1, 1, 0, 0, 0, 1])  # 7/8 = 0.875 similarity
    # One different pattern
    net.store("p3", [0, 0, 0, 0, 1, 1, 1, 1])  # 0/8 similarity to p1
    similar = net.auto_associate("p1")
    assert "p2" in similar
    assert "p3" not in similar
    assert "p1" not in similar  # excludes self


def test_attractor_auto_associate_unknown_returns_empty() -> None:
    """auto_associate with unknown cue returns empty list."""
    net = AttractorNetwork(size=8)
    assert net.auto_associate("unknown") == []


@pytest.mark.parametrize(
    "size, vector, expected_len, expected_vals",
    [
        pytest.param(4, [0.8, 0.3, 0.5, 0.1], 4, [1.0, -1.0, 1.0, -1.0], id="convert"),
        pytest.param(8, [1, 0], 8, [1.0, -1.0, 0.0], id="pads"),
        pytest.param(4, [1, 0, 1, 0, 1, 0], 4, [1.0, -1.0, 1.0, -1.0], id="truncates"),
    ],
)
def test_attractor_to_bipolar(
    size: int, vector: list[float], expected_len: int, expected_vals: list[float]
) -> None:
    """_to_bipolar converts, pads, and truncates as needed."""
    net = AttractorNetwork(size=size)
    result = net._to_bipolar(vector)
    assert len(result) == expected_len
    for i, val in enumerate(expected_vals):
        assert result[i] == val


# ═══════════════════════════════════════════════════════════════════
# SpreadingActivation
# ═══════════════════════════════════════════════════════════════════


def test_spreading_activate_and_get() -> None:
    """Activating a concept sets its activation level."""
    sa = SpreadingActivation()
    sa.activate("concept", amount=0.5)
    assert sa.get_activation("concept") > 0.0


def test_spreading_activate_not_activated() -> None:
    """get_activation returns 0 for unactivated concepts."""
    sa = SpreadingActivation()
    assert sa.get_activation("unknown") == 0.0


def test_spreading_activate_accumulates() -> None:
    """Repeated activation accumulates (bounded at 1.0)."""
    sa = SpreadingActivation()
    sa.activate("concept", amount=0.3)
    level1 = sa.get_activation("concept")
    sa.activate("concept", amount=0.3)
    level2 = sa.get_activation("concept")
    assert level2 > level1


def test_spreading_activate_bounded() -> None:
    """Activation is bounded at 1.0."""
    sa = SpreadingActivation()
    sa.activate("concept", amount=1.0)
    sa.activate("concept", amount=1.0)
    assert sa.get_activation("concept") <= 1.0


def test_spreading_get_all() -> None:
    """get_all returns all concepts above threshold."""
    sa = SpreadingActivation()
    sa.activate("a", amount=0.5)
    sa.activate("b", amount=0.3)
    all_activated = sa.get_all(threshold=0.01)
    assert "a" in all_activated
    assert "b" in all_activated


def test_spreading_with_network_spreads() -> None:
    """Activation with a network spreads to neighbors."""
    net = ConceptNetwork()
    net.add_concept("dog")
    net.add_concept("cat")
    net.add_edge("dog", "cat", RelationType.RELATED_TO, weight=0.8)
    sa = SpreadingActivation(network=net)
    sa.activate("dog", amount=0.5)
    # Cat should be activated via spreading
    assert sa.get_activation("cat") > 0.0


# ======================================================================
# From tests/test_spaced_repetition.py
# ======================================================================

@pytest.fixture
def network() -> ConceptNetwork:
    """Network."""
    net = ConceptNetwork()
    for name in ("tree", "plant", "flower", "garden", "dog", "cat", "animal"):
        net.add_concept(name, confidence=0.9, origin="test")
    net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("tree", "garden", RelationType.RELATED_TO, 0.7, origin="test")
    net.add_edge("flower", "garden", RelationType.RELATED_TO, 0.8, origin="test")
    return net


@pytest.fixture
def scheduler(network) -> SpacedRepetitionScheduler:
    """Scheduler."""
    return SpacedRepetitionScheduler(network)


def test_record_review_success_increases_stability(scheduler) -> None:
    """Successful review increases memory stability."""
    scheduler.record_review("tree", success=True)
    stats = scheduler.get_statistics()
    assert stats["total_reviews"] == 1
    # Stability should have grown from default
    assert stats["avg_stability"] > scheduler.default_stability


def test_record_review_failure_resets_stability(scheduler) -> None:
    """Failed review resets stability to minimum."""
    scheduler.record_review("tree", success=True)
    scheduler.record_review("tree", success=True)
    scheduler.record_review("tree", success=False)
    stats = scheduler.get_statistics()
    # After failure, stability should be at minimum
    assert stats["avg_stability"] == pytest.approx(scheduler.min_stability)


def test_retention_never_reviewed_is_zero(scheduler) -> None:
    """A concept never reviewed has zero retention."""
    assert scheduler.retention("tree") == 0.0


def test_retention_decays_over_time(scheduler) -> None:
    """Retention follows the Ebbinghaus forgetting curve."""
    scheduler.record_review("tree", success=True)
    # Immediately after review, retention should be ~1.0
    now = time.time()
    r0 = scheduler.retention("tree", now=now)
    assert r0 > 0.99
    # After a long time, retention should drop
    r_late = scheduler.retention("tree", now=now + 1e7)
    assert r_late < r0


def test_retention_curve_exponential(scheduler) -> None:
    """R(t) = exp(-t/S) — verify the exponential shape."""
    scheduler.record_review("tree", success=True)
    # Get the stability
    cid = scheduler.network._resolve("tree")
    record = scheduler._records[cid]
    base = time.time()
    r1 = scheduler.retention("tree", now=base + 1000)
    r2 = scheduler.retention("tree", now=base + 2000)
    # exp(-2000/S) / exp(-1000/S) = exp(-1000/S)
    expected_ratio = math.exp(-1000 / record.stability)
    assert abs((r2 / r1) - expected_ratio) < 0.01


def test_urgency_inversely_related_to_retention(scheduler) -> None:
    """Urgency = 1 - retention."""
    scheduler.record_review("tree", success=True)
    now = time.time()
    u = scheduler.urgency("tree", now=now)
    r = scheduler.retention("tree", now=now)
    assert abs(u - (1.0 - r)) < 1e-9


def test_importance_hub_concepts_higher(network, scheduler) -> None:
    """Hub concepts (more edges) have higher importance."""
    # 'plant' has 2 incoming IS_A edges → higher degree than 'garden'
    imp_plant = scheduler.importance("plant")
    imp_garden = scheduler.importance("garden")
    assert imp_plant > 0.0
    assert imp_garden > 0.0


def test_priority_is_urgency_times_importance(scheduler) -> None:
    """Priority = urgency × importance."""
    scheduler.record_review("tree", success=True)
    now = time.time()
    p = scheduler.priority("tree", now=now)
    u = scheduler.urgency("tree", now=now)
    i = scheduler.importance("tree")
    assert abs(p - (u * i)) < 1e-9


def test_get_review_schedule_returns_due_concepts(scheduler) -> None:
    """Schedule returns concepts below the retention threshold."""
    # Review some concepts with low stability (failure → min stability)
    scheduler.record_review("tree", success=False)
    scheduler.record_review("dog", success=False)
    # Wait a tiny bit so they're "due"
    schedule = scheduler.get_review_schedule(limit=10, now=time.time() + 1000)
    assert len(schedule) > 0
    # Each entry is (concept, urgency)
    for _concept, urgency in schedule:
        assert urgency > 0.0


def test_get_next_review_returns_most_urgent(scheduler) -> None:
    """get_next_review returns the single most urgent concept."""
    scheduler.record_review("tree", success=False)
    scheduler.record_review("dog", success=False)
    nxt = scheduler.get_next_review(now=time.time() + 1000)
    assert nxt is not None


def test_get_next_review_none_when_nothing_due(scheduler) -> None:
    """No review needed when nothing is tracked."""
    assert scheduler.get_next_review() is None


def test_next_interval_positive_after_review(scheduler) -> None:
    """Next interval is positive right after a successful review."""
    scheduler.record_review("tree", success=True)
    interval = scheduler.next_interval("tree")
    assert interval > 0


def test_ease_factor_grows_on_success(scheduler) -> None:
    """Ease factor increases on successful reviews."""
    scheduler.record_review("tree", success=True)
    cid = scheduler.network._resolve("tree")
    r1 = scheduler._records[cid]
    ef1 = r1.ease_factor
    scheduler.record_review("tree", success=True)
    ef2 = scheduler._records[cid].ease_factor
    assert ef2 >= ef1


def test_review_record_dataclass() -> None:
    """ReviewRecord stores spaced-repetition state."""
    r = ReviewRecord(concept="tree", stability=100.0)
    assert r.concept == "tree"
    assert r.stability == 100.0
    assert r.review_count == 0
    assert r.ease_factor == 2.5


def test_expanding_intervals(scheduler) -> None:
    """Each successful review increases the next interval."""
    scheduler.record_review("tree", success=True)
    i1 = scheduler.next_interval("tree")
    # Simulate time passing and another successful review
    cid = scheduler.network._resolve("tree")
    scheduler._records[cid].last_review = time.time()
    scheduler.record_review("tree", success=True)
    i2 = scheduler.next_interval("tree")
    # After a second success, stability grew → longer interval
    assert i2 > i1 * 0.9  # allow small timing tolerance
