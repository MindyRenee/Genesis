"""Dreams are dreams: sleep content must not wear waking labels.

Genesis legitimately dreams while asleep. What it must not do is emit
dream content on the waking-thought channel, count dreams as thinking, or
generate waking thoughts after sleep onset. All three happened, and the
third-party symptom was a terminal that showed ``genesis~ [thought]``
lines all night.

The distinction rides on a single field, ``SpontaneousThought.is_dream``,
which the CLI (``mind/conversation.py``) and the workspace-agency filter
both read. These tests pin that field's contract at the points where it
was being set wrong.
"""

from __future__ import annotations

import logging

import pytest

from genesis_conscious import (
    ConceptNetwork,
    CuriosityEngine,
    ReasoningEngine,
    ReflectionEngine,
    RelationType,
)
from genesis_conscious.sleep import InnerLife
from genesis_conscious.sleep.thoughts import SpontaneousThought


def _inner_life() -> InnerLife:
    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.8)
    net.add_edge("beta", "gamma", RelationType.RELATED_TO, 0.5)
    reasoning = ReasoningEngine(net)
    return InnerLife(
        net,
        CuriosityEngine(net, reasoning),
        ReflectionEngine(net),
        seed=7,
    )


def _nrem_summary():
    from genesis_client.protocol import PHASE_NREM
    from genesis_client.types import NeuroSummary

    return NeuroSummary(
        arousal=0.15,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.3,
        encoding_weight=0.3,
        consolidation_weight=0.8,
        retrieval_weight=0.3,
        phase=PHASE_NREM,
    )


def _emotion():
    """A neutral EmotionalState, built the way the real one is."""
    from genesis_conscious.limbic_system.emotion import (
        EmotionalState,
        EmotionCategory,
    )

    return EmotionalState(
        label=EmotionCategory.NEUTRAL.value,
        cognitive_style="steady",
        valence=0.0,
        alertness=0.3,
        plasticity=0.5,
        creativity=0.5,
        caution=0.2,
        openness_to_engage=0.3,
    )


class _Recorder:
    """Captures _emit() calls so a test can assert on stream kinds."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, kind: str, text: str) -> None:
        self.calls.append((kind, text))


def test_dream_derived_insight_is_announced_as_a_dream() -> None:
    """A novel connection found inside a dream must not print as a thought.

    `detect_insight` runs on every recorded thought, and a dream always
    carries at least two concepts, so dream content reliably produced
    insights. Those were emitted with kind="thought" unconditionally, so a
    night's dreaming printed lines indistinguishable from waking thoughts
    on the same `genesis~` stream.
    """
    il = _inner_life()
    recorder = _Recorder()
    il._emit = recorder  # type: ignore[method-assign,assignment]

    dream = SpontaneousThought(content="alpha and beta", trigger="dream")
    dream.is_dream = True

    il._make_insight(dream, "alpha", "gamma", None)

    kinds = [k for k, _ in recorder.calls]
    assert kinds, "an insight should still be announced"
    assert all(k == "dream" for k in kinds), (
        f"dream-derived insight emitted as {kinds}; dream content must stay "
        f"on the dream channel"
    )


def test_waking_derived_insight_is_still_announced_as_a_thought() -> None:
    """The other direction: waking insights must not be demoted."""
    il = _inner_life()
    recorder = _Recorder()
    il._emit = recorder  # type: ignore[method-assign,assignment]

    waking = SpontaneousThought(content="alpha and beta", trigger="spontaneous")
    assert not waking.is_dream

    il._make_insight(waking, "alpha", "gamma", None)

    kinds = [k for k, _ in recorder.calls]
    assert kinds and all(k == "thought" for k in kinds)


def test_thought_count_separates_dreams_from_waking() -> None:
    """A merged counter made a night of dreams look like a night of thinking."""
    il = _inner_life()
    emotion = _emotion()

    for i in range(3):
        waking = SpontaneousThought(content=f"awake {i}", trigger="spontaneous")
        il._record_thought(waking, emotion)

    for i in range(5):
        dream = SpontaneousThought(content=f"dream {i}", trigger="dream")
        dream.is_dream = True
        il._record_thought(dream, emotion)

    assert il.thought_count == 8, "the total is still every record"
    assert il.waking_thought_count == 3
    assert len(il.recent_dreams) == 5
    assert all(d.is_dream for d in il.recent_dreams)


def test_describe_recent_thoughts_labels_both_populations() -> None:
    """The user-facing line must not present dreams as thoughts."""
    il = _inner_life()
    emotion = _emotion()
    il._record_thought(
        SpontaneousThought(content="awake", trigger="spontaneous"), emotion,
    )
    dream = SpontaneousThought(content="asleep", trigger="dream")
    dream.is_dream = True
    il._record_thought(dream, emotion)

    text = il.describe_recent_thoughts()
    assert "2 spontaneous thoughts" in text
    assert "1 waking" in text, f"waking/dream split missing from: {text!r}"
    assert "1 dream)" in text, f"dream count missing from: {text!r}"


def test_chain_stops_when_sleep_onsets_mid_chain() -> None:
    """A chain begun awake must not keep thinking after she falls asleep.

    `_generate_chain` is only entered from the awake branch, so nothing
    downstream re-checked. A chain runs up to MAX_CHAIN_LENGTH steps at
    CHAIN_THOUGHT_DELAY seconds each, and every one of those was a waking
    thought emitted while asleep, recorded with is_dream=False and fed to
    the agency topic pool as if she had meant it.

    The sleep signal is isolated deliberately: emotion is stubbed so it
    can never end the chain on its own, and sleep is driven through the
    mind flag with the daemon read failing. Otherwise the chain breaks for
    some unrelated reason and the test passes whether or not the check
    exists.
    """
    il = _inner_life()
    emotion = _emotion()

    recorded: list[SpontaneousThought] = []
    original_record = il._record_thought

    def spy(t, e):
        recorded.append(t)
        original_record(t, e)

    il._record_thought = spy  # type: ignore[method-assign]

    # Never let a missing emotion end the chain.
    il._get_current_emotion = lambda: emotion  # type: ignore[method-assign]

    # The daemon read fails for the whole run; sleep is signalled only by
    # the mind's own flag, which is what a real sleep onset looks like
    # before the daemon phase catches up.
    il._get_current_summary = lambda: None  # type: ignore[method-assign]
    state = {"asleep": False, "n": 0}
    il._is_mind_sleeping = lambda: state["asleep"]  # type: ignore[method-assign]

    def fake_thought(emotion_arg, *_a):
        state["n"] += 1
        if state["n"] == 1:
            # She falls asleep right after the first thought.
            state["asleep"] = True
        return SpontaneousThought(content=f"step {state['n']}", trigger="spontaneous")

    il._generate_thought = fake_thought  # type: ignore[method-assign]
    il._generate_seeded_thought = fake_thought  # type: ignore[method-assign]
    il._chain_length = lambda emotion_arg: 8  # type: ignore[method-assign,assignment]
    il._chain_continue_probability = (  # type: ignore[assignment]
        lambda emotion, thought: 1.0
    )

    import genesis_conscious.sleep.inner_life as il_mod

    original_delay = il_mod.CHAIN_THOUGHT_DELAY
    il_mod.CHAIN_THOUGHT_DELAY = 0.0
    try:
        il._generate_chain(emotion)
    finally:
        il_mod.CHAIN_THOUGHT_DELAY = original_delay

    assert state["n"] == 1, (
        f"the waking generator ran {state['n']} times after sleep onset; it "
        f"must stop at the first step"
    )
    assert len(recorded) == 1
    assert recorded[0].is_dream is False, "sanity: step 0 was genuinely awake"


def test_failed_summary_read_does_not_disable_sleep_detection() -> None:
    """A transient IPC failure must not report her as awake.

    The dispatcher and the run loop both need this answer and used to
    compute it separately — one of them as
    `_is_currently_sleeping(summary) if summary else False`, which threw
    away the mind-flag fallback that exists precisely for when the daemon
    read fails. A transient IPC error then reported her awake and ran the
    social-drive and curiosity branches mid-sleep.

    `_sleep_state` is now the single accessor, so assert *it* rather than
    re-testing `_is_currently_sleeping` directly, which would pass whether
    or not the call sites were fixed.
    """
    il = _inner_life()
    il._get_current_summary = lambda: None  # type: ignore[method-assign]

    il._is_mind_sleeping = lambda: True
    assert il._sleep_state() is True, (
        "no summary + mind says asleep must be asleep, not awake"
    )

    il._is_mind_sleeping = lambda: False
    assert il._sleep_state() is False


@pytest.mark.parametrize("phase", ["nrem", "rem", "sleeping"])
def test_dream_phase_is_recognised_as_sleeping(phase: str) -> None:
    """The three phase names the daemon can report all mean asleep."""
    il = _inner_life()

    class _S:
        phase_name = phase

    il._is_mind_sleeping = lambda: False
    assert il._is_currently_sleeping(_S()) is True


def test_dream_generators_are_only_flagged_by_their_callers() -> None:
    """`is_dream` is set at the call site, not inside the generators.

    Every dream generator returns a plain SpontaneousThought with
    `metadata["dream"] = True`; the flag the rest of the system reads is
    applied by `_generate_first_dream_thought` and `_continue_dream_chain`.
    That is fine while those two are the only callers, and this test
    documents the contract so a new direct caller is noticed.
    """
    il = _inner_life()
    summary = _nrem_summary()
    emotion = _emotion()

    il._generate_dream(emotion, summary)

    with il._thoughts_lock:
        dreams = list(il._thoughts)
    if not dreams:
        pytest.skip("dream generation is probabilistic; nothing to assert")
    for d in dreams:
        assert d.is_dream, (
            f"dream {d.content!r} recorded without is_dream — it would be "
            f"displayed and broadcast as a waking thought"
        )


def test_parasomnia_log_is_not_a_thought(caplog) -> None:
    """Parasomnia events are logged, never surfaced as thoughts."""
    il = _inner_life()
    caplog.set_level(logging.WARNING, logger="genesis_conscious.sleep.inner_life")

    recorder = _Recorder()
    il._emit = recorder  # type: ignore[method-assign,assignment]
    il._check_parasomnia(_emotion(), _nrem_summary())

    assert all("thought" not in kind for kind, _ in recorder.calls), (
        f"parasomnia must not emit thoughts, got {recorder.calls}"
    )


# ── Dream-derived vs learning target ───────────────────────────────
#
# `is_dream` answers "generated while asleep". The autonomous learner
# needs a different question: "may this become a learning goal?". The
# two are not the same, and conflating them is what let dream material
# reach the learner.


def _thought(trigger: str, **kw) -> SpontaneousThought:
    t = SpontaneousThought(content=kw.pop("content", "x"), trigger=trigger)
    t.metadata = kw.pop("metadata", None)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


@pytest.mark.parametrize(
    "trigger",
    ["dream", "lucid-dream", "dream-reflection"],
)
def test_dream_derived_triggers_are_not_learning_targets(trigger: str) -> None:
    """Dreams and reflections *on* dreams must not seed learning."""
    t = _thought(trigger)
    assert t.is_dream_derived() is True, f"{trigger} is dream-derived"
    assert t.is_learning_target() is False, (
        f"{trigger} would queue '{t.content}' for the autonomous learner; "
        f"dreams produce affective impressions, not goals"
    )


def test_dream_flag_alone_marks_dream_derived() -> None:
    """A thought stamped is_dream is dream-derived regardless of trigger."""
    t = _thought("spontaneous", is_dream=True)
    assert t.is_dream_derived() is True
    assert t.is_learning_target() is False


def test_dream_metadata_marks_dream_derived() -> None:
    """The generators tag metadata["dream"]; that counts too.

    None of the seven dream generators sets is_dream itself — their
    callers do. Reading the metadata means the predicate is right even if
    a caller forgets.
    """
    t = _thought("spontaneous", metadata={"dream": True})
    assert t.is_dream_derived() is True
    assert t.is_learning_target() is False


def test_hypnagogic_is_not_a_dream_and_remains_a_learning_target() -> None:
    """The drowsy pre-sleep state is not sleep, and must stay learnable.

    Hypnagogia happens while she is still awake — a drifting reflection
    on the day, in the gap before the first dream. Setting its is_dream
    to True would be a second bug in the opposite direction: it would cut
    genuine pre-sleep reflection out of the learner. This test exists to
    stop that "fix".
    """
    t = _thought("hypnagogic", metadata={"hypnagogic": True})

    assert t.is_dream is False, "hypnagogia is not a dream"
    assert t.is_dream_derived() is False
    assert t.is_learning_target() is True, (
        "pre-sleep reflection is waking cognition and may seed learning"
    )


def test_ordinary_waking_thought_is_a_learning_target() -> None:
    for trigger in ("spontaneous", "curiosity", "emotional", "reflect"):
        t = _thought(trigger)
        assert t.is_dream_derived() is False, trigger
        assert t.is_learning_target() is True, trigger


def test_broadcast_carries_the_explicit_learning_flag() -> None:
    """The consumer needs the learning answer, not the sleep answer.

    The subliminal drain used to read `is_dream` and infer. Broadcasting
    both makes the contract explicit, and lets the filter treat a missing
    flag as permissive so a workspace from older state degrades to the old
    behavior instead of silently dropping every topic.
    """
    il = _inner_life()
    captured: list[dict] = []

    class _GW:
        def broadcast(self, content, source, activation, brain_waves=None, metadata=None):
            captured.append(metadata or {})

        def drain_subliminal(self):
            return []

        def tick(self, dt):
            pass

    class _Cog:
        global_workspace = _GW()

        class language:
            current_brain_waves = None

    il._cognition = _Cog()

    dream = _thought("dream", is_dream=True)
    il._broadcast_thought(dream, ["alpha"])
    reflection = _thought("dream-reflection")
    il._broadcast_thought(reflection, ["beta"])
    awake = _thought("hypnagogic")
    il._broadcast_thought(awake, ["gamma"])

    assert captured[0]["is_dream"] is True
    assert captured[0]["is_learning_target"] is False
    assert captured[1]["is_dream"] is False, (
        "a waking reflection is correctly not flagged as a dream"
    )
    assert captured[1]["is_learning_target"] is False, (
        "…but it is still dream-derived, so it must not seed learning"
    )
    assert captured[2]["is_learning_target"] is True


def test_agency_filter_uses_the_learning_flag() -> None:
    """The drain must key off is_learning_target, not is_dream.

    A dream reflection broadcasts with is_dream=False, so the old
    `if item.metadata.get("is_dream"): continue` passed it straight into
    the agency queue — the learner being handed a topic she had only
    dreamed about.
    """
    il = _inner_life()

    class _Item:
        def __init__(self, topics, **meta):
            self.metadata = {"topics": topics, **meta}

    class _GW:
        def __init__(self, items):
            self._items = items

        def drain_subliminal(self):
            return self._items

    dream_reflection = _Item(["alpha"], is_dream=False, is_learning_target=False)
    assert il._absorb_subliminal(_GW([dream_reflection])) == 0
    assert il.agency_topics == [], (
        f"a dream reflection reached the learner: {il.agency_topics}"
    )

    in_sleep_dream = _Item(["beta"], is_dream=True, is_learning_target=False)
    hypnagogic = _Item(["gamma"], is_dream=False, is_learning_target=True)
    assert il._absorb_subliminal(_GW([in_sleep_dream, hypnagogic])) == 1
    assert il.agency_topics == ["gamma"], (
        "only the hypnagogic topic may reach the learner"
    )


def test_absorb_subliminal_permits_content_with_no_flag() -> None:
    """Workspace content from an older state must not starve the learner.

    The flag is additive metadata. Treating its absence as "not a
    learning target" would silently disable agency topics after an
    upgrade until the next save, so absence means permitted.
    """
    il = _inner_life()

    class _Item:
        def __init__(self, topics):
            self.metadata = {"topics": topics}

    class _GW:
        def drain_subliminal(self):
            return [_Item(["alpha"])]

    assert il._absorb_subliminal(_GW()) == 1
    assert il.agency_topics == ["alpha"]
