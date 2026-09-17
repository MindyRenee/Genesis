"""Tests for brain wave states — derivation from neurochemistry and
cognitive gating effects."""

import logging
import sys

from genesis_client.protocol import (
    PHASE_ACTIVE,
    PHASE_DROWSY,
    PHASE_FLOW,
    PHASE_NREM,
    PHASE_OVERWHELMED,
    PHASE_REM,
    PHASE_SLEEPING,
    PHASE_STRESS,
)
from genesis_client.types import NeuroSummary
from genesis_cognitive.brain_waves import (
    BrainWave,
    BrainWaveState,
    SleepStage,
    _sleep_stage_from_summary,
    assess_brain_waves,
)

logger = logging.getLogger(__name__)


def _make_summary(
    arousal=0.5,
    valence=0.0,
    plasticity=0.5,
    phase=0,  # 0=active
    encoding=0.5,
    consolidation=0.5,
    retrieval=0.5,
    global_tone=0.5,
) -> NeuroSummary:
    """Create a NeuroSummary with specified values."""
    return NeuroSummary(
        arousal=arousal,
        valence=valence,
        global_tone=global_tone,
        plasticity_gate=plasticity,
        encoding_weight=encoding,
        consolidation_weight=consolidation,
        retrieval_weight=retrieval,
        phase=phase,
    )


# ─── Derivation tests ────────────────────────────────────────


def test_high_arousal_high_plasticity_gamma():
    """High arousal + high plasticity → gamma dominant."""
    summary = _make_summary(arousal=0.8, plasticity=0.7, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.GAMMA
    assert state.integration > 0.5


def test_high_arousal_low_plasticity_beta():
    """High arousal + low plasticity → beta (alert but not integrating)."""
    summary = _make_summary(arousal=0.8, plasticity=0.2, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA
    assert state.integration < 0.5


def test_moderate_arousal_alpha():
    """Moderate arousal → alpha dominant (reflective filtering)."""
    summary = _make_summary(arousal=0.55, plasticity=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.ALPHA
    assert state.focus > 0.3  # alpha produces focus


def test_low_arousal_theta():
    """Low arousal → theta dominant (memory consolidation)."""
    summary = _make_summary(arousal=0.35, plasticity=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.THETA


def test_very_low_arousal_delta():
    """Very low arousal → delta dominant."""
    summary = _make_summary(arousal=0.15, plasticity=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.DELTA


def test_sleep_phase_delta():
    """Sleep phase → delta dominant."""
    summary = _make_summary(arousal=0.1, phase=PHASE_SLEEPING)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.DELTA
    assert state.powers[BrainWave.DELTA] > 0.4


def test_sleep_phase_has_theta():
    """Sleep phase has theta for dreaming."""
    summary = _make_summary(arousal=0.1, phase=PHASE_SLEEPING)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.THETA] > 0.2


def test_flow_phase_gamma():
    """Flow phase → gamma dominant."""
    summary = _make_summary(arousal=0.7, plasticity=0.6, phase=PHASE_FLOW)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.GAMMA
    assert state.integration > 0.6


def test_stress_phase_beta():
    """Stress phase → beta without gamma integration."""
    summary = _make_summary(arousal=0.7, phase=PHASE_STRESS)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA
    assert state.integration < 0.4  # stress blocks integration


def test_drowsy_phase_theta():
    """Drowsy phase → theta dominant."""
    summary = _make_summary(arousal=0.3, phase=PHASE_DROWSY)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.THETA


def test_overwhelmed_phase_beta():
    """Overwhelmed phase → beta (alert but can't integrate)."""
    summary = _make_summary(arousal=0.8, phase=PHASE_OVERWHELMED)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA


# ─── Cognitive gating tests ──────────────────────────────────


def test_gamma_high_integration():
    """Gamma state has high integration."""
    summary = _make_summary(arousal=0.8, plasticity=0.8, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.integration > 0.5


def test_alpha_high_focus():
    """Alpha state has high focus (filtering)."""
    summary = _make_summary(arousal=0.55, plasticity=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.focus > 0.3


def test_delta_low_engagement():
    """Delta state has low consolidation and focus."""
    summary = _make_summary(arousal=0.1, phase=PHASE_SLEEPING)
    state = assess_brain_waves(summary)
    # Delta has high consolidation (restorative) but low focus
    assert state.consolidation > 0.3
    assert state.focus < 0.3


def test_theta_high_consolidation():
    """Theta state has high consolidation (memory processing)."""
    summary = _make_summary(arousal=0.35, consolidation=0.8, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.consolidation > 0.4


def test_powers_sum_to_one():
    """Brain wave powers sum to approximately 1.0."""
    for arousal in [0.1, 0.3, 0.5, 0.7, 0.9]:
        summary = _make_summary(arousal=arousal, phase=PHASE_ACTIVE)
        state = assess_brain_waves(summary)
        total = sum(state.powers.values())
        assert abs(total - 1.0) < 0.01, f"Powers sum to {total} at arousal {arousal}"


def test_secondary_wave_exists():
    """Secondary wave is always set and different from dominant."""
    for arousal in [0.1, 0.3, 0.5, 0.7, 0.9]:
        summary = _make_summary(arousal=arousal, phase=PHASE_ACTIVE)
        state = assess_brain_waves(summary)
        assert state.secondary != state.dominant


# ─── Description tests ───────────────────────────────────────


def test_describe():
    """Brain wave state can describe itself."""
    summary = _make_summary(arousal=0.8, plasticity=0.7, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    desc = state.describe()
    assert "gamma" in desc
    assert "mind in" in desc


def test_describe_cognition():
    """Brain wave state can describe its cognitive effects."""
    summary = _make_summary(arousal=0.55, plasticity=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    cog = state.describe_cognition()
    assert len(cog) > 0


def test_label_matches_dominant():
    """Label string matches the dominant wave."""
    summary = _make_summary(arousal=0.8, plasticity=0.7, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.label == state.dominant.value


# ─── Plasticity gating tests ─────────────────────────────────


def test_plasticity_gates_integration():
    """Low plasticity reduces integration even with gamma."""
    high_plast = _make_summary(arousal=0.8, plasticity=0.8, phase=PHASE_ACTIVE)
    low_plast = _make_summary(arousal=0.8, plasticity=0.2, phase=PHASE_ACTIVE)

    high_state = assess_brain_waves(high_plast)
    low_state = assess_brain_waves(low_plast)

    assert high_state.integration > low_state.integration


def test_consolidation_weight_affects_theta():
    """Higher consolidation weight boosts theta consolidation."""
    low_consol = _make_summary(arousal=0.35, consolidation=0.2, phase=PHASE_ACTIVE)
    high_consol = _make_summary(arousal=0.35, consolidation=0.9, phase=PHASE_ACTIVE)

    low_state = assess_brain_waves(low_consol)
    high_state = assess_brain_waves(high_consol)

    assert high_state.consolidation > low_state.consolidation


# ─── Sleep stage tests ────────────────────────────────────────


def test_nrem_phase_never_returns_rem():
    """Explicit NREM phase must never produce REM, even with high arousal."""
    summary = _make_summary(arousal=0.6, phase=PHASE_NREM)
    stage = _sleep_stage_from_summary(summary)
    assert stage is not None
    assert stage != SleepStage.REM, "NREM phase must not return REM"


def test_nrem_phase_high_arousal_n1():
    """NREM phase with high arousal → N1 (lightest NREM)."""
    summary = _make_summary(arousal=0.45, phase=PHASE_NREM)
    stage = _sleep_stage_from_summary(summary)
    assert stage == SleepStage.N1


def test_nrem_phase_mid_arousal_n2():
    """NREM phase with mid arousal → N2."""
    summary = _make_summary(arousal=0.25, phase=PHASE_NREM)
    stage = _sleep_stage_from_summary(summary)
    assert stage == SleepStage.N2


def test_nrem_phase_low_arousal_n3():
    """NREM phase with low arousal → N3 (deepest)."""
    summary = _make_summary(arousal=0.10, phase=PHASE_NREM)
    stage = _sleep_stage_from_summary(summary)
    assert stage == SleepStage.N3


def test_rem_phase_returns_rem():
    """Explicit REM phase → REM regardless of arousal."""
    summary = _make_summary(arousal=0.2, phase=PHASE_REM)
    stage = _sleep_stage_from_summary(summary)
    assert stage == SleepStage.REM


def test_non_sleep_phase_returns_none():
    """Non-sleep phase → None."""
    summary = _make_summary(arousal=0.8, phase=PHASE_ACTIVE)
    stage = _sleep_stage_from_summary(summary)
    assert stage is None


# ─── Subsystem wiring tests ───────────────────────────────────


def _make_wave_state(dominant: BrainWave) -> BrainWaveState:
    """Create a BrainWaveState with a specific dominant wave for testing."""
    if dominant == BrainWave.GAMMA:
        return assess_brain_waves(_make_summary(arousal=0.85, plasticity=0.8, phase=PHASE_ACTIVE))
    elif dominant == BrainWave.BETA:
        return assess_brain_waves(_make_summary(arousal=0.8, plasticity=0.2, phase=PHASE_ACTIVE))
    elif dominant == BrainWave.ALPHA:
        return assess_brain_waves(_make_summary(arousal=0.55, plasticity=0.5, phase=PHASE_ACTIVE))
    elif dominant == BrainWave.THETA:
        return assess_brain_waves(_make_summary(arousal=0.35, plasticity=0.5, phase=PHASE_ACTIVE))
    elif dominant == BrainWave.DELTA:
        return assess_brain_waves(_make_summary(arousal=0.1, phase=PHASE_SLEEPING))
    else:
        return assess_brain_waves(_make_summary(arousal=0.5, phase=PHASE_ACTIVE))


def test_curiosity_gamma_more_questions():
    """Gamma state produces more curiosity questions than baseline."""
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.emotion import EmotionalState
    from genesis_cognitive.learning import CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    emo = EmotionalState(
        label="curious", nuance="default", cognitive_style="steady",
        valence=0.3, alertness=0.7, plasticity=0.6, creativity=0.5,
        caution=0.2, openness_to_engage=0.7,
    )
    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    engine = CuriosityEngine(net, reasoning)

    gamma = _make_wave_state(BrainWave.GAMMA)
    delta = _make_wave_state(BrainWave.DELTA)

    gamma_questions = engine.generate_questions(emo, max_questions=2, brain_waves=gamma)
    delta_questions = engine.generate_questions(emo, max_questions=2, brain_waves=delta)

    # Gamma should produce at least as many questions as delta
    assert len(gamma_questions) >= len(delta_questions)


def test_working_memory_gamma_expands_capacity():
    """Gamma state expands working memory capacity."""
    from genesis_cognitive.memory import WorkingMemory

    wm = WorkingMemory()
    summary = _make_summary(arousal=0.5, plasticity=0.5, phase=PHASE_ACTIVE)

    gamma = _make_wave_state(BrainWave.GAMMA)
    delta = _make_wave_state(BrainWave.DELTA)

    wm.update_capacity(summary, gamma)
    gamma_cap = wm.capacity

    wm.update_capacity(summary, delta)
    delta_cap = wm.capacity

    assert gamma_cap > delta_cap


def test_global_workspace_gamma_lowers_threshold():
    """Gamma lowers the ignition threshold — easier broadcast."""
    from genesis_cognitive.global_workspace import GlobalWorkspace

    gw = GlobalWorkspace(ignition_threshold=0.7)

    gamma = _make_wave_state(BrainWave.GAMMA)
    delta = _make_wave_state(BrainWave.DELTA)

    # With gamma, a 0.65 activation should ignite (threshold lowered)
    ignited_gamma = gw.broadcast("test", "src", 0.65, brain_waves=gamma)
    assert ignited_gamma is True

    # Reset workspace
    gw._items.clear()
    gw._broadcast_count = 0

    # With delta, a 0.65 activation should NOT ignite (threshold raised)
    ignited_delta = gw.broadcast("test", "src", 0.65, brain_waves=delta)
    assert ignited_delta is False


def test_drift_diffusion_gamma_faster_accumulation():
    """Gamma speeds up evidence accumulation vs delta."""
    from genesis_cognitive.reasoning import DriftDiffusionModel

    gamma = _make_wave_state(BrainWave.GAMMA)
    delta = _make_wave_state(BrainWave.DELTA)

    # Run two identical DDMs with different brain waves
    ddm_gamma = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm_delta = DriftDiffusionModel(threshold=1.0, noise=0.0)

    for ddm in [ddm_gamma, ddm_delta]:
        ddm.add_option("a")
        ddm.add_option("b")
        ddm.add_evidence("test", "a", 0.3)

    # Tick with gamma — should accumulate faster
    result_gamma = ddm_gamma.tick(dt=1.0, brain_waves=gamma)
    result_delta = ddm_delta.tick(dt=1.0, brain_waves=delta)

    # Gamma should have more evidence on "a" than delta
    if result_gamma is None and result_delta is None:
        # Neither reached threshold — compare evidence
        # Access the accumulator
        a_gamma = ddm_gamma._accumulators.get("a")
        a_delta = ddm_delta._accumulators.get("a")
        if a_gamma and a_delta:
            assert a_gamma.evidence > a_delta.evidence


def test_narrative_delta_suppresses_recording():
    """Delta state suppresses narrative event recording."""
    import time

    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.emotion import EmotionalState
    from genesis_cognitive.narrative import NarrativeEngine
    from genesis_cognitive.self import SelfModel

    model = SelfModel(born_at=int(time.time() * 1000))
    net = ConceptNetwork()
    engine = NarrativeEngine(model, net)

    emo = EmotionalState(
        label="neutral", nuance="default", cognitive_style="steady",
        valence=0.0, alertness=0.5, plasticity=0.5, creativity=0.5,
        caution=0.3, openness_to_engage=0.7,
    )

    delta = _make_wave_state(BrainWave.DELTA)
    gamma = _make_wave_state(BrainWave.GAMMA)

    # Delta should suppress recording
    event_delta = engine.record_event("test", "test", emo, brain_waves=delta)
    assert event_delta is None

    # Gamma should allow recording
    event_gamma = engine.record_event("test", "test", emo, brain_waves=gamma)
    assert event_gamma is not None


def test_dream_synthesis_theta_more_proposals():
    """Theta (REM) state produces more dream synthesis proposals than delta."""
    from genesis_cognitive.concepts import ConceptNetwork
    from genesis_cognitive.sleep import DreamSynthesisEngine

    net = ConceptNetwork()
    # Add some concepts with similar structure
    for i in range(10):
        net.add_concept(f"concept_{i}", origin="learned")

    engine = DreamSynthesisEngine(net)

    theta = _make_wave_state(BrainWave.THETA)
    delta = _make_wave_state(BrainWave.DELTA)

    theta_proposals = engine.synthesize(max_proposals=5, brain_waves=theta)
    delta_proposals = engine.synthesize(max_proposals=5, brain_waves=delta)

    # Theta should allow more proposals than delta
    # (both may be 0 if no structural matches, but theta's effective_max is higher)
    assert len(theta_proposals) >= len(delta_proposals)


def test_sleep_compression_consolidation_scales_homeostasis():
    """High consolidation intensity produces more aggressive downscaling."""
    import tempfile

    from genesis_cognitive.concepts import ConceptNetwork, Edge, RelationType
    from genesis_cognitive.sleep import SleepCompressor

    net = ConceptNetwork()
    net.add_concept("a", origin="learned")
    net.add_concept("b", origin="learned")
    net.add_edge("a", "b", RelationType.IS_A, weight=0.8)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressor = SleepCompressor(tmpdir, homeostasis_scale=0.9)

        # High consolidation → more aggressive downscaling
        compressor._apply_homeostasis(net, consolidation_intensity=1.0)
        high_edges = [e for e in net.edges if e.source == "a" and e.target == "b"]
        high_final = high_edges[0].weight if high_edges else 0.0

        # Reset
        net.replace_edges([Edge(source="a", target="b", relation=RelationType.IS_A, weight=0.8)])

        # Low consolidation → gentler downscaling
        compressor._apply_homeostasis(net, consolidation_intensity=0.1)
        low_edges = [e for e in net.edges if e.source == "a" and e.target == "b"]
        low_final = low_edges[0].weight if low_edges else 0.0

        # High consolidation should have downscaled more
        assert high_final < low_final


def test_voice_brain_waves_modulate_rhythm():
    """Brain wave state modulates voice rhythm."""
    from genesis_cognitive.emotion import EmotionalState
    from genesis_cognitive.language import Voice
    from genesis_cognitive.self import PersonalityTraits

    personality = PersonalityTraits()
    voice = Voice(personality, seed=42)

    emo = EmotionalState(
        label="neutral", nuance="default", cognitive_style="steady",
        valence=0.0, alertness=0.7, plasticity=0.5, creativity=0.5,
        caution=0.3, openness_to_engage=0.7,
    )

    # Long sentence that could be split
    sentences = [
        "this is a very long sentence that could potentially be split "
        "into two separate sentences for clarity"
    ]

    gamma = _make_wave_state(BrainWave.GAMMA)
    delta = _make_wave_state(BrainWave.DELTA)

    # Run multiple times to account for randomness
    gamma_splits = 0
    delta_splits = 0
    for _ in range(20):
        result_gamma = voice._adjust_rhythm(list(sentences), emo, gamma)
        result_delta = voice._adjust_rhythm(list(sentences), emo, delta)
        if len(result_gamma) > len(sentences):
            gamma_splits += 1
        if len(result_delta) > len(sentences):
            delta_splits += 1

    # Delta should split more aggressively than gamma
    assert delta_splits >= gamma_splits


# ─── Brain-wave-driven self-priority tests ────────────────────


def test_derive_self_priority_gamma_dominant():
    """Gamma-dominant brain waves should produce high scheduling priority."""
    from genesis_cognitive.brain_waves import derive_self_priority

    state = BrainWaveState(
        dominant=BrainWave.GAMMA,
        secondary=BrainWave.BETA,
        powers={
            BrainWave.GAMMA: 0.50,
            BrainWave.BETA: 0.25,
            BrainWave.ALPHA: 0.15,
            BrainWave.THETA: 0.07,
            BrainWave.DELTA: 0.03,
        },
        focus=0.7,
        integration=0.8,
        consolidation=0.1,
        label="gamma",
        description="active integration",
    )
    nice, io_class = derive_self_priority(state)
    # Gamma-dominant → high engagement → negative nice (boosted)
    assert nice < 0, f"gamma-dominant should have negative nice: {nice}"
    assert nice >= -5, f"nice should be clamped to -5: {nice}"
    # High integration → fast I/O
    assert io_class in ("best-effort-0", "best-effort-1"), (
        f"high integration should have fast I/O: {io_class}"
    )


def test_derive_self_priority_delta_dominant():
    """Delta-dominant brain waves (deep sleep) should produce lowest priority."""
    from genesis_cognitive.brain_waves import derive_self_priority

    state = BrainWaveState(
        dominant=BrainWave.DELTA,
        secondary=BrainWave.THETA,
        powers={
            BrainWave.DELTA: 0.55,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.12,
            BrainWave.BETA: 0.06,
            BrainWave.GAMMA: 0.02,
        },
        focus=0.1,
        integration=0.05,
        consolidation=0.6,
        label="delta",
        description="deep sleep",
    )
    nice, io_class = derive_self_priority(state)
    # Delta-dominant → lowest priority → high positive nice
    assert nice > 5, f"delta-dominant should have high positive nice: {nice}"
    assert nice <= 10, f"nice should be clamped to 10: {nice}"
    # Delta > 50% → idle I/O
    assert io_class == "idle", f"delta-dominant should have idle I/O: {io_class}"


def test_derive_self_priority_brain_waves_override_body():
    """Brain waves should override the body recommendation when they disagree."""
    from genesis_cognitive.brain_waves import derive_self_priority

    # Gamma-dominant (active thinking) but body says deprioritize (tired)
    state = BrainWaveState(
        dominant=BrainWave.GAMMA,
        secondary=BrainWave.BETA,
        powers={
            BrainWave.GAMMA: 0.45,
            BrainWave.BETA: 0.30,
            BrainWave.ALPHA: 0.15,
            BrainWave.THETA: 0.07,
            BrainWave.DELTA: 0.03,
        },
        focus=0.7,
        integration=0.7,
        consolidation=0.1,
        label="gamma",
        description="active integration",
    )
    # Body recommends nice=10 (deep deprioritize — very tired)
    nice, _ = derive_self_priority(state, body_recommended_nice=10)
    # Brain waves should override: gamma-dominant should still get
    # negative or near-zero nice despite the body saying 10
    assert nice < 5, (
        f"brain waves should override body deprioritization: nice={nice}"
    )


def test_derive_self_priority_clamps():
    """Priority values should be within valid ranges."""
    from genesis_cognitive.brain_waves import derive_self_priority

    # Extreme gamma
    state = BrainWaveState(
        dominant=BrainWave.GAMMA,
        secondary=BrainWave.BETA,
        powers={
            BrainWave.GAMMA: 1.0,
            BrainWave.BETA: 0.0,
            BrainWave.ALPHA: 0.0,
            BrainWave.THETA: 0.0,
            BrainWave.DELTA: 0.0,
        },
        focus=1.0,
        integration=1.0,
        consolidation=0.0,
        label="gamma",
        description="max gamma",
    )
    nice, io_class = derive_self_priority(state)
    assert -5 <= nice <= 10, f"nice out of range: {nice}"
    assert io_class in ("idle", "best-effort-0", "best-effort-1",
                        "best-effort-3", "best-effort-6"), (
        f"invalid io_class: {io_class}"
    )


# ─── Test runner ──────────────────────────────────────────────


def run_all():
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    logger.info(f"\n  Brain wave tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
