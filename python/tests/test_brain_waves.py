"""Tests for brain wave states — derivation from neurochemistry and
cognitive gating effects."""

import logging
import math
import sys
import time

import pytest
from thalamus.brain_waves import (
    BrainWave,
    BrainWaveState,
    SleepStage,
    _sleep_stage_from_summary,
    assess_brain_waves,
)

from genesis_client.protocol import (
    PHASE_ACTIVE,
    PHASE_ALERT,
    PHASE_DROWSY,
    PHASE_FLOW,
    PHASE_NREM,
    PHASE_OVERWHELMED,
    PHASE_REM,
    PHASE_SLEEPING,
    PHASE_STRESS,
    ZONE_IDLE,
    ZONE_SLEEPING,
)
from genesis_client.types import FoldMargins, NeuroSummary

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _fresh_oscillator():
    """Every test meters a fresh field.

    Reported band powers are *measured* — Goertzel over each band's
    emitted trace — so they carry up to ~4 s of emitted history.
    Without a reset, a test's spectrum leaks into the next test's
    reading.
    """
    from thalamus.brain_waves import reset_oscillator

    reset_oscillator()


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
    assert state.dominant == BrainWave.BETA2
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
    """Stress phase → high beta without gamma integration."""
    summary = _make_summary(arousal=0.7, phase=PHASE_STRESS)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA3
    assert state.integration < 0.4  # stress blocks integration


def test_drowsy_phase_theta():
    """Drowsy phase → theta dominant."""
    summary = _make_summary(arousal=0.3, phase=PHASE_DROWSY)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.THETA


def test_overwhelmed_phase_beta():
    """Overwhelmed phase → high beta (alert but can't integrate)."""
    summary = _make_summary(arousal=0.8, phase=PHASE_OVERWHELMED)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA3


# ─── Physiology alignment tests ──────────────────────────────────


def test_drowsy_is_alpha_theta_not_delta():
    """Drowsy (early) is alpha+theta — delta only deepens in."""
    summary = _make_summary(arousal=0.5, phase=PHASE_DROWSY)
    state = assess_brain_waves(summary)
    assert state.dominant in (BrainWave.ALPHA, BrainWave.THETA)
    assert state.powers[BrainWave.ALPHA] + state.powers[BrainWave.THETA] > 0.5
    assert state.powers[BrainWave.DELTA] < 0.2


def test_drowsy_deepens_alpha_dropout():
    """Deeper drowsiness: theta rises, alpha falls out."""
    shallow = assess_brain_waves(_make_summary(arousal=0.5, phase=PHASE_DROWSY))
    deep = assess_brain_waves(_make_summary(arousal=0.3, phase=PHASE_DROWSY))
    assert deep.powers[BrainWave.THETA] > shallow.powers[BrainWave.THETA]
    assert deep.powers[BrainWave.ALPHA] < shallow.powers[BrainWave.ALPHA]


def test_active_low_arousal_resolves_to_drowsy_signature():
    """Active phase with N1-level arousal must read like drowsy, not like alert."""
    active = assess_brain_waves(_make_summary(arousal=0.3, phase=PHASE_ACTIVE))
    drowsy = assess_brain_waves(_make_summary(arousal=0.3, phase=PHASE_DROWSY))
    assert active.dominant == drowsy.dominant == BrainWave.THETA
    assert abs(active.powers[BrainWave.THETA] - drowsy.powers[BrainWave.THETA]) < 0.02


def test_rem_theta_dominant_with_awakening_intensity():
    """REM: theta dominant but beta/gamma at waking metabolic intensity."""
    state = assess_brain_waves(_make_summary(arousal=0.6, phase=PHASE_REM))
    assert state.dominant == BrainWave.THETA
    fast = (
        state.powers[BrainWave.BETA1]
        + state.powers[BrainWave.BETA2]
        + state.powers[BrainWave.BETA3]
        + state.powers[BrainWave.GAMMA]
    )
    assert fast > 0.30, f"REM fast-band sum too low: {fast}"
    assert state.powers[BrainWave.GAMMA] > 0.10, (
        f"REM gamma too low: {state.powers[BrainWave.GAMMA]}"
    )


def test_rem_hippocampus_is_theta_predominant():
    """REM theta is hippocampus-predominant."""
    state = assess_brain_waves(_make_summary(arousal=0.6, phase=PHASE_REM))
    hipp = state.regional_powers["hippocampus"]
    assert max(hipp, key=lambda b: hipp[b]) == BrainWave.THETA


def test_alpha_dominant_at_occipital_for_relaxed_state():
    """Relaxed-wake alpha is occipital."""
    state = assess_brain_waves(_make_summary(arousal=0.55, plasticity=0.5, phase=PHASE_ACTIVE))
    occ = state.regional_powers["occipital"]
    assert max(occ, key=lambda b: occ[b]) == BrainWave.ALPHA


def test_rem_alpha_is_intermittent():
    """REM alpha waxes/wanes on the infraslow envelope."""
    from thalamus.brain_waves import BrainWaveOscillator, _sleep_phase_powers

    osc = BrainWaveOscillator()
    summary = _make_summary(arousal=0.6, phase=PHASE_REM)
    targets, _l, _d = _sleep_phase_powers(summary)
    osc._last_phase_name = "rem"

    osc._amplitude = dict.fromkeys(BrainWave, 0.1)
    for _ in range(40):
        osc._phase[BrainWave.INFRASLOW] = 0.0  # envelope ~ max gain
        osc._step(0.2, targets)
    hi = osc._amplitude[BrainWave.ALPHA]

    osc._amplitude = dict.fromkeys(BrainWave, 0.1)
    for _ in range(40):
        osc._phase[BrainWave.INFRASLOW] = 3.141592653589793  # envelope ~ 0
        osc._step(0.2, targets)
    lo = osc._amplitude[BrainWave.ALPHA]
    assert hi > lo * 4, f"alpha not intermittent: hi={hi}, lo={lo}"


def test_rem_theta_uses_sawtooth_carrier():
    """REM theta PAC uses a sawtooth carrier, not a cosine."""
    from thalamus.brain_waves import _wave_phase_value

    assert _wave_phase_value(BrainWave.THETA, 0.0, "rem") == -1.0
    assert _wave_phase_value(BrainWave.THETA, math.pi, "rem") == 0.0
    assert _wave_phase_value(BrainWave.THETA, 0.0, "nrem") == 1.0
    assert _wave_phase_value(BrainWave.ALPHA, 0.5, "rem") == math.cos(0.5)


def test_sleep_stage_override_drives_nrem_waves():
    """The ultradian stage overrides the arousal-derived stage.

    The daemon clamps arousal into 0.10-0.25 for all of NREM, so
    arousal cannot distinguish N1 from N3 — without the override
    every NREM tick reports slow-wave N3.
    """
    summary = _make_summary(arousal=0.18, phase=PHASE_NREM)

    n1 = assess_brain_waves(summary, SleepStage.N1)
    n2 = assess_brain_waves(summary, SleepStage.N2)
    n3 = assess_brain_waves(summary, SleepStage.N3)

    assert n1.dominant == BrainWave.THETA, "N1 is theta-dominant"
    # N2's background is theta/delta with sigma riding above it —
    # spindles are transient graphoelements, not the standing band.
    assert n2.dominant == BrainWave.THETA, "N2 background is theta-dominant"
    assert n2.powers[BrainWave.SIGMA] > n3.powers[BrainWave.SIGMA]
    assert n3.dominant == BrainWave.DELTA, "N3 is slow-wave dominant"
    # Without the override the same summary reads as N3.
    from thalamus.brain_waves import reset_oscillator

    reset_oscillator()
    assert assess_brain_waves(summary).dominant == BrainWave.DELTA
    reset_oscillator()


def test_n2_waves_carry_sigma_spindles():
    """N2 waves are spindle-rich — the graphoelement that defines it."""
    state = assess_brain_waves(_make_summary(arousal=0.18, phase=PHASE_NREM), SleepStage.N2)
    assert state.powers[BrainWave.SIGMA] > 0.15
    # Spindles are generated centrally.
    central = state.regional_powers["central"]
    assert max(central, key=lambda b: central[b]) == BrainWave.SIGMA


def test_n3_waves_are_frontally_dominant_delta():
    """Slow-wave sleep delta peaks over frontal cortex."""
    state = assess_brain_waves(_make_summary(arousal=0.15, phase=PHASE_NREM), SleepStage.N3)
    frontal = state.regional_powers["frontal"]
    assert max(frontal, key=lambda b: frontal[b]) == BrainWave.DELTA


def test_effective_phase_resolves_waking_low_arousal_to_drowsy():
    """Mind._effective_phase is the chokepoint that keeps layers agreeing."""
    from types import SimpleNamespace

    from frontal_lobe.status import StatusMixin

    effective = StatusMixin._effective_phase

    def _core(zone, phase, arousal):
        return SimpleNamespace(cognitive_zone=zone, emergent_phase=phase, arousal=arousal)

    # Waking phase carrying N1-level arousal is pre-sleep.
    assert effective(None, _core(ZONE_IDLE, PHASE_ACTIVE, 0.30)) == PHASE_DROWSY
    assert effective(None, _core(ZONE_IDLE, PHASE_ALERT, 0.30)) == PHASE_DROWSY
    # Genuinely awake is left alone.
    assert effective(None, _core(ZONE_IDLE, PHASE_ACTIVE, 0.60)) == PHASE_ACTIVE
    assert effective(None, _core(ZONE_IDLE, PHASE_FLOW, 0.60)) == PHASE_FLOW
    assert effective(None, _core(ZONE_IDLE, PHASE_STRESS, 0.60)) == PHASE_STRESS
    # The clamp band is floor/ceiling safe either side of the gate.
    assert effective(None, _core(ZONE_IDLE, PHASE_ACTIVE, 0.36)) == PHASE_ACTIVE
    assert effective(None, _core(ZONE_IDLE, PHASE_ACTIVE, 0.34)) == PHASE_DROWSY
    # Never rewrite a non-waking phase.
    assert effective(None, _core(ZONE_IDLE, PHASE_OVERWHELMED, 0.10)) == PHASE_OVERWHELMED
    assert effective(None, _core(ZONE_IDLE, PHASE_DROWSY, 0.90)) == PHASE_DROWSY
    # Sleeping zone still wins.
    assert effective(None, _core(ZONE_SLEEPING, PHASE_ACTIVE, 0.15)) == PHASE_NREM
    assert effective(None, _core(ZONE_SLEEPING, PHASE_REM, 0.45)) == PHASE_REM


def test_regional_powers_normalized_per_region():
    """Each region's power vector sums to 1."""
    state = assess_brain_waves(_make_summary(arousal=0.55, phase=PHASE_ACTIVE))
    for region, powers in state.regional_powers.items():
        assert abs(sum(powers.values()) - 1.0) < 1e-6, region


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


def test_consolidation_suppressed_when_encoding_dominates():
    """Encoding mode suppresses consolidation drive — write, not replay.

    encoding_weight and consolidation_weight compete (the same
    balance compute_theta_gamma_coupling uses to pick a phase): when
    encoding outranks consolidation, the drive is gated down even
    though the slow-wave bands are present.
    """
    deep = {"arousal": 0.18, "phase": PHASE_NREM}
    # Same consolidation weight in all three — only the encoding
    # balance changes.
    encode_mode = assess_brain_waves(_make_summary(**deep, encoding=0.9, consolidation=0.3))
    balanced = assess_brain_waves(_make_summary(**deep, encoding=0.3, consolidation=0.3))
    replay_mode = assess_brain_waves(_make_summary(**deep, encoding=0.1, consolidation=0.9))
    assert encode_mode.consolidation < balanced.consolidation
    assert replay_mode.consolidation > balanced.consolidation


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
    elif dominant == BrainWave.BETA2:
        return assess_brain_waves(_make_summary(arousal=0.8, plasticity=0.2, phase=PHASE_ACTIVE))
    elif dominant == BrainWave.BETA3:
        return assess_brain_waves(_make_summary(arousal=0.7, phase=PHASE_STRESS))
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
    from frontal_lobe.reasoning import Reasoner
    from limbic_system.curiosity import Curiosity
    from limbic_system.emotion import EmotionalState
    from temporal_lobe.concepts import ConceptNetwork

    emo = EmotionalState(
        label="curious",
        nuance="default",
        cognitive_style="steady",
        valence=0.3,
        alertness=0.7,
        plasticity=0.6,
        creativity=0.5,
        caution=0.2,
        openness_to_engage=0.7,
    )
    net = ConceptNetwork()
    reasoning = Reasoner(net)
    engine = Curiosity(net, reasoning)

    gamma = _make_wave_state(BrainWave.GAMMA)
    delta = _make_wave_state(BrainWave.DELTA)

    gamma_questions = engine.generate_questions(emo, max_questions=2, brain_waves=gamma)
    delta_questions = engine.generate_questions(emo, max_questions=2, brain_waves=delta)

    # Gamma should produce at least as many questions as delta
    assert len(gamma_questions) >= len(delta_questions)


def test_working_memory_gamma_expands_capacity():
    """Gamma state expands working memory capacity."""
    from memory import WorkingMemory

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
    from frontal_lobe.global_workspace import GlobalWorkspace

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
    from frontal_lobe.reasoning import DriftDiffusionModel

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

    from limbic_system.emotion import EmotionalState
    from self import SelfModel
    from temporal_lobe.concepts import ConceptNetwork

    from infrastructure.narrative import Narrator

    model = SelfModel(born_at=int(time.time() * 1000))
    net = ConceptNetwork()
    engine = Narrator(model, net)

    emo = EmotionalState(
        label="neutral",
        nuance="default",
        cognitive_style="steady",
        valence=0.0,
        alertness=0.5,
        plasticity=0.5,
        creativity=0.5,
        caution=0.3,
        openness_to_engage=0.7,
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
    from sleep.dream_synthesis import DreamSynthesis
    from temporal_lobe.concepts import ConceptNetwork

    net = ConceptNetwork()
    # Add some concepts with similar structure
    for i in range(10):
        net.add_concept(f"concept_{i}", origin="learned")

    engine = DreamSynthesis(net)

    theta = _make_wave_state(BrainWave.THETA)
    delta = _make_wave_state(BrainWave.DELTA)

    theta_proposals = engine.synthesize(max_proposals=5, brain_waves=theta)
    delta_proposals = engine.synthesize(max_proposals=5, brain_waves=delta)

    # Theta should allow more proposals than delta
    # (both may be 0 if no structural matches, but theta's effective_max is higher)
    assert len(theta_proposals) >= len(delta_proposals)


def test_synaptic_downscaler_body_pressure_drives_shy():
    """SHY renormalisation is proportional, pressure-driven, and
    non-destructive: every edge loses weight ∝ w × rate × dt, no edge
    is deleted, and higher body sleep pressure downscales faster."""
    from brainstem.architecture import SynapticDownscaler
    from temporal_lobe.concepts import ConceptNetwork, RelationType

    net = ConceptNetwork()
    net.add_concept("a", origin="learned")
    net.add_concept("b", origin="learned")
    net.add_concept("c", origin="learned")
    net.add_edge("a", "b", RelationType.IS_A, weight=0.8)
    net.add_edge("b", "c", RelationType.IS_A, weight=0.2)

    downscaler = SynapticDownscaler()
    downscaler.body_pressure = 1.0  # body still needs sleep

    before = len(net.edges)
    w_strong, w_weak = net.edges[0].weight, net.edges[1].weight
    downscaler.downscale_during_n3(net, dt=1.0)

    # No deletion — the kill floor is gone.
    assert len(net.edges) == before
    # Proportional: the strong edge lost more absolute weight.
    d_strong = w_strong - net.edges[0].weight
    d_weak = w_weak - net.edges[1].weight
    assert d_strong > d_weak > 0.0

    # Higher body pressure → faster renormalisation.
    calm = SynapticDownscaler()
    calm.body_pressure = 0.0
    net2 = ConceptNetwork()
    net2.add_concept("x", origin="learned")
    net2.add_concept("y", origin="learned")
    net2.add_edge("x", "y", RelationType.IS_A, weight=0.8)
    calm.downscale_during_n3(net2, dt=1.0)
    assert net2.edges[0].weight > net.edges[0].weight


def test_voice_brain_waves_modulate_rhythm():
    """Brain wave state modulates voice rhythm."""
    from frontal_lobe.voice import Voice
    from limbic_system.emotion import EmotionalState
    from self import PersonalityTraits

    personality = PersonalityTraits()
    voice = Voice(personality, seed=42)

    emo = EmotionalState(
        label="neutral",
        nuance="default",
        cognitive_style="steady",
        valence=0.0,
        alertness=0.7,
        plasticity=0.5,
        creativity=0.5,
        caution=0.3,
        openness_to_engage=0.7,
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
    from thalamus.brain_waves import derive_self_priority

    state = BrainWaveState(
        dominant=BrainWave.GAMMA,
        secondary=BrainWave.BETA2,
        powers={
            BrainWave.GAMMA: 0.50,
            BrainWave.BETA2: 0.15,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA3: 0.05,
            BrainWave.ALPHA: 0.10,
            BrainWave.THETA: 0.07,
            BrainWave.DELTA: 0.03,
            BrainWave.SIGMA: 0.00,
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
    from thalamus.brain_waves import derive_self_priority

    state = BrainWaveState(
        dominant=BrainWave.DELTA,
        secondary=BrainWave.THETA,
        powers={
            BrainWave.DELTA: 0.55,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.08,
            BrainWave.BETA1: 0.02,
            BrainWave.BETA2: 0.02,
            BrainWave.BETA3: 0.01,
            BrainWave.GAMMA: 0.02,
            BrainWave.SIGMA: 0.03,
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
    from thalamus.brain_waves import derive_self_priority

    # Gamma-dominant (active thinking) but body says deprioritize (tired)
    state = BrainWaveState(
        dominant=BrainWave.GAMMA,
        secondary=BrainWave.BETA2,
        powers={
            BrainWave.GAMMA: 0.45,
            BrainWave.BETA2: 0.20,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA3: 0.05,
            BrainWave.ALPHA: 0.10,
            BrainWave.THETA: 0.07,
            BrainWave.DELTA: 0.03,
            BrainWave.SIGMA: 0.00,
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
    assert nice < 5, f"brain waves should override body deprioritization: nice={nice}"


def test_derive_self_priority_clamps():
    """Priority values should be within valid ranges."""
    from thalamus.brain_waves import derive_self_priority

    # Extreme gamma
    state = BrainWaveState(
        dominant=BrainWave.GAMMA,
        secondary=BrainWave.BETA2,
        powers={
            BrainWave.GAMMA: 1.0,
            BrainWave.BETA1: 0.0,
            BrainWave.BETA2: 0.0,
            BrainWave.BETA3: 0.0,
            BrainWave.ALPHA: 0.0,
            BrainWave.THETA: 0.0,
            BrainWave.DELTA: 0.0,
            BrainWave.SIGMA: 0.0,
        },
        focus=1.0,
        integration=1.0,
        consolidation=0.0,
        label="gamma",
        description="max gamma",
    )
    nice, io_class = derive_self_priority(state)
    assert -5 <= nice <= 10, f"nice out of range: {nice}"
    assert io_class in (
        "idle",
        "best-effort-0",
        "best-effort-1",
        "best-effort-3",
        "best-effort-6",
    ), f"invalid io_class: {io_class}"


# ─── Beta sub-band tests ──────────────────────────────────────


def test_stress_phase_high_beta3():
    """Stress phase → high beta3 (hypervigilance)."""
    summary = _make_summary(arousal=0.7, phase=PHASE_STRESS)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA3
    assert state.powers[BrainWave.BETA3] > 0.25


def test_overwhelmed_phase_high_beta3():
    """Overwhelmed phase → high beta3 (can't integrate)."""
    summary = _make_summary(arousal=0.8, phase=PHASE_OVERWHELMED)
    state = assess_brain_waves(summary)
    assert state.dominant == BrainWave.BETA3
    assert state.powers[BrainWave.BETA3] > 0.25


def test_beta_subbands_distinct():
    """Beta sub-bands are distinct in the power distribution."""
    summary = _make_summary(arousal=0.8, plasticity=0.2, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert BrainWave.BETA1 in state.powers
    assert BrainWave.BETA2 in state.powers
    assert BrainWave.BETA3 in state.powers
    assert state.powers[BrainWave.BETA2] > state.powers[BrainWave.BETA1]
    assert state.powers[BrainWave.BETA2] > state.powers[BrainWave.BETA3]


# ─── Sigma band tests ─────────────────────────────────────────


def test_n2_sleep_has_sigma():
    """N2 sleep has sigma band (sleep spindles)."""
    summary = _make_summary(arousal=0.25, phase=PHASE_NREM)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.SIGMA] > 0.1


def test_waking_no_sigma():
    """Waking states have no sigma activity."""
    summary = _make_summary(arousal=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.SIGMA] < 0.05


# ─── Infraslow and Epsilon band tests ─────────────────────────


def test_deep_sleep_has_infraslow():
    """Deep sleep (N3) has infraslow activity."""
    summary = _make_summary(arousal=0.1, phase=PHASE_NREM)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.INFRASLOW] > 0.03


def test_waking_has_infraslow():
    """Waking states have some infraslow activity."""
    summary = _make_summary(arousal=0.5, phase=PHASE_ACTIVE)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.INFRASLOW] > 0.02


def test_n3_has_epsilon():
    """N3 sleep has epsilon (HFO) activity — sharp wave-ripples."""
    summary = _make_summary(arousal=0.1, phase=PHASE_NREM)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.EPSILON] > 0.01


def test_flow_has_epsilon():
    """Flow state has epsilon (HFO) activity."""
    summary = _make_summary(arousal=0.7, plasticity=0.6, phase=PHASE_FLOW)
    state = assess_brain_waves(summary)
    assert state.powers[BrainWave.EPSILON] > 0.02


# ─── Theta-gamma coupling tests ───────────────────────────────


def test_theta_gamma_coupling_encoding_phase():
    """Encoding bias → gamma peaks near theta peak (0 rad)."""
    from thalamus.brain_waves import compute_theta_gamma_coupling

    state = assess_brain_waves(
        _make_summary(
            arousal=0.5, plasticity=0.5, phase=PHASE_ACTIVE, encoding=0.9, consolidation=0.1
        )
    )
    coupling = compute_theta_gamma_coupling(state, consolidation_weight=0.1, encoding_weight=0.9)
    assert coupling.mode == "encoding"
    assert coupling.preferred_phase < 0.5, (
        f"encoding should peak near theta peak (0 rad), got {coupling.preferred_phase}"
    )


def test_theta_gamma_coupling_retrieval_phase():
    """Retrieval bias → gamma peaks near theta trough (π rad)."""
    from thalamus.brain_waves import compute_theta_gamma_coupling

    state = assess_brain_waves(
        _make_summary(
            arousal=0.5, plasticity=0.5, phase=PHASE_ACTIVE, encoding=0.1, consolidation=0.9
        )
    )
    coupling = compute_theta_gamma_coupling(state, consolidation_weight=0.9, encoding_weight=0.1)
    assert coupling.mode == "retrieval"
    assert coupling.preferred_phase > 2.5, (
        f"retrieval should peak near theta trough (π rad), got {coupling.preferred_phase}"
    )


# ─── Dynamic stimulus tests ───────────────────────────────────


def test_stimulus_modulates_amplitude():
    """Continuous stimulus modulates band amplitude."""
    from thalamus.brain_waves import add_stimulus, reset_oscillator

    reset_oscillator()
    summary = _make_summary(arousal=0.5, phase=PHASE_ACTIVE)
    state_before = assess_brain_waves(summary)
    gamma_before = state_before.powers[BrainWave.GAMMA]

    add_stimulus(BrainWave.GAMMA, 0.3)
    # Powers are metered over the emitted trace — a real band meter
    # needs the window to fill with post-stimulus wave, so let the
    # field run (same pattern as test_stimulus_decays).
    for _ in range(10):
        time.sleep(0.05)
        state_after = assess_brain_waves(summary)
    gamma_after = state_after.powers[BrainWave.GAMMA]

    assert gamma_after > gamma_before, (
        f"stimulus should increase gamma: {gamma_before} → {gamma_after}"
    )
    reset_oscillator()


def test_stimulus_decays():
    """Stimulus decays over time without re-application."""
    from thalamus.brain_waves import _get_oscillator, add_stimulus, reset_oscillator

    reset_oscillator()
    summary = _make_summary(arousal=0.5, phase=PHASE_ACTIVE)
    osc = _get_oscillator()

    # Powers are Goertzel-metered over 4 s of emitted trace, and the
    # stimulus rings down at τ≈5 s — a power reading taken right
    # after the pulse still meters pre-stimulus wave, so the fill
    # transient can outrun the decay. Step the oscillator's own
    # clock instead of racing wall-clock sleeps: one dt-exact step
    # fills the window with stimulated wave.
    add_stimulus(BrainWave.GAMMA, 0.3)
    assess_brain_waves(summary)  # establishes _last_time
    osc._last_time -= 4.0
    state1 = assess_brain_waves(summary)
    gamma1 = state1.powers[BrainWave.GAMMA]

    # Another 4 s: the pulse rings down (~45% per step) while the
    # window refills with progressively weaker wave.
    osc._last_time -= 4.0
    state2 = assess_brain_waves(summary)
    gamma2 = state2.powers[BrainWave.GAMMA]

    assert gamma2 < gamma1, f"stimulus should decay: {gamma1} → {gamma2}"
    assert osc._stimulus[BrainWave.GAMMA] < 0.1, (
        f"register must ring down: {osc._stimulus[BrainWave.GAMMA]}"
    )
    reset_oscillator()


def test_pulse_survives_held_level_on_same_band():
    """Pulses and held levels are separate registers — a pulse on a
    band a continuous channel reports must not be clobbered by the
    next hold. This is what lets vision's alpha-blocking pulse
    coexist with the field probe's held substrate level.
    """
    from thalamus.brain_waves import (
        _get_oscillator,
        add_stimulus,
        hold_stimulus,
        reset_oscillator,
    )

    reset_oscillator()
    add_stimulus(BrainWave.ALPHA, -0.25)
    hold_stimulus(BrainWave.ALPHA, 0.10)
    osc = _get_oscillator()
    assert osc._stimulus[BrainWave.ALPHA] < 0.0, "hold must not erase the pulse"
    assert osc._held[BrainWave.ALPHA] == 0.10
    reset_oscillator()


def test_negative_stimulus_suppresses_band():
    """Inhibitory stimulus (alpha blocking) lowers band power."""
    from thalamus.brain_waves import add_stimulus, reset_oscillator

    reset_oscillator()
    summary = _make_summary(arousal=0.5, phase=PHASE_ACTIVE)

    def mean_alpha(samples: int = 20) -> float:
        # Instantaneous band power rides the oscillation (±0.04), so a
        # single reading cannot resolve a −0.3 stimulus; average a window.
        total = 0.0
        for _ in range(samples):
            time.sleep(0.05)
            total += assess_brain_waves(summary).powers[BrainWave.ALPHA]
        return total / samples

    alpha_before = mean_alpha()
    add_stimulus(BrainWave.ALPHA, -0.3)
    alpha_after = mean_alpha()

    assert alpha_after < alpha_before, (
        f"inhibitory stimulus should suppress alpha: {alpha_before} → {alpha_after}"
    )
    reset_oscillator()


def test_held_stimulus_is_poll_rate_invariant():
    """A continuous channel's reading must not integrate per call.

    The field probe reports the substrate's *current* energy on every
    assess. If that level is integrated once per call, the standing
    stimulus scales with the poll rate — the machine's slow-band
    (1/f) energy then swamps the modeled EEG until an alert body
    reads delta-dominant, which is what the live readiness gate was
    choking on. Hold semantics keep the stimulus at the reported
    level however often it is re-reported.
    """
    from thalamus.brain_waves import (
        _get_oscillator,
        hold_stimulus,
        reset_oscillator,
    )

    reset_oscillator()
    summary = _make_summary(arousal=0.6, phase=PHASE_ACTIVE)

    # ~1 Hz: twenty reports of the same level.
    for _ in range(20):
        hold_stimulus(BrainWave.DELTA, 0.20)
        assess_brain_waves(summary)
        time.sleep(0.02)
    slow = _get_oscillator()._held[BrainWave.DELTA]
    assert slow <= 0.5

    # ~10 Hz: same level, ten times as many reports — no inflation.
    reset_oscillator()
    for _ in range(200):
        hold_stimulus(BrainWave.DELTA, 0.20)
        assess_brain_waves(summary)
        time.sleep(0.002)
    fast = _get_oscillator()._held[BrainWave.DELTA]
    assert abs(fast - slow) < 0.1, (
        f"held level must not grow with poll rate: {slow:.3f} vs {fast:.3f}"
    )
    reset_oscillator()


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


def test_nrem_depth_follows_chemical_arousal() -> None:
    """NREM depth is read off her chemical arousal — which the
    sleep-wake network drives down through the descent (measured
    ~0.49 at onset to ~0.14 once LC silences), so the bands trace a
    real N1 → N2 → N3 architecture. The earlier homeostat mapping
    pinned the stage at N1 whenever adenosine sat low, because a live
    h is a decaying remainder, not a depth."""
    assert _sleep_stage_from_summary(_make_summary(arousal=0.49, phase=PHASE_NREM)) is SleepStage.N1
    assert _sleep_stage_from_summary(_make_summary(arousal=0.28, phase=PHASE_NREM)) is SleepStage.N2
    assert _sleep_stage_from_summary(_make_summary(arousal=0.15, phase=PHASE_NREM)) is SleepStage.N3


def test_nrem_arousal_descent_reports_deepening_stages() -> None:
    """The fallback arousal mapping still descends when arousal does:
    used only when no sleep-cycle tracker is active, it must report the
    full N1→N2→N3 band sweep rather than pinning at N1."""
    descent = [0.48, 0.40, 0.33, 0.26, 0.21, 0.18, 0.14]
    stages = [
        _sleep_stage_from_summary(_make_summary(arousal=a, phase=PHASE_NREM)) for a in descent
    ]
    assert stages == [
        SleepStage.N1,
        SleepStage.N1,
        SleepStage.N2,
        SleepStage.N2,
        SleepStage.N2,
        SleepStage.N3,
        SleepStage.N3,
    ]


# ─── Bistable-switch readout: early-warning stats and the
#     Kramers flicker ──────────────────────────────────────────


def _fold(position: float, bistable: bool = True) -> FoldMargins:
    """A FoldMargins at a normalized hysteresis position."""
    return FoldMargins(
        h=200.0,
        h_sleep_fold=100.0,
        h_wake_fold=300.0,
        position=position,
        bistable=bistable,
    )


def test_early_warning_populates_after_window() -> None:
    """The critical-slowing-down stats report once the window fills.

    d = θ/(α+θ) is measured off the Goertzel ring each assess; at 30
    samples the state carries ar1, variance, return rate and the
    window count — the fold's precursors, not the oscillator's guess.
    """
    summary = _make_summary(arousal=0.4, phase=PHASE_DROWSY)
    state = None
    for _ in range(35):
        state = assess_brain_waves(summary)
    assert state is not None
    ew = state.early_warning
    assert ew["n"] == 35.0
    assert math.isfinite(ew["ar1"]) and -1.0 <= ew["ar1"] <= 1.0
    assert ew["variance"] >= 0.0
    assert 0.0 <= ew["return_rate"] <= 1.0


def test_fold_margins_reach_the_state() -> None:
    """A live body's switch geometry is reported, never inferred."""
    from thalamus.brain_waves import _get_oscillator

    osc = _get_oscillator()
    m = _fold(position=0.7, bistable=True)
    state = osc.assess(_make_summary(arousal=0.4, phase=PHASE_DROWSY), fold=m)
    assert state.fold_position == 0.7
    assert state.bistable is True

    # No reading is no reading — the fields must not fabricate.
    state = osc.assess(_make_summary(arousal=0.4, phase=PHASE_DROWSY), fold=None)
    assert state.fold_position is None
    assert state.bistable is False


def test_kramers_sleepward_excursion_tilts_targets() -> None:
    """A flicker excursion on the wake branch pulls the texture
    sleepward: theta and delta rise, alpha thins — the stage-5
    microstate of the Hori descent riding the targets."""
    from thalamus.brain_waves import _get_oscillator

    osc = _get_oscillator()
    osc._flicker = 0.8  # mid-excursion, no hop bookkeeping needed
    summary = _make_summary(arousal=0.4, phase=PHASE_DROWSY)
    base = dict.fromkeys(BrainWave, 0.2)
    tilted = dict(base)
    osc._kramers_step(0.1, summary, None, tilted)
    assert tilted[BrainWave.THETA] > base[BrainWave.THETA]
    assert tilted[BrainWave.DELTA] > base[BrainWave.DELTA]
    assert tilted[BrainWave.ALPHA] < base[BrainWave.ALPHA]


def test_kramers_wakeward_excursion_on_the_sleep_branch() -> None:
    """Asleep, the same telegraph visits the *wake* texture — alpha
    and beta intrude while theta thins: micro-arousals, the mirror
    image of the descent flicker."""
    from thalamus.brain_waves import _get_oscillator

    osc = _get_oscillator()
    osc._flicker = 0.8
    summary = _make_summary(arousal=0.2, phase=PHASE_NREM)
    base = dict.fromkeys(BrainWave, 0.2)
    tilted = dict(base)
    osc._kramers_step(0.1, summary, None, tilted)
    assert tilted[BrainWave.ALPHA] > base[BrainWave.ALPHA]
    assert tilted[BrainWave.BETA1] > base[BrainWave.BETA1]
    assert tilted[BrainWave.THETA] < base[BrainWave.THETA]


def test_kramers_hop_rate_ordering_near_fold() -> None:
    """The η^{3/2} barrier law: hops near the fold come orders of
    magnitude faster than mid-band.

    η=0.05 gives λ≈0.9 hops/s; η=0.5 gives λ≈0.06. With a seeded
    telegraph over 20 one-second steps the ordering is deterministic:
    the near-fold well must hop more.
    """
    import random as _random

    from thalamus.brain_waves import _get_oscillator

    summary = _make_summary(arousal=0.4, phase=PHASE_DROWSY)
    osc = _get_oscillator()

    def hops_in(position: float, steps: int = 20) -> int:
        osc._rng = _random.Random(0)
        osc._flicker_state = 0
        osc._flicker = 0.0
        osc._last_fold = _fold(position=position, bistable=True)
        hops = 0
        prev = 0
        for _ in range(steps):
            targets = dict.fromkeys(BrainWave, 0.2)
            osc._kramers_step(1.0, summary, None, targets)
            if osc._flicker_state != prev:
                hops += 1
                prev = osc._flicker_state
        return hops

    assert hops_in(0.95) > hops_in(0.5)


def test_monostable_means_no_flicker() -> None:
    """Outside the band there is one attractor: the telegraph never
    hops, and any residual excursion relaxes home instead of
    freezing mid-texture."""
    from thalamus.brain_waves import _get_oscillator

    osc = _get_oscillator()
    summary = _make_summary(arousal=0.4, phase=PHASE_DROWSY)
    osc._last_fold = _fold(position=1.5, bistable=False)
    osc._flicker = 0.5
    for _ in range(20):
        osc._kramers_step(1.0, summary, None, dict.fromkeys(BrainWave, 0.2))
    assert osc._flicker == 0.0
    assert osc._flicker_state == 0
